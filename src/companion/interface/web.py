"""FastAPI app: a small web UI plus a JSON API.

The web UI exists for one reason the CLI cannot serve: clicking a citation
should seek the actual audio to that timestamp, so a learner can verify an
answer by listening rather than by trusting. The CLI remains the guaranteed
interface.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    StreamingResponse,
)
from pydantic import BaseModel, Field

from companion.chat.agent import CompanionAgent
from companion.chat.providers import provider_available
from companion.chat.session import Session
from companion.config import get_settings
from companion.errors import CompanionError
from companion.interface.audio_store import AudioUnavailable, fetch_range
from companion.ingest.pipeline import load_episodes
from companion.retrieve.retriever import get_retriever
from companion.utils.logging import configure_logging, get_logger

log = get_logger("web")
STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(
    title="Fermi Podcast Companion",
    description="Grounded conversation over a small podcast collection.",
    version="1.0.0",
)

# Sessions are in-process and keyed by id. Multi-user infrastructure is an
# explicit non-goal for this trial.
_SESSIONS: dict[str, Session] = {}
_AGENT: CompanionAgent | None = None


def get_agent() -> CompanionAgent:
    global _AGENT
    if _AGENT is None:
        _AGENT = CompanionAgent(get_settings())
    return _AGENT


def _session(session_id: str | None) -> Session:
    if session_id and session_id in _SESSIONS:
        return _SESSIONS[session_id]
    session = Session()
    _SESSIONS[session.session_id] = session
    return session


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    session_id: str | None = None
    episode_id: str | None = None
    mode: str | None = None


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)
    episode_id: str | None = None


@app.exception_handler(CompanionError)
async def companion_error_handler(_: Request, exc: CompanionError) -> JSONResponse:
    """Surface an actionable message rather than a stack trace."""
    log.warn("request failed", error=exc.message)
    return JSONResponse(
        status_code=503,
        content={"error": exc.message, "remedy": exc.remedy},
    )


@app.get("/health")
def liveness() -> dict[str, str]:
    """Liveness probe for the platform's health check.

    Deliberately does no work: it touches no file, loads no model and reads
    no settings, so a slow first model load can never make the platform think
    the service is down. Use /api/health for the substantive readiness view.
    """
    return {"status": "ok"}


@app.get("/api/health")
def health() -> dict[str, Any]:
    settings = get_settings()
    episodes = load_episodes(settings)
    return {
        "status": "ok" if episodes else "not_ingested",
        "episodes": len(episodes),
        "chat_provider": settings.chat_provider,
        "chat_model": settings.chat_model,
        "chat_key_present": provider_available("chat", settings),
        "retrieval_mode": settings.retrieval_mode,
    }


@app.get("/api/episodes")
def list_episodes() -> dict[str, Any]:
    return {"episodes": [episode.to_dict() for episode in load_episodes()]}


@app.get("/api/episodes/{episode_id}")
def get_episode(episode_id: str) -> dict[str, Any]:
    episodes = {episode.episode_id: episode for episode in load_episodes()}
    episode = episodes.get(episode_id)
    if not episode:
        raise HTTPException(status_code=404, detail=f"no episode '{episode_id}'")
    chunks = get_retriever().chunks_for_episode(episode_id)
    return {
        "episode": episode.to_dict(),
        "outline": [
            {"chunk_id": c.chunk_id, "start": c.start, "end": c.end,
             "timestamp": c.timestamp_label, "preview": c.text[:200]}
            for c in chunks
        ],
    }


@app.get("/api/episodes/{episode_id}/audio")
def episode_audio(episode_id: str, request: Request):
    """Serve the original audio so the player can seek into it.

    Range requests are what make seeking work, and both backends preserve
    them: ``FileResponse`` implements Range locally, and the S3 backend
    forwards the header to the object store and relays its 206.
    """
    settings = get_settings()
    episodes = {episode.episode_id: episode for episode in load_episodes()}
    episode = episodes.get(episode_id)
    if not episode:
        raise HTTPException(status_code=404, detail=f"no episode '{episode_id}'")

    if settings.audio_backend == "s3":
        try:
            body, status, headers = fetch_range(
                settings, episode.source_file, request.headers.get("range")
            )
        except AudioUnavailable as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return StreamingResponse(
            body, status_code=status, media_type="audio/mpeg", headers=headers
        )

    path = settings.audio_dir / episode.source_file
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"audio file '{episode.source_file}' is no longer in audio/",
        )
    return FileResponse(path, media_type="audio/mpeg", filename=episode.source_file)


@app.post("/api/search")
def search(request: SearchRequest) -> dict[str, Any]:
    result = get_retriever().retrieve(
        request.query, top_k=request.top_k, episode_id=request.episode_id
    )
    return result.to_dict()


@app.post("/api/chat")
def chat(request: ChatRequest) -> dict[str, Any]:
    settings = get_settings()
    if not provider_available("chat", settings):
        raise HTTPException(
            status_code=503,
            detail=(
                f"No API key for CHAT_PROVIDER={settings.chat_provider}. "
                f"Add it to .env and restart. /api/search works without one."
            ),
        )
    session = _session(request.session_id)
    if request.episode_id is not None:
        session.episode_filter = request.episode_id or None
    response = get_agent().ask(
        request.message, session, episode_id=session.episode_filter,
        mode=request.mode,
    )
    payload = response.to_dict()
    payload["session_id"] = session.session_id
    return payload


@app.post("/api/locate")
def locate(request: ChatRequest) -> dict[str, Any]:
    """Find the moment in the audio that discusses something."""
    settings = get_settings()
    if not provider_available("chat", settings):
        raise HTTPException(status_code=503, detail="No chat API key configured.")
    session = _session(request.session_id)
    response = get_agent().locate(
        request.message, session, episode_id=request.episode_id
    )
    payload = response.to_dict()
    payload["session_id"] = session.session_id
    return payload


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    page = STATIC_DIR / "index.html"
    if not page.exists():
        return HTMLResponse("<h1>UI missing</h1>", status_code=500)
    return HTMLResponse(page.read_text(encoding="utf-8"))


def main() -> int:
    import os

    import uvicorn

    configure_logging()
    settings = get_settings()
    episodes = load_episodes(settings)
    if not episodes:
        print("Nothing ingested yet — run `make ingest` first.")
        return 1
    # Local default stays loopback; a container sets HOST=0.0.0.0 so the
    # platform can reach it, and PORT is assigned by the platform.
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    print(f"Fermi Podcast Companion → http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
