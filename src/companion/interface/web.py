"""FastAPI app: a small web UI plus a JSON API.

The web UI exists for one reason the CLI cannot serve: clicking a citation
should seek the actual audio to that timestamp, so a learner can verify an
answer by listening rather than by trusting. The CLI remains the guaranteed
interface.
"""

from __future__ import annotations

import json
import queue
import threading
import time
import traceback
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
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
from companion.utils.memory import log_rss, rss_mb
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


# Every handler below returns the same JSON shape — {"error", "detail"} — so
# the browser can parse any response without first guessing whether it is
# JSON. Starlette's defaults do not: an unhandled exception yields a
# plain-text "Internal Server Error" body, which is what made the UI report
# "Unexpected token" instead of the real failure.


def _problem(
    status: int, error: str, detail: str = "", **extra: Any
) -> JSONResponse:
    """The single error shape returned by every endpoint."""
    body: dict[str, Any] = {"error": error, "detail": detail}
    body.update(extra)
    return JSONResponse(status_code=status, content=body)


@app.exception_handler(CompanionError)
async def companion_error_handler(_: Request, exc: CompanionError) -> JSONResponse:
    """Surface an actionable message rather than a stack trace."""
    log.warn("request failed", error=exc.message, remedy=exc.remedy)
    return _problem(503, exc.message, exc.remedy, remedy=exc.remedy)


@app.exception_handler(StarletteHTTPException)
async def http_error_handler(
    _: Request, exc: StarletteHTTPException
) -> JSONResponse:
    """404/405/etc. as JSON, including the ones Starlette raises itself."""
    detail = exc.detail if isinstance(exc.detail, str) else ""
    return _problem(exc.status_code, detail or "Request failed.", detail)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(
    _: Request, exc: RequestValidationError
) -> JSONResponse:
    """A malformed request body is a 422 with a readable message, not a list."""
    first = (exc.errors() or [{}])[0]
    where = ".".join(str(p) for p in first.get("loc", ()) if p != "body")
    detail = f"{where}: {first.get('msg', 'invalid')}" if where else str(
        first.get("msg", "invalid request")
    )
    return _problem(422, "The request body was not valid.", detail)


@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """The backstop.

    The full traceback goes to the server log, where it is useful; the client
    gets JSON with the exception type but no internals. Without this, an
    unexpected error reaches the browser as a plain-text body that
    ``response.json()`` cannot parse.
    """
    log.error(
        "unhandled error",
        path=request.url.path,
        error=f"{type(exc).__name__}: {exc}",
        traceback=traceback.format_exc(),
    )
    return _problem(
        500,
        "The server hit an unexpected error.",
        f"{type(exc).__name__}: {exc}",
    )


@app.on_event("startup")
def _report_startup_memory() -> None:
    """Baseline RSS before any model is touched."""
    log_rss("startup (no models loaded)")


@app.on_event("startup")
def _warm_models() -> None:
    """Load the embedding and reranking models before serving traffic.

    Both are lazily constructed on first use, which put ~8s of model loading
    (far more on a small cloud instance) into whichever learner happened to
    ask the first question. Doing it at start-up moves that cost to boot,
    where the platform's start-period already tolerates it. Failure is not
    fatal: the lazy path still works, it is just slow once.
    """
    if not get_settings().warm_models_on_startup:
        log.info("model warm-up disabled; models load on first use")
        return

    def warm() -> None:
        try:
            started = time.perf_counter()
            get_retriever().retrieve("warm up the embedding and reranking models")
            log.info(
                "models warmed", ms=round((time.perf_counter() - started) * 1000, 1)
            )
            log_rss("after warm-up (all models loaded)")
        except Exception as exc:  # noqa: BLE001 - warm-up must never break the app
            log.warn("model warm-up skipped", error=f"{type(exc).__name__}: {exc}")

    # On a background thread so /health answers immediately: a platform health
    # check must not wait for ~17s of model loading (much longer on a small
    # instance) before deciding the service is alive.
    threading.Thread(target=warm, name="model-warmup", daemon=True).start()


@app.get("/health")
def liveness() -> dict[str, str]:
    """Liveness probe for the platform's health check.

    Deliberately does no work: it touches no file, loads no model and reads
    no settings, so a slow first model load can never make the platform think
    the service is down. Use /api/health for the substantive readiness view.
    """
    return {"status": "ok"}


def _models_loaded() -> dict[str, bool]:
    """Which heavy models this process currently holds, without loading any."""
    from companion.ingest import embed
    from companion.retrieve import rerank

    return {
        "embedder": embed._build_cached.cache_info().currsize > 0,
        "reranker": rerank._load_cached.cache_info().currsize > 0,
    }


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
        "rss_mb": round(rss_mb(), 1),
        "models_loaded": _models_loaded(),
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


# How long the stream may go without emitting anything before a keepalive
# comment is sent. Comfortably under any proxy idle timeout.
HEARTBEAT_SECONDS = 3.0

_DONE = object()


def _sse(event: str, data: dict[str, Any]) -> str:
    """One Server-Sent Event. Data is always JSON, including errors."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/api/chat/stream")
def chat_stream(request: ChatRequest) -> StreamingResponse:
    """Stream an answer as Server-Sent Events.

    Why this exists: a hosted deployment sits behind a proxy that gives up on
    a request producing no bytes for too long, and a long answer can take
    tens of seconds to generate. Holding the connection silent until the
    whole answer exists puts the request at the mercy of that timeout. Here
    the first event is sent before generation begins, and deltas flow as
    they are produced, so the connection is never idle.

    Grounding is unchanged — the evidence gate still refuses before any model
    call, and citations are still validated against the passages the model
    saw. Only the final `done` event carries citations, because they cannot
    be validated until the answer is complete.

    Every event's payload is JSON, including failures, so a client never has
    to parse a half-written body.
    """
    settings = get_settings()
    session = _session(request.session_id)
    if request.episode_id is not None:
        session.episode_filter = request.episode_id or None

    def produce(out: "queue.Queue[Any]") -> None:
        """Run the agent on a worker thread, pushing events onto a queue."""
        try:
            for event in get_agent().ask_stream(
                request.message, session,
                episode_id=session.episode_filter, mode=request.mode,
            ):
                out.put(event)
        except CompanionError as exc:
            log.warn("stream failed", error=exc.message)
            out.put({"error": exc.message, "detail": exc.remedy,
                     "remedy": exc.remedy})
        except Exception as exc:  # noqa: BLE001 - must not break the stream
            log.error(
                "unhandled error in stream",
                error=f"{type(exc).__name__}: {exc}",
                traceback=traceback.format_exc(),
            )
            out.put({"error": "The server hit an unexpected error.",
                     "detail": f"{type(exc).__name__}: {exc}"})
        finally:
            out.put(_DONE)

    def events() -> Any:
        # Sent immediately, before any model work: this is the byte that
        # tells the proxy the response has begun.
        yield _sse("start", {"session_id": session.session_id})
        if not provider_available("chat", settings):
            yield _sse("error", {
                "error": f"No API key for CHAT_PROVIDER={settings.chat_provider}.",
                "detail": "Add it to .env and restart. /api/search works without one.",
            })
            return

        # The model can think for several seconds before the first token, and
        # retrieval runs before that. A comment line every few seconds keeps
        # the connection producing bytes throughout, so no proxy can call it
        # idle. SSE comments are ignored by clients.
        out: "queue.Queue[Any]" = queue.Queue()
        worker = threading.Thread(
            target=produce, args=(out,), name="chat-stream", daemon=True
        )
        worker.start()
        while True:
            try:
                event = out.get(timeout=HEARTBEAT_SECONDS)
            except queue.Empty:
                yield ": keepalive\n\n"
                continue
            if event is _DONE:
                break
            if isinstance(event, str):
                yield _sse("delta", {"text": event})
            elif isinstance(event, dict):
                yield _sse("error", event)
            else:
                payload = event.to_dict()
                payload["session_id"] = session.session_id
                yield _sse("done", payload)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",   # ask nginx-style proxies not to buffer
            "Connection": "keep-alive",
        },
    )


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
