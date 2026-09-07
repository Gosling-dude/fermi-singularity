"""End-to-end ingestion: raw audio → transcripts → chunks → search indexes.

The pipeline is idempotent at episode granularity. Each episode's SHA-256 is
recorded in ``data/manifest.json`` alongside the settings that produced its
artefacts; an episode is re-transcribed only when its audio content changes or
the ASR configuration changes. Chunking and indexing are cheap and are always
redone from the stored transcripts, which keeps the two indexes consistent
with each other and with the chunk files.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from companion.config import Settings, get_settings
from companion.ingest.chunk import chunk_transcript
from companion.ingest.index import build_index
from companion.ingest.normalize import AudioFile, discover_audio, normalize_audio
from companion.ingest.transcribe import transcribe
from companion.models import Chunk, Episode, Transcript
from companion.utils.logging import get_logger
from companion.utils.timefmt import format_duration, format_timestamp

log = get_logger("ingest")

MANIFEST_VERSION = 2


@dataclass
class IngestStats:
    """What one ingestion run did — printed at the end and stored in state."""

    episodes: list[Episode] = field(default_factory=list)
    transcribed: list[str] = field(default_factory=list)
    reused: list[str] = field(default_factory=list)
    total_audio_seconds: float = 0.0
    total_segments: int = 0
    total_chunks: int = 0
    asr_seconds: float = 0.0
    total_seconds: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "episodes": [episode.to_dict() for episode in self.episodes],
            "transcribed": self.transcribed,
            "reused": self.reused,
            "total_audio_seconds": round(self.total_audio_seconds, 1),
            "total_segments": self.total_segments,
            "total_chunks": self.total_chunks,
            "asr_seconds": round(self.asr_seconds, 1),
            "total_seconds": round(self.total_seconds, 1),
        }


def _asr_fingerprint(settings: Settings) -> str:
    """Changing any of these invalidates a stored transcript."""
    return f"{settings.asr_provider}:{settings.asr_model}:{settings.asr_language}"


def load_manifest(settings: Settings) -> dict[str, Any]:
    if not settings.manifest_path.exists():
        return {"version": MANIFEST_VERSION, "episodes": {}}
    try:
        payload = json.loads(settings.manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        log.warn("manifest was unreadable; treating every episode as new")
        return {"version": MANIFEST_VERSION, "episodes": {}}
    if payload.get("version") != MANIFEST_VERSION:
        log.warn("manifest version changed; re-ingesting all episodes")
        return {"version": MANIFEST_VERSION, "episodes": {}}
    return payload


def save_manifest(settings: Settings, manifest: dict[str, Any]) -> None:
    settings.manifest_path.write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def _transcript_path(settings: Settings, episode_id: str) -> Path:
    return settings.transcripts_dir / f"{episode_id}.json"


def _write_transcript(settings: Settings, transcript: Transcript) -> None:
    """Persist the machine-readable transcript and a human-readable one."""
    json_path = _transcript_path(settings, transcript.episode_id)
    json_path.write_text(
        json.dumps(transcript.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    lines = [
        f"# {transcript.episode_title}",
        f"# source: {transcript.source_file}",
        f"# duration: {format_duration(transcript.duration)}",
        f"# asr: {transcript.asr_provider} ({transcript.asr_model})",
        "",
    ]
    lines.extend(
        f"[{format_timestamp(segment.start)}] {segment.text}"
        for segment in transcript.segments
    )
    (settings.transcripts_dir / f"{transcript.episode_id}.txt").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def _read_transcript(settings: Settings, episode_id: str) -> Transcript | None:
    path = _transcript_path(settings, episode_id)
    if not path.exists():
        return None
    try:
        return Transcript.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, KeyError):
        log.warn("stored transcript was unreadable; re-transcribing",
                 episode=episode_id)
        return None


def _write_chunks(settings: Settings, episode_id: str, chunks: list[Chunk]) -> None:
    (settings.chunks_dir / f"{episode_id}.json").write_text(
        json.dumps([chunk.to_dict() for chunk in chunks], indent=2,
                   ensure_ascii=False),
        encoding="utf-8",
    )


def _needs_transcription(
    manifest: dict[str, Any], audio: AudioFile, fingerprint: str, settings: Settings
) -> bool:
    record = manifest["episodes"].get(audio.episode_id)
    if not record:
        return True
    if record.get("sha256") != audio.sha256:
        log.info("audio changed since last run", episode=audio.episode_id)
        return True
    if record.get("asr_fingerprint") != fingerprint:
        log.info("ASR configuration changed", episode=audio.episode_id)
        return True
    return not _transcript_path(settings, audio.episode_id).exists()


def _prune_removed_episodes(
    settings: Settings, manifest: dict[str, Any], present: set[str]
) -> None:
    """Drop artefacts for audio that is no longer in audio/.

    Without this, deleting an episode would leave its chunks in the index and
    the system would keep citing an episode the learner cannot listen to.
    """
    for episode_id in list(manifest["episodes"]):
        if episode_id in present:
            continue
        log.warn("episode removed from audio/; discarding its artefacts",
                 episode=episode_id)
        for path in (
            _transcript_path(settings, episode_id),
            settings.transcripts_dir / f"{episode_id}.txt",
            settings.chunks_dir / f"{episode_id}.json",
            settings.normalized_dir / f"{episode_id}.wav",
        ):
            path.unlink(missing_ok=True)
        manifest["episodes"].pop(episode_id, None)


def run_ingestion(
    settings: Settings | None = None, *, force: bool = False
) -> IngestStats:
    """Discover, transcribe, chunk and index every episode in ``audio/``."""
    settings = settings or get_settings()
    settings.ensure_dirs()
    started = time.perf_counter()

    audio_files = discover_audio(settings.audio_dir)
    manifest = load_manifest(settings)
    fingerprint = _asr_fingerprint(settings)
    _prune_removed_episodes(
        settings, manifest, {audio.episode_id for audio in audio_files}
    )

    stats = IngestStats()
    all_chunks: list[Chunk] = []

    for audio in audio_files:
        log.info(
            "episode", name=audio.title, file=audio.path.name,
            duration=format_duration(audio.duration),
        )
        transcript: Transcript | None = None
        if not force and not _needs_transcription(
            manifest, audio, fingerprint, settings
        ):
            transcript = _read_transcript(settings, audio.episode_id)
            if transcript is not None:
                stats.reused.append(audio.episode_id)
                log.info("reusing stored transcript", episode=audio.episode_id)

        if transcript is None:
            wav = normalize_audio(
                audio.path,
                settings.normalized_dir / f"{audio.episode_id}.wav",
                force=force,
            )
            asr_started = time.perf_counter()
            transcript = transcribe(audio, wav, settings)
            stats.asr_seconds += time.perf_counter() - asr_started
            _write_transcript(settings, transcript)
            stats.transcribed.append(audio.episode_id)

        chunks = chunk_transcript(transcript, settings)
        _write_chunks(settings, audio.episode_id, chunks)
        all_chunks.extend(chunks)

        stats.episodes.append(
            Episode(
                episode_id=transcript.episode_id,
                title=transcript.episode_title,
                source_file=transcript.source_file,
                duration=transcript.duration,
                sha256=transcript.source_sha256,
                num_segments=len(transcript.segments),
                num_chunks=len(chunks),
            )
        )
        stats.total_audio_seconds += transcript.duration
        stats.total_segments += len(transcript.segments)
        stats.total_chunks += len(chunks)

        manifest["episodes"][audio.episode_id] = {
            "sha256": audio.sha256,
            "asr_fingerprint": fingerprint,
            "title": audio.title,
            "source_file": audio.path.name,
            "duration": transcript.duration,
            "num_segments": len(transcript.segments),
            "num_chunks": len(chunks),
        }

    build_index(all_chunks, settings)
    stats.total_seconds = time.perf_counter() - started

    manifest["stats"] = stats.as_dict()
    manifest["embed_model"] = settings.embed_model
    save_manifest(settings, manifest)
    return stats


def load_episodes(settings: Settings | None = None) -> list[Episode]:
    """The episode catalogue, read from the manifest written by ingestion."""
    settings = settings or get_settings()
    manifest = load_manifest(settings)
    episodes = [
        Episode(
            episode_id=episode_id,
            title=record["title"],
            source_file=record["source_file"],
            duration=record["duration"],
            sha256=record["sha256"],
            num_segments=record.get("num_segments", 0),
            num_chunks=record.get("num_chunks", 0),
        )
        for episode_id, record in manifest.get("episodes", {}).items()
    ]
    return sorted(episodes, key=lambda episode: episode.title)
