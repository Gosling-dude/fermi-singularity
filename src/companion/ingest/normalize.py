"""Audio discovery, validation and normalisation.

The supplied MP3s are never modified. FFmpeg writes a 16 kHz mono WAV under
``data/normalized/`` — the format faster-whisper wants — and every derived
artefact is keyed by the SHA-256 of the original file so re-running ingestion
is idempotent.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from companion.errors import AudioError
from companion.utils.logging import get_logger

log = get_logger("ingest")

AUDIO_SUFFIXES = {".mp3", ".m4a", ".wav", ".flac", ".ogg", ".opus", ".aac", ".wma"}
SAMPLE_RATE = 16_000


@dataclass
class AudioFile:
    """A discovered source file with its identity and probed duration."""

    path: Path
    episode_id: str
    title: str
    sha256: str
    duration: float


def require_ffmpeg() -> None:
    """Fail early and actionably when FFmpeg is absent."""
    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        return
    raise AudioError(
        "FFmpeg (and ffprobe) were not found on PATH.",
        "Install with `brew install ffmpeg` on macOS, "
        "`sudo apt install ffmpeg` on Debian/Ubuntu, then re-run `make ingest`.",
    )


def sha256_of(path: Path) -> str:
    """Content hash used as the idempotency key for the whole pipeline."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def probe_duration(path: Path) -> float:
    """Duration in seconds via ffprobe; raises ``AudioError`` if unreadable."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "json", str(path),
            ],
            capture_output=True,
            text=True,
            timeout=120,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise AudioError(
            f"'{path.name}' could not be read by ffprobe — it may be corrupt "
            f"or not an audio file.",
            f"Verify the download, then re-run. ffprobe said: "
            f"{(exc.stderr or '').strip()[:200]}",
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise AudioError(f"ffprobe timed out reading '{path.name}'.") from exc

    try:
        duration = float(json.loads(result.stdout)["format"]["duration"])
    except (KeyError, ValueError, json.JSONDecodeError) as exc:
        raise AudioError(
            f"'{path.name}' has no readable duration — likely a corrupt file."
        ) from exc
    if duration <= 0:
        raise AudioError(f"'{path.name}' reports a zero-length stream.")
    return duration


def derive_episode_id(path: Path) -> str:
    """A stable, filesystem-safe id derived from the filename."""
    stem = path.stem.lower()
    slug = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")
    return slug or "episode"


def derive_title(path: Path) -> str:
    """A human title derived from the filename.

    Filenames are the only metadata the raw audio carries, so leading track
    numbers and separators are stripped and the rest is title-cased only when
    it appears to be lowercase slug text.
    """
    stem = path.stem
    stem = re.sub(r"^\s*\d{1,3}\s*[-_.)]\s*", "", stem)
    stem = stem.replace("_", " ").replace("-", " ")
    stem = re.sub(r"\s+", " ", stem).strip()
    if not stem:
        return path.stem
    if stem.islower():
        return stem.title()
    return stem


def discover_audio(audio_dir: Path) -> list[AudioFile]:
    """Find and validate every audio file in ``audio_dir``.

    Returns them sorted by filename so episode ordering is deterministic.
    Never hard-codes filenames: whatever valid audio is present is processed.
    """
    require_ffmpeg()
    if not audio_dir.exists():
        raise AudioError(
            f"Audio directory '{audio_dir}' does not exist.",
            f"Create it and place the supplied episode files inside: "
            f"mkdir -p {audio_dir}",
        )

    candidates = sorted(
        path
        for path in audio_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in AUDIO_SUFFIXES
        and not path.name.startswith(".")
    )
    if not candidates:
        raise AudioError(
            f"No audio files found in '{audio_dir}'.",
            "Place the supplied podcast episodes (.mp3) in that folder, then "
            "run `make ingest`. See audio/README.md for details.",
        )

    discovered: list[AudioFile] = []
    seen_ids: dict[str, Path] = {}
    for path in candidates:
        if path.stat().st_size == 0:
            raise AudioError(
                f"'{path.name}' is empty (0 bytes).",
                "Re-download the file and run `make ingest` again.",
            )
        episode_id = derive_episode_id(path)
        if episode_id in seen_ids:
            # Two filenames slugging to the same id would silently overwrite
            # each other's transcript; disambiguate rather than lose one.
            episode_id = f"{episode_id}_{len(seen_ids)}"
        seen_ids[episode_id] = path
        discovered.append(
            AudioFile(
                path=path,
                episode_id=episode_id,
                title=derive_title(path),
                sha256=sha256_of(path),
                duration=probe_duration(path),
            )
        )
        log.info(
            "discovered",
            file=path.name,
            episode=episode_id,
            duration_s=round(discovered[-1].duration, 1),
        )
    return discovered


def normalize_audio(source: Path, target: Path, *, force: bool = False) -> Path:
    """Decode ``source`` to 16 kHz mono WAV at ``target``.

    The original file is only ever read. Returns the target path.
    """
    require_ffmpeg()
    if target.exists() and not force and target.stat().st_size > 0:
        log.debug("normalized artefact reused", file=target.name)
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".partial.wav")
    command = [
        "ffmpeg", "-nostdin", "-y", "-loglevel", "error",
        "-i", str(source),
        "-ac", "1", "-ar", str(SAMPLE_RATE), "-vn",
        "-c:a", "pcm_s16le",
        str(partial),
    ]
    try:
        subprocess.run(command, capture_output=True, text=True, check=True, timeout=3600)
    except subprocess.CalledProcessError as exc:
        partial.unlink(missing_ok=True)
        raise AudioError(
            f"FFmpeg failed to decode '{source.name}'.",
            f"The file may be corrupt or truncated. FFmpeg said: "
            f"{(exc.stderr or '').strip()[:300]}",
        ) from exc
    except subprocess.TimeoutExpired as exc:
        partial.unlink(missing_ok=True)
        raise AudioError(f"FFmpeg timed out normalising '{source.name}'.") from exc

    # Rename only after a successful decode so an interrupted run never
    # leaves a half-written WAV that a later run would treat as complete.
    partial.replace(target)
    log.info("normalized", file=source.name, output=target.name)
    return target
