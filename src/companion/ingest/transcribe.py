"""Automatic speech recognition over the supplied raw audio.

Transcription is done by us, from the audio, every time — no external or
platform-provided captions are ever read. The local backend is faster-whisper
(CTranslate2); an OpenAI Whisper API backend exists for machines that cannot
run the local model.

If the configured local model cannot be loaded (typically memory pressure on
``medium``), the loader steps down through progressively smaller models and
records which one actually ran in the transcript.
"""

from __future__ import annotations

import time
from pathlib import Path

from companion.config import Settings, get_settings
from companion.errors import ConfigError, TranscriptionError
from companion.ingest.normalize import AudioFile
from companion.models import Segment, Transcript
from companion.utils.logging import get_logger
from companion.utils.timefmt import format_duration

log = get_logger("ingest")

# Ordered largest → smallest. A failed load falls through to the next entry.
FALLBACK_CHAIN = ["large-v3", "medium", "small", "base", "tiny"]

_MODEL_CACHE: dict[tuple[str, str], object] = {}


def _fallback_models(preferred: str) -> list[str]:
    """The preferred model followed by every smaller one in the chain."""
    if preferred in FALLBACK_CHAIN:
        index = FALLBACK_CHAIN.index(preferred)
        return FALLBACK_CHAIN[index:]
    # A custom/HF model id: try it, then fall back to the standard ladder.
    return [preferred, *FALLBACK_CHAIN[FALLBACK_CHAIN.index("small"):]]


def load_local_model(settings: Settings) -> tuple[object, str]:
    """Load faster-whisper, stepping down a size on failure.

    Returns the model and the name that actually loaded, so the transcript
    records the true ASR model rather than the one that was requested.
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise ConfigError(
            "faster-whisper is not installed.",
            "Run `make setup` to install project dependencies.",
        ) from exc

    errors: list[str] = []
    for candidate in _fallback_models(settings.asr_model):
        cache_key = (candidate, settings.asr_compute_type)
        if cache_key in _MODEL_CACHE:
            return _MODEL_CACHE[cache_key], candidate
        try:
            log.info("loading ASR model", model=candidate,
                     compute_type=settings.asr_compute_type)
            model = WhisperModel(
                candidate,
                device="cpu",
                compute_type=settings.asr_compute_type,
                cpu_threads=0,
            )
        except Exception as exc:  # noqa: BLE001 - surfaced below if all fail
            errors.append(f"{candidate}: {type(exc).__name__}: {exc}")
            log.warn(
                "ASR model unavailable, falling back to a smaller model",
                model=candidate,
                error=type(exc).__name__,
            )
            continue
        if candidate != settings.asr_model:
            log.warn(
                "using a smaller ASR model than requested",
                requested=settings.asr_model,
                using=candidate,
            )
        _MODEL_CACHE[cache_key] = model
        return model, candidate

    raise TranscriptionError(
        "No Whisper model could be loaded.",
        "Check disk space and network access for the first model download, "
        "or set ASR_MODEL=tiny in .env. Attempts: " + "; ".join(errors),
    )


def transcribe_local(
    audio: AudioFile, wav_path: Path, settings: Settings
) -> Transcript:
    """Run faster-whisper over the normalised WAV."""
    model, model_name = load_local_model(settings)
    started = time.perf_counter()
    try:
        segment_iter, info = model.transcribe(
            str(wav_path),
            language=settings.asr_language or None,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
            beam_size=5,
            condition_on_previous_text=False,
        )
        segments = [
            Segment(
                id=index,
                start=round(float(raw.start), 3),
                end=round(float(raw.end), 3),
                text=raw.text.strip(),
            )
            for index, raw in enumerate(segment_iter)
            if raw.text and raw.text.strip()
        ]
    except Exception as exc:  # noqa: BLE001 - translated for the operator
        raise TranscriptionError(
            f"ASR failed on '{audio.path.name}': {type(exc).__name__}: {exc}",
            "Try a smaller model (ASR_MODEL=small) or confirm the audio "
            "decodes with `ffplay` / `ffprobe`.",
        ) from exc

    if not segments:
        raise TranscriptionError(
            f"ASR produced no speech for '{audio.path.name}'.",
            "The file may be silent, music-only, or in an unexpected "
            "language. Set ASR_LANGUAGE to the correct language code.",
        )

    elapsed = time.perf_counter() - started
    log.info(
        "transcribed",
        episode=audio.episode_id,
        model=model_name,
        segments=len(segments),
        audio=format_duration(audio.duration),
        asr_time=format_duration(elapsed),
        speed=f"{audio.duration / max(elapsed, 1e-6):.1f}x",
    )
    return Transcript(
        episode_id=audio.episode_id,
        episode_title=audio.title,
        source_file=audio.path.name,
        source_sha256=audio.sha256,
        duration=audio.duration,
        asr_model=model_name,
        asr_provider="local:faster-whisper",
        language=getattr(info, "language", settings.asr_language),
        segments=segments,
    )


def transcribe_openai(
    audio: AudioFile, wav_path: Path, settings: Settings
) -> Transcript:
    """Optional API backend, used when ASR_PROVIDER=openai.

    Still our own ASR pass over the supplied audio — the file is uploaded to a
    speech-to-text model, not to a product that returns a stored transcript.
    """
    import openai

    if not settings.openai_api_key:
        raise ConfigError(
            "ASR_PROVIDER=openai but OPENAI_API_KEY is not set.",
            "Add the key to .env, or set ASR_PROVIDER=local to transcribe "
            "on this machine for free.",
        )
    client = openai.OpenAI(api_key=settings.openai_api_key, timeout=600.0)
    started = time.perf_counter()
    try:
        with wav_path.open("rb") as handle:
            result = client.audio.transcriptions.create(
                model="whisper-1",
                file=handle,
                response_format="verbose_json",
                timestamp_granularities=["segment"],
                language=settings.asr_language or None,
            )
    except openai.APIError as exc:
        raise TranscriptionError(
            f"OpenAI transcription failed for '{audio.path.name}': {exc}",
            "Check OPENAI_API_KEY and your usage limits, or set "
            "ASR_PROVIDER=local.",
        ) from exc

    raw_segments = getattr(result, "segments", None) or []
    segments = [
        Segment(
            id=index,
            start=round(float(item.start), 3),
            end=round(float(item.end), 3),
            text=item.text.strip(),
        )
        for index, item in enumerate(raw_segments)
        if getattr(item, "text", "").strip()
    ]
    if not segments:
        raise TranscriptionError(
            f"OpenAI ASR returned no segments for '{audio.path.name}'."
        )

    log.info(
        "transcribed",
        episode=audio.episode_id,
        model="whisper-1",
        segments=len(segments),
        asr_time=format_duration(time.perf_counter() - started),
    )
    return Transcript(
        episode_id=audio.episode_id,
        episode_title=audio.title,
        source_file=audio.path.name,
        source_sha256=audio.sha256,
        duration=audio.duration,
        asr_model="whisper-1",
        asr_provider="openai",
        language=getattr(result, "language", settings.asr_language),
        segments=segments,
    )


def transcribe(
    audio: AudioFile, wav_path: Path, settings: Settings | None = None
) -> Transcript:
    """Transcribe one episode with the configured ASR backend."""
    settings = settings or get_settings()
    if settings.asr_provider == "openai":
        return transcribe_openai(audio, wav_path, settings)
    return transcribe_local(audio, wav_path, settings)
