"""Reading episode audio for playback, from wherever it happens to live.

The browser verifies an answer by seeking to a citation's timestamp, which it
does with an HTTP Range request — it asks for a byte window, never the whole
file. So the only hard requirement on any backend here is that Range survives
intact.

Two backends, selected by ``AUDIO_BACKEND``:

* ``local`` (default) — the MP3 sits next to the app. Starlette's
  ``FileResponse`` already handles Range correctly, so the local path is left
  exactly as it was.
* ``s3`` — the MP3 lives in private S3-compatible object storage (Cloudflare
  R2). The client's Range header is passed straight through to the store and
  its ``206`` and ``Content-Range`` are relayed back, so the object store does
  the range arithmetic rather than us re-implementing it.

The ``s3`` backend exists so a public deployment can play the episodes without
the audio ever entering the git repository or the container image. The bucket
stays private; credentials live only on the server.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Iterator

from companion.config import Settings
from companion.errors import ConfigError

CHUNK_SIZE = 64 * 1024


class AudioUnavailable(Exception):
    """The requested object is not in the configured store."""


@lru_cache(maxsize=1)
def _client(endpoint: str, key_id: str, secret: str, region: str) -> Any:
    """One boto3 client per process; building it per request is wasteful."""
    try:
        import boto3
        from botocore.config import Config
    except ImportError as exc:  # pragma: no cover - declared dependency
        raise ConfigError(
            "boto3 is not installed, but AUDIO_BACKEND=s3.",
            "Run `make setup`, or set AUDIO_BACKEND=local.",
        ) from exc
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=key_id,
        aws_secret_access_key=secret,
        region_name=region,
        config=Config(signature_version="s3v4", retries={"max_attempts": 3}),
    )


def s3_client(settings: Settings) -> Any:
    """The configured object-store client, or a clear error saying what's missing."""
    missing = [
        name
        for name, value in (
            ("AUDIO_S3_BUCKET", settings.audio_s3_bucket),
            ("AUDIO_S3_ENDPOINT", settings.audio_s3_endpoint),
            ("AWS_ACCESS_KEY_ID", settings.aws_access_key_id),
            ("AWS_SECRET_ACCESS_KEY", settings.aws_secret_access_key),
        )
        if not value
    ]
    if missing:
        raise ConfigError(
            "AUDIO_BACKEND=s3 but these are unset: " + ", ".join(missing) + ".",
            "Set them in the environment, or set AUDIO_BACKEND=local to read "
            "audio from the audio/ directory instead.",
        )
    return _client(
        settings.audio_s3_endpoint,          # type: ignore[arg-type]
        settings.aws_access_key_id,          # type: ignore[arg-type]
        settings.aws_secret_access_key,      # type: ignore[arg-type]
        settings.audio_s3_region,
    )


def fetch_range(
    settings: Settings, key: str, range_header: str | None
) -> tuple[Iterator[bytes], int, dict[str, str]]:
    """Fetch an object, honouring the client's Range header verbatim.

    Returns ``(body, status, headers)``. The Range header is forwarded
    unparsed: the object store already implements the full grammar, including
    open-ended and suffix ranges, and relaying its answer is both simpler and
    more correct than re-deriving the offsets here.
    """
    client = s3_client(settings)
    request: dict[str, Any] = {"Bucket": settings.audio_s3_bucket, "Key": key}
    if range_header:
        request["Range"] = range_header

    try:
        obj = client.get_object(**request)
    except Exception as exc:  # noqa: BLE001 - botocore raises several types
        name = type(exc).__name__
        code = getattr(exc, "response", {}).get("Error", {}).get("Code", "")
        if code in {"NoSuchKey", "404", "NotFound"} or name == "NoSuchKey":
            raise AudioUnavailable(
                f"'{key}' is not in bucket '{settings.audio_s3_bucket}'"
            ) from exc
        if code in {"InvalidRange", "416"}:
            raise AudioUnavailable(f"range not satisfiable for '{key}'") from exc
        raise

    headers = {
        # Advertised on every response so the player knows it may seek.
        "Accept-Ranges": "bytes",
        "Content-Length": str(obj["ContentLength"]),
        # The audio is immutable once ingested, so let the browser keep it.
        "Cache-Control": "private, max-age=86400",
    }
    status = 200
    if "ContentRange" in obj:
        headers["Content-Range"] = obj["ContentRange"]
        status = 206

    return obj["Body"].iter_chunks(CHUNK_SIZE), status, headers
