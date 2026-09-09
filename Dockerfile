# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# Serving image for the Fermi Podcast Companion.
#
# This image SERVES a pre-built index. It never runs ASR and never runs
# `make ingest`: the index in data/ is built once by `make ingest` on a
# machine with the audio, committed, and copied in here. That keeps the image
# small, the start-up instant, and the deployment reproducible.
#
# Consequently faster-whisper's ~1.4 GB model and FFmpeg are NOT installed —
# nothing at request time decodes audio or transcribes.
# ---------------------------------------------------------------------------

# ---- build stage: resolve wheels once ------------------------------------
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build
COPY pyproject.toml ./
COPY src ./src

# CPU-only torch keeps the image ~2.5 GB smaller than the default CUDA wheels.
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --upgrade pip \
 && /opt/venv/bin/pip install \
      --extra-index-url https://download.pytorch.org/whl/cpu \
      .

# Bake the two runtime models into the image so the first request is fast and
# the container needs no network at run time. The ASR model is deliberately
# not fetched — this image never transcribes.
ENV HF_HOME=/opt/hf
RUN /opt/venv/bin/python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('BAAI/bge-small-en-v1.5', device='cpu'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cpu', max_length=512); \
print('models cached')"

# ---- runtime stage --------------------------------------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app/src \
    HF_HOME=/opt/hf \
    # Models are already in the image; never reach out to the Hub at runtime.
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    # Bind where the platform can reach us. PORT is supplied by the platform.
    HOST=0.0.0.0 \
    PORT=8000 \
    # Absolute paths so nothing depends on the working directory.
    DATA_DIR=/app/data \
    AUDIO_DIR=/app/audio \
    VECTOR_DB_PATH=/app/data/index

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/hf /opt/hf

WORKDIR /app

# Application source.
COPY src ./src
COPY pyproject.toml ./

# The pre-built index: the only data the server reads.
#   data/index/       Chroma (chroma.sqlite3 + HNSW segment) and bm25.pkl
#   data/chunks/      passage text the retriever loads
#   data/manifest.json  the episode catalogue
COPY data/index ./data/index
COPY data/chunks ./data/chunks
COPY data/manifest.json ./data/manifest.json

# An empty audio/ directory. No podcast media is ever baked into a layer —
# .dockerignore strips every media file, and an image can be pushed to a
# registry and inspected. In production the browser's audio requests are
# served from private object storage (AUDIO_BACKEND=s3), which keeps
# Range/seek working without redistributing the MP3s. Created rather than
# COPYd so the build cannot fail when the context has no audio at all.
RUN mkdir -p /app/audio

# Fail the build rather than ship a broken index.
RUN python -c "\
import json,pathlib,sys; \
m=json.loads(pathlib.Path('/app/data/manifest.json').read_text()); \
eps=m.get('episodes',{}); \
seg=list(pathlib.Path('/app/data/index').glob('*/data_level0.bin')); \
assert eps, 'manifest has no episodes'; \
assert pathlib.Path('/app/data/index/chroma.sqlite3').exists(), 'chroma db missing'; \
assert pathlib.Path('/app/data/index/bm25.pkl').exists(), 'bm25 index missing'; \
assert seg, 'Chroma HNSW vectors missing (is *.bin gitignored?)'; \
print(f'index OK: {len(eps)} episodes, {len(seg)} vector segment(s)')"

# Run as a non-root user.
RUN useradd --create-home --uid 10001 app && chown -R app:app /app
USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD python -c "import os,urllib.request; \
urllib.request.urlopen(f\"http://127.0.0.1:{os.getenv('PORT','8000')}/health\", timeout=4)"

# Shell form so ${PORT} is expanded from the platform's environment.
CMD ["sh", "-c", "exec uvicorn companion.interface.web:app --host 0.0.0.0 --port ${PORT:-8000} --log-level warning"]
