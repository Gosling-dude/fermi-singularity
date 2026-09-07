"""Search index construction and loading.

Two indexes are built from the same chunks and kept in lockstep:

* a Chroma collection holding dense vectors for semantic matching, and
* a BM25 index over tokenised chunk text for exact terminology matching.

Both are rebuilt together whenever the chunk set changes, so a chunk can never
exist in one index and not the other.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

from companion.config import Settings, get_settings
from companion.errors import IndexError_
from companion.ingest.embed import get_embedder
from companion.models import Chunk
from companion.retrieve.bm25 import BM25Index
from companion.utils.logging import get_logger

log = get_logger("ingest")

COLLECTION_NAME = "podcast_chunks"


def _chroma_client(settings: Settings):
    try:
        import chromadb
        from chromadb.config import Settings as ChromaSettings
    except ImportError as exc:  # pragma: no cover - declared dependency
        raise IndexError_(
            "chromadb is not installed.", "Run `make setup`."
        ) from exc
    settings.vector_db_path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(settings.vector_db_path),
        settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
    )


def build_index(chunks: list[Chunk], settings: Settings | None = None) -> None:
    """Embed every chunk and write both the vector and BM25 indexes."""
    settings = settings or get_settings()
    if not chunks:
        raise IndexError_(
            "No chunks to index.",
            "Run `make ingest` with audio files present in audio/.",
        )

    embedder = get_embedder(settings)
    texts = [chunk.text for chunk in chunks]
    log.info("embedding chunks", count=len(texts), model=embedder.name)
    vectors = embedder.embed_passages(texts)

    client = _chroma_client(settings)
    # Drop and recreate so a shrinking corpus cannot leave orphaned vectors.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:  # noqa: BLE001 - absent on first run
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )

    for start in range(0, len(chunks), 256):
        window = chunks[start : start + 256]
        collection.add(
            ids=[chunk.chunk_id for chunk in window],
            embeddings=vectors[start : start + 256],
            documents=[chunk.text for chunk in window],
            metadatas=[
                {
                    "episode_id": chunk.episode_id,
                    "episode_title": chunk.episode_title,
                    "start": chunk.start,
                    "end": chunk.end,
                    "source_file": chunk.source_file,
                }
                for chunk in window
            ],
        )
    log.info("vector index built", vectors=len(vectors), dim=embedder.dimension)

    bm25 = BM25Index.build(chunks)
    bm25.save(settings.bm25_path)
    log.info("bm25 index built", documents=len(chunks))

    # The refusal threshold is a property of this corpus, so it is computed
    # here rather than hard-coded or fitted to the evaluation cases.
    from companion.retrieve.calibrate import calibrate_relevance_gate
    from companion.retrieve.rerank import get_reranker

    gate = calibrate_relevance_gate(chunks, get_reranker(settings))
    _write_index_meta(settings, chunks, embedder.name, embedder.dimension, gate)


def _write_index_meta(
    settings: Settings,
    chunks: list[Chunk],
    embed_model: str,
    dimension: int,
    relevance_gate: float | None = None,
) -> None:
    meta = {
        "num_chunks": len(chunks),
        "embed_model": embed_model,
        "embed_dimension": dimension,
        "episodes": sorted({chunk.episode_id for chunk in chunks}),
        "collection": COLLECTION_NAME,
        "relevance_gate": relevance_gate,
        "rerank_model": settings.rerank_model if settings.enable_rerank else None,
    }
    (settings.vector_db_path / "index_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )


def load_collection(settings: Settings | None = None):
    """Open the persisted Chroma collection, with a clear error if absent."""
    settings = settings or get_settings()
    client = _chroma_client(settings)
    try:
        return client.get_collection(COLLECTION_NAME)
    except Exception as exc:  # noqa: BLE001 - chroma raises several types
        raise IndexError_(
            "The search index has not been built yet.",
            "Run `make ingest` to transcribe the audio and build the index.",
        ) from exc


def load_chunks(settings: Settings | None = None) -> list[Chunk]:
    """Read every chunk back from ``data/chunks/``."""
    settings = settings or get_settings()
    files = sorted(settings.chunks_dir.glob("*.json"))
    if not files:
        raise IndexError_(
            "No chunk files found in data/chunks/.",
            "Run `make ingest` first.",
        )
    chunks: list[Chunk] = []
    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise IndexError_(
                f"Chunk file '{path.name}' is corrupt.",
                "Delete data/ and re-run `make ingest`.",
            ) from exc
        chunks.extend(Chunk.from_dict(item) for item in payload)
    return chunks


def load_bm25(settings: Settings | None = None) -> BM25Index:
    settings = settings or get_settings()
    if not settings.bm25_path.exists():
        raise IndexError_(
            "The BM25 index is missing.", "Run `make ingest` to rebuild it."
        )
    try:
        return BM25Index.load(settings.bm25_path)
    except (pickle.UnpicklingError, EOFError, AttributeError) as exc:
        raise IndexError_(
            "The BM25 index is corrupt.",
            "Delete data/index/bm25.pkl and re-run `make ingest`.",
        ) from exc


def index_meta(settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or get_settings()
    path = settings.vector_db_path / "index_meta.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
