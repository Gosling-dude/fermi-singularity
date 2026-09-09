"""Embedding backends.

Local sentence-transformers (BAAI/bge-small-en-v1.5) is the default: it runs
free and offline, which keeps the API budget for chat and the judge. BGE
models expect an instruction prefix on *queries* but not on passages, and
getting that asymmetry right is worth a few points of retrieval quality.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from functools import lru_cache

from companion.config import Settings, get_settings
from companion.errors import ConfigError
from companion.utils.logging import get_logger

log = get_logger("ingest")

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder(ABC):
    """Turns text into vectors. Queries and passages are encoded separately."""

    name: str
    dimension: int

    @abstractmethod
    def embed_passages(self, texts: list[str]) -> list[list[float]]: ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]: ...


class LocalEmbedder(Embedder):
    """sentence-transformers, normalised vectors so cosine == dot product."""

    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - declared dependency
            raise ConfigError(
                "sentence-transformers is not installed.",
                "Run `make setup`.",
            ) from exc
        log.info("loading embedding model", model=model_name)
        try:
            self._model = SentenceTransformer(model_name, device="cpu")
        except Exception as exc:  # noqa: BLE001
            raise ConfigError(
                f"Could not load embedding model '{model_name}': {exc}",
                "Check network access for the first download, or set "
                "EMBED_MODEL to a model already in your HuggingFace cache.",
            ) from exc
        self.name = model_name
        # `get_embedding_dimension` is the current name; fall back for older
        # sentence-transformers releases.
        getter = getattr(
            self._model, "get_embedding_dimension", None
        ) or self._model.get_sentence_embedding_dimension
        self.dimension = getter()
        self._is_bge = "bge" in model_name.lower()

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [vector.tolist() for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        prompt = f"{BGE_QUERY_PREFIX}{text}" if self._is_bge else text
        vector = self._model.encode(
            prompt, normalize_embeddings=True, show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vector.tolist()


class OpenAIEmbedder(Embedder):
    """Optional hosted embeddings (EMBED_PROVIDER=openai)."""

    def __init__(self, model_name: str, settings: Settings) -> None:
        import openai

        if not settings.openai_api_key:
            raise ConfigError(
                "EMBED_PROVIDER=openai but OPENAI_API_KEY is not set.",
                "Add the key to .env, or set EMBED_PROVIDER=local to embed "
                "on this machine for free.",
            )
        self._client = openai.OpenAI(api_key=settings.openai_api_key, timeout=120.0)
        self.name = model_name
        self.dimension = 1536 if "small" in model_name else 3072

    def _encode(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(model=self.name, input=texts)
        return [item.embedding for item in response.data]

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), 128):
            vectors.extend(self._encode(texts[start : start + 128]))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._encode([text])[0]


# Guarded because the web app warms models on a background thread; without
# it a request arriving mid-warm-up would load a second copy of the model.
_LOAD_LOCK = threading.Lock()


@lru_cache(maxsize=2)
def _build_cached(provider: str, model: str) -> Embedder:
    settings = get_settings()
    if provider == "openai":
        return OpenAIEmbedder(model, settings)
    return LocalEmbedder(model)


def _build(provider: str, model: str) -> Embedder:
    with _LOAD_LOCK:
        return _build_cached(provider, model)


def get_embedder(settings: Settings | None = None) -> Embedder:
    """The configured embedder, cached for the life of the process."""
    settings = settings or get_settings()
    return _build(settings.embed_provider, settings.embed_model)
