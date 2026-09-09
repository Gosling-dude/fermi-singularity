"""Cross-encoder reranking and the relevance gate.

A bi-encoder embeds the query and the passage independently, so its cosine
similarity measures *topical* proximity. That is enough to rank candidates but
not to decide whether a passage actually answers the question — which is
exactly what a refusal decision needs. A cross-encoder reads the query and the
passage together and scores their relevance jointly, and its output is
separable enough to threshold.

The model is ``cross-encoder/ms-marco-MiniLM-L-6-v2``: 22M parameters, CPU-only,
no API key, a few milliseconds per passage.

Query normalisation matters here. "Take me to the part where they explain what
a bit is" is a request wrapped around a topic; scored verbatim, the cross-encoder
grades the wrapper. Stripping the wrapper before scoring recovers the topic.
"""

from __future__ import annotations

import re
import threading
from functools import lru_cache

from companion.config import Settings, get_settings
from companion.models import RetrievedChunk
from companion.utils.logging import get_logger
from companion.utils.memory import log_rss

log = get_logger("retrieval")

# Conversational wrappers that carry intent but no topical content.
_META_PREFIX = re.compile(
    r"""^\s*(?:
      take\s+me\s+to\s+the\s+(?:part|bit|moment|point)\s+(?:where|when)\s+
        (?:they|he|she|it)\s+(?:explains?|discusses?|talks?\s+about|describes?)|
      where\s+in\s+the\s+(?:audio|episode|episodes)\s+do(?:es)?\s+
        (?:they|we|he|she)\s+(?:talk\s+about|discuss|explain|mention)|
      which\s+episode\s+should\s+i\s+(?:listen\s+to|start\s+with)\s*
        (?:if\s+i\s+want\s+to\s+understand|to\s+understand)?|
      (?:jump|skip|go)\s+to\s+the\s+part\s+(?:where|about)|
      find\s+the\s+part\s+(?:where|about)|
      (?:can\s+you\s+|could\s+you\s+)?(?:please\s+)?
        (?:explain|describe|tell\s+me\s+about|walk\s+me\s+through)|
      what\s+do\s+(?:these|the)\s+episodes\s+say\s+about|
      according\s+to\s+(?:these|the)\s+episodes\s*,?
    )\s*""",
    re.IGNORECASE | re.VERBOSE,
)

# Trailing register requests that add no topical content.
_META_SUFFIX = re.compile(
    r"\s*(?:,?\s*and\s+why)?\s*(?:,?\s*)?(?:simply|in\s+simple\s+terms|"
    r"like\s+i'?m\s+(?:new\s+to\s+this|five)|step\s+by\s+step|"
    r"in\s+this\s+episode|in\s+these\s+episodes)?\s*[?.!]*\s*$",
    re.IGNORECASE,
)


def normalize_query(query: str) -> str:
    """Strip conversational wrapping so relevance is judged on the topic.

    Returns the original query when stripping would leave nothing to score.
    A single word is a perfectly good topic ("entropy"), so only an empty or
    punctuation-only remainder falls back.
    """
    stripped = _META_SUFFIX.sub("", _META_PREFIX.sub("", query)).strip(" ,")
    if len(stripped) < 2:
        return query.strip()
    return stripped


class Reranker:
    """Scores (query, passage) pairs jointly with a cross-encoder."""

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import CrossEncoder

        before = log_rss("before reranker load")
        log.info("loading reranker", model=model_name)
        self._model = CrossEncoder(model_name, device="cpu", max_length=512)
        after = log_rss("after reranker load")
        log.info("reranker memory", cost_mb=round(after - before, 1))
        self.name = model_name

    def score(self, query: str, chunks: list[RetrievedChunk]) -> list[float]:
        if not chunks:
            return []
        pairs = [(query, item.chunk.text) for item in chunks]
        return [float(score) for score in self._model.predict(pairs)]


# Loading is guarded because start-up warms the models on a background
# thread: without it, a request arriving mid-warm-up would miss the cache and
# load a second copy of the model.
_LOAD_LOCK = threading.Lock()


@lru_cache(maxsize=1)
def _load_cached(model_name: str) -> Reranker | None:
    try:
        return Reranker(model_name)
    except Exception as exc:  # noqa: BLE001 - degradation is deliberate
        log.warn(
            "reranker unavailable; falling back to bi-encoder scores only",
            model=model_name,
            error=f"{type(exc).__name__}: {exc}",
        )
        return None


def _load(model_name: str) -> Reranker | None:
    with _LOAD_LOCK:
        return _load_cached(model_name)


def get_reranker(settings: Settings | None = None) -> Reranker | None:
    settings = settings or get_settings()
    if not settings.enable_rerank:
        return None
    return _load(settings.rerank_model)


def rerank(
    query: str, chunks: list[RetrievedChunk], settings: Settings | None = None
) -> tuple[list[RetrievedChunk], float | None]:
    """Reorder candidates by cross-encoder relevance.

    Returns the reordered chunks and the best relevance score, which the
    evidence gate uses to decide whether any answer is supported at all.
    ``None`` means the reranker was unavailable and the caller should fall
    back to bi-encoder similarity.
    """
    settings = settings or get_settings()
    reranker = get_reranker(settings)
    if reranker is None or not chunks:
        return chunks, None

    topic = normalize_query(query)
    scores = reranker.score(topic, chunks)
    for item, score in zip(chunks, scores):
        item.rerank_score = round(score, 4)
    ordered = sorted(
        chunks, key=lambda item: item.rerank_score or float("-inf"), reverse=True
    )
    best = max(scores)
    log.debug(
        "reranked", topic=topic[:60], best=round(best, 3), candidates=len(chunks)
    )
    return ordered, best
