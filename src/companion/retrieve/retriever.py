"""Hybrid retrieval: dense + BM25, fused with Reciprocal Rank Fusion.

Why RRF rather than a weighted score blend: cosine similarities and BM25
scores live on different, corpus-dependent scales, so any fixed weighting
needs re-tuning whenever the corpus changes. RRF consumes only *ranks*, so it
is scale-free and stable across corpora — the right default for a system that
must work on whatever three episodes it is handed.

Retrieval also owns two product behaviours:

* ``episode_id`` filtering, so a learner can scope a question to one episode;
* ``compare`` mode, which guarantees per-episode representation so a
  cross-episode question cannot be answered from whichever episode happens to
  dominate the ranking.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal

from companion.config import Settings, get_settings
from companion.errors import RetrievalError
from companion.ingest.embed import get_embedder
from companion.ingest.index import index_meta, load_bm25, load_chunks, load_collection
from companion.models import Chunk, RetrievedChunk
from companion.retrieve.rerank import rerank
from companion.utils.logging import get_logger

log = get_logger("retrieval")

Mode = Literal["default", "compare"]


@dataclass
class RetrievalResult:
    """Retrieved context plus the trace the eval runner records."""

    query: str
    chunks: list[RetrievedChunk]
    mode: Mode = "default"
    episode_filter: str | None = None
    dense_hits: int = 0
    bm25_hits: int = 0
    latency_ms: float = 0.0
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return not self.chunks

    @property
    def top_score(self) -> float:
        return self.chunks[0].score if self.chunks else 0.0

    @property
    def episodes(self) -> list[str]:
        seen: list[str] = []
        for item in self.chunks:
            if item.chunk.episode_title not in seen:
                seen.append(item.chunk.episode_title)
        return seen

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "mode": self.mode,
            "episode_filter": self.episode_filter,
            "dense_hits": self.dense_hits,
            "bm25_hits": self.bm25_hits,
            "latency_ms": round(self.latency_ms, 1),
            "chunks": [item.to_dict() for item in self.chunks],
            "diagnostics": self.diagnostics,
        }


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]], k: int = 60
) -> dict[str, float]:
    """Fuse ranked id lists into one score map.

    Each list contributes ``1 / (k + rank)`` per document. ``k`` damps the
    influence of the very top of any single list, which is what stops one
    retriever from unilaterally deciding the final order.
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores


class Retriever:
    """Loads the indexes once and serves queries against them."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._chunks: dict[str, Chunk] = {
            chunk.chunk_id: chunk for chunk in load_chunks(self.settings)
        }
        if not self._chunks:
            raise RetrievalError(
                "The chunk store is empty.", "Run `make ingest`."
            )
        self._collection = load_collection(self.settings)
        self._bm25 = load_bm25(self.settings)
        self._embedder = get_embedder(self.settings)
        meta = index_meta(self.settings)
        self.relevance_gate: float | None = (
            self.settings.rerank_gate_override
            if self.settings.rerank_gate_override is not None
            else meta.get("relevance_gate")
        )
        log.info(
            "retriever ready",
            chunks=len(self._chunks),
            episodes=len(self.episode_ids()),
            mode=self.settings.retrieval_mode,
        )

    # --- catalogue ------------------------------------------------------
    def episode_ids(self) -> list[str]:
        seen: list[str] = []
        for chunk in self._chunks.values():
            if chunk.episode_id not in seen:
                seen.append(chunk.episode_id)
        return sorted(seen)

    def chunk(self, chunk_id: str) -> Chunk | None:
        return self._chunks.get(chunk_id)

    def chunks_for_episode(self, episode_id: str) -> list[Chunk]:
        return sorted(
            (c for c in self._chunks.values() if c.episode_id == episode_id),
            key=lambda c: c.start,
        )

    # --- component retrievers -------------------------------------------
    def _dense(self, query: str, k: int, episode_id: str | None) -> list[tuple[str, float]]:
        where = {"episode_id": episode_id} if episode_id else None
        try:
            response = self._collection.query(
                query_embeddings=[self._embedder.embed_query(query)],
                n_results=min(k, len(self._chunks)),
                where=where,
            )
        except Exception as exc:  # noqa: BLE001 - chroma raises broadly
            raise RetrievalError(
                f"The vector index could not be queried: {exc}",
                "Delete data/index/ and re-run `make ingest`.",
            ) from exc
        ids = (response.get("ids") or [[]])[0]
        distances = (response.get("distances") or [[]])[0]
        # Chroma returns cosine *distance*; convert to a similarity so larger
        # is better and the number is interpretable as relevance.
        return [
            (doc_id, 1.0 - float(distance))
            for doc_id, distance in zip(ids, distances)
        ]

    def _lexical(self, query: str, k: int, episode_id: str | None) -> list[tuple[str, float]]:
        hits = self._bm25.search(query, top_k=k * 3 if episode_id else k)
        if episode_id:
            hits = [
                (cid, score)
                for cid, score in hits
                if (c := self._chunks.get(cid)) and c.episode_id == episode_id
            ][:k]
        return hits

    # --- public API ------------------------------------------------------
    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        episode_id: str | None = None,
        mode: Mode = "default",
    ) -> RetrievalResult:
        """Run the configured retrieval strategy for one query."""
        if not query or not query.strip():
            raise RetrievalError(
                "The query is empty.", "Ask a question in natural language."
            )
        settings = self.settings
        top_k = top_k or settings.top_k
        started = time.perf_counter()

        dense = self._dense(query, settings.dense_k, episode_id)
        dense_scores = dict(dense)
        dense_ranks = {cid: rank for rank, (cid, _) in enumerate(dense, start=1)}

        if settings.retrieval_mode == "hybrid":
            lexical = self._lexical(query, settings.bm25_k, episode_id)
        else:
            # Dense-only mode exists so the evaluation baseline is the real
            # production code path with one flag flipped, not a separate
            # re-implementation.
            lexical = []
        bm25_scores = dict(lexical)
        bm25_ranks = {cid: rank for rank, (cid, _) in enumerate(lexical, start=1)}

        ranked_lists = [[cid for cid, _ in dense]]
        if lexical:
            ranked_lists.append([cid for cid, _ in lexical])
        fused = reciprocal_rank_fusion(ranked_lists, k=settings.rrf_k)

        ordered = sorted(fused.items(), key=lambda pair: pair[1], reverse=True)
        candidates = [
            RetrievedChunk(
                chunk=self._chunks[cid],
                score=score,
                dense_rank=dense_ranks.get(cid),
                bm25_rank=bm25_ranks.get(cid),
                dense_score=(
                    round(dense_scores[cid], 4) if cid in dense_scores else None
                ),
                bm25_score=(
                    round(bm25_scores[cid], 4) if cid in bm25_scores else None
                ),
            )
            for cid, score in ordered[: settings.fusion_k]
            if cid in self._chunks
        ]

        # Rerank the fused pool with a cross-encoder. This both improves the
        # final ordering and produces the relevance score the evidence gate
        # uses to decide whether the collection supports an answer at all.
        candidates, best_relevance = rerank(query, candidates, settings)

        if mode == "compare" and settings.enable_compare_mode and not episode_id:
            selected = self._balance_across_episodes(candidates, top_k)
        else:
            selected = candidates[:top_k]

        result = RetrievalResult(
            query=query,
            chunks=selected,
            mode=mode,
            episode_filter=episode_id,
            dense_hits=len(dense),
            bm25_hits=len(lexical),
            latency_ms=(time.perf_counter() - started) * 1000,
            diagnostics={
                "retrieval_mode": settings.retrieval_mode,
                "fusion_pool": len(candidates),
                "max_dense_similarity": (
                    round(max(dense_scores.values()), 4) if dense_scores else None
                ),
                "max_rerank_score": (
                    round(best_relevance, 4) if best_relevance is not None else None
                ),
                "rerank_enabled": best_relevance is not None,
                "relevance_gate": self.relevance_gate,
            },
        )
        log.info(
            "retrieved",
            query=query[:70],
            dense=len(dense),
            bm25=len(lexical),
            selected=len(selected),
            episodes=len(result.episodes),
            latency_ms=round(result.latency_ms, 1),
        )
        return result

    def _balance_across_episodes(
        self, candidates: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        """Round-robin across episodes so comparisons see every side.

        Without this, a strong match in one episode can occupy the whole
        context window and the model ends up "comparing" one episode with
        itself.
        """
        buckets: dict[str, list[RetrievedChunk]] = {}
        for item in candidates:
            buckets.setdefault(item.chunk.episode_id, []).append(item)
        # Preserve fused ordering of episodes: best-scoring episode goes first.
        def bucket_rank(episode: str) -> float:
            head = buckets[episode][0]
            return head.rerank_score if head.rerank_score is not None else head.score

        order = sorted(buckets, key=bucket_rank, reverse=True)
        balanced: list[RetrievedChunk] = []
        depth = 0
        while len(balanced) < top_k:
            added = False
            for episode in order:
                if depth < len(buckets[episode]) and len(balanced) < top_k:
                    balanced.append(buckets[episode][depth])
                    added = True
            if not added:
                break
            depth += 1
        return balanced


_RETRIEVER: Retriever | None = None


def get_retriever(settings: Settings | None = None, *, refresh: bool = False) -> Retriever:
    """Process-wide retriever; loading the indexes twice is wasteful."""
    global _RETRIEVER
    if _RETRIEVER is None or refresh:
        _RETRIEVER = Retriever(settings)
    return _RETRIEVER
