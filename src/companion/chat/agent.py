"""The conversational agent: retrieve, ground, answer, verify.

One turn is:

    message → (rewrite if a follow-up) → hybrid retrieval → evidence gate
            → grounded generation → citation validation → answer

Two guards sit either side of the model. The *evidence gate* refuses before
spending a token when retrieval found nothing relevant, and *citation
validation* checks afterwards that what came back points at real speech.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Iterator

from companion.chat.citations import (
    Citation,
    citation_validity_rate,
    extract_citations,
    format_sources,
    strip_invalid_citations,
    validate_citations,
)
from companion.chat.prompt import (
    NOT_COVERED_MARKER,
    REWRITE_PROMPT,
    SYSTEM_PROMPT,
    build_rewrite_turn,
    build_user_turn,
    cap_context,
)
from companion.chat.providers import (
    LLMProvider,
    LLMResponse,
    get_provider,
    provider_available,
)
from companion.chat.session import Session, Turn
from companion.config import Settings, get_settings
from companion.errors import ProviderError
from companion.ingest.pipeline import load_episodes
from companion.models import Episode, RetrievedChunk
from companion.retrieve.retriever import Retriever, RetrievalResult, get_retriever
from companion.utils.logging import get_logger

log = get_logger("chat")

# Phrasings that signal the learner wants to compare episodes or be taken to a
# moment in the audio. These only pick a retrieval/prompt mode — they never
# decide the content of an answer.
_COMPARE_RE = re.compile(
    r"\b(compare|contrast|differ|difference|both episodes|across (?:the )?episodes"
    r"|versus|vs\.?|how do they|which episode)\b",
    re.IGNORECASE,
)
_LOCATE_RE = re.compile(
    r"\b(take me to|where (?:do|does|did) (?:they|he|she|it)|jump to|point me to"
    r"|which (?:part|moment|timestamp)|play the part|find the part"
    r"|where in the (?:audio|episode))\b",
    re.IGNORECASE,
)

# Deliberately makes no claim beyond the outcome. An earlier version said
# "I searched the transcripts of every episode…", which the evaluation judge
# correctly flagged as an assertion the retrieved passages cannot support.
NOT_COVERED_MESSAGE = (
    "The supplied episodes don't cover this — nothing in this collection "
    "speaks to your question, so I'd rather say so than guess."
)


@dataclass
class ChatResponse:
    """One answer plus everything needed to verify or evaluate it."""

    answer: str
    question: str
    retrieval_query: str
    mode: str
    not_covered: bool
    citations: list[Citation] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)
    retrieval: RetrievalResult | None = None
    llm: LLMResponse | None = None
    latency_ms: float = 0.0
    rewrite_llm: LLMResponse | None = None
    # Per-stage wall-clock, in ms, so a slow turn can be attributed to a
    # stage rather than guessed at.
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def citation_validity(self) -> float:
        return citation_validity_rate(self.citations)

    @property
    def estimated_cost_usd(self) -> float:
        total = self.llm.estimated_cost_usd if self.llm else 0.0
        if self.rewrite_llm:
            total += self.rewrite_llm.estimated_cost_usd
        return total

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "retrieval_query": self.retrieval_query,
            "mode": self.mode,
            "answer": self.answer,
            "not_covered": self.not_covered,
            "citations": [citation.to_dict() for citation in self.citations],
            "citation_validity": round(self.citation_validity, 4),
            "sources": self.sources,
            "retrieval": self.retrieval.to_dict() if self.retrieval else None,
            "llm": self.llm.as_dict() if self.llm else None,
            "rewrite_llm": self.rewrite_llm.as_dict() if self.rewrite_llm else None,
            "latency_ms": round(self.latency_ms, 1),
            "timings": {k: round(v, 1) for k, v in self.timings.items()},
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
        }


def _log_timings(label: str, timings: dict[str, float]) -> None:
    """One line per turn attributing the wall clock to a stage.

    Printed for every request so a slow turn is diagnosed from the log rather
    than reproduced under a profiler.
    """
    order = [
        "rewrite_ms", "dense_ms", "bm25_ms", "rrf_ms", "rerank_ms",
        "retrieval_ms", "gate_ms", "prompt_build_ms", "llm_ms",
        "citations_ms", "total_ms",
    ]
    parts = [f"{k[:-3]}={timings[k]:.0f}ms" for k in order if k in timings]
    extra = []
    if "rewrite_llm_called" in timings:
        extra.append(f"rewrite_llm={'yes' if timings['rewrite_llm_called'] else 'no'}")
    if "output_tokens" in timings:
        extra.append(f"out_tok={timings['output_tokens']:.0f}")
    if "context_chars" in timings:
        extra.append(f"ctx_chars={timings['context_chars']:.0f}")
    log.info("timings " + label, breakdown=" ".join(parts), **{"detail": " ".join(extra)})


def detect_mode(message: str) -> str:
    if _LOCATE_RE.search(message):
        return "locate"
    if _COMPARE_RE.search(message):
        return "compare"
    return "default"


class CompanionAgent:
    """Answers learner questions strictly from the ingested episodes."""

    def __init__(
        self,
        settings: Settings | None = None,
        retriever: Retriever | None = None,
        provider: LLMProvider | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.retriever = retriever or get_retriever(self.settings)
        self._provider = provider
        self.episodes: list[Episode] = load_episodes(self.settings)

    @property
    def provider(self) -> LLMProvider:
        """Built lazily so retrieval-only work needs no API key."""
        if self._provider is None:
            self._provider = get_provider("chat", self.settings)
        return self._provider

    # --- query rewriting -------------------------------------------------
    def rewrite_query(
        self, message: str, session: Session, *, offline: bool = False
    ) -> tuple[str, LLMResponse | None]:
        """Turn a follow-up into a standalone retrieval query.

        A first turn, or a message that already reads as standalone, is used
        as-is: rewriting an already-good query only risks drifting off topic.
        """
        if offline:
            # `--retrieval-only` evaluation must make no LLM call at all, so
            # its numbers are free, reproducible, and identical whether or not
            # a key happens to be configured.
            return message, None
        if not self.settings.enable_query_rewrite or not session.is_followup():
            return message, None
        if not _looks_context_dependent(message):
            return message, None
        if not provider_available("chat", self.settings):
            # No credentials configured; the raw message is a usable fallback.
            return message, None
        try:
            response = self.provider.complete(
                REWRITE_PROMPT,
                [{"role": "user",
                  "content": build_rewrite_turn(session.history(), message)}],
                max_tokens=200,
                effort=self.settings.rewrite_effort,
            )
        except ProviderError as exc:
            # A failed rewrite must not fail the turn; the raw message is a
            # workable, if weaker, query.
            log.warn("query rewrite failed; using the raw message",
                     error=str(exc))
            return message, None
        rewritten = response.text.strip().strip('"').splitlines()[0] if response.text else ""
        if not rewritten or len(rewritten) > 300:
            return message, response
        log.info("query rewritten", original=message[:60], rewritten=rewritten[:60])
        return rewritten, response

    # --- main entry point -------------------------------------------------
    def ask(
        self,
        message: str,
        session: Session | None = None,
        *,
        episode_id: str | None = None,
        mode: str | None = None,
        retrieval_only: bool = False,
    ) -> ChatResponse:
        """Answer one learner message.

        With ``retrieval_only`` the turn stops after retrieval and the
        evidence gate. No LLM is called, so retrieval quality and refusal
        behaviour can be measured with no API key and no cost.
        """
        session = session or Session()
        started = time.perf_counter()
        message = (message or "").strip()
        if not message:
            return ChatResponse(
                answer="I didn't catch a question there — what would you like to know?",
                question=message,
                retrieval_query=message,
                mode="default",
                not_covered=False,
                latency_ms=0.0,
            )

        timings: dict[str, float] = {}
        resolved_mode = mode or detect_mode(message)

        mark = time.perf_counter()
        retrieval_query, rewrite_llm = self.rewrite_query(
            message, session, offline=retrieval_only
        )
        timings["rewrite_ms"] = (time.perf_counter() - mark) * 1000
        timings["rewrite_llm_called"] = float(rewrite_llm is not None)
        effective_episode = episode_id or session.episode_filter

        mark = time.perf_counter()
        retrieval = self.retriever.retrieve(
            retrieval_query,
            episode_id=effective_episode,
            mode="compare" if resolved_mode == "compare" else "default",
        )
        timings["retrieval_ms"] = (time.perf_counter() - mark) * 1000
        for key in ("dense_ms", "bm25_ms", "rrf_ms", "rerank_ms"):
            if key in retrieval.diagnostics:
                timings[key] = retrieval.diagnostics[key]

        # Evidence gate: refuse before calling the model when nothing
        # sufficiently relevant came back. This makes the refusal path cheap
        # and independent of whether the model chooses to comply.
        mark = time.perf_counter()
        gated = self._below_evidence_threshold(retrieval)
        timings["gate_ms"] = (time.perf_counter() - mark) * 1000
        if gated:
            log.info(
                "no sufficient evidence; refusing without an LLM call",
                query=retrieval_query[:70],
                top_score=round(retrieval.top_score, 4),
                max_similarity=retrieval.diagnostics.get("max_dense_similarity"),
            )
            timings["total_ms"] = (time.perf_counter() - started) * 1000
            response = ChatResponse(
                answer=NOT_COVERED_MESSAGE,
                question=message,
                retrieval_query=retrieval_query,
                mode=resolved_mode,
                not_covered=True,
                retrieval=retrieval,
                rewrite_llm=rewrite_llm,
                latency_ms=timings["total_ms"],
                timings=timings,
            )
            _log_timings("refused (no LLM call)", timings)
            session.add(_turn_from(response))
            return response

        if retrieval_only:
            timings["total_ms"] = (time.perf_counter() - started) * 1000
            response = ChatResponse(
                answer="",
                question=message,
                retrieval_query=retrieval_query,
                mode=resolved_mode,
                not_covered=False,
                retrieval=retrieval,
                rewrite_llm=rewrite_llm,
                latency_ms=timings["total_ms"],
                timings=timings,
            )
            session.add(_turn_from(response))
            return response

        mark = time.perf_counter()
        context_chunks = cap_context(
            retrieval.chunks, self.settings.max_context_chars
        )
        if len(context_chunks) < len(retrieval.chunks):
            log.info(
                "context capped",
                kept=len(context_chunks),
                dropped=len(retrieval.chunks) - len(context_chunks),
                limit=self.settings.max_context_chars,
            )
        user_turn = build_user_turn(
            message,
            context_chunks,
            mode=resolved_mode,
            retrieval_query=retrieval_query,
        )
        messages = [*session.history(limit=4), {"role": "user", "content": user_turn}]
        timings["prompt_build_ms"] = (time.perf_counter() - mark) * 1000
        timings["context_chars"] = float(
            sum(len(c.chunk.text) for c in context_chunks)
        )

        mark = time.perf_counter()
        llm = self.provider.complete(
            SYSTEM_PROMPT,
            messages,
            max_tokens=self.settings.chat_max_tokens,
            effort=self.settings.chat_effort,
        )
        timings["llm_ms"] = (time.perf_counter() - mark) * 1000
        timings["output_tokens"] = float(llm.output_tokens)

        mark = time.perf_counter()
        answer, not_covered = _strip_marker(llm.text)
        citations = validate_citations(
            extract_citations(answer), context_chunks, self.episodes
        )
        if any(not citation.valid for citation in citations):
            invalid = [c for c in citations if not c.valid]
            log.warn(
                "dropping citations that failed validation",
                count=len(invalid),
                reasons="; ".join(sorted({c.reason or "" for c in invalid})),
            )
            answer = strip_invalid_citations(answer, citations)

        response = ChatResponse(
            answer=answer.strip(),
            question=message,
            retrieval_query=retrieval_query,
            mode=resolved_mode,
            not_covered=not_covered,
            citations=citations,
            sources=[] if not_covered else format_sources(context_chunks, citations),
            retrieval=retrieval,
            llm=llm,
            rewrite_llm=rewrite_llm,
            latency_ms=(time.perf_counter() - started) * 1000,
            timings=timings,
        )
        timings["citations_ms"] = (time.perf_counter() - mark) * 1000
        timings["total_ms"] = response.latency_ms
        _log_timings("answered", timings)
        log.info(
            "answered",
            mode=resolved_mode,
            not_covered=not_covered,
            citations=len(citations),
            citation_validity=round(response.citation_validity, 2),
            model=llm.model,
            latency_ms=round(response.latency_ms, 1),
            cost_usd=round(response.estimated_cost_usd, 5),
        )
        session.add(_turn_from(response))
        return response

    def ask_stream(
        self,
        message: str,
        session: Session | None = None,
        *,
        episode_id: str | None = None,
        mode: str | None = None,
    ) -> Iterator[str | ChatResponse]:
        """Answer one message, yielding text as it is generated.

        Yields answer deltas, then exactly one ``ChatResponse`` carrying the
        validated citations, sources and timings. Every guard is identical to
        ``ask``: the evidence gate still refuses before any model call, and
        citations are still validated against the passages the model saw —
        streaming changes when bytes leave, not what is allowed to be said.

        A refusal yields no deltas at all, just the final response.
        """
        session = session or Session()
        started = time.perf_counter()
        message = (message or "").strip()
        if not message:
            yield ChatResponse(
                answer="I didn't catch a question there — what would you like to know?",
                question=message, retrieval_query=message, mode="default",
                not_covered=False, latency_ms=0.0,
            )
            return

        timings: dict[str, float] = {}
        resolved_mode = mode or detect_mode(message)

        mark = time.perf_counter()
        retrieval_query, rewrite_llm = self.rewrite_query(message, session)
        timings["rewrite_ms"] = (time.perf_counter() - mark) * 1000
        timings["rewrite_llm_called"] = float(rewrite_llm is not None)
        effective_episode = episode_id or session.episode_filter

        mark = time.perf_counter()
        retrieval = self.retriever.retrieve(
            retrieval_query,
            episode_id=effective_episode,
            mode="compare" if resolved_mode == "compare" else "default",
        )
        timings["retrieval_ms"] = (time.perf_counter() - mark) * 1000
        for key in ("dense_ms", "bm25_ms", "rrf_ms", "rerank_ms"):
            if key in retrieval.diagnostics:
                timings[key] = retrieval.diagnostics[key]

        if self._below_evidence_threshold(retrieval):
            timings["total_ms"] = (time.perf_counter() - started) * 1000
            response = ChatResponse(
                answer=NOT_COVERED_MESSAGE, question=message,
                retrieval_query=retrieval_query, mode=resolved_mode,
                not_covered=True, retrieval=retrieval, rewrite_llm=rewrite_llm,
                latency_ms=timings["total_ms"], timings=timings,
            )
            _log_timings("refused (no LLM call)", timings)
            session.add(_turn_from(response))
            yield response
            return

        context_chunks = cap_context(
            retrieval.chunks, self.settings.max_context_chars
        )
        user_turn = build_user_turn(
            message, context_chunks, mode=resolved_mode,
            retrieval_query=retrieval_query,
        )
        messages = [*session.history(limit=4), {"role": "user", "content": user_turn}]
        timings["context_chars"] = float(
            sum(len(c.chunk.text) for c in context_chunks)
        )

        mark = time.perf_counter()
        llm: LLMResponse | None = None
        emitted = 0
        for event in self.provider.complete_stream(
            SYSTEM_PROMPT, messages,
            max_tokens=self.settings.chat_max_tokens,
            effort=self.settings.chat_effort,
        ):
            if isinstance(event, LLMResponse):
                llm = event
                break
            # The refusal marker is an internal signal, never shown. It only
            # ever appears at the very start, so holding back the first chunk
            # until it is longer than the marker is enough to keep it hidden
            # without buffering the whole answer.
            emitted += len(event)
            if emitted <= len(NOT_COVERED_MARKER) + 2:
                continue
            yield event
        timings["llm_ms"] = (time.perf_counter() - mark) * 1000

        if llm is None:  # pragma: no cover - provider contract violation
            raise ProviderError("The model stream ended without a response.")
        timings["output_tokens"] = float(llm.output_tokens)

        mark = time.perf_counter()
        answer, not_covered = _strip_marker(llm.text)
        citations = validate_citations(
            extract_citations(answer), context_chunks, self.episodes
        )
        if any(not citation.valid for citation in citations):
            answer = strip_invalid_citations(answer, citations)

        response = ChatResponse(
            answer=answer.strip(), question=message,
            retrieval_query=retrieval_query, mode=resolved_mode,
            not_covered=not_covered, citations=citations,
            sources=[] if not_covered else format_sources(context_chunks, citations),
            retrieval=retrieval, llm=llm, rewrite_llm=rewrite_llm,
            latency_ms=(time.perf_counter() - started) * 1000, timings=timings,
        )
        timings["citations_ms"] = (time.perf_counter() - mark) * 1000
        timings["total_ms"] = response.latency_ms
        _log_timings("answered (streamed)", timings)
        log.info(
            "answered", mode=resolved_mode, not_covered=not_covered,
            citations=len(citations),
            citation_validity=round(response.citation_validity, 2),
            model=llm.model, latency_ms=round(response.latency_ms, 1),
            cost_usd=round(response.estimated_cost_usd, 5), streamed=True,
        )
        session.add(_turn_from(response))
        yield response

    def _below_evidence_threshold(self, retrieval: RetrievalResult) -> bool:
        """True when retrieval is too weak to support any grounded answer.

        The fused RRF score cannot be used: it is a rank statistic, so it is
        just as high for the best of a bad candidate set as for the best of a
        good one. Cross-encoder relevance is preferred where available because
        it judges whether a passage *answers* the question; bi-encoder cosine
        only measures topical proximity, and physics-adjacent questions the
        episodes never cover score as highly on it as questions they do.
        Cosine remains the fallback when the reranker could not load.
        """
        if retrieval.is_empty:
            return True
        rerank_score = retrieval.diagnostics.get("max_rerank_score")
        gate = retrieval.diagnostics.get("relevance_gate")
        if rerank_score is not None and gate is not None:
            return rerank_score < gate
        best = retrieval.diagnostics.get("max_dense_similarity")
        if best is None:
            return False
        return best < self.settings.min_relevance

    def locate(
        self, message: str, session: Session | None = None,
        *, episode_id: str | None = None,
    ) -> ChatResponse:
        """Answer a 'take me to the audio' request."""
        return self.ask(message, session, episode_id=episode_id, mode="locate")


def _turn_from(response: ChatResponse) -> Turn:
    return Turn(
        question=response.question,
        answer=response.answer,
        retrieval_query=response.retrieval_query,
        mode=response.mode,
        retrieved=response.retrieval.chunks if response.retrieval else [],
        sources=response.sources,
        not_covered=response.not_covered,
        latency_ms=response.latency_ms,
    )


def _strip_marker(text: str) -> tuple[str, bool]:
    """Detect and remove the not-covered marker the prompt asks for."""
    stripped = text.strip()
    if stripped.upper().startswith(NOT_COVERED_MARKER):
        remainder = stripped[len(NOT_COVERED_MARKER):].lstrip(" :\n-")
        return (remainder or NOT_COVERED_MESSAGE), True
    if NOT_COVERED_MARKER in stripped:
        return stripped.replace(NOT_COVERED_MARKER, "").strip(), True
    return stripped, False


_CONTEXT_WORDS = re.compile(
    r"\b(that|this|those|these|it|its|they|them|he|she|his|her|the (?:second|first|"
    r"third|last|previous|other|same)|again|instead|more simply|simpler|"
    r"explain (?:that|it)|why (?:is|was) (?:that|it))\b",
    re.IGNORECASE,
)


def _looks_context_dependent(message: str) -> bool:
    """Heuristic: does this message need the conversation to make sense?

    Rewriting every follow-up would waste a call and risks corrupting a
    question that was already fine, so only messages carrying an unresolved
    reference (or too short to stand alone) are rewritten.
    """
    if len(message.split()) <= 4:
        return True
    return bool(_CONTEXT_WORDS.search(message))
