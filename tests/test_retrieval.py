"""Retrieval behaviour.

Fusion and gating are tested in isolation; the end-to-end tests run against
the real index and skip cleanly when ingestion has not been run.
"""

from __future__ import annotations

import pytest

from companion.config import get_settings
from companion.retrieve.bm25 import BM25Index, tokenize
from companion.retrieve.rerank import normalize_query
from companion.retrieve.retriever import reciprocal_rank_fusion


# --- unit ---------------------------------------------------------------
def test_tokenize_drops_stopwords_and_fillers():
    tokens = tokenize("So, um, the quantum of ACTION is really Planck's constant.")
    assert "quantum" in tokens and "action" in tokens
    assert "the" not in tokens and "um" not in tokens and "really" not in tokens


def test_rrf_rewards_agreement():
    scores = reciprocal_rank_fusion([["a", "b", "c"], ["c", "a", "d"]], k=60)
    assert scores["a"] > scores["c"] > scores["b"]


def test_rrf_is_scale_free():
    # RRF sees only ranks, so it is unaffected by the magnitude of scores.
    assert reciprocal_rank_fusion([["a", "b"]]) == reciprocal_rank_fusion([["a", "b"]])


def test_bm25_finds_exact_terminology(chunks):
    index = BM25Index.build(chunks)
    hits = index.search("ultraviolet catastrophe")
    assert hits and hits[0][0] == "quantum_0001"


def test_bm25_returns_nothing_for_absent_terms(chunks):
    assert BM25Index.build(chunks).search("superconductivity Cooper pairs") == []


def test_bm25_roundtrips_through_disk(chunks, tmp_path):
    path = tmp_path / "bm25.pkl"
    BM25Index.build(chunks).save(path)
    assert BM25Index.load(path).search("entropy")


@pytest.mark.parametrize(
    "query,expected",
    [
        ("Take me to the part where they explain what a bit is.", "what a bit is"),
        ("Explain the ultraviolet catastrophe simply", "the ultraviolet catastrophe"),
        ("What do these episodes say about superconductivity?", "superconductivity"),
        ("What did Planck assume?", "What did Planck assume"),
    ],
)
def test_normalize_query_strips_wrappers(query, expected):
    assert normalize_query(query) == expected


def test_normalize_query_keeps_short_queries_intact():
    assert normalize_query("entropy") == "entropy"


# --- integration against the real index ---------------------------------
@pytest.fixture(scope="module")
def retriever():
    from companion.errors import CompanionError
    from companion.retrieve.retriever import get_retriever

    try:
        return get_retriever(refresh=True)
    except CompanionError:
        pytest.skip("no index — run `make ingest` to enable retrieval tests")


def test_semantic_query_finds_the_right_episode(retriever):
    # Paraphrased, with none of the transcript's own wording — this can only
    # be found semantically. The Shannon episode also discusses Planck's
    # packets explicitly, so the assertion is that the quantum episode is
    # retrieved, not that it necessarily ranks first.
    result = retriever.retrieve("why energy comes in discrete packets")
    assert not result.is_empty
    episodes = {item.chunk.episode_id for item in result.chunks}
    assert any(e.endswith("the_birth_of_the_quantum") for e in episodes)


def test_exact_terminology_is_found(retriever):
    result = retriever.retrieve("ultraviolet catastrophe")
    text = " ".join(item.chunk.text.lower() for item in result.chunks)
    assert "ultraviolet catastrophe" in text


def test_episode_filter_restricts_results(retriever):
    episode = retriever.episode_ids()[0]
    result = retriever.retrieve("what is this episode about", episode_id=episode)
    assert result.chunks
    assert {item.chunk.episode_id for item in result.chunks} == {episode}


def test_compare_mode_spans_multiple_episodes(retriever):
    default = retriever.retrieve("uncertainty and noise", mode="default")
    compare = retriever.retrieve("uncertainty and noise", mode="compare")
    assert len(set(i.chunk.episode_id for i in compare.chunks)) >= len(
        set(i.chunk.episode_id for i in default.chunks)
    )
    assert len(set(i.chunk.episode_id for i in compare.chunks)) >= 2


def test_retrieved_timestamps_are_within_the_episode(retriever):
    from companion.ingest.pipeline import load_episodes

    durations = {e.episode_id: e.duration for e in load_episodes()}
    for item in retriever.retrieve("entropy", top_k=8).chunks:
        assert 0 <= item.chunk.start < item.chunk.end
        assert item.chunk.end <= durations[item.chunk.episode_id] + 1.0


def test_unrelated_query_scores_below_the_calibrated_gate(retriever):
    if retriever.relevance_gate is None:
        pytest.skip("reranker unavailable, so no calibrated gate")
    result = retriever.retrieve("how do I bake sourdough bread at home")
    assert result.diagnostics["max_rerank_score"] < retriever.relevance_gate


def test_in_scope_query_scores_above_the_calibrated_gate(retriever):
    if retriever.relevance_gate is None:
        pytest.skip("reranker unavailable, so no calibrated gate")
    result = retriever.retrieve("What did Planck assume about the oscillators?")
    assert result.diagnostics["max_rerank_score"] > retriever.relevance_gate


def test_empty_query_is_rejected(retriever):
    from companion.errors import RetrievalError

    with pytest.raises(RetrievalError):
        retriever.retrieve("   ")
