"""Citation validation is what makes an answer checkable."""

from __future__ import annotations

from companion.chat.citations import (
    citation_validity_rate, extract_citations, format_sources,
    strip_invalid_citations, validate_citations,
)


def test_extracts_canonical_form():
    text = 'Planck did X (Ep. "The Birth of the Quantum" 0:30-1:10).'
    citations = extract_citations(text)
    assert len(citations) == 1
    assert citations[0].episode_title == "The Birth of the Quantum"
    assert citations[0].start == 30.0
    assert citations[0].end == 70.0


def test_accepts_en_dash_and_missing_prefix():
    text = ('A (Ep. "The Birth of the Quantum" 0:30–1:10) '
            'B ("Shannon and the Birth of Information" 0:10-0:50)')
    assert len(extract_citations(text)) == 2


def test_ignores_non_citation_parentheses():
    assert extract_citations("This is a note (see below) and (1:00-2:00).") == []


def test_valid_citation_passes(retrieved, episodes):
    citations = validate_citations(
        extract_citations('X (Ep. "The Birth of the Quantum" 0:10-1:00)'),
        retrieved, episodes,
    )
    assert citations[0].valid
    assert citations[0].episode_id == "quantum"


def test_unknown_episode_is_rejected(retrieved, episodes):
    citations = validate_citations(
        extract_citations('X (Ep. "Some Other Show" 0:10-1:00)'), retrieved, episodes
    )
    assert not citations[0].valid
    assert "not in the retrieved context" in citations[0].reason


def test_timestamp_beyond_episode_is_rejected(retrieved, episodes):
    citations = validate_citations(
        extract_citations('X (Ep. "The Birth of the Quantum" 59:00-59:30)'),
        retrieved, episodes,
    )
    assert not citations[0].valid


def test_timestamp_not_overlapping_any_passage_is_rejected(retrieved, episodes):
    # 3:10-3:20 is inside the episode but no retrieved chunk covers it.
    citations = validate_citations(
        extract_citations('X (Ep. "The Birth of the Quantum" 3:10-3:20)'),
        retrieved, episodes,
    )
    assert not citations[0].valid
    assert "does not overlap" in citations[0].reason


def test_inverted_interval_is_rejected(retrieved, episodes):
    citations = validate_citations(
        extract_citations('X (Ep. "The Birth of the Quantum" 1:00-0:30)'),
        retrieved, episodes,
    )
    assert not citations[0].valid
    assert "malformed" in citations[0].reason


def test_validity_rate():
    assert citation_validity_rate([]) == 1.0


def test_strip_removes_only_invalid(retrieved, episodes):
    text = ('Good (Ep. "The Birth of the Quantum" 0:10-1:00) '
            'Bad (Ep. "Fake Show" 0:10-1:00)')
    citations = validate_citations(extract_citations(text), retrieved, episodes)
    cleaned = strip_invalid_citations(text, citations)
    assert 'The Birth of the Quantum" 0:10-1:00' in cleaned
    assert "Fake Show" not in cleaned


def test_sources_reflect_what_was_cited(retrieved, episodes):
    citations = validate_citations(
        extract_citations('X (Ep. "The Birth of the Quantum" 0:10-1:00)'),
        retrieved, episodes,
    )
    sources = format_sources(retrieved, citations)
    assert sources
    assert all(s["episode_title"] == "The Birth of the Quantum" for s in sources)
