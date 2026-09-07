"""Agent behaviour, driven by a scripted provider so no API key is needed."""

from __future__ import annotations

import pytest

from companion.chat.agent import CompanionAgent, detect_mode, _strip_marker
from companion.chat.prompt import NOT_COVERED_MARKER, build_user_turn, render_context
from companion.chat.session import Session
from companion.errors import CompanionError


# --- pure functions -----------------------------------------------------
@pytest.mark.parametrize(
    "message,expected",
    [
        ("Compare what these episodes say about noise", "compare"),
        ("How do these episodes differ on entropy?", "compare"),
        ("Take me to the part where they explain entropy", "locate"),
        ("Where in the audio do they discuss Bell?", "locate"),
        ("What is entropy?", "default"),
    ],
)
def test_detect_mode(message, expected):
    assert detect_mode(message) == expected


def test_strip_marker_detects_refusal():
    answer, refused = _strip_marker(f"{NOT_COVERED_MARKER}\nThe episodes don't cover it.")
    assert refused and answer == "The episodes don't cover it."


def test_strip_marker_leaves_normal_answers():
    answer, refused = _strip_marker("Planck introduced the quantum.")
    assert not refused and answer == "Planck introduced the quantum."


def test_context_labels_every_passage_with_episode_and_timestamp(retrieved):
    context = render_context(retrieved)
    assert 'Episode: "The Birth of the Quantum"' in context
    assert "Timestamp: 0:00–1:30" in context
    assert context.count("[Passage") == len(retrieved)


def test_context_is_explicit_when_nothing_retrieved():
    assert "No transcript passages" in render_context([])


def test_user_turn_carries_the_grounding_task(retrieved):
    turn = build_user_turn("Compare them", retrieved, mode="compare")
    assert "comparison question" in turn
    assert "LEARNER'S QUESTION: Compare them" in turn


# --- agent with a scripted provider -------------------------------------
@pytest.fixture
def agent(scripted_provider, monkeypatch, episodes, retrieved):
    """An agent whose retriever and provider are both stubbed."""
    from companion.config import get_settings

    class StubRetriever:
        def __init__(self):
            self.relevance_gate = -9.98

        def retrieve(self, query, *, top_k=None, episode_id=None, mode="default"):
            from companion.retrieve.retriever import RetrievalResult

            return RetrievalResult(
                query=query, chunks=retrieved, mode=mode,
                dense_hits=3, bm25_hits=2,
                diagnostics={"max_dense_similarity": 0.8,
                             "max_rerank_score": 4.0,
                             "relevance_gate": -9.98},
            )

    def build(replies):
        monkeypatch.setattr(
            "companion.chat.agent.load_episodes", lambda *a, **k: episodes
        )
        return CompanionAgent(
            get_settings(), retriever=StubRetriever(),
            provider=scripted_provider(replies),
        )

    return build


def test_grounded_answer_keeps_valid_citations(agent):
    reply = ('Planck assumed discrete energies '
             '(Ep. "The Birth of the Quantum" 0:10-1:00).')
    response = agent([reply]).ask("What did Planck assume?")
    assert not response.not_covered
    assert len(response.citations) == 1 and response.citations[0].valid
    assert response.sources
    assert response.citation_validity == 1.0


def test_fabricated_citation_is_stripped(agent):
    reply = ('Real (Ep. "The Birth of the Quantum" 0:10-1:00) and '
             'invented (Ep. "A Show That Does Not Exist" 2:00-3:00).')
    response = agent([reply]).ask("What did Planck assume?")
    assert "A Show That Does Not Exist" not in response.answer
    assert 'The Birth of the Quantum" 0:10-1:00' in response.answer
    assert response.citation_validity == 0.5


def test_model_refusal_marker_is_honoured(agent):
    response = agent([f"{NOT_COVERED_MARKER}\nNot discussed."]).ask("Dark matter?")
    assert response.not_covered
    assert response.sources == []


def test_evidence_gate_refuses_without_calling_the_model(agent, monkeypatch):
    built = agent(["should never be used"])

    from companion.retrieve.retriever import RetrievalResult

    def nothing_relevant(query, **kwargs):
        return RetrievalResult(
            query=query, chunks=[],
            diagnostics={"max_dense_similarity": 0.05,
                         "max_rerank_score": -12.0, "relevance_gate": -9.98},
        )

    monkeypatch.setattr(built.retriever, "retrieve", nothing_relevant)
    response = built.ask("What is the capital of Australia?")
    assert response.not_covered
    assert response.llm is None, "the gate must refuse before spending a call"
    assert built._provider.calls == []


def test_low_relevance_triggers_the_gate(agent, monkeypatch):
    built = agent(["unused"])
    from companion.retrieve.retriever import RetrievalResult

    monkeypatch.setattr(
        built.retriever, "retrieve",
        lambda query, **kw: RetrievalResult(
            query=query, chunks=[],
            diagnostics={"max_rerank_score": -11.0, "relevance_gate": -9.98},
        ),
    )
    assert built.ask("sourdough bread").not_covered


def test_followup_reuses_the_session(agent):
    built = agent(["First answer.", "Second answer."])
    session = Session()
    built.ask("What is the main idea?", session)
    built.ask("Explain that more simply.", session)
    assert len(session.turns) == 2
    assert session.turns[0].answer == "First answer."


def test_history_is_passed_to_the_model_on_a_followup(agent):
    built = agent(["First answer.", "Second answer."])
    session = Session()
    built.ask("What is the main idea?", session)
    built.ask("What about the second point?", session)
    last_call = built._provider.calls[-1]
    assert any("First answer." in m["content"] for m in last_call["messages"])


def test_empty_message_is_handled_gracefully(agent):
    response = agent([]).ask("   ")
    assert not response.not_covered
    assert "question" in response.answer.lower()


def test_missing_api_key_raises_an_actionable_error(monkeypatch, episodes):
    from companion.config import Settings
    from companion.chat.providers import get_provider
    from companion.errors import ConfigError

    settings = Settings(CHAT_PROVIDER="anthropic", ANTHROPIC_API_KEY=None)
    with pytest.raises(ConfigError) as exc:
        get_provider("chat", settings)
    assert isinstance(exc.value, CompanionError)
    assert ".env" in (exc.value.remedy or "")
