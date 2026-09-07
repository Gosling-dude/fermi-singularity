"""HTTP surface. These run against the real index and skip without one."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    from companion.ingest.pipeline import load_episodes

    if not load_episodes():
        pytest.skip("no ingested episodes — run `make ingest`")
    from companion.interface.web import app

    return TestClient(app)


def test_health(client):
    payload = client.get("/api/health").json()
    assert payload["status"] in {"ok", "not_ingested"}
    assert "chat_key_present" in payload
    assert payload["retrieval_mode"] in {"hybrid", "dense"}


def test_episodes_listing(client):
    payload = client.get("/api/episodes").json()
    assert payload["episodes"]
    first = payload["episodes"][0]
    assert {"episode_id", "title", "duration", "num_chunks"} <= set(first)


def test_episode_detail_includes_an_outline(client):
    episode_id = client.get("/api/episodes").json()["episodes"][0]["episode_id"]
    payload = client.get(f"/api/episodes/{episode_id}").json()
    assert payload["outline"]
    assert payload["outline"][0]["timestamp"]


def test_unknown_episode_returns_404(client):
    assert client.get("/api/episodes/does-not-exist").status_code == 404


def test_search_returns_scored_passages(client):
    payload = client.post(
        "/api/search", json={"query": "ultraviolet catastrophe", "top_k": 3}
    ).json()
    assert 1 <= len(payload["chunks"]) <= 3
    top = payload["chunks"][0]
    assert {"episode_title", "timestamp", "score", "text"} <= set(top)


def test_search_respects_the_episode_filter(client):
    episode_id = client.get("/api/episodes").json()["episodes"][0]["episode_id"]
    payload = client.post(
        "/api/search", json={"query": "what is this about", "episode_id": episode_id}
    ).json()
    assert {c["episode_id"] for c in payload["chunks"]} == {episode_id}


def test_search_rejects_an_empty_query(client):
    assert client.post("/api/search", json={"query": ""}).status_code == 422


def test_search_rejects_an_out_of_range_top_k(client):
    assert client.post(
        "/api/search", json={"query": "entropy", "top_k": 999}
    ).status_code == 422


def test_audio_is_servable_for_seeking(client):
    episode_id = client.get("/api/episodes").json()["episodes"][0]["episode_id"]
    response = client.get(f"/api/episodes/{episode_id}/audio")
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/mpeg"


def test_chat_without_a_key_is_a_clear_503(client):
    from companion.chat.providers import provider_available

    if provider_available("chat"):
        pytest.skip("an API key is configured, so chat is expected to work")
    response = client.post("/api/chat", json={"message": "What is entropy?"})
    assert response.status_code == 503
    assert "API key" in response.json()["detail"]


def test_index_page_renders(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Fermi Podcast Companion" in response.text
