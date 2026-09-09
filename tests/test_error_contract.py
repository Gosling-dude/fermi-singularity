"""Every response the browser can receive must be JSON.

The UI reported "Unexpected end of JSON input" and "Unexpected token '<'"
because it called ``response.json()`` on bodies that were not JSON: an empty
body, and an HTML error page. The frontend no longer parses blind, and these
tests pin the other half of the contract — that the server never emits a
non-JSON body in the first place, whatever goes wrong underneath.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from companion.errors import ProviderError


@pytest.fixture
def client():
    from companion.interface.web import app

    # raise_server_exceptions=False so the app's own 500 handler runs, which
    # is exactly what a browser would receive.
    return TestClient(app, raise_server_exceptions=False)


def _assert_json_error(response, status: int) -> dict:
    """Every error must be JSON and carry the agreed keys."""
    assert response.status_code == status
    assert response.headers["content-type"].startswith("application/json")
    assert response.content, "body was empty — json() would throw"
    body = json.loads(response.content)          # must not raise
    assert "error" in body and "detail" in body
    assert isinstance(body["error"], str) and body["error"]
    return body


# --- the happy path still works ------------------------------------------
def test_successful_response_is_json(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    assert json.loads(response.content)["status"] in {"ok", "not_ingested"}


def test_health_probe_is_json(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert json.loads(response.content) == {"status": "ok"}


# --- provider failure ------------------------------------------------------
def test_provider_failure_is_json_not_html(client, monkeypatch):
    """A dead upstream must not reach the browser as an unparseable body."""
    from companion.chat import agent as agent_module

    def boom(*_args, **_kwargs):
        raise ProviderError("OpenRouter is unreachable.", "Check your network.")

    monkeypatch.setattr(agent_module.CompanionAgent, "ask", boom)
    body = _assert_json_error(
        client.post("/api/chat", json={"message": "what is the dirac equation"}), 503
    )
    assert "OpenRouter" in body["error"]
    assert body["remedy"] == "Check your network."


def test_provider_timeout_is_json(client, monkeypatch):
    from companion.chat import agent as agent_module

    def slow(*_args, **_kwargs):
        raise ProviderError("OpenRouter request timed out after 60s.", "Re-run.")

    monkeypatch.setattr(agent_module.CompanionAgent, "ask", slow)
    body = _assert_json_error(client.post("/api/chat", json={"message": "hi"}), 503)
    assert "timed out" in body["error"]


def test_malformed_upstream_response_is_json(client, monkeypatch):
    """OpenRouter can return 200 with no choices; that must surface as JSON."""
    from companion.chat import agent as agent_module

    def empty(*_args, **_kwargs):
        raise ProviderError(
            "OpenRouter returned no completion for 'x': no choices returned"
        )

    monkeypatch.setattr(agent_module.CompanionAgent, "ask", empty)
    body = _assert_json_error(client.post("/api/chat", json={"message": "hi"}), 503)
    assert "no completion" in body["error"]


# --- unexpected 500 --------------------------------------------------------
def test_unexpected_exception_is_json_500(client, monkeypatch):
    """The backstop: an ordinary bug must not become a plain-text body."""
    from companion.chat import agent as agent_module

    def explode(*_args, **_kwargs):
        raise RuntimeError("something nobody anticipated")

    monkeypatch.setattr(agent_module.CompanionAgent, "ask", explode)
    body = _assert_json_error(client.post("/api/chat", json={"message": "hi"}), 500)
    assert "RuntimeError" in body["detail"]
    # The traceback belongs in the log, not the response.
    assert "Traceback" not in json.dumps(body)


# --- invalid / malformed requests -----------------------------------------
def test_invalid_json_body_is_json_422(client):
    response = client.post(
        "/api/chat",
        content=b"{not valid json",
        headers={"Content-Type": "application/json"},
    )
    _assert_json_error(response, 422)


def test_missing_required_field_is_json_422(client):
    _assert_json_error(client.post("/api/chat", json={}), 422)


def test_unknown_route_is_json_404(client):
    _assert_json_error(client.get("/api/does-not-exist"), 404)


def test_unknown_episode_is_json_404(client):
    _assert_json_error(client.get("/api/episodes/nope/audio"), 404)


def test_wrong_method_is_json_405(client):
    _assert_json_error(client.get("/api/chat"), 405)


# --- the one endpoint that is legitimately HTML ---------------------------
def test_index_is_html_and_the_ui_never_parses_it(client):
    """/ is HTML by design — the UI must only ever json() the /api/* routes."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")

    page = response.text
    assert "async function readJson" in page, "the guard must ship with the UI"
    # Every fetch must go through the guard rather than .json() directly.
    assert ".then(r => r.json())" not in page
    assert "await res.json()" not in page
