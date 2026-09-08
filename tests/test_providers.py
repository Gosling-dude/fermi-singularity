"""Provider layer, including OpenRouter.

The OpenRouter tests use a fake OpenAI-compatible client, so they assert the
exact request shape and response handling without a network call or a key.
"""

from __future__ import annotations

import types

import pytest

from companion.chat.providers import (
    _extract_json,
    _openrouter_cost,
    estimate_cost,
    get_provider,
    provider_available,
    AnthropicProvider,
    OpenAIProvider,
    OpenRouterProvider,
    _PROVIDERS,
)
from companion.config import Settings
from companion.errors import ConfigError, ProviderError


# --- registry ------------------------------------------------------------
def test_all_three_providers_are_registered():
    assert set(_PROVIDERS) == {"openrouter", "anthropic", "openai"}
    assert _PROVIDERS["openrouter"] is OpenRouterProvider
    assert _PROVIDERS["anthropic"] is AnthropicProvider
    assert _PROVIDERS["openai"] is OpenAIProvider


def _default(field: str):
    """The value shipped in the code, independent of any local .env."""
    return Settings.model_fields[field].default


def test_openrouter_is_the_shipped_default_for_chat_and_judge():
    assert _default("chat_provider") == "openrouter"
    assert _default("judge_provider") == "openrouter"
    assert "/" in _default("chat_model"), "OpenRouter ids are namespaced"
    assert "/" in _default("judge_model")


def test_judge_uses_a_different_vendor_than_chat():
    """Guards the anti-self-preference choice against a careless edit."""
    chat_vendor = _default("chat_model").split("/")[0]
    judge_vendor = _default("judge_model").split("/")[0]
    assert chat_vendor != judge_vendor


def test_one_key_serves_chat_and_judge():
    settings = Settings(
        CHAT_PROVIDER="openrouter", JUDGE_PROVIDER="openrouter",
        OPENROUTER_API_KEY="sk-or-test",
    )
    assert provider_available("chat", settings)
    assert provider_available("judge", settings)


def test_typo_in_chat_provider_gives_an_actionable_error(monkeypatch):
    """A bad .env value must not surface as a pydantic traceback."""
    from companion.config import get_settings

    monkeypatch.setenv("CHAT_PROVIDER", "antropic")  # deliberate typo
    get_settings.cache_clear()
    try:
        with pytest.raises(ConfigError) as exc:
            get_settings()
        assert "CHAT_PROVIDER" in exc.value.message
        assert "openrouter, anthropic, openai" in (exc.value.remedy or "")
    finally:
        get_settings.cache_clear()


def test_get_provider_rejects_an_unregistered_provider():
    """Defence in depth for settings built programmatically."""
    settings = Settings(OPENROUTER_API_KEY="x")
    object.__setattr__(settings, "chat_provider", "notreal")
    with pytest.raises(ConfigError) as exc:
        get_provider("chat", settings)
    assert "openrouter" in (exc.value.remedy or "")


def test_missing_openrouter_key_is_actionable():
    with pytest.raises(ConfigError) as exc:
        get_provider("chat", Settings(CHAT_PROVIDER="openrouter",
                                      OPENROUTER_API_KEY=None))
    assert "OPENROUTER_API_KEY" in exc.value.message
    assert "openrouter.ai" in (exc.value.remedy or "")


def test_no_anthropic_key_is_needed_when_using_openrouter():
    settings = Settings(
        CHAT_PROVIDER="openrouter", JUDGE_PROVIDER="openrouter",
        OPENROUTER_API_KEY="sk-or-test", ANTHROPIC_API_KEY=None,
        OPENAI_API_KEY=None,
    )
    assert provider_available("chat", settings)
    assert provider_available("judge", settings)
    assert settings.api_key_for("anthropic") is None


# --- OpenRouter request/response shape -----------------------------------
class _FakeCompletions:
    def __init__(self, response, recorder):
        self._response = response
        self._recorder = recorder

    def create(self, **kwargs):
        self._recorder.update(kwargs)
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


def _fake_response(content="hello", cost=0.00123, model="anthropic/claude-sonnet-5"):
    usage = types.SimpleNamespace(prompt_tokens=120, completion_tokens=45, cost=cost)
    choice = types.SimpleNamespace(
        message=types.SimpleNamespace(content=content), finish_reason="stop"
    )
    return types.SimpleNamespace(choices=[choice], usage=usage, model=model)


@pytest.fixture
def openrouter(monkeypatch):
    """Build an OpenRouterProvider whose HTTP client is a stub."""

    def build(response, **overrides):
        recorder: dict = {}
        settings = Settings(OPENROUTER_API_KEY="sk-or-test", **overrides)
        provider = OpenRouterProvider.__new__(OpenRouterProvider)
        provider.model = settings.chat_model
        provider.settings = settings
        import openai

        provider._sdk = openai
        provider._client = types.SimpleNamespace(
            chat=types.SimpleNamespace(
                completions=_FakeCompletions(response, recorder)
            )
        )
        return provider, recorder

    return build


def test_request_shape_matches_openrouter(openrouter):
    provider, sent = openrouter(_fake_response())
    provider.complete("SYSTEM", [{"role": "user", "content": "hi"}],
                      max_tokens=1234, effort="high")
    assert sent["model"] == "anthropic/claude-sonnet-5"
    assert sent["max_tokens"] == 1234, "OpenRouter uses max_tokens"
    assert sent["messages"][0] == {"role": "system", "content": "SYSTEM"}
    assert sent["messages"][1] == {"role": "user", "content": "hi"}
    # Actual spend must be requested explicitly.
    assert sent["extra_body"]["usage"] == {"include": True}


def test_reasoning_effort_is_not_sent_by_default(openrouter):
    """Most OpenRouter models reject a reasoning field; default must be safe."""
    provider, sent = openrouter(_fake_response())
    provider.complete("s", [{"role": "user", "content": "x"}], effort="high")
    assert "reasoning" not in sent["extra_body"]


def test_reasoning_effort_is_sent_when_enabled(openrouter):
    provider, sent = openrouter(
        _fake_response(), OPENROUTER_SEND_REASONING=True
    )
    provider.complete("s", [{"role": "user", "content": "x"}], effort="high")
    assert sent["extra_body"]["reasoning"] == {"effort": "high"}


def test_response_is_parsed_with_tokens_and_real_cost(openrouter):
    provider, _ = openrouter(_fake_response(content="  an answer  ", cost=0.0042))
    result = provider.complete("s", [{"role": "user", "content": "x"}])
    assert result.text == "an answer"
    assert result.provider == "openrouter"
    assert result.input_tokens == 120 and result.output_tokens == 45
    assert result.stop_reason == "stop"
    # Provider-reported spend wins over the local pricing table.
    assert result.reported_cost_usd == 0.0042
    assert result.estimated_cost_usd == 0.0042
    assert result.as_dict()["cost_source"] == "provider"


def test_falls_back_to_the_pricing_table_when_cost_is_absent(openrouter):
    provider, _ = openrouter(_fake_response(cost=None))
    result = provider.complete("s", [{"role": "user", "content": "x"}])
    assert result.reported_cost_usd is None
    assert result.as_dict()["cost_source"] == "estimated"


def test_body_level_error_is_surfaced(openrouter):
    """OpenRouter reports upstream failures on a 200 with no choices."""
    broken = types.SimpleNamespace(
        choices=[], usage=None, model="x",
        error={"message": "upstream is down"},
    )
    provider, _ = openrouter(broken)
    with pytest.raises(ProviderError) as exc:
        provider.complete("s", [{"role": "user", "content": "x"}])
    assert "no completion" in exc.value.message


def test_unknown_model_error_is_actionable(openrouter):
    import httpx
    import openai

    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    response = httpx.Response(404, request=request, json={"error": "no such model"})
    error = openai.NotFoundError("not found", response=response, body=None)
    provider, _ = openrouter(error)
    with pytest.raises(ConfigError) as exc:
        provider.complete("s", [{"role": "user", "content": "x"}])
    assert "make models" in (exc.value.remedy or "")


def test_rate_limit_mentions_credits(openrouter):
    import httpx
    import openai

    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    response = httpx.Response(429, request=request, json={"error": "rate limited"})
    provider, _ = openrouter(
        openai.RateLimitError("rate limited", response=response, body=None)
    )
    with pytest.raises(ProviderError) as exc:
        provider.complete("s", [{"role": "user", "content": "x"}])
    assert "credit" in (exc.value.remedy or "").lower()


# --- cost helper ---------------------------------------------------------
@pytest.mark.parametrize(
    "usage,expected",
    [
        (None, None),
        (types.SimpleNamespace(cost=0.005), 0.005),
        (types.SimpleNamespace(cost=None), None),
        (types.SimpleNamespace(cost="0.25"), 0.25),
        (types.SimpleNamespace(cost="nonsense"), None),
        (types.SimpleNamespace(), None),
    ],
)
def test_openrouter_cost_extraction(usage, expected):
    assert _openrouter_cost(usage) == expected


def test_estimate_cost_returns_zero_for_unknown_models():
    assert estimate_cost("some/unlisted-model", 1000, 1000) == 0.0


# --- shared helpers ------------------------------------------------------
def test_extract_json_handles_fenced_output():
    assert _extract_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_extract_json_handles_prose_around_the_object():
    assert _extract_json('Sure! {"a": {"b": 2}} done.') == {"a": {"b": 2}}


def test_extract_json_rejects_non_json():
    with pytest.raises(ProviderError):
        _extract_json("no json here at all")
