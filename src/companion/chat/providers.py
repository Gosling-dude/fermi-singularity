"""LLM provider abstraction.

Chat, query rewriting and the evaluation judge all speak to models through
``LLMProvider``, so swapping Anthropic for OpenAI is an environment change
rather than a code change. Providers are responsible for their own
credential checks, error translation, token accounting and cost estimation.

A note on determinism: current Claude models removed the sampling parameters,
and the current Anthropic SDK no longer accepts ``temperature`` at all, so the
Anthropic provider never sends it. Reproducibility for evaluation therefore
comes from fixed prompts, a pinned model id and a fixed reasoning effort, all
of which are recorded in every eval run. ``temperature`` is still forwarded to
OpenAI, which accepts it.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

from companion.config import Settings, get_settings
from companion.errors import ConfigError, ProviderError
from companion.utils.logging import get_logger

log = get_logger("llm")

Role = Literal["chat", "judge"]

# USD per 1M tokens (input, output), for providers that do not report spend.
# OpenRouter reports actual cost per call, so its models do not need an entry
# here — which also means no hard-coded price can silently go stale.
PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
}

def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimated USD for one call. Unknown models return 0.0 rather than guess."""
    rates = PRICING.get(model)
    if not rates:
        return 0.0
    return (input_tokens * rates[0] + output_tokens * rates[1]) / 1_000_000


@dataclass
class LLMResponse:
    """One completion plus the accounting needed by the eval runner."""

    text: str
    model: str
    provider: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    stop_reason: str | None = None
    # Actual spend reported by the provider, when it reports one. OpenRouter
    # does; it is preferred over the local pricing table because it cannot go
    # stale and it accounts for the model that actually served the request.
    reported_cost_usd: float | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def estimated_cost_usd(self) -> float:
        if self.reported_cost_usd is not None:
            return self.reported_cost_usd
        return estimate_cost(self.model, self.input_tokens, self.output_tokens)

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "provider": self.provider,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "latency_ms": round(self.latency_ms, 1),
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
            "cost_source": (
                "provider" if self.reported_cost_usd is not None else "estimated"
            ),
            "stop_reason": self.stop_reason,
        }


class LLMProvider(ABC):
    """Minimal surface every provider implements."""

    name: str

    def __init__(self, model: str, settings: Settings) -> None:
        self.model = model
        self.settings = settings

    @abstractmethod
    def complete(
        self,
        system: str,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 4000,
        effort: str = "medium",
        temperature: float | None = None,
    ) -> LLMResponse:
        """Return one completion. Raises ``ProviderError`` on any failure."""

    def complete_json(
        self,
        system: str,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 2000,
        effort: str = "medium",
    ) -> dict[str, Any]:
        """Complete and parse a JSON object from the response.

        Models occasionally wrap JSON in prose or a fenced block, so the
        payload is extracted by brace matching rather than a strict parse of
        the whole string.
        """
        response = self.complete(
            system, messages, max_tokens=max_tokens, effort=effort
        )
        return _extract_json(response.text)


def _extract_json(text: str) -> dict[str, Any]:
    """Pull the first balanced JSON object out of a model response."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1] if "```" in cleaned[3:] else cleaned[3:]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    start = cleaned.find("{")
    if start == -1:
        raise ProviderError(
            "Model did not return JSON.",
            "Re-run the case; if it persists, lower the judge's effort or "
            "switch JUDGE_MODEL.",
        )
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(cleaned)):
        char = cleaned[index]
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(cleaned[start : index + 1])
                except json.JSONDecodeError as exc:
                    raise ProviderError(f"Malformed JSON from model: {exc}") from exc
    raise ProviderError("Model returned an unterminated JSON object.")


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, model: str, settings: Settings) -> None:
        super().__init__(model, settings)
        import anthropic

        key = settings.anthropic_api_key
        if not key:
            raise ConfigError(
                "ANTHROPIC_API_KEY is not set.",
                "Copy .env.example to .env and add your key, or switch "
                "provider with CHAT_PROVIDER=openai.",
            )
        self._sdk = anthropic
        self._client = anthropic.Anthropic(
            api_key=key,
            timeout=settings.provider_timeout_seconds,
            max_retries=settings.provider_max_retries,
        )

    def complete(
        self,
        system: str,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 4000,
        effort: str = "medium",
        temperature: float | None = None,
    ) -> LLMResponse:
        # `temperature` is deliberately never sent: the current Anthropic SDK
        # has removed the sampling parameters, and current models reject them.
        # Reasoning depth is controlled by `output_config.effort` instead.
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": messages,
            "output_config": {"effort": effort},
        }

        started = time.perf_counter()
        try:
            response = self._client.messages.create(**kwargs)
        except self._sdk.AuthenticationError as exc:
            raise ConfigError(
                "Anthropic rejected the API key.",
                "Check ANTHROPIC_API_KEY in your .env.",
            ) from exc
        except self._sdk.NotFoundError as exc:
            raise ConfigError(
                f"Model '{self.model}' was not found.",
                "Set CHAT_MODEL to a model your key can access, "
                "e.g. claude-sonnet-5.",
            ) from exc
        except self._sdk.RateLimitError as exc:
            raise ProviderError(
                "Anthropic rate limit reached.",
                "Wait a moment and re-run; the eval runner is resumable.",
            ) from exc
        except self._sdk.APITimeoutError as exc:
            raise ProviderError(
                "Anthropic request timed out after 120s.",
                "Re-run, or reduce CHAT_MAX_TOKENS.",
            ) from exc
        except self._sdk.APIConnectionError as exc:
            raise ProviderError(
                "Could not reach the Anthropic API.", "Check your network."
            ) from exc
        except self._sdk.APIStatusError as exc:
            raise ProviderError(f"Anthropic API error ({exc.status_code}).") from exc
        latency_ms = (time.perf_counter() - started) * 1000

        if response.stop_reason == "refusal":
            raise ProviderError(
                "The model declined this request at the safety layer.",
                "This is a provider-level refusal, distinct from the "
                "product's own 'not covered in these episodes' behaviour.",
            )

        text = "".join(
            block.text for block in response.content if block.type == "text"
        )
        return LLMResponse(
            text=text.strip(),
            model=self.model,
            provider=self.name,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_ms=latency_ms,
            stop_reason=response.stop_reason,
        )


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(self, model: str, settings: Settings) -> None:
        super().__init__(model, settings)
        import openai

        key = settings.openai_api_key
        if not key:
            raise ConfigError(
                "OPENAI_API_KEY is not set.",
                "Copy .env.example to .env and add your key, or switch "
                "provider with CHAT_PROVIDER=anthropic.",
            )
        self._sdk = openai
        self._client = openai.OpenAI(
            api_key=key,
            timeout=settings.provider_timeout_seconds,
            max_retries=settings.provider_max_retries,
        )

    def complete(
        self,
        system: str,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 4000,
        effort: str = "medium",
        temperature: float | None = None,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "max_completion_tokens": max_tokens,
            "messages": [{"role": "system", "content": system}, *messages],
        }
        if temperature is not None:
            payload["temperature"] = temperature

        started = time.perf_counter()
        try:
            response = self._client.chat.completions.create(**payload)
        except self._sdk.AuthenticationError as exc:
            raise ConfigError(
                "OpenAI rejected the API key.", "Check OPENAI_API_KEY in your .env."
            ) from exc
        except self._sdk.NotFoundError as exc:
            raise ConfigError(
                f"Model '{self.model}' was not found.",
                "Set CHAT_MODEL to a model your key can access, e.g. gpt-4o.",
            ) from exc
        except self._sdk.RateLimitError as exc:
            raise ProviderError(
                "OpenAI rate limit reached.", "Wait a moment and re-run."
            ) from exc
        except self._sdk.APITimeoutError as exc:
            raise ProviderError("OpenAI request timed out after 120s.") from exc
        except self._sdk.APIConnectionError as exc:
            raise ProviderError(
                "Could not reach the OpenAI API.", "Check your network."
            ) from exc
        except self._sdk.APIStatusError as exc:
            raise ProviderError(f"OpenAI API error ({exc.status_code}).") from exc
        latency_ms = (time.perf_counter() - started) * 1000

        choice = response.choices[0]
        usage = response.usage
        return LLMResponse(
            text=(choice.message.content or "").strip(),
            model=self.model,
            provider=self.name,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            latency_ms=latency_ms,
            stop_reason=choice.finish_reason,
        )


class OpenRouterProvider(LLMProvider):
    """OpenRouter — one key, many vendors.

    OpenRouter exposes an OpenAI-compatible ``/chat/completions`` endpoint, so
    the official ``openai`` SDK is pointed at its base URL rather than a
    bespoke HTTP client. That keeps retries, timeouts and typed exceptions
    identical to the OpenAI provider.

    Two OpenRouter-specific behaviours are worth noting:

    * ``usage: {"include": true}`` makes the response carry the **actual**
      credits spent, which is reported instead of a locally estimated price.
    * ``reasoning: {"effort": ...}`` is only accepted by reasoning-capable
      models, so it is opt-in (``OPENROUTER_SEND_REASONING``) rather than
      always sent — any of OpenRouter's models can be selected without a 400.
    """

    name = "openrouter"

    def __init__(self, model: str, settings: Settings) -> None:
        super().__init__(model, settings)
        import openai

        key = settings.openrouter_api_key
        if not key:
            raise ConfigError(
                "OPENROUTER_API_KEY is not set.",
                "Add your OpenRouter key to .env "
                "(get one at https://openrouter.ai/keys), or switch provider "
                "with CHAT_PROVIDER=anthropic / openai.",
            )
        self._sdk = openai
        headers = {"X-Title": settings.openrouter_app_title}
        if settings.openrouter_site_url:
            headers["HTTP-Referer"] = settings.openrouter_site_url
        self._client = openai.OpenAI(
            api_key=key,
            base_url=settings.openrouter_base_url,
            default_headers=headers,
            timeout=settings.provider_timeout_seconds,
            max_retries=settings.provider_max_retries,
        )

    def complete(
        self,
        system: str,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 4000,
        effort: str = "medium",
        temperature: float | None = None,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": system}, *messages],
            # Ask OpenRouter to return what the call actually cost.
            "extra_body": {"usage": {"include": True}},
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if self.settings.openrouter_send_reasoning:
            payload["extra_body"]["reasoning"] = {"effort": effort}

        started = time.perf_counter()
        try:
            response = self._client.chat.completions.create(**payload)
        except self._sdk.AuthenticationError as exc:
            raise ConfigError(
                "OpenRouter rejected the API key.",
                "Check OPENROUTER_API_KEY in your .env.",
            ) from exc
        except self._sdk.NotFoundError as exc:
            raise ConfigError(
                f"OpenRouter does not have a model called '{self.model}'.",
                "Model ids are namespaced, e.g. 'anthropic/claude-sonnet-5'. "
                "Run `make models` to list what your key can reach.",
            ) from exc
        except self._sdk.PermissionDeniedError as exc:
            raise ConfigError(
                f"Your OpenRouter key is not allowed to use '{self.model}'.",
                "Some models need extra account setup or credits. "
                "Run `make models` to see the available ids.",
            ) from exc
        except self._sdk.BadRequestError as exc:
            raise ProviderError(
                f"OpenRouter rejected the request for '{self.model}': "
                f"{getattr(exc, 'message', exc)}",
                "If the model is not reasoning-capable, set "
                "OPENROUTER_SEND_REASONING=false (the default).",
            ) from exc
        except self._sdk.RateLimitError as exc:
            raise ProviderError(
                "OpenRouter rate limit or insufficient credits.",
                "Check your credit balance at https://openrouter.ai/credits, "
                "then re-run.",
            ) from exc
        except self._sdk.APITimeoutError as exc:
            raise ProviderError(
                f"OpenRouter request timed out after "
                f"{self.settings.provider_timeout_seconds:.0f}s.",
                "Re-run, or lower CHAT_MAX_TOKENS / raise "
                "PROVIDER_TIMEOUT_SECONDS.",
            ) from exc
        except self._sdk.APIConnectionError as exc:
            raise ProviderError(
                "Could not reach OpenRouter.", "Check your network."
            ) from exc
        except self._sdk.APIStatusError as exc:
            raise ProviderError(
                f"OpenRouter API error ({exc.status_code})."
            ) from exc
        latency_ms = (time.perf_counter() - started) * 1000

        # OpenRouter surfaces upstream failures as a body-level `error` on an
        # otherwise 200 response, so choices can legitimately be absent.
        choices = getattr(response, "choices", None)
        if not choices:
            detail = getattr(response, "error", None) or "no choices returned"
            raise ProviderError(
                f"OpenRouter returned no completion for '{self.model}': {detail}",
                "The upstream provider may be down; try another model with "
                "`make models`.",
            )

        choice = choices[0]
        usage = getattr(response, "usage", None)
        return LLMResponse(
            text=(choice.message.content or "").strip(),
            model=getattr(response, "model", self.model),
            provider=self.name,
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            latency_ms=latency_ms,
            stop_reason=choice.finish_reason,
            reported_cost_usd=_openrouter_cost(usage),
        )


def _openrouter_cost(usage: Any) -> float | None:
    """Pull the actual spend out of an OpenRouter usage object.

    Returns ``None`` when the field is absent so the caller falls back to the
    local pricing table rather than silently reporting zero.
    """
    if usage is None:
        return None
    cost = getattr(usage, "cost", None)
    if cost is None and hasattr(usage, "model_extra"):
        cost = (usage.model_extra or {}).get("cost")
    try:
        return float(cost) if cost is not None else None
    except (TypeError, ValueError):
        return None


def list_openrouter_models(settings: Settings | None = None) -> list[dict[str, Any]]:
    """Fetch OpenRouter's live model catalogue.

    Used by `make models` and by `make doctor` so a misconfigured model id is
    caught with a list of real alternatives instead of a failed run.
    """
    import httpx

    settings = settings or get_settings()
    try:
        response = httpx.get(
            f"{settings.openrouter_base_url}/models", timeout=30.0
        )
        response.raise_for_status()
    except Exception as exc:  # noqa: BLE001 - network is optional here
        raise ProviderError(
            f"Could not fetch the OpenRouter model list: {exc}",
            "Check your network, or browse https://openrouter.ai/models",
        ) from exc
    return response.json().get("data", [])


_PROVIDERS: dict[str, type[LLMProvider]] = {
    "openrouter": OpenRouterProvider,
    "anthropic": AnthropicProvider,
    "openai": OpenAIProvider,
}


def get_provider(role: Role = "chat", settings: Settings | None = None) -> LLMProvider:
    """Build the provider configured for ``chat`` or ``judge``."""
    settings = settings or get_settings()
    if role == "judge":
        provider_name, model = settings.judge_provider, settings.judge_model
    else:
        provider_name, model = settings.chat_provider, settings.chat_model

    provider_cls = _PROVIDERS.get(provider_name)
    if provider_cls is None:
        raise ConfigError(
            f"Unknown provider '{provider_name}'.",
            f"Set {role.upper()}_PROVIDER to one of: "
            f"{', '.join(sorted(_PROVIDERS))}.",
        )
    log.debug("provider resolved", role=role, provider=provider_name, model=model)
    return provider_cls(model, settings)


def provider_available(role: Role = "chat", settings: Settings | None = None) -> bool:
    """True when credentials for ``role`` are present (no network call)."""
    settings = settings or get_settings()
    name = settings.judge_provider if role == "judge" else settings.chat_provider
    return bool(settings.api_key_for(name))
