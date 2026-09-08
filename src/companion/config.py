"""Central, env-driven configuration.

Every tunable lives here so that provider choice, model choice and retrieval
depth can be changed without touching call sites. Values come from the
environment (optionally via a local ``.env``); the defaults are chosen so the
system runs fully locally and for free apart from chat/judge calls.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env", override=False)

Provider = Literal["openrouter", "anthropic", "openai"]
LocalOrApi = Literal["local", "openai"]


class Settings(BaseSettings):
    """Runtime settings resolved from environment variables."""

    model_config = SettingsConfigDict(extra="ignore", case_sensitive=False)

    @model_validator(mode="before")
    @classmethod
    def _blank_means_unset(cls, values: object) -> object:
        """Treat an empty environment variable as absent.

        ``.env`` templates ship keys with no value (``ANTHROPIC_API_KEY=``,
        ``RERANK_GATE_OVERRIDE=``). Without this, pydantic sees an empty
        string and either fails to parse it as a number or stores "" where
        the code expects None.
        """
        if isinstance(values, dict):
            return {
                key: value
                for key, value in values.items()
                if not (isinstance(value, str) and not value.strip())
            }
        return values

    # --- chat -------------------------------------------------------------
    chat_provider: Provider = Field(default="openrouter", alias="CHAT_PROVIDER")
    chat_model: str = Field(
        default="anthropic/claude-sonnet-5", alias="CHAT_MODEL"
    )
    chat_max_tokens: int = Field(default=8000, alias="CHAT_MAX_TOKENS")
    chat_effort: str = Field(default="medium", alias="CHAT_EFFORT")
    rewrite_effort: str = Field(default="low", alias="REWRITE_EFFORT")

    # --- judge (evaluation) ----------------------------------------------
    judge_provider: Provider = Field(default="openrouter", alias="JUDGE_PROVIDER")
    # A different vendor from the chat model on purpose — a judge scoring its
    # own family's output is prone to self-preference bias. OpenRouter makes
    # cross-vendor judging a one-line change on a single key.
    judge_model: str = Field(default="openai/gpt-5-mini", alias="JUDGE_MODEL")
    judge_effort: str = Field(default="medium", alias="JUDGE_EFFORT")
    judge_max_tokens: int = Field(default=4000, alias="JUDGE_MAX_TOKENS")

    # --- credentials ------------------------------------------------------
    openrouter_api_key: str | None = Field(default=None, alias="OPENROUTER_API_KEY")
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")

    # --- OpenRouter specifics --------------------------------------------
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1", alias="OPENROUTER_BASE_URL"
    )
    # Sent as HTTP-Referer / X-Title for OpenRouter's attribution; optional.
    openrouter_site_url: str | None = Field(
        default=None, alias="OPENROUTER_SITE_URL"
    )
    openrouter_app_title: str = Field(
        default="Fermi Podcast Companion", alias="OPENROUTER_APP_TITLE"
    )
    # Only reasoning-capable models accept a reasoning effort. Off by default
    # so any of OpenRouter's models can be selected without a 400.
    openrouter_send_reasoning: bool = Field(
        default=False, alias="OPENROUTER_SEND_REASONING"
    )

    # --- ASR --------------------------------------------------------------
    asr_provider: LocalOrApi = Field(default="local", alias="ASR_PROVIDER")
    asr_model: str = Field(default="medium", alias="ASR_MODEL")
    asr_compute_type: str = Field(default="int8", alias="ASR_COMPUTE_TYPE")
    asr_language: str = Field(default="en", alias="ASR_LANGUAGE")

    # --- embeddings -------------------------------------------------------
    embed_provider: LocalOrApi = Field(default="local", alias="EMBED_PROVIDER")
    embed_model: str = Field(default="BAAI/bge-small-en-v1.5", alias="EMBED_MODEL")

    # --- chunking ---------------------------------------------------------
    chunk_target_seconds: float = Field(default=75.0, alias="CHUNK_TARGET_SECONDS")
    chunk_max_seconds: float = Field(default=110.0, alias="CHUNK_MAX_SECONDS")
    chunk_overlap_seconds: float = Field(default=15.0, alias="CHUNK_OVERLAP_SECONDS")

    # --- retrieval --------------------------------------------------------
    top_k: int = Field(default=5, alias="TOP_K")
    dense_k: int = Field(default=20, alias="DENSE_K")
    bm25_k: int = Field(default=20, alias="BM25_K")
    fusion_k: int = Field(default=8, alias="FUSION_K")
    rrf_k: int = Field(default=60, alias="RRF_K")
    min_relevance: float = Field(default=0.30, alias="MIN_RELEVANCE")

    # --- reranking / relevance gate --------------------------------------
    enable_rerank: bool = Field(default=True, alias="ENABLE_RERANK")
    rerank_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2", alias="RERANK_MODEL"
    )
    # Left unset, the gate calibrated at ingestion time is used. Set this to
    # override it (higher = refuses more readily).
    rerank_gate_override: float | None = Field(
        default=None, alias="RERANK_GATE_OVERRIDE"
    )

    # --- feature flags (used to define the eval baseline) -----------------
    retrieval_mode: Literal["dense", "hybrid"] = Field(
        default="hybrid", alias="RETRIEVAL_MODE"
    )
    enable_query_rewrite: bool = Field(default=True, alias="ENABLE_QUERY_REWRITE")
    enable_compare_mode: bool = Field(default=True, alias="ENABLE_COMPARE_MODE")

    # --- paths ------------------------------------------------------------
    audio_dir: Path = Field(default=PROJECT_ROOT / "audio", alias="AUDIO_DIR")
    data_dir: Path = Field(default=PROJECT_ROOT / "data", alias="DATA_DIR")
    vector_db_path: Path = Field(
        default=PROJECT_ROOT / "data" / "index", alias="VECTOR_DB_PATH"
    )

    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # --- derived paths ----------------------------------------------------
    @property
    def transcripts_dir(self) -> Path:
        return self.data_dir / "transcripts"

    @property
    def chunks_dir(self) -> Path:
        return self.data_dir / "chunks"

    @property
    def normalized_dir(self) -> Path:
        return self.data_dir / "normalized"

    @property
    def manifest_path(self) -> Path:
        return self.data_dir / "manifest.json"

    @property
    def bm25_path(self) -> Path:
        return self.data_dir / "index" / "bm25.pkl"

    def ensure_dirs(self) -> None:
        for path in (
            self.data_dir,
            self.transcripts_dir,
            self.chunks_dir,
            self.normalized_dir,
            self.vector_db_path,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def api_key_for(self, provider: str) -> str | None:
        return {
            "openrouter": self.openrouter_api_key,
            "anthropic": self.anthropic_api_key,
            "openai": self.openai_api_key,
        }.get(provider)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton.

    A bad value in ``.env`` is a configuration mistake, not a bug, so the
    pydantic validation error is translated into a message that names the
    offending variable and the values it accepts.
    """
    from pydantic import ValidationError

    from companion.errors import ConfigError

    try:
        return Settings()
    except ValidationError as exc:
        problems = []
        for error in exc.errors():
            field = ".".join(str(part) for part in error["loc"]) or "(unknown)"
            problems.append(f"{field}: {error['msg']} (got {error.get('input')!r})")
        raise ConfigError(
            "Your .env has an invalid value:\n  " + "\n  ".join(problems),
            "Fix the variable in .env — see .env.example for the accepted "
            "values. CHAT_PROVIDER and JUDGE_PROVIDER must be one of: "
            "openrouter, anthropic, openai.",
        ) from exc


def reload_settings() -> Settings:
    """Drop the cache and re-read the environment (used by tests)."""
    get_settings.cache_clear()
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    return get_settings()


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}
