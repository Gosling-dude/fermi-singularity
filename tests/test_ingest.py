"""Ingestion plumbing: identity, idempotency, error handling and eval loading."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from companion.config import Settings
from companion.errors import AudioError
from companion.ingest.normalize import (
    derive_episode_id, derive_title, discover_audio, sha256_of,
)
from companion.ingest.pipeline import _asr_fingerprint, load_manifest, save_manifest


@pytest.mark.parametrize(
    "filename,episode_id,title",
    [
        ("01 - The Birth of the Quantum.mp3", "01_the_birth_of_the_quantum",
         "The Birth of the Quantum"),
        ("shannon_information.mp3", "shannon_information", "Shannon Information"),
        ("Bell's Theorem — 1964.mp3", "bell_s_theorem_1964", "Bell's Theorem — 1964"),
        # The real supplied filenames: the separator between the series/track
        # prefix and the title is a hyphen, which becomes a space in the title.
        ("Great Papers 01 - Einstein's Special Relativity.mp3",
         "great_papers_01_einstein_s_special_relativity",
         "Great Papers 01 Einstein's Special Relativity"),
        ("Great Papers 09 - Bell's Theorem, 1964.mp3",
         "great_papers_09_bell_s_theorem_1964",
         "Great Papers 09 Bell's Theorem, 1964"),
    ],
)
def test_identity_is_derived_from_the_filename(filename, episode_id, title):
    path = Path(filename)
    assert derive_episode_id(path) == episode_id
    assert derive_title(path) == title


def test_sha256_is_content_addressed(tmp_path):
    a, b, c = tmp_path / "a", tmp_path / "b", tmp_path / "c"
    a.write_bytes(b"same"); b.write_bytes(b"same"); c.write_bytes(b"different")
    assert sha256_of(a) == sha256_of(b) != sha256_of(c)


def test_missing_audio_directory_is_actionable(tmp_path):
    with pytest.raises(AudioError) as exc:
        discover_audio(tmp_path / "nope")
    assert exc.value.remedy


def test_empty_audio_directory_points_at_the_readme(tmp_path):
    with pytest.raises(AudioError) as exc:
        discover_audio(tmp_path)
    assert "audio/README.md" in (exc.value.remedy or "")


def test_zero_byte_file_is_rejected(tmp_path):
    (tmp_path / "broken.mp3").write_bytes(b"")
    with pytest.raises(AudioError) as exc:
        discover_audio(tmp_path)
    assert "empty" in exc.value.message


def test_corrupt_audio_is_rejected_with_a_reason(tmp_path):
    (tmp_path / "broken.mp3").write_bytes(b"this is definitely not an mp3 file")
    with pytest.raises(AudioError) as exc:
        discover_audio(tmp_path)
    assert "ffprobe" in exc.value.message or "corrupt" in exc.value.message


def test_asr_fingerprint_changes_with_configuration():
    a = _asr_fingerprint(Settings(ASR_MODEL="medium"))
    b = _asr_fingerprint(Settings(ASR_MODEL="small"))
    assert a != b


def test_manifest_roundtrip(tmp_path):
    settings = Settings(DATA_DIR=tmp_path)
    settings.ensure_dirs()
    save_manifest(settings, {"version": 2, "episodes": {"x": {"sha256": "abc"}}})
    assert load_manifest(settings)["episodes"]["x"]["sha256"] == "abc"


def test_corrupt_manifest_falls_back_to_a_full_reingest(tmp_path):
    settings = Settings(DATA_DIR=tmp_path)
    settings.ensure_dirs()
    settings.manifest_path.write_text("{ not json")
    assert load_manifest(settings)["episodes"] == {}


def test_manifest_version_bump_forces_reingest(tmp_path):
    settings = Settings(DATA_DIR=tmp_path)
    settings.ensure_dirs()
    settings.manifest_path.write_text(json.dumps({"version": 0, "episodes": {"x": {}}}))
    assert load_manifest(settings)["episodes"] == {}


def test_blank_env_vars_are_treated_as_unset():
    # `.env.example` ships keys with no value; an empty string must not be
    # parsed as a float or stored where None is expected.
    settings = Settings(
        RERANK_GATE_OVERRIDE="", ANTHROPIC_API_KEY="", OPENAI_API_KEY="   "
    )
    assert settings.rerank_gate_override is None
    assert settings.anthropic_api_key is None
    assert settings.openai_api_key is None


def test_explicit_env_values_still_apply():
    assert Settings(RERANK_GATE_OVERRIDE="-5.5").rerank_gate_override == -5.5
    assert Settings(TOP_K="7").top_k == 7
