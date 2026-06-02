"""
tests/test_positive_verification_prompt.py

Tests for the verifier prompt file and build_verifier_user_message().
"""
import re
import pytest
from leuven_expansion.feature_prompts import (
    build_verifier_user_message,
    load_verifier_prompt,
)

SYSTEM_PROMPT = load_verifier_prompt()


# ── Prompt file existence and structure ───────────────────────────────────────

def test_verifier_prompt_loaded():
    assert len(SYSTEM_PROMPT) > 100


def test_verifier_prompt_contains_leuven():
    assert "leuven" in SYSTEM_PROMPT.lower()


def test_verifier_prompt_has_output_fields():
    for field in ("verified_feature_value", "retain_positive", "reason", "confidence"):
        assert field in SYSTEM_PROMPT, f"missing field '{field}' in verifier prompt"


def test_verifier_prompt_no_drm_metadata():
    forbidden = ("drm_list", "critical_lure", "isc_ci", "sleep_list")
    for token in forbidden:
        assert token not in SYSTEM_PROMPT.lower(), f"forbidden token '{token}' in verifier prompt"


def test_verifier_prompt_no_other_tasks():
    """Prompt should not mention DRM, ISC, or unrelated cognitive tasks."""
    for token in ("false memory", "false recognition", "critical lure"):
        assert token not in SYSTEM_PROMPT.lower()


def test_verifier_prompt_value_range_stated():
    """0–4 scale should be explained in the prompt."""
    assert "0" in SYSTEM_PROMPT and "4" in SYSTEM_PROMPT


# ── build_verifier_user_message ───────────────────────────────────────────────

def _make_votes():
    return [
        {"judge": "A", "feature_value": 3.0, "confidence": 0.8, "reason": "dogs are friendly"},
        {"judge": "B", "feature_value": 2.0, "confidence": 0.7, "reason": "somewhat friendly"},
        {"judge": "C", "feature_value": 3.0, "confidence": 0.9, "reason": "definitely friendly"},
    ]


def test_user_message_contains_word():
    msg = build_verifier_user_message(
        word_normalized="dog",
        feature_id=0,
        feature_text="is friendly",
        resolved_feature_value=3.0,
        resolved_confidence=0.85,
        resolution_method="majority",
        first_pass_votes=_make_votes(),
    )
    assert "dog" in msg


def test_user_message_contains_feature_text():
    msg = build_verifier_user_message(
        word_normalized="dog",
        feature_id=0,
        feature_text="is friendly",
        resolved_feature_value=3.0,
        resolved_confidence=0.85,
        resolution_method="majority",
        first_pass_votes=_make_votes(),
    )
    assert "is friendly" in msg


def test_user_message_contains_judge_votes():
    votes = _make_votes()
    msg = build_verifier_user_message(
        word_normalized="dog",
        feature_id=0,
        feature_text="is friendly",
        resolved_feature_value=3.0,
        resolved_confidence=0.85,
        resolution_method="majority",
        first_pass_votes=votes,
    )
    for v in votes:
        assert str(v["feature_value"]) in msg or str(int(v["feature_value"])) in msg


def test_user_message_contains_resolved_value():
    msg = build_verifier_user_message(
        word_normalized="dog",
        feature_id=0,
        feature_text="is friendly",
        resolved_feature_value=3.0,
        resolved_confidence=0.85,
        resolution_method="majority",
        first_pass_votes=_make_votes(),
    )
    assert "3" in msg


def test_user_message_no_drm_fields():
    msg = build_verifier_user_message(
        word_normalized="dog",
        feature_id=0,
        feature_text="is friendly",
        resolved_feature_value=3.0,
        resolved_confidence=0.85,
        resolution_method="majority",
        first_pass_votes=_make_votes(),
    )
    forbidden = ("drm_list", "critical_lure", "isc_ci", "sleep_list", "false memory")
    for token in forbidden:
        assert token not in msg.lower()


def test_user_message_no_other_words():
    """The message should not accidentally include unrelated word lists."""
    msg = build_verifier_user_message(
        word_normalized="dog",
        feature_id=0,
        feature_text="is friendly",
        resolved_feature_value=3.0,
        resolved_confidence=0.85,
        resolution_method="majority",
        first_pass_votes=_make_votes(),
    )
    # Should not contain long word lists in the body
    assert msg.count("\n") < 50


def test_user_message_contains_feature_id():
    msg = build_verifier_user_message(
        word_normalized="dog",
        feature_id=42,
        feature_text="has four legs",
        resolved_feature_value=3.0,
        resolved_confidence=0.85,
        resolution_method="majority",
        first_pass_votes=_make_votes(),
    )
    assert "42" in msg
