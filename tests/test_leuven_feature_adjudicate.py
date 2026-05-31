"""
tests/test_leuven_feature_adjudicate.py

Tests for leuven_expansion/feature_adjudicate.py — cell-level disagreement logic.
"""
import json
from unittest.mock import MagicMock

import pytest

from leuven_expansion.feature_adjudicate import (
    _needs_adjudication,
    _resolve_first_pass,
    resolve_first_pass,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_vote(value, confidence=0.8, ambiguous=False, parse_error="", row_hash="abc"):
    return {
        "feature_value": value,
        "confidence": confidence,
        "ambiguous": ambiguous,
        "parse_error": parse_error,
        "row_hash": row_hash,
    }


def _make_client_returning(value=3.0):
    raw = json.dumps({
        "target_word": "dog",
        "feature_id": 0,
        "feature_value": value,
        "confidence": 0.9,
        "ambiguous": False,
        "reason": "adj reason",
    })
    client = MagicMock()
    client.generate.return_value = (raw, {})
    return client


def _make_prompts():
    return {"A": "A", "B": "B", "C": "C", "adjudicator": "adj"}


# ── _needs_adjudication tests ─────────────────────────────────────────────────

def test_exact_agreement_no_adjudication():
    votes = [_make_vote(3.0), _make_vote(3.0), _make_vote(3.0)]
    needs, _ = _needs_adjudication(votes)
    assert needs is False


def test_small_disagreement_no_adjudication():
    """Range of 1 should not trigger adjudication (with low-confidence dissenter)."""
    votes = [_make_vote(3.0, confidence=0.9), _make_vote(3.0, confidence=0.85), _make_vote(2.0, confidence=0.5)]
    needs, _ = _needs_adjudication(votes)
    assert needs is False


def test_range_two_triggers_adjudication():
    """Range >= 2 triggers adjudication."""
    votes = [_make_vote(0.0), _make_vote(2.0), _make_vote(2.0)]
    needs, reason = _needs_adjudication(votes)
    assert needs is True


def test_zero_vs_three_triggers_adjudication():
    votes = [_make_vote(0.0), _make_vote(3.0), _make_vote(3.0)]
    needs, reason = _needs_adjudication(votes)
    assert needs is True
    assert "zero_vs_high" in reason or "range" in reason


def test_zero_vs_four_triggers_adjudication():
    votes = [_make_vote(0.0), _make_vote(4.0), _make_vote(4.0)]
    needs, reason = _needs_adjudication(votes)
    assert needs is True


def test_high_confidence_dissent_triggers_adjudication():
    """2/1 split where dissenter has confidence >= 0.80, range < 2."""
    votes = [
        _make_vote(3.0, confidence=0.9),
        _make_vote(3.0, confidence=0.85),
        _make_vote(2.0, confidence=0.85),  # high-confidence dissent, range=1
    ]
    needs, reason = _needs_adjudication(votes)
    assert needs is True
    assert "dissent" in reason


def test_low_confidence_dissent_no_adjudication():
    """2/1 split where dissenter has low confidence, range < 2: no adjudication."""
    votes = [
        _make_vote(3.0, confidence=0.9),
        _make_vote(3.0, confidence=0.85),
        _make_vote(2.0, confidence=0.5),  # low-confidence dissent, range=1
    ]
    needs, _ = _needs_adjudication(votes)
    assert needs is False


def test_parse_error_triggers_adjudication():
    votes = [
        _make_vote(3.0),
        _make_vote(3.0),
        _make_vote(None, parse_error="JSON parse error"),
    ]
    votes[2]["feature_value"] = None
    needs, reason = _needs_adjudication(votes)
    assert needs is True
    assert "parse_error" in reason


def test_ambiguous_majority_triggers_adjudication():
    votes = [
        _make_vote(2.0, ambiguous=True, confidence=0.5),
        _make_vote(2.0, ambiguous=True, confidence=0.5),
        _make_vote(2.0, ambiguous=False, confidence=0.5),
    ]
    needs, reason = _needs_adjudication(votes)
    assert needs is True
    assert "ambiguous" in reason


# ── _resolve_first_pass tests ─────────────────────────────────────────────────

def test_unanimous_resolution():
    votes = [_make_vote(3.0), _make_vote(3.0), _make_vote(3.0)]
    res = _resolve_first_pass(votes)
    assert res["final_feature_value"] == 3.0
    assert res["resolution_method"] == "unanimous"
    assert res["needs_human_audit"] is False


def test_small_disagreement_uses_mean():
    votes = [_make_vote(3.0), _make_vote(3.0), _make_vote(2.0)]
    res = _resolve_first_pass(votes)
    expected = (3.0 + 3.0 + 2.0) / 3
    assert abs(res["final_feature_value"] - expected) < 1e-6


# ── Adjudication sees only disputed pair ─────────────────────────────────────

def test_adjudication_input_contains_only_disputed_pair():
    """Adjudicator user message must only reference the disputed word × feature pair."""
    from leuven_expansion.feature_prompts import build_adjudicator_user_message
    votes = [_make_vote(0.0), _make_vote(4.0), _make_vote(3.0)]
    msg = build_adjudicator_user_message("dog", 0, "is an animal", votes)
    # Must not contain DRM metadata
    assert "drm" not in msg.lower()
    assert "list_id" not in msg.lower()
    assert "critical_lure" not in msg.lower()
    # Must contain word and feature
    assert "dog" in msg
    assert "is an animal" in msg


# ── resolve_first_pass integration tests ─────────────────────────────────────

def test_resolve_no_adjudication_for_unanimous():
    votes = [_make_vote(4.0), _make_vote(4.0), _make_vote(4.0)]
    client = MagicMock()
    resolution, adj_votes = resolve_first_pass(
        job_id="test",
        word_normalized="dog",
        feature_id=0,
        feature_text="is an animal",
        votes=votes,
        prompts=_make_prompts(),
        client=client,
        model="test-model",
    )
    # No adjudication called
    client.generate.assert_not_called()
    assert len(adj_votes) == 0
    assert resolution["final_feature_value"] == 4.0


def test_resolve_calls_adjudicator_for_large_disagreement():
    votes = [_make_vote(0.0), _make_vote(4.0), _make_vote(4.0)]
    client = _make_client_returning(value=4.0)

    resolution, adj_votes = resolve_first_pass(
        job_id="test",
        word_normalized="dog",
        feature_id=0,
        feature_text="is an animal",
        votes=votes,
        prompts=_make_prompts(),
        client=client,
        model="test-model",
    )
    assert client.generate.call_count == 3  # three adjudicator calls
    assert len(adj_votes) == 3
    assert resolution["adjudicated"] is True


def test_adjudication_failure_marks_human_audit():
    """If all adjudicators return parse errors, needs_human_audit=True."""
    votes = [_make_vote(0.0), _make_vote(4.0), _make_vote(3.0)]
    bad_client = MagicMock()
    bad_client.generate.return_value = ("{invalid json}", {})

    resolution, _ = resolve_first_pass(
        job_id="test",
        word_normalized="dog",
        feature_id=0,
        feature_text="is an animal",
        votes=votes,
        prompts=_make_prompts(),
        client=bad_client,
        model="test-model",
    )
    assert resolution["needs_human_audit"] is True
