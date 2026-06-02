"""
tests/test_positive_verification_runner.py

Tests for verify_positive() in positive_verifier.py and the
candidate-routing logic in run_jobs.py.
"""
import json
import pytest
from unittest.mock import MagicMock, patch, call
from leuven_expansion.positive_verifier import verify_positive, POSITIVE_THRESHOLD


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_client(response_json: dict):
    """Return a mock vLLM client whose chat.completions.create returns response_json."""
    content = json.dumps(response_json)
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    client = MagicMock()
    client.chat.completions.create.return_value = resp
    return client


def _valid_verifier_response(word="dog", feature_id=0, value=3.0, retain=True):
    return {
        "target_word": word,
        "feature_id": feature_id,
        "verified_feature_value": value,
        "confidence": 0.9,
        "retain_positive": retain,
        "reason": "Clearly applicable.",
    }


def _base_votes():
    return [
        {"judge": "A", "feature_value": 3.0, "confidence": 0.8, "reason": "yes"},
        {"judge": "B", "feature_value": 3.0, "confidence": 0.7, "reason": "yes"},
        {"judge": "C", "feature_value": 2.0, "confidence": 0.6, "reason": "maybe"},
    ]


# ── verify_positive return structure ─────────────────────────────────────────

def test_verify_positive_retained_record():
    client = _make_client(_valid_verifier_response())
    record, vote = verify_positive(
        job_id="j1",
        word_normalized="dog",
        feature_id=0,
        feature_text="is friendly",
        resolved_feature_value=3.0,
        resolved_confidence=0.85,
        resolution_method="majority",
        first_pass_votes=_base_votes(),
        prompt="You are a verifier.",
        client=client,
        model="test-model",
        verification_threshold=1.0,
    )
    assert record["positive_verification_status"] == "retained"
    assert record["verified_feature_value"] == 3.0
    assert record["pre_verification_feature_value"] == 3.0


def test_verify_positive_rejected_when_low_value():
    client = _make_client(_valid_verifier_response(value=0.0, retain=False))
    record, vote = verify_positive(
        job_id="j1",
        word_normalized="dog",
        feature_id=0,
        feature_text="is an object",
        resolved_feature_value=2.0,
        resolved_confidence=0.6,
        resolution_method="majority",
        first_pass_votes=_base_votes(),
        prompt="You are a verifier.",
        client=client,
        model="test-model",
        verification_threshold=1.0,
    )
    assert record["positive_verification_status"] == "rejected"
    assert record["verified_feature_value"] == 0.0


def test_verify_positive_final_value_set_to_zero_on_rejection():
    client = _make_client(_valid_verifier_response(value=0.0, retain=False))
    record, vote = verify_positive(
        job_id="j1",
        word_normalized="dog",
        feature_id=0,
        feature_text="has mass",
        resolved_feature_value=2.0,
        resolved_confidence=0.5,
        resolution_method="majority",
        first_pass_votes=_base_votes(),
        prompt="You are a verifier.",
        client=client,
        model="test-model",
        verification_threshold=1.0,
    )
    assert record["final_feature_value"] == 0.0


def test_verify_positive_retained_preserves_original_value():
    client = _make_client(_valid_verifier_response(value=3.0, retain=True))
    record, vote = verify_positive(
        job_id="j1",
        word_normalized="dog",
        feature_id=0,
        feature_text="is friendly",
        resolved_feature_value=3.0,
        resolved_confidence=0.9,
        resolution_method="majority",
        first_pass_votes=_base_votes(),
        prompt="You are a verifier.",
        client=client,
        model="test-model",
        verification_threshold=1.0,
    )
    assert record["final_feature_value"] == 3.0


def test_verify_positive_vote_dict_has_required_keys():
    client = _make_client(_valid_verifier_response())
    record, vote = verify_positive(
        job_id="j1",
        word_normalized="dog",
        feature_id=0,
        feature_text="is friendly",
        resolved_feature_value=3.0,
        resolved_confidence=0.9,
        resolution_method="majority",
        first_pass_votes=_base_votes(),
        prompt="You are a verifier.",
        client=client,
        model="test-model",
        verification_threshold=1.0,
    )
    for key in ("job_id", "word_normalized", "feature_id", "verified_feature_value",
                "retain_positive", "confidence", "reason"):
        assert key in vote


# ── Parse-error / retry logic ─────────────────────────────────────────────────

def test_parse_error_retried_once():
    bad = MagicMock()
    bad.choices = [MagicMock()]
    bad.choices[0].message.content = "not json"

    good_content = json.dumps(_valid_verifier_response())
    good = MagicMock()
    good.choices = [MagicMock()]
    good.choices[0].message.content = good_content

    client = MagicMock()
    client.chat.completions.create.side_effect = [bad, good]

    record, vote = verify_positive(
        job_id="j1",
        word_normalized="dog",
        feature_id=0,
        feature_text="is friendly",
        resolved_feature_value=3.0,
        resolved_confidence=0.9,
        resolution_method="majority",
        first_pass_votes=_base_votes(),
        prompt="You are a verifier.",
        client=client,
        model="test-model",
        verification_threshold=1.0,
    )
    assert client.chat.completions.create.call_count == 2
    assert record["positive_verification_status"] == "retained"


def test_repeated_parse_failure_sets_parse_error_status():
    bad = MagicMock()
    bad.choices = [MagicMock()]
    bad.choices[0].message.content = "not json at all"

    client = MagicMock()
    client.chat.completions.create.return_value = bad

    record, vote = verify_positive(
        job_id="j1",
        word_normalized="dog",
        feature_id=0,
        feature_text="is friendly",
        resolved_feature_value=3.0,
        resolved_confidence=0.9,
        resolution_method="majority",
        first_pass_votes=_base_votes(),
        prompt="You are a verifier.",
        client=client,
        model="test-model",
        verification_threshold=1.0,
    )
    assert record["positive_verification_status"] == "parse_error"
    assert record.get("needs_human_audit") is True


# ── Candidate routing in run_jobs ─────────────────────────────────────────────

def test_below_positive_threshold_not_sent_to_verifier():
    """Values strictly below POSITIVE_THRESHOLD should get status 'not_candidate'."""
    from leuven_expansion.positive_verifier import route_for_verification
    status = route_for_verification(final_feature_value=0.5, positive_threshold=1.0)
    assert status == "not_candidate"


def test_at_positive_threshold_is_candidate():
    from leuven_expansion.positive_verifier import route_for_verification
    status = route_for_verification(final_feature_value=1.0, positive_threshold=1.0)
    assert status == "candidate"


def test_above_positive_threshold_is_candidate():
    from leuven_expansion.positive_verifier import route_for_verification
    status = route_for_verification(final_feature_value=3.5, positive_threshold=1.0)
    assert status == "candidate"


def test_none_value_not_candidate():
    from leuven_expansion.positive_verifier import route_for_verification
    status = route_for_verification(final_feature_value=None, positive_threshold=1.0)
    assert status == "not_candidate"


# ── POSITIVE_THRESHOLD constant ───────────────────────────────────────────────

def test_default_positive_threshold_is_1():
    assert POSITIVE_THRESHOLD == 1.0
