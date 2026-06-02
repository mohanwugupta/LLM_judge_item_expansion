"""
tests/test_positive_verification_schema.py

Tests for the positive_verification_schema_v1.json and the
validate_verifier_output() function in feature_schema.py.
"""
import json
import pytest
from leuven_expansion.positive_verifier import validate_verifier_output


def _valid_raw(
    word="dog",
    feature_id=0,
    verified_feature_value=3.0,
    confidence=0.9,
    retain_positive=True,
    reason="Dogs are well-known for being friendly.",
):
    return json.dumps({
        "target_word": word,
        "feature_id": feature_id,
        "verified_feature_value": verified_feature_value,
        "confidence": confidence,
        "retain_positive": retain_positive,
        "reason": reason,
    })


# ── Valid responses ────────────────────────────────────────────────────────────

def test_valid_response_parses():
    record, err = validate_verifier_output(_valid_raw(), expected_word="dog", expected_feature_id=0)
    assert err is None
    assert record["verified_feature_value"] == 3.0
    assert record["retain_positive"] is True


def test_valid_response_retain_false():
    raw = _valid_raw(verified_feature_value=0.0, retain_positive=False)
    record, err = validate_verifier_output(raw, expected_word="dog", expected_feature_id=0)
    assert err is None
    assert record["retain_positive"] is False


def test_maximum_value_accepted():
    raw = _valid_raw(verified_feature_value=4.0)
    record, err = validate_verifier_output(raw, expected_word="dog", expected_feature_id=0)
    assert err is None


def test_minimum_value_accepted():
    raw = _valid_raw(verified_feature_value=0.0, retain_positive=False)
    record, err = validate_verifier_output(raw, expected_word="dog", expected_feature_id=0)
    assert err is None


# ── Missing required fields ────────────────────────────────────────────────────

def test_missing_verified_feature_value_fails():
    obj = {
        "target_word": "dog",
        "feature_id": 0,
        "confidence": 0.9,
        "retain_positive": True,
        "reason": "test",
    }
    record, err = validate_verifier_output(json.dumps(obj))
    assert err is not None
    assert record is None


def test_missing_retain_positive_fails():
    obj = {
        "target_word": "dog",
        "feature_id": 0,
        "verified_feature_value": 3.0,
        "confidence": 0.9,
        "reason": "test",
    }
    record, err = validate_verifier_output(json.dumps(obj))
    assert err is not None


def test_missing_reason_fails():
    obj = {
        "target_word": "dog",
        "feature_id": 0,
        "verified_feature_value": 3.0,
        "confidence": 0.9,
        "retain_positive": True,
    }
    record, err = validate_verifier_output(json.dumps(obj))
    assert err is not None


# ── Extra / forbidden fields ───────────────────────────────────────────────────

def test_extra_field_rejected():
    obj = json.loads(_valid_raw())
    obj["drm_list"] = "sleep"
    record, err = validate_verifier_output(json.dumps(obj))
    assert err is not None


# ── Out-of-range values ────────────────────────────────────────────────────────

def test_value_above_4_rejected():
    raw = _valid_raw(verified_feature_value=5.0)
    record, err = validate_verifier_output(raw)
    assert err is not None


def test_value_below_0_rejected():
    raw = _valid_raw(verified_feature_value=-1.0)
    record, err = validate_verifier_output(raw)
    assert err is not None


def test_confidence_above_1_rejected():
    raw = _valid_raw(confidence=1.5)
    record, err = validate_verifier_output(raw)
    assert err is not None


# ── Word / feature_id cross-checks ────────────────────────────────────────────

def test_wrong_word_raises_error():
    raw = _valid_raw(word="cat")
    record, err = validate_verifier_output(raw, expected_word="dog")
    assert err is not None
    assert record is None


def test_wrong_feature_id_raises_error():
    raw = _valid_raw(feature_id=99)
    record, err = validate_verifier_output(raw, expected_feature_id=0)
    assert err is not None


def test_no_expected_word_skips_check():
    raw = _valid_raw(word="anything")
    record, err = validate_verifier_output(raw)
    assert err is None


# ── retain_positive consistency fix ───────────────────────────────────────────

def test_inconsistent_retain_positive_is_recomputed():
    """If retain_positive=True but verified_feature_value=0, trust the number."""
    raw = _valid_raw(verified_feature_value=0.0, retain_positive=True)
    record, err = validate_verifier_output(raw, verification_threshold=1.0)
    assert err is None
    assert record["retain_positive"] is False


def test_inconsistent_retain_false_recomputed():
    """If retain_positive=False but verified_feature_value=3, trust the number."""
    raw = _valid_raw(verified_feature_value=3.0, retain_positive=False)
    record, err = validate_verifier_output(raw, verification_threshold=1.0)
    assert err is None
    assert record["retain_positive"] is True


# ── Python None / True / False literals ───────────────────────────────────────

def test_python_none_in_retain_positive_coerced():
    """Model emitting Python None for retain_positive should be handled."""
    raw = '{"target_word":"dog","feature_id":0,"verified_feature_value":2.0,"confidence":0.8,"retain_positive":None,"reason":"ok"}'
    record, err = validate_verifier_output(raw, expected_word="dog", expected_feature_id=0, verification_threshold=1.0)
    # None -> null -> coerced, retain_positive recomputed from value
    assert err is None
    assert record["retain_positive"] is True
