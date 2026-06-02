"""
tests/test_positive_verification_integration.py

Integration tests: run_jobs output files contain verifier columns,
pre-verification values are preserved, validate_features pre/post metrics
are computed and reported correctly.
"""
import json
import os
import tempfile
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_resolution_row(job_id="j1", word="dog", feature_id=0,
                         final_feature_value=3.0):
    return {
        "job_id": job_id,
        "word_normalized": word,
        "feature_id": feature_id,
        "feature_text": "is friendly",
        "final_feature_value": final_feature_value,
        "resolution_method": "majority",
        "needs_human_audit": False,
        "adjudicated": False,
        "adjudication_trigger": None,
    }


def _mock_verifier_call(verified_value=3.0, retain=True):
    record = {
        "positive_verification_status": "retained" if retain else "rejected",
        "verified_feature_value": verified_value,
        "pre_verification_feature_value": 3.0,
        "final_feature_value": verified_value if retain else 0.0,
        "positive_verification_confidence": 0.9,
        "positive_verification_reason": "ok",
        "needs_human_audit": False,
    }
    vote = {
        "job_id": "j1",
        "word_normalized": "dog",
        "feature_id": 0,
        "verified_feature_value": verified_value,
        "retain_positive": retain,
        "confidence": 0.9,
        "reason": "ok",
    }
    return record, vote


# ── Output file columns ───────────────────────────────────────────────────────

def test_resolutions_csv_has_verification_columns(tmp_path):
    """After run_atomic_jobs with verification enabled, resolutions CSV has
    the five new verification columns."""
    from leuven_expansion.run_jobs import RESOLUTION_COLUMNS
    expected_new = {
        "pre_verification_feature_value",
        "verified_feature_value",
        "positive_verification_status",
        "positive_verification_confidence",
        "positive_verification_reason",
    }
    assert expected_new.issubset(set(RESOLUTION_COLUMNS))


def test_verifier_votes_csv_has_required_columns(tmp_path):
    """positive_verification_votes.csv must have standard vote columns."""
    from leuven_expansion.run_jobs import VERIFIER_VOTE_COLUMNS
    required = {"job_id", "word_normalized", "feature_id", "verified_feature_value",
                "retain_positive", "confidence", "reason"}
    assert required.issubset(set(VERIFIER_VOTE_COLUMNS))


# ── pre_verification value preserved ─────────────────────────────────────────

def test_pre_verification_value_matches_resolved(tmp_path):
    """pre_verification_feature_value must equal the resolved value before
    the verifier modifies final_feature_value."""
    from leuven_expansion.positive_verifier import apply_verification_result
    row = _make_resolution_row(final_feature_value=2.5)
    ver_record, _ = _mock_verifier_call(verified_value=0.0, retain=False)
    updated = apply_verification_result(row, ver_record)
    assert updated["pre_verification_feature_value"] == 2.5
    assert updated["final_feature_value"] == 0.0


def test_rejected_row_final_value_zeroed(tmp_path):
    from leuven_expansion.positive_verifier import apply_verification_result
    row = _make_resolution_row(final_feature_value=2.0)
    ver_record, _ = _mock_verifier_call(verified_value=0.0, retain=False)
    updated = apply_verification_result(row, ver_record)
    assert updated["final_feature_value"] == 0.0


def test_retained_row_final_value_unchanged(tmp_path):
    from leuven_expansion.positive_verifier import apply_verification_result
    row = _make_resolution_row(final_feature_value=3.0)
    ver_record, _ = _mock_verifier_call(verified_value=3.0, retain=True)
    updated = apply_verification_result(row, ver_record)
    assert updated["final_feature_value"] == 3.0


def test_not_candidate_row_unchanged(tmp_path):
    from leuven_expansion.positive_verifier import apply_verification_result
    row = _make_resolution_row(final_feature_value=0.0)
    # Simulate a not_candidate record (no verification was run)
    not_candidate_record = {
        "positive_verification_status": "not_candidate",
        "verified_feature_value": None,
        "pre_verification_feature_value": 0.0,
        "final_feature_value": 0.0,
        "positive_verification_confidence": None,
        "positive_verification_reason": None,
        "needs_human_audit": False,
    }
    updated = apply_verification_result(row, not_candidate_record)
    assert updated["final_feature_value"] == 0.0
    assert updated["positive_verification_status"] == "not_candidate"


# ── validate_features pre/post metrics ───────────────────────────────────────

def test_pre_post_verification_metrics_reported(tmp_path):
    """_compute_verification_delta() should return a dict with pre and post metrics."""
    from leuven_expansion.validate_features import _compute_verification_delta

    resolutions = pd.DataFrame([
        {"word_normalized": "dog", "feature_id": 0, "feature_text": "is friendly",
         "final_feature_value": 3.0, "pre_verification_feature_value": 3.0,
         "positive_verification_status": "retained"},
        {"word_normalized": "dog", "feature_id": 1, "feature_text": "is an object",
         "final_feature_value": 0.0, "pre_verification_feature_value": 2.0,
         "positive_verification_status": "rejected"},
        {"word_normalized": "dog", "feature_id": 2, "feature_text": "has four legs",
         "final_feature_value": 0.0, "pre_verification_feature_value": 0.0,
         "positive_verification_status": "not_candidate"},
    ])
    ground_truth = pd.DataFrame([
        {"word_normalized": "dog", "feature_id": 0, "feature_value": 3.0},
        {"word_normalized": "dog", "feature_id": 1, "feature_value": 0.0},
        {"word_normalized": "dog", "feature_id": 2, "feature_value": 0.0},
    ])

    delta = _compute_verification_delta(resolutions, ground_truth)

    assert "pre_precision" in delta
    assert "post_precision" in delta
    assert "pre_recall" in delta
    assert "post_recall" in delta
    assert "n_candidate_positives" in delta
    assert "n_verified_retained" in delta
    assert "n_verified_rejected" in delta


def test_verification_delta_rejected_reduces_fp(tmp_path):
    """Rejecting a false positive should improve precision."""
    from leuven_expansion.validate_features import _compute_verification_delta

    resolutions = pd.DataFrame([
        {"word_normalized": "dog", "feature_id": 0, "feature_text": "is friendly",
         "final_feature_value": 3.0, "pre_verification_feature_value": 3.0,
         "positive_verification_status": "retained"},
        {"word_normalized": "dog", "feature_id": 1, "feature_text": "is an object",
         "final_feature_value": 0.0, "pre_verification_feature_value": 2.0,
         "positive_verification_status": "rejected"},
    ])
    ground_truth = pd.DataFrame([
        {"word_normalized": "dog", "feature_id": 0, "feature_value": 3.0},
        {"word_normalized": "dog", "feature_id": 1, "feature_value": 0.0},
    ])

    delta = _compute_verification_delta(resolutions, ground_truth)
    assert delta["post_precision"] >= delta["pre_precision"]


def test_verification_counts_correct(tmp_path):
    from leuven_expansion.validate_features import _compute_verification_delta

    resolutions = pd.DataFrame([
        {"word_normalized": "a", "feature_id": 0, "feature_text": "f0",
         "final_feature_value": 3.0, "pre_verification_feature_value": 3.0,
         "positive_verification_status": "retained"},
        {"word_normalized": "a", "feature_id": 1, "feature_text": "f1",
         "final_feature_value": 0.0, "pre_verification_feature_value": 2.0,
         "positive_verification_status": "rejected"},
        {"word_normalized": "a", "feature_id": 2, "feature_text": "f2",
         "final_feature_value": 0.0, "pre_verification_feature_value": 0.0,
         "positive_verification_status": "not_candidate"},
    ])
    ground_truth = pd.DataFrame([
        {"word_normalized": "a", "feature_id": i, "feature_value": 0.0}
        for i in range(3)
    ])

    delta = _compute_verification_delta(resolutions, ground_truth)
    assert delta["n_candidate_positives"] == 2  # retained + rejected
    assert delta["n_verified_retained"] == 1
    assert delta["n_verified_rejected"] == 1
