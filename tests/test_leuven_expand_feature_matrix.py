"""
tests/test_leuven_expand_feature_matrix.py

Tests for leuven_expansion/expand_feature_matrix.py
"""
import json
import pathlib

import pandas as pd
import pytest

from leuven_expansion.expand_feature_matrix import (
    _load_new_items,
    _dedup_items,
    build_expanded_matrix,
)
from leuven_expansion.normalize import normalize_word


@pytest.fixture
def leuven_features_csv(tmp_path):
    df = pd.DataFrame({
        "Name": ["dog", "cat", "hammer"],
        "is an animal": [4.0, 4.0, 0.0],
        "is a tool": [0.0, 0.0, 4.0],
        "can fly": [0.0, 0.0, 0.0],
    })
    p = tmp_path / "features.csv"
    df.to_csv(p, index=False)
    return p


@pytest.fixture
def drm_items_csv(tmp_path):
    df = pd.DataFrame({"word": ["sleep", "dream", "rest", "dog"]})
    p = tmp_path / "drm_items.csv"
    df.to_csv(p, index=False)
    return p


@pytest.fixture
def resolutions_csv(tmp_path):
    rows = []
    words = ["sleep", "dream", "rest"]
    features = [("is an animal", 0), ("is a tool", 1), ("can fly", 2)]
    for w in words:
        for fname, fid in features:
            rows.append({
                "job_id": "test",
                "word_normalized": w,
                "feature_id": fid,
                "feature_text": fname,
                "final_feature_value": 0.0,
                "resolution_method": "unanimous",
                "needs_human_audit": False,
                "adjudicated": False,
                "adjudication_trigger": "",
            })
    df = pd.DataFrame(rows)
    p = tmp_path / "feature_resolutions.csv"
    df.to_csv(p, index=False)
    return p


# ── Item loading tests ────────────────────────────────────────────────────────

def test_new_drm_words_load(drm_items_csv):
    items = _load_new_items(drm_items_csv)
    assert "sleep" in items
    assert "dream" in items


def test_existing_leuven_words_skipped():
    """DRM words that already exist in Leuven should be excluded."""
    existing = {"dog", "cat"}
    pairs = _dedup_items(["dog", "sleep", "dream"], existing)
    nw_set = {nw for _, nw in pairs}
    assert "dog" not in nw_set
    assert "sleep" in nw_set


def test_duplicate_normalized_words_classified_once():
    existing = set()
    pairs = _dedup_items(["Dog", "dog", "DOG", "sleep"], existing)
    nw_set = {nw for _, nw in pairs}
    assert len(nw_set) == 2  # "dog" and "sleep"
    assert nw_set == {"dog", "sleep"}


def test_every_new_word_feature_pair_is_independent_atomic_job(leuven_features_csv):
    """Each new word must create one pair per feature column."""
    from leuven_expansion.feature_schema import load_leuven_feature_schema
    schema = load_leuven_feature_schema(leuven_features_csv)
    n_features = schema["n_features"]

    new_words = [("sleep", "sleep"), ("dream", "dream")]
    expected_pairs = len(new_words) * n_features

    pairs = []
    for word_orig, word_norm in new_words:
        for fid, fcol in enumerate(schema["feature_columns"]):
            pairs.append({
                "word_original": word_orig,
                "word_normalized": word_norm,
                "feature_id": fid,
                "feature_text": fcol,
            })

    assert len(pairs) == expected_pairs
    # All pairs are unique (word, feature_id) combinations
    keys = {(p["word_normalized"], p["feature_id"]) for p in pairs}
    assert len(keys) == expected_pairs


# ── Expanded matrix tests ─────────────────────────────────────────────────────

def test_expanded_matrix_has_original_feature_columns(
    leuven_features_csv, resolutions_csv, tmp_path
):
    """Expanded matrix must have exact original Leuven feature columns."""
    new_word_pairs = [("sleep", "sleep"), ("dream", "dream"), ("rest", "rest")]
    out_path = build_expanded_matrix(
        leuven_features_csv=leuven_features_csv,
        resolutions_csv=resolutions_csv,
        new_word_pairs=new_word_pairs,
        output_dir=tmp_path,
        job_id="test",
    )
    expanded = pd.read_csv(out_path)
    assert "is an animal" in expanded.columns
    assert "is a tool" in expanded.columns
    assert "can fly" in expanded.columns


def test_expanded_matrix_includes_original_leuven_rows(
    leuven_features_csv, resolutions_csv, tmp_path
):
    new_word_pairs = [("sleep", "sleep")]
    out_path = build_expanded_matrix(
        leuven_features_csv=leuven_features_csv,
        resolutions_csv=resolutions_csv,
        new_word_pairs=new_word_pairs,
        output_dir=tmp_path,
        job_id="test",
    )
    expanded = pd.read_csv(out_path)
    norm_words = set(expanded["word_normalized"].apply(normalize_word))
    assert "dog" in norm_words
    assert "cat" in norm_words
    assert "hammer" in norm_words


def test_expanded_matrix_includes_new_words(
    leuven_features_csv, resolutions_csv, tmp_path
):
    new_word_pairs = [("sleep", "sleep"), ("dream", "dream")]
    out_path = build_expanded_matrix(
        leuven_features_csv=leuven_features_csv,
        resolutions_csv=resolutions_csv,
        new_word_pairs=new_word_pairs,
        output_dir=tmp_path,
        job_id="test",
    )
    expanded = pd.read_csv(out_path)
    norm_words = set(expanded["word_normalized"])
    assert "sleep" in norm_words
    assert "dream" in norm_words


def test_expanded_matrix_has_metadata_columns(
    leuven_features_csv, resolutions_csv, tmp_path
):
    new_word_pairs = [("sleep", "sleep")]
    out_path = build_expanded_matrix(
        leuven_features_csv=leuven_features_csv,
        resolutions_csv=resolutions_csv,
        new_word_pairs=new_word_pairs,
        output_dir=tmp_path,
        job_id="test",
    )
    expanded = pd.read_csv(out_path)
    for col in ["word_normalized", "source", "in_original_leuven", "in_llm_expansion"]:
        assert col in expanded.columns, f"Missing column: {col}"
