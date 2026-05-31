"""
tests/test_leuven_feature_coverage.py

Tests for leuven_expansion/compute_feature_coverage.py
"""
import json
import pathlib

import pandas as pd
import pytest

from leuven_expansion.compute_feature_coverage import compute_coverage


@pytest.fixture
def expanded_matrix_csv(tmp_path):
    df = pd.DataFrame({
        "word_normalized": ["dog", "cat", "sleep", "dream"],
        "source": ["leuven_original"] * 2 + ["llm_expansion"] * 2,
        "is an animal": [4.0, 4.0, 0.0, 0.0],
    })
    p = tmp_path / "expanded.csv"
    df.to_csv(p, index=False)
    return p


@pytest.fixture
def drm_items_csv(tmp_path):
    df = pd.DataFrame({
        "word": ["sleep", "dream", "rest", "memory"],
        "list_id": ["list1", "list1", "list1", "list2"],
    })
    p = tmp_path / "drm.csv"
    df.to_csv(p, index=False)
    return p


def test_coverage_rate_computed(expanded_matrix_csv, drm_items_csv, tmp_path):
    stats = compute_coverage(
        expanded_matrix_csv=expanded_matrix_csv,
        drm_items_csv=drm_items_csv,
        output_dir=tmp_path,
    )
    assert "n_drm_items" in stats
    assert "n_covered" in stats
    assert "coverage_rate" in stats
    assert stats["n_drm_items"] == 4
    assert stats["n_covered"] == 2  # sleep and dream are covered
    assert stats["coverage_rate"] == pytest.approx(0.5, abs=0.01)


def test_coverage_report_csv_written(expanded_matrix_csv, drm_items_csv, tmp_path):
    compute_coverage(
        expanded_matrix_csv=expanded_matrix_csv,
        drm_items_csv=drm_items_csv,
        output_dir=tmp_path,
    )
    assert (tmp_path / "coverage_report.csv").exists()
    assert (tmp_path / "coverage_summary.json").exists()


def test_list_coverage_written_when_list_col_present(
    expanded_matrix_csv, drm_items_csv, tmp_path
):
    compute_coverage(
        expanded_matrix_csv=expanded_matrix_csv,
        drm_items_csv=drm_items_csv,
        output_dir=tmp_path,
        list_col="list_id",
    )
    assert (tmp_path / "list_coverage.csv").exists()


def test_complete_list_counted(tmp_path):
    expanded = pd.DataFrame({
        "word_normalized": ["sleep", "dream", "rest"],
        "source": ["llm_expansion"] * 3,
    })
    ep = tmp_path / "expanded.csv"
    expanded.to_csv(ep, index=False)

    drm = pd.DataFrame({
        "word": ["sleep", "dream", "rest"],
        "list_id": ["list1", "list1", "list1"],
    })
    dp = tmp_path / "drm.csv"
    drm.to_csv(dp, index=False)

    stats = compute_coverage(
        expanded_matrix_csv=ep,
        drm_items_csv=dp,
        output_dir=tmp_path,
        list_col="list_id",
    )
    assert stats["n_complete_lists"] == 1
