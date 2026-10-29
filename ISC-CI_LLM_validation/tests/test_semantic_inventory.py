import csv
from pathlib import Path

import numpy as np
import pytest

from semantic_retraining.inventory import CONDITIONS, inventory, load_matrix


def test_exact_fourteen_sources():
    assert len(CONDITIONS) == 14
    assert CONDITIONS["human"] == CONDITIONS["random_human"] == 385
    assert CONDITIONS["v4_ensemble_locked_v2"] == 87315
    assert CONDITIONS["v4_ensemble_calibrated"] == 80699


def test_strict_loader(tmp_path):
    path = tmp_path / "matrix.csv"
    path.write_text("object,f,g\na,1,0\nb,0,1\n")
    frame = load_matrix(path, ["a", "b"])
    assert frame.values.dtype == np.int8
    with pytest.raises(ValueError, match="order"):
        load_matrix(path, ["b", "a"])
    path.write_text("object,f,f\na,1,0\nb,0,1\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_matrix(path, ["a", "b"])
    path.write_text("object,f\na,2\nb,0\n")
    with pytest.raises(ValueError, match="binary"):
        load_matrix(path, ["a", "b"])


def test_all_existing_matrices_resolve_without_writes():
    root = Path(__file__).resolve().parents[1]
    records = inventory(root)
    assert set(records) == set(CONDITIONS)
    assert all(row["objects"] == 293 for row in records.values())
    assert all(row["features"] == CONDITIONS[key] for key, row in records.items())
    assert records["random_human"]["positives"] == 8056
    assert records["human"]["feature_ids"] == records["random_human"]["feature_ids"]
    assert records["human"]["prevalence"] == records["random_human"]["prevalence"]
