import sys
from pathlib import Path

import pandas as pd


VALIDATION = Path(__file__).resolve().parents[1] / "ISC-CI_LLM_validation"
sys.path.insert(0, str(VALIDATION))

from iscci_validation.dataio import load_v4_matrix  # noqa: E402


def test_v4_loader_uses_strict_greater_than_three_and_stable_object_order(tmp_path):
    path = tmp_path / "matrix.csv"
    pd.DataFrame(
        {
            "three_positive": [1, 1, 1, 0, 0],
            "four_positive": [1, 1, 1, 1, 0],
        },
        index=["a", "b", "c", "d", "e"],
    ).to_csv(path)
    loaded = load_v4_matrix(path, ["e", "d", "c", "b", "a"])
    assert loaded.index.tolist() == ["e", "d", "c", "b", "a"]
    assert loaded.columns.tolist() == ["four_positive"]


def test_v4_config_keeps_executed_training_and_evaluation_protocol():
    import json

    config = json.loads(
        (VALIDATION.parent / "configs" / "v4_validation.json").read_text()
    )
    assert config["training"]["epochs"] == 400
    assert config["training"]["batch_size"] == 128
    assert config["training"]["seeds"] == [0, 1, 2, 3]
    assert config["evaluation"]["seed"] == 20260804
    assert set(config["v4"]["conditions"]) == {
        "v4_b_locked_v2",
        "v4_b_calibrated",
        "v4_ensemble_locked_v2",
        "v4_ensemble_calibrated",
    }

