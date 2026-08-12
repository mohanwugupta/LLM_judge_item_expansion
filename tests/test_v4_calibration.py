import numpy as np
import pandas as pd

from calibrate_v4_judgments import (
    apply_threshold,
    cross_validate_thresholds,
    select_threshold,
)


def test_calibration_splits_whole_words_and_reproduces_locked_rule():
    words = [f"word_{index}" for index in range(10)]
    ordinal = pd.DataFrame(
        np.tile([0.0, 1 / 3, 1.0, 4.0], (10, 1)), index=words
    )
    human = pd.DataFrame(
        np.tile([0, 0, 1, 1], (10, 1)), index=words, dtype=np.int8
    )
    candidates = [
        {"threshold_id": "locked", "operator": "gt", "value": 0.0, "is_locked_v2": True},
        {"threshold_id": "ge_1", "operator": "ge", "value": 1.0, "is_locked_v2": False},
    ]
    _, summary, assignments = cross_validate_thresholds(
        ordinal, human, candidates, folds=5, seed=3
    )
    assignment_frame = pd.DataFrame(assignments)
    assert assignment_frame.groupby("word")["fold"].nunique().eq(1).all()
    assert sorted(assignment_frame.groupby("fold").size()) == [2, 2, 2, 2, 2]
    assert np.array_equal(
        apply_threshold(ordinal.to_numpy(), candidates[0]),
        ordinal.to_numpy() > 0,
    )
    selected, gate_met = select_threshold(summary, recall_gate=0.8)
    assert gate_met
    assert selected["threshold_id"] == "ge_1"


def test_threshold_ties_use_precision_then_conservatism():
    summary = pd.DataFrame(
        {
            "threshold_id": ["low", "middle", "high"],
            "positive_recall_mean": [0.9, 0.9, 0.9],
            "MCC_mean": [0.5, 0.5, 0.5],
            "positive_precision_mean": [0.7, 0.8, 0.8],
            "threshold_value": [0.0, 1.0, 2.0],
        }
    )
    selected, gate_met = select_threshold(summary, recall_gate=0.8)
    assert gate_met
    assert selected["threshold_id"] == "high"

