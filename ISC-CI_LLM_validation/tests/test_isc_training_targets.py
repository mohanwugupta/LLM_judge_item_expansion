import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from semantic_retraining.training_adapter import TaskDataset, parse_output_ids


@pytest.mark.parametrize("targets", ["66 67 68;", "66 67 68 ;"])
def test_parser_preserves_final_feature_with_attached_or_separate_terminator(targets):
    assert parse_output_ids(f"name: dogC1 i: 0 351 t: {targets}") == [66, 67, 68]


def test_parser_preserves_single_positive_and_rejects_malformed_patterns():
    assert parse_output_ids("name: goldfishC2 i: 29 352 t: 351;") == [351]
    for pattern in ["t: 66 67", "t: 66; garbage", "t: 1 -2;", "t: ;", "t: 1 x;"]:
        with pytest.raises(ValueError):
            parse_output_ids(pattern)


def test_real_released_patterns_keep_all_positive_targets():
    path = (
        Path(__file__).resolve().parents[1]
        / "upstream/IntegratedSemanticsControl/data/leuven_rum_nozeroes.ex"
    )
    lines = [line for line in path.read_text().splitlines() if "vigerC" not in line]
    corrected = [parse_output_ids(line) for line in lines]
    legacy = [line.split("t:")[1].strip().split()[:-1] for line in lines]
    assert len(lines) == 6007
    assert sum(not ids for ids in legacy) == 1532
    assert all(ids for ids in corrected)
    for old, new in zip(legacy, corrected):
        assert new[:-1] == [int(token) for token in old]
        assert len(new) == len(old) + 1


def test_full_vector_targets_and_positive_pair_selection():
    matrix = pd.DataFrame(
        [[1, 1, 0], [0, 1, 1]], index=["a", "b"], columns=["x", "y", "z"], dtype=np.int8
    )
    dataset = TaskDataset(matrix, {"x": 0, "y": 1, "z": 0}, task_count=3)
    assert dataset.pairs.tolist() == [[0, 0], [0, 1], [1, 0], [1, 1]]
    assert dataset.inactive_tasks == [2]
    np.testing.assert_array_equal(
        dataset.targets([0, 2]).numpy(), [[1, 0, 0], [0, 0, 1]]
    )
    np.testing.assert_array_equal(
        dataset.targets([1, 3]).numpy(), [[0, 1, 0], [0, 1, 0]]
    )
    assert dataset.exposure()["valid_patterns"] == 4


def test_malformed_mappings_fail_without_reordering_or_modification():
    matrix = pd.DataFrame([[1, 0], [0, 1]], columns=["x", "y"], dtype=np.int8)
    original = matrix.copy(deep=True)
    for assignment in [{"x": 0}, {"x": 0, "y": 5}, {"x": 0, "y": "1"}]:
        with pytest.raises(ValueError):
            TaskDataset(matrix, assignment, task_count=2)
    pd.testing.assert_frame_equal(matrix, original)


def test_chunked_targets_match_full_targets_and_preserve_singleton():
    matrix = pd.DataFrame(
        [[1, 1, 1], [0, 0, 1]], columns=["first", "middle", "last"], dtype=np.int8
    )
    dataset = TaskDataset(matrix, {"first": 0, "middle": 0, "last": 1}, task_count=3)
    assert dataset.pairs.tolist() == [[0, 0], [0, 1], [1, 1]]
    full = dataset.targets([0, 1, 2]).numpy()
    np.testing.assert_array_equal(full, [[1, 1, 0], [0, 0, 1], [0, 0, 1]])
    np.testing.assert_array_equal(
        np.concatenate(
            [
                dataset.targets([0, 1, 2], start=0, stop=2).numpy(),
                dataset.targets([0, 1, 2], start=2, stop=3).numpy(),
            ],
            axis=1,
        ),
        full,
    )
    matrix.iloc[:, :] = 0
    np.testing.assert_array_equal(dataset.targets([0, 1, 2]).numpy(), full)


def test_inactive_class_and_object_reporting_does_not_invent_examples():
    matrix = pd.DataFrame(
        [[1, 0], [0, 0]], index=["a", "b"], columns=["x", "y"], dtype=np.int8
    )
    dataset = TaskDataset(matrix, {"x": 0, "y": 1}, task_count=3)
    assert dataset.pairs.tolist() == [[0, 0]]
    assert dataset.inactive_tasks == [1, 2]
    assert dataset.exposure()["objects_without_patterns"] == ["b"]
    assert dataset.exposure()["features_per_task"] == [1, 1, 0]
    assert dataset.exposure()["positive_associations_per_task"] == [1, 0, 0]
