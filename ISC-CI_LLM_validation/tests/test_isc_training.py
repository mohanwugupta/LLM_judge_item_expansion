import numpy as np
import pandas as pd
import pytest
import torch

from semantic_retraining.training_adapter import TaskDataset
from semantic_retraining.train_isc import TrainingConfig, fit


def fixture_dataset():
    matrix = pd.DataFrame(
        [[1, 0, 1, 0], [0, 1, 0, 1]],
        index=["a", "b"],
        columns=list("wxyz"),
        dtype=np.int8,
    )
    return TaskDataset(matrix, {"w": 0, "x": 0, "y": 1, "z": 1}, 2)


def test_synthetic_environment_is_learned_reproducibly_without_pretrained_weights():
    torch.set_num_threads(1)
    dataset = fixture_dataset()
    config = TrainingConfig(
        max_epochs=800, batch_size=4, learning_rate=0.01, max_seconds=30
    )
    model, history, result = fit(dataset, 0, config)
    assert result["complete"] and result["converged"]
    assert result["metrics"]["max_absolute_error"] < 0.1
    assert result["metrics"]["positive_recall"] == 1
    assert result["metrics"]["bce"] < result["metrics"]["all_negative_bce"]
    assert len(history) == result["epochs"]
    assert result["examples_seen"] == result["epochs"] * len(dataset)
    assert result["initialization"] == "independent_random_no_pretrained_weights"
    duplicate, other_history, other_result = fit(dataset, 0, config)
    for key, value in model.state_dict().items():
        assert torch.equal(value, duplicate.state_dict()[key])
    assert [r["bce"] for r in history] == [r["bce"] for r in other_history]


def test_interrupted_mid_epoch_resumes_identical_updates_and_exposure():
    torch.set_num_threads(1)
    dataset = fixture_dataset()
    config = TrainingConfig(
        max_epochs=3,
        batch_size=3,
        learning_rate=0.05,
        max_seconds=30,
        stopping="fixed_budget",
    )
    expected, _, final = fit(dataset, 3, config)
    snapshots = []

    def interrupt(state):
        import copy

        snapshots.append(copy.deepcopy(state))
        if state["updates"] == 1:
            raise KeyboardInterrupt("fixture interrupt")

    with pytest.raises(KeyboardInterrupt):
        fit(dataset, 3, config, checkpoint_callback=interrupt, checkpoint_every_steps=1)
    actual, _, resumed = fit(dataset, 3, config, resume_state=snapshots[-1])
    assert resumed["examples_seen"] == final["examples_seen"] == 12
    assert resumed["updates"] == final["updates"] == 6
    for key, value in expected.state_dict().items():
        assert torch.equal(value, actual.state_dict()[key])


def test_missing_convergence_and_resource_limit_are_not_completed():
    dataset = fixture_dataset()
    _, _, result = fit(
        dataset, 1, TrainingConfig(max_epochs=1, batch_size=4, max_seconds=30)
    )
    assert not result["complete"] and result["status"] == "not_converged"
    _, _, result = fit(dataset, 1, TrainingConfig(max_epochs=10, max_seconds=1e-12))
    assert not result["complete"] and result["status"] == "resource_limit"
