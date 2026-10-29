import json

import numpy as np
import pytest
import torch

from test_isc_training import fixture_dataset
from semantic_retraining.extract_representations import extract, load_run, train_run
from semantic_retraining.train_isc import TrainingConfig


def test_checkpoint_extraction_resume_and_source_hash_guard(tmp_path):
    torch.set_num_threads(1)
    dataset = fixture_dataset()
    config = TrainingConfig(
        max_epochs=200, batch_size=4, learning_rate=0.01, max_seconds=30
    )
    directory = tmp_path / "human_seed_0"
    result = train_run(
        dataset,
        "human",
        0,
        config,
        directory,
        taxonomy_sha256="fixture-taxonomy",
        classification_sha256="fixture-reviewed",
    )
    assert result["complete"]
    model, checkpoint, manifest = load_run(directory)
    saved = np.load(directory / "representations.npz", allow_pickle=False)
    np.testing.assert_array_equal(saved["object_ids"], dataset.object_ids)
    np.testing.assert_array_equal(
        saved["activations"], extract(model, dataset.object_ids)
    )
    assert saved["activations"].shape == (2, 64)
    assert (saved["activations"] > 0).all() and (saved["activations"] < 1).all()
    with pytest.raises(FileExistsError):
        train_run(
            dataset,
            "human",
            0,
            config,
            directory,
            "fixture-taxonomy",
            "fixture-reviewed",
        )
    reused = train_run(
        dataset,
        "human",
        0,
        config,
        directory,
        "fixture-taxonomy",
        "fixture-reviewed",
        resume=True,
    )
    assert reused["reused"]
    with pytest.raises(ValueError, match="signature"):
        train_run(
            dataset,
            "human",
            0,
            config,
            directory,
            "changed",
            "fixture-reviewed",
            resume=True,
        )
    with (directory / "representations.npz").open("ab") as f:
        f.write(b"corrupt")
    with pytest.raises(ValueError, match="hash"):
        load_run(directory)


def test_nonconverged_model_cannot_publish_transfer_representations(tmp_path):
    result = train_run(
        fixture_dataset(),
        "human",
        0,
        TrainingConfig(max_epochs=1, max_seconds=30),
        tmp_path / "failed",
        "t",
        "c",
    )
    assert not result["complete"]
    assert not (tmp_path / "failed/representations.npz").exists()
    with pytest.raises(ValueError, match="complete"):
        load_run(tmp_path / "failed")
