import numpy as np
import pandas as pd
import pytest
import torch

from iscci_validation.training import load_trained_model
from semantic_retraining.transfer_embeddings import frozen_embedding, train_context


def test_exact_transfer_order_padding_and_frozen_gradient():
    names = ["a", "b", "c", "d", "e"]
    values = np.arange(5 * 64, dtype=np.float32).reshape(5, 64) / 400
    embedding = frozen_embedding(values, names, names)
    assert torch.equal(embedding[:-1], torch.from_numpy(values))
    assert embedding.shape == (6, 64) and not embedding[-1].count_nonzero()
    for wrong in [names[::-1], names[:-1], ["a"] * 5]:
        with pytest.raises(ValueError, match="object"):
            frozen_embedding(values, names, wrong)
    with pytest.raises(ValueError, match="float32"):
        frozen_embedding(values.astype(np.float64), names, names)


def test_short_context_diagnostic_reuses_exact_original_training_and_freezes_hub(
    tmp_path,
):
    names = ["a", "b", "c", "d", "e"]
    matrix = pd.DataFrame(
        {"x": [1, 1, 1, 1, 0], "y": [0, 1, 1, 1, 1]}, index=names, dtype=np.int8
    )
    values = np.arange(5 * 64, dtype=np.float32).reshape(5, 64) / 400
    result = train_context(
        matrix,
        values,
        names,
        "human",
        "random_human",
        0,
        tmp_path / "context",
        {"semantic_checkpoint_sha256": "fixture"},
        diagnostic=True,
    )
    assert result["diagnostic"] and not result["complete"]
    model, saved = load_trained_model(tmp_path / "context/checkpoint.pt")
    assert torch.equal(model.input_to_independent.weight[:-1], torch.from_numpy(values))
    model.input_to_independent.weight.requires_grad_(False)
    support = torch.tensor([[0, 1], [1, 2]])
    query = torch.tensor([[2, 3], [3, 4]])
    losses = model.losses(
        support,
        torch.ones(2, 2),
        torch.tensor([[0, 0], [1, 1]]),
        query,
        torch.zeros(2, 2),
    )
    (0.5 * losses[0] + 0.5 * (losses[1] + losses[2])).backward()
    assert model.input_to_independent.weight.grad is None
    assert all(
        p.grad is not None
        for n, p in model.named_parameters()
        if n != "input_to_independent.weight"
    )
    assert result["settings"]["epochs"] == 2
    assert saved["provenance"]["semantic_source"] == "human"
    assert saved["provenance"]["context_source"] == "random_human"


def test_loaded_model_cannot_be_relabelled_as_another_source_or_seed(tmp_path):
    from semantic_retraining.transfer_embeddings import load_context

    names = list("abcde")
    matrix = pd.DataFrame(
        {"x": [1, 1, 1, 1, 0], "y": [0, 1, 1, 1, 1]}, index=names, dtype=np.int8
    )
    values = np.full((5, 64), 0.5, dtype=np.float32)
    directory = tmp_path / "context"
    train_context(
        matrix,
        values,
        names,
        "human",
        "v2",
        0,
        directory,
        {"semantic_checkpoint_sha256": "fixture"},
        diagnostic=True,
    )
    with pytest.raises(ValueError, match="identity"):
        load_context(
            directory,
            matrix,
            frozen_embedding(values, names, names),
            expected_identity={
                "semantic_source": "human",
                "context_source": "v2",
                "seed": 1,
            },
        )
