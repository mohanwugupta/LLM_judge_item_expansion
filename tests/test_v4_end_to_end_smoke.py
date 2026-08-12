import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


VALIDATION = Path(__file__).resolve().parents[1] / "ISC-CI_LLM_validation"
sys.path.insert(0, str(VALIDATION))

from iscci_validation.evaluation import evaluate_model, make_evaluation_contexts  # noqa: E402
from iscci_validation.training import train_model  # noqa: E402


def test_one_epoch_cpu_training_and_fixed_activation_extraction():
    matrix = pd.DataFrame(
        {
            "context_a": [1, 1, 1, 1, 0],
            "context_b": [0, 1, 1, 1, 1],
        },
        index=["a", "b", "c", "d", "e"],
    )
    released_embedding = torch.randn(6, 64)
    released_embedding[-1].zero_()
    model, metrics, _ = train_model(
        matrix=matrix,
        released_embedding=released_embedding,
        seed=0,
        epochs=1,
        episodes_per_epoch=2,
        batch_size=2,
        learning_rate=0.001,
        task_loss_weight=0.5,
    )
    assert len(metrics) == 1
    assert torch.equal(model.input_to_independent.weight, released_embedding)
    evaluation = make_evaluation_contexts(
        object_count=5,
        pair_context_count=2,
        rdm_context_count=4,
        context_dependent_rdm_count=2,
        seed=20260804,
    )
    outputs = evaluate_model(model, evaluation, batch_size=2)
    assert outputs["membership_logits"].shape == (7, 5)
    assert np.isfinite(outputs["membership_logits"]).all()
