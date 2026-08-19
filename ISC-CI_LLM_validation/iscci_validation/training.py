from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .dataio import CICODataset, matrix_sha256, set_random_seed
from .modeling import CICOModel, initialize_model


def train_model(
    matrix: pd.DataFrame,
    released_embedding: torch.Tensor,
    seed: int,
    epochs: int,
    episodes_per_epoch: int,
    batch_size: int,
    learning_rate: float,
    task_loss_weight: float,
) -> tuple[CICOModel, pd.DataFrame, float]:
    set_random_seed(seed)
    model = initialize_model(matrix.shape[1], released_embedding, seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    dataset = CICODataset(
        matrix, n_sequences=episodes_per_epoch, n_shot=(1, 2)
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    metrics: list[dict[str, Any]] = []
    started = time.perf_counter()
    model.train()
    for epoch in range(epochs):
        for batch_index, batch in enumerate(loader):
            support_x = batch["support_x"].long()
            support_y = batch["support_y"].float()
            support_c = batch["support_c"].long()
            query_x = batch["query_x"].long()
            query_y = batch["query_y"].float()
            (
                task_loss,
                support_loss,
                query_loss,
                task_pred,
                support_pred,
                query_pred,
            ) = model.losses(support_x, support_y, support_c, query_x, query_y)
            feature_loss = support_loss + query_loss
            loss = task_loss_weight * task_loss + (1 - task_loss_weight) * feature_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                metrics.append(
                    {
                        "epoch": epoch,
                        "batch_idx": batch_index,
                        "task_acc": float(
                            (task_pred.argmax(-1) == support_c[:, 0]).float().mean()
                        ),
                        "support_ft_acc": float(
                            ((support_pred > 0) == support_y.bool()).float().mean()
                        ),
                        "query_ft_acc": float(
                            ((query_pred > 0) == query_y.bool()).float().mean()
                        ),
                        "task_loss": float(task_loss),
                        "support_ft_loss": float(support_loss),
                        "query_ft_loss": float(query_loss),
                        "loss": float(loss),
                    }
                )
    elapsed = time.perf_counter() - started
    return model, pd.DataFrame(metrics), elapsed


def save_checkpoint(
    path: Path,
    model: CICOModel,
    matrix: pd.DataFrame,
    condition: str,
    seed: int,
    epochs: int,
    training_parameters: dict[str, Any],
    provenance: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "condition": condition,
            "seed": seed,
            "epoch": epochs,
            "matrix_sha256": matrix_sha256(matrix),
            "object_names": list(map(str, matrix.index)),
            "task_names": list(map(str, matrix.columns)),
            "model_params": {
                "input_d": model.input_to_independent.num_embeddings,
                "embedding_d": model.input_to_independent.embedding_dim,
                "context_d": model.independent_to_context.out_features,
                "context_dependent_d": model.context_to_dependent.out_features,
                "task_output_d": model.context_to_task_output.out_features,
            },
            "training_parameters": training_parameters,
            "provenance": provenance or {},
        },
        path,
    )


def load_trained_model(path: Path) -> tuple[CICOModel, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    params = checkpoint["model_params"]
    model = CICOModel(**params)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint
