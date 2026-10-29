from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch

from iscci_validation.dataio import matrix_sha256
from iscci_validation.training import load_trained_model, save_checkpoint, train_model
from .inventory import sha256
from .provenance import (
    atomic_json,
    code_hashes,
    digest,
    run_lock,
    validate_output_directory,
)

SETTINGS = {
    "epochs": 400,
    "episodes_per_epoch": 1024,
    "batch_size": 128,
    "learning_rate": 0.001,
    "task_loss_weight": 0.5,
    "support_sizes": [1, 2],
    "freeze_semantic_embedding": True,
}


def frozen_embedding(activations, object_ids, expected_ids):
    if list(object_ids) != list(expected_ids) or len(set(object_ids)) != len(
        object_ids
    ):
        raise ValueError("Semantic object identity/order mismatch")
    if activations.dtype != np.float32:
        raise ValueError(
            "Semantic activations must remain float32 without numerical conversion"
        )
    if (
        activations.shape != (len(object_ids), 64)
        or not np.isfinite(activations).all()
        or (activations < 0).any()
        or (activations > 1).any()
    ):
        raise ValueError("Invalid semantic sigmoid activations")
    return torch.cat([torch.from_numpy(activations.copy()), torch.zeros(1, 64)], dim=0)


def load_context(directory, matrix, embedding, signature=None, expected_identity=None):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if signature is not None and manifest["signature"] != signature:
        raise ValueError("Downstream signature mismatch")
    identity = {
        key: manifest["signature"][key]
        for key in ["semantic_source", "context_source", "seed"]
    }
    if expected_identity is not None and identity != expected_identity:
        raise ValueError("Downstream source/seed identity mismatch")
    for name, expected in manifest["outputs"].items():
        if sha256(directory / name) != expected:
            raise ValueError(f"Downstream artifact hash mismatch: {name}")
    model, checkpoint = load_trained_model(directory / "checkpoint.pt")
    if (
        checkpoint["seed"] != identity["seed"]
        or checkpoint["condition"]
        != f"{identity['semantic_source']}__{identity['context_source']}"
    ):
        raise ValueError("Downstream checkpoint identity mismatch")
    if (
        checkpoint["task_names"] != list(matrix.columns)
        or checkpoint["training_parameters"] != manifest["signature"]["settings"]
    ):
        raise ValueError("Downstream feature identity or training settings mismatch")
    if checkpoint["matrix_sha256"] != matrix_sha256(matrix) or checkpoint[
        "object_names"
    ] != list(matrix.index):
        raise ValueError("Downstream matrix/object order mismatch")
    if not torch.equal(model.input_to_independent.weight, embedding):
        raise ValueError("Transferred semantic activations changed")
    model.input_to_independent.weight.requires_grad_(False)
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError("Nonfinite downstream weights")
    return model, checkpoint, manifest


def train_context(
    matrix,
    activations,
    object_ids,
    semantic_source,
    context_source,
    seed,
    directory,
    provenance,
    diagnostic=False,
    resume=False,
):
    directory = Path(directory)
    validate_output_directory(directory)
    embedding = frozen_embedding(activations, object_ids, list(matrix.index))
    settings = {**SETTINGS}
    if diagnostic:
        settings.update(epochs=2, episodes_per_epoch=16, batch_size=8)
    label = f"{semantic_source}__{context_source}"
    source = {
        **provenance,
        "semantic_source": semantic_source,
        "context_source": context_source,
        "initialization": "transferred_frozen_sigmoid_hub; all_remaining_weights_fresh",
    }
    signature = {
        "semantic_source": semantic_source,
        "context_source": context_source,
        "seed": seed,
        "matrix_sha256": matrix_sha256(matrix),
        "embedding_sha256": digest(activations.tolist()),
        "settings": settings,
        "provenance": source,
        "diagnostic": diagnostic,
        "code": code_hashes("context"),
    }
    with run_lock(directory):
        path = directory / "manifest.json"
        previous = json.loads(path.read_text()) if path.exists() else None
        if previous:
            if not resume:
                raise FileExistsError(
                    f"Existing downstream run requires --resume: {directory}"
                )
            if previous["signature"] != signature:
                raise ValueError("Downstream signature mismatch")
            if (
                previous.get("complete")
                or previous.get("status") == "diagnostic_complete"
            ):
                load_context(directory, matrix, embedding, signature)
                return {**previous, "reused": True}
        elif any(p.name != ".run.lock" for p in directory.iterdir()):
            raise ValueError("Existing downstream files without a matching manifest")
        atomic_json(
            path, {"signature": signature, "complete": False, "status": "running"}
        )
        try:
            model, metrics, seconds = train_model(
                matrix,
                embedding,
                seed,
                settings["epochs"],
                settings["episodes_per_epoch"],
                settings["batch_size"],
                settings["learning_rate"],
                settings["task_loss_weight"],
            )
            expected_updates = (
                settings["epochs"]
                * settings["episodes_per_epoch"]
                // settings["batch_size"]
            )
            if (
                len(metrics) != expected_updates
                or not np.isfinite(metrics.select_dtypes(include="number").values).all()
            ):
                raise ValueError(
                    "Downstream epoch/update/finite-metric validation failed"
                )
            if (
                not torch.equal(model.input_to_independent.weight, embedding)
                or model.input_to_independent.weight.requires_grad
            ):
                raise ValueError(
                    "Semantic embedding was changed or unfrozen during training"
                )
            fd, temporary = tempfile.mkstemp(
                dir=directory, prefix="checkpoint.", suffix=".pt"
            )
            os.close(fd)
            try:
                save_checkpoint(
                    Path(temporary),
                    model,
                    matrix,
                    label,
                    seed,
                    settings["epochs"],
                    settings,
                    source,
                )
                os.replace(temporary, directory / "checkpoint.pt")
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
            metrics.to_csv(directory / "training_metrics.csv", index=False)
            manifest = {
                "signature": signature,
                "complete": not diagnostic,
                "status": "diagnostic_complete" if diagnostic else "complete",
                "diagnostic": diagnostic,
                "settings": settings,
                "updates": len(metrics),
                "elapsed_seconds": seconds,
                "outputs": {
                    name: sha256(directory / name)
                    for name in ["checkpoint.pt", "training_metrics.csv"]
                },
            }
            atomic_json(path, manifest)
            load_context(directory, matrix, embedding, signature)
            return manifest
        except BaseException as exc:
            atomic_json(
                path,
                {
                    "signature": signature,
                    "complete": False,
                    "status": "interrupted",
                    "error": repr(exc),
                },
            )
            raise
