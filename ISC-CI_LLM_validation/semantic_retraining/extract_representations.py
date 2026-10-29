from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from iscci_validation.dataio import matrix_sha256
from .inventory import sha256
from .modeling import ISCModel
from .provenance import (
    atomic_file,
    atomic_json,
    code_hashes,
    digest,
    run_lock,
    validate_output_directory,
)
from .train_isc import fit

PROTOCOL = "isc-independent-32-corrected-targets-v1"


@torch.no_grad()
def extract(model, object_ids):
    if len(object_ids) != model.num_objects or len(set(object_ids)) != len(object_ids):
        raise ValueError("Object identity count or uniqueness mismatch")
    weight = model.item_input_to_hub_weights.weight
    # Canonical CPU sigmoid avoids device-dependent rounding when restoring GPU fits.
    bias = model.item_input_to_hub_weights.bias
    activations = torch.sigmoid(
        torch.nn.functional.linear(
            torch.eye(model.num_objects, dtype=weight.dtype),
            weight.detach().cpu(),
            None if bias is None else bias.detach().cpu(),
        )
    ).numpy()
    if activations.shape != (len(object_ids), 64) or not np.isfinite(activations).all():
        raise ValueError("Invalid sigmoid semantic activation matrix")
    return activations


def load_run(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("protocol") != PROTOCOL or manifest.get("complete") is not True:
        raise ValueError("A complete new semantic run is required")
    if manifest["signature"]["code"] != code_hashes("semantic"):
        raise ValueError("Semantic implementation signature changed")
    for name, expected in manifest["outputs"].items():
        if sha256(directory / name) != expected:
            raise ValueError(f"Semantic artifact hash mismatch: {name}")
    checkpoint = torch.load(
        directory / "checkpoint.pt", map_location="cpu", weights_only=False
    )
    if checkpoint["signature"] != manifest["signature"]:
        raise ValueError("Checkpoint/manifest signature mismatch")
    model = ISCModel(**checkpoint["model_params"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError("Nonfinite semantic weights")
    with np.load(directory / "representations.npz", allow_pickle=False) as saved:
        if list(saved["object_ids"]) != checkpoint["object_ids"] or not np.array_equal(
            saved["activations"], extract(model, checkpoint["object_ids"])
        ):
            raise ValueError("Saved activations differ from checkpoint extraction")
    return model, checkpoint, manifest


def train_run(
    dataset,
    condition,
    seed,
    config,
    directory,
    taxonomy_sha256,
    classification_sha256,
    resume=False,
):
    directory = Path(directory)
    validate_output_directory(directory)
    matrix = pd.DataFrame(
        dataset.values, index=dataset.object_ids, columns=dataset.feature_ids
    )
    signature = {
        "protocol": PROTOCOL,
        "condition": condition,
        "seed": seed,
        "matrix_sha256": matrix_sha256(matrix),
        "feature_tasks_sha256": digest(dataset.feature_tasks.tolist()),
        "taxonomy_sha256": taxonomy_sha256,
        "classification_sha256": classification_sha256,
        "training": asdict(config),
        "code": code_hashes("semantic"),
        "versions": {"torch": torch.__version__, "numpy": np.__version__},
    }
    with run_lock(directory):
        manifest_path = directory / "manifest.json"
        previous = (
            json.loads(manifest_path.read_text()) if manifest_path.exists() else None
        )
        if previous:
            if not resume:
                raise FileExistsError(f"Existing run requires --resume: {directory}")
            if previous["signature"] != signature:
                raise ValueError("Semantic run signature mismatch")
            if previous.get("complete"):
                load_run(directory)
                return {**previous, "reused": True}
        elif any(p.name != ".run.lock" for p in directory.iterdir()):
            raise ValueError("Existing files without a matching semantic manifest")
        atomic_json(
            manifest_path,
            {
                "protocol": PROTOCOL,
                "signature": signature,
                "complete": False,
                "status": "running",
            },
        )
        state = None
        progress = directory / "training_state.pt"
        if previous and progress.exists():
            saved = torch.load(progress, map_location="cpu", weights_only=False)
            if saved["signature"] != signature:
                raise ValueError("Training state signature mismatch")
            state = saved["state"]

        def checkpoint_callback(state):
            atomic_file(
                progress,
                lambda stream: torch.save(
                    {"signature": signature, "state": state}, stream
                ),
            )

        try:
            model, history, result = fit(
                dataset,
                seed,
                config,
                resume_state=state,
                checkpoint_callback=checkpoint_callback,
            )
            checkpoint_callback(result["training_state"])
            summary = {
                key: value for key, value in result.items() if key != "training_state"
            }
            params = {
                "num_objects": len(dataset.object_ids),
                "num_output": len(dataset.feature_ids),
                "num_tasks": dataset.task_count,
            }
            checkpoint = {
                "signature": signature,
                "state_dict": {
                    key: value.cpu() for key, value in model.state_dict().items()
                },
                "model_params": params,
                "object_ids": dataset.object_ids,
                "feature_ids": dataset.feature_ids,
                "feature_tasks": dataset.feature_tasks.tolist(),
                "result": summary,
            }
            atomic_file(
                directory / "checkpoint.pt",
                lambda stream: torch.save(checkpoint, stream),
            )
            atomic_json(directory / "training_metrics.json", history)
            atomic_json(directory / "exposure.json", dataset.exposure())
            outputs = [
                "checkpoint.pt",
                "training_metrics.json",
                "exposure.json",
                "training_state.pt",
            ]
            if result["complete"]:
                activations = extract(model, dataset.object_ids)
                atomic_file(
                    directory / "representations.npz",
                    lambda stream: np.savez(
                        stream,
                        object_ids=np.asarray(dataset.object_ids),
                        activations=activations,
                    ),
                )
                outputs.append("representations.npz")
            manifest = {
                "protocol": PROTOCOL,
                "signature": signature,
                "complete": result["complete"],
                "status": result["status"],
                "result": summary,
                "outputs": {name: sha256(directory / name) for name in outputs},
            }
            atomic_json(manifest_path, manifest)
            return manifest
        except BaseException as exc:
            atomic_json(
                manifest_path,
                {
                    "protocol": PROTOCOL,
                    "signature": signature,
                    "complete": False,
                    "status": "interrupted",
                    "error": repr(exc),
                },
            )
            raise
