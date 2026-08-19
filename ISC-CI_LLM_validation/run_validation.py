#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch

from iscci_validation.dataio import (
    load_human_matrix,
    load_v2_matrix,
    load_v3_matrix,
    load_v4_matrix,
    matrix_sha256,
)
from iscci_validation.provenance import (
    environment_record,
    git_revision,
    sha256_file,
    write_json,
)
from iscci_validation.training import save_checkpoint, train_model


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
UPSTREAM = ROOT / "upstream" / "IntegratedSemanticsControlContextInference"
HUMAN_CSV = (
    UPSTREAM / "data" / "leuven_dataset" / "leuven_combined_features_consolidated.csv"
)
V2_CSV = (
    PROJECT_ROOT
    / "artifacts"
    / "leuven_full_labels"
    / "leuven_full_v2"
    / "feature_resolutions.csv"
)
V3_DIR = ROOT / "artifacts" / "v3_consolidation"
V3_1_DIR = ROOT / "artifacts" / "v3_1_consolidation"
RELEASED_MODEL = UPSTREAM / "models" / "1and2shot_isc-seed3.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the ISC-CI validation conditions.")
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs" / "v3_validation.json"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "artifacts" / "validation_v3"
    )
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--conditions", nargs="+")
    parser.add_argument("--include-v3-1", action="store_true")
    parser.add_argument("--include-v4", action="store_true")
    parser.add_argument(
        "--v4-matrix-dir",
        type=Path,
        help="Directory containing completed V4 binary matrices.",
    )
    parser.add_argument(
        "--base-validation-dir",
        type=Path,
        help="Completed validation directory to copy and reuse when output is absent.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_conditions(
    include_v3_1: bool = False,
    include_v4: bool = False,
    v4_matrix_dir: Path | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    human_counts, human = load_human_matrix(HUMAN_CSV)
    v2, v2_missing_pairs = load_v2_matrix(V2_CSV, human_counts)
    conditions = {"human": human, "v2": v2}
    for prompt in "ABC":
        conditions[f"v3_{prompt}"] = load_v3_matrix(
            V3_DIR / f"v3_{prompt}_training_binary.csv", human.index
        )
    if include_v3_1 or include_v4:
        for prompt in "ABC":
            conditions[f"v3_1_{prompt}"] = load_v3_matrix(
                V3_1_DIR / f"v3_1_{prompt}_training_binary.csv", human.index
            )
    if include_v4:
        if v4_matrix_dir is None:
            raise ValueError("--include-v4 requires --v4-matrix-dir or config v4.matrix_dir")
        v4_files = {
            "v4_b_locked_v2": "v4_b_locked_v2.csv",
            "v4_b_calibrated": "v4_b_calibrated.csv",
            "v4_ensemble_locked_v2": "v4_ensemble_locked_v2.csv",
            "v4_ensemble_calibrated": "v4_ensemble_calibrated.csv",
        }
        for condition, filename in v4_files.items():
            conditions[condition] = load_v4_matrix(
                v4_matrix_dir / filename, human.index
            )
    return conditions, {
        "v2_missing_pairs_filled_zero": v2_missing_pairs,
        "condition_task_counts": {
            condition: int(matrix.shape[1]) for condition, matrix in conditions.items()
        },
        "condition_positive_cells": {
            condition: int(matrix.values.sum()) for condition, matrix in conditions.items()
        },
    }


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    training_config = dict(config["training"])
    epochs = args.epochs if args.epochs is not None else int(training_config["epochs"])
    seeds = args.seeds if args.seeds is not None else list(training_config["seeds"])
    configured_v4_dir = config.get("v4", {}).get("matrix_dir")
    v4_matrix_dir = args.v4_matrix_dir
    if v4_matrix_dir is None and configured_v4_dir:
        configured_path = Path(configured_v4_dir)
        v4_matrix_dir = (
            configured_path
            if configured_path.is_absolute()
            else (PROJECT_ROOT / configured_path).resolve()
        )
    conditions, data_summary = load_conditions(
        args.include_v3_1,
        args.include_v4,
        v4_matrix_dir.resolve() if v4_matrix_dir else None,
    )
    selected_conditions = args.conditions or list(conditions)
    unknown = set(selected_conditions) - set(conditions)
    if unknown:
        raise ValueError(f"Unknown conditions: {sorted(unknown)}")

    output_dir = args.output_dir.resolve()
    base_validation_dir = (
        args.base_validation_dir.resolve() if args.base_validation_dir else None
    )
    if base_validation_dir is not None and not output_dir.exists():
        if base_validation_dir == output_dir:
            raise ValueError("Base and output validation directories must differ")
        shutil.copytree(base_validation_dir, output_dir)
        shutil.copy2(
            base_validation_dir / "manifest.json",
            output_dir / "base_validation_manifest.json",
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(output_dir / "run.log"),
            logging.StreamHandler(),
        ],
    )
    released = torch.load(RELEASED_MODEL, map_location="cpu", weights_only=False)
    released_embedding = released["state_dict"]["input_to_independent.weight"].detach()
    if float(released_embedding[-1].abs().max()) != 0.0:
        raise ValueError("Released padding embedding is not zero")

    matrices_dir = output_dir / "data_matrices"
    matrices_dir.mkdir(exist_ok=True)
    for condition, matrix in conditions.items():
        matrix.to_csv(matrices_dir / f"{condition}.csv")

    run_rows: list[dict[str, object]] = []
    parameters = {
        "epochs": epochs,
        "episodes_per_epoch": int(training_config["episodes_per_epoch"]),
        "batch_size": int(training_config["batch_size"]),
        "support_sizes": list(training_config["support_sizes"]),
        "learning_rate": float(training_config["learning_rate"]),
        "task_loss_weight": float(training_config["task_loss_weight"]),
        "freeze_released_semantic_embedding": True,
    }
    v4_provenance: dict[str, object] = {}
    if args.include_v4:
        assert v4_matrix_dir is not None
        matrix_manifest = v4_matrix_dir / "matrix_manifest.json"
        if not matrix_manifest.exists():
            raise FileNotFoundError(f"Missing V4 matrix manifest: {matrix_manifest}")
        matrix_metadata = json.loads(matrix_manifest.read_text(encoding="utf-8"))
        v4_provenance = {
            "candidate_bank_hash": matrix_metadata["candidate_inventory_hash"],
            "judgment_manifest_hash": matrix_metadata["judgment_manifest_sha256"],
            "calibration_hash": matrix_metadata["calibration_hash"],
            "matrix_manifest_hash": matrix_metadata["matrix_manifest_hash"],
            "matrix_manifest_sha256": sha256_file(matrix_manifest),
            "semantic_embedding_hash": sha256_file(RELEASED_MODEL),
            "training_config_hash": sha256_file(args.config.resolve()),
            "source_commit": git_revision(PROJECT_ROOT),
        }
    for condition in selected_conditions:
        matrix = conditions[condition]
        for seed in seeds:
            run_dir = output_dir / "models" / condition / f"seed_{seed}"
            checkpoint_path = run_dir / "checkpoint.pt"
            metrics_path = run_dir / "training_metrics.csv"
            if checkpoint_path.exists() and not args.force:
                checkpoint = torch.load(
                    checkpoint_path, map_location="cpu", weights_only=False
                )
                if (
                    checkpoint.get("matrix_sha256") == matrix_sha256(matrix)
                    and checkpoint.get("epoch") == epochs
                    and checkpoint.get("training_parameters") == parameters
                    and (
                        not condition.startswith("v4_")
                        or checkpoint.get("provenance") == v4_provenance
                    )
                ):
                    logging.info("Reusing %s seed=%s", condition, seed)
                    run_rows.append(
                        {
                            "condition": condition,
                            "seed": seed,
                            "epochs": epochs,
                            "task_count": matrix.shape[1],
                            "elapsed_seconds": 0.0,
                            "reused": True,
                        }
                    )
                    continue
            logging.info(
                "Training condition=%s seed=%s tasks=%s epochs=%s",
                condition,
                seed,
                matrix.shape[1],
                epochs,
            )
            model, metrics, elapsed = train_model(
                matrix=matrix,
                released_embedding=released_embedding,
                seed=seed,
                epochs=epochs,
                episodes_per_epoch=parameters["episodes_per_epoch"],
                batch_size=parameters["batch_size"],
                learning_rate=parameters["learning_rate"],
                task_loss_weight=parameters["task_loss_weight"],
            )
            run_dir.mkdir(parents=True, exist_ok=True)
            metrics.to_csv(metrics_path, index=False)
            save_checkpoint(
                checkpoint_path,
                model,
                matrix,
                condition,
                seed,
                epochs,
                parameters,
                v4_provenance if condition.startswith("v4_") else None,
            )
            if not torch.equal(
                model.input_to_independent.weight.detach(), released_embedding
            ):
                raise ValueError("Frozen semantic embedding changed during training")
            logging.info(
                "Completed condition=%s seed=%s elapsed=%.1fs", condition, seed, elapsed
            )
            run_rows.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "epochs": epochs,
                    "task_count": matrix.shape[1],
                    "elapsed_seconds": elapsed,
                    "reused": False,
                }
            )
    run_table = pd.DataFrame(run_rows)
    run_table.to_csv(output_dir / "training_runs.csv", index=False)
    manifest = {
        "protocol_version": config["protocol_version"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": str(args.config.resolve()),
        "config_sha256": sha256_file(args.config.resolve()),
        "human_csv_sha256": sha256_file(HUMAN_CSV),
        "v2_csv_sha256": sha256_file(V2_CSV),
        "consolidation_manifest_sha256_by_version": {
            "v3": sha256_file(V3_DIR / "manifest.json"),
            **(
                {"v3.1": sha256_file(V3_1_DIR / "manifest.json")}
                if args.include_v3_1 or args.include_v4
                else {}
            ),
        },
        "v4_matrix_dir": str(v4_matrix_dir) if v4_matrix_dir else None,
        "v4_provenance": v4_provenance,
        "base_validation_dir": (
            str(base_validation_dir) if base_validation_dir is not None else None
        ),
        "base_validation_manifest_sha256": (
            sha256_file(output_dir / "base_validation_manifest.json")
            if (output_dir / "base_validation_manifest.json").exists()
            else None
        ),
        "released_model_sha256": sha256_file(RELEASED_MODEL),
        "released_embedding_frozen": True,
        "released_embedding_shape": list(released_embedding.shape),
        "released_checkpoint_metadata_batch_size": released["dataset_params"].get(
            "batch_size"
        ),
        "executed_batch_size_inferred_from_released_metrics": 128,
        "released_checkpoint_epoch": released["epoch"],
        "training_parameters": parameters,
        "data_summary": data_summary,
        "matrix_sha256_by_condition": {
            condition: matrix_sha256(matrix) for condition, matrix in conditions.items()
        },
        "selected_conditions": selected_conditions,
        "selected_seeds": seeds,
        "environment": environment_record(),
    }
    write_json(output_dir / "manifest.json", manifest)
    logging.info("Training stage complete: %s", output_dir)


if __name__ == "__main__":
    main()
