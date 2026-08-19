#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from iscci_validation.evaluation import (
    ModelSpec,
    binary_recovery_metrics,
    ceiling_normalized_summary,
    evaluate_model,
    make_evaluation_contexts,
    matrix_rdm_comparisons,
    pairwise_model_comparisons,
    summarize_pairwise,
)
from iscci_validation.modeling import CICOModel
from iscci_validation.provenance import sha256_file, write_json
from iscci_validation.training import load_trained_model


ROOT = Path(__file__).resolve().parent
UPSTREAM = ROOT / "upstream" / "IntegratedSemanticsControlContextInference"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ISC-CI validation models.")
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs" / "v3_validation.json"
    )
    parser.add_argument(
        "--validation-dir", type=Path, default=ROOT / "artifacts" / "validation_v3"
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def released_model(path: Path) -> CICOModel:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    params = checkpoint["model_params"]
    model = CICOModel(
        input_d=params["input_d"],
        task_output_d=params["task_output_d"],
        embedding_d=params["embedding_d"],
        context_d=params["context_d"],
        context_dependent_d=params["context_dependent_d"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def discover_models(validation_dir: Path) -> list[ModelSpec]:
    specs: list[ModelSpec] = []
    for checkpoint_path in sorted(
        (validation_dir / "models").glob("*/seed_*/checkpoint.pt")
    ):
        condition = checkpoint_path.parents[1].name
        seed = int(checkpoint_path.parent.name.removeprefix("seed_"))
        specs.append(
            ModelSpec(
                model_id=f"{condition}_seed_{seed}",
                condition=condition,
                seed=seed,
                source="retrained",
                path=checkpoint_path,
            )
        )
    for seed in range(4):
        path = UPSTREAM / "models" / f"1and2shot_isc-seed{seed}.pt"
        specs.append(
            ModelSpec(
                model_id=f"released_seed_{seed}",
                condition="released",
                seed=seed,
                source="released",
                path=path,
            )
        )
    return specs


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    evaluation_config = config["evaluation"]
    validation_dir = args.validation_dir.resolve()
    results_dir = validation_dir / "evaluation"
    model_eval_dir = results_dir / "model_outputs"
    model_eval_dir.mkdir(parents=True, exist_ok=True)
    previous_model_hashes: dict[str, str] = {}
    previous_model_table_path = results_dir / "models_evaluated.csv"
    if previous_model_table_path.exists():
        previous_model_table = pd.read_csv(previous_model_table_path)
        previous_model_hashes = dict(
            zip(previous_model_table["model_id"], previous_model_table["sha256"])
        )
    previous_manifest_path = results_dir / "manifest.json"
    previous_evaluation_config = None
    if previous_manifest_path.exists():
        previous_evaluation_config = json.loads(
            previous_manifest_path.read_text(encoding="utf-8")
        ).get("evaluation_config")
    contexts_unchanged = previous_evaluation_config == evaluation_config

    matrices = {
        path.stem: pd.read_csv(path, index_col=0)
        for path in sorted((validation_dir / "data_matrices").glob("*.csv"))
    }
    required_conditions = {"human", "v2", "v3_A", "v3_B", "v3_C"}
    if not required_conditions.issubset(matrices):
        raise ValueError(
            f"Missing required matrices {sorted(required_conditions - set(matrices))}; "
            f"got {sorted(matrices)}"
        )
    object_names = list(matrices["human"].index.astype(str))
    evaluation = make_evaluation_contexts(
        object_count=len(object_names),
        pair_context_count=int(evaluation_config["pair_contexts"]),
        rdm_context_count=int(evaluation_config["rdm_contexts"]),
        context_dependent_rdm_count=int(
            evaluation_config["context_dependent_rdm_contexts"]
        ),
        seed=int(evaluation_config["seed"]),
    )
    np.savez(results_dir / "evaluation_contexts.npz", **evaluation)
    pd.DataFrame(
        {
            "support_1_index": evaluation["contexts"][:, 0],
            "support_2_index": evaluation["contexts"][:, 1],
            "support_1": [
                object_names[index] for index in evaluation["contexts"][:, 0]
            ],
            "support_2": [
                object_names[index] if index < len(object_names) else "<PAD>"
                for index in evaluation["contexts"][:, 1]
            ],
        }
    ).to_csv(results_dir / "evaluation_contexts.csv", index=False)

    specs = discover_models(validation_dir)
    expected_counts = {condition: 4 for condition in matrices}
    expected_counts["released"] = 4
    actual_counts = pd.Series([spec.condition for spec in specs]).value_counts().to_dict()
    if actual_counts != expected_counts:
        raise ValueError(f"Incomplete model set: expected {expected_counts}, got {actual_counts}")

    for spec in specs:
        output_path = model_eval_dir / f"{spec.model_id}.npz"
        model_sha256 = sha256_file(spec.path.resolve())
        if (
            output_path.exists()
            and not args.force
            and contexts_unchanged
            and previous_model_hashes.get(spec.model_id) == model_sha256
        ):
            continue
        model = (
            released_model(spec.path)
            if spec.source == "released"
            else load_trained_model(spec.path)[0]
        )
        outputs = evaluate_model(model, evaluation)
        np.savez(output_path, **outputs)
        print(f"Evaluated {spec.model_id}")

    pairwise = pairwise_model_comparisons(specs, model_eval_dir)
    pairwise.to_csv(results_dir / "pairwise_model_metrics.csv", index=False)
    summarize_pairwise(pairwise).to_csv(
        results_dir / "pairwise_group_summary.csv", index=False
    )
    ceiling_normalized_summary(pairwise).to_csv(
        results_dir / "ceiling_normalized_summary.csv", index=False
    )
    binary_recovery_metrics(matrices["human"], matrices["v2"]).to_csv(
        results_dir / "v2_binary_recovery.csv", index=False
    )
    matrix_rdm_comparisons(matrices).to_csv(
        results_dir / "input_matrix_rdm_comparisons.csv", index=False
    )
    model_table = pd.DataFrame(
        [
            {
                "model_id": spec.model_id,
                "condition": spec.condition,
                "seed": spec.seed,
                "source": spec.source,
                "path": str(spec.path.resolve()),
                "sha256": sha256_file(spec.path.resolve()),
            }
            for spec in specs
        ]
    )
    model_table.to_csv(results_dir / "models_evaluated.csv", index=False)
    write_json(
        results_dir / "manifest.json",
        {
            "protocol_version": config["protocol_version"],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "validation_manifest_sha256": sha256_file(
                validation_dir / "manifest.json"
            ),
            "evaluation_config": evaluation_config,
            "object_count": len(object_names),
            "context_count": int(len(evaluation["contexts"])),
            "models_evaluated": len(specs),
            "model_count_by_condition": actual_counts,
            "reuse_requires_matching_checkpoint_hash": True,
            "reuse_requires_unchanged_evaluation_config": True,
            "rdm_metric": "cosine distance; Spearman correlation of rank vectors",
            "membership_query_objects": "all 293 Leuven objects for every context",
        },
    )
    print(f"Evaluation complete: {results_dir}")


if __name__ == "__main__":
    main()
