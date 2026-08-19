#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from evaluate_validation import discover_models, released_model
from iscci_validation.provenance import sha256_file, write_json
from iscci_validation.simulations import (
    induction_correlations,
    induction_phenomenon_effects,
    paired_choice_agreement,
    score_arguments,
    similarity_asymmetry_metrics,
    similarity_context_metrics,
    similarity_domain_correlations,
)
from iscci_validation.training import load_trained_model


ROOT = Path(__file__).resolve().parent
UPSTREAM = ROOT / "upstream" / "IntegratedSemanticsControlContextInference"
DATA = UPSTREAM / "data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recreate the ISC-CI paper simulations for every validation model."
    )
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs" / "v3_validation.json"
    )
    parser.add_argument(
        "--validation-dir", type=Path, default=ROOT / "artifacts" / "validation_v3"
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def attach_model_columns(
    frame: pd.DataFrame, model_id: str, condition: str, seed: int
) -> pd.DataFrame:
    frame.insert(0, "seed", seed)
    frame.insert(0, "condition", condition)
    frame.insert(0, "model_id", model_id)
    return frame


def singularize_object_columns(
    frame: pd.DataFrame, plural_to_singular: dict[str, str]
) -> pd.DataFrame:
    frame = frame.copy()
    object_columns = [
        column
        for column in frame.columns
        if any(token in column for token in ("Premise", "Conclusion", "Distractor"))
    ]
    for column in object_columns:
        frame[column] = frame[column].replace(plural_to_singular)
    return frame


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    validation_dir = args.validation_dir.resolve()
    output_dir = validation_dir / "paper_simulations"
    per_model_dir = output_dir / "per_model"
    per_model_dir.mkdir(parents=True, exist_ok=True)
    matrices = {
        path.stem: pd.read_csv(path, index_col=0)
        for path in (validation_dir / "data_matrices").glob("*.csv")
    }
    object_names = list(matrices["human"].index.astype(str))
    singular_plural = pd.read_csv(
        DATA / "leuven_dataset" / "leuven_singular_to_plural.csv", index_col=0
    )
    plural_to_singular = dict(
        zip(singular_plural["plural"], singular_plural["singular"])
    )
    specs = discover_models(validation_dir)
    expected_model_count = 4 * (len(matrices) + 1)
    if len(specs) != expected_model_count:
        raise ValueError(
            f"Expected {expected_model_count} models for {len(matrices)} matrices plus "
            f"released checkpoints, found {len(specs)}"
        )

    induction_data = pd.read_csv(
        DATA / "generalization_experiments" / "induction_arguments.csv", index_col=0
    )
    phenomena_data = pd.read_csv(
        DATA / "generalization_experiments" / "inductive_phenomena.csv", index_col=0
    )
    similarity_domain_data = pd.read_csv(
        DATA / "similarity_experiments" / "similarity_in_domain.csv", index_col=0
    )
    asymmetry_data = pd.read_csv(
        DATA / "similarity_experiments" / "similarity_asymmetry.csv", index_col=0
    )
    similarity_context_data = pd.read_csv(
        DATA / "similarity_experiments" / "similarity_in_context.csv", index_col=0
    )
    thematic_data = pd.read_csv(
        DATA / "generalization_experiments" / "thematic_arguments.csv", index_col=0
    )
    nonmonotonicity_data = pd.read_csv(
        DATA
        / "generalization_experiments"
        / "context_dependent_nonmonotonicity.csv",
        index_col=0,
    )
    induction_data = singularize_object_columns(induction_data, plural_to_singular)
    phenomena_data = singularize_object_columns(phenomena_data, plural_to_singular)
    similarity_domain_data = singularize_object_columns(
        similarity_domain_data, plural_to_singular
    )
    asymmetry_data = singularize_object_columns(asymmetry_data, plural_to_singular)
    similarity_context_data = singularize_object_columns(
        similarity_context_data, plural_to_singular
    )
    thematic_data = singularize_object_columns(thematic_data, plural_to_singular)
    nonmonotonicity_data = singularize_object_columns(
        nonmonotonicity_data, plural_to_singular
    )

    result_names = [
        "induction_correlations",
        "induction_phenomena",
        "similarity_domain_correlations",
        "similarity_asymmetry",
        "paired_choice_agreement",
        "similarity_context",
    ]
    combined: dict[str, list[pd.DataFrame]] = {name: [] for name in result_names}
    for spec in specs:
        completion_path = per_model_dir / f"{spec.model_id}_complete.json"
        reusable = False
        if completion_path.exists() and not args.force:
            completion = json.loads(completion_path.read_text(encoding="utf-8"))
            reusable = completion.get("model_sha256") == sha256_file(
                spec.path.resolve()
            ) and all(
                (per_model_dir / f"{spec.model_id}_{name}.csv").exists()
                for name in result_names
            )
        if reusable:
            for name in result_names:
                combined[name].append(
                    pd.read_csv(per_model_dir / f"{spec.model_id}_{name}.csv")
                )
            continue
        print(f"Running paper simulations for {spec.model_id}")
        model = (
            released_model(spec.path)
            if spec.source == "released"
            else load_trained_model(spec.path)[0]
        )
        induction_scores = score_arguments(model, induction_data, object_names)
        phenomenon_scores = score_arguments(model, phenomena_data, object_names)
        results = {
            "induction_correlations": induction_correlations(
                induction_data, induction_scores
            ),
            "induction_phenomena": induction_phenomenon_effects(
                phenomena_data, phenomenon_scores
            ),
            "similarity_domain_correlations": similarity_domain_correlations(
                similarity_domain_data, model, object_names
            ),
            "similarity_asymmetry": similarity_asymmetry_metrics(
                asymmetry_data, model, object_names
            ),
            "paired_choice_agreement": pd.concat(
                [
                    paired_choice_agreement(
                        thematic_data, model, object_names, "thematic_arguments"
                    ),
                    paired_choice_agreement(
                        nonmonotonicity_data,
                        model,
                        object_names,
                        "context_dependent_nonmonotonicity",
                    ),
                ],
                ignore_index=True,
            ),
            "similarity_context": similarity_context_metrics(
                similarity_context_data,
                model,
                object_names,
                seed=int(config["evaluation"]["seed"]) + spec.seed,
            ),
        }
        for name, frame in results.items():
            frame = attach_model_columns(
                frame, spec.model_id, spec.condition, spec.seed
            )
            frame.to_csv(per_model_dir / f"{spec.model_id}_{name}.csv", index=False)
            combined[name].append(frame)
        write_json(
            completion_path,
            {
                "model_id": spec.model_id,
                "model_sha256": sha256_file(spec.path.resolve()),
                "protocol_version": config["protocol_version"],
            },
        )

    for name, frames in combined.items():
        pd.concat(frames, ignore_index=True).to_csv(
            output_dir / f"{name}.csv", index=False
        )
    completion_protocol_counts: dict[str, int] = {}
    for spec in specs:
        completion = json.loads(
            (per_model_dir / f"{spec.model_id}_complete.json").read_text(
                encoding="utf-8"
            )
        )
        protocol = str(completion["protocol_version"])
        completion_protocol_counts[protocol] = (
            completion_protocol_counts.get(protocol, 0) + 1
        )
    write_json(
        output_dir / "manifest.json",
        {
            "protocol_version": config["protocol_version"],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model_count": len(specs),
            "completion_protocol_version_counts": completion_protocol_counts,
            "reuse_requires_matching_checkpoint_hash": True,
            "source_notebooks": [
                str((UPSTREAM / "1. Part 1 Inductive Inference.ipynb").resolve()),
                str((UPSTREAM / "2. Part 2 Similarity.ipynb").resolve()),
            ],
            "adaptation_notes": [
                "Notebook calculations moved to a deterministic command-line runner.",
                "Pandas aggregations explicitly select numeric columns for pandas 2 compatibility.",
                "The stochastic LCA receives stable per-model/per-row seeds.",
                "Behavioral display labels are mapped through leuven_singular_to_plural.csv.",
            ],
            "lca": {"simulations": 100, "steps": 500, "burn_in": 100},
        },
    )
    print(f"Paper simulations complete: {output_dir}")


if __name__ == "__main__":
    main()
