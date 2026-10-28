#!/usr/bin/env python3
"""Refresh derived reports from hash-verified saved probes, without fitting models."""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from evaluate_validation import released_model
from iscci_validation.behavior import load_tasks, trial_statistics
from iscci_validation.behavior_fidelity import item_fidelity
from iscci_validation.dataio import load_human_matrix, matrix_sha256
from iscci_validation.provenance import sha256_file, write_json
from iscci_validation.training import load_trained_model
from run_human_behavior_validation import (
    DATA,
    ROOT,
    UPSTREAM,
    geometry_behavior_association,
    input_geometry,
    registry,
    resolve,
    summary_table,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/human_behavior_validation.json"
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    output = resolve(config["output_directory"])
    manifest_path = output / "manifest.json"
    previous = json.loads(manifest_path.read_text())
    parent_hash = sha256_file(manifest_path)
    if previous["config"] != config or not previous["fidelity_gate"]["passed"]:
        raise ValueError(
            "Derived refresh requires unchanged inputs and a passed fidelity gate"
        )
    for record in previous["human_datasets"].values():
        if sha256_file(Path(record["path"])) != record["sha256"]:
            raise ValueError("Original human data changed")
    _, human = load_human_matrix(
        DATA / "leuven_dataset/leuven_combined_features_consolidated.csv"
    )
    conditions = config["required_conditions"] + config["optional_conditions"]
    matrices, models = registry(
        [resolve(p) for p in config["validation_directories"]],
        conditions,
        human,
        output / "models",
    )
    if {f"{c}_seed_{s}": sha256_file(p) for (c, s), p in models.items()} != previous[
        "checkpoint_sha256"
    ]:
        raise ValueError("Model cohort changed; run the main validator instead")
    if {c: matrix_sha256(m) for c, m in matrices.items()} != previous["matrix_sha256"]:
        raise ValueError("Feature matrices changed; run the main validator instead")
    for seed in range(4):
        if (
            sha256_file(UPSTREAM / "models" / f"1and2shot_isc-seed{seed}.pt")
            != previous["released_checkpoint_sha256"][str(seed)]
        ):
            raise ValueError("Released reference checkpoint changed")
    torch.set_num_threads(1)
    tasks = load_tasks(DATA)
    probe_checks = []
    for condition, seed, path in [
        ("released", s, UPSTREAM / "models" / f"1and2shot_isc-seed{s}.pt")
        for s in range(4)
    ] + [("human", s, models[("human", s)]) for s in range(4)]:
        model = (
            released_model(path)
            if condition == "released"
            else load_trained_model(path)[0]
        )
        checks = item_fidelity(
            model,
            tasks,
            list(human.index),
            UPSTREAM,
            config["fidelity"]["item_absolute_tolerance"],
        )
        checks["condition"], checks["seed"] = condition, seed
        probe_checks.append(checks)
    probe_checks = pd.concat(probe_checks, ignore_index=True)
    if not probe_checks["passed"].all():
        raise ValueError("Original-model architecture/probe parity failed")
    snapshot = output / f"manifest_before_derived_refresh_{parent_hash[:12]}.json"
    if not snapshot.exists():
        shutil.copy2(manifest_path, snapshot)
    probe_checks.to_csv(output / "fidelity_item_probe_checks.csv", index=False)
    refreshed_trials = []
    for directory in sorted((output / "per_model").iterdir()):
        bundle_path = directory / "manifest.json"
        if not bundle_path.exists():
            continue
        bundle = json.loads(bundle_path.read_text())
        if bundle["signature"]["plural_mapping_sha256"] != sha256_file(
            DATA / "leuven_dataset/leuven_singular_to_plural.csv"
        ):
            raise ValueError("Original plural preprocessing changed")
        raw_paths = sorted(directory.glob("*_trials.csv"))
        for path in [*raw_paths, *sorted(directory.glob("*_items.csv"))]:
            if bundle["output_sha256"].get(path.name) != sha256_file(path):
                raise ValueError(f"Saved item/trial probes changed: {path}")
        frame = pd.concat([pd.read_csv(path) for path in raw_paths], ignore_index=True)
        target = directory / "trial_statistics.csv"
        old_hash = sha256_file(target)
        backup = directory / f"trial_statistics_before_refresh_{old_hash[:12]}.csv"
        if not backup.exists():
            shutil.copy2(target, backup)
        trial_statistics(frame).to_csv(target, index=False)
        bundle.setdefault("derived_refresh_history", []).append(
            {
                "parent_bundle_sha256": sha256_file(bundle_path),
                "parent_trial_statistics_sha256": old_hash,
                "derived_implementation_sha256": sha256_file(
                    ROOT / "iscci_validation/behavior.py"
                ),
                "raw_probes_unchanged": True,
            }
        )
        bundle["output_sha256"][target.name] = sha256_file(target)
        write_json(bundle_path, bundle)
        table = pd.read_csv(target)
        table["model_id"] = directory.name
        refreshed_trials.append(table)
    pd.concat(refreshed_trials, ignore_index=True).to_csv(
        output / "all_individual_trial_statistics.csv", index=False
    )
    representations = pd.read_csv(output / "representation_per_seed.csv")
    for condition, matrix in matrices.items():
        for column, value in input_geometry(matrix, matrices["human"]).items():
            representations.loc[representations["condition"] == condition, column] = (
                value
            )
    representations.to_csv(output / "representation_per_seed.csv", index=False)
    summary = pd.read_csv(output / "behavior_summary.csv")
    lower = pd.read_csv(output / "similarity_levels_1_2_summary.csv")
    reliability = pd.read_csv(output / "human_reliability.csv")
    table = summary_table(
        conditions,
        matrices,
        models,
        summary,
        representations,
        False,
        lower,
        reliability,
    )
    table.to_csv(output / "condition_summary.csv", index=False)
    geometry_behavior_association(summary, representations, config).to_csv(
        output / "geometry_behavior_association.csv", index=False
    )
    sensitivity = representations.copy()
    sensitivity["input_object_rdm_spearman"] = sensitivity[
        "input_object_rdm_defined_pairs_spearman"
    ]
    geometry_behavior_association(summary, sensitivity, config).to_csv(
        output / "geometry_behavior_defined_pairs_sensitivity.csv", index=False
    )
    for level in ("items", "trials"):
        path = output / f"similarity_levels_1_2_{level}.csv"
        frame = pd.read_csv(path)
        frame["prediction_status"] = np.where(
            np.isfinite(frame["prediction"]), "defined", "undefined_zero_feature_vector"
        )
        frame.to_csv(path, index=False)
    manifest = {
        **previous,
        "derived_refresh": {
            "parent_manifest_sha256": parent_hash,
            "parent_manifest_snapshot": str(snapshot),
            "script_sha256": sha256_file(Path(__file__)),
            "current_implementation_sha256": {
                p.name: sha256_file(p)
                for p in [
                    ROOT / "run_human_behavior_validation.py",
                    ROOT / "iscci_validation/behavior.py",
                    ROOT / "iscci_validation/behavior_fidelity.py",
                    UPSTREAM / "models.py",
                    UPSTREAM / "analysis.py",
                ]
            },
            "refreshed_at": datetime.now(timezone.utc).isoformat(),
            "changes": [
                "Item-level parity rechecked against the unedited upstream model graph and original analysis on all eight released/reconstructed human checkpoints.",
                "Induction individual-response errors use original Human outcomes, not missing observed cells from other concatenated tasks.",
                "Primary input geometry uses original validation zero-vector/rank convention; finite-pair sensitivity remains separate.",
                "Condition table includes direct human-prediction metrics, uncertainty, and reliability.",
                "No saved model/item predictions, feature judgments, thresholds, matrices, training settings, or fitted CV mappings changed.",
            ],
            "derived_output_sha256": {
                p.name: sha256_file(p) for p in output.glob("*.csv")
            },
        },
    }
    write_json(manifest_path, manifest)
    print(f"Refreshed derived reports from unchanged probes: {output}")


if __name__ == "__main__":
    main()
