#!/usr/bin/env python3
"""Recreate the readonly baseline and original-graph QA for the finalized V4 run."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from iscci_validation.behavior import load_tasks
from iscci_validation.behavior_fidelity import item_fidelity
from iscci_validation.provenance import sha256_file, write_json
from iscci_validation.training import load_trained_model


ROOT = Path(__file__).resolve().parent
OLD = ROOT / "artifacts/human_behavior_validation_statistical"
NEW = ROOT / "artifacts/human_behavior_validation_v4_20261002"
VALIDATION = ROOT / "artifacts/validation_v4_20261002"
OUTPUT = ROOT / "reports/v4_20261002"
UPSTREAM = ROOT / "upstream/IntegratedSemanticsControlContextInference"
TASKS = (
    "induction",
    "phenomena",
    "similarity",
    "asymmetry",
    "context",
    "thematic",
    "nonmonotonicity",
)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    producer = {"implementation_sha256": sha256_file(Path(__file__))}
    files = []
    for directory in sorted((OLD / "per_model").iterdir()):
        if not directory.is_dir():
            continue
        for task in TASKS:
            for level in ("items", "trials"):
                name = f"{task}_{level}.csv"
                original = sha256_file(directory / name)
                current = sha256_file(NEW / "per_model" / directory.name / name)
                files.append(
                    {
                        "model": directory.name,
                        "file": name,
                        "original_sha256": original,
                        "current_sha256": current,
                        "identical": original == current,
                    }
                )
    failures = sum(not row["identical"] for row in files)
    if len(files) != 504 or failures:
        raise ValueError("Baseline prediction files changed or are incomplete")
    write_json(
        OUTPUT / "baseline_prediction_parity.json",
        {
            **producer,
            "complete": True,
            "files_compared": len(files),
            "mismatches": failures,
            "files": files,
        },
    )
    tables = {}
    for name in ("behavior_per_seed.csv", "behavior_summary.csv"):
        original = pd.read_csv(OLD / name)
        current = pd.read_csv(NEW / name)
        current = current[
            current.condition.isin(original.condition.unique())
        ].reset_index(drop=True)
        original = original.reset_index(drop=True)
        pd.testing.assert_frame_equal(original, current, atol=1e-12, rtol=1e-12)
        numerical = original.select_dtypes("number").columns
        error = float((original[numerical] - current[numerical]).abs().max().max())
        tables[name] = {
            "rows": len(original),
            "max_numeric_difference": error,
            "unchanged_within": 1e-12,
        }
    write_json(
        OUTPUT / "baseline_summary_parity.json",
        {
            **producer,
            "complete": True,
            "tables": tables,
        },
    )
    paths = sorted((VALIDATION / "models").glob("v4_*/seed_*/checkpoint.pt"))
    if len(paths) != 20:
        raise ValueError("Expected all 20 completed V4 checkpoints")
    names = list(pd.read_csv(VALIDATION / "data_matrices/human.csv", index_col=0).index)
    tasks = load_tasks(UPSTREAM / "data")
    graph_checks = []
    for path in paths:
        model, checkpoint = load_trained_model(path)
        metrics = pd.read_csv(path.parent / "training_metrics.csv")
        if checkpoint["epoch"] != 400 or len(metrics) != 3200:
            raise ValueError(f"Incomplete training: {path}")
        if not np.isfinite(metrics.to_numpy()).all() or not all(
            torch.isfinite(value).all() for value in model.state_dict().values()
        ):
            raise ValueError(f"Nonfinite training: {path}")
        check = item_fidelity(model, tasks, names, UPSTREAM, 1e-5)
        check["condition"], check["seed"] = checkpoint["condition"], checkpoint["seed"]
        graph_checks.append(check)
        print(
            f"Original-graph parity: {checkpoint['condition']} seed {checkpoint['seed']}",
            flush=True,
        )
    checks = pd.concat(graph_checks, ignore_index=True)
    if len(checks) != 40 or not checks.passed.all():
        raise ValueError("Original-graph parity failed")
    checks.to_csv(OUTPUT / "v4_original_graph_parity.csv", index=False)
    write_json(
        OUTPUT / "v4_original_graph_parity.json",
        {
            **producer,
            "complete": True,
            "checks": len(checks),
            "passed": True,
            "max_absolute_error": float(checks.max_absolute_error.max()),
            "checkpoint_sha256": {
                str(path.relative_to(ROOT)): sha256_file(path) for path in paths
            },
        },
    )
    print("All finalized V4 fidelity audits passed", flush=True)


if __name__ == "__main__":
    main()
