#!/usr/bin/env python3
"""Build traceable raw, locked, calibrated, and source-only V4 matrices."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from calibrate_v4_judgments import apply_threshold
from leuven_expansion.feature_schema import load_candidate_feature_schema
from leuven_expansion.v4 import sha256_file, stable_json_hash, write_json


ROOT = Path(__file__).resolve().parent
DEFAULT_HUMAN = ROOT / "data" / "leuven_combined_features_consolidated.csv"


def load_and_validate_cells(
    candidate_bank: Path,
    resolved_values: Path,
    judgment_manifest: Path,
    words: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    bank = pd.read_csv(candidate_bank, dtype=str).fillna("")
    schema = load_candidate_feature_schema(candidate_bank)
    manifest = json.loads(judgment_manifest.read_text(encoding="utf-8"))
    if not manifest.get("complete"):
        raise ValueError("V4 judgment manifest is not complete")
    if manifest.get("candidate_inventory_hash") != schema["candidate_inventory_hash"]:
        raise ValueError("Candidate-bank hash differs from the judgment manifest")
    if manifest.get("candidate_bank_sha256") != sha256_file(candidate_bank):
        raise ValueError("Candidate-bank file differs from the judged file")

    cells = pd.read_csv(resolved_values, dtype={"candidate_id": str, "target_word": str})
    required = {"candidate_id", "target_word", "resolved_value"}
    if missing := required - set(cells.columns):
        raise ValueError(f"Resolved values are missing columns: {sorted(missing)}")
    if cells.duplicated(["candidate_id", "target_word"]).any():
        count = int(cells.duplicated(["candidate_id", "target_word"]).sum())
        raise ValueError(f"V4 resolved values contain {count} duplicate cells")
    values = pd.to_numeric(cells["resolved_value"], errors="coerce")
    if values.isna().any():
        raise ValueError("Unresolved cells cannot enter V4 matrices")
    if not values.between(0, 4).all():
        raise ValueError("V4 resolved values must be within [0, 4]")
    if set(cells["candidate_id"]) != set(schema["candidate_ids"]):
        raise ValueError("Resolved candidate IDs do not match the frozen bank")
    if set(cells["target_word"]) != set(words):
        raise ValueError("Resolved target words do not match the frozen Leuven inventory")
    expected_count = len(schema["candidate_ids"]) * len(words)
    if len(cells) != expected_count:
        raise ValueError(
            f"Expected {expected_count} exhaustive cells, found {len(cells)}"
        )
    cells = cells.assign(resolved_value=values.astype(np.float32))
    return bank, cells, manifest


def pivot_cells(
    cells: pd.DataFrame, words: list[str], candidate_ids: list[str], value: str
) -> pd.DataFrame:
    matrix = cells.pivot(index="target_word", columns="candidate_id", values=value)
    matrix = matrix.reindex(index=words, columns=candidate_ids)
    if matrix.isna().any().any():
        raise ValueError(f"Matrix {value} is incomplete after stable-ID alignment")
    matrix.index.name = "word"
    matrix.columns.name = None
    return matrix


def retention_summary(name: str, matrix: pd.DataFrame) -> dict[str, Any]:
    positive_counts = matrix.sum(axis=0)
    retained = positive_counts.gt(3)
    return {
        "matrix": name,
        "candidate_count_before_retention": int(matrix.shape[1]),
        "candidate_count_after_strict_gt_3": int(retained.sum()),
        "positive_cells_before_retention": int(matrix.to_numpy().sum()),
        "positive_cells_after_retention": int(
            matrix.loc[:, retained].to_numpy().sum()
        ),
        "matrix_density_before_retention": float(matrix.to_numpy().mean()),
        "matrix_density_after_retention": (
            float(matrix.loc[:, retained].to_numpy().mean()) if retained.any() else 0.0
        ),
        "retention_rule": "positive_object_count > 3",
    }


def source_membership(bank: pd.DataFrame, cells: pd.DataFrame) -> np.ndarray:
    source_by_candidate = {
        row.candidate_id: set(json.loads(row.source_words))
        for row in bank.itertuples(index=False)
    }
    return np.fromiter(
        (
            str(word) in source_by_candidate[str(candidate)]
            for candidate, word in zip(cells["candidate_id"], cells["target_word"])
        ),
        dtype=bool,
        count=len(cells),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-bank", type=Path, required=True)
    parser.add_argument("--resolved-values", type=Path, required=True)
    parser.add_argument("--threshold", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--human-features", type=Path, default=DEFAULT_HUMAN)
    parser.add_argument("--fixed-bank", type=Path)
    parser.add_argument("--judgment-manifest", type=Path)
    args = parser.parse_args()

    candidate_bank = args.candidate_bank.resolve()
    resolved_values = args.resolved_values.resolve()
    threshold_path = args.threshold.resolve()
    output = args.output_dir.resolve()
    judgment_manifest = (
        args.judgment_manifest.resolve()
        if args.judgment_manifest
        else resolved_values.parent / "judgment_manifest.json"
    )
    fixed_bank_path = (
        args.fixed_bank.resolve()
        if args.fixed_bank
        else candidate_bank.parent / "candidate_bank_v3_1_b_175.csv"
    )
    human_counts = pd.read_csv(
        args.human_features.resolve(), index_col=0, encoding="ISO-8859-1"
    )
    words = list(map(str, human_counts.index))
    bank, cells, judgments = load_and_validate_cells(
        candidate_bank, resolved_values, judgment_manifest, words
    )
    schema = load_candidate_feature_schema(candidate_bank)
    candidate_ids = list(schema["candidate_ids"])
    threshold = json.loads(threshold_path.read_text(encoding="utf-8"))
    selected_rule = threshold["selected_rule"]
    cells["resolved_binary_locked_v2"] = cells["resolved_value"].gt(0).astype(np.int8)
    cells["resolved_binary_calibrated"] = apply_threshold(
        cells["resolved_value"].to_numpy(), selected_rule
    )
    cells["source_generated"] = source_membership(bank, cells)
    cells["resolved_binary_source_only"] = (
        cells["resolved_binary_calibrated"].astype(bool) & cells["source_generated"]
    ).astype(np.int8)

    ensemble_raw = pivot_cells(cells, words, candidate_ids, "resolved_value")
    ensemble_locked = pivot_cells(
        cells, words, candidate_ids, "resolved_binary_locked_v2"
    ).astype(np.int8)
    ensemble_calibrated = pivot_cells(
        cells, words, candidate_ids, "resolved_binary_calibrated"
    ).astype(np.int8)
    ensemble_source_only = pivot_cells(
        cells, words, candidate_ids, "resolved_binary_source_only"
    ).astype(np.int8)
    fixed_bank = pd.read_csv(fixed_bank_path, dtype=str).fillna("")
    fixed_bank["fixed_v3_1_b_order"] = pd.to_numeric(
        fixed_bank["fixed_v3_1_b_order"], errors="raise"
    ).astype(int)
    fixed_ids = fixed_bank.sort_values("fixed_v3_1_b_order")["candidate_id"].tolist()
    if len(fixed_ids) != 175 or not set(fixed_ids).issubset(candidate_ids):
        raise ValueError("The locked 175-context V3.1-B inventory is not a V4 bank subset")

    matrices = {
        "v4_b_raw": ensemble_raw.loc[:, fixed_ids],
        "v4_b_locked_v2": ensemble_locked.loc[:, fixed_ids],
        "v4_b_calibrated": ensemble_calibrated.loc[:, fixed_ids],
        "v4_ensemble_raw": ensemble_raw,
        "v4_ensemble_locked_v2": ensemble_locked,
        "v4_ensemble_calibrated": ensemble_calibrated,
        "v4_ensemble_source_only": ensemble_source_only,
    }
    output.mkdir(parents=True, exist_ok=True)
    for name, matrix in matrices.items():
        matrix.to_csv(output / f"{name}.csv")

    provenance_columns = [
        column
        for column in [
            "candidate_id",
            "target_word",
            "resolved_value",
            "resolved_binary_locked_v2",
            "resolved_binary_calibrated",
            "source_generated",
            "resolved_binary_source_only",
            "confidence",
            "ambiguous",
            "resolution_method",
            "adjudicated",
            "needs_human_audit",
        ]
        if column in cells.columns
    ]
    provenance_path = output / "cell_provenance.parquet"
    try:
        cells[provenance_columns].to_parquet(
            provenance_path, index=False, compression="zstd"
        )
    except ImportError as error:
        raise RuntimeError(
            "Writing required cell_provenance.parquet needs pyarrow or fastparquet"
        ) from error

    inventory = pd.DataFrame(
        [
            retention_summary(name, matrix)
            for name, matrix in matrices.items()
            if "raw" not in name
        ]
    )
    inventory.to_csv(output / "context_inventory_comparison.csv", index=False)
    source_positive = cells["source_generated"] & cells["resolved_binary_calibrated"].eq(1)
    source_negative = cells["source_generated"] & cells["resolved_binary_calibrated"].eq(0)
    completed_positive = ~cells["source_generated"] & cells["resolved_binary_calibrated"].eq(1)
    pruning_completion = {
        "source_positive_cells": int(source_positive.sum()),
        "source_cells_pruned_by_judges": int(source_negative.sum()),
        "new_cells_added_by_completion": int(completed_positive.sum()),
        "completion_to_source_ratio": (
            float(completed_positive.sum() / source_positive.sum())
            if source_positive.sum()
            else None
        ),
    }
    write_json(output / "pruning_completion.json", pruning_completion)
    manifest = {
        "protocol_version": "v4-matrix-construction-1.0.0",
        "candidate_bank": str(candidate_bank),
        "candidate_bank_sha256": sha256_file(candidate_bank),
        "candidate_inventory_hash": schema["candidate_inventory_hash"],
        "fixed_bank": str(fixed_bank_path),
        "fixed_bank_sha256": sha256_file(fixed_bank_path),
        "resolved_values": str(resolved_values),
        "resolved_values_sha256": sha256_file(resolved_values),
        "judgment_manifest": str(judgment_manifest),
        "judgment_manifest_sha256": sha256_file(judgment_manifest),
        "judgment_protocol_hash": judgments.get("protocol_hash"),
        "threshold": str(threshold_path),
        "threshold_sha256": sha256_file(threshold_path),
        "calibration_hash": threshold.get("calibration_hash"),
        "selected_rule": selected_rule,
        "word_count": len(words),
        "ensemble_candidate_count": len(candidate_ids),
        "fixed_candidate_count": len(fixed_ids),
        "context_retention_rule": "positive_object_count > 3",
        "matrix_sha256": {
            name: sha256_file(output / f"{name}.csv") for name in matrices
        },
        "cell_provenance_sha256": sha256_file(provenance_path),
        "pruning_completion": pruning_completion,
    }
    manifest["matrix_manifest_hash"] = stable_json_hash(manifest)
    write_json(output / "matrix_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
