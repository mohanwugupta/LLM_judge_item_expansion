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
    if manifest.get("resolved_values_sha256") and manifest[
        "resolved_values_sha256"
    ] != sha256_file(resolved_values):
        raise ValueError("Resolved-value file differs from the finalized manifest")

    cells = pd.read_csv(
        resolved_values, dtype={"candidate_id": str, "target_word": str}
    )
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
    bank_ids = set(schema["candidate_ids"])
    if not set(cells["candidate_id"]).issubset(bank_ids) or not set(
        cells["target_word"]
    ).issubset(words):
        raise ValueError(
            "Resolved cells do not match the frozen candidate/word inventories"
        )
    expected_count = len(schema["candidate_ids"]) * len(words)
    unresolved_count = manifest.get("unresolved_cells", 0)
    if type(unresolved_count) is not int or unresolved_count < 0:
        raise ValueError("Invalid unresolved-cell count in manifest")
    excluded = set(manifest.get("excluded_candidate_ids", []))
    if unresolved_count:
        if manifest.get("unresolved_policy") != "exclude_incomplete_candidates":
            raise ValueError(
                "Unresolved cells need an explicit candidate-exclusion policy"
            )
        cap = manifest.get("max_unresolved_cells", 0)
        if type(cap) is not int or unresolved_count > cap:
            raise ValueError("Unresolved cells exceed the authorized cap")
        audit_path = resolved_values.parent / "unresolved_cells.csv"
        if not audit_path.exists() or sha256_file(audit_path) != manifest.get(
            "unresolved_cells_sha256"
        ):
            raise ValueError(
                "Unresolved-cell audit differs from the finalized manifest"
            )
        failed = pd.read_csv(
            audit_path, dtype={"candidate_id": str, "target_word": str}
        )
        keys = ["candidate_id", "target_word"]
        if len(failed) != unresolved_count or failed.duplicated(keys).any():
            raise ValueError(
                "Unresolved-cell audit has incorrect coverage or duplicates"
            )
        if not set(failed["candidate_id"]).issubset(bank_ids) or not set(
            failed["target_word"]
        ).issubset(words):
            raise ValueError("Unresolved-cell audit contains unknown candidates/words")
        if not failed["reason"].eq("adjudicator_failed_missing_final_value").all():
            raise ValueError("Only recorded failed adjudications may be excluded")
        if excluded != set(failed["candidate_id"]) or manifest.get(
            "excluded_candidate_count"
        ) != len(excluded):
            raise ValueError(
                "Excluded candidates do not exactly match the unresolved audit"
            )
        if len(
            cells.loc[cells["candidate_id"].isin(excluded), keys].merge(
                failed[keys], on=keys
            )
        ):
            raise ValueError("A cell cannot be both resolved and unresolved")
        combined_counts = (
            cells.groupby("candidate_id")
            .size()
            .add(failed.groupby("candidate_id").size(), fill_value=0)
        )
        if (
            set(combined_counts.index) != bank_ids
            or not combined_counts.eq(len(words)).all()
        ):
            raise ValueError(
                "Resolved and unresolved cells do not cover the frozen cross-product"
            )
    elif excluded:
        raise ValueError(
            "Candidate exclusions cannot be declared without unresolved cells"
        )
    if len(cells) + unresolved_count != expected_count:
        raise ValueError(
            f"Expected {expected_count} exhaustive cells, found {len(cells)} resolved and {unresolved_count} unresolved"
        )
    cells = cells.assign(resolved_value=values.astype(np.float32))
    # No missing judgment is converted into a negative; drop entire affected contexts.
    if excluded:
        cells = cells.loc[~cells["candidate_id"].isin(excluded)].copy()
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
        "positive_cells_after_retention": int(matrix.loc[:, retained].to_numpy().sum()),
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
    excluded_ids = set(judgments.get("excluded_candidate_ids", []))
    candidate_ids = [c for c in schema["candidate_ids"] if c not in excluded_ids]
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
    if len(fixed_ids) != 175 or not set(fixed_ids).issubset(schema["candidate_ids"]):
        raise ValueError(
            "The locked 175-context V3.1-B inventory is not a V4 bank subset"
        )
    fixed_excluded = [c for c in fixed_ids if c in excluded_ids]
    fixed_ids = [c for c in fixed_ids if c not in excluded_ids]
    if not fixed_ids:
        raise ValueError(
            "No complete fixed-B candidates remain after declared exclusions"
        )

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
    source_positive = cells["source_generated"] & cells[
        "resolved_binary_calibrated"
    ].eq(1)
    source_negative = cells["source_generated"] & cells[
        "resolved_binary_calibrated"
    ].eq(0)
    completed_positive = ~cells["source_generated"] & cells[
        "resolved_binary_calibrated"
    ].eq(1)
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
        "frozen_candidate_count": schema["n_features"],
        "unresolved_cells": judgments.get("unresolved_cells", 0),
        "unresolved_policy": judgments.get("unresolved_policy", "none"),
        "excluded_candidate_ids": judgments.get("excluded_candidate_ids", []),
        "excluded_candidate_count": len(excluded_ids),
        "fixed_excluded_candidate_ids": fixed_excluded,
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
