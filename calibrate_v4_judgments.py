#!/usr/bin/env python3
"""Freeze a V4 ordinal decision threshold using completed V2 and human cells."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.model_selection import KFold

from leuven_expansion.v4 import sha256_file, stable_json_hash, write_json


ROOT = Path(__file__).resolve().parent
DEFAULT_V2 = ROOT / "artifacts" / "leuven_full_labels" / "leuven_full_v2" / "feature_resolutions.csv"
DEFAULT_HUMAN = ROOT / "data" / "leuven_combined_features_consolidated.csv"


def load_calibration_matrices(
    v2_resolved: Path, human_features: Path
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    counts = pd.read_csv(human_features, index_col=0, encoding="ISO-8859-1")
    human_all = counts.gt(3).astype(np.int8)
    retained_columns = human_all.columns[human_all.sum(axis=0).gt(3)]
    human = human_all.loc[:, retained_columns]
    retained_ids = counts.columns.get_indexer(retained_columns)
    if (retained_ids < 0).any():
        raise ValueError("Could not map retained human columns to stable feature IDs")

    resolved = pd.read_csv(v2_resolved)
    required = {"word_normalized", "feature_id", "final_feature_value"}
    if missing := required - set(resolved.columns):
        raise ValueError(f"V2 resolutions are missing columns: {sorted(missing)}")
    resolved = resolved.loc[resolved["feature_id"].isin(retained_ids)].copy()
    duplicate_count = int(
        resolved.duplicated(["word_normalized", "feature_id"]).sum()
    )
    if duplicate_count:
        raise ValueError(f"V2 retained calibration cells contain {duplicate_count} duplicates")
    ordinal = resolved.pivot(
        index="word_normalized", columns="feature_id", values="final_feature_value"
    ).reindex(index=counts.index, columns=retained_ids)
    if ordinal.isna().any().any():
        missing_cells = int(ordinal.isna().sum().sum())
        raise ValueError(f"V2 retained calibration matrix has {missing_cells} missing cells")
    ordinal.columns = retained_columns
    values = ordinal.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or ((values < 0) | (values > 4)).any():
        raise ValueError("V2 ordinal values must be finite and within [0, 4]")
    return ordinal, human, {
        "word_count": int(len(human)),
        "retained_human_feature_count": int(human.shape[1]),
        "calibration_cell_count": int(human.size),
        "human_positive_rule": "participant_count > 3",
        "human_context_retention_rule": "positive_object_count > 3",
        "human_positive_cells": int(human.to_numpy().sum()),
        "v2_ordinal_values": sorted(map(float, np.unique(values))),
    }


def threshold_candidates(ordinal: pd.DataFrame) -> list[dict[str, Any]]:
    positive_values = sorted(
        float(value) for value in np.unique(ordinal.to_numpy()) if float(value) > 0
    )
    if not positive_values:
        raise ValueError("V2 calibration data contain no positive ordinal values")
    candidates = [
        {
            "threshold_id": "gt_0_locked_v2",
            "operator": "gt",
            "value": 0.0,
            "is_locked_v2": True,
            "conservatism": 0.0,
        }
    ]
    for value in positive_values:
        # The smallest positive cut is equivalent to >0 and is already represented.
        if value == positive_values[0]:
            continue
        candidates.append(
            {
                "threshold_id": f"ge_{value:g}",
                "operator": "ge",
                "value": value,
                "is_locked_v2": False,
                "conservatism": value,
            }
        )
    return candidates


def apply_threshold(values: np.ndarray, rule: dict[str, Any]) -> np.ndarray:
    if rule["operator"] == "gt":
        return (values > float(rule["value"])).astype(np.int8)
    if rule["operator"] == "ge":
        return (values >= float(rule["value"])).astype(np.int8)
    raise ValueError(f"Unsupported threshold operator: {rule['operator']}")


def _mean_row_cosine(left: np.ndarray, right: np.ndarray) -> float:
    numerator = np.einsum("ij,ij->i", left, right, dtype=np.float64)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    cosine = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=denominator > 0,
    )
    return float(np.mean(cosine))


def _object_rdm_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 3:
        return float("nan")
    left_rdm = np.nan_to_num(pdist(left, metric="cosine"), nan=0.0)
    right_rdm = np.nan_to_num(pdist(right, metric="cosine"), nan=0.0)
    if np.ptp(left_rdm) == 0 or np.ptp(right_rdm) == 0:
        return float("nan")
    result = spearmanr(left_rdm, right_rdm)
    return float(result.statistic)


def score_predictions(
    human: np.ndarray, predicted: np.ndarray
) -> dict[str, float]:
    truth = human.reshape(-1)
    guess = predicted.reshape(-1)
    return {
        "positive_precision": float(precision_score(truth, guess, zero_division=0)),
        "positive_recall": float(recall_score(truth, guess, zero_division=0)),
        "positive_f1": float(f1_score(truth, guess, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, guess)),
        "MCC": float(matthews_corrcoef(truth, guess)),
        "matrix_density": float(np.mean(guess)),
        "word_vector_cosine": _mean_row_cosine(human, predicted),
        "input_object_RDM_correlation": _object_rdm_correlation(human, predicted),
    }


def cross_validate_thresholds(
    ordinal: pd.DataFrame,
    human: pd.DataFrame,
    candidates: list[dict[str, Any]],
    folds: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
    fold_records: list[dict[str, Any]] = []
    assignments: list[dict[str, Any]] = []
    values = ordinal.to_numpy(dtype=np.float64)
    targets = human.to_numpy(dtype=np.int8)
    for fold, (_, heldout) in enumerate(splitter.split(values)):
        heldout_words = ordinal.index[heldout].tolist()
        assignments.extend(
            {"word": str(word), "fold": fold} for word in heldout_words
        )
        for rule in candidates:
            predicted = apply_threshold(values[heldout], rule)
            fold_records.append(
                {
                    "threshold_id": rule["threshold_id"],
                    "operator": rule["operator"],
                    "threshold_value": rule["value"],
                    "is_locked_v2": rule["is_locked_v2"],
                    "fold": fold,
                    "heldout_word_count": int(len(heldout)),
                    **score_predictions(targets[heldout], predicted),
                }
            )
    fold_metrics = pd.DataFrame(fold_records)
    metric_columns = [
        "positive_precision",
        "positive_recall",
        "positive_f1",
        "balanced_accuracy",
        "MCC",
        "matrix_density",
        "word_vector_cosine",
        "input_object_RDM_correlation",
    ]
    summary = (
        fold_metrics.groupby(
            ["threshold_id", "operator", "threshold_value", "is_locked_v2"],
            observed=True,
            as_index=False,
        )[metric_columns]
        .agg(["mean", "std", "min", "max"])
    )
    summary.columns = [
        "_".join(str(part) for part in column if str(part))
        if isinstance(column, tuple)
        else str(column)
        for column in summary.columns
    ]
    # Add pooled out-of-fold metrics. Each word appears in exactly one held-out fold.
    for rule in candidates:
        pooled = score_predictions(targets, apply_threshold(values, rule))
        mask = summary["threshold_id"] == rule["threshold_id"]
        for metric, value in pooled.items():
            summary.loc[mask, f"{metric}_pooled"] = value
    return fold_metrics, summary, assignments


def select_threshold(
    summary: pd.DataFrame, recall_gate: float
) -> tuple[pd.Series, bool]:
    eligible = summary.loc[summary["positive_recall_mean"] >= recall_gate].copy()
    gate_met = not eligible.empty
    pool = eligible if gate_met else summary.copy()
    pool = pool.sort_values(
        ["MCC_mean", "positive_precision_mean", "threshold_value"],
        ascending=[False, False, False],
        kind="stable",
    )
    return pool.iloc[0], gate_met


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-resolved", type=Path, default=DEFAULT_V2)
    parser.add_argument("--human-features", type=Path, default=DEFAULT_HUMAN)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "artifacts" / "v4" / "judgments"
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--recall-gate", type=float, default=0.80)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    threshold_path = output_dir / "judgment_threshold.json"
    downstream = output_dir.parent / "validation"
    if not threshold_path.exists() and downstream.exists() and any(downstream.iterdir()):
        raise ValueError("Calibration must be frozen before downstream V4 evaluation")

    ordinal, human, data_summary = load_calibration_matrices(
        args.v2_resolved.resolve(), args.human_features.resolve()
    )
    candidates = threshold_candidates(ordinal)
    fold_metrics, summary, assignments = cross_validate_thresholds(
        ordinal, human, candidates, args.folds, args.seed
    )
    selected, gate_met = select_threshold(summary, args.recall_gate)
    selected_id = str(selected["threshold_id"])
    selected_rule = next(
        candidate for candidate in candidates if candidate["threshold_id"] == selected_id
    )
    frozen = {
        "protocol_version": "v4-v2-word-fold-calibration-1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selection_rule": (
            "highest mean held-out-word MCC subject to mean positive recall >= gate; "
            "ties: precision then more conservative threshold"
        ),
        "recall_gate": args.recall_gate,
        "recall_gate_met": gate_met,
        "fold_count": args.folds,
        "split_unit": "word",
        "split_seed": args.seed,
        "selected_rule": selected_rule,
        "selected_cross_validated_metrics": {
            key: (bool(value) if isinstance(value, (np.bool_, bool)) else float(value))
            for key, value in selected.items()
            if key not in {"threshold_id", "operator"}
        },
        "locked_v2_rule": {"operator": "gt", "value": 0.0},
        "candidate_rules": candidates,
        "data_summary": data_summary,
        "v2_resolved": str(args.v2_resolved.resolve()),
        "v2_resolved_sha256": sha256_file(args.v2_resolved.resolve()),
        "human_features": str(args.human_features.resolve()),
        "human_features_sha256": sha256_file(args.human_features.resolve()),
        "leakage_controls": [
            "no V4 values",
            "no human phrase mapping",
            "no ISC-CI outputs",
            "no behavioral outcomes",
        ],
    }
    frozen["calibration_hash"] = stable_json_hash(
        {key: value for key, value in frozen.items() if key != "created_at"}
    )
    if threshold_path.exists():
        existing = json.loads(threshold_path.read_text(encoding="utf-8"))
        if existing.get("calibration_hash") != frozen["calibration_hash"]:
            raise ValueError("Existing frozen V4 threshold differs; refusing to overwrite")
        print(json.dumps(existing, indent=2))
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    fold_metrics.to_csv(output_dir / "calibration_metrics.csv", index=False)
    summary.to_csv(output_dir / "calibration_summary.csv", index=False)
    pd.DataFrame(assignments).sort_values(["fold", "word"]).to_csv(
        output_dir / "calibration_word_folds.csv", index=False
    )
    write_json(threshold_path, frozen)
    print(json.dumps(frozen, indent=2))


if __name__ == "__main__":
    main()
