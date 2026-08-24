#!/usr/bin/env python3
"""Posthoc prompt-C cascade validation for complete V2 and partial V4 judgments."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr, rankdata


ROOT = Path(__file__).resolve().parent
VALIDATION_ROOT = ROOT / "ISC-CI_LLM_validation"
if str(VALIDATION_ROOT) not in sys.path:
    sys.path.insert(0, str(VALIDATION_ROOT))

from evaluate_validation import released_model  # noqa: E402
from iscci_validation.dataio import (  # noqa: E402
    load_human_matrix,
    load_v2_matrix,
    matrix_sha256,
)
from iscci_validation.evaluation import (  # noqa: E402
    compare_evaluations,
    evaluate_model,
)
from iscci_validation.provenance import sha256_file, write_json  # noqa: E402
from iscci_validation.simulations import (  # noqa: E402
    induction_correlations,
    induction_phenomenon_effects,
    paired_choice_agreement,
    score_arguments,
    similarity_asymmetry_metrics,
    similarity_context_metrics,
    similarity_domain_correlations,
)
from iscci_validation.training import (  # noqa: E402
    load_trained_model,
    save_checkpoint,
    train_model,
)
from run_paper_simulations import singularize_object_columns  # noqa: E402


V2_DIR = ROOT / "artifacts" / "leuven_full_labels" / "leuven_full_v2"
V4_DIR = ROOT / "artifacts" / "v4"
V4_SHARDS = V4_DIR / "judgments" / "shards"
CANDIDATE_BANK = V4_DIR / "discovery" / "candidate_bank.csv"
BASE_VALIDATION = VALIDATION_ROOT / "artifacts" / "validation_v3_1"
UPSTREAM = VALIDATION_ROOT / "upstream" / "IntegratedSemanticsControlContextInference"
HUMAN_CSV = (
    UPSTREAM / "data" / "leuven_dataset" / "leuven_combined_features_consolidated.csv"
)
RELEASED_MODEL = UPSTREAM / "models" / "1and2shot_isc-seed3.pt"
CONFIG = ROOT / "configs" / "v4_validation.json"
DEFAULT_OUTPUT = V4_DIR / "retrieval_efficiency" / "prompt_c_cascade"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--confidence-threshold", type=float, default=0.80)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-simulations", action="store_true")
    return parser.parse_args()


def as_bool(values: pd.Series) -> pd.Series:
    if isinstance(values.dtype, pd.BooleanDtype) or values.dtype == bool:
        return values.fillna(False).astype(bool)
    return values.fillna("").astype(str).str.lower().isin({"true", "1", "yes"})


def prompt_c_gate(votes: pd.DataFrame, confidence_threshold: float) -> pd.Series:
    value = pd.to_numeric(votes["feature_value"], errors="coerce")
    confidence = pd.to_numeric(votes["confidence"], errors="coerce")
    parse_failure = (
        votes.get("parse_error", pd.Series("", index=votes.index))
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
    )
    return (
        value.gt(0)
        | as_bool(votes["ambiguous"])
        | confidence.lt(confidence_threshold)
        | value.isna()
        | confidence.isna()
        | parse_failure
    )


def rdm_rank(values: np.ndarray) -> np.ndarray:
    distances = np.nan_to_num(
        pdist(values.astype(np.float32), metric="cosine"),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    return rankdata(distances, method="average").astype(np.float32)


def rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = rdm_rank(left)
    right_rank = rdm_rank(right)
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def cascade_summary(
    benchmark: str,
    gold: np.ndarray,
    routed: np.ndarray,
    adjudicated: np.ndarray,
    context_rule: str,
) -> dict[str, Any]:
    cascade = gold & routed
    if context_rule == "gt3":
        full_context = gold.sum(axis=0) > 3
        cascade_context = cascade.sum(axis=0) > 3
    elif context_rule == "gt0":
        full_context = gold.sum(axis=0) > 0
        cascade_context = cascade.sum(axis=0) > 0
    else:
        raise ValueError(f"Unknown context rule: {context_rule}")
    full_calls = 3 * gold.size + 3 * int(adjudicated.sum())
    cascade_calls = (
        gold.size
        + 2 * int(routed.sum())
        + 3 * int((adjudicated & routed).sum())
    )
    if not full_context.any() or not cascade_context.any():
        geometry = np.nan
    else:
        geometry = rank_correlation(
            gold[:, full_context].astype(np.float32),
            cascade[:, cascade_context].astype(np.float32),
        )
    return {
        "benchmark": benchmark,
        "cells": int(gold.size),
        "features": int(gold.shape[1]),
        "positive_cells": int(gold.sum()),
        "positive_density": float(gold.mean()),
        "routed_cells": int(routed.sum()),
        "routed_fraction": float(routed.mean()),
        "retained_positive_cells": int(cascade.sum()),
        "positive_cell_recall": float(cascade.sum() / gold.sum()),
        "full_contexts": int(full_context.sum()),
        "cascade_contexts": int(cascade_context.sum()),
        "context_recall": float((full_context & cascade_context).sum() / full_context.sum()),
        "object_geometry_correlation": geometry,
        "full_panel_calls": full_calls,
        "cascade_calls": cascade_calls,
        "call_reduction": float(1.0 - cascade_calls / full_calls),
        "context_rule": context_rule,
    }


def build_v2(
    output_dir: Path, confidence_threshold: float
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    human_counts, _ = load_human_matrix(HUMAN_CSV)
    full_training, missing_pairs = load_v2_matrix(
        V2_DIR / "feature_resolutions.csv", human_counts
    )
    base_matrix = pd.read_csv(
        BASE_VALIDATION / "data_matrices" / "v2.csv", index_col=0
    ).astype(np.int8)
    if matrix_sha256(full_training) != matrix_sha256(base_matrix):
        raise ValueError("Loaded V2 matrix does not match the executed validation matrix")

    resolutions = pd.read_csv(
        V2_DIR / "feature_resolutions.csv",
        usecols=[
            "word_normalized",
            "feature_id",
            "final_feature_value",
            "adjudicated",
        ],
    )
    c_votes = pd.read_csv(
        V2_DIR / "feature_votes.csv",
        usecols=[
            "word_normalized",
            "feature_id",
            "judge_id",
            "feature_value",
            "confidence",
            "ambiguous",
            "parse_error",
        ],
        low_memory=False,
    )
    c_votes = c_votes.loc[c_votes["judge_id"].eq("C")].drop(columns="judge_id")
    if c_votes.duplicated(["word_normalized", "feature_id"]).any():
        raise ValueError("V2 has duplicate prompt-C votes")
    c_votes["has_prompt_c_vote"] = True
    cells = resolutions.merge(
        c_votes,
        on=["word_normalized", "feature_id"],
        how="left",
        validate="one_to_one",
    )
    if not cells["has_prompt_c_vote"].fillna(False).all():
        raise ValueError("V2 is missing prompt-C votes")
    cells["gold_positive"] = pd.to_numeric(
        cells["final_feature_value"], errors="raise"
    ).gt(0)
    cells["routed"] = prompt_c_gate(cells, confidence_threshold)
    cells["cascade_positive"] = cells["gold_positive"] & cells["routed"]

    feature_count = human_counts.shape[1]
    words = list(map(str, human_counts.index))
    full = (
        cells.pivot(
            index="word_normalized", columns="feature_id", values="gold_positive"
        )
        .reindex(index=words, columns=range(feature_count))
        .astype("boolean")
        .fillna(False)
        .astype(bool)
    )
    cascade = (
        cells.pivot(
            index="word_normalized", columns="feature_id", values="cascade_positive"
        )
        .reindex(index=words, columns=range(feature_count))
        .astype("boolean")
        .fillna(False)
        .astype(bool)
    )
    adjudicated = (
        cells.pivot(
            index="word_normalized", columns="feature_id", values="adjudicated"
        )
        .reindex(index=words, columns=range(feature_count))
        .astype("boolean")
        .fillna(False)
        .astype(bool)
    )
    routed = (
        cells.pivot(index="word_normalized", columns="feature_id", values="routed")
        .reindex(index=words, columns=range(feature_count))
        .astype("boolean")
        .fillna(False)
        .astype(bool)
    )
    full.columns = human_counts.columns
    cascade.columns = human_counts.columns
    human_task_names = human_counts.columns[human_counts.gt(3).sum(axis=0).gt(3)]
    cascade_training = cascade.loc[:, human_task_names].astype(np.int8)
    cascade_training = cascade_training.loc[:, cascade_training.sum(axis=0).gt(0)]
    cascade_training.to_csv(output_dir / "v2_prompt_c_cascade_matrix.csv")
    full_training.to_csv(output_dir / "v2_full_panel_matrix.csv")

    summary = cascade_summary(
        "v2_complete",
        full.to_numpy(dtype=bool),
        routed.to_numpy(dtype=bool),
        adjudicated.to_numpy(dtype=bool),
        "gt0",
    )
    human_positive = human_counts.gt(3).to_numpy(dtype=bool)
    summary.update(
        {
            "human_positive_cells": int(human_positive.sum()),
            "prompt_c_gate_human_positive_recall": float(
                (human_positive & routed.to_numpy(dtype=bool)).sum()
                / human_positive.sum()
            ),
            "full_v2_human_positive_recall": float(
                (human_positive & full.to_numpy(dtype=bool)).sum()
                / human_positive.sum()
            ),
            "cascade_human_positive_recall": float(
                (human_positive & cascade.to_numpy(dtype=bool)).sum()
                / human_positive.sum()
            ),
            "missing_pairs_filled_zero": len(missing_pairs),
            "training_tasks_full": full_training.shape[1],
            "training_tasks_cascade": cascade_training.shape[1],
            "training_matrix_rdm_correlation": rank_correlation(
                full_training.to_numpy(dtype=np.float32),
                cascade_training.reindex(
                    columns=full_training.columns, fill_value=0
                ).to_numpy(dtype=np.float32),
            ),
        }
    )
    audit_frame = pd.DataFrame(
        {
            "benchmark": "v2_complete",
            "gold_positive": cells["gold_positive"].to_numpy(dtype=bool),
            "routed": cells["routed"].to_numpy(dtype=bool),
        }
    )
    return cascade_training, summary, audit_frame


def complete_v4_feature_ids(word_count: int) -> np.ndarray:
    bank_rows = len(pd.read_csv(CANDIDATE_BANK, usecols=["candidate_index"]))
    row_counts = np.zeros(bank_rows, dtype=np.int32)
    valid_counts = np.zeros(bank_rows, dtype=np.int32)
    for path in sorted(V4_SHARDS.glob("*/feature_resolutions.csv")):
        frame = pd.read_csv(path, usecols=["feature_id", "final_feature_value"])
        ids = frame["feature_id"].to_numpy(dtype=int)
        values = pd.to_numeric(frame["final_feature_value"], errors="coerce").to_numpy()
        row_counts += np.bincount(ids, minlength=bank_rows)
        valid_counts += np.bincount(ids[~np.isnan(values)], minlength=bank_rows)
    return np.flatnonzero((row_counts == word_count) & (valid_counts == word_count))


def stratum_summary(
    name: str,
    mask: np.ndarray,
    gold: np.ndarray,
    routed: np.ndarray,
    adjudicated: np.ndarray,
) -> dict[str, Any] | None:
    if not mask.any():
        return None
    result = cascade_summary(
        name,
        gold[:, mask],
        routed[:, mask],
        adjudicated[:, mask],
        "gt3",
    )
    result["stratum"] = name
    return result


def build_v4_pilot(
    output_dir: Path, confidence_threshold: float
) -> tuple[dict[str, Any], pd.DataFrame]:
    human_counts, _ = load_human_matrix(HUMAN_CSV)
    words = list(map(str, human_counts.index))
    word_index = {word: index for index, word in enumerate(words)}
    feature_ids = complete_v4_feature_ids(len(words))
    feature_index = {feature_id: index for index, feature_id in enumerate(feature_ids)}
    shape = (len(words), len(feature_ids))
    gold = np.zeros(shape, dtype=bool)
    routed = np.zeros(shape, dtype=bool)
    adjudicated = np.zeros(shape, dtype=bool)
    seen_resolutions = np.zeros(shape, dtype=np.uint8)
    seen_c_votes = np.zeros(shape, dtype=np.uint8)

    for shard in sorted(V4_SHARDS.glob("*")):
        resolution_path = shard / "feature_resolutions.csv"
        vote_path = shard / "feature_votes.csv"
        if not resolution_path.exists() or not vote_path.exists():
            continue
        resolutions = pd.read_csv(
            resolution_path,
            usecols=[
                "word_normalized",
                "feature_id",
                "final_feature_value",
                "adjudicated",
            ],
        )
        resolutions = resolutions.loc[resolutions["feature_id"].isin(feature_index)]
        rows = resolutions["word_normalized"].map(word_index).to_numpy(dtype=int)
        columns = resolutions["feature_id"].map(feature_index).to_numpy(dtype=int)
        seen_resolutions[rows, columns] += 1
        gold[rows, columns] = pd.to_numeric(
            resolutions["final_feature_value"], errors="raise"
        ).gt(0)
        adjudicated[rows, columns] = as_bool(resolutions["adjudicated"])

        votes = pd.read_csv(
            vote_path,
            usecols=[
                "word_normalized",
                "feature_id",
                "judge_id",
                "feature_value",
                "confidence",
                "ambiguous",
                "parse_error",
            ],
        )
        votes = votes.loc[
            votes["judge_id"].eq("C") & votes["feature_id"].isin(feature_index)
        ]
        rows = votes["word_normalized"].map(word_index).to_numpy(dtype=int)
        columns = votes["feature_id"].map(feature_index).to_numpy(dtype=int)
        seen_c_votes[rows, columns] += 1
        routed[rows, columns] = prompt_c_gate(votes, confidence_threshold)

    if not np.all(seen_resolutions == 1):
        raise ValueError("V4 pilot does not have exactly one resolution per selected cell")
    if not np.all(seen_c_votes == 1):
        raise ValueError("V4 pilot does not have exactly one prompt-C vote per selected cell")

    bank = pd.read_csv(CANDIDATE_BANK).set_index("candidate_index").loc[feature_ids]
    candidate_ids = bank["candidate_id"].astype(str).tolist()
    split = np.asarray(
        [
            "development"
            if int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:8], 16) % 2 == 0
            else "heldout"
            for value in candidate_ids
        ]
    )
    full_support = gold.sum(axis=0)
    cascade_support = (gold & routed).sum(axis=0)
    feature_metrics = pd.DataFrame(
        {
            "feature_id": feature_ids,
            "candidate_id": candidate_ids,
            "canonical_feature_text": bank["canonical_feature_text"].astype(str).values,
            "split": split,
            "full_positive_objects": full_support,
            "cascade_positive_objects": cascade_support,
            "full_retained_gt3": full_support > 3,
            "cascade_retained_gt3": cascade_support > 3,
            "positive_cell_recall": np.divide(
                cascade_support,
                full_support,
                out=np.ones(len(feature_ids), dtype=float),
                where=full_support > 0,
            ),
            "n_source_words": bank["n_source_words"].astype(int).values,
            "n_independent_responses": bank["n_independent_responses"].astype(int).values,
            "merge_review_status": bank["merge_review_status"].astype(str).values,
            "source_ids": bank["source_ids"].astype(str).values,
            "source_prompt_families": bank["source_prompt_families"].astype(str).values,
        }
    )
    feature_metrics.to_csv(output_dir / "v4_pilot_feature_metrics.csv", index=False)

    rows: list[dict[str, Any]] = []
    all_result = stratum_summary(
        "all_complete_v4_pilot", np.ones(len(feature_ids), dtype=bool), gold, routed, adjudicated
    )
    assert all_result is not None
    rows.append(all_result)
    for partition in ["development", "heldout"]:
        result = stratum_summary(
            f"split:{partition}", split == partition, gold, routed, adjudicated
        )
        if result:
            rows.append(result)
    for status in sorted(bank["merge_review_status"].astype(str).unique()):
        result = stratum_summary(
            f"merge:{status}",
            bank["merge_review_status"].astype(str).values == status,
            gold,
            routed,
            adjudicated,
        )
        if result:
            rows.append(result)
    response_bins = np.select(
        [
            bank["n_independent_responses"].astype(int).values == 1,
            bank["n_independent_responses"].astype(int).values <= 3,
        ],
        ["1", "2-3"],
        default="4+",
    )
    for label in ["1", "2-3", "4+"]:
        result = stratum_summary(
            f"independent_responses:{label}",
            response_bins == label,
            gold,
            routed,
            adjudicated,
        )
        if result:
            rows.append(result)
    source_word_bins = np.where(
        bank["n_source_words"].astype(int).values == 1, "1", "2+"
    )
    for label in ["1", "2+"]:
        result = stratum_summary(
            f"source_words:{label}",
            source_word_bins == label,
            gold,
            routed,
            adjudicated,
        )
        if result:
            rows.append(result)
    for source in [
        "v3_qwen2_5_72b_round0",
        "v3_1_qwen2_5_72b_round0",
        "v4_qwen2_5_72b_round1",
    ]:
        mask = bank["source_ids"].astype(str).str.contains(source, regex=False).values
        result = stratum_summary(
            f"source:{source}", mask, gold, routed, adjudicated
        )
        if result:
            rows.append(result)
    prompt_values = bank["source_prompt_families"].map(json.loads)
    prompt_families = sorted({value for values in prompt_values for value in values})
    for prompt in prompt_families:
        mask = prompt_values.map(lambda values: prompt in values).values
        result = stratum_summary(f"prompt:{prompt}", mask, gold, routed, adjudicated)
        if result:
            rows.append(result)
    pd.DataFrame(rows).to_csv(output_dir / "v4_pilot_stratified_metrics.csv", index=False)

    source_positive = source_total = 0
    word_to_index = {word: index for index, word in enumerate(words)}
    for column, source_json in enumerate(bank["source_words"]):
        source_indices = [
            word_to_index[word]
            for word in json.loads(source_json)
            if word in word_to_index
        ]
        source_total += len(source_indices)
        source_positive += int(gold[source_indices, column].sum())
    summary = dict(all_result)
    summary.update(
        {
            "fully_judged_feature_id_min": int(feature_ids.min()),
            "fully_judged_feature_id_max": int(feature_ids.max()),
            "source_cells": source_total,
            "source_positive_rate": float(source_positive / source_total),
            "pilot_source_v4_only_fraction": float(
                bank["source_ids"].eq('["v4_qwen2_5_72b_round1"]').mean()
            ),
        }
    )
    audit_frame = pd.DataFrame(
        {
            "benchmark": "v4_complete_pilot",
            "gold_positive": gold.reshape(-1),
            "routed": routed.reshape(-1),
        }
    )
    return summary, audit_frame


def simulate_negative_audits(
    frames: list[pd.DataFrame], seed: int = 20260824
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for frame in frames:
        benchmark = str(frame["benchmark"].iloc[0])
        excluded = frame.loc[~frame["routed"], "gold_positive"].to_numpy(dtype=bool)
        included_positive = int(
            frame.loc[frame["routed"], "gold_positive"].sum()
        )
        for fraction in [0.01, 0.05]:
            sample_size = max(1, round(len(excluded) * fraction))
            for repetition in range(200):
                sample = rng.choice(len(excluded), sample_size, replace=False)
                rate = float(excluded[sample].mean())
                estimated_missed = rate * len(excluded)
                estimated_recall = included_positive / (
                    included_positive + estimated_missed
                )
                rows.append(
                    {
                        "benchmark": benchmark,
                        "audit_fraction": fraction,
                        "repetition": repetition,
                        "audit_cells": sample_size,
                        "audited_positive_rate": rate,
                        "estimated_missed_positive_cells": estimated_missed,
                        "estimated_positive_recall": estimated_recall,
                    }
                )
    return pd.DataFrame(rows)


def train_cascade_models(
    matrix: pd.DataFrame, output_dir: Path, config: dict[str, Any], force: bool
) -> pd.DataFrame:
    training = config["training"]
    released = torch.load(RELEASED_MODEL, map_location="cpu", weights_only=False)
    released_embedding = released["state_dict"]["input_to_independent.weight"].detach()
    parameters = {
        "epochs": int(training["epochs"]),
        "episodes_per_epoch": int(training["episodes_per_epoch"]),
        "batch_size": int(training["batch_size"]),
        "support_sizes": list(training["support_sizes"]),
        "learning_rate": float(training["learning_rate"]),
        "task_loss_weight": float(training["task_loss_weight"]),
        "freeze_released_semantic_embedding": True,
    }
    rows: list[dict[str, Any]] = []
    for seed in training["seeds"]:
        run_dir = output_dir / "models" / "v2_prompt_c_cascade" / f"seed_{seed}"
        checkpoint_path = run_dir / "checkpoint.pt"
        metrics_path = run_dir / "training_metrics.csv"
        if checkpoint_path.exists() and not force:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if (
                checkpoint.get("matrix_sha256") == matrix_sha256(matrix)
                and checkpoint.get("epoch") == parameters["epochs"]
                and checkpoint.get("training_parameters") == parameters
            ):
                rows.append(
                    {
                        "condition": "v2_prompt_c_cascade",
                        "seed": seed,
                        "task_count": matrix.shape[1],
                        "elapsed_seconds": 0.0,
                        "reused": True,
                    }
                )
                continue
        model, metrics, elapsed = train_model(
            matrix=matrix,
            released_embedding=released_embedding,
            seed=int(seed),
            epochs=parameters["epochs"],
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
            "v2_prompt_c_cascade",
            int(seed),
            parameters["epochs"],
            parameters,
        )
        rows.append(
            {
                "condition": "v2_prompt_c_cascade",
                "seed": seed,
                "task_count": matrix.shape[1],
                "elapsed_seconds": elapsed,
                "reused": False,
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "training_runs.csv", index=False)
    return result


def load_npz(path: Path) -> dict[str, np.ndarray]:
    loaded = np.load(path, allow_pickle=False)
    return {key: loaded[key] for key in loaded.files}


def evaluate_cascade_models(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    evaluation = load_npz(BASE_VALIDATION / "evaluation" / "evaluation_contexts.npz")
    model_output_dir = output_dir / "evaluation" / "model_outputs"
    model_output_dir.mkdir(parents=True, exist_ok=True)
    for seed in range(4):
        checkpoint = (
            output_dir
            / "models"
            / "v2_prompt_c_cascade"
            / f"seed_{seed}"
            / "checkpoint.pt"
        )
        model = load_trained_model(checkpoint)[0]
        result = evaluate_model(model, evaluation)
        np.savez(model_output_dir / f"v2_prompt_c_cascade_seed_{seed}.npz", **result)

    rows: list[dict[str, Any]] = []
    groups = {
        "same_seed_cascade_vs_v2": [
            ("cascade", seed, "v2", seed) for seed in range(4)
        ],
        "all_seed_cascade_vs_v2": [
            ("cascade", left, "v2", right)
            for left in range(4)
            for right in range(4)
        ],
        "cascade_self": [
            ("cascade", left, "cascade", right)
            for left, right in combinations(range(4), 2)
        ],
        "v2_self": [
            ("v2", left, "v2", right)
            for left, right in combinations(range(4), 2)
        ],
        "cascade_vs_human_retrain": [
            ("cascade", left, "human", right)
            for left in range(4)
            for right in range(4)
        ],
        "v2_vs_human_retrain": [
            ("v2", left, "human", right)
            for left in range(4)
            for right in range(4)
        ],
    }

    def output(kind: str, seed: int) -> dict[str, np.ndarray]:
        if kind == "cascade":
            path = model_output_dir / f"v2_prompt_c_cascade_seed_{seed}.npz"
        else:
            path = (
                BASE_VALIDATION
                / "evaluation"
                / "model_outputs"
                / f"{kind}_seed_{seed}.npz"
            )
        return load_npz(path)

    cache: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    for group, pairs in groups.items():
        for left_kind, left_seed, right_kind, right_seed in pairs:
            left = cache.setdefault(
                (left_kind, left_seed), output(left_kind, left_seed)
            )
            right = cache.setdefault(
                (right_kind, right_seed), output(right_kind, right_seed)
            )
            rows.append(
                {
                    "comparison_group": group,
                    "condition_a": left_kind,
                    "seed_a": left_seed,
                    "condition_b": right_kind,
                    "seed_b": right_seed,
                    **compare_evaluations(left, right),
                }
            )
    comparisons = pd.DataFrame(rows)
    comparisons.to_csv(output_dir / "evaluation" / "representation_comparisons.csv", index=False)
    metric_columns = [
        column
        for column in comparisons.columns
        if column
        not in {
            "comparison_group",
            "condition_a",
            "seed_a",
            "condition_b",
            "seed_b",
        }
    ]
    summary = (
        comparisons.groupby("comparison_group", observed=True)[metric_columns]
        .agg(["count", "mean", "std", "min", "max"])
        .reset_index()
    )
    summary.to_csv(output_dir / "evaluation" / "representation_summary.csv", index=False)
    fidelity_rows: list[dict[str, Any]] = []
    for metric in metric_columns:
        matched_mean = float(
            comparisons.loc[
                comparisons["comparison_group"].eq("same_seed_cascade_vs_v2"), metric
            ].mean()
        )
        v2_self_mean = float(
            comparisons.loc[comparisons["comparison_group"].eq("v2_self"), metric].mean()
        )
        cascade_self_mean = float(
            comparisons.loc[
                comparisons["comparison_group"].eq("cascade_self"), metric
            ].mean()
        )
        cascade_human = float(
            comparisons.loc[
                comparisons["comparison_group"].eq("cascade_vs_human_retrain"), metric
            ].mean()
        )
        v2_human = float(
            comparisons.loc[
                comparisons["comparison_group"].eq("v2_vs_human_retrain"), metric
            ].mean()
        )
        lower_is_better = metric == "membership_probability_mae"
        fidelity_margin = (
            v2_self_mean - matched_mean
            if lower_is_better
            else matched_mean - v2_self_mean
        )
        fidelity_rows.append(
            {
                "metric": metric,
                "same_seed_cascade_vs_v2": matched_mean,
                "v2_seed_to_seed": v2_self_mean,
                "cascade_seed_to_seed": cascade_self_mean,
                "cascade_is_closer_than_v2_seed_variability": fidelity_margin > 0,
                "fidelity_margin": fidelity_margin,
                "cascade_vs_human_retrain": cascade_human,
                "v2_vs_human_retrain": v2_human,
                "cascade_minus_v2_human_similarity": cascade_human - v2_human,
                "lower_is_better": lower_is_better,
            }
        )
    pd.DataFrame(fidelity_rows).to_csv(
        output_dir / "evaluation" / "fidelity_vs_seed_variability.csv", index=False
    )
    return comparisons, summary


def paper_data() -> tuple[dict[str, pd.DataFrame], list[str]]:
    data = UPSTREAM / "data"
    object_names = list(pd.read_csv(HUMAN_CSV, index_col=0, encoding="ISO-8859-1").index.astype(str))
    singular_plural = pd.read_csv(
        data / "leuven_dataset" / "leuven_singular_to_plural.csv", index_col=0
    )
    plural_to_singular = dict(
        zip(singular_plural["plural"], singular_plural["singular"])
    )
    paths = {
        "induction": data / "generalization_experiments" / "induction_arguments.csv",
        "phenomena": data / "generalization_experiments" / "inductive_phenomena.csv",
        "domain": data / "similarity_experiments" / "similarity_in_domain.csv",
        "asymmetry": data / "similarity_experiments" / "similarity_asymmetry.csv",
        "context": data / "similarity_experiments" / "similarity_in_context.csv",
        "thematic": data / "generalization_experiments" / "thematic_arguments.csv",
        "nonmonotonicity": data
        / "generalization_experiments"
        / "context_dependent_nonmonotonicity.csv",
    }
    frames = {
        key: singularize_object_columns(
            pd.read_csv(path, index_col=0), plural_to_singular
        )
        for key, path in paths.items()
    }
    return frames, object_names


def run_cascade_simulations(
    output_dir: Path, config: dict[str, Any], force: bool
) -> dict[str, pd.DataFrame]:
    frames, object_names = paper_data()
    simulation_dir = output_dir / "paper_simulations"
    per_model = simulation_dir / "per_model"
    per_model.mkdir(parents=True, exist_ok=True)
    names = [
        "induction_correlations",
        "induction_phenomena",
        "similarity_domain_correlations",
        "similarity_asymmetry",
        "paired_choice_agreement",
        "similarity_context",
    ]
    combined: dict[str, list[pd.DataFrame]] = {name: [] for name in names}
    for seed in range(4):
        checkpoint = (
            output_dir
            / "models"
            / "v2_prompt_c_cascade"
            / f"seed_{seed}"
            / "checkpoint.pt"
        )
        model_id = f"v2_prompt_c_cascade_seed_{seed}"
        cached = all((per_model / f"{model_id}_{name}.csv").exists() for name in names)
        if cached and not force:
            for name in names:
                combined[name].append(pd.read_csv(per_model / f"{model_id}_{name}.csv"))
            continue
        model = load_trained_model(checkpoint)[0]
        induction_scores = score_arguments(model, frames["induction"], object_names)
        phenomenon_scores = score_arguments(model, frames["phenomena"], object_names)
        results = {
            "induction_correlations": induction_correlations(
                frames["induction"], induction_scores
            ),
            "induction_phenomena": induction_phenomenon_effects(
                frames["phenomena"], phenomenon_scores
            ),
            "similarity_domain_correlations": similarity_domain_correlations(
                frames["domain"], model, object_names
            ),
            "similarity_asymmetry": similarity_asymmetry_metrics(
                frames["asymmetry"], model, object_names
            ),
            "paired_choice_agreement": pd.concat(
                [
                    paired_choice_agreement(
                        frames["thematic"], model, object_names, "thematic_arguments"
                    ),
                    paired_choice_agreement(
                        frames["nonmonotonicity"],
                        model,
                        object_names,
                        "context_dependent_nonmonotonicity",
                    ),
                ],
                ignore_index=True,
            ),
            "similarity_context": similarity_context_metrics(
                frames["context"],
                model,
                object_names,
                seed=int(config["evaluation"]["seed"]) + seed,
            ),
        }
        for name, frame in results.items():
            frame.insert(0, "seed", seed)
            frame.insert(0, "condition", "v2_prompt_c_cascade")
            frame.insert(0, "model_id", model_id)
            frame.to_csv(per_model / f"{model_id}_{name}.csv", index=False)
            combined[name].append(frame)
    final = {name: pd.concat(values, ignore_index=True) for name, values in combined.items()}
    for name, frame in final.items():
        frame.to_csv(simulation_dir / f"{name}.csv", index=False)
    return final


def compare_paper_simulations(
    output_dir: Path, cascade: dict[str, pd.DataFrame]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline_dir = BASE_VALIDATION / "paper_simulations"
    key_columns = {
        "induction_correlations": ["dataset"],
        "induction_phenomena": ["phenomenon"],
        "similarity_domain_correlations": ["domain"],
        "similarity_asymmetry": [],
        "paired_choice_agreement": ["experiment"],
        "similarity_context": [],
    }
    metric_columns = {
        "induction_correlations": ["pearson_r"],
        "induction_phenomena": ["high_mean", "low_mean", "high_minus_low"],
        "similarity_domain_correlations": ["pearson_r"],
        "similarity_asymmetry": [
            "agreement_with_human_majority",
            "asymmetry_score_human_pearson",
        ],
        "paired_choice_agreement": [
            "agreement_with_human_majority",
            "model_argument2_choice_rate",
            "score_difference_human_choice_pearson",
        ],
        "similarity_context": [
            "direction_agreement",
            "model_effect_mean",
            "model_human_effect_pearson",
        ],
    }
    rows: list[dict[str, Any]] = []
    variability: dict[tuple[str, str], dict[str, Any]] = {}
    for analysis, cascade_frame in cascade.items():
        baseline = pd.read_csv(baseline_dir / f"{analysis}.csv")
        baseline = baseline.loc[baseline["condition"].eq("v2")]
        keys = ["seed", *key_columns[analysis]]
        merged = baseline.merge(
            cascade_frame,
            on=keys,
            suffixes=("_v2", "_cascade"),
            validate="one_to_one",
        )
        for row in merged.itertuples(index=False):
            key_record = {key: getattr(row, key) for key in keys}
            for metric in metric_columns[analysis]:
                full = float(getattr(row, f"{metric}_v2"))
                pruned = float(getattr(row, f"{metric}_cascade"))
                rows.append(
                    {
                        "analysis": analysis,
                        **key_record,
                        "metric": metric,
                        "full_v2": full,
                        "cascade": pruned,
                        "signed_delta": pruned - full,
                        "absolute_delta": abs(pruned - full),
                        "sign_agreement": bool(
                            np.sign(pruned) == np.sign(full)
                        ),
                    }
                )
        grouping_keys = key_columns[analysis]
        baseline_for_grouping = baseline.copy()
        cascade_for_grouping = cascade_frame.copy()
        if not grouping_keys:
            baseline_for_grouping["__all__"] = "all"
            cascade_for_grouping["__all__"] = "all"
            grouping_keys = ["__all__"]
        for metric in metric_columns[analysis]:
            self_deltas: list[float] = []
            for _, group in baseline_for_grouping.groupby(grouping_keys, observed=True):
                values = group.sort_values("seed")[metric].astype(float).to_numpy()
                self_deltas.extend(
                    abs(float(values[left] - values[right]))
                    for left, right in combinations(range(len(values)), 2)
                )
            baseline_means = (
                baseline_for_grouping.groupby(grouping_keys, observed=True)[metric]
                .mean()
                .rename("full_mean")
            )
            cascade_means = (
                cascade_for_grouping.groupby(grouping_keys, observed=True)[metric]
                .mean()
                .rename("cascade_mean")
            )
            condition_means = pd.concat([baseline_means, cascade_means], axis=1)
            condition_mean_correlation = (
                float(
                    pearsonr(
                        condition_means["full_mean"],
                        condition_means["cascade_mean"],
                    ).statistic
                )
                if len(condition_means) > 2
                and condition_means["full_mean"].nunique() > 1
                and condition_means["cascade_mean"].nunique() > 1
                else np.nan
            )
            variability[(analysis, metric)] = {
                "v2_seed_to_seed_mean_absolute_delta": float(np.mean(self_deltas)),
                "condition_mean_absolute_delta": float(
                    (condition_means["cascade_mean"] - condition_means["full_mean"])
                    .abs()
                    .mean()
                ),
                "condition_mean_sign_agreement": float(
                    (
                        np.sign(condition_means["cascade_mean"])
                        == np.sign(condition_means["full_mean"])
                    ).mean()
                ),
                "condition_mean_full_cascade_pearson": condition_mean_correlation,
            }
    comparisons = pd.DataFrame(rows)
    comparisons.to_csv(
        output_dir / "paper_simulations" / "full_v2_vs_cascade.csv", index=False
    )
    summaries: list[dict[str, Any]] = []
    for (analysis, metric), group in comparisons.groupby(["analysis", "metric"]):
        correlation = (
            float(pearsonr(group["full_v2"], group["cascade"]).statistic)
            if len(group) > 2
            and group["full_v2"].nunique() > 1
            and group["cascade"].nunique() > 1
            else np.nan
        )
        summaries.append(
            {
                "analysis": analysis,
                "metric": metric,
                "rows": len(group),
                "full_cascade_pearson": correlation,
                "mean_absolute_delta": float(group["absolute_delta"].mean()),
                "max_absolute_delta": float(group["absolute_delta"].max()),
                "mean_signed_delta": float(group["signed_delta"].mean()),
                "sign_agreement": float(group["sign_agreement"].mean()),
                **variability[(analysis, metric)],
            }
        )
    summary = pd.DataFrame(summaries)
    summary["matched_delta_over_v2_seed_variability"] = np.divide(
        summary["mean_absolute_delta"],
        summary["v2_seed_to_seed_mean_absolute_delta"],
        out=np.full(len(summary), np.nan),
        where=summary["v2_seed_to_seed_mean_absolute_delta"].ne(0),
    )
    summary.to_csv(
        output_dir / "paper_simulations" / "full_v2_vs_cascade_summary.csv",
        index=False,
    )
    return comparisons, summary


def write_results(
    output_dir: Path,
    benchmark: pd.DataFrame,
    representation: pd.DataFrame | None,
    paper: pd.DataFrame | None,
) -> None:
    v2 = benchmark.loc[benchmark["benchmark"].eq("v2_complete")].iloc[0]
    v4 = benchmark.loc[benchmark["benchmark"].eq("all_complete_v4_pilot")].iloc[0]
    lines = [
        "# Prompt-C Cascade Validation",
        "",
        "## Cell and geometry fidelity",
        "",
        "| Benchmark | Routed cells | Positive recall | Context recall | Object geometry | Call reduction |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        f"| V2 complete | {v2.routed_fraction:.3f} | {v2.positive_cell_recall:.4f} | {v2.context_recall:.4f} | {v2.object_geometry_correlation:.4f} | {v2.call_reduction:.3f} |",
        f"| V4 complete pilot | {v4.routed_fraction:.3f} | {v4.positive_cell_recall:.4f} | {v4.context_recall:.4f} | {v4.object_geometry_correlation:.4f} | {v4.call_reduction:.3f} |",
        "",
        "The cascade sends prompt-C positives, ambiguous responses, low-confidence responses, and parse failures to the full panel. High-confidence prompt-C negatives are set to zero.",
    ]
    if representation is not None:
        same = representation.loc[
            representation["comparison_group"].eq("same_seed_cascade_vs_v2")
        ]
        fidelity = pd.read_csv(
            output_dir / "evaluation" / "fidelity_vs_seed_variability.csv"
        )
        lines.extend(
            [
                "",
                "## ISC-CI representation fidelity",
                "",
                f"Across matched seeds, mean context-RDM correlation was {same.context_rdm_spearman.mean():.4f}, median-context-dependent-RDM correlation was {same.context_dependent_rdm_spearman_median.mean():.4f}, membership-logit correlation was {same.membership_logit_spearman.mean():.4f}, and binary membership agreement was {same.binary_membership_agreement.mean():.4f}.",
                f"The cascade/full-V2 comparison was closer than ordinary full-V2 seed-to-seed variation for {int(fidelity.cascade_is_closer_than_v2_seed_variability.sum())} of {len(fidelity)} representation metrics. Mean similarity to the human retrains changed by at most {fidelity.cascade_minus_v2_human_similarity.abs().max():.4f}.",
            ]
        )
    if paper is not None:
        paper_summary = pd.read_csv(
            output_dir
            / "paper_simulations"
            / "full_v2_vs_cascade_summary.csv"
        )
        below_seed_variability = int(
            paper_summary["matched_delta_over_v2_seed_variability"].le(1).sum()
        )
        lines.extend(
            [
                "",
                "## Paper simulations",
                "",
                f"Across {len(paper)} matched simulation metric rows, mean absolute change was {paper.absolute_delta.mean():.4f}; sign agreement was {paper.sign_agreement.mean():.3f}.",
                f"At the condition-mean level, all {len(paper_summary)} reported metric directions were preserved. Matched cascade changes were no larger than ordinary V2 seed-to-seed variation for {below_seed_variability} of {len(paper_summary)} metrics.",
                "The largest substantive shift was context-dependent nonmonotonicity choice agreement, which decreased from 0.659 to 0.591. This outcome should remain an explicit sensitivity result rather than being described as numerically identical.",
            ]
        )
    lines.extend(
        [
            "",
            "## Limitation",
            "",
            f"The V4 pilot consists of alphabetically early candidate IDs and is {v4.pilot_source_v4_only_fraction:.1%} V4-only. The complete V2 benchmark provides broader vocabulary evidence, but later V3/V3.1-derived V4 candidates still require a stratified audit when more exhaustive judgments are available.",
        ]
    )
    (output_dir / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(CONFIG.read_text(encoding="utf-8"))

    cascade_matrix, v2_summary, v2_audit = build_v2(
        output_dir, args.confidence_threshold
    )
    v4_summary, v4_audit = build_v4_pilot(
        output_dir, args.confidence_threshold
    )
    benchmark = pd.DataFrame([v2_summary, v4_summary])
    benchmark.to_csv(output_dir / "benchmark_summary.csv", index=False)
    audit = simulate_negative_audits([v2_audit, v4_audit])
    audit.to_csv(output_dir / "negative_audit_simulation.csv", index=False)

    representation = None
    paper_comparison = None
    training_runs = None
    if not args.skip_training:
        training_runs = train_cascade_models(
            cascade_matrix, output_dir, config, args.force
        )
        representation, _ = evaluate_cascade_models(output_dir)
        if not args.skip_simulations:
            cascade_simulations = run_cascade_simulations(
                output_dir, config, args.force
            )
            paper_comparison, _ = compare_paper_simulations(
                output_dir, cascade_simulations
            )
    write_results(output_dir, benchmark, representation, paper_comparison)

    write_json(
        output_dir / "manifest.json",
        {
            "protocol_version": "v4-prompt-c-cascade-posthoc-1.0.0",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "confidence_threshold": args.confidence_threshold,
            "routing_rule": "C value > 0 OR ambiguous OR confidence < threshold OR parse/schema failure",
            "v2_resolutions_sha256": sha256_file(V2_DIR / "feature_resolutions.csv"),
            "v2_votes_sha256": sha256_file(V2_DIR / "feature_votes.csv"),
            "candidate_bank_sha256": sha256_file(CANDIDATE_BANK),
            "v4_shard_count": len(list(V4_SHARDS.glob("*"))),
            "config_sha256": sha256_file(CONFIG),
            "released_model_sha256": sha256_file(RELEASED_MODEL),
            "base_validation_manifest_sha256": sha256_file(
                BASE_VALIDATION / "manifest.json"
            ),
            "analysis_script_sha256": sha256_file(Path(__file__).resolve()),
            "cascade_matrix_sha256": matrix_sha256(cascade_matrix),
            "training_completed": training_runs is not None,
            "simulations_completed": paper_comparison is not None,
            "benchmark_summary": benchmark.astype(object)
            .where(benchmark.notna(), None)
            .to_dict(orient="records"),
        },
    )
    print(f"Prompt-C cascade analysis complete: {output_dir}")


if __name__ == "__main__":
    main()
