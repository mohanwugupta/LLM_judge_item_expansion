from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import pdist
from scipy.special import expit
from scipy.stats import rankdata
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    jaccard_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

from .modeling import CICOModel


METRICS = (
    "context_rdm_spearman",
    "context_dependent_rdm_spearman_median",
    "membership_logit_spearman",
    "binary_membership_agreement",
    "membership_probability_mae",
)
HIGHER_IS_BETTER = {
    "context_rdm_spearman": True,
    "context_dependent_rdm_spearman_median": True,
    "membership_logit_spearman": True,
    "binary_membership_agreement": True,
    "membership_probability_mae": False,
}


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    condition: str
    seed: int
    source: str
    path: Path


def make_evaluation_contexts(
    object_count: int,
    pair_context_count: int,
    rdm_context_count: int,
    context_dependent_rdm_count: int,
    seed: int,
) -> dict[str, np.ndarray]:
    padding_index = object_count
    singletons = np.column_stack(
        [np.arange(object_count, dtype=np.int64), np.full(object_count, padding_index)]
    )
    all_pairs = np.asarray(list(combinations(range(object_count), 2)), dtype=np.int64)
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(all_pairs), size=pair_context_count, replace=False)
    pair_contexts = all_pairs[chosen]
    contexts = np.vstack([singletons, pair_contexts])
    rdm_indices = np.sort(
        rng.choice(len(contexts), size=rdm_context_count, replace=False)
    )
    cd_indices = np.sort(
        rng.choice(len(contexts), size=context_dependent_rdm_count, replace=False)
    )
    return {
        "contexts": contexts,
        "rdm_indices": rdm_indices,
        "context_dependent_rdm_indices": cd_indices,
        "query_objects": np.arange(object_count, dtype=np.int64),
    }


def _ranked_cosine_rdm(values: np.ndarray) -> np.ndarray:
    distances = pdist(values, metric="cosine")
    distances = np.nan_to_num(distances, nan=0.0, posinf=0.0, neginf=0.0)
    return rankdata(distances, method="average").astype(np.float32)


def evaluate_model(
    model: CICOModel,
    evaluation: dict[str, np.ndarray],
    batch_size: int = 32,
) -> dict[str, np.ndarray]:
    model.eval()
    contexts = torch.from_numpy(evaluation["contexts"]).long()
    query_values = torch.from_numpy(evaluation["query_objects"]).long()
    rdm_index_set = set(evaluation["context_dependent_rdm_indices"].tolist())
    context_rows: list[np.ndarray] = []
    logit_rows: list[np.ndarray] = []
    dependent_by_index: dict[int, np.ndarray] = {}
    with torch.no_grad():
        for start in range(0, len(contexts), batch_size):
            stop = min(start + batch_size, len(contexts))
            support = contexts[start:stop]
            context_rep = model.get_context_rep(support)
            query = query_values.unsqueeze(0).repeat(stop - start, 1)
            dependent = model.get_context_dependent_rep(query, context_rep)
            logits = model.dependent_to_ft_output(dependent).squeeze(-1)
            context_rows.append(context_rep.cpu().numpy())
            logit_rows.append(logits.cpu().numpy())
            for local_index, global_index in enumerate(range(start, stop)):
                if global_index in rdm_index_set:
                    dependent_by_index[global_index] = dependent[
                        local_index
                    ].cpu().numpy()

    context_reps = np.concatenate(context_rows, axis=0)
    logits = np.concatenate(logit_rows, axis=0).astype(np.float32)
    context_rdm_ranks = _ranked_cosine_rdm(
        context_reps[evaluation["rdm_indices"]]
    )
    dependent_rdm_ranks = np.stack(
        [
            _ranked_cosine_rdm(dependent_by_index[int(index)])
            for index in evaluation["context_dependent_rdm_indices"]
        ]
    )
    return {
        "context_rdm_ranks": context_rdm_ranks,
        "context_dependent_rdm_ranks": dependent_rdm_ranks,
        "membership_logits": logits,
        "membership_logit_ranks": rankdata(
            logits.reshape(-1), method="average"
        ).astype(np.float32),
    }


def _pearson_from_ranks(left: np.ndarray, right: np.ndarray) -> float:
    left_centered = left.astype(np.float64) - float(np.mean(left))
    right_centered = right.astype(np.float64) - float(np.mean(right))
    denominator = float(
        np.linalg.norm(left_centered) * np.linalg.norm(right_centered)
    )
    return float(left_centered @ right_centered / denominator) if denominator else np.nan


def compare_evaluations(
    left: dict[str, np.ndarray], right: dict[str, np.ndarray]
) -> dict[str, float]:
    context_dependent_correlations = [
        _pearson_from_ranks(left_row, right_row)
        for left_row, right_row in zip(
            left["context_dependent_rdm_ranks"],
            right["context_dependent_rdm_ranks"],
        )
    ]
    left_logits = left["membership_logits"]
    right_logits = right["membership_logits"]
    return {
        "context_rdm_spearman": _pearson_from_ranks(
            left["context_rdm_ranks"], right["context_rdm_ranks"]
        ),
        "context_dependent_rdm_spearman_median": float(
            np.nanmedian(context_dependent_correlations)
        ),
        "membership_logit_spearman": _pearson_from_ranks(
            left["membership_logit_ranks"], right["membership_logit_ranks"]
        ),
        "binary_membership_agreement": float(
            np.mean((left_logits > 0) == (right_logits > 0))
        ),
        "membership_probability_mae": float(
            np.mean(np.abs(expit(left_logits) - expit(right_logits)))
        ),
    }


def comparison_group(left_condition: str, right_condition: str) -> str:
    conditions = {left_condition, right_condition}
    if len(conditions) == 1:
        condition = left_condition
        if condition == "human":
            return "human_retrain_vs_retrain"
        if condition == "released":
            return "released_vs_released"
        return f"{condition}_vs_{condition}"
    if conditions == {"released", "human"}:
        return "released_vs_human_retrain"
    if "human" in conditions:
        candidate = next(value for value in conditions if value != "human")
        return f"{candidate}_vs_human_retrain"
    if "released" in conditions:
        candidate = next(value for value in conditions if value != "released")
        return f"released_vs_{candidate}"
    return "_vs_".join(sorted(conditions))


def pairwise_model_comparisons(
    specs: Iterable[ModelSpec], evaluation_dir: Path
) -> pd.DataFrame:
    specs = list(specs)
    cache: dict[str, dict[str, np.ndarray]] = {}

    def load(model_id: str) -> dict[str, np.ndarray]:
        if model_id not in cache:
            loaded = np.load(evaluation_dir / f"{model_id}.npz", allow_pickle=False)
            cache[model_id] = {key: loaded[key] for key in loaded.files}
        return cache[model_id]

    rows: list[dict[str, Any]] = []
    for left, right in combinations(specs, 2):
        metrics = compare_evaluations(load(left.model_id), load(right.model_id))
        rows.append(
            {
                "model_a": left.model_id,
                "condition_a": left.condition,
                "seed_a": left.seed,
                "model_b": right.model_id,
                "condition_b": right.condition,
                "seed_b": right.seed,
                "comparison_group": comparison_group(
                    left.condition, right.condition
                ),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def summarize_pairwise(pairwise: pd.DataFrame) -> pd.DataFrame:
    return (
        pairwise.groupby("comparison_group", observed=True)[list(METRICS)]
        .agg(["count", "mean", "std", "min", "max"])
        .reset_index()
    )


def ceiling_normalized_summary(pairwise: pd.DataFrame) -> pd.DataFrame:
    ceiling = pairwise[
        pairwise["comparison_group"] == "human_retrain_vs_retrain"
    ]
    ceiling_means = ceiling[list(METRICS)].mean()
    candidate_groups = [
        group
        for group in pairwise["comparison_group"].unique()
        if group.endswith("_vs_human_retrain")
        and group not in {"released_vs_human_retrain"}
    ]
    rows: list[dict[str, Any]] = []
    for group in sorted(candidate_groups):
        values = pairwise[pairwise["comparison_group"] == group]
        for metric in METRICS:
            candidate_mean = float(values[metric].mean())
            ceiling_mean = float(ceiling_means[metric])
            normalized = (
                candidate_mean / ceiling_mean
                if HIGHER_IS_BETTER[metric]
                else ceiling_mean / candidate_mean
            )
            rows.append(
                {
                    "comparison_group": group,
                    "metric": metric,
                    "candidate_mean": candidate_mean,
                    "human_self_ceiling_mean": ceiling_mean,
                    "ceiling_normalized_ratio": normalized,
                    "higher_is_better": HIGHER_IS_BETTER[metric],
                }
            )
    return pd.DataFrame(rows)


def binary_recovery_metrics(human: pd.DataFrame, v2: pd.DataFrame) -> pd.DataFrame:
    shared = human.columns.intersection(v2.columns)
    truth = human.loc[:, shared].values.reshape(-1)
    prediction = v2.loc[:, shared].values.reshape(-1)
    rows = [
        {
            "comparison": "v2_vs_human_shared_382_features",
            "cells": len(truth),
            "positive_truth": int(truth.sum()),
            "positive_prediction": int(prediction.sum()),
            "accuracy": accuracy_score(truth, prediction),
            "balanced_accuracy": balanced_accuracy_score(truth, prediction),
            "precision": precision_score(truth, prediction, zero_division=0),
            "recall": recall_score(truth, prediction, zero_division=0),
            "f1": f1_score(truth, prediction, zero_division=0),
            "jaccard": jaccard_score(truth, prediction, zero_division=0),
            "matthews_correlation": matthews_corrcoef(truth, prediction),
        }
    ]
    return pd.DataFrame(rows)


def matrix_rdm_comparisons(matrices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    ranks: dict[str, np.ndarray] = {}
    for condition, matrix in matrices.items():
        values = matrix.values.astype(np.float32)
        ranks[condition] = _ranked_cosine_rdm(values)
    rows: list[dict[str, Any]] = []
    for left, right in combinations(sorted(matrices), 2):
        rows.append(
            {
                "condition_a": left,
                "condition_b": right,
                "object_rdm_spearman": _pearson_from_ranks(
                    ranks[left], ranks[right]
                ),
            }
        )
    return pd.DataFrame(rows)
