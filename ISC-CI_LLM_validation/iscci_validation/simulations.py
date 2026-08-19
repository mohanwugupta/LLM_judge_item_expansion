from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from scipy.special import expit, softmax
from scipy.stats import binom, pearsonr

from .modeling import CICOModel


def _name_to_index(object_names: Sequence[str]) -> dict[str, int]:
    return {str(name): index for index, name in enumerate(object_names)}


def score_arguments(
    model: CICOModel,
    arguments: pd.DataFrame,
    object_names: Sequence[str],
    batch_size: int = 512,
) -> np.ndarray:
    lookup = _name_to_index(object_names)
    padding = len(object_names)
    scores = np.empty(len(arguments), dtype=np.float32)
    groups: dict[int, list[int]] = {1: [], 2: [], 3: []}
    supports: list[list[int]] = []
    conclusions: list[int] = []
    for row_index, row in arguments.reset_index(drop=True).iterrows():
        premise_names = [
            row.get(column)
            for column in ("Premise 1", "Premise 2", "Premise 3")
            if pd.notna(row.get(column)) and str(row.get(column)).strip()
        ]
        if not premise_names or len(premise_names) > 3:
            raise ValueError(f"Unsupported premises at row {row_index}: {premise_names}")
        try:
            premise_indices = [lookup[str(name)] for name in premise_names]
            conclusion_index = lookup[str(row["Conclusion"])]
        except KeyError as error:
            raise ValueError(f"Unknown Leuven object at row {row_index}: {error}") from error
        if len(premise_indices) == 1:
            premise_indices.append(padding)
        supports.append(premise_indices)
        conclusions.append(conclusion_index)
        groups[len(premise_names)].append(row_index)

    model.eval()
    with torch.no_grad():
        for group_indices in groups.values():
            for start in range(0, len(group_indices), batch_size):
                selected = group_indices[start : start + batch_size]
                support = torch.tensor(
                    [supports[index] for index in selected], dtype=torch.long
                )
                conclusion = torch.tensor(
                    [[conclusions[index]] for index in selected], dtype=torch.long
                )
                context = model.get_context_rep(support)
                output = model.get_ft_output(conclusion, context).squeeze(-1).squeeze(-1)
                scores[selected] = output.cpu().numpy()
    return scores


def induction_correlations(
    data: pd.DataFrame, scores: np.ndarray
) -> pd.DataFrame:
    working = data.copy()
    working["ISC-CI"] = scores
    rows = []
    for dataset, group in working.groupby("Dataset", observed=True):
        finite = np.isfinite(group["Human"]) & np.isfinite(group["ISC-CI"])
        correlation, p_value = pearsonr(
            group.loc[finite, "Human"], group.loc[finite, "ISC-CI"]
        )
        rows.append(
            {
                "dataset": dataset,
                "n": int(finite.sum()),
                "pearson_r": correlation,
                "p_value": p_value,
            }
        )
    return pd.DataFrame(rows)


def induction_phenomenon_effects(
    data: pd.DataFrame, scores: np.ndarray
) -> pd.DataFrame:
    working = data.copy()
    working["ISC-CI"] = expit(scores)
    grouped = (
        working.groupby(["Phenomenon", "Argument Group"], observed=True)["ISC-CI"]
        .agg(["count", "mean", "sem"])
        .reset_index()
    )
    rows = []
    for phenomenon, group in grouped.groupby("Phenomenon", observed=True):
        by_argument = group.set_index("Argument Group")
        high = float(by_argument.loc["High", "mean"])
        low = float(by_argument.loc["Low", "mean"])
        rows.append(
            {
                "phenomenon": phenomenon,
                "high_mean": high,
                "low_mean": low,
                "high_minus_low": high - low,
                "predicted_direction": "expected" if high > low else "opposite",
                "n_high": int(by_argument.loc["High", "count"]),
                "n_low": int(by_argument.loc["Low", "count"]),
            }
        )
    return pd.DataFrame(rows)


def similarity_domain_correlations(
    data: pd.DataFrame,
    model: CICOModel,
    object_names: Sequence[str],
) -> pd.DataFrame:
    averaged = (
        data.groupby(["Premise 1", "Conclusion", "Domain"], observed=True)[
            "Normalized Similarity"
        ]
        .mean()
        .reset_index()
    )
    forward = averaged[["Premise 1", "Conclusion"]].copy()
    forward["Premise 2"] = np.nan
    forward["Premise 3"] = np.nan
    reverse = forward.copy()
    reverse["Premise 1"] = forward["Conclusion"]
    reverse["Conclusion"] = forward["Premise 1"]
    scores = (
        expit(score_arguments(model, forward, object_names))
        + expit(score_arguments(model, reverse, object_names))
    ) / 2
    averaged["ISC-CI"] = scores
    rows = []
    for domain, group in averaged.groupby("Domain", observed=True):
        correlation, p_value = pearsonr(
            group["Normalized Similarity"], group["ISC-CI"]
        )
        rows.append(
            {
                "domain": domain,
                "n": len(group),
                "pearson_r": correlation,
                "p_value": p_value,
            }
        )
    return pd.DataFrame(rows)


def _paired_arguments(
    grouped: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    first = grouped[
        ["Argument 1-Premise 1", "Argument 1-Premise 2", "Argument 1-Conclusion"]
    ].copy()
    second = grouped[
        ["Argument 2-Premise 1", "Argument 2-Premise 2", "Argument 2-Conclusion"]
    ].copy()
    first.columns = ["Premise 1", "Premise 2", "Conclusion"]
    second.columns = ["Premise 1", "Premise 2", "Conclusion"]
    first["Premise 3"] = np.nan
    second["Premise 3"] = np.nan
    return first, second


def paired_choice_agreement(
    data: pd.DataFrame,
    model: CICOModel,
    object_names: Sequence[str],
    label: str,
) -> pd.DataFrame:
    argument_columns = [
        "Argument 1-Premise 1",
        "Argument 1-Premise 2",
        "Argument 1-Conclusion",
        "Argument 2-Premise 1",
        "Argument 2-Premise 2",
        "Argument 2-Conclusion",
    ]
    grouped = (
        data.groupby(argument_columns, dropna=False, observed=True)["Argument 2 Chosen"]
        .agg(["mean", "count"])
        .reset_index()
    )
    first, second = _paired_arguments(grouped)
    first_scores = score_arguments(model, first, object_names)
    second_scores = score_arguments(model, second, object_names)
    human_majority = grouped["mean"].values > 0.5
    model_choice = second_scores > first_scores
    return pd.DataFrame(
        [
            {
                "experiment": label,
                "argument_pairs": len(grouped),
                "agreement_with_human_majority": float(
                    np.mean(model_choice == human_majority)
                ),
                "model_argument2_choice_rate": float(np.mean(model_choice)),
                "human_argument2_choice_rate": float(grouped["mean"].mean()),
                "score_difference_human_choice_pearson": float(
                    pearsonr(second_scores - first_scores, grouped["mean"].values)[0]
                ),
            }
        ]
    )


def similarity_asymmetry_metrics(
    data: pd.DataFrame,
    model: CICOModel,
    object_names: Sequence[str],
) -> pd.DataFrame:
    argument_columns = [
        "Argument 1-Premise 1",
        "Argument 1-Premise 2",
        "Argument 1-Conclusion",
        "Argument 2-Premise 1",
        "Argument 2-Premise 2",
        "Argument 2-Conclusion",
    ]
    grouped = (
        data.groupby(argument_columns, dropna=False, observed=True)["Argument 2 Chosen"]
        .agg(["mean", "count"])
        .reset_index()
    )
    grouped["effect_size"] = np.maximum(grouped["mean"], 1 - grouped["mean"])
    grouped["p_value"] = binom.sf(
        grouped["effect_size"] * grouped["count"] - 1,
        grouped["count"],
        0.5,
    )
    significant = grouped[grouped["p_value"] <= 0.05].copy()
    first, second = _paired_arguments(significant)
    first_scores = score_arguments(model, first, object_names)
    second_scores = score_arguments(model, second, object_names)
    human_majority = significant["mean"].values > 0.5
    model_choice = second_scores > first_scores
    return pd.DataFrame(
        [
            {
                "significant_human_pairs": len(significant),
                "agreement_with_human_majority": float(
                    np.mean(model_choice == human_majority)
                ),
                "asymmetry_score_human_pearson": float(
                    pearsonr(
                        second_scores - first_scores,
                        significant["mean"].values - 0.5,
                    )[0]
                ),
            }
        ]
    )


def human_similarity_context_effects(data: pd.DataFrame) -> pd.DataFrame:
    index_columns = [
        "Premise 1",
        "Conclusion 1",
        "Conclusion 2",
        "Distractor 1",
        "Distractor 2",
    ]
    grouped = (
        data.groupby(["Participant Group", *index_columns], observed=True)[
            ["Conclusion 1 Chosen", "Conclusion 2 Chosen"]
        ]
        .mean()
        .reset_index()
    )
    pivoted = grouped.pivot(
        index=index_columns,
        columns="Participant Group",
        values=["Conclusion 1 Chosen", "Conclusion 2 Chosen"],
    )
    pivoted.columns = [
        f"{measure}-{group}" for measure, group in pivoted.columns.to_flat_index()
    ]
    pivoted = pivoted.reset_index()
    pivoted["Conclusion 1 Effect"] = (
        pivoted["Conclusion 1 Chosen-Distractor 1"]
        - pivoted["Conclusion 1 Chosen-Distractor 2"]
    )
    pivoted["Conclusion 2 Effect"] = (
        pivoted["Conclusion 2 Chosen-Distractor 2"]
        - pivoted["Conclusion 2 Chosen-Distractor 1"]
    )
    pivoted["Context Effect"] = (
        pivoted["Conclusion 1 Effect"] + pivoted["Conclusion 2 Effect"]
    )
    return pivoted


def _lca_choice_probabilities(
    drift_rates_by_context: np.ndarray,
    rng: np.random.Generator,
    simulations: int = 100,
    steps: int = 500,
    burn_in: int = 100,
) -> np.ndarray:
    sigma = 0.2
    beta = 0.6
    lambda_ = 0.94
    activations = np.zeros((simulations, 3), dtype=np.float64)
    choice_counts = np.zeros(3, dtype=np.int64)
    for step in range(steps):
        contexts = rng.integers(0, len(drift_rates_by_context), size=simulations)
        drift = drift_rates_by_context[contexts]
        inputs = np.maximum(0.0, 2 * drift - drift.sum(axis=1, keepdims=True))
        noise = rng.normal(0.0, sigma, size=(simulations, 3))
        first = np.maximum(
            0.0,
            lambda_ * activations[:, 0]
            + (1 - lambda_)
            * (
                inputs[:, 0]
                - beta * (activations[:, 1] + activations[:, 2])
                + noise[:, 0]
            ),
        )
        second = np.maximum(
            0.0,
            lambda_ * activations[:, 1]
            + (1 - lambda_)
            * (
                inputs[:, 1]
                - beta * (first + activations[:, 2])
                + noise[:, 1]
            ),
        )
        third = np.maximum(
            0.0,
            lambda_ * activations[:, 2]
            + (1 - lambda_)
            * (inputs[:, 2] - beta * (first + second) + noise[:, 2]),
        )
        activations = np.column_stack([first, second, third])
        if step >= burn_in:
            choice_counts += np.bincount(activations.argmax(axis=1), minlength=3)
    return choice_counts / choice_counts.sum()


def _stable_seed(base_seed: int, *values: str) -> int:
    payload = "\0".join([str(base_seed), *map(str, values)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def model_similarity_context_effect(
    row: pd.Series,
    model: CICOModel,
    object_names: Sequence[str],
    seed: int,
) -> float:
    lookup = _name_to_index(object_names)
    premise = lookup[str(row["Premise 1"])]
    options_base = [
        lookup[str(row["Conclusion 1"])],
        lookup[str(row["Conclusion 2"])],
    ]
    output_values: dict[str, float] = {}
    model.eval()
    with torch.no_grad():
        for distractor_name in ("Distractor 1", "Distractor 2"):
            options = options_base + [lookup[str(row[distractor_name])]]
            option_tensor = torch.tensor(options, dtype=torch.long).unsqueeze(1)
            drift_rows = []
            for option in options:
                support = torch.tensor([[premise, option]], dtype=torch.long)
                context = model.get_context_rep(support).repeat(3, 1)
                logits = model.get_ft_output(option_tensor, context).squeeze().numpy()
                drift_rows.append(softmax(0.5 * logits))
            rng = np.random.default_rng(
                _stable_seed(seed, str(row.name), distractor_name)
            )
            choices = _lca_choice_probabilities(np.stack(drift_rows), rng)
            output_values[f"c1-{distractor_name}"] = float(choices[0])
            output_values[f"c2-{distractor_name}"] = float(choices[1])
    return (
        output_values["c1-Distractor 1"]
        - output_values["c1-Distractor 2"]
        + output_values["c2-Distractor 2"]
        - output_values["c2-Distractor 1"]
    )


def similarity_context_metrics(
    data: pd.DataFrame,
    model: CICOModel,
    object_names: Sequence[str],
    seed: int,
) -> pd.DataFrame:
    human_effects = human_similarity_context_effects(data)
    selected = human_effects[human_effects["Context Effect"] > 0].copy()
    selected["model_effect"] = selected.apply(
        model_similarity_context_effect,
        model=model,
        object_names=object_names,
        seed=seed,
        axis=1,
    )
    return pd.DataFrame(
        [
            {
                "positive_human_context_sets": len(selected),
                "direction_agreement": float(np.mean(selected["model_effect"] > 0)),
                "model_effect_mean": float(selected["model_effect"].mean()),
                "model_human_effect_pearson": float(
                    pearsonr(selected["model_effect"], selected["Context Effect"])[0]
                ),
                "lca_simulations": 100,
                "lca_steps": 500,
                "lca_burn_in": 100,
            }
        ]
    )
