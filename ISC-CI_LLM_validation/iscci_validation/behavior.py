"""External human tests using the released stimuli and ISC-CI probes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import binom, rankdata, ttest_ind, ttest_rel
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold

from .simulations import (
    _paired_arguments,
    human_similarity_context_effects,
    model_similarity_context_predictions,
    score_arguments,
)


TASK_FILES = {
    "induction": "generalization_experiments/induction_arguments.csv",
    "phenomena": "generalization_experiments/inductive_phenomena.csv",
    "thematic": "generalization_experiments/thematic_arguments.csv",
    "nonmonotonicity": "generalization_experiments/context_dependent_nonmonotonicity.csv",
    "similarity": "similarity_experiments/similarity_in_domain.csv",
    "asymmetry": "similarity_experiments/similarity_asymmetry.csv",
    "context": "similarity_experiments/similarity_in_context.csv",
}
PAIR_COLUMNS = [
    f"Argument {argument}-{part}"
    for argument in (1, 2)
    for part in ("Premise 1", "Premise 2", "Conclusion")
]
CONTEXT_COLUMNS = [
    "Premise 1",
    "Conclusion 1",
    "Conclusion 2",
    "Distractor 1",
    "Distractor 2",
]
PHENOMENON_NAMES = {
    "Monotonicity": "In-Category Monotonicity",
    "Non-Monotonicity (In-Category)": "In-Category Non-Monotonicity",
    "Non-Monotonicity (Cross-Category)": "Cross-Category Non-Monotonicity",
}


def stable_id(task: str, values: list[object]) -> str:
    text = json.dumps([task, *["" if pd.isna(v) else str(v) for v in values]])
    return hashlib.sha256(text.encode()).hexdigest()[:20]


def load_tasks(data_dir: Path) -> dict[str, pd.DataFrame]:
    mapping = pd.read_csv(
        data_dir / "leuven_dataset/leuven_singular_to_plural.csv", index_col=0
    )
    plural_to_singular = dict(zip(mapping["plural"], mapping["singular"]))
    tasks = {}
    for task, filename in TASK_FILES.items():
        frame = pd.read_csv(data_dir / filename, index_col=0)
        frame.insert(0, "source_row", frame.index.astype(str))
        for column in frame:
            if any(
                token in column for token in ("Premise", "Conclusion", "Distractor")
            ):
                if column not in ("Conclusion 1 Chosen", "Conclusion 2 Chosen"):
                    frame[column] = frame[column].replace(plural_to_singular)
        tasks[task] = frame.reset_index(drop=True)
    return tasks


def item_table(task: str, data: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    if task in ("induction", "phenomena"):
        keys = ["Dataset", "Premise 1", "Premise 2", "Premise 3", "Conclusion"]
        if task == "phenomena":
            keys += ["Phenomenon", "Argument Group"]
        items = data.copy()
        items["human"] = data["Human"] if task == "induction" else np.nan
        items["dataset"] = (
            data["Dataset"] if task == "induction" else data["Phenomenon"]
        )
    elif task == "similarity":
        keys = ["Premise 1", "Conclusion", "Domain"]
        items = (
            data.groupby(keys, dropna=False)["Normalized Similarity"]
            .agg(human="mean", human_n="count")
            .reset_index()
        )
        items["dataset"] = items["Domain"]
    elif task in ("asymmetry", "thematic", "nonmonotonicity"):
        keys = PAIR_COLUMNS
        items = (
            data.groupby(keys, dropna=False)["Argument 2 Chosen"]
            .agg(human="mean", human_n="count")
            .reset_index()
        )
        items["dataset"] = task
    else:
        keys = CONTEXT_COLUMNS
        items = human_similarity_context_effects(data)
        items["human"] = items["Context Effect"]
        items["dataset"] = task
    items["task"] = task
    items["item_id"] = [stable_id(task, row) for row in items[keys].values]
    if items["item_id"].duplicated().any():
        # Repeated argument observations must retain distinct source rows.
        items["item_id"] = [
            stable_id(task, [key, source])
            for key, source in zip(items["item_id"], items["source_row"])
        ]
    if task == "similarity":
        items["cluster_id"] = [
            stable_id(task, [domain, *sorted([a, b])])
            for a, b, domain in items[keys].values
        ]
    elif task in ("induction", "phenomena"):
        # Related arguments share a CV/bootstrap cluster across premise variants.
        items["cluster_id"] = [
            stable_id(task, [d, a, b])
            for d, a, b in items[["dataset", "Premise 1", "Conclusion"]].values
        ]
    else:
        items["cluster_id"] = items["item_id"]
    items["paper_selected"] = True
    if task == "asymmetry":
        effects = np.maximum(items["human"], 1 - items["human"])
        items["selection_p"] = binom.sf(
            effects * items["human_n"] - 1, items["human_n"], 0.5
        )
        items["paper_selected"] = items["selection_p"] <= 0.05
    elif task == "context":
        items["paper_selected"] = items["human"] > 0
    return items, keys


def predict_task(task, data, model, object_names, simulation_seed):
    items, keys = item_table(task, data)
    if task in ("induction", "phenomena"):
        items["raw_logit"] = score_arguments(model, items, object_names)
        items["prediction"] = (
            items["raw_logit"] if task == "induction" else expit(items["raw_logit"])
        )
    elif task == "similarity":
        forward = items[["Premise 1", "Conclusion"]].copy()
        forward["Premise 2"] = np.nan
        forward["Premise 3"] = np.nan
        reverse = forward.copy()
        reverse["Premise 1"], reverse["Conclusion"] = (
            forward["Conclusion"],
            forward["Premise 1"],
        )
        items["forward_logit"] = score_arguments(model, forward, object_names)
        items["reverse_logit"] = score_arguments(model, reverse, object_names)
        items["prediction"] = (
            expit(items["forward_logit"]) + expit(items["reverse_logit"])
        ) / 2
    elif task in ("asymmetry", "thematic", "nonmonotonicity"):
        first, second = _paired_arguments(items)
        items["argument1_logit"] = score_arguments(model, first, object_names)
        items["argument2_logit"] = score_arguments(model, second, object_names)
        items["prediction"] = items["argument2_logit"] - items["argument1_logit"]
        items["model_choice"] = items["prediction"] > 0
    else:
        values = pd.DataFrame(
            [
                model_similarity_context_predictions(
                    row, model, object_names, simulation_seed
                )
                for _, row in items.iterrows()
            ],
            index=items.index,
        )
        items = pd.concat([items, values], axis=1)
        items["prediction"] = items["model_effect"]
    # Save a prediction for every original observation, not just the paper-selected subset.
    prediction_columns = [
        c for c in items if c not in data or c in ("item_id", "cluster_id")
    ]
    if task in ("induction", "phenomena"):
        trials = items.copy()
    else:
        trials = data.merge(
            items[[*keys, *prediction_columns]],
            on=keys,
            how="left",
            validate="many_to_one",
        )
        if task == "similarity":
            trials["observed"] = trials["Normalized Similarity"]
        elif task == "context":
            trials["observed_c1"] = trials["Conclusion 1 Chosen"].astype(float)
            trials["observed_c2"] = trials["Conclusion 2 Chosen"].astype(float)
            group = trials["Participant Group"]
            if not group.isin(["Distractor 1", "Distractor 2"]).all():
                raise ValueError("Unknown original context participant group")
            trials["predicted_c1"] = np.where(
                group == "Distractor 1",
                trials["c1-Distractor 1"],
                trials["c1-Distractor 2"],
            )
            trials["predicted_c2"] = np.where(
                group == "Distractor 1",
                trials["c2-Distractor 1"],
                trials["c2-Distractor 2"],
            )
            trials["predicted_distractor"] = (
                1 - trials["predicted_c1"] - trials["predicted_c2"]
            )
        else:
            trials["observed"] = trials["Argument 2 Chosen"].astype(float)
    if len(trials) != len(data) or trials["prediction"].isna().any():
        raise ValueError(f"Incomplete original trial predictions for {task}")
    return items, trials


def safe_correlation(left, right, method="pearson"):
    left, right = np.asarray(left, float), np.asarray(right, float)
    finite = np.isfinite(left) & np.isfinite(right)
    left, right = left[finite], right[finite]
    if len(left) < 3 or np.ptp(left) == 0 or np.ptp(right) == 0:
        return np.nan
    if method == "spearman":
        left, right = rankdata(left), rankdata(right)
    left, right = left - left.mean(), right - right.mean()
    return float(
        np.clip(left @ right / (np.linalg.norm(left) * np.linalg.norm(right)), -1, 1)
    )


def cross_validated_mapping(items: pd.DataFrame, folds: int) -> pd.DataFrame:
    result = items.copy()
    result["cv_prediction"] = np.nan
    result["cv_fold"] = -1
    result["cv_mapping_intercept"] = np.nan
    result["cv_mapping_slope"] = np.nan
    for _, group in items.groupby(["task", "dataset"], sort=True):
        group = group[np.isfinite(group["human"]) & np.isfinite(group["prediction"])]
        n_groups = group["cluster_id"].nunique()
        if n_groups < 3:
            continue
        splitter = GroupKFold(n_splits=min(folds, n_groups))
        for fold, (train, test) in enumerate(
            splitter.split(group, groups=group["cluster_id"])
        ):
            fit = group.iloc[train]
            held = group.iloc[test]
            mapping = LinearRegression().fit(fit[["prediction"]], fit["human"])
            prediction = mapping.predict(held[["prediction"]])
            if held["task"].iloc[0] in ("asymmetry", "thematic", "nonmonotonicity"):
                prediction = np.clip(prediction, 0, 1)
            result.loc[held.index, "cv_prediction"] = prediction
            result.loc[held.index, "cv_fold"] = fold
            result.loc[held.index, "cv_mapping_intercept"] = float(mapping.intercept_)
            result.loc[held.index, "cv_mapping_slope"] = float(mapping.coef_[0])
    return result


def metrics_for_values(task, human, predicted, selected, cv, include_tests=True):
    result = {
        "pearson_r": safe_correlation(predicted, human),
        "spearman_r": safe_correlation(predicted, human, "spearman"),
        "prediction_coverage": float(
            (np.isfinite(predicted) & np.isfinite(human)).mean()
        ),
    }
    finite = np.isfinite(cv) & np.isfinite(human)
    result["cv_rmse"] = (
        float(np.sqrt(np.mean((cv[finite] - human[finite]) ** 2)))
        if finite.any()
        else np.nan
    )
    result["cv_mae"] = (
        float(np.mean(abs(cv[finite] - human[finite]))) if finite.any() else np.nan
    )
    if task in ("similarity", "induction"):
        result["paper_statistic"] = result["pearson_r"]
    elif task in ("asymmetry", "thematic", "nonmonotonicity"):
        agreement = (predicted > 0) == (human > 0.5)
        result["directional_agreement"] = float(agreement.mean())
        result["model_argument2_rate"] = float((predicted > 0).mean())
        if task == "asymmetry":
            result["paper_statistic"] = float(
                ((predicted[selected] > 0) == (human[selected] > 0.5)).mean()
            )
            r = safe_correlation(predicted[selected], human[selected])
            result["paper_asymmetry_r_squared"] = r**2
            k = int(((predicted[selected] > 0) == (human[selected] > 0.5)).sum())
            if include_tests:
                result["paper_binomial_p"] = float(
                    binom.sf(k - 1, int(selected.sum()), 0.5)
                )
        else:
            # Figures 5/6 measure argument-2 preference, not majority agreement.
            result["paper_statistic"] = result["model_argument2_rate"]
            k = int((predicted > 0).sum())
            if include_tests:
                result["paper_binomial_p"] = float(binom.sf(k - 1, len(predicted), 0.5))
    elif task == "context":
        result["directional_agreement"] = float(((predicted > 0) == (human > 0)).mean())
        result["paper_statistic"] = float((predicted[selected] > 0).mean())
        result["direct_rmse"] = float(np.sqrt(np.mean((predicted - human) ** 2)))
        k = int((predicted[selected] > 0).sum())
        if include_tests:
            result["paper_binomial_p"] = float(
                binom.sf(k - 1, int(selected.sum()), 0.5)
            )
    else:
        result["paper_statistic"] = np.nan
    return result


def metric_arrays(items):
    return (
        items["human"].to_numpy(float),
        items["prediction"].to_numpy(float),
        items["paper_selected"].to_numpy(bool),
        items.get("cv_prediction", pd.Series(np.nan, index=items.index)).to_numpy(
            float
        ),
    )


def bootstrap_seed_metrics(task, arrays, indices, seeds):
    # Retain replacement multiplicities/order, but calculate each seed only once.
    cache = {}
    results = []
    for seed in seeds:
        seed = int(seed)
        if seed not in cache:
            cache[seed] = metrics_for_values(
                task, *[values[indices] for values in arrays[seed]], include_tests=False
            )
        results.append(cache[seed])
    return results


def metrics_for_items(items: pd.DataFrame) -> dict[str, float]:
    return metrics_for_values(items["task"].iloc[0], *metric_arrays(items))


def trial_statistics(trials: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (task, dataset), group in trials.groupby(["task", "dataset"]):
        row = {"task": task, "dataset": dataset, "observations": len(group)}
        if task == "context":
            observed = np.column_stack(
                [
                    group["observed_c1"],
                    group["observed_c2"],
                    1 - group["observed_c1"] - group["observed_c2"],
                ]
            )
            probabilities = group[
                ["predicted_c1", "predicted_c2", "predicted_distractor"]
            ].values
            row["multiclass_brier"] = float(
                np.mean(np.sum((observed - probabilities) ** 2, axis=1))
            )
            row["log_loss"] = float(
                -np.mean(
                    np.sum(observed * np.log(np.clip(probabilities, 1e-12, 1)), axis=1)
                )
            )
        elif task != "phenomena":
            observed = group["Human"] if task == "induction" else group["observed"]
            mapped = group["cv_prediction"]
            finite = np.isfinite(mapped) & np.isfinite(observed)
            row["cv_trial_rmse"] = float(
                np.sqrt(np.mean((mapped[finite] - observed[finite]) ** 2))
            )
            row["cv_trial_mae"] = float(np.mean(abs(mapped[finite] - observed[finite])))
            if task in ("thematic", "nonmonotonicity", "asymmetry"):
                row["individual_choice_agreement"] = float(
                    np.mean(group["model_choice"] == observed)
                )
                row["cv_brier"] = float(
                    np.mean((mapped[finite] - observed[finite]) ** 2)
                )
        rows.append(row)
    return pd.DataFrame(rows)


def observed_paper_choices(tasks):
    participants, summary = [], []
    for task in ("thematic", "nonmonotonicity"):
        data = tasks[task]
        fractions = data.groupby("participant_id")["Argument 2 Chosen"].mean()
        for participant, value in fractions.items():
            participants.append(
                {
                    "task": task,
                    "participant_id": participant,
                    "argument2_preference": value,
                    "source": TASK_FILES[task],
                }
            )
        items, _ = item_table(task, data)
        k = int((fractions > 0.5).sum())
        summary.append(
            {
                "task": task,
                "participants": len(fractions),
                "paper_participant_median": fractions.median(),
                "participant_mean": fractions.mean(),
                "participants_preferring_argument2": k,
                "participant_binomial_p": binom.sf(k - 1, len(fractions), 0.5),
                "item_mean_argument2_rate": items["human"].mean(),
                "measure": "Figures 5/6: within-participant fraction choosing argument 2",
            }
        )
    return pd.DataFrame(participants), pd.DataFrame(summary)


def phenomenon_statistics(items):
    rows = []
    for phenomenon, group in items.groupby("Phenomenon"):
        name = PHENOMENON_NAMES.get(phenomenon, phenomenon)
        high = group[group["Argument Group"] == "High"]
        low = group[group["Argument Group"] == "Low"]
        if "Monotonicity" in name:
            paired = high.merge(
                low, on=["Premise 1", "Conclusion"], suffixes=("_high", "_low")
            )
            test = ttest_rel(paired["prediction_high"], paired["prediction_low"])
            test_name = "paired_t"
        else:
            test = ttest_ind(high["prediction"], low["prediction"])
            test_name = "unpaired_t_equal_variance"
        rows.append(
            {
                "phenomenon": name,
                "high_mean": high["prediction"].mean(),
                "low_mean": low["prediction"].mean(),
                "t": test.statistic,
                "p": test.pvalue,
                "test": test_name,
                "observed_human_judgments_available": False,
            }
        )
    return pd.DataFrame(rows)


def summarize_behavior(predictions, repetitions, seed):
    per_seed, summary = [], []
    rng = np.random.default_rng(seed)
    for (condition, task, dataset), group in predictions.groupby(
        ["condition", "task", "dataset"], sort=True
    ):
        if task == "phenomena":
            continue
        by_seed = {
            int(s): g.sort_values("item_id").reset_index(drop=True)
            for s, g in group.groupby("seed")
        }
        seed_values = list(by_seed)
        reference = by_seed[seed_values[0]]
        for value in by_seed.values():
            if not value["item_id"].equals(reference["item_id"]):
                raise ValueError("Seeds must have identical behavioral items")
        evaluated = {s: metrics_for_items(g) for s, g in by_seed.items()}
        arrays = {s: metric_arrays(g) for s, g in by_seed.items()}
        for s, metrics in evaluated.items():
            per_seed.append(
                {
                    "condition": condition,
                    "task": task,
                    "dataset": dataset,
                    "seed": s,
                    "items": len(reference),
                    "paper_selected_items": int(reference["paper_selected"].sum()),
                    **metrics,
                }
            )
        clusters = [
            np.flatnonzero(reference["cluster_id"].values == c)
            for c in reference["cluster_id"].unique()
        ]
        draws = {
            m: [] for m in next(iter(evaluated.values())) if m != "paper_binomial_p"
        }
        for _ in range(repetitions):
            indices = np.concatenate(
                [clusters[i] for i in rng.integers(0, len(clusters), len(clusters))]
            )
            seeds = rng.choice(seed_values, len(seed_values), replace=True)
            draw_metrics = bootstrap_seed_metrics(task, arrays, indices, seeds)
            for metric in draws:
                values = [m[metric] for m in draw_metrics if np.isfinite(m[metric])]
                draws[metric].append(np.mean(values) if values else np.nan)
        for metric in draws:
            values = np.array([m[metric] for m in evaluated.values()])
            finite = values[np.isfinite(values)]
            samples = np.array(draws[metric])
            samples = samples[np.isfinite(samples)]
            summary.append(
                {
                    "condition": condition,
                    "task": task,
                    "dataset": dataset,
                    "metric": metric,
                    "seed_count": len(seed_values),
                    "items": len(reference),
                    "mean": float(finite.mean()) if len(finite) else np.nan,
                    "seed_sd": float(finite.std(ddof=1)) if len(finite) > 1 else np.nan,
                    "ci_low": (
                        float(np.quantile(samples, 0.025)) if len(samples) else np.nan
                    ),
                    "ci_high": (
                        float(np.quantile(samples, 0.975)) if len(samples) else np.nan
                    ),
                    "finite_bootstrap_draws": len(samples),
                    "ci_method": "crossed_cluster_item_and_seed_percentile",
                    "cv_bootstrap_note": "CV predictions fixed; mapping not refit inside bootstrap",
                }
            )
    return pd.DataFrame(per_seed), pd.DataFrame(summary)


def paired_condition_contrasts(predictions, pairs, repetitions, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for left, right in pairs:
        for (task, dataset), task_data in predictions.groupby(["task", "dataset"]):
            if task == "phenomena":
                continue
            left_data = task_data[task_data["condition"] == left]
            right_data = task_data[task_data["condition"] == right]
            seeds = sorted(set(left_data["seed"]) & set(right_data["seed"]))
            if not seeds:
                continue
            aligned = {}
            for s in seeds:
                a = (
                    left_data[left_data["seed"] == s]
                    .sort_values("item_id")
                    .reset_index(drop=True)
                )
                b = (
                    right_data[right_data["seed"] == s]
                    .sort_values("item_id")
                    .reset_index(drop=True)
                )
                if not a["item_id"].equals(b["item_id"]):
                    raise ValueError("Condition contrasts require identical trials")
                aligned[s] = (a, b)
            ref = aligned[seeds[0]][0]
            arrays = {
                s: (metric_arrays(a), metric_arrays(b)) for s, (a, b) in aligned.items()
            }
            left_arrays = {s: pair[0] for s, pair in arrays.items()}
            right_arrays = {s: pair[1] for s, pair in arrays.items()}
            clusters = [
                np.flatnonzero(ref["cluster_id"].values == c)
                for c in ref["cluster_id"].unique()
            ]
            metrics = ("paper_statistic", "spearman_r", "cv_rmse")
            actual = {m: [] for m in metrics}
            boot = {m: [] for m in metrics}
            for a, b in aligned.values():
                x, y = metrics_for_items(a), metrics_for_items(b)
                for m in metrics:
                    actual[m].append(x[m] - y[m])
            for _ in range(repetitions):
                idx = np.concatenate(
                    [clusters[i] for i in rng.integers(0, len(clusters), len(clusters))]
                )
                sampled_seeds = rng.choice(seeds, len(seeds), replace=True)
                values = {m: [] for m in metrics}
                left_metrics = bootstrap_seed_metrics(
                    task, left_arrays, idx, sampled_seeds
                )
                right_metrics = bootstrap_seed_metrics(
                    task, right_arrays, idx, sampled_seeds
                )
                for x, y in zip(left_metrics, right_metrics):
                    for m in metrics:
                        values[m].append(x[m] - y[m])
                for m in metrics:
                    finite = [v for v in values[m] if np.isfinite(v)]
                    boot[m].append(np.mean(finite) if finite else np.nan)
            for m in metrics:
                values = np.array(actual[m])
                values = values[np.isfinite(values)]
                draws = np.array(boot[m])
                draws = draws[np.isfinite(draws)]
                rows.append(
                    {
                        "condition_a": left,
                        "condition_b": right,
                        "task": task,
                        "dataset": dataset,
                        "metric": m,
                        "difference_a_minus_b": (
                            values.mean() if len(values) else np.nan
                        ),
                        "ci_low": np.quantile(draws, 0.025) if len(draws) else np.nan,
                        "ci_high": np.quantile(draws, 0.975) if len(draws) else np.nan,
                        "paired_seeds": len(seeds),
                        "paired_items": len(ref),
                        "finite_bootstrap_draws": len(draws),
                    }
                )
    return pd.DataFrame(
        rows,
        columns=[
            "condition_a",
            "condition_b",
            "task",
            "dataset",
            "metric",
            "difference_a_minus_b",
            "ci_low",
            "ci_high",
            "paired_seeds",
            "paired_items",
            "finite_bootstrap_draws",
        ],
    )


def reliability_tables(tasks, repetitions, seed):
    """Available individual responses; no reliability invented for aggregate-only tasks."""
    rows, participants = [], []
    rng = np.random.default_rng(seed)
    for task, data in tasks.items():
        if "participant_id" not in data:
            rows.append(
                {
                    "task": task,
                    "dataset": "all",
                    "status": "unavailable_no_individual_human_responses",
                }
            )
            continue
        groups = data.groupby("Domain") if task == "similarity" else [(task, data)]
        for dataset, group in groups:
            keys = (
                ["Premise 1", "Conclusion"]
                if task == "similarity"
                else CONTEXT_COLUMNS if task == "context" else PAIR_COLUMNS
            )
            measure = (
                "Normalized Similarity" if task == "similarity" else "Argument 2 Chosen"
            )
            if task != "context":
                total = group.groupby(keys, dropna=False)[measure].agg(["sum", "count"])
                for participant, subset in group.groupby("participant_id"):
                    own = subset.groupby(keys, dropna=False)[measure].agg(
                        ["sum", "count", "mean"]
                    )
                    joined = own.join(total, rsuffix="_total")
                    rest_mean = (joined["sum_total"] - joined["sum"]) / (
                        joined["count_total"] - joined["count"]
                    )
                    participants.append(
                        {
                            "task": task,
                            "dataset": dataset,
                            "participant_id": participant,
                            "loo_mean_pearson": safe_correlation(
                                joined["mean"], rest_mean
                            ),
                            "paper_sum_rest_pearson": safe_correlation(
                                joined["mean"], joined["sum_total"] - joined["mean"]
                            ),
                            "items": len(joined),
                        }
                    )
            ids = group["participant_id"].unique()
            split_correlations = []
            for _ in range(repetitions):
                shuffled = rng.permutation(ids)
                first = group[group["participant_id"].isin(shuffled[: len(ids) // 2])]
                second = group[group["participant_id"].isin(shuffled[len(ids) // 2 :])]
                if task == "context":
                    a = human_similarity_context_effects(first).set_index(keys)[
                        "Context Effect"
                    ]
                    b = human_similarity_context_effects(second).set_index(keys)[
                        "Context Effect"
                    ]
                else:
                    a = first.groupby(keys, dropna=False)[measure].mean()
                    b = second.groupby(keys, dropna=False)[measure].mean()
                aligned = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
                split_correlations.append(safe_correlation(aligned["a"], aligned["b"]))
            finite = np.array(split_correlations)
            finite = finite[np.isfinite(finite)]
            corrected = 2 * finite / (1 + finite)
            rows.append(
                {
                    "task": task,
                    "dataset": dataset,
                    "status": "estimated_from_original_participants",
                    "participants": len(ids),
                    "split_half_pearson_mean": finite.mean() if len(finite) else np.nan,
                    "spearman_brown_reliability_mean": (
                        corrected.mean() if len(corrected) else np.nan
                    ),
                    "spearman_brown_low": (
                        np.quantile(corrected, 0.025) if len(corrected) else np.nan
                    ),
                    "spearman_brown_high": (
                        np.quantile(corrected, 0.975) if len(corrected) else np.nan
                    ),
                    "interpretation": "reliability estimate, not a guaranteed model noise ceiling",
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(participants)


def notebook_reference(upstream: Path) -> pd.DataFrame:
    rows = []
    for notebook, cell_index, task, metric in [
        ("1. Part 1 Inductive Inference.ipynb", 4, "induction", "paper_statistic"),
        ("2. Part 2 Similarity.ipynb", 4, "similarity", "paper_statistic"),
        ("2. Part 2 Similarity.ipynb", 7, "asymmetry", "paper_statistic"),
        ("2. Part 2 Similarity.ipynb", 8, "asymmetry", "paper_asymmetry_r_squared"),
        ("2. Part 2 Similarity.ipynb", 11, "context", "paper_statistic"),
    ]:
        cell = json.loads((upstream / notebook).read_text())["cells"][cell_index]
        matches = [
            trace
            for output in cell.get("outputs", [])
            for trace in output.get("data", {})
            .get("application/vnd.plotly.v1+json", {})
            .get("data", [])
            if trace.get("name") == "ISC-CI"
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Missing/ambiguous saved ISC-CI reference: {notebook} cell {cell_index}"
            )
        trace = matches[0]
        for dataset, value in zip(trace["x"], trace["y"]):
            rows.append(
                {
                    "task": task,
                    "dataset": task if dataset == "ISC-CI" else dataset,
                    "metric": metric,
                    "reference": float(value),
                    "source": notebook,
                    "cell": cell_index,
                    "reference_model": "released_seed_3",
                }
            )
    return pd.DataFrame(rows)
