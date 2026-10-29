from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform

from iscci_validation.behavior import safe_correlation
from .inventory import CONDITIONS, sha256
from .provenance import atomic_json, code_hashes, validate_output_directory, digest
from .run_crosses import all_pairs, planned_jobs


def read_table(path):
    if not path.exists() or path.stat().st_size < 2:
        return pd.DataFrame()
    return pd.read_csv(path)


def verified_evaluation(directory, sources):
    path = directory / "manifest.json"
    if not path.exists():
        return None
    manifest = json.loads(path.read_text())
    for name, expected in manifest["outputs"].items():
        if sha256(directory / name) != expected:
            raise ValueError(f"Saved evaluation input changed: {directory / name}")
        sources[str(directory / name)] = expected
    sources[str(path)] = sha256(path)
    return manifest


def recorded_job(job, sources):
    directory = Path(job["directory"])
    path = directory / "manifest.json"
    if not path.exists():
        return "planned"
    record = json.loads(path.read_text())
    sources[str(path)] = sha256(path)
    if not record.get("complete"):
        return record.get("status", "incomplete")
    scope = "context" if "context_source" in job else "semantic"
    if record["signature"].get("code") != code_hashes(scope):
        return "stale_implementation"
    for name, expected in record["outputs"].items():
        if not (directory / name).exists() or sha256(directory / name) != expected:
            return "corrupt"
        sources[str(directory / name)] = expected
    return "complete"


def save_figure(fig, name, directory):
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    for suffix in ["svg", "pdf", "png"]:
        fig.savefig(directory / f"{name}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_figures(output, destination, summary, semantic, context):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "svg.fonttype": "none", "pdf.fonttype": 42})
    directory = destination / "figures"
    made = []
    jobs = planned_jobs(output)
    learning = []
    for job in jobs["semantic"]:
        path = Path(job["directory"]) / "training_metrics.json"
        if path.exists():
            frame = pd.DataFrame(json.loads(path.read_text()))
            frame["condition"], frame["seed"] = job["semantic_source"], job["seed"]
            learning.append(frame)
    if learning:
        curves = pd.concat(learning, ignore_index=True)
        curves.to_csv(
            destination / "source_data/semantic_learning_curves.csv", index=False
        )
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")
        for condition, group in curves.groupby("condition"):
            mean = group.groupby("epoch")[
                ["bce", "max_absolute_error", "positive_recall"]
            ].mean()
            for ax, metric in zip(axes, mean):
                ax.plot(mean.index, mean[metric], label=condition, linewidth=1)
                ax.set(xlabel="Epoch", ylabel=metric.replace("_", " "))
        axes[1].axhline(0.1, linestyle="--", color="black", linewidth=0.6)
        axes[2].legend(fontsize=6, bbox_to_anchor=(1.02, 1), loc="upper left")
        save_figure(fig, "semantic_learning", directory)
        made.append("semantic_learning")
    if semantic:
        semantic_scores = read_table(
            output / "evaluations/semantic/semantic_similarity_summary.csv"
        )
        raw_scores = read_table(
            output / "evaluations/semantic/raw_similarity_summary.csv"
        )
        if not semantic_scores.empty:
            semantic_scores = semantic_scores[
                semantic_scores["metric"] == "paper_statistic"
            ]
            raw_scores = raw_scores[raw_scores["metric"] == "paper_statistic"]
            domains = list(semantic_scores["dataset"].unique())
            fig, axes = plt.subplots(3, 3, figsize=(21, 17), layout="constrained")
            for ax, domain in zip(axes.flat, domains):
                for level, frame, offset, color in [
                    ("Raw features", raw_scores, -0.13, "#777777"),
                    ("Pretrained semantics", semantic_scores, 0.13, "#007f73"),
                ]:
                    selected = (
                        frame[frame["dataset"] == domain]
                        .set_index("condition")
                        .reindex(CONDITIONS)
                    )
                    mean = selected["mean"].values
                    low, high = selected["ci_low"].values, selected["ci_high"].values
                    ax.hlines(
                        np.arange(14) + offset, low, high, linewidth=0.7, color=color
                    )
                    ax.plot(
                        mean,
                        np.arange(14) + offset,
                        "o",
                        markersize=3,
                        color=color,
                        label=level,
                    )
                ax.set(
                    yticks=np.arange(14),
                    yticklabels=list(CONDITIONS),
                    xlim=(-1, 1),
                    title=str(domain),
                    xlabel="Original Pearson r",
                )
                ax.tick_params(labelsize=8)
                ax.invert_yaxis()
            axes.flat[0].legend(fontsize=8)
            for ax in list(axes.flat)[len(domains) :]:
                ax.set_visible(False)
            save_figure(fig, "human_similarity_raw_vs_semantic", directory)
            made.append("human_similarity_raw_vs_semantic")
        task = read_table(output / "evaluations/semantic/task_geometry.csv")
        if not task.empty:
            counts = (
                task.groupby(["condition", "task_id"])["features"]
                .first()
                .unstack()
                .reindex(CONDITIONS)
            )
            counts.to_csv(destination / "source_data/task_feature_counts.csv")
            fig, ax = plt.subplots(figsize=(14, 5), layout="constrained")
            im = ax.imshow(np.log1p(counts), aspect="auto", cmap="viridis")
            ax.set(
                xticks=np.arange(32),
                xticklabels=np.arange(32),
                yticks=np.arange(14),
                yticklabels=list(CONDITIONS),
                xlabel="Reconstructed task ID",
            )
            fig.colorbar(im, ax=ax, label="log(1 + feature count)")
            save_figure(fig, "task_coverage", directory)
            made.append("task_coverage")
            fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
            mean = task.groupby("task_id")[
                ["independent_spearman", "dependent_spearman"]
            ].mean()
            ax.plot(
                mean.index,
                mean["independent_spearman"],
                marker="o",
                markersize=3,
                label="Independent hub",
            )
            ax.plot(
                mean.index,
                mean["dependent_spearman"],
                marker="o",
                markersize=3,
                label="Task-dependent representation",
            )
            ax.set(
                xlabel="Reconstructed task ID",
                ylabel="Task-feature RDM Spearman correlation",
            )
            ax.legend()
            save_figure(fig, "task_control_geometry", directory)
            made.append("task_control_geometry")
        transfer = read_table(
            output / "evaluations/semantic/feature_space_transfer.csv"
        )
        if not transfer.empty:
            grid = (
                transfer.groupby(["semantic_source", "feature_source"])["spearman"]
                .mean()
                .unstack()
                .reindex(index=CONDITIONS, columns=CONDITIONS)
            )
            fig, ax = plt.subplots(figsize=(12, 8), layout="constrained")
            im = ax.imshow(grid, cmap="coolwarm", vmin=-1, vmax=1)
            ax.set(
                xticks=np.arange(14),
                xticklabels=np.arange(14),
                yticks=np.arange(14),
                yticklabels=list(CONDITIONS),
                xlabel="Feature source ID",
                ylabel="Semantic source",
            )
            fig.colorbar(
                im, ax=ax, label="Semantic-to-feature RDM Spearman correlation"
            )
            save_figure(fig, "semantic_feature_space_transfer", directory)
            made.append("semantic_feature_space_transfer")
        with np.load(
            output / "evaluations/semantic/rdms.npz", allow_pickle=False
        ) as saved:
            for prefix, title in [
                ("raw", "Feature input"),
                ("semantic", "Independent sigmoid hub, prespecified seed 0"),
            ]:
                fig, axes = plt.subplots(4, 4, figsize=(14, 13), layout="constrained")
                for ax, condition in zip(axes.flat, CONDITIONS):
                    key = (
                        f"raw__{condition}"
                        if prefix == "raw"
                        else f"semantic__{condition}__seed_0"
                    )
                    if key in saved:
                        distances = saved[key]
                        if prefix == "semantic":
                            from scipy.stats import rankdata

                            finite = np.isfinite(distances)
                            distances = distances.copy()
                            distances[finite] = rankdata(distances[finite]) / max(
                                1, finite.sum()
                            )
                        ax.imshow(squareform(distances), cmap="viridis", vmin=0, vmax=1)
                    else:
                        ax.text(0.5, 0.5, "Not run", ha="center", va="center")
                    ax.set_title(condition, fontsize=8)
                    ax.set_xticks([])
                    ax.set_yticks([])
                for ax in list(axes.flat)[14:]:
                    ax.set_visible(False)
                fig.suptitle(
                    title
                    + (
                        "; ranked cosine dissimilarities"
                        if prefix == "semantic"
                        else "; cosine dissimilarities"
                    )
                    + "; frozen object order"
                )
                save_figure(fig, f"{prefix}_object_rdms", directory)
                made.append(f"{prefix}_object_rdms")
    if context:
        behavior = read_table(
            output
            / f"evaluations/context/phase_{context['phase']}/behavior_summary.csv"
        )
        selected = behavior[behavior["metric"] == "paper_statistic"]
        for task, group in selected.groupby("task"):
            datasets = list(group["dataset"].unique())
            cols = min(3, len(datasets))
            rows = (len(datasets) + cols - 1) // cols
            fig, axes = plt.subplots(
                rows,
                cols,
                figsize=(6 * cols, 5 * rows),
                squeeze=False,
                layout="constrained",
            )
            for ax, dataset in zip(axes.flat, datasets):
                table = group[group["dataset"] == dataset].copy()
                table[["semantic_source", "context_source"]] = table[
                    "condition"
                ].str.split("__", expand=True)
                grid = table.pivot(
                    index="semantic_source", columns="context_source", values="mean"
                ).reindex(index=CONDITIONS, columns=CONDITIONS)
                im = ax.imshow(grid.values, cmap="viridis", aspect="auto")
                short = [f"{i}: {c}" for i, c in enumerate(CONDITIONS)]
                ax.set(
                    xticks=np.arange(14),
                    xticklabels=np.arange(14),
                    yticks=np.arange(14),
                    yticklabels=short,
                    xlabel="Context source ID",
                    ylabel="Semantic source",
                    title=str(dataset),
                )
                ax.tick_params(labelsize=7)
                fig.colorbar(im, ax=ax, shrink=0.7, label="Original paper statistic")
            for ax in list(axes.flat)[len(datasets) :]:
                ax.set_visible(False)
            save_figure(fig, f"crossed_behavior_{task}", directory)
            made.append(f"crossed_behavior_{task}")
        contrasts = read_table(
            output
            / f"evaluations/context/phase_{context['phase']}/paired_contrasts.csv"
        )
        if not contrasts.empty:
            a = contrasts["condition_a"].str.split("__", expand=True)
            b = contrasts["condition_b"].str.split("__", expand=True)
            matched = contrasts[
                (a[0] == a[1])
                & (b[0] == b[1])
                & (contrasts["metric"] == "paper_statistic")
            ]
            for task, group in matched.groupby("task"):
                datasets = list(group["dataset"].unique())
                cols = min(3, len(datasets))
                rows = (len(datasets) + cols - 1) // cols
                fig, axes = plt.subplots(
                    rows,
                    cols,
                    figsize=(8 * cols, 5 * rows),
                    squeeze=False,
                    layout="constrained",
                )
                for ax, dataset in zip(axes.flat, datasets):
                    table = group[group["dataset"] == dataset]
                    means, low, high = [
                        table[column].values
                        for column in ["difference_a_minus_b", "ci_low", "ci_high"]
                    ]
                    ax.hlines(
                        np.arange(len(table)), low, high, linewidth=0.8, color="#007f73"
                    )
                    ax.plot(
                        means, np.arange(len(table)), "o", markersize=3, color="#007f73"
                    )
                    labels = [
                        f"{left.split('__')[0]} - {right.split('__')[0]}"
                        for left, right in table[
                            ["condition_a", "condition_b"]
                        ].itertuples(index=False, name=None)
                    ]
                    ax.axvline(0, color="black", linewidth=0.6)
                    ax.set(
                        yticks=np.arange(len(table)),
                        yticklabels=labels,
                        title=str(dataset),
                        xlabel="Difference in original paper statistic",
                    )
                    ax.tick_params(labelsize=7)
                    ax.invert_yaxis()
                for ax in list(axes.flat)[len(datasets) :]:
                    ax.set_visible(False)
                save_figure(fig, f"matched_pipeline_contrasts_{task}", directory)
                made.append(f"matched_pipeline_contrasts_{task}")
    return made


def report(root, output, destination, config, phase):
    output, destination = Path(output), Path(destination)
    validate_output_directory(destination)
    sources = {}
    semantic = verified_evaluation(output / "evaluations/semantic", sources)
    context = verified_evaluation(
        output / f"evaluations/context/phase_{phase}", sources
    )
    jobs = planned_jobs(output)
    statuses = {
        job["id"]: recorded_job(job, sources)
        for group in jobs.values()
        for job in group
    }
    raw = (
        read_table(output / "evaluations/semantic/raw_geometry.csv")
        if semantic
        else pd.DataFrame()
    )
    sem_repr = (
        read_table(output / "evaluations/semantic/representation_per_seed.csv")
        if semantic
        else pd.DataFrame()
    )
    ctx_repr = (
        read_table(
            output / f"evaluations/context/phase_{phase}/representation_per_seed.csv"
        )
        if context
        else pd.DataFrame()
    )
    behavior = (
        read_table(output / f"evaluations/context/phase_{phase}/behavior_summary.csv")
        if context
        else pd.DataFrame()
    )
    rows = []
    for s, c in all_pairs():
        completed = sum(
            statuses[f"{s}__{c}_seed_{seed}"] == "complete" for seed in range(4)
        )
        pair_states = {statuses[f"{s}__{c}_seed_{seed}"] for seed in range(4)}
        row = {
            "semantic_source": s,
            "context_source": c,
            "context_features": CONDITIONS[c],
            "semantic_features": CONDITIONS[s],
            "expected_seeds": 4,
            "completed_seeds": completed,
            "status": (
                "complete"
                if completed == 4
                else (
                    "partial"
                    if completed
                    else (
                        "planned"
                        if pair_states == {"planned"}
                        else "blocked:" + ",".join(sorted(pair_states - {"planned"}))
                    )
                )
            ),
        }
        if not raw.empty:
            for source, prefix in [(s, "semantic_input"), (c, "context_input")]:
                selected = raw[raw["condition"] == source]
                row[f"{prefix}_leuven_rdm_spearman"] = selected.iloc[0]["spearman"]
        if not sem_repr.empty:
            selected = sem_repr[
                (sem_repr["condition"] == s) & (sem_repr["scope"] == "all")
            ]
            row["semantic_activation_agreement_mean"] = selected[
                "semantic_rdm_spearman"
            ].mean()
            row["semantic_activation_agreement_seed_sd"] = selected[
                "semantic_rdm_spearman"
            ].std()
        if not ctx_repr.empty:
            selected = ctx_repr[ctx_repr["condition"] == f"{s}__{c}"]
            for metric in [
                "context_rdm_spearman",
                "context_dependent_rdm_spearman_median",
                "membership_logit_spearman",
            ]:
                row[f"iscci_{metric}_mean"] = selected[metric].mean()
                row[f"iscci_{metric}_seed_sd"] = selected[metric].std()
        if not behavior.empty:
            selected = behavior[behavior["condition"] == f"{s}__{c}"]
            for _, statistic in selected.iterrows():
                prefix = f"behavior__{statistic['task']}__{statistic['dataset']}__{statistic['metric']}"
                for measure in [
                    "mean",
                    "seed_sd",
                    "ci_low",
                    "ci_high",
                    "seed_count",
                    "items",
                ]:
                    row[f"{prefix}__{measure}"] = statistic[measure]
        rows.append(row)
    summary = pd.DataFrame(rows)
    destination.mkdir(parents=True, exist_ok=True)
    data = destination / "source_data"
    data.mkdir(exist_ok=True)
    summary.to_csv(data / "all_196_conditions.csv", index=False)
    pd.DataFrame(
        [
            {**job, "status": statuses[job["id"]]}
            for group in jobs.values()
            for job in group
        ]
    ).to_csv(data / "all_840_planned_models.csv", index=False)
    # Keep the original individual human reliability and all external outcomes available.
    for directory in [
        output / "evaluations/semantic",
        output / f"evaluations/context/phase_{phase}",
    ]:
        if directory.exists():
            for path in directory.glob("*.csv"):
                if path.name not in {
                    "all_item_predictions.csv",
                    "raw_similarity_items.csv",
                    "semantic_similarity_items.csv",
                }:
                    frame = read_table(path)
                    frame.to_csv(data / f"{directory.name}_{path.name}", index=False)
    associations = []
    metrics = [
        c
        for c in summary
        if c.startswith("behavior__") and c.endswith("__paper_statistic__mean")
    ]
    for metric in metrics:
        for subset, group in [
            (
                "matched_diagonal",
                summary[summary["semantic_source"] == summary["context_source"]],
            ),
            ("context_fixed_human", summary[summary["context_source"] == "human"]),
            ("semantic_fixed_human", summary[summary["semantic_source"] == "human"]),
        ]:
            for geometry in [
                "semantic_input_leuven_rdm_spearman",
                "context_input_leuven_rdm_spearman",
                "semantic_activation_agreement_mean",
            ]:
                if geometry in group:
                    valid = np.isfinite(group[geometry]) & np.isfinite(group[metric])
                    associations.append(
                        {
                            "subset": subset,
                            "behavior": metric,
                            "geometry": geometry,
                            "sources": int(valid.sum()),
                            "spearman": safe_correlation(
                                group.loc[valid, geometry],
                                group.loc[valid, metric],
                                "spearman",
                            ),
                            "pearson": safe_correlation(
                                group.loc[valid, geometry], group.loc[valid, metric]
                            ),
                            "interpretation": "Descriptive fixed related source inventory; not independent datasets or causal evidence",
                        }
                    )
    pd.DataFrame(associations).to_csv(
        data / "geometry_behavior_association.csv", index=False
    )
    figures = make_figures(output, destination, summary, semantic, context)
    sem_count = sum(statuses[j["id"]] == "complete" for j in jobs["semantic"])
    ctx_count = sum(statuses[j["id"]] == "complete" for j in jobs["context"])
    complete = (
        sem_count == 56
        and ctx_count == 784
        and bool(context and context["complete"] and len(context["models"]) == 784)
    )
    text = f"""# Independent ISC Semantic Retraining

## Status

Approved 32-class reconstruction with corrected complete semantic targets.
Semantic models: {sem_count}/56 complete. ISC-CI models: {ctx_count}/784 complete.
The factorial contains 196 related source pairs; missing runs are not omitted.
Experiment complete: {complete}. A generated report is not evidence that training ran.

## Separate Evaluation Levels

1. Raw retained feature matrices predict the original human similarity judgments.
2. Newly pretrained sigmoid semantic activations predict the same judgments.
3. Fully trained ISC-CI models predict all original paper tasks using the frozen code.

Full source/condition table: source_data/all_196_conditions.csv. Every planned model:
source_data/all_840_planned_models.csv. Item predictions remain in the hashed
evaluation directories; figures never train models or generate predictions.
Original human reliability tables accompany the new human-trained baseline when
behavioral evaluation exists. Unavailable reliability is not invented.

## Interpretation Rules

Behavior intervals use the original 2000-draw crossed human-item/seed procedure;
representation SD is variation across training seeds, not a human noise ceiling.
Raw matrices have one realization, not four independent seeds. Domain means are
descriptive averages of correlations, never pooled correlations. Preference rates
are not accuracy. RDMs retain undefined pairs and report their coverage.

Main source/context effects and interactions are decomposed within paired seeds,
not inferred from 196 independent datasets. Prespecified matched-pipeline and
single-axis contrasts retain the original paired item/seed bootstrap. The full
contrast design and unavailable contrasts remain exported. Geometry-behavior
associations are exploratory descriptions of a fixed, related source inventory.

No human behavioral result may change taxonomy assignments, feature decisions,
thresholds, training schedules, or model selection. The original 33-class numeric
dictionary remains unresolved; this is not exact reproduction of the historical
350-object encoder. Existing completed experiments are unchanged.

## Figure Notes

Learning curves average available seeds within source; BCE magnitude is sensitive
to inventory sparsity and is not sufficient evidence of feature learning. Task
geometry reports both improvement and deterioration, with undefined classes left
missing. Object RDMs use frozen object order and prespecified seed 0, not the best
performing seed. Crossed heatmaps report each original task/domain separately.
Generated figure families: {', '.join(figures) if figures else 'none; required model evaluations have not run'}.
"""
    (destination / "RESULTS.md").write_text(text)
    result = {
        "report_generated": True,
        "protocol": config["protocol"],
        "config_sha256": digest(config),
        "experiment_complete": complete,
        "semantic_models_complete": sem_count,
        "context_models_complete": ctx_count,
        "source_pairs": 196,
        "figures": figures,
        "inputs": sources,
        "report_outputs": {
            str(p.relative_to(destination)): sha256(p)
            for p in destination.rglob("*")
            if p.is_file() and p.name != "report_manifest.json"
        },
    }
    atomic_json(destination / "report_manifest.json", result)
    return result
