from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import pdist

from iscci_validation.behavior import (
    item_table,
    cross_validated_mapping,
    safe_correlation,
    load_tasks,
    summarize_behavior,
    paired_condition_contrasts,
    reliability_tables,
    phenomenon_statistics,
)
from .extract_representations import extract, load_run
from .gates import saved_path, verify_history
from .inventory import CONDITIONS, load_matrix, matrix_path, sha256
from .provenance import atomic_json, code_hashes, digest
from .training_adapter import TaskDataset


def scoring_functions():
    from run_human_behavior_validation import run_predictions

    return {
        "predict": run_predictions,
        "summarize": summarize_behavior,
        "contrasts": paired_condition_contrasts,
    }


def finite_mean(values):
    values = np.asarray(values, dtype=float)
    valid = values[np.isfinite(values)]
    return float(valid.mean()) if len(valid) else np.nan


def rdm_statistics(left, right, categories=None):
    valid = np.isfinite(left) & np.isfinite(right)
    masks = {"all": np.ones(len(left), dtype=bool)}
    if categories is not None:
        i, j = np.triu_indices(len(categories), 1)
        within = np.asarray(categories)[i] == np.asarray(categories)[j]
        masks.update(within=within, between=~within)
    return {
        name: {
            "spearman": safe_correlation(
                left[valid & mask], right[valid & mask], "spearman"
            ),
            "pearson": safe_correlation(left[valid & mask], right[valid & mask]),
            "valid_pairs": int((valid & mask).sum()),
            "total_pairs": int(mask.sum()),
        }
        for name, mask in masks.items()
    }


def rdm_agreement(left, right, categories=None):
    return rdm_statistics(pdist(left, "cosine"), pdist(right, "cosine"), categories)[
        "all"
    ]


def similarity_predictions(
    values, names, original_data, condition, seed, level, folds=5
):
    items, _ = item_table("similarity", original_data)
    index = {name: i for i, name in enumerate(names)}
    if len(index) != len(names) or len(values) != len(names):
        raise ValueError("Invalid object mapping for original human similarity trials")
    predictions = []
    # Only two feature rows are converted at a time, even for 87k-wide matrices.
    for a, b in items[["Premise 1", "Conclusion"]].itertuples(index=False, name=None):
        left, right = values[index[a]].astype(float), values[index[b]].astype(float)
        denominator = np.linalg.norm(left) * np.linalg.norm(right)
        predictions.append(
            float(np.sum(left * right) / denominator) if denominator > 0 else np.nan
        )
    items["prediction"] = predictions
    items["prediction_status"] = np.where(
        np.isfinite(predictions), "defined", "undefined_zero_feature_vector"
    )
    items["condition"], items["seed"], items["level"] = condition, seed, level
    return cross_validated_mapping(items, folds)


def task_geometry(model, dataset):
    rows = []
    objects = torch.eye(model.num_objects)
    hub = extract(model, dataset.object_ids)
    hub_rdm = pdist(hub, "cosine")
    for task in range(dataset.task_count):
        selected = dataset.feature_tasks == task
        targets = dataset.values[:, selected]
        positives = targets.any(axis=1)
        task_input = torch.zeros(model.num_objects, dataset.task_count)
        task_input[:, task] = 1
        with torch.no_grad():
            dependent = model.get_context_dependent_rep([objects, task_input]).numpy()
        feature_rdm = (
            pdist(targets, "cosine")
            if selected.any()
            else np.full(len(hub_rdm), np.nan)
        )
        dependent_rdm = pdist(dependent, "cosine")
        common = (
            np.isfinite(feature_rdm) & np.isfinite(hub_rdm) & np.isfinite(dependent_rdm)
        )
        independent = safe_correlation(feature_rdm[common], hub_rdm[common], "spearman")
        controlled = safe_correlation(
            feature_rdm[common], dependent_rdm[common], "spearman"
        )
        rows.append(
            {
                "task_id": task,
                "features": int(selected.sum()),
                "positive_objects": int(positives.sum()),
                "valid_pairs": int(common.sum()),
                "total_pairs": len(common),
                "independent_spearman": independent,
                "dependent_spearman": controlled,
                "dependent_minus_independent": controlled - independent,
                "inactive": task in dataset.inactive_tasks,
            }
        )
    return pd.DataFrame(rows)


def factorial_effects(per_seed, metric="paper_statistic", expected_sources=None):
    rows = []
    for (task, dataset, seed), group in per_seed.groupby(["task", "dataset", "seed"]):
        if group.duplicated(["semantic_source", "context_source"]).any():
            raise ValueError("Duplicate source pair in factorial effects")
        table = group.pivot(
            index="semantic_source", columns="context_source", values=metric
        )
        if expected_sources is not None:
            table = table.reindex(index=expected_sources, columns=expected_sources)
        if table.isna().any().any():
            rows.append(
                {
                    "task": task,
                    "dataset": dataset,
                    "seed": seed,
                    "metric": metric,
                    "status": "not_estimable_unbalanced_or_nonfinite",
                    "observed_cells": int(np.isfinite(table.values).sum()),
                    "required_cells": table.size,
                    "semantic_effect": np.nan,
                    "context_effect": np.nan,
                    "interaction": np.nan,
                }
            )
            continue
        grand, semantic, context = (
            table.values.mean(),
            table.mean(axis=1),
            table.mean(axis=0),
        )
        for s in table.index:
            for c in table.columns:
                rows.append(
                    {
                        "task": task,
                        "dataset": dataset,
                        "seed": seed,
                        "metric": metric,
                        "semantic_source": s,
                        "context_source": c,
                        "grand_mean": grand,
                        "semantic_effect": semantic[s] - grand,
                        "context_effect": context[c] - grand,
                        "interaction": table.loc[s, c]
                        - semantic[s]
                        - context[c]
                        + grand,
                        "inference": "Fixed-cohort descriptive decomposition within paired seed; cells are not independent datasets",
                    }
                )
    return pd.DataFrame(rows)


def behavior_config(root, config):
    behavior = json.loads((root / config["behavior_config"]).read_text())
    historical = json.loads((root / config["frozen_behavior_manifest"]).read_text())[
        "config"
    ]
    if behavior != historical:
        raise ValueError(
            "Behavior protocol must equal the frozen original evaluation configuration"
        )
    return behavior


def evaluate_semantics(root, output, conditions, config):
    verify_history(root, config)
    if not config["semantic"]["schedule_locked"]:
        raise ValueError(
            "Lock training-only settings before inspecting human semantic judgments"
        )
    behavior = behavior_config(root, config)
    data_dir = root / "upstream/IntegratedSemanticsControlContextInference/data"
    tasks = load_tasks(data_dir)
    names = list(load_matrix(matrix_path(root, "human")).index)
    categories = (
        pd.read_csv(data_dir / "leuven_dataset/leuven_categories.csv")
        .set_index("Name")["Category 1"]
        .reindex(names)
    )
    if categories.isna().any():
        raise ValueError("Original category labels do not cover the object IDs")
    sources = list(dict.fromkeys(["human", *conditions]))
    models, checkpoints, hashes = {}, {}, {}
    for condition in sources:
        for seed in range(4):
            directory = output / "isc_models" / condition / f"seed_{seed}"
            model, checkpoint, manifest = load_run(directory)
            if (
                checkpoint["object_ids"] != names
                or checkpoint["signature"]["training"] != config["semantic"]["full"]
                or checkpoint["signature"]["condition"] != condition
                or checkpoint["signature"]["seed"] != seed
            ):
                raise ValueError(
                    "Semantic checkpoint ordering or locked training settings differ"
                )
            if condition == "human" and not checkpoint["result"]["converged"]:
                raise ValueError(
                    "Human semantic reference did not learn its own environment"
                )
            models[condition, seed], checkpoints[condition, seed] = model, checkpoint
            hashes[f"{condition}_seed_{seed}"] = sha256(directory / "checkpoint.pt")
    destination = output / "evaluations/semantic"
    signature = {
        "checkpoints": hashes,
        "code": code_hashes(),
        "config_sha256": digest(config),
    }
    previous_path = destination / "manifest.json"
    if previous_path.exists():
        previous = json.loads(previous_path.read_text())
        if previous["signature"] == signature and all(
            sha256(destination / p) == h for p, h in previous["outputs"].items()
        ):
            return previous
    destination.mkdir(parents=True, exist_ok=True)
    raw_rdms, raw_items = {}, []
    human_matrix = load_matrix(matrix_path(root, "human"), names)
    human_raw_rdm = pdist(human_matrix.values, "cosine")
    raw_geometry = []
    for condition in CONDITIONS:
        matrix = load_matrix(matrix_path(root, condition), names)
        raw_rdms[condition] = pdist(matrix.values, "cosine")
        raw_geometry.append(
            {
                "condition": condition,
                **rdm_statistics(raw_rdms[condition], human_raw_rdm)["all"],
            }
        )
        raw_items.append(
            similarity_predictions(
                matrix.values,
                names,
                tasks["similarity"],
                condition,
                0,
                "raw_feature_matrix",
                behavior["analysis"]["cross_validation_folds"],
            )
        )
    embeddings = {key: extract(model, names) for key, model in models.items()}
    rdms = {key: pdist(values, "cosine") for key, values in embeddings.items()}
    representations, semantic_items, control, transfer, stability = [], [], [], [], []
    for (condition, seed), model in models.items():
        semantic_items.append(
            similarity_predictions(
                embeddings[condition, seed],
                names,
                tasks["similarity"],
                condition,
                seed,
                "independently_pretrained_semantic_activations",
                behavior["analysis"]["cross_validation_folds"],
            )
        )
        references = [
            rdms["human", other]
            for other in range(4)
            if condition != "human" or other != seed
        ]
        for scope in ["all", "within", "between"]:
            comparisons = [
                rdm_statistics(rdms[condition, seed], reference, categories.values)[
                    scope
                ]
                for reference in references
            ]
            representations.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "scope": scope,
                    "semantic_rdm_spearman": float(
                        finite_mean([r["spearman"] for r in comparisons])
                    ),
                    "valid_pairs": comparisons[0]["valid_pairs"],
                    "reference_seeds": len(references),
                }
            )
        matrix = load_matrix(matrix_path(root, condition), names)
        assignments = dict(
            zip(
                checkpoints[condition, seed]["feature_ids"],
                checkpoints[condition, seed]["feature_tasks"],
            )
        )
        task_results = task_geometry(model, TaskDataset(matrix, assignments, 32))
        task_results["condition"], task_results["seed"] = condition, seed
        control.append(task_results)
        for context, raw_rdm in raw_rdms.items():
            transfer.append(
                {
                    "semantic_source": condition,
                    "feature_source": context,
                    "seed": seed,
                    **rdm_statistics(rdms[condition, seed], raw_rdm)["all"],
                }
            )
    for condition in sources:
        for a, b in combinations(range(4), 2):
            stability.append(
                {
                    "condition": condition,
                    "seed_a": a,
                    "seed_b": b,
                    **rdm_statistics(rdms[condition, a], rdms[condition, b])["all"],
                }
            )
    raw_items, semantic_items = pd.concat(raw_items), pd.concat(semantic_items)
    frames = {
        "raw_similarity_items.csv": raw_items,
        "semantic_similarity_items.csv": semantic_items,
        "raw_geometry.csv": pd.DataFrame(raw_geometry),
        "representation_per_seed.csv": pd.DataFrame(representations),
        "task_geometry.csv": pd.concat(control),
        "feature_space_transfer.csv": pd.DataFrame(transfer),
        "cross_seed_stability.csv": pd.DataFrame(stability),
    }
    for prefix, predictions in [("raw", raw_items), ("semantic", semantic_items)]:
        per_seed, summary = summarize_behavior(
            predictions,
            behavior["analysis"]["bootstrap_repetitions"],
            behavior["analysis"]["seed"],
        )
        (
            frames[f"{prefix}_similarity_per_seed.csv"],
            frames[f"{prefix}_similarity_summary.csv"],
        ) = (per_seed, summary)
    for name, frame in frames.items():
        frame.to_csv(destination / name, index=False)
    np.savez_compressed(
        destination / "rdms.npz",
        **{f"raw__{c}": r for c, r in raw_rdms.items()},
        **{f"semantic__{c}__seed_{s}": r for (c, s), r in rdms.items()},
    )
    atomic_json(destination / "object_ids.json", names)
    manifest = {
        "complete": True,
        "signature": signature,
        "conditions": sources,
        "human_training_gate_passed": True,
        "outcomes_may_not_change_training": True,
        "outputs": {
            name: sha256(destination / name)
            for name in [*frames, "rdms.npz", "object_ids.json"]
        },
    }
    atomic_json(previous_path, manifest)
    return manifest


SOURCE_CONTRASTS = [
    ("human", "random_human"),
    ("v2", "human"),
    ("v3_1_A", "v3_A"),
    ("v3_1_B", "v3_B"),
    ("v3_1_C", "v3_C"),
    ("v4_b_calibrated", "v3_1_B"),
    ("v4_ensemble_calibrated", "v4_ensemble_source_only"),
    ("v4_ensemble_calibrated", "v2"),
    ("v4_ensemble_calibrated", "human"),
    ("v4_b_calibrated", "v4_b_locked_v2"),
    ("v4_ensemble_calibrated", "v4_ensemble_locked_v2"),
    ("v4_ensemble_calibrated", "v4_b_calibrated"),
]


def contrast_design():
    rows = []
    for left, right in SOURCE_CONTRASTS:
        rows.append(
            {
                "axis": "matched_pipeline",
                "held_source": "diagonal",
                "left": f"{left}__{left}",
                "right": f"{right}__{right}",
            }
        )
        for held in CONDITIONS:
            rows.extend(
                [
                    {
                        "axis": "semantic",
                        "held_source": held,
                        "left": f"{left}__{held}",
                        "right": f"{right}__{held}",
                    },
                    {
                        "axis": "context",
                        "held_source": held,
                        "left": f"{held}__{left}",
                        "right": f"{held}__{right}",
                    },
                ]
            )
    return pd.DataFrame(rows)


def evaluate_context_models(root, output, jobs, config, phase):
    from iscci_validation.behavior_fidelity import item_fidelity
    from iscci_validation.evaluation import (
        make_evaluation_contexts,
        evaluate_model,
        compare_evaluations,
    )
    from .transfer_embeddings import frozen_embedding, load_context

    verify_history(root, config)
    behavior = behavior_config(root, config)
    destination = output / f"evaluations/context/phase_{phase}"
    cache = output / "evaluations/context"
    tasks = load_tasks(
        root / "upstream/IntegratedSemanticsControlContextInference/data"
    )
    human = load_matrix(matrix_path(root, "human"))
    names = list(human.index)
    evaluation = behavior["evaluation"]
    probes = make_evaluation_contexts(
        len(names),
        evaluation["pair_contexts"],
        evaluation["rdm_contexts"],
        evaluation["context_dependent_rdm_contexts"],
        evaluation["seed"],
    )
    destination.mkdir(parents=True, exist_ok=True)
    semantic_manifest = json.loads(
        (output / "evaluations/semantic/manifest.json").read_text()
    )
    if semantic_manifest.get("complete") is not True or semantic_manifest["signature"][
        "config_sha256"
    ] != digest(config):
        raise ValueError(
            "Current Stage 1 semantic evaluation must precede downstream evaluation"
        )
    signature = {
        "code": code_hashes(),
        "behavior_config": behavior,
        "semantic_stage": digest(semantic_manifest),
    }
    protocol_path = cache / "protocol.json"
    if (
        protocol_path.exists()
        and json.loads(protocol_path.read_text())["behavior_config"] != behavior
    ):
        raise ValueError(
            "Completed predictions cannot be rewritten under a different behavioral protocol"
        )
    atomic_json(protocol_path, signature)
    reference_evaluations = []
    for seed in range(4):
        semantic, _, _ = load_run(output / "isc_models/human" / f"seed_{seed}")
        embedding = frozen_embedding(extract(semantic, names), names, names)
        directory = output / "iscci_models/human__human" / f"seed_{seed}"
        model, _, manifest = load_context(
            directory,
            human,
            embedding,
            expected_identity={
                "semantic_source": "human",
                "context_source": "human",
                "seed": seed,
            },
        )
        if not manifest["complete"] or manifest["diagnostic"]:
            raise ValueError("Four full human/human reference models are required")
        reference_evaluations.append(evaluate_model(model, probes))
    all_items, checks, representations, completed, blocked = [], [], [], [], []
    for job in jobs:
        directory = Path(job["directory"])
        if not (directory / "manifest.json").exists():
            blocked.append({**job, "reason": "not trained"})
            continue
        semantic_source, context_source, seed = (
            job["semantic_source"],
            job["context_source"],
            job["seed"],
        )
        semantic, sem_checkpoint, sem_manifest = load_run(
            output / "isc_models" / semantic_source / f"seed_{seed}"
        )
        matrix = load_matrix(matrix_path(root, context_source), names)
        embedding = frozen_embedding(extract(semantic, names), names, names)
        model, checkpoint, manifest = load_context(
            directory,
            matrix,
            embedding,
            expected_identity={
                "semantic_source": semantic_source,
                "context_source": context_source,
                "seed": seed,
            },
        )
        if not manifest["complete"] or manifest["diagnostic"]:
            blocked.append({**job, "reason": "incomplete or diagnostic checkpoint"})
            continue
        label = f"{semantic_source}__{context_source}"
        check = item_fidelity(
            model,
            tasks,
            names,
            root / "upstream/IntegratedSemanticsControlContextInference",
            behavior["fidelity"]["item_absolute_tolerance"],
        )
        check["condition"], check["seed"] = label, seed
        checks.append(check)
        if not check["passed"].all():
            pd.concat(checks).to_csv(destination / "graph_fidelity.csv", index=False)
            raise ValueError(
                "Original model/scorer parity failed before human prediction"
            )
        items = scoring_functions()["predict"](
            model,
            label,
            seed,
            tasks,
            names,
            behavior,
            cache,
            directory / "checkpoint.pt",
        )
        items["semantic_source"], items["context_source"] = (
            semantic_source,
            context_source,
        )
        all_items.append(items)
        candidate = evaluate_model(model, probes)
        np.savez_compressed(
            cache / f"{label}_seed_{seed}_representations.npz", **candidate
        )
        comparisons = [
            compare_evaluations(candidate, ref)
            for n, ref in enumerate(reference_evaluations)
            if label != "human__human" or n != seed
        ]
        representations.append(
            {
                "condition": label,
                "semantic_source": semantic_source,
                "context_source": context_source,
                "seed": seed,
                **{
                    metric: finite_mean([r[metric] for r in comparisons])
                    for metric in comparisons[0]
                },
            }
        )
        completed.append(
            {"id": job["id"], "checkpoint_sha256": sha256(directory / "checkpoint.pt")}
        )
    if not all_items:
        raise ValueError(
            "No complete downstream models available for the selected phase"
        )
    predictions = pd.concat(all_items, ignore_index=True)
    per_seed, summary = summarize_behavior(
        predictions,
        behavior["analysis"]["bootstrap_repetitions"],
        behavior["analysis"]["seed"],
    )
    labels = predictions[
        ["condition", "semantic_source", "context_source"]
    ].drop_duplicates()
    per_seed = per_seed.merge(labels, on="condition", validate="many_to_one")
    design = contrast_design()
    counts = per_seed.groupby("condition")["seed"].nunique()
    available = set(counts[counts == 4].index)
    ready = design[design["left"].isin(available) & design["right"].isin(available)]
    pairs = list(
        dict.fromkeys(ready[["left", "right"]].itertuples(index=False, name=None))
    )
    contrasts = paired_condition_contrasts(
        predictions,
        pairs,
        behavior["analysis"]["bootstrap_repetitions"],
        behavior["analysis"]["seed"],
    )
    frames = {
        "all_item_predictions.csv": predictions,
        "behavior_per_seed.csv": per_seed,
        "behavior_summary.csv": summary,
        "graph_fidelity.csv": pd.concat(checks),
        "representation_per_seed.csv": pd.DataFrame(representations),
        "paired_contrasts.csv": contrasts,
        "contrast_design.csv": design,
        "blocked_contrasts.csv": design.loc[~design.index.isin(ready.index)],
        "factorial_effects_per_seed.csv": factorial_effects(
            per_seed, expected_sources=list(CONDITIONS)
        ),
        "blocked_models.csv": pd.DataFrame(blocked),
    }
    frozen_humans = root / config["frozen_behavior_manifest"]
    for name in [
        "human_reliability.csv",
        "human_participant_reliability.csv",
        "paper_human_choice_statistics.csv",
    ]:
        frames[name] = pd.read_csv(frozen_humans.parent / name)
    for (condition, seed), selected in predictions[
        predictions["task"] == "phenomena"
    ].groupby(["condition", "seed"]):
        result = phenomenon_statistics(selected)
        result["condition"], result["seed"] = condition, seed
        name = f"phenomena_{condition}_seed_{seed}.csv"
        frames[name] = result
    for name, frame in frames.items():
        frame.to_csv(destination / name, index=False)
    manifest = {
        "complete": not blocked,
        "phase": phase,
        "signature": signature,
        "models": completed,
        "planned_models": len(jobs),
        "blocked_models": len(blocked),
        "outputs": {name: sha256(destination / name) for name in frames},
    }
    atomic_json(destination / "manifest.json", manifest)
    return manifest
