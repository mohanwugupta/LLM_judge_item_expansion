#!/usr/bin/env python3
"""Gated external behavioral validation, with item-level predictions and provenance."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import pdist
from scipy.stats import t as student_t

from evaluate_validation import released_model
from iscci_validation.behavior import (
    TASK_FILES,
    cross_validated_mapping,
    load_tasks,
    metrics_for_items,
    observed_paper_choices,
    paired_condition_contrasts,
    phenomenon_statistics,
    predict_task,
    reliability_tables,
    safe_correlation,
    summarize_behavior,
    trial_statistics,
)
from iscci_validation.behavior_fidelity import fidelity_gate, item_fidelity
from iscci_validation.dataio import load_human_matrix, load_v4_matrix, matrix_sha256
from iscci_validation.evaluation import (
    _pearson_from_ranks,
    _ranked_cosine_rdm,
    compare_evaluations,
    evaluate_model,
    make_evaluation_contexts,
)
from iscci_validation.provenance import environment_record, sha256_file, write_json
from iscci_validation.training import load_trained_model, save_checkpoint, train_model

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
UPSTREAM = ROOT / "upstream/IntegratedSemanticsControlContextInference"
DATA = UPSTREAM / "data"
EXPECTED_TRAINING = {
    "seeds": [0, 1, 2, 3],
    "epochs": 400,
    "episodes_per_epoch": 1024,
    "batch_size": 128,
    "support_sizes": [1, 2],
    "learning_rate": 0.001,
    "task_loss_weight": 0.5,
    "freeze_released_semantic_embedding": True,
}


def resolve(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (PROJECT / path).resolve()


def input_geometry(matrix, human):
    distances = pdist(matrix.values, "cosine")
    return {
        "input_object_rdm_spearman": _pearson_from_ranks(
            _ranked_cosine_rdm(matrix.values.astype(np.float32)),
            _ranked_cosine_rdm(human.values.astype(np.float32)),
        ),
        "input_object_rdm_defined_pairs_spearman": safe_correlation(
            distances, pdist(human.values, "cosine"), "spearman"
        ),
        "input_objects_without_features": int((matrix.sum(axis=1) == 0).sum()),
        "input_rdm_undefined_pairs": int((~np.isfinite(distances)).sum()),
        "input_rdm_primary_convention": "original validation ranks: undefined cosine distance -> 0; defined-pair sensitivity also reported",
    }


def registry(directories, conditions, human, extra_models):
    matrices = {"human": human.astype(np.int8)}
    models = {}
    for directory in directories:
        if not directory.exists():
            continue
        for path in sorted((directory / "data_matrices").glob("*.csv")):
            if path.stem not in conditions:
                continue
            matrix = pd.read_csv(path, index_col=0)
            if list(matrix.index) != list(human.index):
                raise ValueError(f"Object order mismatch: {path}")
            if not np.isin(matrix.values, [0, 1]).all():
                raise ValueError(f"Nonbinary matrix: {path}")
            matrix = matrix.astype(np.int8)
            if path.stem in matrices and matrix_sha256(
                matrices[path.stem]
            ) != matrix_sha256(matrix):
                raise ValueError(f"Conflicting matrices for {path.stem}")
            matrices[path.stem] = matrix
        for path in sorted((directory / "models").glob("*/seed_*/checkpoint.pt")):
            condition = path.parents[1].name
            if condition not in conditions:
                continue
            seed = int(path.parent.name.removeprefix("seed_"))
            key = (condition, seed)
            if key in models and sha256_file(models[key]) != sha256_file(path):
                raise ValueError(f"Conflicting checkpoints for {key}")
            models[key] = path
    for condition in conditions:
        for seed in range(4):
            path = extra_models / condition / f"seed_{seed}/checkpoint.pt"
            if path.exists():
                if (condition, seed) in models and sha256_file(
                    models[(condition, seed)]
                ) != sha256_file(path):
                    raise ValueError(
                        f"Conflicting extra checkpoint for {(condition, seed)}"
                    )
                models[(condition, seed)] = path
    return matrices, models


def validate_checkpoint(path, condition, seed, matrix, embedding, training):
    model, checkpoint = load_trained_model(path)
    expected = {k: v for k, v in training.items() if k != "seeds"}
    if checkpoint.get("condition") != condition or checkpoint.get("seed") != seed:
        raise ValueError(f"Checkpoint identity mismatch: {path}")
    if (
        checkpoint.get("epoch") != 400
        or checkpoint.get("training_parameters") != expected
    ):
        raise ValueError(f"Checkpoint is not a complete frozen-protocol run: {path}")
    if checkpoint.get("object_names") != list(matrix.index):
        raise ValueError(f"Checkpoint stimulus order mismatch: {path}")
    if checkpoint.get("matrix_sha256") != matrix_sha256(matrix):
        raise ValueError(f"Checkpoint matrix mismatch: {path}")
    dimensions = checkpoint["model_params"]
    if any(
        dimensions[k] != value
        for k, value in (
            ("input_d", 294),
            ("embedding_d", 64),
            ("context_d", 128),
            ("context_dependent_d", 128),
        )
    ):
        raise ValueError(f"Checkpoint architecture mismatch: {path}")
    if dimensions["task_output_d"] != matrix.shape[1]:
        raise ValueError(f"Checkpoint task head mismatch: {path}")
    if not torch.equal(model.input_to_independent.weight.detach(), embedding):
        raise ValueError(f"Frozen semantic activations differ: {path}")
    return model


def run_predictions(
    model, condition, seed, tasks, names, config, output, checkpoint_path
):
    destination = output / "per_model" / f"{condition}_seed_{seed}"
    destination.mkdir(parents=True, exist_ok=True)
    implementation = [
        ROOT / "run_human_behavior_validation.py",
        ROOT / "iscci_validation/behavior.py",
        ROOT / "iscci_validation/simulations.py",
        ROOT / "iscci_validation/modeling.py",
    ]
    signature = {
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "protocol": config,
        "data_sha256": {t: sha256_file(DATA / p) for t, p in TASK_FILES.items()},
        "plural_mapping_sha256": sha256_file(
            DATA / "leuven_dataset/leuven_singular_to_plural.csv"
        ),
        "implementation_sha256": {p.name: sha256_file(p) for p in implementation},
        "object_names": names,
    }
    manifest = destination / "manifest.json"
    previous = json.loads(manifest.read_text()) if manifest.exists() else {}
    required_outputs = [
        f"{task}_{level}.csv" for task in tasks for level in ("items", "trials")
    ]
    required_outputs.append("trial_statistics.csv")
    required_outputs.append("paper_participant_predictions.csv")
    if previous.get("signature") == signature and all(
        (destination / name).exists()
        and previous.get("output_sha256", {}).get(name)
        == sha256_file(destination / name)
        for name in required_outputs
    ):
        return pd.concat(
            [pd.read_csv(destination / f"{task}_items.csv") for task in tasks],
            ignore_index=True,
        )
    all_items = []
    all_trials = []
    for task, data in tasks.items():
        items, trials = predict_task(
            task, data, model, names, int(config["evaluation"]["seed"])
        )
        # Predictions are written before any fitted measurement mapping or aggregation.
        for frame, level in ((items, "items"), (trials, "trials")):
            frame.insert(0, "seed", seed)
            frame.insert(0, "condition", condition)
            frame.insert(0, "model_id", f"{condition}_seed_{seed}")
            frame.to_csv(destination / f"{task}_{level}.csv", index=False)
        mapped = cross_validated_mapping(
            items, int(config["analysis"]["cross_validation_folds"])
        )
        mapped.to_csv(destination / f"{task}_items.csv", index=False)
        trials = trials.merge(
            mapped[["item_id", "cv_prediction", "cv_fold"]],
            on="item_id",
            how="left",
            validate="many_to_one",
        )
        trials.to_csv(destination / f"{task}_trials.csv", index=False)
        all_items.append(mapped)
        all_trials.append(trials)
    trial_statistics(pd.concat(all_trials, ignore_index=True)).to_csv(
        destination / "trial_statistics.csv", index=False
    )
    participant_rows = []
    for frame in all_trials:
        if frame["task"].iloc[0] not in ("thematic", "nonmonotonicity"):
            continue
        for participant, group in frame.groupby("participant_id"):
            participant_rows.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "task": group["task"].iloc[0],
                    "participant_id": participant,
                    "human_argument2_fraction": group["observed"].mean(),
                    "model_argument2_fraction": group["model_choice"].mean(),
                    "original_trials": len(group),
                }
            )
    pd.DataFrame(participant_rows).to_csv(
        destination / "paper_participant_predictions.csv", index=False
    )
    write_json(
        manifest,
        {
            "signature": signature,
            "output_sha256": {
                name: sha256_file(destination / name) for name in required_outputs
            },
        },
    )
    return pd.concat(all_items, ignore_index=True)


def simple_metrics(predictions):
    rows = []
    for (condition, seed, task, dataset), group in predictions.groupby(
        ["condition", "seed", "task", "dataset"]
    ):
        if task != "phenomena":
            rows.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "task": task,
                    "dataset": dataset,
                    **metrics_for_items(group),
                }
            )
    return pd.DataFrame(rows)


def load_ready_v4(matrices, conditions, matrix_dir):
    manifest_path = matrix_dir / "matrix_manifest.json"
    if not manifest_path.exists():
        return {}
    metadata = json.loads(manifest_path.read_text())
    required = (
        "candidate_inventory_hash",
        "judgment_manifest_sha256",
        "calibration_hash",
        "matrix_manifest_hash",
    )
    if any(key not in metadata for key in required):
        raise ValueError("Incomplete V4 matrix provenance")
    judgment_path = matrix_dir.parent / "judgments/judgment_manifest.json"
    if not judgment_path.exists() or not json.loads(judgment_path.read_text()).get(
        "complete"
    ):
        raise ValueError(
            "V4 judgments are not finalized; invalid cells may not be zero-filled"
        )
    if sha256_file(judgment_path) != metadata["judgment_manifest_sha256"]:
        raise ValueError("V4 matrix/judgment provenance mismatch")
    provenance = {**metadata, "matrix_manifest_sha256": sha256_file(manifest_path)}
    for condition in conditions:
        path = matrix_dir / f"{condition}.csv"
        if condition.startswith("v4_") and path.exists():
            matrix = load_v4_matrix(path, matrices["human"].index)
            if condition in matrices and matrix_sha256(matrix) != matrix_sha256(
                matrices[condition]
            ):
                raise ValueError(
                    f"Finalized V4 matrix differs from existing training: {condition}"
                )
            matrices[condition] = matrix
    return provenance


def train_missing(matrices, models, conditions, config, output, embedding, provenance):
    training = config["training"]
    parameters = {key: value for key, value in training.items() if key != "seeds"}
    for condition in conditions:
        if condition not in matrices:
            continue
        if "source_only" in condition and not any(c == condition for c, _ in models):
            continue
        for seed in training["seeds"]:
            if (condition, seed) in models:
                continue
            if condition.startswith("v4_") and not provenance:
                continue
            print(
                f"Training {condition} seed {seed} with the frozen 400-epoch protocol",
                flush=True,
            )
            model, metrics, _ = train_model(
                matrices[condition],
                embedding,
                seed,
                400,
                training["episodes_per_epoch"],
                training["batch_size"],
                training["learning_rate"],
                training["task_loss_weight"],
            )
            destination = output / "models" / condition / f"seed_{seed}"
            destination.mkdir(parents=True, exist_ok=True)
            metrics.to_csv(destination / "training_metrics.csv", index=False)
            path = destination / "checkpoint.pt"
            save_checkpoint(
                path,
                model,
                matrices[condition],
                condition,
                seed,
                400,
                parameters,
                provenance,
            )
            models[(condition, seed)] = path


def representation_table(matrices, models, conditions, embedding, config, output):
    names = list(matrices["human"].index)
    human_rdm = pdist(matrices["human"].values, "cosine")
    semantic_rdm = pdist(embedding[:-1].numpy(), "cosine")
    contexts = make_evaluation_contexts(
        len(names),
        int(config["evaluation"]["pair_contexts"]),
        int(config["evaluation"]["rdm_contexts"]),
        int(config["evaluation"]["context_dependent_rdm_contexts"]),
        int(config["evaluation"]["seed"]),
    )
    np.savez(output / "representation_probes.npz", **contexts)
    evaluations = {}
    for (condition, seed), path in models.items():
        if condition not in matrices or condition not in conditions:
            continue
        model = validate_checkpoint(
            path, condition, seed, matrices[condition], embedding, config["training"]
        )
        evaluations[(condition, seed)] = evaluate_model(model, contexts)
    rows = []
    for condition in conditions:
        if condition not in matrices:
            continue
        geometry = input_geometry(matrices[condition], matrices["human"])
        for seed in range(4):
            if (condition, seed) not in evaluations:
                continue
            comparisons = [
                compare_evaluations(
                    evaluations[(condition, seed)], evaluations[("human", s)]
                )
                for s in range(4)
                if not (condition == "human" and s == seed)
            ]
            row = {
                "condition": condition,
                "seed": seed,
                **geometry,
                "semantic_activation_rdm_agreement": 1.0,
                "semantic_activation_rdm_vs_leuven": safe_correlation(
                    semantic_rdm, human_rdm, "spearman"
                ),
                "semantic_activation_note": "shared frozen human-trained embeddings; not independently learned from LLM features",
                "features": matrices[condition].shape[1],
            }
            for metric in comparisons[0]:
                row[metric] = float(np.mean([c[metric] for c in comparisons]))
            rows.append(row)
    return pd.DataFrame(rows)


def geometry_behavior_association(summary, representations, config):
    geometry = representations.groupby("condition")["input_object_rdm_spearman"].mean()
    rows = []
    rng = np.random.default_rng(config["analysis"]["seed"])
    for (task, dataset, metric), group in summary.groupby(
        ["task", "dataset", "metric"]
    ):
        if metric not in ("paper_statistic", "pearson_r", "spearman_r"):
            continue
        # Conditions are observations here; four seeds are not four independent matrices.
        paired = group.set_index("condition")[["mean"]].join(geometry).dropna()
        paired = paired.drop(index=["human", "released"], errors="ignore")
        x, y = paired["input_object_rdm_spearman"].values, paired["mean"].values
        r = safe_correlation(x, y, "spearman")
        permutations = (
            [
                safe_correlation(x, rng.permutation(y), "spearman")
                for _ in range(config["analysis"]["geometry_permutations"])
            ]
            if np.isfinite(r)
            else []
        )
        p = (
            (1 + sum(abs(v) >= abs(r) for v in permutations)) / (1 + len(permutations))
            if permutations
            else np.nan
        )
        rows.append(
            {
                "task": task,
                "dataset": dataset,
                "metric": metric,
                "conditions": len(paired),
                "spearman_r": r,
                "pearson_r": safe_correlation(x, y),
                "permutation_p": p,
                "scope": "exploratory condition-level association; related conditions are not independent replications; no feature or training selection",
            }
        )
    return pd.DataFrame(rows)


def summary_table(
    conditions,
    matrices,
    models,
    summary,
    representations,
    blocked,
    lower_summary=None,
    reliability=None,
):
    rows = []
    for condition in conditions:
        seeds = [s for s in range(4) if (condition, s) in models]
        row = {
            "condition": condition,
            "seed_count": len(seeds),
            "matrix_available": condition in matrices,
            "features": (
                matrices[condition].shape[1] if condition in matrices else np.nan
            ),
            "status": (
                "blocked_by_fidelity_gate"
                if blocked
                else (
                    "evaluated"
                    if len(seeds) == 4
                    else "missing_or_incomplete_condition"
                )
            ),
        }
        if condition in matrices:
            row.update(input_geometry(matrices[condition], matrices["human"]))
        if len(representations):
            rep = representations[representations["condition"] == condition]
            for column in (
                "semantic_activation_rdm_agreement",
                "semantic_activation_rdm_vs_leuven",
                "context_rdm_spearman",
                "context_dependent_rdm_spearman_median",
                "membership_logit_spearman",
            ):
                row[column + "_mean"] = rep[column].mean()
                row[column + "_seed_sd"] = rep[column].std()
                error = (
                    student_t.ppf(0.975, len(rep) - 1)
                    * rep[column].std()
                    / np.sqrt(len(rep))
                    if len(rep) > 1
                    else np.nan
                )
                row[column + "_seed_ci_low"] = rep[column].mean() - error
                row[column + "_seed_ci_high"] = rep[column].mean() + error
        if len(summary):
            for _, result in summary[
                (summary["condition"] == condition)
                & summary["metric"].isin(
                    [
                        "paper_statistic",
                        "pearson_r",
                        "spearman_r",
                        "cv_rmse",
                        "directional_agreement",
                        "paper_asymmetry_r_squared",
                        "direct_rmse",
                    ]
                )
            ].iterrows():
                label = f"{result['task']}:{result['dataset']}:{result['metric']}"
                for field in ("mean", "seed_sd", "ci_low", "ci_high"):
                    row[f"{label}:{field}"] = result[field]
        if lower_summary is not None and len(lower_summary):
            for _, result in lower_summary[
                (lower_summary["base_condition"] == condition)
                & (lower_summary["metric"] == "paper_statistic")
            ].iterrows():
                label = f"{result['level']}:{result['dataset']}"
                for field in ("mean", "ci_low", "ci_high"):
                    row[f"{label}:{field}"] = result[field]
        rows.append(row)
    table = pd.DataFrame(rows)
    if reliability is not None:
        for _, result in reliability.iterrows():
            if result["status"] == "estimated_from_original_participants":
                label = f"human_reliability:{result['task']}:{result['dataset']}"
                for field in (
                    "spearman_brown_reliability_mean",
                    "spearman_brown_low",
                    "spearman_brown_high",
                ):
                    table[f"{label}:{field}"] = result[field]
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/human_behavior_validation.json"
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--fidelity-only", action="store_true")
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Do not train missing seeds after the gate passes",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if config["training"] != EXPECTED_TRAINING:
        raise ValueError(
            "External tests must use the unchanged frozen training protocol"
        )
    output = (
        args.output_dir.resolve()
        if args.output_dir
        else resolve(config["output_directory"])
    )
    output.mkdir(parents=True, exist_ok=True)
    conditions = config["required_conditions"] + config["optional_conditions"]
    _, human = load_human_matrix(
        DATA / "leuven_dataset/leuven_combined_features_consolidated.csv"
    )
    directories = [resolve(path) for path in config["validation_directories"]]
    matrices, models = registry(directories, conditions, human, output / "models")
    provenance = load_ready_v4(matrices, conditions, PROJECT / "artifacts/v4/matrices")
    tasks = load_tasks(DATA)
    names = list(human.index.astype(str))
    released_path = UPSTREAM / "models/1and2shot_isc-seed3.pt"
    released = released_model(released_path)
    embedding = released.input_to_independent.weight.detach().clone()
    torch.set_num_threads(1)
    gate_predictions, item_checks = [], []
    for condition, seed, path in [
        ("released", seed, UPSTREAM / "models" / f"1and2shot_isc-seed{seed}.pt")
        for seed in range(4)
    ] + [
        ("human", s, models[("human", s)]) for s in range(4) if ("human", s) in models
    ]:
        print(f"Fidelity evaluation: {condition} seed {seed}", flush=True)
        model = (
            released_model(path)
            if condition == "released"
            else validate_checkpoint(
                path, condition, seed, human, embedding, config["training"]
            )
        )
        check = item_fidelity(
            model, tasks, names, UPSTREAM, config["fidelity"]["item_absolute_tolerance"]
        )
        check["condition"], check["seed"] = condition, seed
        item_checks.append(check)
        gate_predictions.append(
            run_predictions(model, condition, seed, tasks, names, config, output, path)
        )
    gate_predictions = pd.concat(gate_predictions, ignore_index=True)
    item_checks = pd.concat(item_checks, ignore_index=True)
    gate_metrics = simple_metrics(gate_predictions)
    item_checks.to_csv(output / "fidelity_item_probe_checks.csv", index=False)
    gate_metrics.to_csv(output / "fidelity_human_per_seed.csv", index=False)
    checks, gate = fidelity_gate(
        UPSTREAM, gate_metrics, item_checks, config["fidelity"]
    )
    checks.to_csv(output / "fidelity_statistic_checks.csv", index=False)
    write_json(output / "fidelity_gate.json", gate)
    reliability, participants = reliability_tables(
        tasks, config["analysis"]["reliability_repetitions"], config["analysis"]["seed"]
    )
    reliability.to_csv(output / "human_reliability.csv", index=False)
    participants.to_csv(output / "human_participant_reliability.csv", index=False)
    human_participants, human_statistics = observed_paper_choices(tasks)
    human_participants.to_csv(
        output / "paper_human_participant_choices.csv", index=False
    )
    human_statistics.to_csv(output / "paper_human_choice_statistics.csv", index=False)
    behavioral_summary = pd.DataFrame()
    representations = pd.DataFrame()
    lower_summary = pd.DataFrame()
    if gate["passed"] and not args.fidelity_only:
        if not args.evaluate_only:
            train_missing(
                matrices, models, conditions, config, output, embedding, provenance
            )
        predictions = [gate_predictions[gate_predictions["condition"] == "human"]]
        for condition in conditions:
            if condition == "human" or condition not in matrices:
                continue
            if any((condition, s) not in models for s in range(4)):
                continue
            for seed in range(4):
                path = models[(condition, seed)]
                print(
                    f"External behavioral evaluation: {condition} seed {seed}",
                    flush=True,
                )
                model = validate_checkpoint(
                    path,
                    condition,
                    seed,
                    matrices[condition],
                    embedding,
                    config["training"],
                )
                predictions.append(
                    run_predictions(
                        model, condition, seed, tasks, names, config, output, path
                    )
                )
        predictions = pd.concat(predictions, ignore_index=True)
        predictions.to_csv(output / "all_item_predictions.csv", index=False)
        individual_statistics = []
        evaluated_ids = {
            f"{condition}_seed_{int(seed)}"
            for condition, seed in predictions[["condition", "seed"]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        } | {f"released_seed_{seed}" for seed in range(4)}
        for model_id in sorted(evaluated_ids):
            table = pd.read_csv(
                output / "per_model" / model_id / "trial_statistics.csv"
            )
            table["model_id"] = model_id
            individual_statistics.append(table)
        pd.concat(individual_statistics, ignore_index=True).to_csv(
            output / "all_individual_trial_statistics.csv", index=False
        )
        print("Bootstrapping human-item and model-seed uncertainty", flush=True)
        per_seed, behavioral_summary = summarize_behavior(
            predictions,
            config["analysis"]["bootstrap_repetitions"],
            config["analysis"]["seed"],
        )
        per_seed.to_csv(output / "behavior_per_seed.csv", index=False)
        behavioral_summary.to_csv(output / "behavior_summary.csv", index=False)
        phenomena = []
        for (condition, seed), group in predictions[
            predictions["task"] == "phenomena"
        ].groupby(["condition", "seed"]):
            stats = phenomenon_statistics(group)
            stats["condition"], stats["seed"] = condition, seed
            phenomena.append(stats)
        pd.concat(phenomena, ignore_index=True).to_csv(
            output / "paper_phenomenon_tests.csv", index=False
        )
        v4s = [c for c in conditions if c.startswith("v4_") and "source_only" not in c]
        pairs = [
            (v4, reference)
            for v4 in v4s
            for reference in (
                "v3_1_B",
                "v2",
                "human",
                "v4_source_only",
                "v4_b_source_only",
                "v4_ensemble_source_only",
            )
        ]
        pairs += [
            ("v4_b_calibrated", "v4_b_locked_v2"),
            ("v4_ensemble_calibrated", "v4_ensemble_locked_v2"),
            ("v4_ensemble_locked_v2", "v4_b_locked_v2"),
            ("v4_ensemble_calibrated", "v4_b_calibrated"),
        ]
        paired_condition_contrasts(
            predictions,
            pairs,
            config["analysis"]["bootstrap_repetitions"],
            config["analysis"]["seed"],
        ).to_csv(output / "paired_condition_contrasts.csv", index=False)
        representations = representation_table(
            matrices, models, conditions, embedding, config, output
        )
        representations.to_csv(output / "representation_per_seed.csv", index=False)
        geometry_behavior_association(
            behavioral_summary, representations, config
        ).to_csv(output / "geometry_behavior_association.csv", index=False)
        sensitivity = representations.copy()
        sensitivity["input_object_rdm_spearman"] = sensitivity[
            "input_object_rdm_defined_pairs_spearman"
        ]
        geometry_behavior_association(behavioral_summary, sensitivity, config).to_csv(
            output / "geometry_behavior_defined_pairs_sensitivity.csv", index=False
        )
        # Levels 1/2 are restricted to the existing human similarity trials, not new tasks.
        similarity = tasks["similarity"]
        names_to_index = {name: i for i, name in enumerate(names)}
        lower_items, lower_trials = [], []
        from iscci_validation.behavior import item_table

        for condition, matrix in matrices.items():
            for level, values in (
                ("raw_feature_matrix", matrix.values),
                ("pretrained_semantic_activations", embedding[:-1].numpy()),
            ):
                item, keys = item_table("similarity", similarity)
                a = values[[names_to_index[v] for v in item["Premise 1"]]]
                b = values[[names_to_index[v] for v in item["Conclusion"]]]
                denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
                item["prediction"] = np.divide(
                    np.sum(a.astype(float) * b, axis=1),
                    denominator,
                    out=np.full(len(item), np.nan),
                    where=denominator > 0,
                )
                item["prediction_status"] = np.where(
                    denominator > 0, "defined", "undefined_zero_feature_vector"
                )
                item = cross_validated_mapping(
                    item, config["analysis"]["cross_validation_folds"]
                )
                item["condition"], item["level"] = condition, level
                lower_items.append(item)
                trial = similarity.merge(
                    item[
                        [
                            *keys,
                            "prediction",
                            "cv_prediction",
                            "cv_fold",
                            "item_id",
                            "cluster_id",
                            "prediction_status",
                        ]
                    ],
                    on=keys,
                    validate="many_to_one",
                )
                trial["condition"], trial["level"] = condition, level
                lower_trials.append(trial)
        lower_predictions = pd.concat(lower_items, ignore_index=True)
        lower_predictions.to_csv(
            output / "similarity_levels_1_2_items.csv", index=False
        )
        pd.concat(lower_trials, ignore_index=True).to_csv(
            output / "similarity_levels_1_2_trials.csv", index=False
        )
        lower_predictions["condition"] = (
            lower_predictions["condition"] + ":" + lower_predictions["level"]
        )
        lower_predictions["seed"] = 0
        print(
            "Bootstrapping separate raw-feature and semantic-activation similarity levels",
            flush=True,
        )
        _, lower_summary = summarize_behavior(
            lower_predictions,
            config["analysis"]["bootstrap_repetitions"],
            config["analysis"]["seed"],
        )
        lower_summary[["base_condition", "level"]] = lower_summary[
            "condition"
        ].str.split(":", expand=True)
        lower_summary.to_csv(output / "similarity_levels_1_2_summary.csv", index=False)
    summary_table(
        conditions,
        matrices,
        models,
        behavioral_summary,
        representations,
        not gate["passed"],
        lower_summary,
        reliability,
    ).to_csv(output / "condition_summary.csv", index=False)
    complete = (
        gate["passed"]
        and not args.fidelity_only
        and all(
            c in matrices and (c, s) in models
            for c in config["required_conditions"]
            for s in range(4)
        )
    )
    manifest = {
        "protocol_version": config["protocol_version"],
        "complete": complete,
        "status": (
            "blocked_fidelity"
            if not gate["passed"]
            else (
                "fidelity_only"
                if args.fidelity_only
                else "complete" if complete else "partial_missing_conditions"
            )
        ),
        "fidelity_gate": gate,
        "config": config,
        "config_sha256": sha256_file(args.config),
        "human_datasets": {
            task: {
                "path": str(DATA / file),
                "sha256": sha256_file(DATA / file),
                "observations": len(tasks[task]),
            }
            for task, file in TASK_FILES.items()
        },
        "paper_sha256": sha256_file(PROJECT / "ISCCI_2026Jan_ArXiv.pdf"),
        "released_checkpoint_sha256": {
            str(seed): sha256_file(UPSTREAM / "models" / f"1and2shot_isc-seed{seed}.pt")
            for seed in range(4)
        },
        "notebook_sha256": {p.name: sha256_file(p) for p in UPSTREAM.glob("*.ipynb")},
        "checkpoint_sha256": {
            f"{c}_seed_{s}": sha256_file(path) for (c, s), path in models.items()
        },
        "matrix_sha256": {
            condition: matrix_sha256(matrix) for condition, matrix in matrices.items()
        },
        "implementation_sha256": {
            path.name: sha256_file(path)
            for path in [
                Path(__file__),
                *sorted((ROOT / "iscci_validation").glob("*.py")),
            ]
        },
        "missing_required_seeds": [
            f"{c}_seed_{s}"
            for c in config["required_conditions"]
            for s in range(4)
            if (c, s) not in models
        ],
        "v4_matrix_provenance": provenance,
        "environment": environment_record(),
        "notes": [
            "Original model-selected stimuli are unchanged; their selection limits claims of independent external generalization.",
            "Original human-defined paper subsets are fixed across model conditions; full-item supplemental results are also saved.",
            "Phenomena CSV contains synthetic arguments, not item-level human responses; it is not a human-judgment benchmark.",
            "One common stochastic LCA seed is used across ALL conditions AND training seeds.",
            "No feature judgments, thresholds, matrices, or training settings are chosen from behavioral outcomes.",
        ],
    }
    write_json(output / "manifest.json", manifest)
    failure_rows = checks[checks["required"] & ~checks["passed"]]
    status_lines = [
        "# Human-Behavior Validation Status",
        "",
        f"Status: `{manifest['status']}`.",
        "",
        f"Original-code item probes: {'PASS' if gate['original_item_probe_checks_passed'] else 'FAIL'}.",
        f"Failed required statistic/source checks: {gate['failed_required_statistic_checks']}.",
        "",
        "Released deterministic absolute tolerance: 1e-5. Reconstructed human means must lie inside the 95% reference-seed prediction interval (four released human runs). This is replication consistency, not formal equivalence.",
        "",
        "## Fidelity Differences",
        "",
        "| Condition | Task / dataset | Reference | Observed | Absolute difference |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for _, row in failure_rows.iterrows():
        status_lines.append(
            f"| {row['condition']} | {row['task']} / {row['dataset']} ({row['metric']}) | {row['reference']:.6f} | {row['observed']:.6f} | {row['absolute_error']:.6f} |"
        )
    status_lines += [
        "",
        "## Interpretation",
        "",
        "Independent reconstructed seeds are assessed against released human-run variation, following the user's explicit statistical-replication decision. The initial strict numerical audit is preserved in the sibling human_behavior_validation directory.",
        "",
        "The PDF reports 36/38 context directions; the saved notebook reports 37/38. This is a stochastic-source discrepancy, not a feature-quality result.",
        "",
        (
            "Finalized V4 matrix provenance is available."
            if provenance
            else "Finalized V4 matrix provenance is not available; missing cells must not be filled with zeros to enable training."
        ),
        "",
        "`condition_summary.csv` records available/missing conditions. Behavioral comparisons, central contrast intervals, and geometry-behavior associations are not completed when the gate is blocked.",
        "",
        "Protocol, original measures, reliability, limitations, and resume commands: `../../HUMAN_BEHAVIOR_VALIDATION.md`.",
        "",
    ]
    (output / "STATUS.md").write_text("\n".join(status_lines), encoding="utf-8")
    print(
        f"Behavioral validation status: {manifest['status']}; outputs: {output}",
        flush=True,
    )
    if not gate["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
