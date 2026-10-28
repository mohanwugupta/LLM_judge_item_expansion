#!/usr/bin/env python3
"""Add one prevalence-preserving human-sized random-context control.

Reuses the frozen training, prediction, scoring, bootstrap, and plotting code.
Original 13-condition scientific files are read only. No LLM calls are made.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import subprocess
import sys

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
import torch

from evaluate_validation import released_model
from iscci_validation.behavior import (
    cross_validated_mapping,
    item_table,
    load_tasks,
    paired_condition_contrasts,
    phenomenon_statistics,
    safe_correlation,
    summarize_behavior,
)
from iscci_validation.dataio import load_human_matrix, matrix_sha256
from iscci_validation.behavior_fidelity import item_fidelity
from iscci_validation.evaluation import (
    compare_evaluations,
    evaluate_model,
    make_evaluation_contexts,
)
from iscci_validation.provenance import environment_record, sha256_file, write_json
from run_human_behavior_validation import (
    DATA,
    EXPECTED_TRAINING,
    UPSTREAM,
    input_geometry,
    run_predictions,
    summary_table,
    validate_checkpoint,
)
from run_validation import train_run

ROOT = Path(__file__).resolve().parent
CONDITION = "random_human"
BASE_BEHAVIOR = ROOT / "artifacts/human_behavior_validation_v4_20261002"
BASE_VALIDATION = ROOT / "artifacts/validation_v4_20261002"
BASE_REPORT = ROOT / "reports/v4_20261002"
DEFAULT_OUTPUT = ROOT / "artifacts/random_feature_control_20261028"
DEFAULT_REPORT = ROOT / "reports/v4_20261028_random_control"


def randomize_features(human: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Independently permute each column, preserving context sizes, not row degrees."""
    if human.index.has_duplicates or human.columns.has_duplicates:
        raise ValueError("Randomization needs unique object and feature IDs")
    values = human.to_numpy()
    if not np.isin(values, [0, 1]).all():
        raise ValueError("Randomization requires a finalized binary matrix")
    rng = np.random.default_rng(seed)
    shuffled = np.column_stack(
        [rng.permutation(values[:, j]) for j in range(values.shape[1])]
    )
    result = pd.DataFrame(
        shuffled, index=human.index.copy(), columns=human.columns.copy(), dtype=np.int8
    )
    if not np.array_equal(result.sum(axis=0), human.sum(axis=0)):
        raise ValueError("Randomization changed feature prevalence")
    return result


def append_frozen_csv(source: Path, new: pd.DataFrame, target: Path):
    """Keep the original CSV bytes as a prefix; only append the new condition."""
    if source.resolve() == target.resolve():
        raise ValueError("Cannot overwrite an original table")
    with source.open(newline="") as stream:
        columns = next(csv.reader(stream))
    if set(new.columns) - set(columns):
        raise ValueError(
            f"Unexpected new columns for {source.name}: {set(new.columns)-set(columns)}"
        )
    if not source.read_bytes().endswith(b"\n"):
        raise ValueError(f"Original CSV lacks terminal newline: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    new.reindex(columns=columns).to_csv(target, mode="a", header=False, index=False)
    with source.open("rb") as a, target.open("rb") as b:
        for block in iter(lambda: a.read(1024 * 1024), b""):
            if b.read(len(block)) != block:
                raise ValueError(
                    f"Original CSV bytes changed in combined output: {source}"
                )
    return {
        "source": str(source),
        "sha256": sha256_file(source),
        "baseline_bytes_preserved": True,
        "appended_rows": len(new),
    }


def link_readonly(source: Path, target: Path):
    source = source.resolve()
    if target.is_symlink():
        if target.resolve() != source:
            raise ValueError(f"Conflicting existing link: {target}")
    elif target.exists():
        raise ValueError(f"Will not replace existing file/directory: {target}")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(source, target_is_directory=source.is_dir())


def train_seed(matrix, embedding, seed, parameters, run_dir, provenance):
    torch.set_num_threads(1)
    return train_run(
        matrix, embedding, CONDITION, seed, parameters, run_dir, provenance
    )


class Control:
    def __init__(self, args):
        self.args = args
        self.output = args.output_dir.resolve()
        self.report = args.report_dir.resolve()
        for path in [self.output, self.report]:
            if path == BASE_BEHAVIOR or path == BASE_VALIDATION or path == BASE_REPORT:
                raise ValueError(
                    "Use new output directories; frozen experiment cannot be overwritten"
                )
            if any(p in path.parents for p in [BASE_BEHAVIOR, BASE_VALIDATION]):
                raise ValueError("Do not write inside original scientific artifacts")
            path.mkdir(parents=True, exist_ok=True)
        self.base = json.loads((BASE_BEHAVIOR / "manifest.json").read_text())
        self.config = self.base["config"]
        if (
            not self.base["complete"]
            or not self.base["fidelity_gate"]["passed"]
            or self.config["training"] != EXPECTED_TRAINING
            or self.base["protocol_version"] != "iscci-external-human-behavior-1.1.0"
        ):
            raise ValueError(
                "A completed frozen-protocol evaluation and passed fidelity gate are required"
            )
        # Reuse the gate only if the data and scientific implementation still match it.
        for name, expected in self.base["implementation_sha256"].items():
            path = (
                ROOT / name
                if name == "run_human_behavior_validation.py"
                else ROOT / "iscci_validation" / name
            )
            if sha256_file(path) != expected:
                raise ValueError(f"Frozen scientific implementation changed: {path}")
        for dataset in self.base["human_datasets"].values():
            if sha256_file(Path(dataset["path"])) != dataset["sha256"]:
                raise ValueError(f"Frozen human data changed: {dataset['path']}")
        frozen_figures = json.loads(
            (BASE_REPORT / "figures_final/figure_manifest.json").read_text()
        )
        self.guard = {
            item["path"]: item["sha256"] for item in frozen_figures["sources"]
        }
        self.verify_originals()
        _, self.human = load_human_matrix(
            DATA / "leuven_dataset/leuven_combined_features_consolidated.csv"
        )
        if self.human.shape != (293, 385):
            raise ValueError("Expected the frozen 293-object, 385-feature human matrix")
        if matrix_sha256(self.human) != self.base["matrix_sha256"]["human"]:
            raise ValueError("Human control matrix differs from completed experiment")
        self.matrix = randomize_features(self.human, args.randomization_seed)
        self.validation = self.output / "validation"
        self.behavior = self.output / "behavior"
        self.combined = self.output / "combined_behavior"
        for path in [self.validation / "data_matrices", self.behavior, self.combined]:
            path.mkdir(parents=True, exist_ok=True)
        matrix_path = self.validation / "data_matrices" / f"{CONDITION}.csv"
        if matrix_path.exists():
            existing = pd.read_csv(matrix_path, index_col=0)
            if not np.isin(existing.values, [0, 1]).all() or matrix_sha256(
                existing.astype(np.int8)
            ) != matrix_sha256(self.matrix):
                raise ValueError(
                    "Existing control has a different random matrix; use a new directory"
                )
        else:
            self.matrix.to_csv(matrix_path)
        self.released_path = UPSTREAM / "models/1and2shot_isc-seed3.pt"
        if (
            sha256_file(self.released_path)
            != self.base["released_checkpoint_sha256"]["3"]
        ):
            raise ValueError("Frozen released encoder checkpoint changed")
        self.embedding = (
            released_model(self.released_path)
            .input_to_independent.weight.detach()
            .clone()
        )
        self.provenance = {
            "protocol_version": "iscci-random-human-context-control-1.0.0",
            "randomization": "independent within-feature object permutation",
            "randomization_seed": args.randomization_seed,
            "randomization_rng": "numpy.default_rng/PCG64",
            "randomization_realizations": 1,
            "human_matrix_sha256": matrix_sha256(self.human),
            "matrix_sha256": matrix_sha256(self.matrix),
            "released_checkpoint_sha256": sha256_file(self.released_path),
            "scientific_implementation_sha256": self.base["implementation_sha256"],
        }
        audit = pd.DataFrame(
            {
                "feature_id": self.human.columns,
                "human_positive_objects": self.human.sum(axis=0).values,
                "random_positive_objects": self.matrix.sum(axis=0).values,
            }
        )
        audit.to_csv(self.output / "feature_prevalence_audit.csv", index=False)
        pd.DataFrame(
            {
                "object": self.human.index,
                "human_positive_features": self.human.sum(axis=1).values,
                "random_positive_features": self.matrix.sum(axis=1).values,
            }
        ).to_csv(self.output / "object_degree_audit.csv", index=False)
        write_json(
            self.output / "randomization_manifest.json",
            {
                **self.provenance,
                "shape": list(self.matrix.shape),
                "positive_cells": int(self.matrix.values.sum()),
                "feature_prevalences_preserved": True,
                "object_degrees_preserved": False,
                "training_seeds": [0, 1, 2, 3],
                "uncertainty_scope": "one fixed shuffled matrix; intervals do not quantify variation across randomization realizations",
                "external_test_policy": "Randomization fixed before inspecting its behavioral results; no realization selection or tuning",
            },
        )

    def verify_originals(self):
        for name, digest in self.guard.items():
            if sha256_file(Path(name)) != digest:
                raise ValueError(f"Original experiment input changed: {name}")

    def checkpoints(self):
        return {
            seed: self.validation / "models" / CONDITION / f"seed_{seed}/checkpoint.pt"
            for seed in range(4)
        }

    def train(self):
        parameters = {k: v for k, v in EXPECTED_TRAINING.items() if k != "seeds"}
        pending, rows = [], []
        for seed, path in self.checkpoints().items():
            if path.exists():
                validate_checkpoint(
                    path,
                    CONDITION,
                    seed,
                    self.matrix,
                    self.embedding,
                    EXPECTED_TRAINING,
                )
                saved = torch.load(path, map_location="cpu", weights_only=False)
                if saved.get("provenance") != self.provenance:
                    raise ValueError(f"Random-control provenance mismatch: {path}")
                print(f"Reusing complete random-control seed {seed}", flush=True)
                rows.append(
                    dict(
                        condition=CONDITION,
                        seed=seed,
                        epochs=400,
                        task_count=385,
                        reused=True,
                    )
                )
            else:
                pending.append(
                    (
                        self.matrix,
                        self.embedding.numpy(),
                        seed,
                        parameters,
                        path.parent,
                        self.provenance,
                    )
                )
        if pending:
            print(
                f"Training {len(pending)} human-sized random-control seeds; 400 epochs each",
                flush=True,
            )
            with ProcessPoolExecutor(
                max_workers=self.args.workers,
                mp_context=multiprocessing.get_context("spawn"),
            ) as pool:
                futures = [pool.submit(train_seed, *job) for job in pending]
                for future in as_completed(futures):
                    row = future.result()
                    rows.append(row)
                    print(
                        f"Finished seed {row['seed']}: {row['elapsed_seconds']:.1f}s",
                        flush=True,
                    )
        pd.DataFrame(rows).sort_values("seed").to_csv(
            self.validation / "training_runs.csv", index=False
        )
        write_json(
            self.validation / "manifest.json",
            {
                **self.provenance,
                "complete": True,
                "training_parameters": parameters,
                "checkpoint_sha256": {
                    str(s): sha256_file(p) for s, p in self.checkpoints().items()
                },
                "environment": environment_record(),
            },
        )
        self.verify_originals()

    def lower_levels(self, tasks):
        item_frames, trial_frames = [], []
        names_to_index = {name: i for i, name in enumerate(self.human.index)}
        for level, values in [
            ("raw_feature_matrix", self.matrix.values),
            ("pretrained_semantic_activations", self.embedding[:-1].numpy()),
        ]:
            items, keys = item_table("similarity", tasks["similarity"])
            a = values[[names_to_index[name] for name in items["Premise 1"]]]
            b = values[[names_to_index[name] for name in items["Conclusion"]]]
            denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
            items["prediction"] = np.divide(
                np.sum(a.astype(float) * b, axis=1),
                denominator,
                out=np.full(len(items), np.nan),
                where=denominator > 0,
            )
            items["prediction_status"] = np.where(
                denominator > 0, "defined", "undefined_zero_feature_vector"
            )
            items = cross_validated_mapping(
                items, self.config["analysis"]["cross_validation_folds"]
            )
            items["condition"], items["level"] = CONDITION, level
            item_frames.append(items)
            trials = tasks["similarity"].merge(
                items[
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
            trials["condition"], trials["level"] = CONDITION, level
            trial_frames.append(trials)
        items, trials = pd.concat(item_frames, ignore_index=True), pd.concat(
            trial_frames, ignore_index=True
        )
        summary_input = items.copy()
        summary_input["condition"] += ":" + summary_input.level
        summary_input["seed"] = 0
        _, summary = summarize_behavior(
            summary_input,
            self.config["analysis"]["bootstrap_repetitions"],
            self.config["analysis"]["seed"],
        )
        summary[["base_condition", "level"]] = summary.condition.str.split(
            ":", expand=True
        )
        return items, trials, summary

    def representations(self):
        config = self.config["evaluation"]
        contexts = make_evaluation_contexts(
            293,
            config["pair_contexts"],
            config["rdm_contexts"],
            config["context_dependent_rdm_contexts"],
            config["seed"],
        )
        # Reuse the original cached human probes only after exact query-array checks.
        with np.load(
            BASE_VALIDATION / "evaluation/evaluation_contexts.npz"
        ) as original:
            if set(original.files) != set(contexts) or any(
                not np.array_equal(original[key], value)
                for key, value in contexts.items()
            ):
                raise ValueError(
                    "Representation probes differ from original experiment"
                )
        np.savez(self.behavior / "representation_probes.npz", **contexts)
        humans = []
        models_table = pd.read_csv(
            BASE_VALIDATION / "evaluation/models_evaluated.csv"
        ).set_index("model_id")
        for seed in range(4):
            checkpoint = BASE_VALIDATION / "models/human" / f"seed_{seed}/checkpoint.pt"
            expected = self.base["checkpoint_sha256"][f"human_seed_{seed}"]
            if (
                sha256_file(checkpoint) != expected
                or models_table.loc[f"human_seed_{seed}", "sha256"] != expected
            ):
                raise ValueError(
                    "Cached human reference checkpoint does not match frozen experiment"
                )
            with np.load(
                BASE_VALIDATION / "evaluation/model_outputs" / f"human_seed_{seed}.npz"
            ) as saved:
                humans.append({key: saved[key] for key in saved.files})
        geometry = input_geometry(self.matrix, self.human)
        rows = []
        for seed, path in self.checkpoints().items():
            model = validate_checkpoint(
                path, CONDITION, seed, self.matrix, self.embedding, EXPECTED_TRAINING
            )
            evaluation = evaluate_model(model, contexts)
            np.savez_compressed(
                self.behavior / f"representation_seed_{seed}.npz", **evaluation
            )
            comparisons = [compare_evaluations(evaluation, human) for human in humans]
            rows.append(
                {
                    "condition": CONDITION,
                    "seed": seed,
                    **geometry,
                    "semantic_activation_rdm_agreement": 1.0,
                    "semantic_activation_rdm_vs_leuven": safe_correlation(
                        pdist(self.embedding[:-1].numpy(), "cosine"),
                        pdist(self.human.values, "cosine"),
                        "spearman",
                    ),
                    "semantic_activation_note": "shared frozen human-trained embeddings; not independently learned from LLM features",
                    "features": 385,
                    **{
                        metric: float(np.mean([r[metric] for r in comparisons]))
                        for metric in comparisons[0]
                    },
                }
            )
        return pd.DataFrame(rows)

    def evaluate(self):
        torch.set_num_threads(1)
        tasks = load_tasks(DATA)
        predictions, fidelity_checks = [], []
        for seed, path in self.checkpoints().items():
            model = validate_checkpoint(
                path, CONDITION, seed, self.matrix, self.embedding, EXPECTED_TRAINING
            )
            metrics = pd.read_csv(path.parent / "training_metrics.csv")
            if (
                len(metrics) != 3200
                or not np.isfinite(metrics.to_numpy()).all()
                or not all(torch.isfinite(v).all() for v in model.state_dict().values())
            ):
                raise ValueError(
                    f"Incomplete or nonfinite random-control training: {path}"
                )
            checks = item_fidelity(model, tasks, list(self.human.index), UPSTREAM, 1e-5)
            checks["condition"], checks["seed"] = CONDITION, seed
            if not checks.passed.all():
                raise ValueError(
                    f"Original-paper model graph mismatch: random seed {seed}"
                )
            fidelity_checks.append(checks)
            print(
                f"Original-paper behavioral predictions: random seed {seed}", flush=True
            )
            predictions.append(
                run_predictions(
                    model,
                    CONDITION,
                    seed,
                    tasks,
                    list(self.human.index),
                    self.config,
                    self.behavior,
                    path,
                )
            )
        predictions = pd.concat(predictions, ignore_index=True)
        pd.concat(fidelity_checks, ignore_index=True).to_csv(
            self.behavior / "original_graph_parity.csv", index=False
        )
        predictions.to_csv(self.behavior / "all_item_predictions.csv", index=False)
        print(
            "Bootstrapping random-control human-item/seed uncertainty (2,000 draws)",
            flush=True,
        )
        per_seed, summary = summarize_behavior(
            predictions,
            self.config["analysis"]["bootstrap_repetitions"],
            self.config["analysis"]["seed"],
        )
        if len(per_seed) != 72 or set(per_seed.seed) != {0, 1, 2, 3}:
            raise ValueError(
                "Random control must have all 18 original task/dataset outcomes for four seeds"
            )
        lower_items, lower_trials, lower_summary = self.lower_levels(tasks)
        representations = self.representations()
        statistics = []
        for seed in range(4):
            frame = pd.read_csv(
                self.behavior
                / "per_model"
                / f"{CONDITION}_seed_{seed}/trial_statistics.csv"
            )
            frame["model_id"] = f"{CONDITION}_seed_{seed}"
            statistics.append(frame)
        human_items = pd.concat(
            [
                pd.concat(
                    [
                        pd.read_csv(
                            BASE_BEHAVIOR
                            / "per_model"
                            / f"human_seed_{seed}"
                            / f"{task}_items.csv"
                        )
                        for task in tasks
                    ],
                    ignore_index=True,
                )
                for seed in range(4)
            ],
            ignore_index=True,
        )
        contrasts = paired_condition_contrasts(
            pd.concat([human_items, predictions], ignore_index=True),
            [("human", CONDITION)],
            self.config["analysis"]["bootstrap_repetitions"],
            self.config["analysis"]["seed"],
        )
        phenomena = []
        for seed, frame in predictions[predictions.task.eq("phenomena")].groupby(
            "seed"
        ):
            stats = phenomenon_statistics(frame)
            stats["condition"], stats["seed"] = CONDITION, seed
            phenomena.append(stats)
        models = {(CONDITION, seed): path for seed, path in self.checkpoints().items()}
        condition = summary_table(
            [CONDITION],
            {"human": self.human, CONDITION: self.matrix},
            models,
            summary,
            representations,
            False,
            lower_summary,
            pd.read_csv(BASE_BEHAVIOR / "human_reliability.csv"),
        )
        new_tables = {
            "behavior_per_seed.csv": per_seed,
            "behavior_summary.csv": summary,
            "representation_per_seed.csv": representations,
            "condition_summary.csv": condition,
            "similarity_levels_1_2_items.csv": lower_items,
            "similarity_levels_1_2_trials.csv": lower_trials,
            "similarity_levels_1_2_summary.csv": lower_summary,
            "paired_condition_contrasts.csv": contrasts,
            "all_individual_trial_statistics.csv": pd.concat(
                statistics, ignore_index=True
            ),
            "paper_phenomenon_tests.csv": pd.concat(phenomena, ignore_index=True),
        }
        parity = {}
        for name, frame in new_tables.items():
            frame.to_csv(self.behavior / name, index=False)
            parity[name] = append_frozen_csv(
                BASE_BEHAVIOR / name, frame, self.combined / name
            )
        for name in [
            "human_reliability.csv",
            "human_participant_reliability.csv",
            "paper_human_participant_choices.csv",
            "paper_human_choice_statistics.csv",
            "fidelity_gate.json",
        ]:
            link_readonly(BASE_BEHAVIOR / name, self.combined / name)
        for path in sorted((BASE_BEHAVIOR / "per_model").iterdir()):
            if path.is_dir():
                link_readonly(path, self.combined / "per_model" / path.name)
        for path in sorted((self.behavior / "per_model").iterdir()):
            link_readonly(path, self.combined / "per_model" / path.name)
        for path in sorted((BASE_VALIDATION / "data_matrices").glob("*.csv")):
            link_readonly(
                path, self.output / "combined_validation/data_matrices" / path.name
            )
        link_readonly(
            self.validation / "data_matrices" / f"{CONDITION}.csv",
            self.output / "combined_validation/data_matrices" / f"{CONDITION}.csv",
        )
        manifest = {
            **self.base,
            "complete": True,
            "status": "complete_with_random_control",
            "parent_manifest": str(BASE_BEHAVIOR / "manifest.json"),
            "parent_manifest_sha256": sha256_file(BASE_BEHAVIOR / "manifest.json"),
            "random_control": self.provenance,
            "baseline_table_parity": parity,
            "random_control_implementation_sha256": sha256_file(Path(__file__)),
            "missing_required_seeds": [],
        }
        manifest["checkpoint_sha256"] = {
            **self.base["checkpoint_sha256"],
            **{
                f"{CONDITION}_seed_{s}": sha256_file(p)
                for s, p in self.checkpoints().items()
            },
        }
        manifest["matrix_sha256"] = {
            **self.base["matrix_sha256"],
            CONDITION: matrix_sha256(self.matrix),
        }
        self.verify_originals()
        manifest["original_source_hashes_verified_unchanged"] = True
        write_json(self.combined / "manifest.json", manifest)
        write_json(
            self.behavior / "manifest.json",
            {
                "complete": True,
                "condition": CONDITION,
                "protocol": self.provenance,
                "behavior_protocol": self.config,
                "baseline_table_parity": parity,
            },
        )
        print(
            f"Random control evaluated and appended; original 13-condition table bytes preserved: {self.combined}",
            flush=True,
        )

    def plot(self):
        if not (self.combined / "manifest.json").exists():
            raise ValueError("Complete the random-control evaluation before plotting")
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "plot_finalized_v4_results.py"),
                "--include-random-control",
                "--behavior-dir",
                str(self.combined),
                "--validation-dir",
                str(self.output / "combined_validation"),
                "--output-dir",
                str(self.report / "figures_final"),
            ],
            check=True,
            cwd=ROOT.parent,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--randomization-seed", type=int, default=20261028)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--stage", choices=["all", "train", "evaluate", "plot"], default="all"
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    control = Control(args)
    if args.stage in ("all", "train"):
        control.train()
    if args.stage in ("all", "evaluate"):
        control.evaluate()
    if args.stage in ("all", "plot"):
        control.plot()
    control.verify_originals()


if __name__ == "__main__":
    main()
