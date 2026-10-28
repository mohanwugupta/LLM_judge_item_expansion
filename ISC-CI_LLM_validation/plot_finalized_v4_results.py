#!/usr/bin/env python3
"""Render frozen ISC-CI results. No checkpoints, inference, or judgment payloads.

Dependencies: numpy, pandas, scipy, matplotlib. See FINAL_FIGURES.md.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/iscci-figure-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parent
MAIN = [
    "human",
    "v2",
    "v3_B",
    "v3_1_B",
    "v4_b_calibrated",
    "v4_ensemble_source_only",
    "v4_ensemble_calibrated",
]
ALL = MAIN + [
    "v3_A",
    "v3_C",
    "v3_1_A",
    "v3_1_C",
    "v4_b_locked_v2",
    "v4_ensemble_locked_v2",
]
RANDOM = "random_human"
LABELS = dict(
    zip(
        ALL,
        [
            "Human-norm-trained ISC-CI",
            "V2",
            "V3-B",
            "V3.1-B",
            "V4 fixed-B, calibrated",
            "V4 source-only",
            "V4 completed ensemble",
            "V3-A",
            "V3-C",
            "V3.1-A",
            "V3.1-C",
            "V4 fixed-B, locked",
            "V4 ensemble, locked",
        ],
    )
)
SHORT = {
    c: s
    for c, s in zip(
        ALL,
        [
            "Human-norm-trained",
            "V2",
            "V3-B",
            "V3.1-B",
            "V4 fixed-B (cal.)",
            "V4 source-only",
            "V4 ensemble (cal.)",
            "V3-A",
            "V3-C",
            "V3.1-A",
            "V3.1-C",
            "V4 fixed-B (locked)",
            "V4 ensemble (locked)",
        ],
    )
}
# Okabe-Ito colors; the expanded family also varies markers and line styles.
COLORS = dict(
    zip(
        ALL,
        [
            "#333333",
            "#0072B2",
            "#E69F00",
            "#CC79A7",
            "#009E73",
            "#56B4E9",
            "#D55E00",
            "#E69F00",
            "#E69F00",
            "#CC79A7",
            "#CC79A7",
            "#009E73",
            "#D55E00",
        ],
    )
)
MARKERS = dict(
    zip(ALL, ["o", "s", "^", "D", "P", "v", "X", "<", ">", "h", "*", "s", "D"])
)
LABELS[RANDOM] = "Random-feature ISC-CI (human-sized)"
SHORT[RANDOM] = "Random features (385)"
COLORS[RANDOM] = "#7F7F7F"
MARKERS[RANDOM] = "8"
DOMAINS = [
    "Birds",
    "Clothing",
    "Fish",
    "Insects",
    "Mammals",
    "Reptiles",
    "Tools",
    "Vehicles",
    "Weapons",
]
INDUCTION = [
    "Bhatia - Exp 1",
    "Bhatia - Exp 3",
    "Han - Exp 2",
    "Osherson - Exp 4",
    "Rips",
]
TASK_LABELS = {
    "thematic": "Context-specific categorization (paper Fig. 6)",
    "nonmonotonicity": "Context-dependent nonmonotonicity (paper Fig. 5)",
}
CONTRASTS = [
    ("v4_b_calibrated", "v3_1_B"),
    ("v4_ensemble_calibrated", "v4_ensemble_source_only"),
    ("v4_b_calibrated", "v4_b_locked_v2"),
    ("v4_ensemble_calibrated", "v4_ensemble_locked_v2"),
]
DEFINITIONS = {
    "input_rdm": "Spearman of object cosine distances; primary historical undefined distance -> 0; defined-pair sensitivity separate",
    "similarity": "Original domain-specific Pearson r with mean normalized human judgments",
    "induction": "Original dataset-specific Pearson r with observed human ratings",
    "context_direction": "Fraction model E > 0 on 38 sets with observed human E > 0",
    "context_effect": "E = (c1-Distractor1 - c2-Distractor1) + (c2-Distractor2 - c1-Distractor2); full three-choice denominator",
    "asymmetry": "Argument-2 logit minus argument-1 logit; human argument-2 fraction; original direction subset p <= .05 (34/50 pairs)",
    "preferences": "Original item-level fraction model argument-2 logit > argument-1 logit; not accuracy; participant distributions and medians separate",
    "intervals": "Saved percentile 95% crossed cluster-item/training-seed bootstrap; CV predictions fixed, mapping not refit",
    "common_pairs": "Domain-specific intersection of finite raw predictions/human values across seven main conditions; points only, no new bootstrap",
    "representation": "Saved per-seed agreement with independently seeded reconstructed human models; replicate SD is not a 95% CI",
}


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def check_unique(frame, keys, description):
    if frame.duplicated(keys).any():
        raise ValueError(f"Duplicate {description}: {keys}")


def check_seeds(frame, conditions=None):
    conditions = ALL if conditions is None else conditions
    if set(frame.condition) != set(conditions):
        raise ValueError(
            f"Expected exactly {len(conditions)} specified conditions, without released references"
        )
    for condition, rows in frame.groupby("condition"):
        if set(rows.seed) != {0, 1, 2, 3}:
            raise ValueError(f"{condition}: expected training seeds 0-3")


def finite_correlation(human, model):
    mask = np.isfinite(human) & np.isfinite(model)
    x, y = np.asarray(human)[mask], np.asarray(model)[mask]
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan
    return float(pearsonr(x, y).statistic)


def common_pair_scores(raw, trained, shared, conditions=MAIN, finite_conditions=None):
    """Stable item join; never replace sparse missing values with zero."""
    check_unique(raw, ["condition", "dataset", "item_id"], "raw similarity rows")
    check_unique(
        trained, ["condition", "seed", "dataset", "item_id"], "trained similarity rows"
    )
    result, membership = [], []
    for domain in DOMAINS:
        data = raw[raw.dataset.eq(domain) & raw.condition.isin(conditions)]
        wide = data.pivot(
            index="item_id", columns="condition", values="prediction"
        ).reindex(columns=conditions)
        # Adding a negative control must not silently change the old sensitivity subset.
        finite_conditions = (
            conditions if finite_conditions is None else finite_conditions
        )
        valid = np.isfinite(wide[finite_conditions]).all(axis=1)
        observed = (
            data[data.condition.eq("human")]
            .set_index("item_id")
            .human.reindex(wide.index)
        )
        ids = wide.index[valid & np.isfinite(observed)]
        for item_id in wide.index:
            membership.append(
                dict(
                    dataset=domain,
                    item_id=item_id,
                    common_finite=bool(item_id in ids),
                    subset="seven-main-condition finite raw intersection",
                )
            )
        for level, rows, selected in [
            ("raw_feature_matrix", data, conditions),
            ("fully_trained", trained[trained.dataset.eq(domain)], conditions),
            (
                "pretrained_semantic_activations",
                shared[shared.dataset.eq(domain)],
                ["shared_encoder"],
            ),
        ]:
            for condition in selected:
                subset = rows[rows.condition.eq(condition) & rows.item_id.isin(ids)]
                groups = (
                    subset.groupby("seed") if "seed" in subset else [(np.nan, subset)]
                )
                values = []
                for seed, group in groups:
                    if set(group.item_id) != set(ids):
                        raise ValueError(
                            f"Common-pair join incomplete: {condition}/{domain}/{seed}"
                        )
                    ordered = group.set_index("item_id").reindex(ids)
                    if not np.allclose(
                        ordered.human, observed.reindex(ids), equal_nan=True
                    ):
                        raise ValueError("Human observations differ between levels")
                    estimate = finite_correlation(ordered.human, ordered.prediction)
                    result.append(
                        dict(
                            condition=condition,
                            task="similarity",
                            dataset=domain,
                            level=level,
                            seed=seed,
                            items=len(ids),
                            original_items=len(wide),
                            coverage=len(ids) / len(wide),
                            mean=estimate,
                            metric="paper_statistic",
                            ci_low=np.nan,
                            ci_high=np.nan,
                            uncertainty="not estimated for changed subset",
                            subset="seven-main-condition finite raw intersection",
                        )
                    )
                    values.append(estimate)
                if level == "fully_trained":
                    result.append(
                        dict(
                            condition=condition,
                            task="similarity",
                            dataset=domain,
                            level=level,
                            seed=np.nan,
                            items=len(ids),
                            original_items=len(wide),
                            coverage=len(ids) / len(wide),
                            mean=np.mean(values),
                            metric="paper_statistic",
                            ci_low=np.nan,
                            ci_high=np.nan,
                            uncertainty="descriptive four-seed mean; no subset CI",
                            subset="seven-main-condition finite raw intersection",
                        )
                    )
    return pd.DataFrame(result), pd.DataFrame(membership)


def verify_effects(frame, task):
    if task == "context":
        expected = (
            frame["c1-Distractor 1"]
            - frame["c2-Distractor 1"]
            + frame["c2-Distractor 2"]
            - frame["c1-Distractor 2"]
        )
        human = (
            frame["Conclusion 1 Chosen-Distractor 1"]
            - frame["Conclusion 2 Chosen-Distractor 1"]
            + frame["Conclusion 2 Chosen-Distractor 2"]
            - frame["Conclusion 1 Chosen-Distractor 2"]
        )
        if not np.allclose(human, frame.human, atol=1e-12):
            raise ValueError("Human context orientation changed")
        for d in (1, 2):
            probabilities = frame[[f"c1-Distractor {d}", f"c2-Distractor {d}"]]
            if ((probabilities < 0) | (probabilities > 1)).any().any() or (
                probabilities.sum(axis=1) > 1 + 1e-12
            ).any():
                raise ValueError("Invalid saved three-alternative probabilities")
    else:
        expected = frame.argument2_logit - frame.argument1_logit
    # Saved model logits/predictions were float32; separate CSV rounding can
    # differ by up to 5e-7 on recomputed subtraction (below fidelity tolerance).
    if not np.allclose(expected, frame.prediction, atol=1e-6, rtol=1e-7):
        raise ValueError(f"Saved {task} score does not match original definition")


class Package:
    def __init__(self, args):
        self.args = args
        self.main = MAIN + ([RANDOM] if args.include_random_control else [])
        self.all = ALL + ([RANDOM] if args.include_random_control else [])
        self.out = args.output_dir.resolve()
        if (
            args.include_random_control
            and self.out == (ROOT / "reports/v4_20261002/figures_final").resolve()
        ):
            raise ValueError(
                "Use a new --output-dir for the random-control extension; preserve the frozen figure package"
            )
        protected = [
            args.behavior_dir.resolve(),
            args.validation_dir.resolve(),
            args.report_dir.resolve(),
        ]
        if any(self.out == p or self.out in p.parents for p in protected):
            raise ValueError(
                "Output must be a new subdirectory, not an input directory or its ancestor"
            )
        if any(p in self.out.parents for p in protected[:2]):
            raise ValueError(
                "Do not write plots inside scientific artifact directories"
            )
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "source_data").mkdir(exist_ok=True)
        self.sources, self.figures, self.blocked, self.captions = {}, [], [], []
        self.matrix_cache = {}
        plt.rcParams.update(
            {
                "font.size": 10,
                "axes.titlesize": 11,
                "axes.labelsize": 10,
                "legend.fontsize": 9,
                "svg.fonttype": "none",
                "pdf.fonttype": 42,
                "axes.spines.top": False,
                "axes.spines.right": False,
            }
        )

    def record(self, path):
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Required frozen artifact missing: {path}")
        digest = sha256(path)
        old = self.sources.get(str(path))
        if old and old != digest:
            raise ValueError(f"Input changed during rendering: {path}")
        self.sources[str(path)] = digest
        return path

    def read(self, path, **kwargs):
        return pd.read_csv(self.record(path), **kwargs)

    def json(self, path):
        return json.loads(self.record(path).read_text())

    def stage(self, name, function):
        try:
            print(f"Rendering {name}", flush=True)
            function()
        except FileNotFoundError as error:
            self.blocked.append({"panel_group": name, "missing_input": str(error)})
            print(f"BLOCKED {name}: {error}", flush=True)

    def source(self, name, data, **defaults):
        frame = data.copy()
        for column, value in defaults.items():
            if column not in frame:
                frame[column] = value
            elif pd.notna(value):
                frame[column] = frame[column].fillna(value)
        standard = {
            "condition": "not_applicable",
            "level": "not_applicable",
            "dataset": "all",
            "task": "representation",
            "metric": "descriptive",
            "subset": "frozen original items",
            "seed": np.nan,
            "items": np.nan,
            "coverage": np.nan,
            "uncertainty": "descriptive; no confidence interval",
        }
        for column, value in standard.items():
            if column not in frame:
                frame[column] = value
            elif pd.notna(value):
                frame[column] = frame[column].fillna(value)
        frame["display_label"] = frame.condition.map(LABELS).fillna(frame.condition)
        if "ci_method" in frame:
            frame["uncertainty"] = frame.ci_method.fillna(frame.uncertainty)
        if "estimate" not in frame:
            frame["estimate"] = np.nan
        if "mean" in frame:
            frame["estimate"] = frame["estimate"].fillna(frame["mean"])
        if "prediction" in frame:
            frame["estimate"] = frame["estimate"].fillna(frame["prediction"])
        target = self.out / "source_data" / f"{name}.csv"
        frame.to_csv(target, index=False)
        return str(target.relative_to(self.out))

    def save(self, fig, name, data, caption, **defaults):
        name = name.replace("_all13", f"_all{len(self.all)}")
        sources = []
        if "panel" in data:
            for panel, rows in data.groupby("panel", sort=False):
                sources.append(self.source(f"{name}_{panel}", rows, **defaults))
        else:
            sources.append(self.source(name, data, **defaults))
        files = []
        for suffix in ("svg", "pdf", "png"):
            path = self.out / f"{name}.{suffix}"
            fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
            files.append(path.name)
        plt.close(fig)
        self.figures.append({"figure": name, "files": files, "source_data": sources})
        self.captions.append(
            f"## {name}\n\n{caption}\n\nFiles: [{name}.pdf]({name}.pdf). Source data: "
            + ", ".join(f"[{Path(s).name}]({s})" for s in sources)
            + ".\n"
        )

    def initialize(self):
        b = self.args.behavior_dir
        self.protocol = self.json(b / "manifest.json")
        if (
            not self.protocol["complete"]
            or not self.protocol["fidelity_gate"]["passed"]
        ):
            raise ValueError(
                "Completed authoritative evaluation and passed fidelity gate required"
            )
        if self.protocol["protocol_version"] != "iscci-external-human-behavior-1.1.0":
            raise ValueError(
                "This reporting package requires the frozen common-LCA-seed protocol 1.1.0"
            )
        self.behavior = self.read(b / "behavior_summary.csv")
        self.seeds = self.read(b / "behavior_per_seed.csv")
        self.rep = self.read(b / "representation_per_seed.csv")
        self.lower = self.read(b / "similarity_levels_1_2_summary.csv")
        self.reliability = self.read(b / "human_reliability.csv")
        self.contrasts = self.read(b / "paired_condition_contrasts.csv")
        self.conditions = self.read(b / "condition_summary.csv")
        check_unique(
            self.behavior,
            ["condition", "task", "dataset", "metric"],
            "behavior summaries",
        )
        check_unique(
            self.seeds, ["condition", "task", "dataset", "seed"], "seed statistics"
        )
        check_unique(self.rep, ["condition", "seed"], "representations")
        check_seeds(self.seeds, self.all)
        check_seeds(self.rep, self.all)
        if self.args.include_random_control and not self.protocol.get("random_control"):
            raise ValueError(
                "Random control requires a completed, provenance-recorded extension"
            )
        self.conditions = (
            self.conditions.set_index("condition").reindex(self.all).reset_index()
        )
        if (
            not self.conditions.seed_count.eq(4).all()
            or not self.conditions.status.eq("evaluated").all()
        ):
            raise ValueError("Incomplete final condition table")
        self.conditions.insert(
            1, "display_label", self.conditions.condition.map(LABELS)
        )
        self.conditions.to_csv(self.out / "all_condition_summary.csv", index=False)
        self.json(self.args.report_dir / "report_manifest.json")
        self.record(self.args.report_dir / "RESULTS.md")
        self.anchor_checks()
        self.source("all_behavioral_metrics", self.behavior, level="fully_trained")
        self.source(
            "all_representation_seeds",
            self.rep,
            level="fully_trained",
            uncertainty="four independent seed replicates; not a CI",
        )
        self.source(
            "all_paired_contrasts",
            self.contrasts.rename(columns={"condition_a": "condition"}),
            level="fully_trained",
            uncertainty=DEFINITIONS["intervals"],
        )

    def anchor_checks(self):
        # Rounded PRD values are validation targets only, never plotted inputs.
        expected = [
            (1.0, 0.559, 0.498, 0.947),
            (0.821, 0.511, 0.554, 0.954),
            (0.090, 0.478, 0.450, 0.849),
            (0.241, 0.534, 0.425, 0.757),
            (0.762, 0.500, 0.506, 0.961),
            (0.641, 0.575, 0.513, 0.921),
            (0.821, 0.520, 0.532, 0.928),
        ]
        rows = []
        for condition, anchors in zip(MAIN, expected):
            behavior = self.seeds[self.seeds.condition.eq(condition)]
            actual = [
                self.rep[
                    self.rep.condition.eq(condition)
                ].input_object_rdm_spearman.iloc[0]
            ]
            actual += [
                behavior[behavior.task.eq(task)].paper_statistic.mean()
                for task in ["similarity", "induction", "context"]
            ]
            for metric, value, anchor in zip(
                [
                    "input_rdm",
                    "similarity_unweighted_domain_mean",
                    "induction_unweighted_dataset_mean",
                    "context_direction",
                ],
                actual,
                anchors,
            ):
                if abs(value - anchor) > 0.00055:
                    raise ValueError(
                        f"Frozen-report anchor mismatch: {condition}/{metric}: {value} vs {anchor}"
                    )
                rows.append(
                    dict(
                        condition=condition,
                        metric=metric,
                        estimate=value,
                        rounded_prd_anchor=anchor,
                        passed=True,
                    )
                )
        for condition, level, anchor in [
            ("v4_ensemble_source_only", "raw_feature_matrix", 0.593),
            ("v4_ensemble_calibrated", "raw_feature_matrix", 0.524),
            ("human", "pretrained_semantic_activations", 0.539),
        ]:
            values = self.lower[
                self.lower.base_condition.eq(condition)
                & self.lower.level.eq(level)
                & self.lower.metric.eq("paper_statistic")
            ]
            if len(values) != 9 or abs(values["mean"].mean() - anchor) > 0.00055:
                raise ValueError(
                    f"Lower-level frozen anchor mismatch: {condition}/{level}"
                )
            rows.append(
                dict(
                    condition=condition,
                    level=level,
                    metric="similarity_unweighted_domain_mean",
                    estimate=values["mean"].mean(),
                    rounded_prd_anchor=anchor,
                    passed=True,
                )
            )
        self.source("frozen_anchor_checks", pd.DataFrame(rows))

    def matrix(self, condition):
        if condition in self.matrix_cache:
            return self.matrix_cache[condition]
        path = self.record(
            self.args.validation_dir / "data_matrices" / f"{condition}.csv"
        )
        cache = self.out / "cache"
        cache.mkdir(exist_ok=True)
        target = cache / f"{condition}_{self.sources[str(path)]}.npz"
        if target.is_file():
            with np.load(target) as data:
                result = {key: data[key] for key in data.files}
        else:
            # Parse the finalized wide CSV one row at a time, not 87k pandas columns.
            words, values = [], []
            with path.open(newline="") as stream:
                rows = csv.reader(stream)
                columns = next(rows)[1:]
                for row in rows:
                    a = np.asarray(row[1:], dtype=np.float32)
                    if len(a) != len(columns) or not np.isin(a, [0, 1]).all():
                        raise ValueError(f"Invalid finalized binary matrix: {path}")
                    words.append(row[0])
                    values.append(a)
            matrix = np.asarray(values, dtype=np.float32)
            if (
                len(words) != 293
                or len(set(words)) != 293
                or len(set(columns)) != len(columns)
            ):
                raise ValueError(
                    f"Expected 293 unique objects and unique contexts: {path}"
                )
            result = dict(
                words=np.asarray(words),
                features=np.asarray(columns),
                object_counts=matrix.sum(axis=1),
                feature_counts=matrix.sum(axis=0),
                rdm=squareform(pdist(matrix, metric="cosine")),
            )
            # A zero-vector diagonal is undefined too, not a meaningful zero distance.
            zero = result["object_counts"] == 0
            result["rdm"][zero, :] = np.nan
            result["rdm"][:, zero] = np.nan
            np.savez_compressed(target, **result)
        expected = int(self.rep[self.rep.condition.eq(condition)].features.iloc[0])
        if len(result["features"]) != expected:
            raise ValueError(f"Retained context count mismatch: {condition}")
        human_words = (
            result["words"] if condition == "human" else self.matrix("human")["words"]
        )
        if not np.array_equal(result["words"], human_words):
            raise ValueError(
                f"Object order differs from frozen human matrix: {condition}"
            )
        saved_empty = int(
            self.rep[
                self.rep.condition.eq(condition)
            ].input_objects_without_features.iloc[0]
        )
        if (result["object_counts"] == 0).sum() != saved_empty:
            raise ValueError(f"Feature-empty count mismatch: {condition}")
        self.matrix_cache[condition] = result
        return result

    def ordered_objects(self):
        words = self.matrix("human")["words"]
        path = self.args.categories
        if path and path.is_file():
            categories = self.read(path)
            check_unique(categories, ["Name"], "original Leuven category names")
            table = categories[categories.Name.isin(words)].copy()
            if set(table.Name) != set(words):
                raise ValueError(
                    "Original category file does not cover frozen 293 objects"
                )
            table = table.rename(columns={"Name": "object", "Category 1": "category"})
            # Original file/category order, not model-based clustering.
            category_order = list(dict.fromkeys(table.category))
            table["category_rank"] = table.category.map(
                dict(zip(category_order, range(len(category_order))))
            )
            table = table.sort_values("category_rank", kind="stable")
            definition = (
                "original Leuven Category 1 order; within-category original file order"
            )
        else:
            table = pd.DataFrame({"object": words, "category": "Frozen training order"})
            definition = "frozen human matrix order; original category file unavailable"
        table["rdm_index"] = np.arange(len(table))
        self.source("object_order", table, subset=definition, metric="ordering")
        return table, np.array([list(words).index(w) for w in table.object])

    def structure(self, conditions, prefix):
        geometry = (
            self.rep.drop_duplicates("condition")
            .set_index("condition")
            .loc[conditions]
            .reset_index()
        )
        geometry = geometry[
            [
                "condition",
                "features",
                "input_object_rdm_spearman",
                "input_object_rdm_defined_pairs_spearman",
                "input_objects_without_features",
                "input_rdm_undefined_pairs",
                "input_rdm_primary_convention",
            ]
        ]
        geometry["finite_pair_coverage"] = 1 - geometry.input_rdm_undefined_pairs / (
            293 * 292 / 2
        )
        geometry["coverage"] = geometry.finite_pair_coverage
        fig, axes = plt.subplots(
            1, 2, figsize=(13, max(5, len(conditions) * 0.47)), layout="constrained"
        )
        for i, c in enumerate(conditions):
            row = geometry[geometry.condition.eq(c)].iloc[0]
            axes[0].scatter(row.features, i, color=COLORS[c], marker=MARKERS[c], s=60)
            axes[0].annotate(
                f"{int(row.features):,}",
                (row.features, i),
                xytext=(8, 0),
                textcoords="offset points",
                va="center",
                fontsize=9,
            )
            axes[1].plot(
                [
                    row.input_object_rdm_spearman,
                    row.input_object_rdm_defined_pairs_spearman,
                ],
                [i, i],
                color=COLORS[c],
                alpha=0.5,
            )
            axes[1].scatter(
                row.input_object_rdm_spearman,
                i,
                color=COLORS[c],
                marker=MARKERS[c],
                s=45,
            )
            axes[1].scatter(
                row.input_object_rdm_defined_pairs_spearman,
                i,
                facecolors="none",
                edgecolors=COLORS[c],
                marker="o",
                s=85,
            )
            axes[1].text(
                1.08,
                i,
                f"{row.finite_pair_coverage:.1%} / {int(row.input_objects_without_features)}",
                va="center",
                fontsize=9,
            )
        for ax in axes:
            ax.set_yticks(range(len(conditions)), [SHORT[c] for c in conditions])
            ax.set_ylim(len(conditions) - 0.4, -0.7)
            ax.grid(axis="x", alpha=0.2)
        axes[0].set_xscale("log")
        axes[0].set_xlim(60, 400000)
        axes[0].set_xlabel("Retained contexts (log scale)")
        axes[0].set_title("A. Feature inventory")
        axes[1].set_xlim(-0.08, 1.43)
        axes[1].set_xticks([0, 0.25, 0.5, 0.75, 1])
        axes[1].set_xlabel("Input-RDM Spearman agreement with Leuven")
        axes[1].set_title("B. Geometry; right column: finite pairs / empty objects")
        axes[1].legend(
            handles=[
                Line2D(
                    [],
                    [],
                    marker="o",
                    color="black",
                    linestyle="none",
                    label="Historical convention",
                ),
                Line2D(
                    [],
                    [],
                    marker="o",
                    markerfacecolor="white",
                    color="black",
                    linestyle="none",
                    label="Defined pairs only",
                ),
            ],
            loc="upper left" if self.args.include_random_control else "lower left",
            fontsize=8,
        )
        geometry["panel"] = "geometry"
        geometry["metric"] = "input_object_rdm_spearman"
        geometry["estimate"] = geometry.input_object_rdm_spearman
        inventory = geometry.copy()
        inventory["panel"] = "inventory"
        inventory["metric"] = "retained_contexts"
        inventory["estimate"] = inventory.features
        self.save(
            fig,
            prefix,
            pd.concat([inventory, geometry], ignore_index=True),
            "Question: does completion recover feature structure? Retained-context counts use a log axis. Input geometry uses the frozen historical rank convention; open circles use saved defined-pair sensitivity. Coverage excludes undefined off-diagonal cosine pairs out of 42,778; empty objects have no positive retained features. Human agreement is 1 by definition. No training-seed uncertainty applies to input matrices. Completion improves coarse geometry, not necessarily behavior.",
            level="raw_feature_matrix",
            metric="input_object_rdm_spearman",
        )

    def distributions(self):
        fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout="constrained")
        frames = []
        specs = [
            ("object_counts", "Positive features / concept", False),
            ("feature_counts", "Concepts / feature", False),
            (
                "object_counts",
                "Positive-feature fraction (denominator: retained contexts)",
                True,
            ),
            ("feature_counts", "Feature prevalence (denominator: 293 concepts)", True),
        ]
        for ax, (field, title, normalize) in zip(axes.flat, specs):
            panel = field + ("_fraction" if normalize else "_count")
            for c in self.main:
                matrix = self.matrix(c)
                counts = matrix[field]
                denominator = (
                    len(matrix["features"]) if field == "object_counts" else 293
                )
                values = counts / denominator if normalize else counts
                x = np.sort(values)
                ax.step(
                    x,
                    np.arange(1, len(x) + 1) / len(x),
                    where="post",
                    color=COLORS[c],
                    linestyle=["-", "--", "-."][self.all.index(c) % 3],
                    label=SHORT[c],
                )
                frames.append(
                    pd.DataFrame(
                        dict(
                            condition=c,
                            panel=panel,
                            item_id=(
                                matrix["words"]
                                if field == "object_counts"
                                else matrix["features"]
                            ),
                            estimate=values,
                            positive_count=counts,
                            denominator=denominator,
                            items=len(values),
                            metric=panel,
                        )
                    )
                )
            ax.set_title(title, fontsize=10)
            ax.set_ylabel("Cumulative fraction")
            ax.set_ylim(0, 1.03)
            if not normalize:
                ax.set_xscale("symlog", linthresh=1)
                ax.set_xlim(
                    0, max(self.matrix(c)[field].max() for c in self.main) * 1.08
                )
                ax.set_xlabel("Count (symlog; zero retained)")
            else:
                ax.set_xlim(0, 1)
                ax.set_xlabel("Fraction")
            ax.grid(alpha=0.2)
        axes[0, 1].legend(fontsize=8, loc="lower right")
        self.save(
            fig,
            "figure1b_feature_distributions",
            pd.concat(frames),
            "Question: how do discovery and completion change density? Empirical cumulative distributions include every retained feature and all 293 concepts, including feature-empty concepts. Upper panels show counts; lower panels normalize by retained feature inventory or 293 concepts. These are descriptive distributions, not seed replicates or confidence intervals. A large inventory alone does not imply dense individual representations.",
            level="raw_feature_matrix",
        )

    def rdms(self, conditions, name):
        order, indices = self.ordered_objects()
        ncols = 2 if len(conditions) > 1 else 1
        fig, axes = plt.subplots(
            int(np.ceil(len(conditions) / ncols)),
            ncols,
            figsize=(11, 5 * int(np.ceil(len(conditions) / ncols))),
            squeeze=False,
            layout="constrained",
        )
        sources = []
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad("#f3c6cf")
        groups = order.groupby("category", sort=False).rdm_index.agg(["min", "max"])
        for ax, c in zip(axes.flat, conditions):
            matrix = self.matrix(c)
            rdm = matrix["rdm"][np.ix_(indices, indices)]
            img = ax.imshow(
                rdm, vmin=0, vmax=1, cmap=cmap, interpolation="none", rasterized=True
            )
            centers = (groups["min"] + groups["max"]) / 2
            ax.set_xticks(centers, groups.index, rotation=60, ha="right", fontsize=8)
            ax.set_yticks(centers, groups.index, fontsize=8)
            for end in groups["max"].iloc[:-1]:
                ax.axvline(end + 0.5, color="white", alpha=0.3, linewidth=0.4)
                ax.axhline(end + 0.5, color="white", alpha=0.3, linewidth=0.4)
            ax.set_title(SHORT[c])
            i, j = np.triu_indices(293, 1)
            sources.append(
                pd.DataFrame(
                    dict(
                        condition=c,
                        panel=c,
                        object_1=order.object.to_numpy()[i],
                        object_2=order.object.to_numpy()[j],
                        item_id=[f"{a}:{b}" for a, b in zip(i, j)],
                        estimate=rdm[i, j],
                        defined=np.isfinite(rdm[i, j]),
                        items=len(i),
                        metric="cosine_distance",
                    )
                )
            )
        for ax in list(axes.flat)[len(conditions) :]:
            ax.set_visible(False)
        fig.colorbar(
            img,
            ax=list(axes.flat)[: len(conditions)],
            shrink=0.7,
            label="Cosine distance (pink = undefined)",
        )
        self.save(
            fig,
            name,
            pd.concat(sources),
            "Question: what object structure changes? All heatmaps use the same original Leuven category/file order, 293 objects, and distance scale 0-1. Pink marks undefined zero-vector cosine distances, including zero-vector diagonals; these are not valid zero distances. Off-diagonal source tables retain missing cells. Ordering is independent of model results. No model inference or large judgment scan was performed.",
            level="raw_feature_matrix",
        )

    def similarity_inputs(self):
        if hasattr(self, "trained_similarity"):
            return
        b = self.args.behavior_dir
        items = self.read(b / "similarity_levels_1_2_items.csv")
        check_unique(
            items, ["condition", "level", "dataset", "item_id"], "lower-level items"
        )
        self.raw_items = items[items.level.eq("raw_feature_matrix")].copy()
        semantic = items[items.level.eq("pretrained_semantic_activations")]
        reference = (
            semantic[semantic.condition.eq("human")]
            .set_index(["dataset", "item_id"])
            .sort_index()
        )
        for c in self.all:
            candidate = (
                semantic[semantic.condition.eq(c)]
                .set_index(["dataset", "item_id"])
                .sort_index()
            )
            if not reference.index.equals(candidate.index) or not np.array_equal(
                reference.prediction, candidate.prediction
            ):
                raise ValueError(
                    "Semantic baseline is not shared identically across conditions"
                )
        self.shared_items = reference.reset_index()
        self.shared_items["condition"] = "shared_encoder"
        rows = []
        for c in self.main:
            for seed in range(4):
                rows.append(
                    self.read(
                        b / "per_model" / f"{c}_seed_{seed}" / "similarity_items.csv"
                    )
                )
        self.trained_similarity = pd.concat(rows, ignore_index=True)
        check_unique(
            self.trained_similarity,
            ["condition", "seed", "dataset", "item_id"],
            "trained items",
        )
        coverage = (
            self.raw_items.groupby(["condition", "dataset"])
            .prediction.agg(items="size", finite_items=lambda x: np.isfinite(x).sum())
            .reset_index()
        )
        coverage["coverage"] = coverage.finite_items / coverage["items"]
        self.raw_coverage = coverage
        self.raw_coverage["estimate"] = self.raw_coverage.coverage
        self.source(
            "raw_similarity_item_coverage",
            coverage,
            task="similarity",
            level="raw_feature_matrix",
            metric="finite_prediction_coverage",
        )

    def similarity(self, conditions, name, common=False):
        self.similarity_inputs()
        if common:
            scores, membership = common_pair_scores(
                self.raw_items,
                self.trained_similarity,
                self.shared_items,
                conditions=self.main,
                finite_conditions=MAIN,
            )
            self.source(
                "common_finite_pair_membership",
                membership,
                task="similarity",
                level="raw_feature_matrix",
            )
            self.source("common_finite_pair_per_seed_scores", scores, task="similarity")
            trained = scores[scores.level.eq("fully_trained") & scores.seed.isna()]
            raw = scores[scores.level.eq("raw_feature_matrix")]
            baseline = scores[scores.level.eq("pretrained_semantic_activations")]
        else:
            trained = self.behavior[
                self.behavior.task.eq("similarity")
                & self.behavior.metric.eq("paper_statistic")
            ].copy()
            trained["level"] = "fully_trained"
            raw = self.lower[
                self.lower.level.eq("raw_feature_matrix")
                & self.lower.metric.eq("paper_statistic")
            ].copy()
            raw["condition"] = raw.base_condition
            baseline = self.lower[
                self.lower.level.eq("pretrained_semantic_activations")
                & self.lower.base_condition.eq("human")
                & self.lower.metric.eq("paper_statistic")
            ].copy()
            baseline["condition"] = "shared_encoder"
        fig, axes = plt.subplots(
            3, 3, figsize=(17, 14 if len(conditions) <= 8 else 22), layout="constrained"
        )
        exported = []
        minimum = min(raw.ci_low.min(), trained.ci_low.min())
        limits = (
            min(-0.3, np.floor(minimum * 10) / 10) if np.isfinite(minimum) else -0.3,
            1.02,
        )
        for ax, domain in zip(axes.flat, DOMAINS):
            base = baseline[baseline.dataset.eq(domain)].iloc[0]
            ax.axvline(base["mean"], color="#777777", linestyle="--", linewidth=1.2)
            if not common:
                ax.axvspan(base.ci_low, base.ci_high, color="#999999", alpha=0.13)
            reliability = self.reliability[
                self.reliability.task.eq("similarity")
                & self.reliability.dataset.eq(domain)
            ].iloc[0]
            rel = dict(
                condition="observed_human_reliability",
                task="similarity",
                dataset=domain,
                level="observed_human",
                metric="spearman_brown_reliability_mean",
                mean=reliability.spearman_brown_reliability_mean,
                ci_low=reliability.spearman_brown_low,
                ci_high=reliability.spearman_brown_high,
                uncertainty="saved 100 random participant split-half distribution; not a model CI",
                items=reliability.participants,
                panel=domain,
            )
            # The split-half range is a distribution summary, not a confidence interval.
            ax.plot(
                [rel["ci_low"], rel["ci_high"]],
                [-0.8, -0.8],
                color="#666666",
                linewidth=2,
            )
            ax.scatter(rel["mean"], -0.8, marker="|", color="#333333", s=100)
            exported.append(pd.DataFrame([rel]))
            for i, c in enumerate(conditions):
                for table, offset, fill in [(raw, -0.14, False), (trained, 0.14, True)]:
                    row = table[table.dataset.eq(domain) & table.condition.eq(c)].iloc[
                        0
                    ]
                    self.interval_point(ax, row, i + offset, c, fill)
                    record = row.to_dict()
                    record["panel"] = domain
                    if table is raw:
                        cov = self.raw_coverage[
                            self.raw_coverage.condition.eq(c)
                            & self.raw_coverage.dataset.eq(domain)
                        ].iloc[0]
                        record["original_items"] = cov["items"]
                        record["finite_items"] = (
                            row["items"] if common else cov.finite_items
                        )
                        record["coverage"] = record["finite_items"] / cov["items"]
                    exported.append(pd.DataFrame([record]))
            record = base.to_dict()
            record["panel"] = domain
            exported.append(pd.DataFrame([record]))
            if common:
                subtitle = f"N = {int(base['items'])}/{int(base.original_items)} common pairs; no subset CI"
            else:
                cov = self.raw_coverage[
                    self.raw_coverage.dataset.eq(domain)
                    & self.raw_coverage.condition.isin(conditions)
                ]
                subtitle = f"N = {int(base['items'])}; raw finite {int(cov.finite_items.min())}-{int(cov.finite_items.max())}"
            ax.set_title(f"{domain}\n{subtitle}", fontsize=10)
            self.format_axis(
                ax,
                conditions,
                limits,
                "Pearson r with observed human similarity",
                top=-1.3,
            )
        fig.legend(
            handles=[
                Line2D(
                    [],
                    [],
                    marker="o",
                    markerfacecolor="white",
                    color="black",
                    linestyle="none",
                    label="Raw features (open)",
                ),
                Line2D(
                    [],
                    [],
                    marker="o",
                    color="black",
                    linestyle="none",
                    label="Trained ISC-CI (filled)",
                ),
                Line2D(
                    [],
                    [],
                    color="#777777",
                    linestyle="--",
                    label="Shared pretrained encoder (once/domain)",
                ),
                Line2D(
                    [],
                    [],
                    color="#666666",
                    marker="|",
                    label="Observed humans: split-half Spearman-Brown",
                ),
            ],
            loc="outside lower center",
            ncol=2,
        )
        self.save(
            fig,
            name,
            pd.concat(exported, ignore_index=True),
            "Question: which levels predict actual human similarity? Nine original domains use Pearson r, never one pooled correlation. Open markers are raw binary-feature cosines, filled markers are four-seed ISC-CI means. The shared frozen human-pretrained encoder appears once per domain (human-condition saved baseline interval). The upper gray marker/range summarizes observed-human split-half Spearman-Brown reliability, not an unconditional noise ceiling. "
            + (
                "This sensitivity restricts all levels to the domain-specific intersection of finite raw predictions across seven main conditions; item counts and inclusion IDs are exported. Points have no new subset confidence intervals. "
                if common
                else "Primary statistics preserve original coverage; sparse cosines stay missing. Intervals are the saved 95% item/seed percentile bootstrap (one nontrained representation for raw/encoder, four training seeds for ISC-CI). Finite raw counts and missingness accompany every domain. "
            )
            + "Source-only's similarity advantage is already present in raw features; coarse Leuven recovery is not a universal behavioral ranking.",
            task="similarity",
        )

    @staticmethod
    def interval_point(ax, row, y, condition, filled=True):
        if np.isfinite(row.get("ci_low", np.nan)) and np.isfinite(
            row.get("ci_high", np.nan)
        ):
            ax.plot(
                [row.ci_low, row.ci_high],
                [y, y],
                color=COLORS[condition],
                alpha=0.65,
                linewidth=1.5,
            )
        ax.scatter(
            row["mean"],
            y,
            marker=MARKERS[condition],
            s=45,
            facecolors=COLORS[condition] if filled else "white",
            edgecolors=COLORS[condition],
            zorder=3,
        )

    @staticmethod
    def format_axis(ax, conditions, limits, xlabel, top=-0.7):
        ax.set_yticks(
            range(len(conditions)), [SHORT[c] for c in conditions], fontsize=9
        )
        ax.set_ylim(len(conditions) - 0.4, top)
        ax.set_xlim(*limits)
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", alpha=0.2)

    def induction(self, conditions, name):
        fig, axes = plt.subplots(
            2, 3, figsize=(17, 10 if len(conditions) <= 8 else 15), layout="constrained"
        )
        selected = self.behavior[
            self.behavior.task.eq("induction")
            & self.behavior.metric.eq("paper_statistic")
            & self.behavior.condition.isin(conditions)
        ].copy()
        for ax, dataset in zip(axes.flat, INDUCTION):
            frame = selected[selected.dataset.eq(dataset)]
            for i, c in enumerate(conditions):
                self.interval_point(ax, frame[frame.condition.eq(c)].iloc[0], i, c)
            self.format_axis(
                ax, conditions, (-0.5, 1), "Pearson r with observed human induction"
            )
            ax.set_title(
                f"{dataset}\nN = {int(frame['items'].iloc[0])} human-rated arguments"
            )
        axes.flat[-1].set_visible(False)
        selected["panel"] = selected.dataset
        self.save(
            fig,
            name,
            selected,
            "Question: are gains consistent across induction datasets? Original Pearson correlations for five datasets, with saved crossed item/seed 95% intervals and four-seed means; N is the original argument count. Synthetic phenomenon examples are excluded. Completion helps some induction tasks and not others. Intervals are not simultaneous multiple-comparison intervals.",
            level="fully_trained",
            task="induction",
        )

    def preferences(self, conditions, name):
        b = self.args.behavior_dir
        humans = self.read(b / "paper_human_participant_choices.csv")
        statistics = self.read(b / "paper_human_choice_statistics.csv")
        fig, axes = plt.subplots(
            1, 2, figsize=(15, 6 if len(conditions) <= 8 else 9), layout="constrained"
        )
        selected = self.behavior[
            self.behavior.task.isin(TASK_LABELS)
            & self.behavior.metric.eq("paper_statistic")
            & self.behavior.condition.isin(conditions)
        ].copy()
        source = []
        for ax, task in zip(axes, TASK_LABELS):
            human = humans[humans.task.eq(task)]
            summary = statistics[statistics.task.eq(task)].iloc[0]
            ax.boxplot(
                [human.argument2_preference.to_numpy()],
                positions=[-1.35],
                orientation="horizontal",
                widths=0.5,
                showfliers=False,
                manage_ticks=False,
                patch_artist=True,
                boxprops={"facecolor": "#dddddd"},
                medianprops={"color": "black"},
            )
            jitter = np.linspace(-0.17, 0.17, len(human))
            ax.scatter(
                human.argument2_preference,
                -1.35 + jitter,
                color="#666666",
                s=10,
                alpha=0.7,
            )
            ax.axvline(
                summary.paper_participant_median,
                color="#666666",
                linestyle=":",
                linewidth=1,
            )
            ax.text(
                0.02,
                -0.7,
                f"Observed participants N={len(human)}; median={summary.paper_participant_median:.3f}",
                fontsize=9,
            )
            frame = selected[selected.task.eq(task)]
            for i, c in enumerate(conditions):
                self.interval_point(ax, frame[frame.condition.eq(c)].iloc[0], i, c)
            self.format_axis(
                ax,
                conditions,
                (0, 1.02),
                "Argument-2 preference rate (not accuracy)",
                top=-2,
            )
            ax.set_title(TASK_LABELS[task])
            frame = frame.copy()
            frame["panel"] = task
            source.append(frame)
            observed = human.rename(
                columns={
                    "argument2_preference": "estimate",
                    "participant_id": "item_id",
                }
            ).copy()
            observed["condition"], observed["level"], observed["panel"] = (
                "observed_human",
                "observed_human",
                task,
            )
            observed["dataset"] = task
            observed["metric"], observed["items"] = (
                "participant_argument2_fraction",
                len(human),
            )
            source.append(observed)
        self.save(
            fig,
            name,
            pd.concat(source, ignore_index=True),
            "Question: do models reproduce contextual argument preferences? 'thematic' is context-specific versus taxonomic categorization (supplied paper Fig. 6); 'nonmonotonicity' contrasts single- versus two-premise arguments (Fig. 5). Human dots/boxes are participant fractions, with their participant median shown separately from original model item-level rates. Model points use saved four-seed means and crossed item/seed 95% intervals. Rates are not accuracy and human participant medians are not individual-item fit. The completed ensemble does not recover both preferences to the human-trained baseline. Companion item-fit and paired-contrast tables distinguish overall preference from predicting which items humans choose.",
            level="fully_trained",
        )
        metrics = self.behavior[self.behavior.task.isin(TASK_LABELS)].copy()
        self.source(
            "preference_item_fit_all_conditions", metrics, level="fully_trained"
        )
        participant_rows = []
        for c in conditions:
            for seed in range(4):
                participant_rows.append(
                    self.read(
                        b
                        / "per_model"
                        / f"{c}_seed_{seed}"
                        / "paper_participant_predictions.csv"
                    )
                )
        participant_predictions = pd.concat(participant_rows)
        participant_predictions["estimate"] = (
            participant_predictions.model_argument2_fraction
        )
        participant_predictions["items"] = participant_predictions.original_trials
        self.source(
            name + "_definition_matched_participants",
            participant_predictions,
            level="fully_trained",
            metric="within_participant_argument2_fraction",
            subset="same original participant trials",
        )
        self.captions[-1] += (
            "\nCompanion sources: [item-fit metrics](source_data/preference_item_fit_all_conditions.csv), "
            f"[definition-matched participant predictions](source_data/{name}_definition_matched_participants.csv), "
            "and [saved paired contrasts](source_data/all_paired_contrasts.csv).\n"
        )

    def effects(self, task):
        attribute = task + "_items"
        if not hasattr(self, attribute):
            frames = []
            for c in self.all:
                for seed in range(4):
                    frame = self.read(
                        self.args.behavior_dir
                        / "per_model"
                        / f"{c}_seed_{seed}"
                        / f"{task}_items.csv"
                    )
                    check_unique(frame, ["item_id"], f"{task} items {c}/{seed}")
                    verify_effects(frame, task)
                    expected_n, expected_selected = (
                        (41, 38) if task == "context" else (50, 34)
                    )
                    if (
                        len(frame) != expected_n
                        or frame.paper_selected.sum() != expected_selected
                    ):
                        raise ValueError(f"Unexpected frozen {task} subset: {c}/{seed}")
                    frames.append(frame)
            all_items = pd.concat(frames, ignore_index=True)
            reference = all_items[
                all_items.condition.eq("human") & all_items.seed.eq(0)
            ].set_index("item_id")
            for (c, seed), frame in all_items.groupby(["condition", "seed"]):
                candidate = frame.set_index("item_id").reindex(reference.index)
                if set(frame.item_id) != set(reference.index) or not np.array_equal(
                    candidate.human, reference.human
                ):
                    raise ValueError(f"Unstable {task} human item join: {c}/{seed}")
            setattr(self, attribute, all_items)
        return getattr(self, attribute)

    def direction(self, task, conditions, name):
        self.effects(task)
        frame = self.behavior[
            self.behavior.task.eq(task)
            & self.behavior.metric.eq("paper_statistic")
            & self.behavior.condition.isin(conditions)
        ].copy()
        n = 38 if task == "context" else 34
        fig, ax = plt.subplots(
            figsize=(10, max(5, len(conditions) * 0.43)), layout="constrained"
        )
        for i, c in enumerate(conditions):
            self.interval_point(ax, frame[frame.condition.eq(c)].iloc[0], i, c)
        self.format_axis(
            ax, conditions, (0, 1.02), f"Original direction agreement (denominator {n})"
        )
        ax.set_title(
            f"{'Context effects: 38 human-positive sets' if task=='context' else 'Asymmetry: 34 significant human pairs'}"
        )
        self.save(
            fig,
            name,
            frame,
            f"Question: does the model favor the human direction? Original {task} direction statistic on {n} fixed human-defined items, not the full scatter subset. Saved crossed item/seed 95% intervals accompany four-seed means. A high direction score does not establish matched magnitudes or item-wise correlation.",
            task=task,
            level="fully_trained",
            subset=f"original {n} human-selected items",
            selected_items=n,
        )

    def scatter(self, task, conditions, name):
        items = self.effects(task)
        averaged = (
            items.groupby(["condition", "item_id"], sort=False)
            .agg(
                human=("human", "first"),
                prediction=("prediction", "mean"),
                selected=("paper_selected", "first"),
            )
            .reset_index()
        )
        ncols = min(3, len(conditions))
        nrows = int(np.ceil(len(conditions) / ncols))
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(5 * ncols, 4.5 * nrows),
            squeeze=False,
            layout="constrained",
        )
        xmin, xmax = (
            (
                min(-0.2, averaged.human.min() - 0.08),
                max(1.0, averaged.human.max() + 0.08),
            )
            if task == "context"
            else (0, 1)
        )
        ymin, ymax = (
            min(0, averaged.prediction.min()) - 0.15,
            averaged.prediction.max() + 0.15,
        )
        for ax, c in zip(axes.flat, conditions):
            frame = averaged[averaged.condition.eq(c)]
            for selected, marker in [(True, "o"), (False, "x")]:
                rows = frame[frame.selected.eq(selected)]
                ax.scatter(
                    rows.human,
                    rows.prediction,
                    marker=marker,
                    s=32,
                    color=COLORS[c],
                    alpha=0.85,
                )
            ax.axhline(0, color="#888888", linewidth=0.8)
            ax.axvline(0 if task == "context" else 0.5, color="#888888", linewidth=0.8)
            if task == "context":
                ax.plot(
                    [xmin, xmax],
                    [xmin, xmax],
                    linestyle=":",
                    color="#999999",
                    linewidth=1,
                )
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.set_xlabel(
                "Observed human context E"
                if task == "context"
                else "Observed human argument-2 fraction"
            )
            ax.set_ylabel(
                "Model context E (four-seed mean)"
                if task == "context"
                else "Model argument-2 minus argument-1 logit"
            )
            ax.set_title(SHORT[c])
            ax.grid(alpha=0.15)
        for ax in list(axes.flat)[len(conditions) :]:
            ax.set_visible(False)
        selected_items = items[items.condition.isin(conditions)].copy()
        selected_items["panel"] = selected_items.condition
        selected_items["items"] = 41 if task == "context" else 50
        selected_items["coverage"] = 1.0
        self.save(
            fig,
            name,
            selected_items,
            "Question: does the model predict effect magnitude and item variation? "
            + (
                "All 41 original context sets are plotted, not just the 38 human-positive sets; crosses mark the three nonpositive-human sets. E=(p(a|c_a)-p(b|c_a))+(p(b|c_b)-p(a|c_b)) for both humans and models, retaining all three choice alternatives. The identity line is justified by the shared probability-difference units, not a fitted mapping. "
                if task == "context"
                else "All 50 original asymmetry pairs are plotted; crosses mark the 16 pairs outside the 34 significant-human direction subset. Model magnitude is the argument-2 minus argument-1 logit, while human magnitude is argument-2 choice fraction; these are different scales, so no identity line is drawn. "
            )
            + "Dots are four-seed means; the source tables retain stable item IDs, every seed, observed humans, stimulus orientation, and saved CV predictions. Common axes include every condition. Zero/reference lines show direction. No new mapping, temperature, or rescaling was fit. High direction can coexist with weak magnitude prediction.",
            task=task,
            level="fully_trained",
            metric="item_effect",
            uncertainty="scatter: four-seed descriptive mean; underlying seed rows exported",
        )

    def effect_table(self):
        items = self.effects("context")
        mean_effects = (
            items.groupby(["condition", "seed"])
            .agg(human_mean=("human", "mean"), model_mean=("prediction", "mean"))
            .reset_index()
        )
        mean_effects["model_minus_human_mean"] = (
            mean_effects.model_mean - mean_effects.human_mean
        )
        self.source(
            "context_mean_effect_per_seed",
            mean_effects,
            task="context",
            level="fully_trained",
            metric="descriptive_mean_effect",
            items=41,
        )
        metrics = self.behavior[self.behavior.task.eq("context")].copy()
        self.source(
            "context_effect_metrics_all_conditions",
            metrics,
            task="context",
            level="fully_trained",
        )
        means = mean_effects.groupby("condition").mean(numeric_only=True)
        table = (
            metrics.pivot(index="condition", columns="metric", values="mean")
            .join(means.drop(columns="seed"))
            .reindex(self.all)
        )
        table.insert(0, "display_label", [LABELS[c] for c in self.all])
        table.to_csv(self.out / "context_effect_summary.csv")
        rows = [
            [
                SHORT[c],
                *[
                    f"{table.loc[c, col]:.3f}"
                    for col in [
                        "pearson_r",
                        "spearman_r",
                        "direct_rmse",
                        "human_mean",
                        "model_mean",
                        "model_minus_human_mean",
                    ]
                ],
            ]
            for c in self.main
        ]
        fig, ax = plt.subplots(figsize=(13, 4), layout="constrained")
        ax.axis("off")
        artist = ax.table(
            cellText=rows,
            colLabels=[
                "Condition",
                "Pearson r",
                "Spearman r",
                "Direct RMSE",
                "Human mean E",
                "Model mean E",
                "Mean difference",
            ],
            loc="center",
            colWidths=[0.23, 0.115, 0.115, 0.115, 0.13, 0.13, 0.135],
        )
        artist.auto_set_font_size(False)
        artist.set_fontsize(10)
        artist.scale(1, 1.9)
        ax.set_title(
            "Context magnitudes: all 41 sets; descriptive four-seed means", pad=12
        )
        self.save(
            fig,
            "figure4c_context_magnitude_table",
            table.loc[self.main]
            .reset_index()
            .melt(
                id_vars=["condition", "display_label"],
                value_vars=[
                    "pearson_r",
                    "spearman_r",
                    "direct_rmse",
                    "human_mean",
                    "model_mean",
                    "model_minus_human_mean",
                ],
                var_name="metric",
                value_name="estimate",
            ),
            "Question: are contextual shifts calibrated? Pearson/Spearman and direct probability-effect RMSE are saved all-41-item metrics averaged across seeds. Mean human E, mean model E, and model-minus-human mean E are descriptive transformations of unchanged predictions, with per-seed source rows and saved metric intervals supplied separately. Mean difference is not a complete item-wise calibration test; saved CV RMSE/MAE remain separately labeled in source tables. Weak effect correlations can accompany high 38-item direction agreement.",
            task="context",
            level="fully_trained",
            items=41,
        )
        self.source(
            "asymmetry_effect_metrics_all_conditions",
            self.behavior[self.behavior.task.eq("asymmetry")],
            task="asymmetry",
            level="fully_trained",
        )

    def representations(self):
        fields = [
            "input_object_rdm_spearman",
            "context_rdm_spearman",
            "context_dependent_rdm_spearman_median",
            "membership_logit_spearman",
        ]
        titles = [
            "Input object geometry",
            "Inferred-context geometry",
            "Context-dependent geometry",
            "Membership-logit ranks",
        ]
        fig, axes = plt.subplots(2, 2, figsize=(14, 13), layout="constrained")
        frames = []
        for ax, metric, title in zip(axes.flat, fields, titles):
            for i, c in enumerate(self.all):
                values = self.rep[self.rep.condition.eq(c)][metric]
                if metric.startswith("input"):
                    ax.scatter(
                        values.iloc[0], i, color=COLORS[c], marker=MARKERS[c], s=45
                    )
                else:
                    ax.scatter(
                        values,
                        i + np.linspace(-0.12, 0.12, 4),
                        color=COLORS[c],
                        marker=MARKERS[c],
                        s=25,
                        alpha=0.65,
                    )
                    ax.plot(
                        [
                            values.mean() - values.std(ddof=1),
                            values.mean() + values.std(ddof=1),
                        ],
                        [i, i],
                        color=COLORS[c],
                    )
                self.format_axis(
                    ax,
                    self.all,
                    (-0.08 if self.args.include_random_control else 0, 1.04),
                    "Agreement with reconstructed human reference",
                )
                frames.append(
                    pd.DataFrame(
                        dict(
                            condition=c,
                            seed=(
                                np.nan
                                if metric.startswith("input")
                                else self.rep[self.rep.condition.eq(c)].seed
                            ),
                            estimate=(
                                [values.iloc[0]]
                                if metric.startswith("input")
                                else values.to_numpy()
                            ),
                            metric=metric,
                            panel=metric,
                            uncertainty=(
                                "input matrix has no training seeds"
                                if metric.startswith("input")
                                else "points: four seeds; horizontal segment: mean +/- seed SD, NOT 95% CI"
                            ),
                        )
                    )
                )
            ax.set_title(title)
        self.save(
            fig,
            "supplement_representation_all13",
            pd.concat(frames),
            "Question: which representations are preserved? Input, inferred-context, context-dependent, and membership-rank agreements remain separate. Input has one fixed value; trained panels show four replicates and mean +/- seed SD, not a 95% confidence interval. The frozen human-pretrained semantic encoder is shared, so its activation agreement is exactly 1 by construction, not independently learned LLM semantic recovery. References are not pooled into the four-seed reconstructed human control.",
            level="representation",
        )

    def forests(self):
        for a, b in CONTRASTS + (
            [("human", RANDOM)] if self.args.include_random_control else []
        ):
            frame = self.contrasts[
                self.contrasts.condition_a.eq(a)
                & self.contrasts.condition_b.eq(b)
                & self.contrasts.metric.eq("paper_statistic")
            ].copy()
            check_unique(
                frame, ["task", "dataset"], "paired original-outcome contrasts"
            )
            if len(frame) != 18:
                raise ValueError(f"Expected all 18 original outcomes: {a} minus {b}")
            # Correlations and proportions remain on separate axes despite both being bounded.
            fig, axes = plt.subplots(
                1,
                2,
                figsize=(15, 8),
                layout="constrained",
                gridspec_kw={"width_ratios": [1.45, 1]},
            )
            for ax, kind in zip(axes, ["correlation", "proportion"]):
                rows = frame[
                    frame.task.isin(["similarity", "induction"])
                    == (kind == "correlation")
                ].copy()
                rows["outcome"] = rows.task + ": " + rows.dataset
                for i, row in enumerate(rows.itertuples()):
                    ax.plot(
                        [row.ci_low, row.ci_high],
                        [i, i],
                        color=COLORS[a],
                        linewidth=1.5,
                    )
                    ax.scatter(
                        row.difference_a_minus_b,
                        i,
                        color=COLORS[a],
                        marker=MARKERS[a],
                        s=45,
                    )
                ax.set_yticks(range(len(rows)), rows.outcome, fontsize=9)
                ax.set_ylim(len(rows) - 0.4, -0.6)
                bound = max(
                    0.8,
                    np.ceil(max(rows.ci_low.abs().max(), rows.ci_high.abs().max()) * 10)
                    / 10,
                )
                ax.set_xlim(-bound, bound)
                ax.axvline(0, color="#666666", linestyle=":")
                ax.set_xlabel(
                    "Difference in Pearson r"
                    if kind == "correlation"
                    else "Difference in direction / argument-2 rate"
                )
                ax.set_title(
                    "14 correlation outcomes"
                    if kind == "correlation"
                    else "4 original proportion outcomes"
                )
                ax.grid(axis="x", alpha=0.2)
            fig.suptitle(f"{SHORT[a]} minus {SHORT[b]}", fontsize=13)
            frame["panel"] = np.where(
                frame.task.isin(["similarity", "induction"]),
                "correlation",
                "proportion",
            )
            frame["condition"], frame["estimate"] = a, frame.difference_a_minus_b
            self.save(
                fig,
                f"supplement_contrast_{a}_minus_{b}",
                frame,
                "Question: what changes under completion or calibration? All 18 saved original outcomes are shown, including unfavorable contrasts. Positive means condition A exceeds B in the original metric, not universally better human prediction: argument-2 rates require comparison with observed preference. Correlation and proportion differences have separate axes. Intervals are saved paired crossed cluster-item/seed percentile 95% intervals, unadjusted for multiplicity. Supplemental Spearman and saved CV-error contrasts are retained in the all-paired-contrasts source table; CV maps were not refit in bootstraps.",
                level="fully_trained",
                uncertainty=DEFINITIONS["intervals"],
            )

    def coverage(self):
        self.similarity_inputs()
        self.structure(self.all, "supplement_inventory_geometry_coverage_all13")
        values = self.raw_coverage.pivot(
            index="condition", columns="dataset", values="coverage"
        ).reindex(index=self.all, columns=DOMAINS)
        fig, ax = plt.subplots(figsize=(12, 8), layout="constrained")
        im = ax.imshow(values, vmin=0, vmax=1, cmap="cividis", aspect="auto")
        ax.set_xticks(range(9), DOMAINS, rotation=35, ha="right")
        ax.set_yticks(range(len(self.all)), [SHORT[c] for c in self.all])
        for i in range(len(self.all)):
            for j in range(9):
                ax.text(
                    j,
                    i,
                    f"{values.iloc[i,j]:.0%}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white" if values.iloc[i, j] < 0.5 else "black",
                )
        fig.colorbar(im, ax=ax, label="Finite raw-feature pair fraction")
        ax.set_title(
            "Raw-feature similarity coverage; missing cosine is not negative evidence"
        )
        self.save(
            fig,
            "supplement_raw_pair_coverage_all13",
            self.raw_coverage,
            "Question: how much raw similarity evidence is defined? Fraction finite raw-feature cosine predictions on each original domain's human-rated pairs. Counts and finite counts are exported. Sparse zero-vector predictions remain missing. The primary comparisons keep original coverage; the separate seven-condition common-pair analysis tests sensitivity without changing original results. Source-only and completed inventories differ; fixed-B completion retains the same 175-feature vocabulary as V3.1-B.",
            task="similarity",
            level="raw_feature_matrix",
            metric="finite_prediction_coverage",
        )

    def finalize(self):
        changed = [p for p, digest in self.sources.items() if sha256(Path(p)) != digest]
        if changed:
            raise ValueError(f"Original inputs changed: {changed}")
        source_note = (
            "Figure precedents: NOVA arXiv:2505.10718v1 (15 May 2025), Figures 4-5, "
            "reviewed HTML https://arxiv.org/html/2505.10718v1; journal DOI 10.1111/tops.70037 is a separate version. "
            "ISC-CI: supplied ISCCI_2026Jan_ArXiv.pdf (104 pages), Figures 2, 5-8, 10; "
            "versioned reference https://osf.io/preprints/psyarxiv/fxc87_v3 (web access returned 403; local supplied paper reviewed). "
            "Figure numbering follows that local PDF, not an assumed later revision."
        )
        methods = """# Final ISC-CI Results: Figure Captions

## Shared Methods

Reporting only: no judgments, training, checkpoint inference, model simulations, or giant-cell evaluation. Thirteen conditions have seeds 0-3; released checkpoints are fidelity references, never extra human replicates. All models share the frozen human-pretrained 64-dimensional semantic encoder. Only context norms change. Each condition received 409,600 episodes: the completed ensemble receives roughly five episodes per retained context. Source-only versus completion also changes feature inventory and applicability, so salience, weighting, and exposure are not isolated causes.

Locked scores use value > 0, calibrated scores value >= 1, retained contexts positive_object_count > 3. The 127 unresolved scores caused exclusion of 64 entire candidate columns; missing scores were not encoded as negative judgments. Fixed-B lost no contexts. No human behavioral outcome changed thresholds, features, stimuli, settings, or condition selection. V3-B is prompt-matched, not selected by its result.

Saved 95% bootstrap intervals cross item clusters and model seeds; raw features/shared activations have one nontrained representation. CV mappings were fitted in saved held-out folds and not refit in bootstraps. Seed SD and reliability split-half distribution ranges are not model confidence intervals. Common-pair sensitivities have point estimates only. Missing panels are documented explicitly, never filled from historical LCA runs.

"""
        if self.args.include_random_control:
            methods = methods.replace("Thirteen conditions", "Fourteen conditions")
            methods += (
                "## Random-feature Control\n\n"
                "The original 13 conditions are reused unchanged; four additional random-context models were trained before this reporting pass. "
                "Random-feature ISC-CI uses the human matrix's 293 objects and 385 retained features, with each feature's object assignments "
                f"independently permuted once using seed {self.protocol['random_control']['randomization_seed']}. Context count, individual feature prevalence, total positives, encoder, "
                "architecture, and training budget are matched to the human control. Per-object feature counts are not preserved. "
                "Training seeds 0-3 share this one fixed random matrix; intervals do not cover randomization-realization uncertainty. "
                "This is a meaningful-context control conditional on the same human-pretrained encoder, not an encoder ablation. "
                "The common-finite-pair sensitivity retains the original seven-condition intersection when adding the random row.\n\n"
            )
        (self.out / "captions.md").write_text(
            methods + source_note + "\n\n" + "\n".join(self.captions)
        )
        interpretation = """# Final Results Interpretation

## Established

The experiment replaces ISC-CI context-feature norms while retaining the common human-pretrained semantic encoder and the frozen original tasks. Deterministic released-checkpoint statistics reproduced within numerical tolerance, and reconstructed-human seeds passed statistical consistency checks. The authoritative common-LCA-seed outputs remain separate from historical stochastic notebook realizations. This fidelity gate establishes consistent evaluation, not formal equivalence of candidate models.

[Figure 1](figure1a_inventory_geometry.pdf) shows that completion improves agreement with Leuven object geometry. The vocabulary-matched fixed-B calibrated comparison moves input-RDM agreement from about .241 (V3.1-B) to .762 with the same 175 contexts. The completed ensemble reaches about .821, versus .641 for source-only. The [distribution panels](figure1b_feature_distributions.pdf) and [RDMs](figure1c_object_rdms.pdf) distinguish inventory size, density, and undefined sparse cosines.

These geometric gains are not a universal behavioral gain. [Figure 2](figure2_similarity.pdf) keeps all nine original domains separate. Source-only has descriptive mean trained similarity r about .575 versus .520 for the completed ensemble. That advantage already appears in raw-feature prediction (.593 versus .524), before training. The shared pretrained encoder's descriptive mean is .539. Sparse V3/V3.1 coverage is shown separately; [common-finite-pair sensitivity](figure2s_common_finite_pairs.pdf) changes only the displayed subset, not the original score. These unweighted domain means are descriptive, not pooled paper statistics.

[Figure 3A](figure3a_induction.pdf) shows task-dependent induction changes: completion improves several datasets, but not every dataset. [Figure 3B](figure3b_contextual_preferences.pdf) shows a different limitation. The completed ensemble's argument-2 rates are about .293 for context-specific categorization and .491 for nonmonotonicity, while observed participant medians are about .843 and .848. Participant medians and model item rates are explicitly different summaries; the exported definition-matched participant predictions and item-fit metrics provide the relevant detailed comparisons. A preference rate is not accuracy.

[Figure 4A](figure4a_context_direction.pdf) and [Figure 4B](figure4b_context_magnitude.pdf) separate direction from magnitude. Many conditions favor the human direction on the original 38 positive sets, yet the all-41-item human/model effect correlation is weak and average model shifts differ substantially. Both scores preserve the full three-choice denominator; no normalization or fitted main-panel scaling removes the discrepancy. The [magnitude table](figure4c_context_magnitude_table.pdf) separates direct error, item-wise correlation, and descriptive mean difference. Asymmetry uses its own 34/50 subsets, not the context denominators.

All thirteen conditions and their seed means, SDs, original task statistics, uncertainty, and representational levels remain in [the complete table](all_condition_summary.csv). [Representational supplements](supplement_representation_all13.pdf) show that input recovery, inferred-context recovery, dependent geometry, and membership recovery can diverge. Four prespecified forest plots retain favorable and unfavorable outcomes; their unadjusted intervals are not a universal ranking. Greater selectivity and greater inventory coverage do not consistently improve every human task.

## Unresolved

Source-only/completion differences combine inventory, applicability, redundancy, and effective training exposure. The ensemble has roughly five training episodes per context under the common fixed budget; this may matter, but cannot explain an advantage already visible before training. This experiment does not isolate these causes. Related conditions are not independent replications, and no new outcome-driven optimization was performed. Better global Leuven RDM recovery does not guarantee better within-domain or contextual human prediction; the saved exploratory geometry/behavior associations vary by task and remain unadjusted.

The reconstructed-human fidelity gate checks replication consistency, not formal candidate equivalence. Reliability summaries characterize the available original participants, not guaranteed noise ceilings for models. Original model-selected stimuli limit claims of independent generalization. Raw-feature missingness, and CV maps fixed during bootstrap, remain measurement limitations.

## Implications

LLM-derived norms can preserve useful object structure and some behavioral predictions, but the present evidence supports task-specific validity rather than a wholly human-data-free cognitive model. The common encoder still contains human-derived information. The completed ensemble is neither uniformly superior nor behaviorally interchangeable with human norms.

Arbitrary-word feature discovery and false-memory/DRM predictions remain future external tests. Current figures evaluate prerequisites: retained feature structure and human judgments on the paper's fixed stimuli. They do not establish arbitrary-word generalization or validated false-memory behavior. Any later changes to training budget, feature weighting, or feature discovery require a new separately documented experiment, not retrospective selection using these external human outcomes.
"""
        if self.args.include_random_control:
            values = self.seeds[self.seeds.condition.eq(RANDOM)]
            means = {
                task: values[values.task.eq(task)].paper_statistic.mean()
                for task in ["similarity", "induction", "context"]
            }
            raw_mean = self.lower[
                self.lower.base_condition.eq(RANDOM)
                & self.lower.level.eq("raw_feature_matrix")
                & self.lower.metric.eq("paper_statistic")
            ]["mean"].mean()
            input_rdm = self.rep[
                self.rep.condition.eq(RANDOM)
            ].input_object_rdm_spearman.iloc[0]
            interpretation = interpretation.replace(
                "All thirteen conditions", "All fourteen conditions"
            )
            interpretation = interpretation.replace(
                "supplement_representation_all13.pdf",
                "supplement_representation_all14.pdf",
            )
            interpretation += (
                "\n## Random-Feature Negative Control (October 28, 2026)\n\n"
                f"The added human-sized control has input-RDM agreement {input_rdm:.3f}, raw-feature domain-mean similarity r {raw_mean:.3f}, "
                f"trained domain-mean similarity r {means['similarity']:.3f}, dataset-mean induction r {means['induction']:.3f}, "
                f"and 38-item context-direction agreement {means['context']:.3f}. Correlation means are descriptive unweighted means, not pooled statistics. "
                "The [human-minus-random contrast](supplement_contrast_human_minus_random_human.pdf) retains all original task outcomes and saved-unit differences.\n\n"
                "All encoder weights and the human-sized architecture/training budget are matched; only context memberships are shuffled, preserving each feature's prevalence. "
                "Any behavior retained by the random control is compatible with contributions from the shared semantic encoder and the remaining trained machinery; "
                "it cannot be attributed exclusively to the encoder. A difference from the human control tests meaningful assignments at the human context count, "
                "not every LLM inventory size. Per-object feature counts change, and one random matrix is shared across four training seeds, "
                "so this is not a distribution over independent random-feature realizations. No result was used to choose or revise the randomization.\n"
            )
        if self.blocked:
            interpretation += (
                "\n## Blocked Panels\n\n"
                + "\n".join(
                    f"- {r['panel_group']}: {r['missing_input']}" for r in self.blocked
                )
                + "\n"
            )
        (self.out / "FINAL_RESULTS_INTERPRETATION.md").write_text(interpretation)
        manifest = dict(
            protocol=self.protocol["protocol_version"],
            complete=not self.blocked,
            main_condition_order=self.main,
            all_condition_order=self.all,
            display_labels={c: LABELS[c] for c in self.all},
            styles={c: {"color": COLORS[c], "marker": MARKERS[c]} for c in self.all},
            random_control=self.protocol.get("random_control"),
            metric_definitions=DEFINITIONS,
            definition_references={
                "matrix_cosines": "iscci_validation/evaluation.py::_ranked_cosine_rdm (same scipy pdist definition; undefined visualization preserved)",
                "similarity_and_induction": "iscci_validation/behavior.py::metrics_for_values",
                "context_effects": "iscci_validation/simulations.py::human_similarity_context_effects and model_similarity_context_predictions",
                "asymmetry_and_subsets": "iscci_validation/behavior.py::item_table and predict_task",
            },
            sources=[{"path": p, "sha256": h} for p, h in self.sources.items()],
            inherited_provenance={
                k: self.protocol.get(k)
                for k in [
                    "matrix_sha256",
                    "human_datasets",
                    "implementation_sha256",
                    "paper_sha256",
                    "config_sha256",
                ]
            },
            input_hashes_verified_unchanged=True,
            figures=self.figures,
            blocked_panels=self.blocked,
            source_note=source_note,
            renderer_sha256=sha256(Path(__file__)),
            rendering={
                "png_dpi": 300,
                "matplotlib": matplotlib.__version__,
                "numpy": np.__version__,
                "pandas": pd.__version__,
            },
            cli={k: str(v) for k, v in vars(self.args).items()},
            no_model_calls=True,
        )
        panel_sources = {
            name for figure in self.figures for name in figure["source_data"]
        }
        obsolete = {
            "figure1a_inventory_geometry_geometry_inventory.csv",
            "supplement_inventory_geometry_coverage_all13_geometry_inventory.csv",
        }
        manifest["auxiliary_source_data"] = [
            {"path": str(path.relative_to(self.out)), "sha256": sha256(path)}
            for path in sorted((self.out / "source_data").glob("*.csv"))
            if str(path.relative_to(self.out)) not in panel_sources
            and path.name not in obsolete
        ]
        (self.out / "figure_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        index = [
            "# Final ISC-CI Figure Package",
            "",
            f"Four main figures are split into readable panels. All {len(self.all)} four-seed conditions are retained in the supplements and tables. The plotting command does not run models.",
            "",
            "- [Interpretation](FINAL_RESULTS_INTERPRETATION.md)",
            "- [Captions, methods, and panel-source links](captions.md)",
            "- [All-condition summary](all_condition_summary.csv)",
            "- [Context magnitude summary](context_effect_summary.csv)",
            "- [Provenance and blocked-input record](figure_manifest.json)",
            "- [Visual QA](visual_qa.json)",
            "",
            "## Main Figures",
            "",
        ]
        for figure in self.figures:
            if figure["figure"].startswith("figure"):
                name = figure["figure"]
                index.append(
                    f"- [{name}]({name}.pdf): [PNG]({name}.png), [SVG]({name}.svg)"
                )
        index += ["", "## Supplements", ""]
        for figure in self.figures:
            if figure["figure"].startswith("supplement"):
                name = figure["figure"]
                index.append(
                    f"- [{name}]({name}.pdf): [PNG]({name}.png), [SVG]({name}.svg)"
                )
        index += [
            "",
            "## Replay",
            "",
            "From the project root:",
            "",
            "```bash",
            "python ISC-CI_LLM_validation/plot_finalized_v4_results.py",
            "```",
            "",
            "Dependencies and explicit input/output options: [FINAL_FIGURES.md](../../../FINAL_FIGURES.md).",
            "",
        ]
        if self.args.include_random_control:
            index += [
                "",
                "Random-control replay (reuses completed checkpoints):",
                "",
                "```bash",
                "python ISC-CI_LLM_validation/run_random_feature_control.py --stage plot",
                "```",
                "",
            ]
        (self.out / "INDEX.md").write_text("\n".join(index))
        print(
            f"Wrote {len(self.figures)} figure filesets to {self.out}; blocked groups: {len(self.blocked)}",
            flush=True,
        )


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--behavior-dir",
        type=Path,
        default=ROOT / "artifacts/human_behavior_validation_v4_20261002",
    )
    p.add_argument(
        "--validation-dir", type=Path, default=ROOT / "artifacts/validation_v4_20261002"
    )
    p.add_argument("--report-dir", type=Path, default=ROOT / "reports/v4_20261002")
    p.add_argument(
        "--include-random-control",
        action="store_true",
        help="Add the completed human-sized random-feature control; requires extension artifacts",
    )
    p.add_argument(
        "--output-dir", type=Path, default=ROOT / "reports/v4_20261002/figures_final"
    )
    p.add_argument(
        "--categories",
        type=Path,
        default=ROOT.parent.parent
        / "IntegratedSemanticsControlContextInference/data/leuven_dataset/leuven_categories.csv",
    )
    p.add_argument(
        "--allow-missing",
        action="store_true",
        help="Return success for explicitly documented blocked panels; default returns failure after rendering available groups",
    )
    return p


def main():
    args = parser().parse_args()
    package = Package(args)
    package.initialize()  # Missing core artifacts are a strict preflight failure.
    main_conditions, all_conditions = package.main, package.all
    all_suffix = f"all{len(all_conditions)}"
    stages = [
        (
            "Figure 1A",
            lambda: package.structure(main_conditions, "figure1a_inventory_geometry"),
        ),
        ("Figure 1B", package.distributions),
        (
            "Figure 1C",
            lambda: package.rdms(
                ["human", "v3_B", "v4_ensemble_source_only", "v4_ensemble_calibrated"]
                + ([RANDOM] if args.include_random_control else []),
                "figure1c_object_rdms",
            ),
        ),
        ("Figure 2", lambda: package.similarity(main_conditions, "figure2_similarity")),
        (
            "Figure 2 sensitivity",
            lambda: package.similarity(
                main_conditions, "figure2s_common_finite_pairs", common=True
            ),
        ),
        ("Figure 3A", lambda: package.induction(main_conditions, "figure3a_induction")),
        (
            "Figure 3B",
            lambda: package.preferences(
                main_conditions, "figure3b_contextual_preferences"
            ),
        ),
        (
            "Figure 4A",
            lambda: package.direction(
                "context", main_conditions, "figure4a_context_direction"
            ),
        ),
        (
            "Figure 4B",
            lambda: package.scatter(
                "context",
                [c for c in main_conditions if c != "v2"],
                "figure4b_context_magnitude",
            ),
        ),
        ("Figure 4C", package.effect_table),
        (
            "All-condition similarity",
            lambda: package.similarity(
                all_conditions, f"supplement_similarity_{all_suffix}"
            ),
        ),
        (
            "All-condition induction",
            lambda: package.induction(
                all_conditions, f"supplement_induction_{all_suffix}"
            ),
        ),
        (
            "All-condition preferences",
            lambda: package.preferences(
                all_conditions, f"supplement_preferences_{all_suffix}"
            ),
        ),
        (
            "All-condition context direction",
            lambda: package.direction(
                "context", all_conditions, f"supplement_context_direction_{all_suffix}"
            ),
        ),
        ("Representations", package.representations),
        ("Paired contrasts", package.forests),
        ("Coverage", package.coverage),
        (
            "Remaining RDMs",
            lambda: package.rdms(
                [
                    c
                    for c in ALL
                    if c
                    not in [
                        "human",
                        "v3_B",
                        "v4_ensemble_source_only",
                        "v4_ensemble_calibrated",
                    ]
                ][:5],
                "supplement_object_rdms_1",
            ),
        ),
        (
            "Remaining RDMs 2",
            lambda: package.rdms(
                ["v3_1_A", "v3_1_C", "v4_b_locked_v2", "v4_ensemble_locked_v2"],
                "supplement_object_rdms_2",
            ),
        ),
        (
            "Asymmetry direction",
            lambda: package.direction(
                "asymmetry",
                all_conditions,
                f"supplement_asymmetry_direction_{all_suffix}",
            ),
        ),
    ]
    for task in ["context", "asymmetry"]:
        for start in range(0, len(all_conditions), 6):
            subset = all_conditions[start : start + 6]
            name = f"supplement_{task}_magnitude_{start//6+1}"
            stages.append(
                (name, lambda t=task, cs=subset, n=name: package.scatter(t, cs, n))
            )
    for name, function in stages:
        package.stage(name, function)
    package.finalize()
    if package.blocked and not args.allow_missing:
        raise SystemExit(
            "Required panels blocked; see figure_manifest.json. Available panels were rendered. No model reruns."
        )


if __name__ == "__main__":
    main()
