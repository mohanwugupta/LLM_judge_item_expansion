#!/usr/bin/env python3
"""Summarize completed, unchanged V4 validation outputs without model selection."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from iscci_validation.provenance import sha256_file, write_json


ROOT = Path(__file__).resolve().parent
VALIDATION = ROOT / "artifacts/validation_v4_20261002"
BEHAVIOR = ROOT / "artifacts/human_behavior_validation_v4_20261002"
MATRICES = ROOT / "artifacts/v4_matrices"
REPORT = ROOT / "reports/v4_20261002"


def markdown_table(frame):
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def mean_sd(values):
    return f"{values.mean():.3f} +/- {values.std(ddof=1):.3f}"


def main():
    behavior_manifest = json.loads((BEHAVIOR / "manifest.json").read_text())
    if (
        not behavior_manifest["complete"]
        or not behavior_manifest["fidelity_gate"]["passed"]
    ):
        raise ValueError(
            "Final report requires all requested seeds and a passed fidelity gate"
        )
    model_manifest = json.loads((VALIDATION / "manifest.json").read_text())
    simulation_manifest = json.loads(
        (VALIDATION / "paper_simulations/manifest.json").read_text()
    )
    matrix_manifest = json.loads((MATRICES / "matrix_manifest.json").read_text())
    conditions = pd.read_csv(BEHAVIOR / "condition_summary.csv")
    per_seed = pd.read_csv(BEHAVIOR / "behavior_per_seed.csv")
    representations = pd.read_csv(BEHAVIOR / "representation_per_seed.csv")
    evaluation_manifest = json.loads(
        (VALIDATION / "evaluation/manifest.json").read_text()
    )
    expected_models = len(representations) + 4
    if (
        evaluation_manifest["models_evaluated"] != expected_models
        or simulation_manifest["model_count"] != expected_models
    ):
        raise ValueError(
            "Representational or simulation manifest still describes an older cohort"
        )
    if (
        not conditions["seed_count"].eq(4).all()
        or not conditions["status"].eq("evaluated").all()
    ):
        raise ValueError("Incomplete condition table")
    REPORT.mkdir(parents=True, exist_ok=True)
    profile_audit = pd.read_csv(REPORT / "feature_profile_redundancy.csv")
    geometry, tasks = [], []
    for condition in conditions["condition"]:
        representation = representations[representations["condition"] == condition]
        behavior = per_seed[per_seed["condition"] == condition]
        geometry.append(
            {
                "Condition": condition,
                "Contexts": int(representation["features"].iloc[0]),
                "Input RDM": f"{representation['input_object_rdm_spearman'].iloc[0]:.3f}",
                "Context RDM": mean_sd(representation["context_rdm_spearman"]),
                "Dependent RDM": mean_sd(
                    representation["context_dependent_rdm_spearman_median"]
                ),
                "Membership ranks": mean_sd(
                    representation["membership_logit_spearman"]
                ),
            }
        )
        row = {"Condition": condition}
        for task, label in [
            ("similarity", "Similarity r"),
            ("induction", "Induction r"),
            ("asymmetry", "Asymmetry direction"),
            ("context", "Context direction"),
            ("thematic", "Thematic argument 2"),
            ("nonmonotonicity", "Nonmonotonic argument 2"),
        ]:
            values = (
                behavior[behavior["task"] == task]
                .groupby("seed")["paper_statistic"]
                .mean()
            )
            row[label] = mean_sd(values)
        tasks.append(row)
    geometry = pd.DataFrame(geometry)
    tasks = pd.DataFrame(tasks)
    lower = pd.read_csv(BEHAVIOR / "similarity_levels_1_2_summary.csv")
    levels = (
        lower[lower.metric.eq("paper_statistic")]
        .groupby(["base_condition", "level"])["mean"]
        .mean()
        .unstack()
    )
    levels["fully_trained"] = (
        per_seed[per_seed.task.eq("similarity")]
        .groupby("condition")
        .paper_statistic.mean()
    )
    levels = levels.reindex(conditions["condition"])
    levels.to_csv(REPORT / "three_level_similarity_overview.csv")
    level_display = levels.reset_index().rename(
        columns={"condition": "Condition", "base_condition": "Condition"}
    )
    for column in level_display.columns[1:]:
        level_display[column] = level_display[column].map(lambda value: f"{value:.3f}")
    geometry.to_csv(REPORT / "representational_overview.csv", index=False)
    tasks.to_csv(REPORT / "behavioral_overview.csv", index=False)
    conditions.to_csv(REPORT / "all_condition_summary.csv", index=False)
    contrasts = pd.read_csv(BEHAVIOR / "paired_condition_contrasts.csv")
    central_pairs = {
        ("v4_b_calibrated", "v3_1_B"),
        ("v4_b_calibrated", "human"),
        ("v4_b_calibrated", "v2"),
        ("v4_ensemble_calibrated", "v3_1_B"),
        ("v4_ensemble_calibrated", "v4_ensemble_source_only"),
        ("v4_ensemble_calibrated", "v2"),
        ("v4_ensemble_calibrated", "human"),
        ("v4_ensemble_calibrated", "v4_b_calibrated"),
        ("v4_ensemble_calibrated", "v4_ensemble_locked_v2"),
        ("v4_b_calibrated", "v4_b_locked_v2"),
    }
    central = contrasts[
        [
            pair in central_pairs
            for pair in zip(contrasts.condition_a, contrasts.condition_b)
        ]
    ]
    central.to_csv(REPORT / "central_behavioral_contrasts.csv", index=False)
    parity = json.loads((REPORT / "baseline_prediction_parity.json").read_text())
    graph_parity = json.loads((REPORT / "v4_original_graph_parity.json").read_text())
    if parity["mismatches"] or not graph_parity["passed"]:
        raise ValueError("Prediction or original-graph QA failed")

    def contrast(left, right, task, dataset):
        row = contrasts[
            (contrasts.condition_a == left)
            & (contrasts.condition_b == right)
            & (contrasts.task == task)
            & (contrasts.dataset == dataset)
            & (contrasts.metric == "paper_statistic")
        ].iloc[0]
        return (
            f"{row.difference_a_minus_b:+.3f} [{row.ci_low:+.3f}, {row.ci_high:+.3f}]"
        )

    notes = [
        "# Finalized V4 Results",
        "",
        f"Completed {len(conditions)} conditions with four seeds each ({len(representations)} trained models); the paper simulations also include four released references.",
        "",
        "All 33 required fidelity checks pass: exact deterministic released-checkpoint reproduction and statistical reconstructed-human replication consistency. This is not a formal equivalence test.",
        "",
        f"Additional QA: {parity['files_compared']} previous-condition item/trial CSVs are byte-identical; their per-seed statistics and bootstrap summaries have zero numerical differences. All 20 new models agree with the unedited paper model graph/scorer in {graph_parity['checks']} checks (maximum error {graph_parity['max_absolute_error']:.2g}).",
        "",
        "## Interpretation",
        "",
        "Completion strongly improves Leuven feature-matrix recovery, but does not establish general behavioral equivalence or a uniformly better cognitive model. Fixed-B calibrated completion raises input geometry from .241 to .762 and the descriptive induction average from .425 to .506, while the similarity average falls from .534 to .500. The calibrated ensemble reaches .821 input geometry, essentially the V2 level, without matching the human-trained model's inferred-context representations.",
        "",
        "Source-only has lower input agreement (.641) but higher descriptive similarity prediction (.575) than the completed ensemble (.520). Completion improves Bhatia Exp 1, Bhatia Exp 3, and Han Exp 2 induction relative to source-only, while other induction datasets do not show the same improvement. This is a task-dependent result, not evidence to change the frozen training procedure.",
        "",
        f"Completed ensemble minus source-only, original argument-2 preference: thematic {contrast('v4_ensemble_calibrated', 'v4_ensemble_source_only', 'thematic', 'thematic')}; nonmonotonicity {contrast('v4_ensemble_calibrated', 'v4_ensemble_source_only', 'nonmonotonicity', 'nonmonotonicity')}. These are paired crossed item/seed 95% intervals, not simultaneous multiple-comparison intervals. The original observed participant preference medians are .843 and .848; the completed ensemble rates are only .293 and .491. These original qualitative effects are not recovered to the human-trained baseline level.",
        "",
        "Calibration produces modest, task-dependent changes rather than a consistent behavioral advantage over locked V2 decisions. Broad discovery adds many distinct membership profiles, but the completed ensemble's context and dependent geometry are weaker than fixed-B despite its better input geometry. Unequal per-context episode exposure and changing feature applicability/weighting are plausible explanations, not causes isolated by this run.",
        "",
        "Across 12 nonhuman conditions, geometry-behavior relationships vary by task: positive for Bhatia Exp 1 and Han Exp 2, negative for Osherson Exp 4 and the two argument-preference measures, and mixed across similarity domains. These exploratory associations do not support a universal rule that better Leuven recovery implies better human prediction. Global object geometry and fine-grained within-domain or contextual judgments need not agree.",
        "",
        "## Representational Agreement",
        "",
        markdown_table(geometry),
        "",
        "Seed mean +/- SD. Input geometry has no seed variation. The semantic-activation agreement is exactly 1 by construction because every condition shares the frozen released human-pretrained encoder; it is not independent semantic recovery.",
        "",
        "## Original Human Tasks",
        "",
        markdown_table(tasks),
        "",
        "Similarity and induction columns are descriptive averages of the paper's domain/dataset correlations (nine and five respectively), calculated within seed. They are not pooled paper statistics. Direction columns retain the original human-defined subsets; thematic/nonmonotonicity are original argument-2 preference rates, not majority accuracy. Higher preference is not universally better: compare observed human proportions and individual-item fits in the complete tables.",
        "",
        "Every individual dataset's original measure, direct correlations, prediction errors, cross-validated rating mapping, crossed item/seed 95% intervals, and human reliability are in `all_condition_summary.csv` and the underlying behavior tables. All paired contrasts are preserved, including unfavorable ones; no condition was selected from these outcomes.",
        "",
        "## Three Evaluation Levels",
        "",
        markdown_table(level_display),
        "",
        "Descriptive average across the same nine original similarity domains. Raw matrices and the shared semantic encoder have no training-seed variation; item-bootstrap intervals and coverage are saved separately. Sparse V3/V3.1 undefined raw cosines remain missing, not invented negatives. All V4 matrices have zero feature-empty objects.",
        "",
        "Source-only's raw feature matrix already predicts similarity better than the completed ensemble (.593 versus .524), before neural training. Consequently, neural episode exposure alone cannot explain this particular similarity difference. The shared human-pretrained semantic baseline is .539; fully trained V4 source-only is .575 and the completed calibrated ensemble .520. None of these averages establishes a pooled or equivalence-test conclusion.",
        "",
        "## Central Comparisons",
        "",
        "`central_behavioral_contrasts.csv` reports all original tasks for fixed-B versus V3.1-B, V4 versus source-only, V2, and human, locked versus calibrated, and ensemble versus fixed-B. Intervals jointly resample human item clusters and paired model seeds. CV mappings are not refit inside the bootstrap, as in the frozen protocol.",
        "",
        "`geometry_behavior_association.csv` tests Leuven geometry versus behavior across nonhuman CONDITIONS, not pseudo-independent seeds. Conditions are related, sample size is small, and permutation p-values are exploratory and unadjusted across outcomes. Defined-pair sensitivity is separate from the historical zero-vector RDM convention.",
        "",
        "## Limitations",
        "",
        "- All matrices and thresholds were frozen before external behavior. No behavior-driven feature repair, deletion, threshold, or training tuning.",
        "- The approved cascade accepts high-confidence prompt-C negatives. Completed cell accounting does not imply that every cell received a full judge panel, or establish counterfactual recall for the skipped panels.",
        "- 127 unknown scores remain disclosed; all 64 affected feature columns were excluded, rather than treating missing scores as negatives. Fixed-B loses no contexts.",
        "- Completion versus source-only also changes retained inventory and episode exposure; fixed-B versus V3.1-B holds vocabulary constant.",
        "- Each condition gets 409,600 episodes. The ~81-87k-context ensembles get about five episodes/context, far fewer than human or fixed-B. Do not reinterpret this frozen-budget comparison as convergence-matched training.",
        "- The same human-pretrained semantic encoder remains in every model. These results do not demonstrate a wholly human-data-free semantic model.",
        "- Original model-selected stimuli and synthetic phenomenon arguments limit claims of independent behavioral generalization. Human split-half/participant reliability is not necessarily a model noise ceiling.",
        "",
        "## Feature-Profile Audit",
        "",
        markdown_table(profile_audit),
        "",
        "These are distinct binary membership patterns on the same 293 objects, not independent semantic dimensions or proven proposition equivalence. No duplicate columns were removed, merged, or reweighted.",
        "",
        "## Artifacts",
        "",
        "- `ISC-CI_LLM_validation/V4_20261002_RUN.md`: decisions and exact commands.",
        "- `artifacts/v4_matrices/`: seven matrices, cell provenance, exclusions, retention and completion counts.",
        "- `artifacts/validation_v4_20261002/`: cumulative checkpoints, training logs, human self-ceiling, common-probe comparisons, and original-paper simulations.",
        "- `artifacts/human_behavior_validation_v4_20261002/`: raw item/trial predictions, participant predictions, CV fits, all three evaluation levels, uncertainty, reliability, contrasts, and exploratory geometry-behavior analyses.",
        "- `reports/v4_20261002/source_snapshot/`: execution source files. See report manifest for hashes.",
    ]
    (REPORT / "RESULTS.md").write_text("\n".join(notes) + "\n")
    inputs = [
        BEHAVIOR / name
        for name in (
            "manifest.json",
            "condition_summary.csv",
            "behavior_per_seed.csv",
            "paired_condition_contrasts.csv",
            "geometry_behavior_association.csv",
            "representation_per_seed.csv",
            "human_reliability.csv",
            "similarity_levels_1_2_summary.csv",
        )
    ] + [
        VALIDATION / "manifest.json",
        VALIDATION / "evaluation/manifest.json",
        VALIDATION / "paper_simulations/manifest.json",
        MATRICES / "matrix_manifest.json",
        REPORT / "feature_profile_redundancy.csv",
        REPORT / "feature_profile_audit_manifest.json",
        REPORT / "baseline_prediction_parity.json",
        REPORT / "baseline_summary_parity.json",
        REPORT / "v4_original_graph_parity.csv",
        REPORT / "v4_original_graph_parity.json",
    ]
    write_json(
        REPORT / "report_manifest.json",
        {
            "complete": True,
            "input_sha256": {
                str(path.relative_to(ROOT)): sha256_file(path) for path in inputs
            },
            "snapshot_sha256": {
                path.name: sha256_file(path)
                for path in sorted((REPORT / "source_snapshot").glob("*.py"))
            },
            "summary_implementation_sha256": sha256_file(Path(__file__)),
            "condition_count": len(conditions),
            "model_count": len(representations),
            "training_parameters": model_manifest["training_parameters"],
            "matrix_manifest_hash": matrix_manifest["matrix_manifest_hash"],
            "paper_simulation_manifest": simulation_manifest,
        },
    )
    print(REPORT / "RESULTS.md")


if __name__ == "__main__":
    main()
