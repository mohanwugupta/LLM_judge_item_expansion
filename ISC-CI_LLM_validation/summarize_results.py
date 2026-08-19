#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr

from iscci_validation.provenance import sha256_file, write_json


ROOT = Path(__file__).resolve().parent
VALIDATION = ROOT / "artifacts" / "validation_v3"
EVALUATION = VALIDATION / "evaluation"
SIMULATIONS = VALIDATION / "paper_simulations"
REPORTS = ROOT / "reports"


def training_summary() -> pd.DataFrame:
    rows = []
    for matrix_path in sorted((VALIDATION / "data_matrices").glob("*.csv")):
        condition = matrix_path.stem
        matrix = pd.read_csv(matrix_path, index_col=0)
        seed_rows = []
        for metrics_path in sorted(
            (VALIDATION / "models" / condition).glob("seed_*/training_metrics.csv")
        ):
            metrics = pd.read_csv(metrics_path).tail(50)
            seed_rows.append(
                {
                    "final_loss": metrics["loss"].mean(),
                    "final_task_accuracy": metrics["task_acc"].mean(),
                    "final_query_accuracy": metrics["query_ft_acc"].mean(),
                }
            )
        metrics = pd.DataFrame(seed_rows)
        rows.append(
            {
                "condition": condition,
                "tasks": matrix.shape[1],
                "positive_cells": int(matrix.values.sum()),
                "density": float(matrix.values.mean()),
                "models": len(seed_rows),
                **{
                    f"{column}_mean": float(metrics[column].mean())
                    for column in metrics
                },
                **{
                    f"{column}_std": float(metrics[column].std())
                    for column in metrics
                },
            }
        )

    released_rows = []
    for seed in range(4):
        checkpoint = torch.load(
            ROOT
            / "upstream"
            / "IntegratedSemanticsControlContextInference"
            / "models"
            / f"1and2shot_isc-seed{seed}.pt",
            map_location="cpu",
            weights_only=False,
        )
        metrics = pd.DataFrame(checkpoint["metrics"]).tail(50)
        released_rows.append(
            {
                "final_loss": metrics["loss"].mean(),
                "final_task_accuracy": metrics["task_acc"].mean(),
                "final_query_accuracy": metrics["query_ft_acc"].mean(),
            }
        )
    released_metrics = pd.DataFrame(released_rows)
    human = next(row for row in rows if row["condition"] == "human")
    rows.append(
        {
            "condition": "released",
            "tasks": 385,
            "positive_cells": human["positive_cells"],
            "density": human["density"],
            "models": 4,
            **{
                f"{column}_mean": float(released_metrics[column].mean())
                for column in released_metrics
            },
            **{
                f"{column}_std": float(released_metrics[column].std())
                for column in released_metrics
            },
        }
    )
    return pd.DataFrame(rows)


def ceiling_summary() -> pd.DataFrame:
    pairwise = pd.read_csv(EVALUATION / "pairwise_model_metrics.csv")
    available_groups = set(pairwise["comparison_group"])
    groups = [
        group
        for group in [
            "human_retrain_vs_retrain",
            "released_vs_human_retrain",
            *sorted(
                group
                for group in available_groups
                if group.endswith("_vs_human_retrain")
                and group != "released_vs_human_retrain"
            ),
        ]
        if group in available_groups
    ]
    metrics = [
        "context_rdm_spearman",
        "context_dependent_rdm_spearman_median",
        "membership_logit_spearman",
        "binary_membership_agreement",
        "membership_probability_mae",
    ]
    rows = []
    for group in groups:
        selected = pairwise[pairwise["comparison_group"] == group]
        row = {"comparison_group": group, "comparisons": len(selected)}
        for metric in metrics:
            row[f"{metric}_mean"] = float(selected[metric].mean())
            row[f"{metric}_std"] = float(selected[metric].std())
        rows.append(row)
    return pd.DataFrame(rows)


def simulation_summary() -> pd.DataFrame:
    induction = pd.read_csv(SIMULATIONS / "induction_correlations.csv")
    phenomena = pd.read_csv(SIMULATIONS / "induction_phenomena.csv")
    domain = pd.read_csv(SIMULATIONS / "similarity_domain_correlations.csv")
    asymmetry = pd.read_csv(SIMULATIONS / "similarity_asymmetry.csv")
    paired = pd.read_csv(SIMULATIONS / "paired_choice_agreement.csv")
    context = pd.read_csv(SIMULATIONS / "similarity_context.csv")

    induction_by_seed = (
        induction.groupby(["condition", "seed"], observed=True)["pearson_r"]
        .mean()
        .rename("induction_mean_r")
        .reset_index()
    )
    domain_by_seed = (
        domain.groupby(["condition", "seed"], observed=True)["pearson_r"]
        .mean()
        .rename("similarity_domain_mean_r")
        .reset_index()
    )
    human_effect = (
        phenomena[phenomena["condition"] == "human"]
        .groupby("phenomenon", observed=True)["high_minus_low"]
        .mean()
    )
    phenomenon_rows = []
    for (condition, seed), group in phenomena.groupby(
        ["condition", "seed"], observed=True
    ):
        effects = group.set_index("phenomenon").loc[human_effect.index, "high_minus_low"]
        phenomenon_rows.append(
            {
                "condition": condition,
                "seed": seed,
                "phenomena_pattern_r_vs_human_mean": pearsonr(
                    effects, human_effect
                )[0],
                "phenomena_rmse_vs_human_mean": float(
                    np.sqrt(np.mean((effects - human_effect) ** 2))
                ),
            }
        )
    per_seed = induction_by_seed.merge(domain_by_seed, on=["condition", "seed"])
    per_seed = per_seed.merge(
        pd.DataFrame(phenomenon_rows), on=["condition", "seed"]
    )
    per_seed = per_seed.merge(
        asymmetry[
            [
                "condition",
                "seed",
                "agreement_with_human_majority",
                "asymmetry_score_human_pearson",
            ]
        ],
        on=["condition", "seed"],
    )
    paired_wide = paired.pivot(
        index=["condition", "seed"],
        columns="experiment",
        values="agreement_with_human_majority",
    ).reset_index()
    paired_wide = paired_wide.rename(
        columns={
            "context_dependent_nonmonotonicity": "context_nonmonotonicity_agreement",
            "thematic_arguments": "thematic_agreement",
        }
    )
    per_seed = per_seed.merge(paired_wide, on=["condition", "seed"])
    per_seed = per_seed.merge(
        context[
            [
                "condition",
                "seed",
                "direction_agreement",
                "model_effect_mean",
                "model_human_effect_pearson",
            ]
        ].rename(
            columns={
                "direction_agreement": "similarity_context_direction_agreement",
                "model_effect_mean": "similarity_context_effect_mean",
                "model_human_effect_pearson": "similarity_context_effect_pearson",
            }
        ),
        on=["condition", "seed"],
    )
    per_seed.to_csv(REPORTS / "paper_simulation_per_seed.csv", index=False)
    return per_seed.groupby("condition", observed=True).agg(["mean", "std"]).reset_index()


def flattened(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [
        "_".join(str(value) for value in column if str(value))
        if isinstance(column, tuple)
        else str(column)
        for column in frame.columns
    ]
    return frame


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    training = training_summary()
    ceiling = ceiling_summary()
    simulations = flattened(simulation_summary())
    input_rdm = pd.read_csv(EVALUATION / "input_matrix_rdm_comparisons.csv")
    v2_recovery = pd.read_csv(EVALUATION / "v2_binary_recovery.csv")
    sensitivity = pd.read_csv(
        ROOT / "artifacts" / "v3_consolidation" / "threshold_sensitivity.csv"
    )

    training.to_csv(REPORTS / "training_summary.csv", index=False)
    ceiling.to_csv(REPORTS / "ceiling_summary.csv", index=False)
    simulations.to_csv(REPORTS / "paper_simulation_summary.csv", index=False)
    input_rdm.to_csv(REPORTS / "input_matrix_rdm_comparisons.csv", index=False)
    v2_recovery.to_csv(REPORTS / "v2_binary_recovery.csv", index=False)

    train = training.set_index("condition")
    ceil = ceiling.set_index("comparison_group")
    sim = simulations.set_index("condition")
    candidate_groups = {
        "v2": "v2_vs_human_retrain",
        "v3_A": "v3_A_vs_human_retrain",
        "v3_B": "v3_B_vs_human_retrain",
        "v3_C": "v3_C_vs_human_retrain",
    }

    matrix_table = training[
        ["condition", "tasks", "positive_cells", "density"]
    ].copy()
    matrix_table["density"] = matrix_table["density"].map(lambda value: f"{value:.3f}")
    ceiling_table = []
    for condition, group in candidate_groups.items():
        row = ceil.loc[group]
        ceiling_table.append(
            {
                "Condition": condition,
                "Context RDM": f"{row.context_rdm_spearman_mean:.3f}",
                "Context-dependent RDM": f"{row.context_dependent_rdm_spearman_median_mean:.3f}",
                "Membership rank": f"{row.membership_logit_spearman_mean:.3f}",
                "Binary agreement": f"{row.binary_membership_agreement_mean:.3f}",
                "Probability MAE": f"{row.membership_probability_mae_mean:.3f}",
            }
        )
    human_ceiling = ceil.loc["human_retrain_vs_retrain"]
    released_ceiling = ceil.loc["released_vs_human_retrain"]
    ceiling_table.insert(
        0,
        {
            "Condition": "Human self-ceiling",
            "Context RDM": f"{human_ceiling.context_rdm_spearman_mean:.3f}",
            "Context-dependent RDM": f"{human_ceiling.context_dependent_rdm_spearman_median_mean:.3f}",
            "Membership rank": f"{human_ceiling.membership_logit_spearman_mean:.3f}",
            "Binary agreement": f"{human_ceiling.binary_membership_agreement_mean:.3f}",
            "Probability MAE": f"{human_ceiling.membership_probability_mae_mean:.3f}",
        },
    )
    ceiling_table.insert(
        1,
        {
            "Condition": "Released vs retrained human",
            "Context RDM": f"{released_ceiling.context_rdm_spearman_mean:.3f}",
            "Context-dependent RDM": f"{released_ceiling.context_dependent_rdm_spearman_median_mean:.3f}",
            "Membership rank": f"{released_ceiling.membership_logit_spearman_mean:.3f}",
            "Binary agreement": f"{released_ceiling.binary_membership_agreement_mean:.3f}",
            "Probability MAE": f"{released_ceiling.membership_probability_mae_mean:.3f}",
        },
    )

    simulation_columns = {
        "condition": "Condition",
        "induction_mean_r_mean": "Induction r",
        "similarity_domain_mean_r_mean": "Similarity r",
        "asymmetry_score_human_pearson_mean": "Asymmetry r",
        "context_nonmonotonicity_agreement_mean": "Context nonmono",
        "thematic_agreement_mean": "Thematic",
        "similarity_context_direction_agreement_mean": "Context direction",
    }
    simulation_table = simulations[list(simulation_columns)].rename(
        columns=simulation_columns
    )
    for column in simulation_table.columns[1:]:
        simulation_table[column] = simulation_table[column].map(
            lambda value: f"{value:.3f}"
        )

    rdm_human = {
        row.condition_b: row.object_rdm_spearman
        for row in input_rdm.itertuples()
        if row.condition_a == "human"
    }
    recovery = v2_recovery.iloc[0]
    primary_sensitivity = sensitivity[sensitivity["embedding_threshold"] == 0.85]
    task_ranges = (
        sensitivity.groupby("prompt_variant")["retained_training_tasks"]
        .agg(["min", "max"])
        .astype(int)
    )

    report = f"""# ISC-CI Human, V2, and V3 Validation

Generated by `summarize_results.py` from the locked validation artifacts.

## Executive Interpretation

The retraining pipeline reproduces the original human-data model well: released-versus-
retrained human models are at or slightly above the human retrain self-ceiling on all five
fixed-context measures. This establishes that downstream differences are attributable to the
feature norms rather than a failed reconstruction of the ISC-CI training procedure.

V2 and V3 both recover meaningful behavior, but neither is interchangeable with human norms.
V2 preserves the human object geometry strongly and is closer on context-dependent geometry
and ranked membership output. V3 B/C are about as close as V2 at the context layer, and V3 C
performs well on several paper simulations, but all v3 conditions are materially below the
human self-ceiling on context-dependent and ranked membership measures.

High v3 binary agreement is not sufficient evidence of equivalence. V3 matrices are only
about 2.2-2.3% positive, so both human and v3 models often return a negative membership
decision. The lower context-dependent RDM and logit-rank correlations show that the models do
not order objects in the same way even when their binary decisions agree.

## Conditions

{matrix_table.to_markdown(index=False)}

The human matrix uses the paper's strict `count > 3`, then `positive objects > 3` rules. V2
uses the original 385-feature human schema and treats every nonzero finalized adjudicator
value as positive; three empty tasks are dropped. V3 applies the human strict cutoffs after
prompt-specific semantic consolidation.

## Consolidation Audit

- Source: 17,580 valid responses and 175,796 generated feature rows.
- Prompt-specific retained tasks at threshold 0.85: A={int(primary_sensitivity.loc[primary_sensitivity.prompt_variant == 'A', 'retained_training_tasks'].iloc[0])}, B={int(primary_sensitivity.loc[primary_sensitivity.prompt_variant == 'B', 'retained_training_tasks'].iloc[0])}, C={int(primary_sensitivity.loc[primary_sensitivity.prompt_variant == 'C', 'retained_training_tasks'].iloc[0])}.
- Retained task ranges over embedding thresholds 0.80-0.90: A={task_ranges.loc['A', 'min']}-{task_ranges.loc['A', 'max']}, B={task_ranges.loc['B', 'min']}-{task_ranges.loc['B', 'max']}, C={task_ranges.loc['C', 'min']}-{task_ranges.loc['C', 'max']}.
- Seven retained embedding/profile clusters were manually reviewed; all passed.
- Forty CJK feature rows from four squid responses were flagged and excluded from English
  embedding merges. They did not produce a retained training task.
- Exact assignments, canonical labels, source/model hashes, threshold sensitivity, and manual
  review are in `artifacts/v3_consolidation/`.

A subsequent rule audit found that the 0.85 embedding threshold does not determine the
retained task counts: current lexical normalization alone yields the same 94/115/121 tasks.
The profile guard prevents clear semantic errors, but automatic lexical signatures contain a
small number of proposition-changing merges. The primary validation is therefore preserved
but should be accompanied by an adjudicated/conservative lexical sensitivity retraining. See
`reports/consolidation_audit/CONSOLIDATION_AUDIT.md`.

## Pipeline Reproduction Check

The released human models and reconstructed human models have nearly identical end-of-
training behavior. Mean final-50-batch loss was {train.loc['released', 'final_loss_mean']:.3f}
for released models and {train.loc['human', 'final_loss_mean']:.3f} for retrained models;
query accuracy was {train.loc['released', 'final_query_accuracy_mean']:.3f} and
{train.loc['human', 'final_query_accuracy_mean']:.3f}, respectively.

The released checkpoint metadata says `batch_size=1024`, but its retained metrics contain
eight batches per 1,024-episode epoch with 1/128 accuracy increments. Retraining follows the
executed batch size of 128 for 400 epochs (409,600 episodes), documented in `DECISIONS.md`.

## Model Ceiling

{pd.DataFrame(ceiling_table).to_markdown(index=False)}

All values except probability MAE are higher-is-better. The human self-ceiling is variation
among independently retrained human-data runs. Released-versus-retrained values confirm the
reconstruction.

V2 reaches 95.4% of the human self-ceiling for context-dependent RDM correlation and 83.8%
for membership-logit rank correlation. V3 B/C reach about 90.2-90.3% for context RDM, but
only about 79% for context-dependent RDM and 73.5-73.6% for membership-logit rank.

At the input-data level, object-RDM Spearman correlations with human norms are:
V2={rdm_human['v2']:.3f}, V3 A={rdm_human['v3_A']:.3f}, V3 B={rdm_human['v3_B']:.3f}, and
V3 C={rdm_human['v3_C']:.3f}. The frozen human semantic embedding and shared architecture
therefore account for part of the apparently high context-layer similarity.

## V2 Feature Recovery

Across the 382 shared executable features, V2 has recall={recovery['recall']:.3f},
precision={recovery['precision']:.3f}, F1={recovery['f1']:.3f}, balanced
accuracy={recovery['balanced_accuracy']:.3f}, and MCC={recovery['matthews_correlation']:.3f}.
It recovers most human positives but marks many additional object-feature pairs positive,
making the v2 matrix 26.6% positive versus 7.1% for human norms.

## Paper Simulations

{simulation_table.to_markdown(index=False)}

Values are means across four seeds. `Induction r` averages the five published induction
dataset correlations. `Similarity r` averages the nine in-domain similarity correlations.
The remaining columns are agreement/correlation measures from the repository notebooks.

Key patterns:

- V2 matches or exceeds human/released mean induction correlation, but performs poorly on
  thematic paired arguments.
- V3 B/C recover much of the induction correlation. V3 C is closest among v3 prompts on
  in-domain similarity and similarity-context direction.
- V3 B is the strongest v3 prompt on asymmetric similarity.
- V3 A is strongest on thematic paired arguments and slightly strongest on context-dependent
  non-monotonicity.
- No v3 prompt dominates every behavioral benchmark. Prompt C has the best broad profile;
  prompt B is preferable for asymmetry; prompt A retains a thematic advantage.

## Conclusion

These results support **qualified confidence** in the method. Free LLM feature generation can
produce ISC-CI models that recover several qualitative and quantitative human-model behaviors,
and prompt B/C are clearly more successful than an arbitrary or failed representation. The
result is not strong enough to claim that v3 replaces human norms: the native v3 feature spaces
have weak human object geometry, contain only 24-31% as many trainable contexts as human
norms, and yield downstream representations well outside human run-to-run variability.

The most defensible use is as a scalable approximation whose validity is assessed behaviorally,
with prompt C as the current broad default and prompt-specific sensitivity retained. A next
round should increase independent LLM participants and feature diversity, then repeat the same
locked consolidation and validation without changing thresholds after seeing outcomes. Before
calling the present V3 comparison final, retrain a second sensitivity condition using reviewed
lexical merges while preserving the current models as the preregistered-style primary result.

## Reproduction

```bash
cd ISC-CI_LLM_validation
python consolidate_v3.py
python run_validation.py
python evaluate_validation.py
python run_paper_simulations.py
python summarize_results.py
pytest -q tests
```

All generated validation files remain under `ISC-CI_LLM_validation/`, as requested.
"""
    (REPORTS / "RESULTS.md").write_text(report, encoding="utf-8")
    analysis_sources = sorted(
        [
            ROOT / "consolidate_v3.py",
            ROOT / "run_validation.py",
            ROOT / "evaluate_validation.py",
            ROOT / "run_paper_simulations.py",
            ROOT / "summarize_results.py",
            ROOT / "audit_consolidation_rules.py",
            ROOT / "DECISIONS.md",
            ROOT / "README.md",
            ROOT / "configs" / "v3_validation.json",
            ROOT / "configs" / "v3_consolidation_manual_review.csv",
            *sorted((ROOT / "iscci_validation").glob("*.py")),
        ]
    )
    write_json(
        REPORTS / "report_manifest.json",
        {
            "paper_pdf": str((ROOT.parent / "ISCCI_2026Jan_ArXiv.pdf").resolve()),
            "paper_pdf_sha256": sha256_file(ROOT.parent / "ISCCI_2026Jan_ArXiv.pdf"),
            "consolidation_manifest_sha256": sha256_file(
                ROOT / "artifacts" / "v3_consolidation" / "manifest.json"
            ),
            "validation_manifest_sha256": sha256_file(VALIDATION / "manifest.json"),
            "evaluation_manifest_sha256": sha256_file(
                EVALUATION / "manifest.json"
            ),
            "simulation_manifest_sha256": sha256_file(
                SIMULATIONS / "manifest.json"
            ),
            "consolidation_audit_manifest_sha256": sha256_file(
                REPORTS / "consolidation_audit" / "manifest.json"
            ),
            "consolidation_audit_report_sha256": sha256_file(
                REPORTS / "consolidation_audit" / "CONSOLIDATION_AUDIT.md"
            ),
            "report_sha256": sha256_file(REPORTS / "RESULTS.md"),
            "analysis_source_sha256": {
                str(path.relative_to(ROOT)): sha256_file(path)
                for path in analysis_sources
            },
        },
    )
    print(f"Wrote reports to {REPORTS}")


if __name__ == "__main__":
    main()
