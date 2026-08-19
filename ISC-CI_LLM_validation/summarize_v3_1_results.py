#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import summarize_results as shared
from iscci_validation.evaluation import HIGHER_IS_BETTER, METRICS
from iscci_validation.provenance import sha256_file, write_json


ROOT = Path(__file__).resolve().parent
PAPER = ROOT.parent / "ISCCI_2026Jan_ArXiv.pdf"
GENERATION = (
    ROOT.parent
    / "artifacts"
    / "leuven_feature_generation"
    / "v3.1"
    / "leuven_v3.1_qwen2_5_72b"
)
VALIDATION = ROOT / "artifacts" / "validation_v3_1"
EVALUATION = VALIDATION / "evaluation"
SIMULATIONS = VALIDATION / "paper_simulations"
CONSOLIDATION = ROOT / "artifacts" / "v3_1_consolidation"
V3_CONSOLIDATION = ROOT / "artifacts" / "v3_consolidation"
REPORTS = ROOT / "reports" / "v3_1"

SIMULATION_MEASURES = {
    "induction_mean_r_mean": "mean induction r",
    "similarity_domain_mean_r_mean": "mean in-domain similarity r",
    "asymmetry_score_human_pearson_mean": "asymmetry r",
    "context_nonmonotonicity_agreement_mean": "context nonmonotonicity agreement",
    "thematic_agreement_mean": "thematic agreement",
    "similarity_context_direction_agreement_mean": "similarity-context direction agreement",
}


def human_rdm_by_condition(input_rdm: pd.DataFrame) -> dict[str, float]:
    result: dict[str, float] = {}
    for row in input_rdm.itertuples():
        if row.condition_a == "human":
            result[str(row.condition_b)] = float(row.object_rdm_spearman)
        elif row.condition_b == "human":
            result[str(row.condition_a)] = float(row.object_rdm_spearman)
    return result


def consolidation_summary() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for version, path in [("v3", V3_CONSOLIDATION), ("v3.1", CONSOLIDATION)]:
        manifest = json.loads((path / "manifest.json").read_text())
        for prompt in manifest["primary_summary"]:
            rows.append(
                {
                    "version": version,
                    "prompt": prompt["prompt_variant"],
                    "source_phrase_types": prompt["phrase_types"],
                    "consolidated_clusters": prompt["clusters_total"],
                    "retained_tasks": prompt["retained_training_tasks"],
                    "positive_cells": prompt["training_positive_cells"],
                    "density": prompt["training_density"],
                }
            )
    return pd.DataFrame(rows)


def paired_delta_table(
    training: pd.DataFrame,
    ceiling: pd.DataFrame,
    simulations: pd.DataFrame,
    input_rdm: pd.DataFrame,
) -> pd.DataFrame:
    train = training.set_index("condition")
    ceil = ceiling.set_index("comparison_group")
    sim = simulations.set_index("condition")
    rdm = human_rdm_by_condition(input_rdm)
    rows: list[dict[str, object]] = []

    for prompt in "ABC":
        old = f"v3_{prompt}"
        new = f"v3_1_{prompt}"
        measures: list[tuple[str, float, float, bool]] = [
            ("retained tasks", train.loc[old, "tasks"], train.loc[new, "tasks"], True),
            (
                "positive cells",
                train.loc[old, "positive_cells"],
                train.loc[new, "positive_cells"],
                True,
            ),
            ("input object RDM vs human", rdm[old], rdm[new], True),
        ]
        old_ceiling = ceil.loc[f"{old}_vs_human_retrain"]
        new_ceiling = ceil.loc[f"{new}_vs_human_retrain"]
        for metric in METRICS:
            measures.append(
                (
                    metric,
                    old_ceiling[f"{metric}_mean"],
                    new_ceiling[f"{metric}_mean"],
                    HIGHER_IS_BETTER[metric],
                )
            )
        for column, label in SIMULATION_MEASURES.items():
            measures.append((label, sim.loc[old, column], sim.loc[new, column], True))

        for measure, old_value, new_value, higher_is_better in measures:
            rows.append(
                {
                    "prompt": prompt,
                    "measure": measure,
                    "v3": float(old_value),
                    "v3_1": float(new_value),
                    "delta_v3_1_minus_v3": float(new_value - old_value),
                    "higher_is_better": higher_is_better,
                    "v3_1_improved": (
                        new_value > old_value
                        if higher_is_better
                        else new_value < old_value
                    ),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    shared.VALIDATION = VALIDATION
    shared.EVALUATION = EVALUATION
    shared.SIMULATIONS = SIMULATIONS
    shared.REPORTS = REPORTS

    training = shared.training_summary()
    ceiling = shared.ceiling_summary()
    simulations = shared.flattened(shared.simulation_summary())
    input_rdm = pd.read_csv(EVALUATION / "input_matrix_rdm_comparisons.csv")
    deltas = paired_delta_table(training, ceiling, simulations, input_rdm)
    consolidation = consolidation_summary()

    training.to_csv(REPORTS / "training_summary.csv", index=False)
    ceiling.to_csv(REPORTS / "ceiling_summary.csv", index=False)
    simulations.to_csv(REPORTS / "paper_simulation_summary.csv", index=False)
    input_rdm.to_csv(REPORTS / "input_matrix_rdm_comparisons.csv", index=False)
    deltas.to_csv(REPORTS / "v3_1_vs_v3_deltas.csv", index=False)
    consolidation.to_csv(REPORTS / "consolidation_comparison.csv", index=False)

    matrix_table = training[
        ["condition", "tasks", "positive_cells", "density", "models"]
    ].copy()
    matrix_table["density"] = matrix_table["density"].map(lambda value: f"{value:.3f}")
    consolidation_table = consolidation.copy()
    consolidation_table["density"] = consolidation_table["density"].map(
        lambda value: f"{value:.3f}"
    )

    ceiling_columns = [
        "comparison_group",
        "context_rdm_spearman_mean",
        "context_dependent_rdm_spearman_median_mean",
        "membership_logit_spearman_mean",
        "binary_membership_agreement_mean",
        "membership_probability_mae_mean",
    ]
    ceiling_table = ceiling[ceiling_columns].copy()
    ceiling_table.columns = [
        "Comparison",
        "Context RDM",
        "Context-dependent RDM",
        "Membership rank",
        "Binary agreement",
        "Probability MAE",
    ]
    for column in ceiling_table.columns[1:]:
        ceiling_table[column] = ceiling_table[column].map(lambda value: f"{value:.3f}")

    simulation_columns = {"condition": "Condition", **SIMULATION_MEASURES}
    simulation_table = simulations[list(simulation_columns)].rename(
        columns=simulation_columns
    )
    for column in simulation_table.columns[1:]:
        simulation_table[column] = simulation_table[column].map(
            lambda value: f"{value:.3f}"
        )

    primary_deltas = deltas[
        deltas["measure"].isin(
            [
                "retained tasks",
                "input object RDM vs human",
                "context_rdm_spearman",
                "context_dependent_rdm_spearman_median",
                "membership_logit_spearman",
                "mean induction r",
                "mean in-domain similarity r",
            ]
        )
    ].copy()
    for column in ["v3", "v3_1", "delta_v3_1_minus_v3"]:
        primary_deltas[column] = primary_deltas[column].map(lambda value: f"{value:.3f}")

    ceiling_index = ceiling.set_index("comparison_group")
    simulation_index = simulations.set_index("condition")
    rdm_to_human = human_rdm_by_condition(input_rdm)
    human_ceiling = ceiling_index.loc["human_retrain_vs_retrain"]
    v3_1_b_ceiling = ceiling_index.loc["v3_1_B_vs_human_retrain"]
    b_context_ceiling = (
        v3_1_b_ceiling["context_rdm_spearman_mean"]
        / human_ceiling["context_rdm_spearman_mean"]
    )
    b_dependent_ceiling = (
        v3_1_b_ceiling["context_dependent_rdm_spearman_median_mean"]
        / human_ceiling["context_dependent_rdm_spearman_median_mean"]
    )
    b_membership_ceiling = (
        v3_1_b_ceiling["membership_logit_spearman_mean"]
        / human_ceiling["membership_logit_spearman_mean"]
    )

    report = f"""# ISC-CI V3.1 Validation

Generated by `summarize_v3_1_results.py` from the cumulative V3.1 validation artifact.

## Design

V3.1 changes only the three feature-generation prompts. It uses the same Qwen2.5-72B model,
293 Leuven objects, 20 paired participant seeds per object and prompt, locked consolidation
parameters, strict `response count > 3` and `positive objects > 3` cutoffs, ISC-CI
architecture, frozen semantic embedding, training episodes, optimizer, 400 epochs, seeds 0-3,
fixed evaluation contexts, and paper-simulation code as V3.

All 17,580 planned V3.1 responses completed with no parse or request errors. Five retained
semantic merges were reviewed. Four equivalent merges passed; `driven on highways` versus
`driven on roads` was rejected and split before matrix construction. Retained task counts are
unchanged over embedding thresholds 0.80-0.90.

{consolidation_table.to_markdown(index=False)}

## Training Matrices

{matrix_table.to_markdown(index=False)}

## Fixed-Context Human Comparison

{ceiling_table.to_markdown(index=False)}

Except probability MAE, higher values indicate greater model similarity. Human self-comparison
is the run-to-run ceiling; released-versus-retrained human is the reconstruction check. Every
candidate row averages all 16 candidate-seed by human-seed comparisons.

## Paired V3.1 Minus V3 Changes

{primary_deltas.to_markdown(index=False)}

These are prompt-matched descriptive deltas, not independent statistical tests: V3 and V3.1
used the same model family and paired generation seeds, but each ISC-CI condition has only four
training seeds. Full metric deltas are in `v3_1_vs_v3_deltas.csv`.

## Paper Simulations

{simulation_table.to_markdown(index=False)}

Values are means across four seeds. The analyses reproduce the two paper notebooks: five
induction datasets, seven induction phenomena, nine similarity domains, asymmetric similarity,
thematic and context-dependent paired choices, and the stochastic similarity-context LCA.

## Interpretation

Prompt B is the clearest V3.1 improvement, but the gain is not universal. It produces 175
trainable tasks versus 115 for V3 B and raises input object-RDM similarity to human norms from
{rdm_to_human['v3_B']:.3f} to {rdm_to_human['v3_1_B']:.3f}. All five fixed-context measures
move toward the retrained human models. Its context, context-dependent, and membership-rank
correlations reach {b_context_ceiling:.1%}, {b_dependent_ceiling:.1%}, and
{b_membership_ceiling:.1%} of the human run-to-run ceiling, respectively. In the paper
simulations, B improves in-domain similarity
({simulation_index.loc['v3_B', 'similarity_domain_mean_r_mean']:.3f} to
{simulation_index.loc['v3_1_B', 'similarity_domain_mean_r_mean']:.3f}), asymmetry, thematic
agreement, and context-dependent nonmonotonicity, while induction and similarity-context
direction decline.

Prompt C is mixed: it improves context-dependent geometry and gives the strongest free-
generation induction result ({simulation_index.loc['v3_1_C', 'induction_mean_r_mean']:.3f}),
but membership rank, asymmetry, thematic agreement, and similarity-context direction worsen.
Prompt A improves induction and asymmetry but loses substantially on context-dependent geometry,
membership rank, in-domain similarity, thematic agreement, and similarity-context direction.

The defensible conclusion is that V3.1 successfully improved the first-to-mind B condition,
especially recurring feature coverage and object geometry, but did not raise all prompts or all
behaviors. Even B remains outside human run-to-run variability and far below V2's input object-
RDM correlation of {rdm_to_human['v2']:.3f}. V3.1 is therefore evidence that prompt design can
materially improve these generated norms, not evidence that the norms are interchangeable with
human feature data. These comparisons remain descriptive because there are four ISC-CI seeds
per condition and V3.1 uses the same Qwen model family as V3.

## Reproduction

```bash
cd ISC-CI_LLM_validation
python consolidate_v3.py \\
  --config configs/v3_1_validation.json \\
  --long-csv ../artifacts/leuven_feature_generation/v3.1/leuven_v3.1_qwen2_5_72b/generated_features_long.csv \\
  --output-dir artifacts/v3_1_consolidation \\
  --artifact-prefix v3_1 \\
  --manual-review-csv configs/v3_1_consolidation_manual_review.csv
python run_validation.py \\
  --config configs/v3_1_validation.json \\
  --output-dir artifacts/validation_v3_1 \\
  --base-validation-dir artifacts/validation_v3 \\
  --include-v3-1
python evaluate_validation.py \\
  --config configs/v3_1_validation.json \\
  --validation-dir artifacts/validation_v3_1
python run_paper_simulations.py \\
  --config configs/v3_1_validation.json \\
  --validation-dir artifacts/validation_v3_1
python summarize_v3_1_results.py
pytest -q tests
```
"""
    (REPORTS / "RESULTS.md").write_text(report, encoding="utf-8")

    source_paths = [
        ROOT / "consolidate_v3.py",
        ROOT / "run_validation.py",
        ROOT / "evaluate_validation.py",
        ROOT / "run_paper_simulations.py",
        ROOT / "summarize_results.py",
        ROOT / "summarize_v3_1_results.py",
        ROOT / "configs" / "v3_validation.json",
        ROOT / "configs" / "v3_1_validation.json",
        ROOT / "configs" / "v3_1_consolidation_manual_review.csv",
        ROOT / "DECISIONS.md",
        ROOT / "README.md",
        *sorted((ROOT / "iscci_validation").glob("*.py")),
    ]
    write_json(
        REPORTS / "report_manifest.json",
        {
            "paper_pdf": str(PAPER.resolve()),
            "paper_pdf_sha256": sha256_file(PAPER),
            "generation_manifest": str((GENERATION / "manifest.json").resolve()),
            "generation_manifest_sha256": sha256_file(GENERATION / "manifest.json"),
            "generation_long_csv_sha256": sha256_file(
                GENERATION / "generated_features_long.csv"
            ),
            "consolidation_manifest_sha256": sha256_file(
                CONSOLIDATION / "manifest.json"
            ),
            "validation_manifest_sha256": sha256_file(VALIDATION / "manifest.json"),
            "evaluation_manifest_sha256": sha256_file(EVALUATION / "manifest.json"),
            "paper_simulation_manifest_sha256": sha256_file(
                SIMULATIONS / "manifest.json"
            ),
            "results_sha256": sha256_file(REPORTS / "RESULTS.md"),
            "analysis_source_sha256": {
                str(path.relative_to(ROOT)): sha256_file(path) for path in source_paths
            },
        },
    )
    print(f"Wrote V3.1 reports to {REPORTS}")


if __name__ == "__main__":
    main()
