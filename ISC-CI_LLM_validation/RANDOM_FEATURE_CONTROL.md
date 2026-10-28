# Human-Sized Random-Feature Control

## Design Locked October 28, 2026

The investigator approved one human-sized negative-control condition, `random_human`,
before inspecting its results. Independently permute object assignments within each
column of the original retained human binary matrix. Use `numpy.default_rng(20261028)`
(PCG64), one fixed realization, 293 objects, and 385 features. Keep object and feature
IDs/order fixed. Every feature retains exactly its original prevalence; total positives
remain 8,056. Do not impose or tune an outcome criterion on the shuffled matrix.

This destroys meaningful object-feature assignments and joint feature structure, but
does not preserve per-object feature counts. Feature labels are metadata, not textual
inputs to the neural network. No feature is pruned or reweighted for this control.

Train four seeds (0-3) on that **same** random matrix with the existing `train_run` /
`train_model` implementation: the same frozen 64-dimensional released semantic encoder,
400 epochs, 1,024 episodes/epoch, batch 128, Adam 0.001, 0.5 task-loss weight, support
sizes 1/2, and the same 128-dimensional context/dependent architecture. This is
409,600 episodes per seed. Only the randomized human-size memberships differ from
the reconstructed human control; its task head has the same 385 outputs.

## Evaluation And Preservation

Reuse the completed original-paper human validation protocol 1.1.0, original human
stimuli, preprocessing, queries, scoring, five-fold CV mappings, and common LCA seed.
Require the original fidelity gate to have passed and check that its scientific code,
human data, human matrix, and encoder checkpoint hashes still match. Validate all new
checkpoints, 3,200 training updates, finite metrics/weights, and parity with the
unedited paper model/scorer. The gate is reused, not weakened or refitted.

Generate every new model's item/trial predictions and participant preferences with
`run_predictions`; use the existing 2,000-draw crossed item/seed bootstrap and original
representation probes. Reuse cached human representation evaluations only with matched
human checkpoint hashes and exactly matched probe arrays. Preserve raw-feature,
shared-encoder, and fully trained evaluation levels. Compute one paired human-minus-
random contrast for every original outcome; do not label preference rates accuracy.

No old model is retrained and no old predictions, scores, uncertainty intervals, or
scientific artifacts are overwritten. Combined CSVs retain the original bytes as a
prefix and append only the random condition. Original per-model files and matrices
are exposed through links; only the dedicated runner writes new random outputs.
All 211 source files recorded in the original figure manifest are hash-checked before
and after the extension. Original seven-condition common-finite-pair inclusion stays
fixed when the new control is plotted. Original four contrasts stay unchanged; the
human-minus-random contrast is an explicitly later negative-control addition.

## Replay

From the project root, using the existing scientific Python environment:

```bash
python ISC-CI_LLM_validation/run_random_feature_control.py
```

The verified local interpreter is
`/Users/mg9965/.pyenv/versions/3.11.8/bin/python3`. The runner reuses complete
checkpoints and cached complete per-model predictions. Interrupted training for an
unsaved seed restarts that seed's frozen 400-epoch run, not from an intermediate epoch.
Bootstrap/report tables are reproducibly regenerated. No LLM or cluster calls occur.
Stages can be invoked separately:

```bash
python ISC-CI_LLM_validation/run_random_feature_control.py --stage train
python ISC-CI_LLM_validation/run_random_feature_control.py --stage evaluate
python ISC-CI_LLM_validation/run_random_feature_control.py --stage plot
```

`--output-dir`, `--report-dir`, `--workers`, and `--randomization-seed` are explicit
options. A different randomization cannot overwrite a saved control: use a different
directory and specify that realization independently of its human behavioral result.
Changing training/evaluation settings is not supported by this control protocol.

The reporting-only invocation, after evaluation, is:

```bash
python ISC-CI_LLM_validation/plot_finalized_v4_results.py \
  --include-random-control \
  --behavior-dir ISC-CI_LLM_validation/artifacts/random_feature_control_20261028/combined_behavior \
  --validation-dir ISC-CI_LLM_validation/artifacts/random_feature_control_20261028/combined_validation \
  --output-dir ISC-CI_LLM_validation/reports/v4_20261028_random_control/figures_final
```

## Outputs

- `artifacts/random_feature_control_20261028/randomization_manifest.json`: fixed RNG,
  matrix hashes, audit scope, and uncertainty limitations.
- `feature_prevalence_audit.csv`, `object_degree_audit.csv`: preserved/changed marginals.
- `validation/`: random matrix, four checkpoints, metrics, and training provenance.
- `behavior/`: only the new condition's predictions, summaries, contrasts, probe
  outputs, and original-graph checks. Synthetic phenomena stay separate from humans.
- `combined_behavior/`: cumulative 14-condition tables and original per-model links;
  the full aggregate original item table is not duplicated because plotting uses the
  authoritative per-model tables. New random aggregate predictions remain in `behavior/`.
- `combined_validation/`: original matrix links plus the random matrix.
- `reports/v4_20261028_random_control/figures_final/INDEX.md`: updated original graph
  families with the added row, panel data, methods, interpretation, and QA. Fourteen-
  condition supplement filenames use `all14`; the original `all13` package is preserved.

Test with:

```bash
python -m pytest -q ISC-CI_LLM_validation/tests/test_random_feature_control.py \
  ISC-CI_LLM_validation/tests/test_final_results_figures.py
```

## Interpretation Limits

Completed results (four-seed means; similarity/induction are descriptive unweighted
means over nine domains/five datasets, not pooled correlations):

| Original measure | Human-trained | Random-feature-trained |
| --- | ---: | ---: |
| Similarity Pearson r, domain mean | .559 | .542 |
| Induction Pearson r, dataset mean | .498 | .471 |
| Asymmetry direction, 34 pairs | .757 | .551 |
| Context direction, 38 sets | .947 | .882 |
| Context-specific argument-2 rate | .853 | .625 |
| Nonmonotonicity argument-2 rate | .788 | .628 |

The randomized matrix has input-RDM Spearman agreement -.004529 and raw-feature
domain-mean similarity r .020783. The shared pretrained encoder's descriptive mean
is .538601. All nine human-minus-random similarity intervals include zero; this is
not a formal equivalence finding. Other original outcomes differ more, and a
difference in argument-2 preference is not automatically better human prediction.
All 18 original outcomes, including unfavorable differences, are retained in the
paired table and added forest plot. Intervals are unadjusted for multiplicity.

All eight new original-graph/scorer checks passed (maximum absolute difference
1.67e-6, tolerance 1e-5). The original 13-condition numerical tables and all 211
original figure-input hashes remain unchanged; ten cumulative CSVs preserve the
original byte prefixes. There are 25 passing focused tests. The completed figure
package has 31 PDF/SVG/PNG filesets, including the new comparison.

The matched human-versus-random comparison tests meaningful context assignments
**conditional on the same human-pretrained encoder**. It does not remove that encoder,
identify an additive percentage of performance caused by it, or eliminate generic
learned-network contributions. Comparisons against differently sized LLM inventories
are not capacity/density-matched ablations. One fixed shuffled matrix means item/seed
intervals do not quantify variability across randomization realizations. No behavioral
result may select a new randomization, feature rule, or training setting.
