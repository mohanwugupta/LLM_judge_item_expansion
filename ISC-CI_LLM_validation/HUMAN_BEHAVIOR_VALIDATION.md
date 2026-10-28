# External Human-Behavior Validation

Protocol added 2026-10-01. This is downstream testing of semantic norm replacement,
not a way to tune the feature-generation or completion pipeline.

## Sources and Fidelity Gate

The pinned original implementation is in
`upstream/IntegratedSemanticsControlContextInference/`. Its two notebooks, seven
task CSVs, plural-to-singular mapping, and released seed-3 checkpoint are the
reference. No new behavioral stimuli, human ratings, or benchmark are generated.
The PDF is `../ISCCI_2026Jan_ArXiv.pdf`.

Run from the parent project directory:

```bash
python3 ISC-CI_LLM_validation/run_human_behavior_validation.py --fidelity-only
python3 ISC-CI_LLM_validation/run_human_behavior_validation.py
```

The second command trains missing seeds only AFTER the gate passes. An evaluation-only
run can use `--evaluate-only`. Exit status 2 means fidelity is blocked, not a crashed
simulation. See `artifacts/human_behavior_validation_statistical/fidelity_gate.json` and
`fidelity_statistic_checks.csv` before interpreting any cross-condition result.

Original item probes and released deterministic statistics use a 1e-5 absolute
tolerance, specified before the first run. The initial strict reconstructed-human
numerical gate is preserved in `artifacts/human_behavior_validation/`.

After that human-only audit, the user explicitly chose exact released replication
plus statistical reconstructed replication. Protocol 1.1 checks whether each
reconstructed four-seed mean lies inside the 95% released-human-seed prediction
interval: reference mean +/- t(.975, df=3) * reference seed SD * sqrt(1 + 1/4).
All four released seeds receive identical probes. This is a run-variability
consistency envelope, NOT a formal equivalence test; four seeds provide limited
precision. A zero-variance reference requires identical reconstruction within
roundoff. No best-seed selection is allowed. The criterion was specified after
inspecting initial HUMAN results, before any LLM-condition behavior, and is not
described as preregistered. It changes no features or training settings.

Current outputs are in `artifacts/human_behavior_validation_statistical/`.

An additional source-consistency check records a real discrepancy: PDF page 73
reports 36/38 correct context directions; the saved similarity notebook cell 11
reports 37/38. The LCA is stochastic. Its equations, temperature, 100 simulations,
500 steps, and 100-step burn-in are unchanged, but all conditions and all training
seeds now receive the SAME stochastic probe seed. Exact reproduction of a saved
stochastic realization cannot be inferred from numerical agreement of deterministic
calculations. The historical stochastic notebook number and source-consistency
check remain diagnostics rather than exact-number gates in protocol 1.1. Required
deterministic checks are unchanged. Neither the random seed nor LCA parameters
were changed to obtain a favorable result; the fixed probe happened to match the PDF.

The runner separately checks item predictions against the unedited upstream model
graph AND `analysis.calc_argument_strength` for all 1,067 observed induction arguments and
7,000 synthetic phenomenon arguments, for the released model and each human model.
Tests also compare the vectorized LCA to the original scalar sequential-update
equations with identical random draws.

## Conditions and Training

The registry reads the completed cumulative V3.1 directory and V4 validation
directory, deduplicating identical checkpoints and rejecting conflicting copies.
Human, V2, V3-A/B/C, V3.1-B, and the four V4 locked/calibrated fixed-B/ensemble
conditions are required. V3.1-A/C and named V4 source-only conditions are included
when available. Source-only conditions are not created from scratch; if partially
trained and their finalized matrix exists, missing seeds can be completed.

New conditions receive seeds 0, 1, 2, 3, 400 epochs, 1,024 episodes per epoch,
batch size 128, one-/two-shot support sets, Adam at .001, task-loss weight .5,
64-dimensional frozen released seed-3 semantic embeddings, and unchanged 128-unit
context and context-dependent layers. Only task-head size follows feature count.
Existing checkpoints must match the matrix, object order, full training protocol,
architecture, completed epoch count, identity, and frozen embedding exactly.

V4 training requires a finalized matrix manifest and complete judgment manifest
with matching hashes. The October 1 pretraining audit found 2,005 invalid cells
and no finalized matrices; this stage does not repair, zero-fill, or threshold them.
Behavioral outcomes cannot change feature judgments, retained features, thresholds,
architecture, episode budget, seeds, or condition selection.

## Behavioral Measures

| Original dataset | Original measure preserved | Additional direct validation |
| --- | --- | --- |
| Induction arguments (five datasets) | Within-dataset Pearson correlation of model logits and observed ratings | Spearman; grouped-CV rating error |
| Within-domain similarity (nine domains) | Pearson of averaged forward/reverse sigmoid outputs and mean normalized human similarity | Spearman; individual-response CV error |
| Directed asymmetry | Direction agreement for the same 34 human-significant pairs; asymmetry R-squared; binomial test | All 50 pairs, magnitude correlations, individual choices, CV Brier/error |
| Multi-alternative context similarity | Context-effect direction agreement on the same 38 human-positive sets; binomial test | All 41 sets, magnitude correlations, effect error, per-trial three-choice probabilities/Brier/log loss |
| Context-dependent nonmonotonicity | Within-participant fraction choosing the single-premise argument (argument 2) | Model choices on those exact trials; majority agreement and human preference correlations |
| Thematic categorization | Within-participant fraction choosing the context-specific option (argument 2) | Model choices on those exact trials; majority agreement and human preference correlations |
| Inductive phenomena | Original sigmoid high/low effects and notebook paired/unpaired t-tests | Synthetic probes only: the supplied CSV has NO item-level human outcomes |

For the last two human choice tasks, the old runner's majority-agreement measure
is supplemental: it does not replace Figures 5/6's original argument-2 preference
measure. Original observed participant preferences are saved separately.

Every original row has a source-row identifier and a saved prediction. No model is
evaluated only on a favorable subset: the paper's human-defined subsets are fixed
across conditions, and all-item supplemental predictions remain available. The
original model-selected induction/categorization stimuli are retained, but their
selection limits claims of an entirely independent test of the original model.

## Separate Evaluation Levels

1. Raw object-by-feature cosine similarity predicts the original within-domain
   human similarity judgments, without a trained ISC-CI network.
2. Frozen pretrained semantic-activation cosine similarity predicts those same
   human judgments. These activations are shared across conditions; agreement of
   1.0 is a control property, not evidence of independent semantic learning.
3. Fully trained ISC-CI models answer the original induction/similarity queries,
   with exactly the same scoring calculations across conditions.

The condition summary separately includes input RDM agreement with Leuven,
semantic-activation agreement, context/context-dependent RDM agreement, and
membership agreement relative to independently seeded human models. Human
self-comparisons exclude the identical model. Pairwise baseline comparisons are
first averaged per candidate seed rather than treating 16 seed pairs as independent.

Primary input geometry retains the historical validation convention: undefined
cosine distances for zero feature vectors are set to zero before ranking. This is
not a claim that those objects are semantically identical. Defined-pair correlations,
zero-object counts, and undefined-pair counts are also reported as a sensitivity.
Raw-feature prediction of human similarity remains undefined for these vectors;
rows are retained and coverage reported, not silently treated as negatives.

## Mapping, Uncertainty, and Comparisons

Original uncalibrated statistics remain primary. Supplemental human-scale error
uses five-fold grouped CV linear mappings, clipped to [0,1] for choice proportions.
Reverse similarity pairs share a fold. Induction arguments with the same dataset,
first premise, and conclusion share a fold. No held-out rating is used to fit its
mapping. Fold IDs and fitted coefficients are saved. These measurement fits never
enter feature construction or training.

Uncertainty in behavioral item statistics resamples both item clusters and model
seeds (2,000 crossed bootstrap repetitions, percentile 95% intervals). Related
arguments and reverse pairs stay together. Central contrasts use identical item
draws and paired training seeds. CV error intervals condition on the saved
out-of-fold mappings; mappings are NOT refit in each bootstrap. This limitation is
explicit in outputs. The intervals concern generalization across human items and
model seeds, not a full participant-level generative model.

Original within-domain participant-versus-other-participant reliability is reported,
with an aligned leave-one-out mean version alongside the notebook's sum-rest
calculation. Participant split-half correlations and Spearman-Brown estimates are
also saved for datasets with individual responses. These are reliability estimates,
not guaranteed model noise ceilings. Aggregate-only induction datasets are marked
unavailable rather than assigned invented reliability values.

Contrasts cover V4 versus V3.1-B, source-only when available, V2, and human;
calibrated versus locked; and ensemble versus fixed-B at matched selectivity.
Geometry-versus-behavior analyses treat CONDITIONS, not four seeds, as observations,
exclude the defining human reference, and report Pearson/Spearman associations and
exploratory permutation tests. Related conditions are not independent replications.
No feature or training choices are made from these associations.

## Outputs and Current Status

All new files are inside `ISC-CI_LLM_validation/`.

- `per_model/*/*_items.csv`: unaggregated model predictions, human means, immutable
  selections, CV fold predictions and coefficients.
- `per_model/*/*_trials.csv`: predictions joined to every original human observation.
- `per_model/*/trial_statistics.csv`: individual-response errors and choice metrics.
- `all_individual_trial_statistics.csv`: all evaluated/released trial statistics.
- `fidelity_*`: gate checks, human/released per-seed diagnostics, failure reasons.
- `human_reliability.csv`, `human_participant_reliability.csv`: reliability context.
- `paper_human_*`: original observed participant-level choice dependent measures.
- `behavior_per_seed.csv`, `behavior_summary.csv`: external behavioral results.
- `paired_condition_contrasts.csv`: central differences and paired intervals.
- `representation_per_seed.csv`: separate model-versus-human representation metrics.
- `similarity_levels_1_2_{items,trials}.csv`: separate raw/semantic predictions.
- `geometry_behavior_association.csv`: exploratory geometry/behavior relationship.
- `geometry_behavior_defined_pairs_sensitivity.csv`: separate finite-pair sensitivity.
- `condition_summary.csv`: one row per condition, explicitly including blocked/missing
  conditions; no invented V4 behavior when models do not exist.
- `manifest.json`: source, data, checkpoint, config, and environment provenance.

The initial October 1 fidelity run was BLOCKED under the strict numerical gate. Item-level original-code
checks pass. Sixteen deterministic released-checkpoint statistics reproduce within
1e-5; stochastic context agrees with the PDF's 36/38 rather than the notebook's
37/38. Reconstructed human statistics differ beyond 1e-5. The user then authorized
statistical reconstructed replication, which passes the specified reference-seed
consistency envelope. See the CURRENT statistical manifest and `STATUS.md` for
execution status. Full V4 comparisons still require finalized matrices and four
trained seeds; unit tests are not a substitute for a completed run.

The statistical run completed all eight available conditions (32 models, four seeds
each), plus four released reference models. No additional training was needed for
those conditions. See `artifacts/human_behavior_validation_statistical/RESULTS.md`.
Every V4 row remains explicitly missing. Resume with the same main command after
upstream judgment and matrix finalization; the runner then trains missing eligible
seeds before evaluating them. A partial run can exit successfully but still has
`complete: false`: always check the manifest rather than only the shell exit code.

Two derived-report corrections are recorded with parent manifest snapshots:
induction trial errors must use the original `Human` column when tasks are stacked;
historical RDM ranks and defined-pair sensitivity must not be conflated. The helper
`refresh_human_behavior_derived.py` rebuilds reports only after checking cohort,
data, matrix, checkpoint, preprocessing, and raw-probe hashes, and reruns original
graph fidelity. It does not refit mappings or rescore behavioral items. Both fixes
are also in the main runner. Saved probe implementation signatures remain intact;
refresh provenance records the newer derived implementation separately.
