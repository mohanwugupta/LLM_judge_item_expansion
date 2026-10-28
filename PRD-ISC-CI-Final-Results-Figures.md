# PRD: Interpretable final ISC-CI results, including V3

**Status:** Ready for implementation  
**Scope:** Reporting and visualization of completed experiments  
**Source experiment:** Finalized October 2, 2026 validation; 13 conditions, four training seeds per condition

## 1. Objective

Create a reproducible figure package that explains what the completed experiments establish about replacing human semantic feature norms with LLM-derived feature spaces, and how the resulting ISC-CI models predict actual human judgments.

The immediate deliverable is four main figures, supplementary comparisons covering every completed condition, and a short interpretation report. Use existing matrices, predictions, human observations, and statistical outputs. Do not initiate new LLM judgments, training, model simulations, or the approximately 39-million-cell evaluation.

The longer-term research goal is generating useful semantic feature spaces for arbitrary words and eventually studying false memory. This deliverable assesses the current prerequisite: feature-space quality and human behavioral prediction. It does not establish DRM or false-memory performance, generalization to arbitrary words, or a wholly human-data-free model.

## 2. Questions the figures must answer

1. How do V2, V3, V3.1, and V4 differ in feature density, object geometry, and coverage?
2. Which representations and trained models predict human similarity judgments, within each original domain?
3. Do improvements in Leuven feature geometry translate into improvements in induction and contextual choice?
4. Do models recover the direction of human context effects, their magnitude, and their variation across items?
5. What changes when applying completion to a fixed vocabulary, versus comparing the large completed ensemble with source-only features?

Do not collapse these into one overall performance score. A model can recover coarse geometry or effect direction while poorly predicting individual judgments or effect sizes.

## 3. Conditions and display labels

Use the exact condition IDs from saved artifacts. These IDs are case-sensitive. The main figures contain seven conditions in the following order.

| Condition ID | Display label | Role | Retained contexts |
| --- | --- | --- | ---: |
| `human` | Human-norm-trained ISC-CI | Reconstructed human-norm training control | 385 |
| `v2` | V2 | Earlier generated feature space | 382 |
| `v3_B` | V3-B | Original V3, prompt B | 115 |
| `v3_1_B` | V3.1-B | Expanded B vocabulary before V4 completion | 175 |
| `v4_b_calibrated` | V4 fixed-B, calibrated | Completion of the same 175-feature vocabulary | 175 |
| `v4_ensemble_source_only` | V4 source-only | Judged/calibrated features restricted to source associations | 2,356 |
| `v4_ensemble_calibrated` | V4 completed ensemble | Broad discovery plus calibrated completion | 80,699 |

V3-B is the primary V3 comparator because it matches the B prompt used by V3.1-B and fixed-B V4. This is a design-based choice, not selection of the best V3 result. Do not average A/B/C into a fictional single V3 model.

Mandatory supplementary figures and machine-readable tables include all 13 conditions: the seven above plus `v3_A`, `v3_C`, `v3_1_A`, `v3_1_C`, `v4_b_locked_v2`, and `v4_ensemble_locked_v2`. Distinguish reconstructed `human` models from any released-reference checkpoints; do not pool references into the four-seed human control.

## 4. Paper precedents and their application

The plotting design draws on the following reviewed papers. Figure numbering refers to the reviewed versions; identify the version used in the final source note.

| Source | Useful precedent | Application here |
| --- | --- | --- |
| NOVA, *AI-Enhanced Semantic Feature Norms for 786 Concepts*, Figure 4 | Feature-count/prevalence distributions and ordered object cosine-distance matrices | Figure 1: feature-space structure and Leuven geometry |
| NOVA, Figure 5 | Validation against actual participant choices | Keep human behavioral observations distinct from human feature norms and model controls |
| ISC-CI, *An Integrated Model of Semantics and Control, Part 2: Solving the Similarity Paradox Through Context Inference*, Figure 7 | Similarity correlations by nine domains; participant reliability | Figure 2: domain-specific similarity prediction |
| ISC-CI, Figure 2 | Induction correlations across five datasets | Figure 3: dataset-specific induction prediction |
| ISC-CI, Figures 5–6 | Original context-dependent argument preferences | Figure 3: observed participant preferences and predicted argument-2 rates |
| ISC-CI, Figure 10 | Direction of context-dependent similarity shifts | Figure 4: direction plus item-level magnitude |
| ISC-CI, Figure 8 | Similarity asymmetry, including direction and magnitude | Supplement: asymmetry direction and item-level fit |

References: [NOVA journal article](https://doi.org/10.1111/tops.70037), [NOVA preprint](https://arxiv.org/abs/2505.10718), [ISC-CI preprint](https://osf.io/preprints/psyarxiv/fxc87_v3), and [ISC-CI code and data](https://github.com/tylergiallanza/IntegratedSemanticsControlContextInference).

Do not copy NOVA's feature-verification d-prime or triadic-choice accuracy into this evaluation: those require different human labels and stimuli. NOVA's t-SNE display is optional illustration, not evidence of recovered geometry, and is not a required main figure.

## 5. Existing inputs and code reuse

Inspect the repository and reuse its loaders, condition registry, scoring definitions, ordering, and saved uncertainty outputs. The paths below are documented experiment locations relative to the project root; discover exact filenames and schemas rather than assuming undocumented APIs exist.

| Input | Location or documented output | Purpose |
| --- | --- | --- |
| Final scientific report and source snapshot | `ISC-CI_LLM_validation/reports/v4_20261002/` | Frozen interpretation, definitions, source hashes |
| Finalized run record | `ISC-CI_LLM_validation/V4_20261002_RUN.md` | Protocol and provenance |
| Feature matrices and provenance | `ISC-CI_LLM_validation/artifacts/v4_matrices/`, plus existing earlier-condition matrix locations | Density, coverage, feature profiles, object RDMs |
| Representational summaries | `ISC-CI_LLM_validation/artifacts/validation_v4_20261002/` | Input, inferred-context, dependent-representation, membership agreement |
| Authoritative human validation | `ISC-CI_LLM_validation/artifacts/human_behavior_validation_v4_20261002/` | Human observations, item predictions, evaluation levels, reliability, intervals, contrasts |
| Final summary tables | `all_condition_summary.csv`, `behavior_summary.csv`, `behavior_per_seed.csv` | Human behavioral metrics and condition summaries |
| Paired contrasts | `central_behavioral_contrasts.csv` | Existing paired crossed item/seed intervals |
| Exploratory associations | `geometry_behavior_association.csv` | Optional condition-level geometry/behavior analysis |

Relevant existing code includes `run_human_behavior_validation.py`, `evaluate_validation.py`, `run_paper_simulations.py`, `audit_v4_feature_profiles.py`, `summarize_finalized_v4.py`, and `iscci_validation/dataio.py` under the validation project. Read and reuse their definitions; do not invoke their full pipelines to make plots. Do not use `resume_finalized_v4_analysis.py` as the plotting entry point.

The supplied standalone paper-simulation CSVs and older `condition_summary.csv` are not a substitute for the final authoritative human-validation tables. Historical simulation outputs use a different LCA seed schedule. Keep protocols separate and visibly label any historical supplemental reproduction. Never fill missing authoritative values from a different protocol.

### Lightweight plotting entry point

Add a dedicated reporting script, for example `ISC-CI_LLM_validation/plot_finalized_v4_results.py`, that reads frozen artifacts and writes only to a new report directory. The proposed filename and CLI are implementation requirements, not claims that this script already exists.

Support explicit input/output directories and a strict default that fails clearly when a required artifact is absent. Rendering must work from saved matrices/tables without checkpoint inference. Avoid loading the multi-gigabyte original judgment payload; use finalized matrix exports and audit summaries. Use cached RDMs where available; otherwise compute only the small 293-by-293 object RDMs with the existing definition.

If a required panel needs predictions or statistics that were not saved, list the missing input and mark the panel blocked. Continue rendering the available figures. Do not silently rerun the model or fabricate an interval. Only lightweight transformations of existing observations/predictions are in scope.

## 6. Main figures

### Figure 1 — Feature-space structure and Leuven geometry

**Question:** What did discovery and completion change in the input representation?

Required components:

- Retained-context counts for the seven main conditions, using a clearly labeled log axis because inventories differ greatly.
- Distributions of positive features per concept and concepts per feature. Include normalized positive-feature fractions and feature prevalence so an 80,699-column ensemble is interpretable alongside a 115-column V3 matrix. State denominators and distinguish counts from fractions.
- Object cosine-distance RDMs with the same 293-object order and color scale. Main examples: human norms, V3-B, V4 source-only, and V4 completed ensemble. Supply remaining main-condition RDMs in the supplement. Reuse the original category order when available; otherwise freeze an order independent of model performance.
- Input-RDM agreement with Leuven for all seven conditions, alongside finite-pair coverage and feature-empty object counts.

If these components overcrowd one page, export separate labeled panels under Figure 1 rather than shrink labels. Human input agreement is 1 by definition. V3/V3.1 historical RDMs used a zero-vector convention; show the saved defined-pair sensitivity separately and make this limitation visible. Do not present arbitrary zero-vector imputation as valid cosine information.

Caption takeaway: completion improves recovery of Leuven object geometry; that does not by itself establish better human behavioral prediction.

### Figure 2 — Predicting actual human similarity judgments

**Question:** How much do raw features and trained ISC-CI contribute, within each domain?

- Nine facets: Birds, Clothing, Fish, Insects, Mammals, Reptiles, Tools, Vehicles, and Weapons. Use the original domain-specific Pearson correlation definition.
- For each of the seven conditions, show raw-feature prediction and fully trained ISC-CI prediction using distinct markers, with saved 95% intervals where available.
- Show the shared human-pretrained semantic-activation baseline once per domain, clearly labeled as shared by every condition. Do not repeat it as seven independent results.
- Show available participant reliability distributions or summaries distinctly from the human-norm-trained model. Label the exact reliability calculation; do not call it an unconditional model noise ceiling.
- Report item/pair counts and missingness. Sparse V3/V3.1 raw cosines remain missing. Include a required common-finite-pair sensitivity comparison within each domain, covering the seven main conditions, to reveal differences due to coverage.

The primary panel preserves the frozen evaluation's original coverage and statistics. The common-pair panel is explicitly a coverage sensitivity analysis and may differ numerically. If uncertainty for this subset is unavailable, show coverage and point estimates without inventing intervals.

Do not pool all domains into one correlation. A descriptive across-domain mean may appear in the caption or summary table only if explicitly labeled as an unweighted mean of nine correlations.

### Figure 3 — Induction and contextual argument preferences

**Question:** Are behavioral gains consistent across tasks?

**Part A:** Five facets for the original induction datasets: Bhatia Experiment 1, Bhatia Experiment 3, Han Experiment 2, Osherson Experiment 4, and Rips. Plot the original dataset-specific Pearson correlations for the seven trained conditions with saved intervals and sample sizes. Do not use synthetic `induction_phenomena.csv` examples as human validation.

**Part B:** Two panels for the original thematic/context-specific argument task and context-dependent nonmonotonicity task. Plot observed participant argument-2 preference distributions, when saved, and each model's original argument-2 choice rate. Map each pipeline task name to the reviewed paper's task/stimuli before labeling it.

These rates are preference measures, not percent-correct accuracy. A high rate is useful only insofar as it matches the relevant observed human outcome. Do not mix a pooled human mean from a legacy CSV with a participant median from the paper. If participant distributions are unavailable, use a clearly labeled, definition-matched saved human summary.

Use saved individual-item agreement/error metrics and paired contrasts in the accompanying table to distinguish overall preference from prediction of which items people choose. Do not label an observed median as the individual-item agreement reference.

Caption takeaway: completion has task-dependent effects; the large ensemble does not recover every original contextual preference effect.

### Figure 4 — Context-effect direction versus magnitude

**Question:** Does a model merely favor the human direction, or predict how strongly context changes each judgment?

Required components:

1. Direction agreement for all seven main conditions on the original 38 human-positive context sets, with saved uncertainty and the denominator printed.
2. Item-level scatterplots of human context effect versus model context effect across all 41 original sets. Main small multiples: human-norm-trained, V3-B, V3.1-B, fixed-B calibrated, source-only, and completed ensemble; put V2 in the supplement if space requires. Use common axes and identify nonpositive human-effect sets with a separate marker.
3. A compact table of saved effect-size correlation and prediction error, plus descriptive mean human/model effects and their clearly defined difference. Include paired contrasts when saved. Do not present mean effect difference as a complete measure of individual-item calibration.

Use the authoritative paper score, verified against existing code:

`E = [p(a | c_a) - p(b | c_a)] + [p(b | c_b) - p(a | c_b)]`.

The probabilities retain the full three-alternative choice denominator; do not renormalize over only a and b. Preserve context/alternative orientation and stable item IDs when joining human and model tables. Use the authoritative common-LCA-seed outputs, not an average assembled from historical runs.

Use zero reference lines. Add an identity line only if human and model scores use genuinely comparable units and definitions; otherwise label the scale difference. Do not z-score away the magnitude difference in the main panel, tune model temperature, or fit a new outcome-driven rescaling. If a seed-mean scatter is shown, say so and preserve per-seed predictions in the exported data.

Caption takeaway: high direction agreement can coexist with a much larger or smaller average shift and weak prediction of which items have the strongest shifts. Agreement with the human-trained model is separate from agreement with observed humans.

## 7. Mandatory supplements

1. **All-condition coverage:** extend behavioral panels and summary tables to all 13 conditions, explicitly including V3 A/B/C and V3.1 A/B/C. Preserve the same scales and definitions as the main figures.
2. **Representational agreement:** separate panels for input RDM, inferred-context RDM, context-dependent RDM, and membership-rank agreement. Do not combine these metrics into a score. Explain that semantic-activation agreement equals 1 by construction because the human-pretrained encoder is shared.
3. **Prespecified contrasts:** display the saved paired differences and 95% intervals for calibrated fixed-B minus V3.1-B, completed calibrated ensemble minus source-only, and calibrated minus locked versions within fixed-B and ensemble. Include favorable and unfavorable original behavioral outcomes. Preserve each metric's units and direction; do not declare unadjusted exploratory contrasts a universal ranking.
4. **Similarity asymmetry:** direction agreement on the original 34 significant human pairs, and item-level human/model asymmetry magnitude using the saved broader item set and exact scoring definition. State different subset sizes explicitly; do not reuse context-effect denominators.
5. **Coverage and missing-data sensitivity:** show finite-pair raw-feature coverage, feature-empty objects, and the saved historical-versus-defined-pair geometry comparison. Source-only/completed comparisons change retained inventory as well as applicability; fixed-B/V3.1-B is the vocabulary-matched contrast.

Optional: a condition-level Leuven-geometry versus behavioral-performance plot using saved `geometry_behavior_association.csv`. If included, label it exploratory, use one point per nonhuman condition, and disclose related conditions, small sample size, and unadjusted comparisons. Four seeds are not four independent conditions.

## 8. Statistical and interpretation rules

- Reuse the frozen original scoring, human-defined subsets, saved paired contrasts, and existing uncertainty definitions. The paired contrast intervals use crossed item/seed resampling; CV mappings were not refit within those bootstraps. Describe that limitation where relevant.
- Display seed variation distinctly from human-item uncertainty. A four-seed SD is not a 95% confidence interval. Input matrices and raw-feature scores have no training-seed variation; do not fabricate four replicates for them.
- Keep raw score correlation, cross-validated rating prediction, majority agreement, preference rate, and effect direction clearly separated. If mapped-prediction metrics are shown, use only the saved cross-validated fits.
- Keep human feature norms, participant observations, reconstructed human-trained models, released references, and shared pretrained activations distinct in legends and captions.
- No threshold tuning, feature deletion/merging/reweighting, model selection, stimulus selection, or training changes based on these results. Preserve locked `value > 0`, calibrated `value >= 1`, and retention `positive_object_count > 3`.
- The 127 unresolved cell scores led to exclusion of 64 entire candidate columns. They were not encoded as negative judgments; fixed-B lost no contexts. Disclose this once in methods rather than dominate the result figures with it.
- All models use the same frozen human-pretrained 64-dimensional semantic encoder. This experiment replaces context norms, not the entire human-derived semantic pipeline.
- Every condition received 409,600 training episodes. Much larger context inventories receive much less exposure per context. The completed/source-only contrast also changes inventory and applicability. These results do not isolate the causal contribution of salience, weighting, or episode exposure.
- Existing fidelity checks establish released-checkpoint reproduction and reconstructed-human replication consistency. They are not a formal behavioral equivalence test.
- Avoid claims of human applicability accuracy for unmatched feature vocabularies, human-like context magnitudes based only on direction agreement, or validated false-memory prediction.

## 9. Known numerical anchors for QA

These are rounded frozen-report checks, not numbers to hardcode into plots. Read exact values from authoritative artifacts. Similarity and induction below are descriptive unweighted means across nine and five original correlations, respectively.

| Condition | Input-RDM agreement | Similarity mean r | Induction mean r | Context direction |
| --- | ---: | ---: | ---: | ---: |
| Human-norm-trained | 1.000 | .559 | .498 | .947 |
| V2 | .821 | .511 | .554 | .954 |
| V3-B | .090 | .478 | .450 | .849 |
| V3.1-B | .241 | .534 | .425 | .757 |
| V4 fixed-B, calibrated | .762 | .500 | .506 | .961 |
| V4 source-only | .641 | .575 | .513 | .921 |
| V4 completed ensemble | .821 | .520 | .532 | .928 |

For source-only versus completed calibrated ensemble, raw-feature similarity means are .593 versus .524. Thus the similarity difference is already present before neural training. The shared pretrained similarity mean is .539. Do not conclude that lower per-context training exposure alone explains this difference.

If plots disagree with these anchors beyond rounding, investigate protocol, level, subset, and averaging differences. Common-finite-pair sensitivity outputs are intentionally separate and are not expected to reproduce original-coverage means.

## 10. Output package and presentation

Write outputs to a new versioned report directory, for example `ISC-CI_LLM_validation/reports/v4_20261002/figures_final/`, without changing original artifacts.

Required deliverables:

- Four main figures and mandatory supplements, as SVG, PDF, and 300-dpi PNG; export large multipart figures as separate panels when needed.
- A source-data CSV for every panel, retaining condition ID, display label, evaluation level, dataset/task, metric, item ID or subset definition, seed when applicable, sample size/coverage, estimate, and uncertainty definition. Item-level panels retain underlying human and per-seed model observations.
- `captions.md` with the question, scoring definition, subset, uncertainty, takeaway, and relevant limitation for each figure.
- `FINAL_RESULTS_INTERPRETATION.md`, approximately one to two pages, separating established results, unresolved explanations, and implications for the arbitrary-word/false-memory goal. Link figures and the all-condition table.
- `figure_manifest.json` recording source paths and available hashes, selected protocol, condition/order configuration, metric definitions, figure filenames, and any missing/blocked panel inputs. Reuse existing provenance; hashing or scanning the giant judgment file is unnecessary.
- A documented one-command plotting invocation, dependencies, and expected inputs. The command must render reports without model calls or training.

Use consistent condition order and colors across figures, colorblind-accessible colors plus marker/line distinctions, legible labels, and common scales within comparable panels. Use direct condition labels where feasible. Avoid dense bar charts with many overlapping categories. Do not hide poor-performing datasets, missing values, or unfavorable contrasts.

## 11. Acceptance criteria

The task is complete when:

1. All four main figures render and answer the specified questions, with V3-B in the main comparisons and all V3/V3.1 variants in supplements.
2. All 13 conditions appear in the final summary; every trained condition has seeds 0–3. Released references remain separately labeled.
3. Every main plotted number traces to an authoritative saved artifact or a documented lightweight transformation, with stable joins, no duplicate observations, and explicit missingness.
4. Plot summaries reproduce the appropriate frozen numerical anchors within rounding; any changed subset is clearly labeled and does not replace the original measure silently.
5. Context direction and context magnitude are both visible; preference rates are not labeled accuracy; observed humans are not confused with human-trained models.
6. Sparse raw-feature coverage and the shared human-pretrained encoder are disclosed. Intervals carry the correct resampling definition.
7. Readable source-data tables, captions, interpretation, provenance, and the replay command accompany the images. Rendered SVG/PDF/PNG outputs receive visual QA for clipping, unreadable labels, and misleading axes.
8. Original inputs, predictions, checkpoints, scoring, and thresholds remain unchanged. No new judgments, training, checkpoint inference, or large evaluation occurs.

Implement this as a finishing/reporting task. When inputs are missing, document the precise gap rather than expanding the experiment. The desired outcome is an interpretable account of the completed evidence, not another round of runs.
