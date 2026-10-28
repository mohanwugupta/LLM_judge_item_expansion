# ISC-CI LLM Validation Decision Log

## 2026-08-08: Validate V3.1 as a Prompt-Only Follow-Up

### Decision

Treat V3.1 A, B, and C as three new ISC-CI conditions and compare each with the corresponding
V3 prompt, V2, retrained human models, and released human checkpoints. Hold every
consolidation, training, fixed-context evaluation, and paper-simulation parameter equal to the
locked V3 protocol. Do not retune thresholds after observing V3.1.

### Source and Integrity

The source is
`artifacts/leuven_feature_generation/v3.1/leuven_v3.1_qwen2_5_72b/generated_features_long.csv`.
Its production manifest records Qwen2.5-72B-Instruct, 293 words, prompts A/B/C, 20 simulated
participants per word and prompt, temperature 0.8, paired seeds, 17,580 of 17,580 valid
responses, and zero parse or request errors. Consolidation uses the same strict
`response_count > 3` and `positive_objects > 3` cutoffs as V3.

`configs/v3_1_validation.json` deliberately duplicates every substantive value in
`configs/v3_validation.json`; only the protocol identifier differs. The comparison therefore
tests the V3.1 prompts, not a changed threshold or training procedure.

### Consolidation Review

Prompt variants remain separate. At the primary threshold, five semantic/profile clusters
would be retained and all were reviewed. Four were equivalent and passed. The proposed merge
of `driven on highways` with `driven on roads` was rejected because a highway is a narrower
road type. The consolidation code splits rejected semantic clusters back into lexical groups
before response counts and the training cutoff are calculated. The broader `driven on roads`
feature remains a retained singleton; the highway feature does not meet the task cutoff.

The final V3.1 task counts are A=88, B=175, and C=128. These counts and positive-cell totals
are identical at embedding thresholds 0.80, 0.825, 0.85, 0.875, and 0.90, so the result is not
driven by the selected embedding threshold.

### Reuse and Provenance

`artifacts/validation_v3/` remains unchanged. The V3.1 run copies it to
`artifacts/validation_v3_1/`, preserves its prior manifest as
`base_validation_manifest.json`, and reuses checkpoints only when matrix hash, epoch count,
and training parameters match. It then trains 12 V3.1 checkpoints (A/B/C by seeds 0-3),
evaluates all conditions on the same fixed contexts, and runs the paper simulations only for
models without completion artifacts. The cumulative manifests hash both consolidation stages
and the copied base validation.

### Interpretation Constraint

V3.1 is a prompt iteration on the same model family, not an independent replication. Report
prompt-matched descriptive deltas and four-seed variation, but do not treat the 12 V3.1 runs as
independent evidence about model-family generalization. The same no-applicability-judgment
limitation documented for V3 also applies to V3.1.

## 2026-08-04: Consolidate V3 Feature Generations

### Decision

Treat prompt variants A, B, and C as separate feature-norm conditions. Consolidate
near-equivalent feature phrases within each prompt before constructing the ISC-CI training
matrix. Do not pool responses across prompts.

### Why

V3 recreated the Leuven free feature-generation task. The released Leuven matrix used by
ISC-CI is already consolidated, whereas the V3 artifact stores exact normalized strings and
explicitly states that semantic merging was not performed. Treating every wording variant as
a distinct feature would therefore create an avoidable mismatch with the human pipeline.

### Operational Definition

1. Begin with `generated_features_long.csv`; one valid model response is treated as one
   simulated participant response.
2. Use the supplied normalized feature text as the phrase type.
3. Within each prompt, merge deterministic lexical variants and conservatively merge
   semantic near-duplicates using normalized `all-MiniLM-L6-v2` embeddings plus the
   similarity of their 293-object generation profiles.
4. Require every cross-pair in a proposed multi-phrase cluster to satisfy the semantic and
   profile criteria. This complete-link constraint prevents transitive chaining.
5. Block embedding-only merges when one phrase adds a substantive modifier to the other
   (for example, `has a tail` versus `has a long tail`). Lexically equivalent signatures may
   still merge.
6. Ignore non-substantive frequency scaffolding (`often`, `usually`, `typically`, and similar
   terms) when forming lexical signatures, but preserve conjunctions and substantive
   alternatives. Thus `metal and wood` remains distinct from `metal or wood`, and `made of
   wool` remains distinct from `made of wool or silk`.
7. Do not use the English sentence-embedding model to merge phrases containing CJK text.
   Such phrases retain their exact-string identities and are flagged in the assignments.
8. If one response contains two phrases assigned to the same cluster, count the cluster once
   for that response.
9. Construct object-by-cluster response counts, binarize using the paper's strict
   `response_count > 3` rule, and retain clusters positive for strictly more than three objects.
10. Train separate ISC-CI models for A, B, and C using the released architecture, frozen
   semantic embedding, episode sampler, loss, optimizer, 400 epochs, and seeds 0-3.

### Primary and Sensitivity Specifications

The primary embedding threshold is `0.85`; object-profile cosine must be at least `0.50`.
Embedding thresholds `0.80`, `0.825`, `0.85`, `0.875`, and `0.90` are reported as a
sensitivity analysis with the profile threshold held fixed. The exact parameters live in
`configs/v3_validation.json`.

The automated method is intentionally conservative. It is a reproducible approximation to
the human consolidation applied before release of the Leuven matrix, not a claim that sentence
embedding proximity proves logical equivalence. Every assignment, cluster member, within-
cluster similarity, source hash, and model revision is written to the consolidation artifact.
This allows a future manual or LLM-adjudicated consolidation to replace the current mapping
without changing the downstream training protocol.

All embedding/profile clusters retained by the primary training cutoff were manually reviewed
and recorded in `configs/v3_consolidation_manual_review.csv`. The consolidation command
fails if that reviewed set no longer exactly matches the retained semantic clusters. During
method development, the review caught and rejected a merge of `made of wool` with `made of
wool or silk`; preserving conjunctions and substantive qualifiers prevents that merge in the
locked implementation.

Nearest-neighbor candidates are computed by exact, chunked PyTorch cosine products. An
initial FAISS implementation was rejected after the locally installed native binary crashed
on macOS (`exit 139`); no FAISS-derived assignments were produced or retained.

### Revisit Triggers

- Manual review finds systematic over-merging or under-merging.
- An LLM adjudication run becomes available for candidate phrase pairs.
- The original Leuven consolidation instructions or unconsolidated source become available.
- Validation conclusions change materially across the recorded threshold sensitivity range.

## 2026-08-04: Follow Executed Checkpoint Batch Size

The released checkpoint dictionary records `batch_size=1024`, but every epoch in its retained
metrics has batch indices 0-7 and accuracy increments of `1/128`. Thus the model actually
executed eight batches of 128 examples over each 1,024-episode epoch. Retraining follows the
executed behavior (`batch_size=128`) rather than the inconsistent metadata field. It uses 400
epochs, matching the released checkpoint's `epoch=400`, for 409,600 sampled episodes. The
paper describes this approximately as 500,000 episodes.

## Reconstructed V2 Rule

V2 remains locked to the 385 features retained by the original human Leuven preprocessing.
The finalized adjudicator value is converted directly to binary (`0` is absent; any nonzero
value is present), with no additional rating or object-frequency threshold. Three human-schema
features have no V2 positives and cannot generate support episodes, leaving 382 V2 tasks. This
reproduces the retained prior run log (`human_tasks=385`, `llm_tasks=382`) and the user's
instruction that the adjudicator decision is final.

## Recovery Note

The earlier validation source and reports disappeared from this Git-ignored directory before
the V3 analysis. Reconstruction uses the clean upstream repository snapshots, released model
checkpoint metadata, the retained V2 run log, and surviving compiled ceiling modules. New
runs record source hashes and protocol manifests so this can be detected and audited.

## 2026-08-04: Define Comparability Against a Human Self-Ceiling

Comparability is evaluated on fixed task-label-independent probes rather than each model's
native training accuracy. The evaluation uses all 293 singleton supports, 1,024 sampled
unordered support pairs, and all 293 real query objects for every support context. Context
RDMs use 512 fixed contexts and context-dependent RDMs use 128 fixed contexts. The sampling
seed is `20260804`, and the same probes are used for every model.

Five measures are retained: context RDM Spearman correlation, median context-dependent RDM
Spearman correlation, membership-logit Spearman correlation, binary membership agreement,
and membership-probability mean absolute error. Human self-ceiling values are calculated from
all pairs of the four independently retrained human models. Candidate-to-human values average
all 16 pairs between a candidate condition's four seeds and the four retrained human seeds.
The four released checkpoints are also compared with all four retrained human models as an
independent reconstruction check.

Binary agreement is not interpreted alone because the V3 matrices are sparse. Input-matrix
object RDMs and the continuous/rank output measures are reported to distinguish shared
negative decisions from genuinely similar representational geometry. These choices are
implemented in `iscci_validation/evaluation.py`; exact probes and model outputs are retained
under `artifacts/validation_v3/evaluation/`.

## 2026-08-04: Recreate the Paper Simulations

The analyses in the two released notebooks were moved into a deterministic command-line
runner without changing their substantive calculations. It evaluates every human, V2, V3,
and released checkpoint on the five induction datasets, seven induction phenomena, nine
similarity domains, asymmetric similarity, thematic and context-dependent paired choices,
and the similarity-context LCA simulation.

Necessary execution adaptations are limited to explicitly selecting numeric columns for
pandas 2, mapping behavioral labels through `leuven_singular_to_plural.csv`, and assigning
stable per-model/per-row random seeds to the stochastic LCA. The LCA retains 100 simulations,
500 steps, and a 100-step burn-in. Per-model outputs and completion records are preserved
under `artifacts/validation_v3/paper_simulations/`; the manifest records notebook paths and
adaptations.

Prompt C is the current broad default because it has the strongest overall V3 profile, not
because it wins every benchmark. Prompt B is better for asymmetric similarity, while prompt A
is better for thematic arguments and slightly better for context-dependent non-monotonicity.
This choice must be revisited if additional participants, models, prompts, consolidation
rules, or behavioral benchmarks materially alter that profile.

## 2026-08-04: Audit the V3 Consolidation Rules

A post-validation diagnostic sweep compared exact-only, conservative lexical, current
lexical, stricter, primary, and relaxed embedding/profile rules. It is recorded under
`reports/consolidation_audit/` and does not modify the locked primary matrices or models.

The embedding threshold is not responsible for the small V3 task inventory. With the current
lexical signatures, removing semantic merges leaves A/B/C at 94/115/121 tasks, identical to
the primary counts. Embedding thresholds from 0.80 through 0.90 also leave those task counts
unchanged when the profile threshold remains 0.50. The semantic/profile rule contributes only
two primary positive cells, both in C.

The 0.50 profile guard remains necessary. Relaxing or removing it admits non-equivalent and
sometimes contradictory pairs. The rule requiring revision is automatic lexical merging:
it can erase modality and frequency distinctions or conflate derivational roles. The clearest
retained error pools `can be washed` with `washed frequently`; another cluster pools `musical
instrument` with `used for musical instruments`, although that second error is currently
subthreshold and does not change a positive object.

Therefore, keep the current embedding/profile/complete-link safeguards, but do not treat the
primary lexical consolidation as final. A follow-up should use safe inflectional normalization
plus manual adjudication for modality, frequency, and derivational changes, then retrain those
matrices as a reported sensitivity condition without replacing the original primary results.
## 2026-10-01: External Human-Behavior Stage

Added the separate downstream protocol in `HUMAN_BEHAVIOR_VALIDATION.md` and the
runner `run_human_behavior_validation.py`. Reuse original human observations,
stimuli, mappings, probes, and paper measures. Save every trial prediction before
aggregation. Keep raw-feature, frozen-semantic, and fully trained evaluations
separate. Four seeds and the existing frozen 400-epoch protocol remain unchanged.
Human behavior is an external test, never a feature/threshold/training selection tool.

Default fidelity tolerances were fixed at 1e-5 before running. Original item-probe
parity passes, and the released seed-3 checkpoint reproduces the 16 deterministic
notebook statistics. Reconstructed human models do not numerically match that one
released seed. A distinct source check finds PDF page 73 reports 36/38 context
directions while the saved notebook reports 37/38. The common-seed stochastic probe
matches the PDF. The strict requested gate remains failed, with no silent tolerance
relaxation, seed selection, candidate behavioral evaluation, or new training.

V4 also remains blocked by unfinalized judgments/matrices from the October 1 audit.
No invalid cells were converted to negatives. Optional source-only models are
evaluated when already trained; this stage does not invent a source-only benchmark.
The original synthetic phenomena CSV lacks human responses, and originally
model-selected stimuli limit independence claims; both caveats are explicit.

### User Clarification and Protocol 1.1

The user explicitly selected: "Exact released replication; statistical reconstructed
replication." Preserve the initial strict audit in `artifacts/human_behavior_validation/`.
The current protocol uses exact deterministic released-seed-3 reproduction plus
reconstructed four-seed means inside the 95% released-human-seed prediction envelope
(t, df=3, reference SD * sqrt(1+1/4)). All four released models receive identical probes.
This is statistical replication consistency, not formal equivalence; four reference
seeds limit precision. The rule was specified after initial human-only inspection,
before LLM-condition behavioral evaluation, and must not be called preregistered.

The saved stochastic notebook value and PDF/notebook source discrepancy remain
reported diagnostics, not deterministic numerical gates. No LCA seed or parameters
were changed to achieve a target. Protocol 1.1 passes all 33 required statistical and
deterministic checks. Current outputs use `artifacts/human_behavior_validation_statistical/`.
This authorizes downstream evaluation of available models, but not training from
unfinalized V4 data or changing any upstream feature or training decision.

### Completed Available-Condition Behavior and Derived QA

Protocol 1.1 evaluated human, V2, V3-A/B/C, and V3.1-A/B/C: 32 existing
four-seed models, plus four released reference checkpoints. No new training or
upstream changes were required. All V4 conditions remain missing, not estimated.
The original model graph and scoring agree at every induction/phenomenon query
for the eight human/reference models (16 checks; maximum error 1.91e-6).

Derived QA corrected induction trial errors after heterogeneous task concatenation
and separated historical input-RDM ranks from defined-pair sensitivity. Parent
manifests, original trial-statistic files, raw-probe hashes, and implementation
lineage were retained. No model predictions or CV fits were changed. The main
runner now emits the corrected reports, consolidated trial metrics, and human
reliability in the condition table. Zero-vector raw similarity remains undefined,
with explicit coverage rather than invented scores. These are evaluation/reporting
corrections, not behavior-driven feature, threshold, or training selection.

Exploratory geometry-behavior associations use seven nonhuman CONDITIONS, not 28
pseudo-independent seeds. Permutation p-values are unadjusted, with related
conditions and many outcomes; they do not establish a general geometry-validity
relationship or justify selecting a feature configuration. See the statistical
run's `RESULTS.md`, complete per-task tables, uncertainty, and manifest.

### V4 Finalizer Failure and Authorized Exact-ID Recovery

The finalizer failed because original shard 0 had 47 invalid resolved values despite
complete cell coverage; all 32 original sidecars were incomplete (2,005 failures).
After explanation, the investigator selected recovery of exact-ID formatting cases
from saved responses first, then retrying remaining cells on the cluster.
The recorded conservative policy recovered exactly 1,878 cells locally, leaving
127. Original CSV bytes/failed rows and raw adjudicator responses are preserved;
no semantic scores were guessed, and no LLM calls or training ran locally.
Sidecars remain unchanged pending full revalidation. Details, hashes, and cluster
steps: `reports/v4_exact_id_recovery_20261001/RECOVERY.md`.

The previously unexplained protocol relocation mismatch is now resolved. Replaying
the cluster paths and recomputing their dependent legacy hashes exactly reproduces
the production hash `8d70b3db7fc70fc07d589d97d0297ee851e40c0d637048e9ec0cb710bc66d73f`.
No recorded hash was overwritten. Finalization gains comprehensive manifest
diagnostics and validation-only path replay, without weakened content checks.
Resolution CSVs now use the existing Git LFS convention. A CPU-only recovery
launcher and small portable evidence bundle allow the same repairs on the cluster;
the original four-80-GB-GPU production job resources are unchanged.

### Investigator Amendment: Finalize Without Retrying 127 Cells

The investigator explicitly authorized accepting the remaining 127 failed
adjudications and proceeding without a GPU retry. The finalization launcher now
applies the saved exact-ID recovery inside its CPU job and opts into a cap of
127 recorded missing `adjudicator_failed` scores. Protocol, partition, row
coverage, duplicate, range, and vote checks remain enforced on actual CSVs.
Failures remain in a hashed unresolved-cell audit; no unknown score becomes zero.

Training matrices exclude all 64 candidate columns containing those failures,
leaving 133,017 of the 133,081 frozen candidates before the existing retention
rule. No excluded candidate overlaps the fixed-B 175-feature inventory. The bank,
valid judgments, thresholds, training settings, and human control are unchanged.
Judgment finalization and complete semantic judgments are distinguished explicitly
in manifests (`complete: true`, `judgments_complete: false`, declared exclusions).
The matrix builder validates the failure audit and carries exclusions into its
provenance. This is missing-data handling chosen independently of behavioral
outcomes, not a behavior-driven selection of features. It supersedes the remaining
retry requirement; see `reports/v4_finalize_exclusions_20261001/FINALIZE.md`.

## 2026-10-02: Finalized V4 Training and External Behavior Completed

The downloaded finalization passes the frozen-bank and resolved-file hashes,
exhaustive accounted-for coverage, duplicate/range checks, and explicit failure
audit. Exclude 64 whole incomplete feature columns, none in fixed-B. Keep `>0`,
frozen `>=1`, and strict `>3` context retention unchanged. No behavior feedback.

The investigator approved four ensemble source-only seeds before that condition's
behavioral results. Train 20 new models with the frozen 400-epoch/409,600-episode
procedure and reuse all 32 prior trained checkpoints. Isolated spawned workers,
equivalent integer task sampling, categorical input storage, and duplicate-seed
bootstrap caching change execution only; regression tests establish equivalence.
The original 504 baseline prediction CSVs and baseline statistics/intervals remain
unchanged. Original graph/scorer parity passes for all 20 new models.

The released exact / reconstructed statistical gate passes all 33 required checks.
Common-probe comparisons and original-paper simulations cover 56 models including
released references. External behavior covers 13 four-seed conditions, every
original human dataset, saved individual predictions, CV-only rating mappings,
2,000 crossed item/seed bootstrap draws, human reliability, and all three levels.
No required seeds are missing. See `V4_20261002_RUN.md` and
`reports/v4_20261002/RESULTS.md` for source hashes, commands, and full results.

Completion improves raw Leuven geometry but has mixed behavioral consequences:
fixed-B improves some induction/context measures over V3.1-B; completed ensembles
lose thematic/nonmonotonicity preference fidelity versus source-only and human.
Global RDM recovery does not uniformly predict human behavior across conditions.
This is not behavioral equivalence or an entirely human-data-free semantic model.
Do not tune thresholds, feature decisions, or training from these external results.
The production cascade's fast negatives lack counterfactual full-panel ground
truth, and equal episode budgets give large ensembles only ~5 episodes/context.

## 2026-10-05: Frozen Final-Results Figure Package

Implement `PRD-ISC-CI-Final-Results-Figures.md` as reporting only. No judgments,
training, checkpoint inference, model simulations, new bootstrap intervals, or
giant-cell evaluation. Use authoritative common-LCA-seed human-validation protocol
1.1.0, not historical paper-simulation seed schedules. Preserve all 13 conditions
and seeds 0-3; released references are never added to reconstructed-human replicates.

Main comparisons are the prespecified seven conditions, with V3-B chosen by prompt
matching rather than performance. Raw features, shared pretrained activations,
fully trained predictions, and observed human reliability remain separate.
Common-finite-pair sensitivity intersects stable similarity item IDs across those
seven conditions within each domain and reports point estimates only. Original
coverage statistics and saved intervals remain unchanged. RDM heatmaps use original
Leuven category/file order and preserve undefined zero-vector cosines as missing;
historical imputation and defined-pair sensitivity are displayed separately.

Context direction uses 38 sets; magnitude uses all 41 with the original
three-alternative probability denominator. Asymmetry keeps its distinct 34/50
subsets and logit-versus-human-choice scales. Recomputed CSV logit subtraction is
checked within 1e-6 because float32/export rounding reaches 5e-7, below the frozen
fidelity tolerance; no saved score is changed. The supplied local paper labels
nonmonotonicity Figure 5 and context-specific categorization Figure 6.

All four main figures, mandatory supplements, per-panel source tables, captions,
interpretation, source hashes, and missing-input records are generated under
`reports/v4_20261002/figures_final/`. See `FINAL_FIGURES.md` for the one-command
renderer and independent PDF/SVG/PNG QA. Input hashes are verified unchanged.
Figures do not establish model equivalence, arbitrary-word generalization, or
false-memory/DRM validity. No external human outcome revises scientific settings.

## 2026-10-28: Human-Sized Random-Feature Negative Control

The investigator explicitly approved a human-sized random control rather than a
fixed-B- or ensemble-sized control. Shuffle each of the 385 human feature columns
independently over the same 293 objects with fixed RNG seed 20261028 (PCG64).
Preserve every feature's prevalence and all 8,056 positive cells; do not preserve
object degrees or select a realization using human behavior. Use one fixed random
matrix for four training seeds, not four different matrices mislabeled as pure
training-seed uncertainty.

Reuse the exact frozen encoder, architecture, 400-epoch training procedure, original
human tasks, common stochastic evaluation seed, CV scoring, representation probes,
and 2,000-draw crossed item/seed bootstrap. New original-graph checks must pass.
Add a paired human-minus-random contrast, without changing the earlier contrasts.
Never retrain or overwrite the 13 completed conditions. Combined CSVs preserve
their original byte prefixes. Redraw the same figure families into the versioned
October 28 report; keep the original common-finite-pair mask and separate raw,
shared-pretrained, and fully trained levels.

Interpret this as a test of meaningful contexts conditional on the human-pretrained
encoder, not an encoder-removal ablation or a numerical decomposition of its causal
contribution. Four-seed intervals do not include randomization-realization variability.
Details and replay are in `RANDOM_FEATURE_CONTROL.md`; all new scientific outputs
are under `artifacts/random_feature_control_20261028/`.
