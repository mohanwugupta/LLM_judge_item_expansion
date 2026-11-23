# Stage 1 Dataset Correction: Audit and Proposed Plan

Date: 2026-11-22. Status: **plan only; no retraining authorized or executed**.
Only documentation and read-only audit tables were added. Production code,
configuration, labels, checkpoints, and previous results remain unchanged.

## 1. Finding and Responsibility

The methodological concern is correct. The 293-object/385-feature adapted human
condition is not the original ISC Stage 1 baseline. It is the filtered ISC-CI
feature matrix fed into an ISC trainer. The same narrowed object universe and
downstream retention rules also entered the adapted LLM conditions. Using the
original constructor and optimizer did not repair this experimental confound.

There is an important qualification: the separate `original_human` reference
already uses the full 350-object Stage 1 dataset and original tasks/auxiliaries.
Its ten existing fits are not the mistaken 293-object baseline and need not be
discarded. What was wrong was treating the adapted cohort as an adequately
controlled semantic-feature replacement study after verifying engine parity.
Earlier audits identified the inventory difference; the workflow nevertheless
continued without resolving it. That was an experimental-design failure, not
evidence that the data were previously unavailable. I should have stopped the
source-only interpretation rather than treating the difference as a caveat.

The discrepancy is a strong plausible explanation for weak adapted-human hub
recovery, but it is not yet a uniquely demonstrated cause. Task grouping,
auxiliaries, target parsing and sample exposure also differ. The existing results
cannot identify the contribution of object/feature coverage alone.

## 2. Source Verification

Authoritative local git trees, with no tracked source edits:

- [ISC at 4ecc76d](https://github.com/tylergiallanza/IntegratedSemanticsControl/tree/4ecc76d4817fb927d2ad79dfbe1ff44ea72affc0).
- [ISC-CI at 29e1c67](https://github.com/tylergiallanza/IntegratedSemanticsControlContextInference/tree/29e1c672215b8cbe3dc6bfbfd1d414fbf8915d14).

Live GitHub page/raw retrieval was attempted but returned cache misses. These
findings verify the pinned repository revisions used for the study, not current
GitHub HEAD. Local untracked Python caches are not source changes. Paths below
are relative to `ISC-CI_LLM_validation/`; `ISC/` and `CI/` abbreviate its
`upstream/IntegratedSemanticsControl/` and
`upstream/IntegratedSemanticsControlContextInference/` directories.

| Evidence | Actual implementation |
| --- | --- |
| `ISC/data.py:11-35,113-175,248-261` | 350 object inputs, 2541 semantic outputs, 33 semantic task IDs; original loader adds two size, three category and 350 naming outputs, across three auxiliary tasks |
| `ISC/data/leuven_rum_nozeroes.ex` | Numeric semantic feature IDs 0-2540, task suffixes C1-C33, 6007 semantic examples after excluding `viger` |
| `ISC/data/leuven_size.csv` | Canonical ordered 350 objects; size/category metadata available for all 350 |
| `ISC/models.py:129-166,236-255,293-325` | Original initialization/architecture/Adam/loss/minibatch loop; loader defaults to 30 training epochs |
| `CI/data.py:35-47,72` | Threshold raw feature frequencies, then retain features occurring on sufficiently many objects |
| `CI/models.py:9-13` | Defaults of 293 objects and 385 feature-context outputs belong to ISC-CI, not ISC semantic pretraining |
| Released `CI/models/1and2shot_isc-seed0.pt` metadata | Actual `input_d=294` includes padding; `task_output_d=385`; both rater and object cutoffs are 3 |
| `isc_training.py:219-225` | Existing `original_human` branch directly calls original `make_training_data`; asserts 7057 x 2896 targets |
| `isc_training.py:147-154,226-243` | Adapted branch explicitly requires 293-row matrices, including `human`, and applies Qwen assignments |
| `semantic_retraining/native_isc.py:91-132` | Adapted branch has 32 semantic plus three auxiliary tasks; size/category only for 266 verified objects, naming for 293 |

Independent data inspection confirms the raw downstream matrix is 293 x 1995.
Applying `(frequency > 3)` and then `(positive_objects > 3)` gives exactly the
293 x 385 adapted-human matrix, cell for cell. This is a confirmed cross-stage
dataset substitution, not a conjecture from dimension names.

The correct full original inventory is 350 x 2541 semantic features, **2896 total
outputs**, 36 tasks, and 7057 examples: 6007 semantic plus 1050 auxiliary.

## 3. Correct Baselines and Comparison Design

### Literal release reference

Use the existing full-data `original_human` fits as the original Stage 1
reference after checking their frozen hashes. Keep original numeric assignments,
object ordering and all auxiliary supervision; never Qwen-reclassify this
reference. Existing core RSA is .36748/.50796 versus released .37665/.52402,
within the previously frozen .05 tolerance. That gate covers core RSA, not every
behavioral finding. No new fit or evaluation was run to obtain these numbers.

Proposed explicit name: `human_stage1_350_original_tasks_literal`. This is a
provenance alias in a new registry, not a rename or overwrite of old files.
The old `human` condition becomes `human_stage2_reduced_293_385` in new report
labels only; retain its original machine identifiers in historical manifests.
It is a restricted-data diagnostic, not the primary Stage 1 baseline.

### Matched scientific human/LLM comparison

The goal is to hold the **350 object identities, task system, auxiliary labels,
target policy, model recipe and evaluation** fixed while changing semantic data.
Do not replace the human baseline alone and then compare it with the old
293-object LLM fits as if this accomplished that goal.

There is a pre-existing target-policy issue that must also be matched:

- The released parser at `ISC/data.py:142` discards the final positive feature ID
  on every semantic example. It loses 6007 associations and creates 1532 empty
  semantic target rows.
- Full parsing retains 37683 semantic positives versus 31676 in literal loading.
  The existing `original_dataset(corrected=True)` helper preserves all 7057
  source examples and original task assignments, with identical auxiliaries,
  while restoring those IDs; it yields no empty semantic rows.
- The old LLM adaptation already uses complete targets, as previously approved.
  Comparing that with a literal-loader human reference changes two things.

Recommendation: retain literal loading for release replication; propose a new
`human_stage1_350_original_tasks_fulltargets` comparator using the full original
data and approved complete-target correction. Apply that same target policy to
every new LLM condition. Keep the original trainer otherwise unchanged. This is
an explicitly repaired, matched scientific comparison, not exact target-level
replay of the release. Do not deliberately discard arbitrary LLM positives to
imitate a serialization bug. Finalize this protocol before any retraining.

For the cleanest applicability-only contrast, use all **2541 original feature
IDs and original feature-task memberships**, with LLM rather than human
applicability values. This requires a verified ID-to-feature-text dictionary.
The released `.ex` has IDs, and `object_features_in_context.pkl` contains numeric
arrays, not that dictionary. The downstream 1995 text descriptions must not be
assumed to be a complete or positionally aligned replacement for the 2541 IDs.
That mapping remains a prerequisite for a full original-inventory V2 condition.

Discovery conditions legitimately have different numbers of semantic outputs.
Report that as part of the feature-source intervention, rather than forcing
2541 outputs or claiming a fixed-capacity comparison. Their original 350 naming
and five size/category outputs must remain identical to the human comparator.

## 4. What Additional Data Are Needed

### Object coverage is not simply 350 minus 293

The existing independently verified one-to-one crosswalk covers 266 original
objects. Therefore **84 original rows currently lack verified reusable data**:
30 fruits, 28 vegetables, and 26 other names/identities requiring review or new
collection. There are also 27 unverified current names, including category
conflicts. Some aliases may be resolvable without new calls; do not automatically
reuse all 27 or treat the current 293 as a strict subset of the original 350.

Review translations, numbered homonyms and category conflicts against source
metadata before freezing the manifest. A convenient spelling normalization or
nearest saved embedding is not evidence of concept identity. The previously
identified released embedding reuse includes questionable mappings; it is
provenance for release replay, not permission to merge distinct objects here.

The audit CSVs list every unresolved original/current identity. At present, 84
times the inventory size is a **conservative row-extension workload**, before
new discovery, additional feature columns and legacy missing-cell repairs:

| Existing inventory | Features | Cells on 84 unverified/uncovered original rows |
| --- | ---: | ---: |
| V2 retained | 382 | 32088 |
| V2 all existing candidates | 1995 | 167580 |
| V3 A / B / C retained | 94 / 115 / 121 | 7896 / 9660 / 10164 |
| V3.1 A / B / C retained | 88 / 175 / 128 | 7392 / 14700 / 10752 |
| V4 fixed-B locked or calibrated | 175 | 14700 per matrix; same underlying new judgments can serve both frozen resolvers |
| V4 ensemble source-only retained | 2356 | 197904 potential applicability cells, not a recommendation to complete a source-only condition |

These are cells, not API calls, judge votes or a final cluster budget. They are
not all additive because some feature inventories overlap. Full completed V4
ensembles remain deferred, as previously requested.

### Condition-specific corrections

**V2:** recover the full Stage 1 feature-text/task mapping first. Reuse archived
judgments only for verified matching objects AND identical mapped feature texts
under compatible prompt/protocol semantics. Existing V2 covers the downstream
1995 candidates, not demonstrably all 2541 original features. Missing feature
columns require judgments for all 350 objects, not just new rows. Rebuild the
retention inventory from Stage 1 policy, not the old 382 retained columns.

The actual existing V2 resolutions contain 584532 unique cells of 584535 expected,
with valid final scores in [0,4]. Three are missing: vulture / `intelligent`,
piranha / `looks tough`, and alligator / `is many women's nightmare`. Do not
claim the old export is complete merely from its manifest. Some valid scores
are fractional; this is not a parse error. Keep raw adjudication scores and
explicitly freeze the already approved binary conversion, rather than applying
the human rater-count cutoff to judge scores or assuming missing means zero.

**V3/V3.1:** generate features for all original objects not covered by verified
reuse, with the frozen prompt/model/sampling procedure and reviewed display
names. Reuse existing responses only with compatible identity/protocol evidence.
Consolidate the full 350-object discovery data. Source-only generation is not
equivalent to atomic judging; completing every existing feature against every
object would create a different condition. Existing unfiltered consolidated
inventories have 12271/11584/12221 columns for V3 A/B/C and 13212/8552/11354 for
V3.1 A/B/C, far more than the retained training matrices. Decide Stage 1
consolidation/retention rules before collecting or training; do not transfer
the Stage 2 `>3`/`>3` filters by default. Preserve count-valued evidence.

**V4 fixed-B:** extending the frozen 175 features to the original object universe
is useful but is explicitly a restricted-inventory comparison. Generate/recover
new-object discovery data before freezing a broader Stage 1 bank. For every new
candidate column, perform applicability judging on all 350 objects. For old
candidates, reuse only verified compatible cells and judge the uncovered rows.
Reuse the established cascade/resolvers only with recorded versions; unknown or
failed cells are never implicit negatives. Locked/calibrated decision rules must
not be selected again using downstream human behavior.

**Source-only and random controls:** keep source-only membership distinct from
completion. A corrected random control must shuffle a specified full Stage 1
semantic inventory across the same 350 objects while preserving prevalence,
tasks and all original auxiliary labels; the old 385-feature random control
does not match the full-human baseline. Neither control is regenerated now.

### Task labels are a separate missing-data problem

Original human task IDs already exist for all 2541 features; no Qwen run is
needed for that baseline. The current archive supplies only 32-class Qwen
assignments. Even among the downstream V2 1995 descriptions, only 423 have an
old 32-class annotation; 1572 do not. Completeness reported previously was for
retained features, not the original full inventory or a 33-class system.

The paper's missing named class was previously identified as `Associated object`,
but its original numeric ID must not be guessed. A 33-class taxonomy and verified
class-to-original-task correspondence are prerequisites for a matched novel-
feature pipeline. Human original assignments stay unchanged. All descriptions
affected by moving from 32 to 33 classes need fresh review/classification; merely
adding an unused input unit does not repair old memberships. Newly discovered
features also need classifications. Existing labels remain historical evidence.

If exact grouping for novel features cannot be verified, call that an explicit
reconstructed-task factor, not a feature-source-only manipulation. A separately
labeled full-human/Qwen-grouping bridge could help isolate grouping error, but
it is an optional future control requiring approval, not part of this audit.
The prior approval for a 32-class historical run does not authorize silently
rewriting it or declaring 32 equivalent to 33.

## 5. Frozen Stage 1 Procedure and Proposed Implementation Boundary

Preserve direct imports of the pinned original model and `train` method:
350 object inputs, hub64, task16, dependent128, sigmoid paths, biases enabled,
original uniform initialization, Adam .05, mean BCEWithLogits over the complete
output vector, NumPy-shuffled minibatches of 64, 30 epochs. With 33 semantic
plus three auxiliary tasks the context input has 36 units. Auxiliary output
indices follow the semantic block: two size, three category, then 350 naming.
Use original metadata for **all** 350 objects, including original treatment of
insects as `other`; do not substitute a more intuitive category rule.

Preserve one example per source-defined positive object/task combination and
original auxiliary examples for all objects; do not add a full 350 x 36 grid,
rebalance task sampling, introduce masked-only loss, normalize embeddings or
change early stopping. Check source-pair parity on original human data. For new
feature data, use the same declared positive-pair construction rule. Positive
pair counts, output counts, class frequencies and updates per epoch can still
change as a consequence of semantic data; disclose these, rather than silently
equalizing updates or claiming perfectly equal effective loss weighting.

Proposed implementation, **not executed here**:

1. New Stage 1 inventory module/registry independent of the downstream matrix
   registry. Canonical 350-object order comes only from original metadata.
2. Separate literal/full-target policy names and fingerprinted task assignments.
   Reuse original parsing and the existing corrected helper where appropriate;
   do not duplicate neural-network code.
3. New 350-object adapter appends the same original 355 auxiliary outputs to
   every condition; replaces the 293-only shape check and partial-metadata
   adapter only in a new protocol, leaving v1 frozen.
4. Coverage and cluster work manifests with stable object/feature IDs, exact
   pending cells, source provenance, classifier coverage and resume signatures.
   Inspect absent/LFS-only data before claiming reuse. No jobs submitted now.
5. New output namespace, e.g. `artifacts/isc_stage1_350_<version>/`, with references
   to immutable old checkpoints. No in-place config edits, moves or overwrites.

Changing a training matrix is not a need to rewrite the original trainer.
The central repair is the dataset boundary and matched supervision contract.

## 6. Validation Gates Before Any Retraining

Use TDD for the implementation phase; these are planned checks, **not newly
passed gates**:

- Exactly 350 canonical objects, unique IDs, original row order; no downstream
  filtering or 293-only shape assumption anywhere in Stage 1 preparation.
- Original human: 2541 semantic outputs, 355 auxiliary outputs, 36 tasks,
  original per-feature tasks and 7057 ordered examples. Literal tensors must
  match unmodified `make_training_data` exactly.
- Corrected human: independently parse all positive IDs; restore exactly 6007
  associations, preserve source pairs and all auxiliary targets. Compare this
  policy with LLM targets, not an asymmetrically buggy human loader.
- Every condition has all 1050 original auxiliary examples with identical
  labels, naming order and output block. No missing metadata in the full domain.
- Complete object-feature/task-label coverage with explicit unknown states;
  every reused cell has identity, feature-text and protocol evidence. No NaN,
  failure or absent row becomes a negative by default.
- No Stage 2 rater/object retention rule used accidentally. Prespecified source-
  specific binary conversion and consolidation rules are frozen without human
  behavioral tuning. Any exclusion has a reason and changes inventory hashes.
- Direct original-model initialization, logits, loss, gradients and Adam update
  parity; existing parity tests reused. Compare input/target tensors independently,
  not only adapter output against another helper sharing the same logic.
- Record semantic/auxiliary examples, positive counts, task histograms, actual
  updates and train-fit measures per seed. Do not require convergence-based
  changes to recover an expected outcome at a fixed original budget.
- Use existing ten full-human references for replication calibration. Four
  prespecified matched seeds per new scientific condition; shared seed labels do
  not make hidden coordinates comparable when inventories differ. Compare RDMs,
  not unitwise coordinate averages, and retain seed/item uncertainty.
- Reuse frozen original core RSA stimuli/statistics/tolerance; retain unresolved
  full-paper discrepancies. No tolerance or schedule selected from new results.
- Only after Stage 1 passes: explicitly map/select the Stage 2 object subset and
  append padding for ISC-CI. Do not retrain Stage 1 on that subset. Resolve the
  remaining identity ambiguities before interpreting downstream comparisons.
- Human behavioral data remain external tests, not inputs to feature judgments,
  taxonomy repair, filtering, threshold selection or training-setting selection.
- Verify hashes of old checkpoints, matrices, protocols and reports before/after.

## 7. Audit Outputs and Current Decision

`stage1_scope_audit_20261122_v1/` contains source SHA-256 fingerprints, exact
counts, all 84 unresolved original rows, the 27 unverified current identities,
and fixed-inventory workload estimates. Counts were recomputed from source
metadata, `.ex` records, matrices, the classification archive and raw V2 CSV.
No model prediction, gradient, optimization or behavioral simulation was run.

**Decision:** stop treating the old adapted cohort as a source-only Stage 1
comparison. Promote the already correct full-human reference in the proposed
new design; resolve common target policy, full object coverage, task grouping
and feature-inventory scope before implementation and any new training. The
existing results remain valid records of restricted-data models, not evidence
that replacing human semantic norms alone caused the observed geometry gap.
