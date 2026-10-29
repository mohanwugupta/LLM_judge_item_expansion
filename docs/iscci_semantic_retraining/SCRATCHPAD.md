# ISC Semantic Retraining Scratchpad

## 2026-10-28: Initialization / M0 RED preparation

Goal: implement PRD-isc-creation.md as a separate, gated extension. No existing
judgments, matrices, checkpoints, predictions, or figure outputs may change.

Inspected: validation README, RANDOM_FEATURE_CONTROL.md, DECISIONS.md,
V4_20261002_RUN.md, FINAL_FIGURES.md; run_validation.py; dataio.py; modeling.py;
training.py; existing evaluation and human-behavior interfaces. The original ISC
checkout is present at upstream/IntegratedSemanticsControl (not the adjacent
ISC-CI repository). Its revision is 4ecc76d4817fb927d2ad79dfbe1ff44ea72affc0.
Read its models.py and data.py. No downloaded code or pretrained weights are
needed to initialize a new semantic model.

Known issues to resolve: Appendix A taxonomy and original task-ID ordering;
authoritative human feature mapping; frozen classifier identity; classification
human review; a training-only resource/convergence pilot. Full training is blocked
until these gates pass. No behavioral outcomes will select these settings.

M0 test command (from project root):
`/Users/mg9965/.pyenv/versions/3.11.8/bin/python3 -m pytest -q ISC-CI_LLM_validation/tests/test_semantic_inventory.py`

Next: run the new inventory tests before implementing the module; record RED,
then implement strict ordered binary loading and immutable source inventory.

M0 outcome: RED collection failed as intended with ModuleNotFoundError for
semantic_retraining. Implemented inventory.py. Same command GREEN: 3 passed
(6.70 s). `git check-ignore docs/iscci_semantic_retraining/SCRATCHPAD.md`
returned 1 with no output: this log is not ignored. No existing inputs changed.

## 2026-10-28: Taxonomy source investigation and approved deviation

Visually inspected Appendix A, Table A1, PDF pages 111-113 of the original ISC
manuscript hosted by Princeton. The prose claims 33 semantic classes, but there
are 32 named rows. Released data use all 33 numeric contexts. The public source
contains no task-name dictionary. Original Wu-Barsalou (2009) uses 37 subtypes,
not this modified inventory. Downloaded the original Leuven archive linked from
Gert Storms' KU Leuven resources page. Inspected all 58 XLS workbooks / 590 sheets:
no taxonomy-header matches or feature-type dictionary found. Parsing uses xlrd
2.0.2 installed only in /private/tmp/iscci-taxonomy-audit-deps.

Investigator decision: **Prepare explicit 32-class reconstruction**. This changes
the primary task vocabulary from the PRD's unresolved 33 to the 32 documented
classes. Do not claim original numeric ID correspondence or exact released
350-object encoder reproduction. A future verified 33-class run must use a new
protocol/output namespace. All conditions, including human/random_human, use
the same reconstructed vocabulary. Human assignments are reconstructed, not
ground-truth annotations. This decision precedes any new behavioral evaluation.

Next: M1 strict taxonomy validation and versioned 32-class specification; then
classification infrastructure. No real feature classifications or training yet.

M1 RED: `python3 -m pytest -q ISC-CI_LLM_validation/tests/test_task_taxonomy.py`
failed collection with missing semantic_retraining.taxonomy. Implemented strict
taxonomy.py and the investigator-approved task_taxonomy_32_reconstructed.json.
M1 + M0 GREEN: same interpreter as M0, `-m pytest -q
ISC-CI_LLM_validation/tests/test_task_taxonomy.py
ISC-CI_LLM_validation/tests/test_semantic_inventory.py`: 5 passed (6.57 s).
New stable IDs are Table A1 row order, explicitly not original context IDs.
Descriptions/neighbor guidance are reconstruction metadata. No extra class added.

## M2: Metadata classification infrastructure

RED: `python3 -m pytest -q ISC-CI_LLM_validation/tests/test_feature_classification.py`
failed on missing semantic_retraining.classify_features. Implemented exact-text,
batched transactional SQLite cache, fixed classifier configuration, full response
validation, uncertainty and human-review freeze gates, and strict retained-ID
description joins. GREEN: 3 passed (9.55 s), including all real retained IDs in
all 14 inventories. No LLM requests performed. Actual classifications, separate
ambiguity adjudication, and stratified human review remain pending; this is an
infrastructure gate, not a claim that M2's scientific deliverable is complete.
M3 onward may be fixture-tested but cannot train real models before M2 is ready.

Additional audit finding: data.load_leuven parses output tokens with `[:-1]`,
but the terminator is attached to the last ID (e.g. `351;`). This discards one
true output in every semantic row. 1,532 / 6,007 rows become all-zero. The input
file itself preselects nonempty object-task pairs; adding 1,050 auxiliary rows
gives the paper's 7,057 examples. Requested an explicit decision about preserving
the PRD target formula versus reproducing this parsing bug. Fixture tests use
the stated PRD formula; no real data adapter is authorized for training yet.

## 2026-10-28: Investigator approval of target correction / M3

Investigator: "Well let's correct it. It's likely an actual bug from the original
programmer." Decision: use complete semantic targets and positive object-task
pairs for the new 32-class experiment. Do not reproduce the final-ID deletion.
Preserve the released checkout and all completed experiments, and clearly label
the new adapter as corrected rather than byte-equivalent to the released parser.
This approval does not establish how every historical checkpoint was trained.

Expanded regression tests before implementing training_adapter.py: attached and
separate semicolon; singleton positives; malformed text; all 6,007 real semantic
patterns; task-irrelevant zero targets; exact feature slices; immutable matrix
snapshot; invalid assignments; inactive tasks and objects without examples.

RED command: `/Users/mg9965/.pyenv/versions/3.11.8/bin/python3 -m pytest -q
ISC-CI_LLM_validation/tests/test_isc_training_targets.py`.
Outcome: intended ModuleNotFoundError for semantic_retraining.training_adapter.
Implemented corrected legacy-ID parser plus matrix-native TaskDataset. No input
matrix or legacy source file is rewritten. Targets retain all positive features
of the selected task and explicitly suppress all other feature classes.

M3 GREEN: same command, **8 passed (1.29 s)**. All 6,007 corrected released rows
retain their final feature; all 1,532 singleton-positive rows remain positive.
New matrix-native targets follow the complete target formula, not legacy token
deletion. No real training was started.

## M4: Architecture fidelity infrastructure

RED: `/Users/mg9965/.pyenv/versions/3.11.8/bin/python3 -m pytest -q
ISC-CI_LLM_validation/tests/test_isc_architecture.py` failed with the intended
missing semantic_retraining.modeling module. Tests execute the ISCModel class AST
from the unedited local upstream file, avoiding unrelated imports and training
side effects. Both bias modes must have identical initialized state dictionaries
and bitwise-identical logits and activations. Tests additionally require a
task-independent 64D sigmoid hub, 16D task and 128D dependent layers, full-vector
BCE gradients, and ordinary PyTorch train/eval modes.

Implemented modeling.py without reading pretrained weights. Default no-hidden-
bias mode matches the published hidden architecture and released no-bias model;
the direct output bias remains trainable, initialized at -2, as in executable
upstream. The alternate biased mode is supported only for parity testing, not
silently mixed into an experiment. Optimizer/training orchestration will remain
separate rather than shadow nn.Module.train as the old class does.

M4 GREEN: same interpreter, `-m pytest -q
ISC-CI_LLM_validation/tests/test_isc_architecture.py
ISC-CI_LLM_validation/tests/test_isc_training_targets.py`: **12 passed (2.22 s)**.
Both bias variants match the unedited reference bitwise at initialization and
forward evaluation. These are architecture checks, not a completed semantic
learning or behavioral-reproduction claim.

Refactor: ran `python3 -m black ISC-CI_LLM_validation/semantic_retraining` and the
five new test modules. Added AUDIT.md and archived the original PRD alongside
this log. No original source was patched. Original upstream `git status
--porcelain` returns no changes. Rehashed both completed figure manifests:
**236 distinct frozen input files match their recorded SHA-256 values**.

`git diff --check` over the entire worktree encountered the pre-existing Git LFS
clean filter trying to write .git/lfs/tmp, which this sandbox disallows. Scoped
`git diff --check -- ISC-CI_LLM_validation/README.md` passes. No LFS shard files
were changed or repaired. `git check-ignore` for the three documentation files
and training_adapter.py returns 1 with no output: none is excluded.

Regression in progress: `python3 -m pytest -q ISC-CI_LLM_validation/tests` using
the same verified interpreter. Full semantic training remains blocked on actual
classifications/review and a separately locked training-only pilot schedule.

Regression complete: `/Users/mg9965/.pyenv/versions/3.11.8/bin/python3 -m pytest
-q ISC-CI_LLM_validation/tests`: **74 passed in 22.67 s**. This includes every
existing validation test plus inventory, taxonomy, classification-cache, corrected
target, real released-pattern, and architecture-parity tests. No tests skipped.

Next concrete action for the remaining PRD: freeze the classifier model/revision
and execution endpoint, classify the real descriptions, adjudicate ambiguity,
and obtain stratified human review before freezing mappings. Then implement and
test training-only convergence/resource diagnostics. M5-M12 scientific execution,
full semantic training, crossing jobs, and new behavioral results are not complete.

## M5: Semantic Learning and Exact Resume (2026-10-28)

New tests first: `python3 -m pytest -q
ISC-CI_LLM_validation/tests/test_isc_training.py` (same verified interpreter as
above). RED: missing semantic_retraining.train_isc. Implemented random-only
initialization, full-vector BCE/Adam, explicit convergence versus fixed budgets,
maximum-error diagnostics, negative baseline, exposure, and mid-epoch resume of
optimizer, RNG and minibatch cursor.

Initial synthetic fixture at upstream constructor LR .05 failed the unchanged
max-error <.1 criterion: error ~.5000, with task cues collapsed. Training-only
diagnostic on this four-pattern fixture: LR .01 reached .08155 in 92 epochs;
.005 reached .09904 in 124; .001 reached .09957 in 354. The fixture test now
explicitly uses LR .01, retaining all success assertions and the 800-epoch cap.
This is not a selected real-data training schedule, nor a behavioral-data fit.
The public constructor's .05 default is retained in TrainingConfig for audit
transparency; real settings still require a human-reference training-only pilot.

M5 GREEN: focused training tests: 3 passed (2.24 s). Interrupted mid-epoch
continuation matches uninterrupted weights, optimizer updates, and exposure.
Investigator selected the same Qwen 72B model used to generate the feature spaces
for classification. Existing cluster scripts identify Qwen2.5-72B-Instruct; record
its actual local model revision rather than guessing a Hugging Face commit.

## M6: Exact Large-Output Objective

RED: `python3 -m pytest -q ISC-CI_LLM_validation/tests/test_large_output_equivalence.py`
failed importing missing backward_bce and estimate_memory. Implemented output
streaming with summed BCE normalized by the full batch-by-feature count; all
features remain in both the objective and evaluation. No negative sampling,
class weighting, or task-only loss. Tests compare dense loss and every parameter
gradient in float64 for nondivisible chunks, and assert bounded target requests.

M6 GREEN: dense/chunked equivalence plus training regressions: 7 passed (2.12 s).
No real ensemble training executed. Resource estimates explicitly identify their
lower-bound nature; measured device benchmarks remain necessary before full jobs.

## M7: Checkpoints and Sigmoid Extraction

RED: `python3 -m pytest -q ISC-CI_LLM_validation/tests/test_semantic_representations.py`
failed importing missing extract_representations. Implemented atomic checkpoint
and progress writes, run locking, complete/incomplete manifests, source/taxonomy/
classification/config/code/version signatures, exact-state resume, and ordered
sigmoid activation export. Nonconverged convergence-mode runs cannot publish
transfer representations. Reuse validates every artifact hash and recomputes
activations from the checkpoint, without reading old pretrained weights.

M7 GREEN: 2 passed (2.19 s), including corruption, changed-source rejection,
explicit resume, and no embeddings for failed convergence.

## M8/M9: Frozen Transfer and Original CICO Training

RED: `python3 -m pytest -q ISC-CI_LLM_validation/tests/test_embedding_transfer.py`
failed importing missing transfer_embeddings. Implemented exact float32 transfer,
strict identity/order checks, original zero padding row, and reuse of the existing
train_model and save_checkpoint functions with unchanged production settings.
Two-epoch diagnostics are explicitly incomplete scientifically and cannot become
400-epoch models. Other weights initialize freshly from each paired seed.
Interrupted downstream seeds restart the existing deterministic learning loop;
semantic-stage optimizer/cursor resume remains exact. Output guards refuse old
workspace artifact directories before writing. Both stages use new namespaces.

GREEN with semantic extraction and original training protocol regressions:
8 passed (4.84 s). Tests check exact transferred bits, order failures, gradient
freezing, source labels, and existing sampler/spawn equivalence. No real model
training performed. All new fits so far use tiny synthetic fixtures.

## M2 Completion Infrastructure: Qwen, Adjudication, Human Review

RED regression command: `python3 -m pytest -q
ISC-CI_LLM_validation/tests/test_feature_classification.py -k
'reviews_cover or separate_adjudication'`: 2 failures, as intended (unstratified
review accepted; separate instruction unsupported). Added strict per-source,
nonempty-class review coverage and an independently logged ambiguity prompt.

New workflow tests first failed collection on missing classification_workflow.
Implemented full local weight-shard/config fingerprinting, deterministic Qwen
transport, raw request/response audit, incremental exact-description caches,
separate adjudication, append-only review templates, and immutable per-condition
freezes. Human/random share assignments. Reviewing/freezing human first does not
require classifying all 87k ensemble features. Later conditions append without
changing the human freeze or invalidating its per-condition source hash.

One sampling fixture initially omitted the required uncertainty flag and failed
with KeyError. Corrected the fixture to supply complete classification metadata;
no review assertion or uncertainty requirement was weakened. Combined workflow,
classification, and crossing tests: **10 passed (8.41 s)**.

Additional RED: incremental freeze test failed on the missing task-coverage CSV.
Added all 32 class rows, positive associations, valid object-task counts, sparse
and inactive flags, reviewed agreement, original uncertainty, object coverage,
and feature coverage. Freeze snapshots retain classification evidence and human
reviews, not just their hashes. Combined workflow/report tests: **5 passed
(19.25 s)**. Real Qwen labels and human reviews remain pending.

## M10: Registry and Readonly Planning

RED: test_semantic_crosses.py failed importing run_crosses. Implemented the full
56-semantic / 784-context registry and explicit phased subsets: 14 diagonals,
64 configurations through controls, 132 remaining LLM off-diagonals. Dry-run
includes source IDs, seeds, output locations, config hash, memory estimates, and
blocking gates without creating output directories. No phase auto-promotes.
The investigator's 32-class and corrected-target approvals are encoded in the
separate protocol/configuration. The prespecified pilot LLM source is V3.1-B,
chosen by design correspondence, not new behavioral performance.

## M11: Original Human Evaluation and Reporting

RED: test_semantic_evaluation.py failed importing evaluate_crosses. Implemented
original nine-domain similarity items and CV, common-defined RDM coverage,
within/between-category agreement, all class-dependent versus independent
geometry, cross-seed stability, and feature-space transfer. New context evaluation
calls the existing run_predictions, summarize_behavior, paired contrasts, and
original graph/scorer fidelity without changing their scientific code.

GREEN: **4 passed (2.96 s)**, including scorer identity, missingness, and paired-
seed factorial decomposition. Stage 1 outputs remain separate from full ISC-CI
human outcomes. Reuse the original 2000 crossed item/seed draws, common LCA and
original human reliability. Main source/context effects and interactions are
descriptive within paired seeds; controlled source-axis contrasts use the frozen
paired bootstrap, not an independence assumption across 196 cells.

Reporting RED: missing report_results module. Added all-196 condition and all-840
model tables, explicit planned/partial/complete states, three evaluation levels,
saved-data-only figures and hash manifests, and fixed-cohort geometry/behavior
associations. Combined report/evaluation/CLI checks: **8 passed (20.21 s)**.
These tests do not claim actual human trial predictions have been generated.

## Execution and Cluster Infrastructure

CLI RED: missing cli module. Added stage, source, condition, paired seed, dry-run,
resume, worker, device benchmark, output, classification, and review arguments.
Synthetic end-to-end smoke stays outside real model directories. CLI tests:
**3 passed (3.05 s)**. Missing scientific gates fail before training writes.

Cluster script RED: test_semantic_cluster.py failed because the script was absent.
Added run_semantic_classification.sh, matching the existing module/conda/cache/
vLLM launch pattern and 4 x 80-GB GPU, 8 CPU, 32-GB RAM request. It defaults to an
isolated eight-description Qwen smoke; non-smoke defaults to human labels, not
the full inventory. Prepare/preflight and actual-weight fingerprinting precede
server startup. Uses SQLite resume, health checks, and server cleanup. Cluster
static/syntax plus CLI tests: **4 passed (2.96 s)**. Nothing submitted to Slurm.

## M12: Preservation and Strict Gates

RED: test_semantic_retraining_regression.py failed importing gates. Implemented
frozen-history/hash verification, portable saved-path resolution, strict current-
code/inventory/taxonomy gates, and reviewed per-matrix assignment checks. Tests
reject corrupted inputs, stale/failed gates, and old artifact output paths.
Combined preservation/registry GREEN: **5 passed (1.23 s)**.

Next: finish integration verification, run the full regression suite and bounded
synthetic benchmark, write reproduction instructions, and audit remaining gaps.
No full Qwen classification, real semantic pilot, real model retraining, or new
human behavioral evaluation has been performed during implementation.

## 2026-10-28: Integration Hardening / Final RED-to-GREEN Checks

Commands below use the M0 interpreter and `-m pytest -q`; test paths are under
`ISC-CI_LLM_validation/tests/`. Each regression was written before its fix.

- `test_embedding_transfer.py -k relabelled`: RED rejected the unsupported
  expected_identity argument. Added strict semantic/context/seed identity checks;
  GREEN: 3 passed (1.89 s).
- `test_semantic_retraining_regression.py -k plotting_only`: RED unsupported
  code_hashes scope. Scoped immutable training dependencies separately from the
  complete preflight code signature; combined GREEN: 9 passed (3.26 s).
- `test_semantic_evaluation.py -k modified_behavioral`: RED did not reject
  changed behavioral settings. Enforced equality to frozen original evaluation;
  GREEN: 5 passed (2.45 s), then integration 6 passed (2.71 s). Finite-only means
  avoid empty-cell warnings without inventing outcomes.
- `test_semantic_reporting.py -k saved_behavior`: RED missing matched-pipeline
  contrast figures. Added saved-data-only comparison intervals and control axes;
  GREEN: 2 passed (19.61 s). Interval endpoints are plotted directly, not clipped
  to means. Synthetic PNG/PDF fixture was visually inspected: labels/intervals
  fit; this is not a scientific-results figure.
- `test_semantic_cluster.py -k cpu_stage`: RED absent CPU launcher. Added explicit
  one-stage CPU job with condition/seed/config forwarding and no auto-promotion;
  combined GREEN: 4 passed (4.74 s).
- `test_semantic_crosses.py -k canonical`: RED absent default chunk_size in the
  locked config. Canonicalized TrainingConfig before signatures; combined
  crossing/CLI GREEN: 7 passed (2.87 s).
- `test_semantic_cli.py -k entire_requested`: RED allowed workers to start when
  another requested mapping was missing. Validate the entire cohort first.
- `test_semantic_crosses.py -k context_memory`: RED estimated 103809 parameters
  instead of actual 101761 for human CICO. Separate exact downstream graph and
  sampler memory estimates; GREEN: 4 passed (1.42 s).
- `test_semantic_cli.py -k unconverged`: RED returned process status 0 instead of
  2 for unsuccessful convergence. Aggregate stage success now requires complete
  scientific runs (or explicit diagnostic completion), so Slurm sees failure.
  Final complete-suite GREEN below includes this regression.

Canonical CPU sigmoid extraction prevents a GPU/CPU one-ULP restore mismatch.
Every context evaluation validates the expected source pair/seed. Full phase-5
evaluation covers all 196 pairs, not only the last 132 newly trained pairs.
Unbalanced/nonfinite factorial cells report not_estimable. Output tables include
all 196 pairs and 840 planned models; missing runs remain explicit. Synthetic
smokes use code-versioned directories, preserving earlier toy checkpoints.

## 2026-10-28: Local Verification and Scientific Handoff

No Qwen API calls, real semantic training, real CICO training, or Slurm submissions
were performed. Real task mappings still require investigator review.

Executed from project root with the M0 interpreter, MPLCONFIGDIR under
`/private/tmp`, and OMP/MKL/OpenBLAS thread counts set to 1:

1. `ISC-CI_LLM_validation/run_semantic_retraining.py --stage prepare`: all
   fourteen frozen inventories resolved; 87,679 unique semantic descriptions.
2. `ISC-CI_LLM_validation/run_semantic_retraining.py --stage preflight`: final
   **115 passed (25.47 s)**, no warnings/stderr; **316 frozen files unchanged**.
   Full-code gate signature:
   `f8490530c9a5412fe01c21306cb2c361a8c86990f4f11f24fc3a41c301c0d574`.
   Historical fidelity manifest SHA:
   `bd1a3d078106cc73944b7bbba42771b31721cbe88352e92fe9648f38e1b810be`.
3. `ISC-CI_LLM_validation/run_semantic_retraining.py --stage benchmark --device cpu`:
   passed finite full-BCE updates at 88, 385 and 87,315 outputs. Largest update
   took 0.225 s with process high-water RSS 711,802,880 bytes (about 0.663 GiB).
   One bounded synthetic update is not proof of real-model convergence or a
   forecast of full runtime; cold/warm timing differences are not scaling claims.
4. `ISC-CI_LLM_validation/run_semantic_retraining.py --stage smoke`: bounded
   five-object synthetic ISC-to-CICO integration only; result recorded below.
5. `ISC-CI_LLM_validation/run_semantic_retraining.py --stage report --phase 3`:
   saved planned tables for all 196 conditions, zero scientific models, no fake
   figures, experiment_complete false. Original reports are not modified.

`bash -n` passed for both new Slurm launchers. Scoped `git diff --check` passed.
`git check-ignore` for scratchpad, workflow, CLI and classification launcher
returned 1/no output: these files are not silently ignored. They are new worktree
files, not committed automatically. Existing 32 V4 shard-resolution modifications
and the unrelated recovery script were left untouched.

Documentation: AUDIT.md now distinguishes implemented infrastructure from pending
scientific gates. WORKFLOW.md records exact cluster commands, required ignored
inputs, review/freeze protocol, pilot/schedule lock, staged training, resume
semantics, three evaluation levels, and historical CICO missing-value convention.

Next action: cluster Qwen eight-description smoke, then human-vocabulary
classification and stratified review. Freeze reviewed human/random mappings,
run the training-only human pilot, document and lock its full schedule before
expanding. No human behavioral results may select labels, filters, or settings.

Final versioned synthetic smoke: passed, 91 semantic epochs, maximum error
0.0802, positive recall 1.0, followed by the unchanged two-epoch downstream
diagnostic. Saved gate explicitly records real_models_trained = 0. A post-smoke
readonly preservation command initially used the nonexistent config key
output_root; corrected to output_directory (no pipeline code change).
Corrected post-smoke check passed: all 316 historical files still match the
preflight snapshot.

## 2026-10-29: Missing Cluster Matrices / Input Transfer

Investigator's isc_task_classes_1640624 job stopped at prepare because the cluster
lacked validation_v4_20261002/data_matrices/human.csv. Confirmed all saved matrices
exist locally; the 13-condition matrix directory is about 100 MiB. They are
Git-ignored generated inputs and are not delivered by git pull. No model or
judgment changes are appropriate, and old training must not be rerun to fix this.

RED command: M0 interpreter `-m pytest -q
ISC-CI_LLM_validation/tests/test_semantic_inventory.py -k missing_inventory`.
Two failures: empty checkout reported only the first missing file; partial
checkout attempted to parse the existing file before reporting other absences.
Added initial all-condition existence validation with explicit transfer guidance.
Regression fixtures preserve every existing byte and assert no files are created.
WORKFLOW.md now gives Mac-to-cluster rsync for the frozen matrices and the saved
random control, preserving paths without replacing existing files. It also
requires login-node prepare/preflight before another GPU allocation and notes
that additional ignored historical inputs may need transfer as well.

No network transfers, cluster submissions, regenerated matrices, or model calls
were performed during this repair. GREEN/full regression results follow below.

GREEN: inventory tests 5 passed (4.04 s). Full CLI preflight: **117 passed
(17.72 s)**, no stderr, 316 historical files unchanged. Signature:
`63398b16c4997892b9a22fcbce01482e233caa97009bd231a3f6fc74986e8db7`.
`git diff --check` passed. Local benchmark and versioned synthetic smoke were
refreshed against this code revision; these never authorize real training.

## 2026-10-29: Investigator-Requested Git LFS Publication

Investigator asked to push the missing matrices to the repository instead of
requiring a separate transfer. Publish exactly the 14 saved matrices with scoped
Git LFS attributes, force-adding these ignored files only. No force push or
history rewrite is needed; GitHub main was confirmed at local parent 96416d5.
Other generated artifacts remain ignored. No matrices were regenerated.

RED regression: M0 interpreter `-m pytest -q
ISC-CI_LLM_validation/tests/test_semantic_inventory.py -k configured_for_git_lfs`
first exposed a test transport error (Git's -z stdin also requires NUL-delimited
input); corrected the fixture, then confirmed the intended failure: filter
unspecified rather than lfs for the frozen matrices. Added the two narrow
.gitattributes patterns. Updated missing-input instructions to git pull plus
filtered git lfs pull, retaining rsync as an optional fallback.

GREEN full CLI preflight: **118 passed (19.76 s)**, no stderr, **316 historical
files unchanged**. Signature:
`17fdc1965517dd97ed66d370d8ffe9e6101bb038d06473d11ddec5964b7294fb`.
Inspected staged human and largest ensemble files: both are three-line LFS
pointers, not CSV payloads in ordinary Git objects. Largest payload is 52,915,127
bytes. Staged diff whitespace check passed. Commit/push contains this publication
and the immediately preceding missing-input diagnostic/tests/documentation fix.
