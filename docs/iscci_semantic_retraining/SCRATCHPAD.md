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

## 2026-10-30: Complete Cluster Input Closure

Investigator retrieved all 14 matrices successfully. Prepare then failed on
v3_A_cluster_labels.csv; preflight independently failed on the original figure
manifest. The preceding matrix-only publication was insufficient for a fresh
checkout. Audited the complete dependency closure: frozen history and its source
files/checkpoints, all retained feature-description joins, original human tasks
and references, semantic graph/target fixtures, and downstream reliability tables.

Added a deterministic Git LFS input bundle and relative-path/checksum manifest.
run_semantic_inputs.py --install uses only the standard library, validates all
payloads before writing, skips matching files, accepts only exact LFS pointers,
and stops on conflicting scientific files. It does not regenerate data or weaken
fidelity gates. The classification launcher calls it before prepare/preflight.
All scientific outputs remain separate from these existing reference inputs.

TDD RED: `python -m pytest -q ISC-CI_LLM_validation/tests/test_semantic_inputs.py`
failed collection on missing semantic_inputs. Implemented payload/hash validation,
idempotence, conflict protection, traversal/link/duplicate-member rejection and
CLI installation. Initial GREEN: 13 passed (0.17 s).

First real bundle build correctly stopped on a byte-level category-file mismatch:
the historical adjacent-repository file uses CRLF; the upstream checkout copy
uses LF. They must not be substituted under an exact SHA gate. Package the exact
historical bytes in inputs/reference_sources without changing either original.
Additional fixture tests cover reference aliases and deterministic output.

Portable path regression first failed because an original-machine absolute path
was preferred even when the current checkout had its own file. Another RED
showed no support for the byte-exact packaged external reference. Prefer current
checkout paths and packaged references before legacy fallbacks. Combined initial
GREEN: 20 passed (1.60 s). Validation and isolated-checkout results follow below.

Final frozen package: **345 files**, 378,994,665 compressed bytes (about 361 MiB),
SHA-256 `a29d404e23f48fe5922946719a02c702ebd7db374f790f667ef7f6476d1e6948`.
Manifest includes all 14 matrices, six prompt-specific label tables, candidate
descriptions, frozen figure/history sources and prior checkpoints, original
upstream task data/code/notebooks, and the exact external category reference.
It excludes raw shard judgments, Qwen weights, and new experiment results.

Verified bootstrap with `python -S` (no scientific site packages). The existing
workspace needed only the new byte-exact reference copy; no pre-existing file was
replaced. Repeated install is idempotent. Original-workspace preflight:
**137 passed (35.81 s)**; 316 historical files unchanged, no stderr.

Fresh-checkout reproduction under `/private/tmp/iscci-clean-inputs-fvfHk5`:
created local clone metadata without checkout, exported staged tracked files with
GIT_LFS_SKIP_SMUDGE=1, and materialized only the published bundle. No ignored
original-workspace artifacts were copied separately. Standard-library installer
restored 331 files and verified all 345. Second invocation installed 0.
Prepare resolved all 14 conditions / 87,679 descriptions. **Preflight passed all
137 tests (33.14 s)** with 316 preserved files, no stderr. Both environments have
the same gate signature:
`66faef6b785e65d7aeb18d8337e787681fea052cd90382783b3821c97567050b`.
Historical fidelity manifest SHA remains
`bd1a3d078106cc73944b7bbba42771b31721cbe88352e92fe9648f38e1b810be`.

Additional tests cover all missing/changed paths, stdlib-only CLI, deterministic
reference-alias packaging, complete real input closure, unsafe paths (including
root), symlink destinations, corrupt archive/hash/payload, duplicate members and
non-overwrite on conflicts. `bash -n` and whitespace checks passed. The staged
archive is a three-line LFS pointer. No Qwen calls, new judgments, scientific
model training, altered evaluation settings, or Slurm submissions occurred.

Next cluster action: pull the fix and only the frozen-input LFS bundle, run
run_semantic_inputs.py --install, then prepare/preflight on the login node before
submitting the Qwen smoke job. Do not delete differing cluster results: the
installer intentionally stops for a comparison instead of overwriting them.

## 2026-10-30: Investigator Defers Full V4 Categorization

Investigator requested postponing the full V4 task-categorization workload while
running other conditions. Interpreted full V4 as the two 80k-plus completed
ensemble inventories, not the smaller fixed-B or source-only controls. Explicit
default config defers v4_ensemble_locked_v2 and v4_ensemble_calibrated. All 14
frozen matrices, original checkpoints and scientific results remain unchanged.
This is a compute-driven scheduling decision, not behavioral model selection.

Readonly exact-text count: enabled 12 sources require 2,751 descriptions versus
87,679 in the full design, a 96.86% reduction in descriptions, not a runtime
benchmark. Human-only pilot default remains unchanged. CLASSIFY_CONDITION=all
now means enabled sources; INCLUDE_DEFERRED=1 / --include-deferred is a deliberate
override. Freeze, training, Phase-5 cumulative evaluation and reporting use the
same cohort. Never wait for ensemble classifications to freeze the active set.

RED command: M0 interpreter `-m pytest -q
ISC-CI_LLM_validation/tests/test_semantic_crosses.py
ISC-CI_LLM_validation/tests/test_semantic_cli.py -k
'defer or default_cohort or default_classification or invalid_deferral'`.
Ten failures confirmed missing cohort selection, unguarded deferred sources,
classify/freeze requesting all labels and phase-5 evaluation reintroducing all
784 downstream models. Added shared validated source selection and explicit
override; targeted GREEN: 10 passed (2.56 s).

Reporting RED: unrun report listed only planned; partial geometry crashed on an
absent source's iloc[0]. Reports now retain deferred rows and missing geometry,
with active-cohort completeness separate from full-design completeness. Full
registry remains 56 semantic / 784 context models; 8 / 208 respectively are
deferred. Active design: 48 semantic / 576 context, 144 pairs; staged control
phase 54 pairs and remaining phase 90. Prespecified 14-source factorial statistics
are not silently redefined. Combined planner/CLI/report GREEN: 22 passed (5.59 s).

Updated the old default dry-plan expectation from 56 to 48 to match the approved
schedule; the original full-design dry-run assertion is retained under explicit
include_deferred=True. Added guards against removing human/random/pilot baselines,
unknown IDs or duplicate deferrals. Classification workflow/prompts/cache schema
are unchanged, preserving existing exact-description cache provenance. No model
requests, real training or cluster submissions were performed. Full regression
and actual plan/report checks follow below.

Final full preflight GREEN: **148 passed (33.21 s)**, no stderr; all 316 frozen
historical files unchanged. Gate signature:
`0e36b0b347043b751fe21552610b047ce01ec104c0419939253b9a0303a51c6f`.
Both Slurm launchers pass bash -n; git diff --check passes. Actual CLI dry-run
selects 48 semantic jobs with 12 active sources and both large ensembles listed
as deferred; no scientific artifacts are written by planning. Planned-only
report generation retains 196 pairs and shows the 144-pair active cohort,
without claiming any new model completion. Changes are local, not committed or
pushed in this turn; no production job has been launched.

## 2026-10-30: Expose Cluster Preflight Failure Details

Investigator reported only the generic Tests or preservation failed message.
This does not identify a failed test or prove a changed scientific input. The
cluster's saved stdout/stderr/returncode are still required to diagnose the real
failure; do not assume a dependency or numerical-tolerance issue without them.

Added failing regressions before changes for three outcomes: pytest failure,
before/after preservation mismatch, and preservation verification raising after
tests. RED command: M0 interpreter `-m pytest -q
ISC-CI_LLM_validation/tests/test_semantic_retraining_regression.py -k preflight_failure`.
All three failed because the raised error omitted the actual record path; tests
also require captured failure details and an explicit failed saved gate.

gates.preflight now records separate tests_passed/preservation_passed flags,
changed paths and before/after hashes, a preservation error if any, and Python
identity. It prints actual pytest failure output to stderr and cites the full
record path. Post-test preservation exceptions persist a failed gate rather than
leaving an older successful record. No acceptance criterion or tolerance changed.
WORKFLOW.md supplies an older-version-compatible standard-library command to
print the failed cluster record without dumping all source hashes.

These diagnostics and the prior compute-driven V4 deferral remain local/unpushed.
No claim that the user's underlying cluster failure is fixed is justified until
the saved test output is inspected. Verification results follow below.

GREEN preservation/diagnostic regressions: 9 passed (1.40 s). Full actual local
preflight: **151 passed (33.46 s)**, no stderr, preservation_passed true with
zero changed files and all 316 frozen sources preserved. Signature:
`702d9ccdf41e69c8f04bba4101f63d91c5e0ec5212b0b82dff0f61e2659f6e36`.
The local success does not reproduce or diagnose the cluster's failure. Next:
obtain the cluster's saved stdout/stderr/returncode before choosing any fix.

## 2026-10-30: Cluster Failure Identified as Missing PyArrow

Investigator supplied saved pytest output: returncode 2, collection aborted in
test_v4_downstream_reports.py -> audit_v4_feature_profiles.py -> import
pyarrow.parquet, raising ModuleNotFoundError: No module named pyarrow. This is
a test-collection/environment failure, not evidence of a scientific tolerance
failure. Old records' None preservation fields do not certify a new check.

The traceback uses `/usr/lib64/python3.9/`; earlier cluster model logs used Python
3.13 under PromptControlText. Verified launcher explicitly activates that Conda
environment. First activate it and inspect sys.executable/version, then install
PyArrow there if missing. Use python -m pip, not system pip or sudo. Do not
upgrade Torch/vLLM or change scientific acceptance criteria to fix this import.

Inspected pyproject.toml: only pytest settings, no dependency declaration. Updated
WORKFLOW dependency list and explicit login-node activation/import/install
commands. Deferred V4 task classification does not remove legacy Parquet audit
tests from full preflight. No frozen scientific helper was edited or test skipped.
Local pip metadata confirms PyArrow 26.0.0 / Python 3.11.8; checked Apache Arrow's
official installation documentation. This is an environment-only remedy requiring
no repository update on the cluster. No cluster package installation or job
submission can be claimed from this Mac; investigator must rerun the import and
preflight in the activated cluster environment. Existing code verification remains
151 local tests / 316 preserved inputs; this turn changes documentation only.

## 2026-10-30: Two Cluster Fidelity Tests Hit pandas API Removal

Read investigator attachment 64970fac-2bbe-4b84-8c3e-597ee56e46cf/Pasted text.txt.
Conda interpreter is now correct. Cluster preflight collected tests and reported
135 passed / 2 failed: test_item_probe_matches_unedited_original and
test_fidelity_detects_model_graph_drift. Both raise AttributeError at
pd.errors.SettingWithCopyWarning before the original/reconstructed comparison.
This is a pandas compatibility failure, not observed prediction disagreement.

Verified unchanged behavior_fidelity.py references the warning at line 56 and
that the passing local runtime is Python 3.11.8 / pandas 2.3.2 with that attribute.
Official pandas 3 release notes document removal; pandas 2.3.2 publishes Linux
CPython 3.13 wheels. Do not alter this hash-frozen scientific helper or the saved
manifest to handle the environment mismatch. Pin pandas to 2.3.2 in the activated
cluster environment and verify the actual imported path/version before preflight.

Added a narrow reusable constraints file and documented immediate pip command,
including the risk that visible user-site packages can differ from the Conda
prefix. This is not a full environment lock. No Torch/vLLM upgrade, scientific
source edit, skipped test or changed tolerance is authorized by this fix. The
cluster remedy uses a direct package pin and needs no repository update. Local
recheck of the two reported failing tests follows below; cluster verification
remains the investigator's next action.

Local targeted recheck with pandas 2.3.2: **2 passed (5.21 s)**. The drift test
still rejects deliberately altered predictions; the unchanged reference probe
passes with the original 1e-5 tolerance. Whitespace check passed. No cluster
package change or successful cluster preflight has been claimed.

## 2026-10-30: Investigator Confirms Successful Cluster Preflight

Investigator installed pandas 2.3.2 and verified its imported path under
`/home/mg9965/.conda/envs/PromptControlText/lib/python3.13/site-packages/`.
The subsequently supplied cluster record reports **137 passed (119.73 s)**,
pytest returncode 0, empty stderr, passed true and **316 preserved sources**.
Published-code signature:
`66faef6b785e65d7aeb18d8337e787681fea052cd90382783b3821c97567050b`.
Historical fidelity manifest SHA remains
`bd1a3d078106cc73944b7bbba42771b31721cbe88352e92fe9648f38e1b810be`.
This confirms the cluster environment remedy without changing frozen scientific
code, tolerance or inputs. It is not a completed classification or training run.

Next: eight-description Qwen smoke via SMOKE_TEST=1. The compute-driven V4
deferral, improved failure diagnostics and compatibility constraint remain
local/unpushed; avoid CLASSIFY_CONDITION=all on the current cluster revision
until the deferral update has been published and retrieved. Human-only smoke
and human-only classification do not request the full V4 ensembles. No Slurm
job was submitted by this agent and no new model calls were made in this turn.

## 2026-10-30: Qwen Smoke Passed / Publish Production Cohort

Investigator reported the cluster eight-description Qwen smoke passed, with
smoke_only true, features 8, and production_labels_modified false. HTTP health
and completion calls returned 200; server/worker shutdown is normal cleanup.
This validates the endpoint, output parsing and isolated smoke transport, not
the full classification/review or scientific training stages.

Investigator now requests the production command. Checked published HEAD:
CLASSIFY_CONDITION=all still selects all 14 there; therefore publish the already
requested/tested deferral changes before recommending that command. Publication
also includes the accumulated actionable preflight diagnostics and pandas
compatibility constraint/documentation. No frozen scientific helper, checkpoint,
feature membership or numeric tolerance is changed.

Production command after a successful pull uses SMOKE_TEST=0,
CLASSIFY_CONDITION=all, INCLUDE_DEFERRED=0: all 12 enabled conditions / 2,751
unique descriptions, not the two large completed V4 ensembles. Four 80-GB GPUs,
8 CPUs, 32-GB RAM and existing job time limit remain unchanged. The launcher
refreshes inputs/preflight before model startup and resumes valid cached batches.
The job only classifies/adjudicates features; stratified human review and freeze,
training-only pilot, and explicit full-schedule lock still precede real training.
No cluster job is submitted by this agent. Current regression confirmation is
recorded below before commit/push.

Publication preflight: **151 passed (24.52 s)**, no stderr, preservation passed
with zero changes across 316 sources. Gate signature remains
`702d9ccdf41e69c8f04bba4101f63d91c5e0ec5212b0b82dff0f61e2659f6e36`.
Both Slurm launchers passed bash -n, and whitespace checks passed. GitHub main
was confirmed at ed51a40 before publication; use a normal fast-forward push.
