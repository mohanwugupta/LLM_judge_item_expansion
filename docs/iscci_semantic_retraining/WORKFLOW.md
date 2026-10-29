# Independent ISC Semantic Retraining

## Scope and Status

This is a separate, versioned extension, not a replacement of completed results.
The implementation uses the investigator-approved **32-class reconstruction** and
**corrected complete targets**. The original 33-class numeric dictionary remains
unresolved. See AUDIT.md and SCRATCHPAD.md for source evidence and approvals.

The code supports classification, training, exact resume, transfer, all 196
crossings, original human evaluation, and saved-result reporting. Real Qwen
classification and human review have not yet run. The full semantic schedule is
intentionally **unlocked** until a training-only human pilot and resource checks
justify it. Code completion is not completion of the scientific experiment.

Run commands from `LLM_judge_item_expansion`. The verified local interpreter is
`/Users/mg9965/.pyenv/versions/3.11.8/bin/python3`. Commands below use `python`
for the existing scientific/cluster environment.

## Required Inputs

Preserve/materialize these existing inputs on the execution machine:

- Thirteen retained matrices under
  `ISC-CI_LLM_validation/artifacts/validation_v4_20261002/data_matrices/`.
- The saved `random_human.csv` under
  `artifacts/random_feature_control_20261028/validation/data_matrices/` inside the
  validation folder. Do not regenerate it.
- V3/V3.1 `*_cluster_labels.csv` files in their existing consolidation directories.
- Materialized `artifacts/v4/discovery/candidate_bank.csv`, for descriptions only.
- Original-paper data/code under validation `upstream/`. ISC revision:
  `4ecc76d4817fb927d2ad79dfbe1ff44ea72affc0`; ISC-CI revision:
  `29e1c672215b8cbe3dc6bfbfd1d414fbf8915d14`.
- The frozen behavior manifest, existing model checkpoints, and files referenced
  by the original and random-control figure manifests, for fidelity/preservation.

Many inputs are Git-ignored. **Pushing code does not upload them.** Preflight
reports missing files before loading Qwen or training. The 10-GB raw V4 judgment
payload is not needed for this extension. Old checkpoints are historical fidelity
inputs only; no old semantic weights enter new semantic learning.

Python dependencies are the existing NumPy, pandas, PyTorch, SciPy, scikit-learn,
matplotlib, Pillow, and pytest stack. Cluster classification additionally uses the
already established vLLM/Transformers environment. No new API account is needed.

## Initial Checks

```bash
python ISC-CI_LLM_validation/run_semantic_retraining.py --stage plan --phase 3 --dry-run
python ISC-CI_LLM_validation/run_semantic_retraining.py --stage prepare
python ISC-CI_LLM_validation/run_semantic_retraining.py --stage preflight
python ISC-CI_LLM_validation/run_semantic_retraining.py --stage smoke
```

`prepare` verifies all 14 matrices and writes ordered inventory/description
metadata, never memberships. `preflight` runs the test suite and rechecks frozen
inputs, scientific implementation, human data, and the original fidelity gate.
`smoke` learns a tiny synthetic task, extracts sigmoid activations, transfers them,
and runs a two-epoch CICO diagnostic. It is not a real human model and does not
count toward the 56/784 scientific runs.

Default output:
`ISC-CI_LLM_validation/artifacts/semantic_retraining_20261028_32_corrected/`.
Default reports:
`ISC-CI_LLM_validation/reports/semantic_retraining_20261028_32_corrected/`.
`--output-dir` and `--report-dir` can select new namespaces. Writes inside old
experiment directories are rejected. Dry-run does not create output directories.

## Qwen Classification on the Cluster

Use the same Qwen2.5-72B-Instruct checkpoint used for the feature spaces. The
launcher copies the existing environment/cache/server conventions and requests
**four 80-GB GPUs, 8 CPUs, and 32 GB RAM**. It does not submit subsequent jobs.

First run an isolated eight-description API smoke:

```bash
sbatch --export=ALL,SMOKE_TEST=1 ISC-CI_LLM_validation/run_semantic_classification.sh
```

The script performs prepare/preflight and hashes the actual local model shards
before vLLM startup. Fingerprinting 72B weights requires reading those files and
can take several minutes. This identifies the actual checkpoint rather than
guessing a revision from its name. The smoke cache is separate from production.

Then classify just the human vocabulary for the initial pilot:

```bash
sbatch --export=ALL,SMOKE_TEST=0,CLASSIFY_CONDITION=human ISC-CI_LLM_validation/run_semantic_classification.sh
```

Overrides include `PROJECT_DIR`, `MODEL_PATH`, `CONDA_ENV`, `VLLM_PORT`, and
`OUTPUT_DIR`. The default model path is the same
`/scratch/gpfs/JORDANAT/mg9965/models/Qwen--Qwen2.5-72B-Instruct` used previously.
Initial and ambiguity passes use fixed temperature 0 and seed 0, separate
instructions/caches, and complete request/response records. Only feature
descriptions and taxonomy definitions are sent; no memberships or human behavior.

Exact descriptions share metadata labels, not object assignments. Batches commit
transactionally to SQLite; resubmission continues completed work. A changed
model fingerprint/protocol cannot silently reuse the cache. Malformed responses
stop without committing their batch, while prior valid batches remain reusable.

## Human Review and Freezing

Review `classification/human_review.csv` within the new output directory.
Fill `task_id` (0-31) and `reviewer` for the selected descriptions. The template
samples nonempty task/source strata and includes unresolved ambiguities. It is
not sufficient to approve one unrelated label: every nonempty class in a source
needs reviewed coverage, and every still-uncertain assignment needs resolution.
Review uses descriptions and taxonomy, not applicability patterns or behavior.

```bash
python ISC-CI_LLM_validation/run_semantic_retraining.py --stage freeze --condition human
```

This freezes human and random_human together with identical task assignments.
The random matrix remains the single saved realization. Frozen files include
classification evidence, human review, source/model hashes, and class/object/
feature coverage audits. No sparse class or feature is discarded. Inactive class
IDs remain stable and generate no empty-positive training examples.

Later classify V3.1-B for the matched pilot, or all remaining descriptions:

```bash
sbatch --export=ALL,SMOKE_TEST=0,CLASSIFY_CONDITION=v3_1_B ISC-CI_LLM_validation/run_semantic_classification.sh
# Or, after the initial pilot is satisfactory:
sbatch --export=ALL,SMOKE_TEST=0,CLASSIFY_CONDITION=all ISC-CI_LLM_validation/run_semantic_classification.sh
```

The review template appends new descriptions without overwriting earlier review.
Freeze another condition with `--stage freeze --condition v3_1_B`, or the entire
inventory with `--stage freeze` once all conditions are ready. Existing frozen
assignments cannot be revised in place. An unresolved label blocks pretraining; no convenient fallback class
or membership/feature filtering is applied.

## Human Pilot and Schedule Lock

```bash
python ISC-CI_LLM_validation/run_semantic_retraining.py --stage pilot --phase 1 --resume
python ISC-CI_LLM_validation/run_semantic_retraining.py --stage context --phase 1 --resume
python ISC-CI_LLM_validation/run_semantic_retraining.py --stage benchmark --device cpu
```

The pilot uses only reviewed human features, no pretrained weights, and no
size/category/naming supervision. Provisional pilot parameters are in
`configs/semantic_retraining.json`: these are not an approved full schedule.
Inspect native loss, positive recall, worst absolute error, negative baseline,
updates, and exposure. The convergence criterion is maximum error <0.1 over the
complete output vector. The code never substitutes CICO's 400 epochs for ISC.

The benchmark performs bounded synthetic updates at 88, 385, and 87,315 outputs.
It uses the same full BCE objective, streamed in chunks; equivalence of loss and
all gradients is tested. It records throughput and memory. Process RSS is a
high-water mark, not incremental model allocation. A GPU semantic schedule needs
a benchmark on that device (`--device cuda`), not a CPU benchmark substituted for
GPU evidence. The CPU launcher below assumes a CPU training configuration.

Before full training, set `semantic.full` to explicit TrainingConfig parameters,
`semantic.schedule_rationale` to the training-only evidence, and
`semantic.schedule_locked` to true. Choose limits/stopping/failure policy before
inspecting new human behavioral outcomes. Do not automatically copy the synthetic
fixture's learning rate or assume the provisional pilot budget is adequate.

The first full run freezes an experiment lock. Later changes require a new
experiment namespace. Code, taxonomy, mappings, matrix, configuration, and
completion criteria are checked on resume. A resource-limited or nonconverged
convergence-mode run cannot publish transfer activations. Fixed-budget comparisons
can be specified explicitly, with convergence status still reported; the human
reference must learn its own environment before downstream expansion.

Semantic training resumes optimizer, RNG, minibatch order/cursor, and exposure.
Representations are canonical CPU sigmoid activations, not weights or preactivations.
Downstream incomplete seeds restart the original deterministic learning loop;
completed seeds are validated and skipped. No completed run is silently replaced.
Plotting-only edits do not invalidate training weights, but substantive training
changes do. Preflight must be refreshed when its code signature changes.

## Phased Production

Every stage is explicit. No command below automatically runs its successor.
Four seeds are paired across stages and conditions; they are not four random
feature matrices. V3.1-B is the prespecified LLM pilot by design correspondence.

```bash
# Phase 2: human, fixed random_human, and V3.1-B, four seeds each.
python ISC-CI_LLM_validation/run_semantic_retraining.py --stage semantic --phase 2 --workers 4 --resume
python ISC-CI_LLM_validation/run_semantic_retraining.py --stage semantic-evaluate --phase 2
python ISC-CI_LLM_validation/run_semantic_retraining.py --stage context --phase 2 --workers 4 --resume
python ISC-CI_LLM_validation/run_semantic_retraining.py --stage evaluate --phase 2

# Phase 3: all 14 semantic sources and all 14 matched diagonal pairs.
python ISC-CI_LLM_validation/run_semantic_retraining.py --stage semantic --phase 3 --workers 4 --resume
python ISC-CI_LLM_validation/run_semantic_retraining.py --stage semantic-evaluate --phase 3
python ISC-CI_LLM_validation/run_semantic_retraining.py --stage context --phase 3 --workers 4 --resume
python ISC-CI_LLM_validation/run_semantic_retraining.py --stage evaluate --phase 3

# Phase 4: diagonals plus either-axis human/random controls, 64 unique pairs.
python ISC-CI_LLM_validation/run_semantic_retraining.py --stage context --phase 4 --workers 4 --resume
python ISC-CI_LLM_validation/run_semantic_retraining.py --stage evaluate --phase 4

# Phase 5: train the remaining 132 pairs; evaluate the complete 196-pair cohort.
python ISC-CI_LLM_validation/run_semantic_retraining.py --stage context --phase 5 --workers 4 --resume
python ISC-CI_LLM_validation/run_semantic_retraining.py --stage evaluate --phase 5
python ISC-CI_LLM_validation/run_semantic_retraining.py --stage report --phase 5
```

`--condition`, `--semantic-source`, `--context-source`, and `--seed` allow bounded
subsets. The entire requested source cohort is checked before training workers
start. Stage 1 evaluation must exist for each semantic source before full CICO
training. CICO retains 400 epochs, 1024 episodes/epoch, batch 128, Adam .001,
task-loss weight .5, one/two supports, zero padding, and frozen transferred hubs.
Other CICO weights initialize fresh for each (S,C,seed).

A separate CPU launcher requests 8 CPUs / 32 GB RAM and defaults to synthetic
smoke. For example:

```bash
sbatch --export=ALL,PIPELINE_STAGE=smoke ISC-CI_LLM_validation/run_semantic_stage.sh
sbatch --export=ALL,PIPELINE_STAGE=pilot,PHASE=1 ISC-CI_LLM_validation/run_semantic_stage.sh
sbatch --export=ALL,PIPELINE_STAGE=semantic,PHASE=2,WORKERS=4 ISC-CI_LLM_validation/run_semantic_stage.sh
```

Optional `CONDITION`, `SEMANTIC_SOURCE`, `CONTEXT_SOURCE`, `SEED`, `CONFIG_PATH`,
and `OUTPUT_DIR` are forwarded. Explicit seed arrays must be 0-3. Classification
uses the four-GPU launcher; CPU scientific stages do not reserve those GPUs.

## Evaluation and Outputs

Original human datasets, aliases, queries, probes, scoring, CV folds, common LCA,
and 2000-draw item/seed procedure are unchanged. Changed behavioral settings are
rejected against the frozen protocol. Every model gets item/trial predictions and
an original graph/scorer parity check. Available human reliability is reused.

Separate outputs cover raw features, pretrained semantic activations, and full
ISC-CI behavior. Reports include all 196 cells and all 840 planned scientific
models, even when not run. Planned, partial, diagnostic, failed, and completed
states are not interchangeable. Stage 1 covers source RDMs, category geometry,
cross-seed stability, task-sensitive control, and transfer across feature spaces.
Raw/semantic geometry uses common defined pairs with coverage reporting. Original
CICO common-probe comparisons retain the historical zero fallback for undefined
values rather than silently changing the existing evaluation convention.

Main effects/interactions are descriptive within paired seeds and require a
balanced finite factorial. Unbalanced phases explicitly report nonestimability;
prespecified controlled source-axis and matched-pipeline contrasts retain the
original paired item/seed intervals. The 196 cells are not independent datasets.
Preference is not accuracy; averaged domain correlations are not pooled r.
Exploratory geometry-behavior associations retain these limitations.

Figures are saved-data-only SVG/PDF/300-dpi PNG with hashed source tables. They
include native learning, task coverage/control, raw and semantic RDMs, human
similarity, feature-space transfer, crossed human behavior, and prespecified
contrast intervals. Semantic RDM panels display ranked cosine dissimilarity so
the sigmoid offset does not obscure relative structure. Missing results stay
missing; reports do not invent plots or numerical outcomes.
