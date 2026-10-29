# ISC Semantic Retraining Audit

Date: 2026-10-28. Scope: a new extension under `ISC-CI_LLM_validation/`.
Completed experiments, original code, and original feature memberships are inputs,
not targets for modification. Pipeline infrastructure is implemented; scientific
execution remains gated on reviewed classifications and a locked schedule.

## Confirmed Local Interfaces

| Purpose | Path under ISC-CI_LLM_validation |
| --- | --- |
| Existing training orchestration | `run_validation.py::train_run` |
| Existing ISC-CI learning loop | `iscci_validation/training.py::train_model` |
| Existing CICO graph and frozen embedding initialization | `iscci_validation/modeling.py` |
| Binary matrices, matrix hashes, original episode sampler | `iscci_validation/dataio.py` |
| Checkpoint saving and restoration | `iscci_validation/training.py` |
| Common representational probes | `iscci_validation/evaluation.py` |
| Original human observations and item-level scoring | `run_human_behavior_validation.py`, `iscci_validation/behavior.py` |
| Existing graph/scorer parity | `iscci_validation/behavior_fidelity.py` |
| Readonly final-results rendering | `plot_finalized_v4_results.py` |
| Original ISC graph and loader | `upstream/IntegratedSemanticsControl/models.py`, `data.py` |
| Released no-hidden-bias ISC checkpoint | `upstream/IntegratedSemanticsControl/models/csc_model_0-nobias.torch` |
| Original ISC paper | `upstream/ISC_original_paper.pdf` |

The adjacent `IntegratedSemanticsControlContextInference` repository contains
ISC-CI, not the initial ISC semantic model. The original ISC checkout is already
present in the validation upstream directory, revision
`4ecc76d4817fb927d2ad79dfbe1ff44ea72affc0`.

Reviewed: README.md, RANDOM_FEATURE_CONTROL.md, DECISIONS.md,
V4_20261002_RUN.md, FINAL_FIGURES.md; original ISC models.py, data.py and Appendix A.
The dependency stack is existing NumPy, pandas, PyTorch, SciPy, and pytest.
PDF/XLS tools were used only for the source audit, not as training dependencies.

## Frozen Inputs

Thirteen retained matrices resolve under
`artifacts/validation_v4_20261002/data_matrices/`. The saved random control resolves
under `artifacts/random_feature_control_20261028/validation/data_matrices/`.
`semantic_retraining/inventory.py::inventory` verifies all fourteen dimensions,
exact object order, binary contents, unique object/feature IDs, and file hashes
before/after reads. It returns file SHA-256, canonical int8 matrix hash, object and
feature order, positive counts, and feature prevalences without writing inputs.

| Condition | Retained Features |
| --- | ---: |
| human | 385 |
| random_human | 385 |
| v2 | 382 |
| v3_A / v3_B / v3_C | 94 / 115 / 121 |
| v3_1_A / v3_1_B / v3_1_C | 88 / 175 / 128 |
| v4_b_locked_v2 / v4_b_calibrated | 175 / 175 |
| v4_ensemble_locked_v2 | 87315 |
| v4_ensemble_calibrated | 80699 |
| v4_ensemble_source_only | 2356 |

All have 293 objects in identical order. Random_human is the existing one fixed
PCG64 realization, seed 20261028, not a new randomization. It has 8056 positives,
the same columns and per-column prevalence as human, but different object degrees.
The human task mapping must also be reused unchanged for random_human.

V2/human feature IDs are readable labels. V3/V3.1 descriptions join to the saved
`*_cluster_labels.csv` files. V4 descriptions join candidate_id to
canonical_feature_text in the existing candidate bank. Tests verify every retained
ID resolves without changing a matrix. Exact description caching shares metadata
only, never object-feature memberships.

Existing CICO uses a zero padding row at index 293 after the 293 real objects.
One/two-shot episodes use the unchanged sampler, 1024 episodes per epoch, batch
128, Adam 0.001, task-loss weight 0.5, 400 epochs, seeds 0-3. These are downstream
settings and must not automatically become the semantic-stage training schedule.

## Source Discrepancies and Decisions

### 1. Semantic Taxonomy: Approved 32-Class Reconstruction

The original paper claims 33 semantic classes under five families. Visual
inspection of Table A1 on PDF pages 111-113 finds 32 named semantic rows. The
released semantic patterns use numeric tasks 1-33, but include no name dictionary.
The raw Leuven archive linked by the authors contains 58 workbooks / 590 sheets;
inspection found no feature-type dictionary. Wu and Barsalou (2009) describes a
different 37-subtype system; it cannot silently substitute for the ISC variant.

The investigator approved an explicitly labeled 32-class reconstruction on
2026-10-28. `configs/task_taxonomy_32_reconstructed.json` preserves those 32 names
and five families. New IDs 0-31 follow table row order and do not claim to reproduce
historical numeric context IDs. Operational descriptions and confusable neighbors
are reconstructed guidance. A future verified 33-class experiment must use a new
taxonomy/protocol and new outputs.

No authoritative mapping of the retained 385 human features was found. Human
assignments therefore require the same documented classification and review
process as LLM descriptions; they are not original ground-truth annotations.

### 2. Released Target Parser: Investigator-Approved Correction

Original data.py line 142 uses `split()[:-1]` to remove what it treats as a separate
terminator. The file actually attaches `;` to the last numeric feature ID. For
`t: 66 67 68;`, the released loader retains only 66 and 67. For `t: 351;`, it
retains no positive output. All 6007 semantic rows lose their final feature, and
1532 become entirely zero. Adding the loader's 1050 auxiliary rows explains the
paper's 7057 total patterns.

The investigator approved correcting this on 2026-10-28 after the consequences
were explained. `semantic_retraining/training_adapter.py::parse_output_ids`
parses the terminator rather than deleting a token. The new matrix-native
`TaskDataset` avoids the text format and implements exactly:

`y[o,t,f] = F[o,f] * (feature_task[f] == t)`.

Positive object-task pairs are selected before target construction, consistent
with the input file's preselection. Empty-positive pairs are not invented, and
inactive class IDs are retained. Task-irrelevant features have zero targets.
No auxiliary size, category, or naming targets are imported.

This is a corrected matched new experiment, not literal reproduction of the
released parser. The original checkout is unchanged for fidelity comparisons.
The finding does not establish which parser trained every historical checkpoint.

### 3. Architecture and Biases

The original graph uses 64D sigmoid object hubs, 16D sigmoid task representations,
128D sigmoid dependent representations, and direct hub/dependent output pathways.
Weights initialize uniformly in [-0.01, 0.01]. Loss is full-vector BCE with logits.

The paper describes fixed hidden biases of zero and output biases of -2.
Executable no-hidden-bias mode omits hidden biases but always creates a trainable
direct-output bias, initialized at -2. The released no-bias checkpoint contains
that bias with learned values different from -2. The constructor's alternate
default mode enables biases throughout. These are distinct variants.

The reconstruction exposes both modes for graph fidelity tests and defaults to
no-hidden-bias mode with the executable trainable output bias. Tests require
bitwise equality against original initialization, logits, and hidden activations
in both modes. No pretrained checkpoint is read. The new class retains ordinary
PyTorch train/eval modes instead of overriding nn.Module.train with a learning
loop. Separate training orchestration implements full-vector loss, convergence
checks, exact semantic optimizer/cursor resume, and failure-safe checkpoints.

### 4. Training Schedule Remains Unlocked

The paper describes maximum absolute error <0.1. The released class accepts a
fixed epoch budget; its loader defaults to 30 epochs and constructor Adam learning
rate 0.05. It does not enforce the paper's convergence criterion. Do not infer
that 400 CICO epochs reproduce original semantic training.

A training-only human pilot, synthetic learning checks, and large-output
loss/gradient equivalence and resource benchmarks must precede a locked schedule.
No human behavioral results may choose a schedule, feature mapping, or checkpoint.

## Pinned Source Hashes

| Source | SHA-256 |
| --- | --- |
| Original ISC models.py | `dfe42e77d70e32e6fac266bb97077adddfa1d1895ade77959d78c421c69d3c6c` |
| Original ISC data.py | `0ff1f8a168c69d8c527f200b138f6e5d944417a7b9b4555e8d02c1dc5801f893` |
| Original semantic .ex patterns | `f91a3e3a5af19b35e51f5fb5a77a9e8a34756ad8f63c67408eddc609a11e9aeb` |
| Original ISC PDF | `178562dfb98e6db074eae378d24c663493110cc660c4b77d05d2eaa3925923e9` |

## Remaining Gates

Implemented and tested: readonly inventory; taxonomy and corrected targets;
original graph initialization/forward parity; deterministic Qwen classification,
ambiguity adjudication, review and immutable mapping freezes; streamed full-vector
loss/gradient equivalence; semantic learning, exact resume and sigmoid extraction;
bitwise frozen CICO transfer; phased 56-semantic/784-CICO orchestration; original
human evaluation and saved-result reports. Dedicated classification and CPU-stage
Slurm launchers default to bounded smoke tests. No cluster jobs were submitted.

The investigator selected the existing Qwen2.5-72B-Instruct checkpoint. The
launcher fingerprints actual local weights before serving it, rather than guessing
a revision. Classification tests use fixtures; no real Qwen calls have run.
Actual prepare resolved 87,679 unique descriptions across all fourteen sources.
Local synthetic learning, transfer, and bounded 87,315-output resource checks
passed; these do not establish convergence on the real scientific matrices.

Still pending: real classifications/adjudication, stratified human review and
frozen assignments; training-only human pilot; documented full-schedule lock;
real semantic/downstream training; new human trial predictions and scientific
results. The existing human-behavior evaluation remains the external test, not a
tuning source. The 32-class reconstruction and corrected-target experiment must
not be described as a literal 33-class historical reproduction.

## Primary Sources

- [Original ISC source](https://github.com/tylergiallanza/IntegratedSemanticsControl)
- [Original ISC manuscript and Appendix A](https://princetonuniversity.github.io/NEU-PSY-502/_static/pdf/Class%2017/isc.pdf)
- [Authors' Leuven resources page](https://ppw.kuleuven.be/gert-storms/resources)
- [Wu and Barsalou 2009](https://barsaloulab.org/Online_Articles/2009-Wu_Barsalou-Acta_Psychologica-simulation_conceptual_combination.pdf)
