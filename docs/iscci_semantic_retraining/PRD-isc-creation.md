# PRD: End-to-End ISC Semantic Retraining With LLM Feature Spaces

**Status:** Ready for coding-agent implementation  
**Repository:** Current local `IntegratedSemanticsControlContextInference` repository  
**Primary workspace:** `ISC-CI_LLM_validation/`  
**Method:** Test-driven development (TDD), mandatory RED → GREEN testing, persistent scratchpad  
**Date:** October 28, 2026

## 1. Project objective

Determine whether LLM-generated semantic feature norms can replace human Leuven norms throughout the Integrated Semantics and Control (ISC) and ISC–Context Inference (ISC-CI) pipeline.

Previous experiments trained ISC-CI using different feature matrices but retained the same frozen, human-pretrained 64-dimensional semantic representations. The randomized-feature control demonstrated that substantial human similarity prediction survives even when meaningful feature memberships are destroyed.

This project removes that dependency by independently training ISC semantic representations from each of the 14 existing feature spaces.

The implementation must preserve the original ISC mechanism: **learning representations of objects and tasks through task-conditioned semantic property prediction**.

The experiment will then cross independently trained semantic representations with the existing feature matrices used by ISC-CI.

### Scientific questions

1. Can LLM-generated feature norms produce human-like context-independent semantic representations from scratch?
2. Does the learned network emphasize task-relevant semantic information when provided a task cue?
3. How do feature inventory, elicitation procedure, applicability judgments, and completion affect learned semantic structure?
4. Can LLM-learned representations support human-like similarity, induction, and contextual reasoning in ISC-CI?
5. Do semantic representations generalize across independently generated feature spaces?
6. How much downstream performance persists when meaningful feature assignments are randomized during semantic training, context training, or both?

## 2. Non-negotiable engineering instructions

### Test-driven development

**Every implementation milestone must follow RED → GREEN → REFACTOR.**

1. Write the expected behavior and corresponding test before implementing functionality.
2. Run the test and confirm that it fails for the intended reason (RED).
3. Implement the smallest change necessary to pass (GREEN).
4. Run the test again and record the passing result.
5. Refactor only while preserving passing tests.
6. Run relevant existing regression tests.
7. Record the implementation, commands, failures, and decisions in the scratchpad.

Do not manufacture RED states by intentionally breaking working code. For changes to existing functionality, write a regression test that exposes the missing behavior or previously untested requirement.

Do not weaken assertions, skip tests, or modify scientific definitions merely to obtain a GREEN result.

### Scratchpad

Create a persistent file:

`docs/iscci_semantic_retraining/SCRATCHPAD.md`

Keep it under version control if repository policy permits. Confirm the file is not silently excluded by `.gitignore`.

After each milestone, append:

- Milestone and date
- Goal and implementation status
- Files inspected or changed
- Test written and exact command
- RED result, with expected versus actual failure
- Implementation performed
- GREEN result
- Regression results
- Scientific assumptions and unresolved questions
- Changes to experiment configuration
- Next concrete action

Use this scratchpad to preserve progress between coding-agent sessions. Do not overwrite earlier entries.

### Scientific integrity

- Do not modify, overwrite, or retrain any previously completed experiment.
- Do not change existing feature judgments, consolidation rules, or thresholds.
- Do not generate new applicability judgments.
- Do not pool feature memberships across conditions.
- Do not select training configurations or feature spaces using human behavioral performance.
- Do not introduce human-pretrained semantic weights into the new semantic models.
- Do not launch expensive full-scale experiments until all relevant tests and preflight checks pass.
- All outputs must be reproducible from saved configurations, model seeds, source hashes, and classification metadata.

## 3. Repository audit and initialization

Before implementation, inspect the actual local repository, rather than relying on its public GitHub version.

Locate and document:

- The current ISC-CI `train_run` and `train_model` implementations.
- The existing matrix loading, training, checkpointing, evaluation, and plotting pipelines.
- The original ISC model implementation and pretrained checkpoint.
- Object identity ordering and feature metadata.
- Feature and context index mapping.
- Padding-object behavior.
- Training-seed handling.
- Existing model-fidelity tests.
- The 14 finalized retained feature matrices.
- Relevant artifact directories and immutable hashes.

Read `RANDOM_FEATURE_CONTROL.md`, `DECISIONS.md`, `V4_20261002_RUN.md`, `FINAL_FIGURES.md`, and the current validation README.

Read the original paper's Appendix A and the upstream ISC `models.py` and `data.py`.

Document any discrepancy between the published methodology, released executable code, and current local implementation. In particular, verify how the released ISC code handles biases, output initialization, valid training patterns, and convergence.

**Deliverable:** `AUDIT.md` with confirmed source paths, interfaces, dependencies, missing components, and implementation decisions.

**RED → GREEN gate:** Write failing interface/inventory tests where new loading or validation functionality is needed. Confirm that the final audit is internally consistent and all 14 matrices resolve without altering their contents.

## 4. Feature spaces

Use the exact retained matrices from the completed experiments.

| Condition ID | Features | Role |
|---|---:|---|
| `human` | 385 | Human reference |
| `random_human` | 385 | Fixed shuffled-human control |
| `v2` | 382 | Human feature vocabulary, LLM applicability |
| `v3_A` | 94 | V3 prompt A |
| `v3_B` | 115 | V3 prompt B |
| `v3_C` | 121 | V3 prompt C |
| `v3_1_A` | 88 | V3.1 prompt A |
| `v3_1_B` | 175 | V3.1 prompt B |
| `v3_1_C` | 128 | V3.1 prompt C |
| `v4_b_locked_v2` | 175 | Fixed-B locked judgments |
| `v4_b_calibrated` | 175 | Fixed-B calibrated judgments |
| `v4_ensemble_locked_v2` | 87,315 | Full ensemble, locked |
| `v4_ensemble_calibrated` | 80,699 | Full ensemble, calibrated |
| `v4_ensemble_source_only` | 2,356 | Source-only control |

All conditions have 293 objects, with identical object identities and ordering after explicit validation.

Do not silently remove features that are sparse, unusual, redundant, or difficult to classify.

For `random_human`, reuse the existing saved randomized matrix, generated with `numpy.default_rng(20261028)`. Preserve its object ordering, column ordering, 8,056 positives, and per-feature prevalence. Reuse the human feature-to-task mapping because the column identities are unchanged.

These are 14 conditions, not 14 independent datasets. Several share feature inventories or production procedures.

## 5. Stage 1A: Assign features to the original ISC task taxonomy

### Objective

Reproduce the task-conditioned training environment of the original ISC model.

The original ISC used 33 semantic task classes based on the Wu–Barsalou taxonomy, organized under five broader property families. Three additional auxiliary classes were used for name, category, and size prediction.

For this experiment, implement the **33 semantic classes as the primary task vocabulary**.

### 5.1 Taxonomy specification

Create a machine-readable definition of the 33 task classes using the original paper's Appendix A, Table A1.

For every class, store:

- Stable task ID
- Original class name
- Superordinate family
- Description
- Positive examples
- Potentially confusable neighboring classes

Do not invent new classes or replace the 33 task classes with broad categories such as appearance, behavior, and function alone.

The taxonomy definitions must be versioned and hashed.

### 5.2 Human feature mapping

First determine whether the original Leuven data and repository metadata contain an authoritative feature-to-task mapping for the retained 385 features.

If an exact mapping can be reconstructed, use it.

If not, classify the human retained feature labels using the same documented classification procedure as the LLM features. Record that this is a reconstructed assignment, rather than original ground truth.

Audit the human mapping before proceeding.

### 5.3 LLM feature classification

Classify every retained LLM feature into one of the 33 task classes.

Use the feature's semantic description and the original taxonomy definitions. Do not use human behavioral evaluation outcomes or the feature's 293-dimensional applicability pattern to determine its class.

Recommended workflow:

1. Gather retained feature labels and stable IDs.
2. Reuse verified task labels for exactly identical feature descriptions where appropriate.
3. Classify remaining features with a frozen LLM classifier, fixed prompt, fixed model version, and deterministic settings.
4. Save the predicted task, confidence or uncertainty indicator, and explanation or classification evidence.
5. Recheck ambiguous classifications using a separate documented adjudication pass.
6. Review a stratified sample of classifications across task classes and feature sources.
7. Report classification agreement, ambiguity, and coverage before freezing labels.

Cache identical feature-description classifications to reduce cost. This is permitted metadata reuse, **not pooling object–feature memberships**.

Do not silently assign an ambiguous feature to a convenient category. Flag unresolved assignments, review them, and fail the final pretraining gate if required mappings remain unresolved.

For the very large V4 inventories, classification must be batched, cached, restartable, and auditable. Avoid generating large repeated prompt payloads when the same labels have already been classified.

### 5.4 Classification audits

Report for each condition:

- Features per task class
- Positive object–feature associations per task class
- Number of object–task pairs with at least one positive feature
- Empty and highly sparse task classes
- Classification uncertainty
- Human-reviewed agreement
- Coverage by feature inventory and object

If a task class contains no retained features, preserve its stable identity in the taxonomy but do not invent positive examples.

The procedure for handling inactive classes during training must be specified and tested.

**RED → GREEN tests**

- All valid task IDs resolve to one of the 33 classes.
- Unrecognized or missing assignments fail loudly.
- Identical feature descriptions receive consistent assignments under the locked classifier.
- `human` and `random_human` have identical feature-to-task mappings.
- Feature classification cannot modify the feature matrices.
- All mappings are reproducible from cached outputs and source hashes.
- Classification outputs can be resumed without duplicating completed work.

**Deliverable:** Frozen feature-to-task assignment files and a taxonomy quality report.

## 6. Stage 1B: Reproduce the original ISC training objective

### 6.1 Architecture

Use the original `ISCModel` architecture as the primary implementation.

Preserve:

- One-hot object identity input.
- One-hot task-class input.
- 64-dimensional context-independent semantic hub.
- 16-dimensional task representation.
- 128-dimensional context-dependent representation.
- Sigmoid hidden activations.
- Direct hub-to-output pathway.
- Context-dependent-to-output pathway.
- Binary cross-entropy training objective.

All weights must be initialized independently from scratch for each condition and training seed.

Adapt input and output dimensions to the selected feature space.

Do not silently substitute a plain autoencoder, contrastive model, language embedding, or trainable ISC-CI embedding for the original ISC architecture.

### 6.2 Training targets

Let:

- \(F_{o,f}\) indicate whether object \(o\) possesses feature \(f\).
- \(g(f)\) identify the task class of feature \(f\).
- \(t\) indicate the current task.

The target for object \(o\), task \(t\), and output feature \(f\) is:

\[
y_{o,t,f}=F_{o,f}\cdot\mathbb{1}[g(f)=t]
\]

Thus, a training example activates all features belonging to the supplied task that are true of the object, while all other output units have zero targets.

Use BCE with logits over the full output vector, consistent with original ISC. Do not silently replace this with a loss calculated only over features in the selected task class.

For example, when presented with DOG and the task BEHAVIOR, the model should activate the dog's behavior-related features while suppressing features from other tasks, even when those features are true of dogs.

### 6.3 Construct valid object–task examples

Build a deterministic training-data adapter from each binary object–feature matrix and its taxonomy assignments.

Inspect the original loader to establish its exact rule for valid object–task combinations. Reproduce that rule, including how object–task pairs with no positive targets are handled.

Do not assume that all 293 × 33 combinations were part of the original training procedure.

The paper describes 7,057 valid training patterns from its original 350-object, 36-task environment. The new matrices will produce different counts.

Report example counts and task exposure for each condition.

### 6.4 Auxiliary targets

Primary experiment: **semantic tasks only, without separately imported size, category, or naming targets.**

Reason: the primary scientific question concerns whether the selected feature matrix is sufficient to learn useful semantic representations.

Human-derived size and category targets would introduce a separate source of human semantic information. Naming targets would impose additional object-identity supervision.

Preserve a configuration interface allowing auxiliary tasks to be evaluated later as an explicit ablation. Do not include them silently.

This is a matched new experiment, not an exact reproduction of the released 350-object encoder.

### 6.5 Training and convergence

Original ISC and later ISC-CI used different training schedules. Do not apply ISC-CI's 400-epoch budget to Stage 1 by default.

First reproduce the original ISC training behavior using small fixtures and a human-reference pilot.

The original ISC paper describes training until the maximum absolute output error falls below 0.1. Confirm this criterion against executable code and record any differences.

For the new study:

- Preserve the original loss and architecture.
- Determine a defensible training schedule from training-only diagnostics.
- Use matched training settings across sources.
- Record both update count and actual object–task exposure.
- Save learning curves and convergence information.
- Define a resource limit and failure-to-converge policy before examining human behavioral outcomes.
- Never select a checkpoint based on downstream human behavior.

Because inventories vary from 88 to 87,315 features, identical epochs may not imply equivalent optimization or coverage.

A fixed-budget comparison and a convergence-matched sensitivity analysis may both be useful. Their definitions must be locked before human behavioral evaluation.

### 6.6 Large-output feasibility

The V4 ensemble models require prediction over approximately 80,000–87,000 output features.

Before full training:

1. Benchmark memory use and throughput on representative small, medium, and ensemble conditions.
2. Verify that the implementation can handle large output matrices without numerical instability.
3. Evaluate output chunking or equivalent memory-efficient computation if required.
4. Confirm that any optimization produces the same loss and gradients as the full dense BCE objective within numerical tolerance.
5. Record task exposure and feature coverage.
6. Check performance against an all-negative predictor so low training loss is not mistaken for successful feature learning.

Do not silently introduce negative sampling, class weighting, or a modified loss to accommodate the large models.

If the large models cannot be trained faithfully within available resources, document the limitation and preserve the smaller completed experiments.

### 6.7 Stage 1 outputs

For each condition and seed, save:

- Model checkpoint
- Training configuration
- Task assignments and hashes
- Loss and prediction metrics
- Convergence record
- Exact ordered 293 × 64 semantic activation matrix
- Object identity mapping
- Stage 1 completion manifest

The representations must be extracted through the original sigmoid-activated context-independent layer, not raw weights or preactivation values.

**RED → GREEN tests**

- Correct output targets for known object–task examples.
- Task-irrelevant features are zero in the targets.
- The original architecture is reconstructed correctly.
- BCE loss matches a reference implementation.
- Hub and task representations receive gradients during training.
- A small synthetic task-conditioned dataset can be learned.
- Changing the task cue changes appropriate predictions.
- Semantic hub activations remain independent of the current task.
- A fresh model has no pretrained checkpoint dependency.
- Saved representations exactly match representations recomputed from the checkpoint.
- Large-output optimized training matches the reference loss and gradients.

**Deliverable:** Reproducible ISC training and semantic representation extraction for all 14 conditions.

## 7. Stage 1C: Validate learned semantic representations

Before any new ISC-CI training, evaluate the semantic representations independently.

### Primary measures

- Human similarity correlation across the original nine domains.
- Semantic representation RDM agreement with the human reference.
- Between-category and within-category representation geometry.
- Cross-seed representation stability.
- Training feature-prediction performance.
- Task-conditioned representational changes.

Retain missingness and pair-coverage rules from the existing human validation wherever applicable.

### Test the ISC mechanism

For each semantic task class, calculate how well the context-dependent representation captures the corresponding task-relevant feature geometry.

Compare this against the context-independent hub representation.

The original ISC paper reports stronger task-feature correspondence for context-dependent than context-independent representations.

This provides a direct check that our newly trained model is learning task-sensitive semantic control rather than merely memorizing a binary feature matrix.

Report these measures by task class, with task coverage and valid-pair counts. Do not force an expectation that every individual class must show improvement.

Also examine whether a model trained on one feature space captures semantic structure expressed in another feature space. Keep these representational transfer comparisons separate from the later ISC-CI behavioral tests.

### Evaluation integrity

External human behavioral outcomes must not be used for feature classification revisions, stopping-rule selection, or model tuning.

If the human-reference semantic model fails to learn its own training environment or transfer correctly, stop and diagnose Stage 1 before expanding.

## 8. Stage 2: Transfer semantic representations into ISC-CI

For every completed semantic encoder:

1. Load its ordered 293 × 64 semantic activation matrix.
2. Load the corresponding ISC-CI architecture.
3. Verify exact object identity and ordering.
4. Initialize `input_to_independent` using the learned activations.
5. Handle the padding-object row according to the verified local implementation.
6. Freeze the transferred semantic embedding.
7. Randomly initialize all remaining ISC-CI parameters.
8. Train ISC-CI using the selected downstream feature matrix.
9. Save the resulting model and provenance.

The semantic embedding must remain frozen during ISC-CI training.

All other ISC-CI settings should match the established validation protocol:

- 400 epochs
- 1,024 episodes per epoch
- Batch size 128
- Adam, learning rate 0.001
- Task-loss weight 0.5
- Support sizes 1 and 2
- Paired seeds 0–3

Revalidate these settings against the current local runner.

Preserve the original training sampler and target semantics. Do not change loss weighting or episode allocation without an explicitly labeled additional experiment.

**RED → GREEN tests**

- Object IDs align exactly.
- Missing, duplicated, or reordered objects raise errors.
- Extracted semantic activations survive transfer without numerical changes.
- Padding cannot leak into real object identities or influence evaluated representations unexpectedly.
- Embedding gradients remain disabled during downstream training.
- All other ISC-CI parameters receive appropriate gradients.
- The loaded checkpoint reproduces its saved predictions.
- The original ISC-CI graph and scoring parity checks pass.
- Existing baseline checkpoints and outputs remain unchanged.

## 9. Complete experimental crossing

Let \(S\) identify the source of the semantic representations learned by ISC.

Let \(C\) identify the source of the ISC-CI context and membership matrix.

Train each pair:

\[
\operatorname{ISC\text{-}CI}(S,C)
\]

The experiment contains:

- 14 semantic sources
- 14 downstream context sources
- 196 unique source pairings
- Four paired training seeds per pairing
- Up to 784 newly trained ISC-CI models
- 56 independently trained ISC semantic models

Each semantic checkpoint may be reused across multiple downstream context conditions, but all downstream trainable weights must be newly initialized.

The original released human semantic encoder is retained only as a historical reference, not a substitute for newly trained human encoders.

### Execution stages

**Phase 1 — Pipeline validation**

Train one human semantic encoder and test transfer and downstream functionality using a short diagnostic training run.

**Phase 2 — Matched control pilot**

Train the human, fixed randomized-human, and one prespecified LLM semantic source with four seeds. Compare the Stage 1 representations before running downstream tests.

**Phase 3 — All matched conditions**

Train semantic encoders for all 14 feature sources. Train the 14 diagonal source pairings, where each ISC-CI model uses the feature matrix corresponding to its semantic encoder.

**Phase 4 — Human and randomized crossings**

Run all pairings involving human or randomized-human sources at either stage. This phase, together with the diagonal, covers 64 unique configurations.

**Phase 5 — Complete factorial design**

Run the remaining 132 LLM-to-LLM source pairings after the resource and validation gates pass.

Preserve the complete factorial as the intended design. Phasing determines execution order, not post hoc condition selection.

Do not automatically launch the next phase if source validation, scientific fidelity, test coverage, or resource checks fail.

## 10. Downstream evaluation

Reuse the completed original-paper human behavioral evaluation protocol and scoring functions.

The primary output is performance against actual human behavioral observations, not merely agreement with previously trained human ISC-CI models.

### Required outcomes

- Nine human semantic similarity domains
- Five human induction datasets
- Asymmetric similarity
- Context-effect direction and magnitude
- Context-dependent argument preferences
- Nonmonotonicity argument preferences
- Item-level prediction errors
- Representation agreement and task-specific geometry

Preserve the existing 2,000-draw crossed item/seed uncertainty procedure where appropriate. Distinguish seed variation, item uncertainty, and uncertainty across randomized feature matrices.

Do not interpret argument-2 preference as accuracy.

Report descriptive domain averages only as averages of correlations, not pooled correlations.

### Analysis plan

Estimate the effects of:

1. Semantic training source
2. Downstream context source
3. Their interaction

Prioritize scientifically interpretable source contrasts:

- Human versus randomized semantic training
- Human versus randomized context training
- V2 versus human feature applicability
- V3 versus V3.1 feature elicitation
- V3.1-B versus V4 fixed-B completion
- Locked versus calibrated judgments
- Source-only versus completed ensemble
- Matched LLM pipelines versus mismatched LLM-to-LLM crossings

Report all pairings and prespecified contrasts, including negative results.

Treat the original fixed randomized matrix as one realization. Four training seeds do not constitute four independent random-feature realizations. Additional randomizations, if undertaken, must be labeled as a separate extension.

Do not treat the 196 related source pairs as independent evidence of generalization to 196 different datasets.

## 11. Artifacts and directory layout

Use the existing local validation workspace and naming conventions after completing the repository audit.

Suggested structure:

```text
ISC-CI_LLM_validation/
  semantic_retraining/
    taxonomy.py
    classify_features.py
    training_adapter.py
    train_isc.py
    extract_representations.py
    transfer_embeddings.py
    run_crosses.py
    evaluate_crosses.py
    report_results.py

  configs/
    semantic_retraining.json
    task_taxonomy_33.json
    semantic_crosses.json

  tests/
    test_semantic_inventory.py
    test_task_taxonomy.py
    test_feature_classification.py
    test_isc_training_targets.py
    test_isc_training.py
    test_large_output_equivalence.py
    test_semantic_representations.py
    test_embedding_transfer.py
    test_semantic_crosses.py
    test_semantic_retraining_regression.py

  artifacts/
    semantic_retraining_20261028/
      inventory/
      taxonomy/
      classification/
      isc_models/
      semantic_representations/
      iscci_models/
      evaluations/
      manifests/

  reports/
    semantic_retraining_20261028/
      RESULTS.md
      figures/
      source_data/
```

Keep the PRD, audit, and scratchpad in a tracked documentation location so the scientific reasoning and agent progress remain recoverable independently of Git-ignored training artifacts.

Use actual locally verified module and directory names if they differ from the suggestions above. Avoid unnecessary rewrites of existing modules.

## 12. RED → GREEN milestone gates

Each phase is blocked until its tests pass.

| Milestone | Required failing test (RED) | Required result (GREEN) |
|---|---|---|
| M0: Repository audit | Missing inventory/interface validation | All source paths and matrices resolve |
| M1: Taxonomy | Unmapped or invalid task IDs | Valid 33-class taxonomy |
| M2: Classification | Missing or inconsistent feature assignments | Complete, stable, audited mappings |
| M3: Training adapter | Incorrect object–task targets | Exact target generation |
| M4: ISC architecture | Mismatch against reference forward pass | Verified original-style architecture |
| M5: ISC training | Failure to learn synthetic task-conditioned targets | Reproducible training and correct predictions |
| M6: Large outputs | Dense/chunked loss or gradient mismatch | Numerically equivalent scalable implementation |
| M7: Representation extraction | Incorrect activation or object order | Valid 293 × 64 embeddings |
| M8: Transfer | Changed embeddings or unexpected trainable gradients | Exact frozen transfer |
| M9: ISC-CI training | Downstream parity or configuration mismatch | Fresh valid checkpoints |
| M10: Crosses | Missing, duplicated, or mislabeled pairings | Complete 14 × 14 registry |
| M11: Evaluation | Scoring or original-model parity mismatch | Reproduced frozen evaluation |
| M12: Preservation | Changed baseline hashes | Original outputs unchanged |

At each gate, run the new focused tests and the relevant existing regression suite.

Record both RED and GREEN commands and outcomes in the scratchpad.

A failure in scientific parity, source integrity, or data mapping must block training rather than be downgraded to a warning.

## 13. Reproducibility and execution controls

All expensive execution must be explicitly staged.

Provide command-line interfaces supporting:

- `--stage`
- `--condition`
- `--semantic-source`
- `--context-source`
- `--seed`
- `--dry-run`
- `--resume`
- `--output-dir`
- `--workers`

Use the existing repository's CLI conventions where practical.

Before training, `--dry-run` must show the planned sources, seed count, expected output locations, configuration hashes, and estimated memory requirements without changing scientific artifacts.

Resume only when saved artifacts match source hashes, taxonomy versions, configuration, code provenance, and training completion criteria.

Completed models must never be silently overwritten.

Save checkpoints atomically when practical and preserve manifest records of interrupted runs.

Log computational resources and training duration so differences in feature inventory size and convergence are transparent.

## 14. Final deliverables

### Code

A tested, reproducible pipeline that:

1. Loads the 14 existing matrices.
2. Assigns features to the original 33 ISC tasks.
3. Trains ISC semantic representations independently.
4. Extracts frozen semantic embeddings.
5. Transfers embeddings into ISC-CI.
6. Runs the complete crossing design.
7. Reuses the established human behavioral evaluation.

### Documentation

- Implementation audit
- Taxonomy specification and classification report
- Persistent TDD scratchpad
- Training and convergence configurations
- Reproduction commands
- Checkpoint and dataset manifests
- Final scientific results and limitations

### Figures and tables

- Feature inventory and task-class coverage
- ISC semantic learning curves
- Human-versus-LLM semantic RDMs
- Task-dependent versus task-independent geometry
- Human similarity prediction from newly trained embeddings
- ISC-CI performance across crossed conditions
- Prespecified source contrasts with uncertainty
- Randomized-feature control comparisons
- Training exposure and convergence diagnostics

## 15. Final acceptance criteria

Implementation is complete only when:

1. All 14 matrices and their retained memberships are verified and unchanged.
2. All retained features have auditable 33-class assignments.
3. ISC models are trained from independent random initialization.
4. The original task-conditioned prediction objective and architecture are preserved.
5. No human-pretrained semantic embeddings enter the new semantic training pipeline.
6. The primary experiment uses no additional human-derived size/category supervision.
7. Extracted 64-dimensional semantic representations are validated.
8. Representation transfer into ISC-CI is exact and frozen.
9. The complete 196-pair registry is implemented, with completed runs documented separately from planned runs.
10. All required completed checkpoints and predictions pass fidelity and reproducibility checks.
11. Existing experiment outputs retain their original hashes.
12. The TDD scratchpad records RED → GREEN evidence for every implementation milestone.
13. The final report distinguishes feature-space quality, semantic representation quality, task-dependent control, and downstream behavioral prediction.
14. Any incomplete training condition or unresolved scientific discrepancy is reported rather than hidden.

## 16. Agent execution policy

**Start with repository audit and TDD scaffolding. Do not start full model training immediately.**

Work sequentially through the milestones. Implement and test one vertical slice at a time.

After reaching GREEN, update the scratchpad and continue to the next milestone when the scientific and resource requirements permit.

Do not revise the experimental design based on behavioral results. If an unforeseen implementation decision would materially change the scientific comparison, document the competing options and their consequences before implementing a change.

Keep the original completed experiments immutable and implement this work as a separate, versioned extension.

The intended endpoint is a scientifically defensible end-to-end comparison of independently learned human, LLM, and randomized semantic representations, including how these representations interact with different feature spaces during context inference.
