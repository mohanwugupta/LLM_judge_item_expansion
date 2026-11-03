# Source-Grounded Classifier R2

Decision date: 2026-11-03. The investigator explicitly retained **32 classes**
after the missing class was found. This is a named reconstruction, not literal
reproduction of the historical 33-class model. No source matrix or old label is
changed. Human behavior remains an external test, never a classifier input.

## Source Evidence and Correction

1. [ISC Appendix A](https://princetonuniversity.github.io/NEU-PSY-502/_static/pdf/Class%2017/isc.pdf):
   Table A1, PDF pages 111-113, has 32 semantic rows; Figure A1, PDF page 114
   (printed page 113), includes the additional **Associated object** class. The
   earlier table-only audit missed the figure. The *name* is now verified; its
   correspondence to released numeric context IDs is still unavailable.
2. The same manuscript, PDF page 30, illustrates a dog's eating, running and
   barking as behavior. Eating by the entity is not an action performed on it.
3. [Wu and Barsalou (2009), Appendix A, printed pages 187-188](https://barsaloulab.org/Online_Articles/2009-Wu_Barsalou-Acta_Psychologica-simulation_conceptual_combination.pdf)
   distinguishes intrinsic behavior, external-agent action, purpose, parts,
   non-component properties, and situational associates. Its 37-category coding
   system informs these distinctions; it does not replace ISC's modified system.
4. [The Leuven authors' archive](https://ppw.kuleuven.be/gert-storms/resources)
   supplies the feature strings and object-feature judgments. A fresh scan found
   58 XLS workbooks, 590 sheets, and no taxonomy-like sheet names; representative
   Type-II sheets contain bilingual features and object judgments, not task IDs.
   This is not proof that every cell has been exhaustively searched, nor a
   recovered feature-to-task dictionary.

Local manuscript SHA-256:
`178562dfb98e6db074eae378d24c663493110cc660c4b77d05d2eaa3925923e9`.
Original released checkout: `4ecc76d4817fb927d2ad79dfbe1ff44ea72affc0`.
Source files remain unchanged under `ISC-CI_LLM_validation/upstream/`.

The original semantic `.ex` file contains 2,541 unique feature IDs (0..2540),
all 33 numeric task IDs, and **no feature assigned to multiple tasks** when the
semicolon terminator is parsed correctly. That supports one task for each whole
feature in this experiment, but does not identify the numeric class names.
Wu and Barsalou's word-level coding is not silently substituted for ISC targets.

The spelling/whitespace problems are present in the original data, not evidence
of a corrupt download. In `TypeIIAnimalExemplarFeatureMatrix.xls`, first sheet,
English column B: row 171 is `has large theeth`; row 543 is `has big claws `;
row 22 is `has a tail`; row 30 is `eats fish`; row 413 is `drinks water`.
The classifier must interpret the text without rewriting the inventory.

## Directly Grounded Distinctions

| Distinction | Source | R2 interpretation |
| --- | --- | --- |
| Behavior / action / purpose | ISC training example and Table A1; Wu Appendix A | Who acts, on what, and for what usual goal? A modal verb alone is insufficient. |
| Outside / internal part | ISC Table A1; Wu components | Normally external or enclosed component, not visibility after dissection. |
| Component / property | Same | Parts versus attributes; color, shape and size are not components. |
| Quantity | Same | Count, frequency, intensity; countable noun alone is insufficient. |
| Function / contingency / manner | Same | Usual purpose versus dependency/causation versus how an action occurs. |
| Taxonomy / larger whole | Same | Conceptual kind versus physical containment. |
| Situational classes | ISC Table A1 and Figure A1 | Distinguish place, building, event, living thing, person, time and cultural abstract work. |

These are paraphrased operational instructions. Source examples are conformance
checks, not independent human gold labels for the 385 retained features.

## Explicit Reconstruction Choices

- IDs remain 0..31 in approved Table A1 order. No 33rd ID or auxiliary task is added.
- Associated-object descriptions are `out_of_inventory=true` and `uncertain=true`.
  The closest allowed ID is only provisional; human review must document the
  approximation required by the investigator's 32-class choice. No default
  Location assignment, feature deletion, or silent class merge is authorized.
- The sources do not prescribe a universal tie-break for compound descriptions.
  R2 requests a primary assertion only when clear; otherwise it requires explicit
  alternatives and uncertainty. It does not split feature columns or invent a
  parts-first/material-first priority.
- Review all 385 human-feature descriptions, plus the existing per-source/class
  stratified sample and every unresolved or inventory-gap description. Full human
  review is a quality safeguard, not a claim of recovering original annotations.
- A model's confidence is not validation. The source-rule smoke does not establish
  overall accuracy; actual revised classifications and review remain pending.

## Implementation and Failure Policy

`configs/semantic_classifier_r2_prompt.txt` is the model-facing rulebook.
`configs/semantic_retraining_r2.json` selects the new protocol and fresh output:
`artifacts/semantic_retraining_20261103_classifier_r2/` under the validation root.
The previous `semantic_retraining_20261028_32_corrected` run, its 128 labels,
request history, and retrieved archive remain unchanged. They are not imported
as R2 labels.

Stable IDs are assigned to the sorted set of **exact** strings in all 14 frozen
catalogs. Case, spelling, punctuation and trailing spaces remain significant.
The full registry is saved and hashed; only the enabled 2,751 descriptions are
sent to Qwen. Shared descriptions share annotation metadata, not matrix entries.
An inventory change requires a new namespace; IDs cannot silently rebind.

Each response contains IDs, not echoed descriptions. Server-side
[JSON Schema constrained output](https://docs.vllm.ai/en/v0.19.0/features/structured_outputs/)
is combined with client-side coverage, duplicate, type and allowed-ID checks.
The model, local weight fingerprint, runtime versions, prompt, implementation,
taxonomy, registry and retry policy are recorded. There is no fuzzy text repair.

Two attempts per batch are allowed. Malformed batches split recursively down to
single features; successful leaves commit immediately. Failed singletons remain
missing, other batches continue, and the stage returns failure with an explicit
progress file. Network failures get bounded retries without multiplying calls
across all singletons; configuration/schema HTTP errors fail immediately.
Every raw request/response attempt has a distinct filename. Nothing assigns a
default task to a failed item. Resubmission retries only missing labels under
the same protocol. Changed protocol/signatures cannot reuse cached labels.

The isolated smoke tests six source distinctions, the known inventory gap, and
the original misspelling's transport identity. A matching passing smoke is
required before R2 production. A completed classifier still does not authorize
training: reviewed mappings, the training-only human pilot, fidelity gates and
the unlocked full-training schedule remain separate requirements.

Both Slurm launchers and the CLI now default to R2. To inspect the legacy run,
explicitly select `--config ISC-CI_LLM_validation/configs/semantic_retraining.json`
(or `CONFIG_PATH` in Slurm). Never point R2 at the old output directory.
