# PRD: Independent Leuven Feature-Judgment Expansion for ISC-CI

## 1. Central theme

We will expand the number of items that can be used to train and apply ISC-CI by using an open-source LLM to reproduce the existing Leuven feature judgments for new words.

The primary unit of judgment is:

```text
one target word × one existing Leuven feature
```

Each judgment must be made independently. The LLM should not see other target words, other features, previous judgments, current matrix values, DRM metadata, or ISC-CI outputs in the same prompt.

## 2. Core design principle

The system should behave like a scalable feature-norming instrument.

It should answer one atomic question at a time:

> For a typical participant using the Leuven norming convention, how strongly does this single feature apply to this single word?

It should not answer:

> What features describe this word?
> What category does this word belong to?
> What is the gist of this DRM list?
> Is this word a critical lure?
> How likely is this word to cause false memory?
> How should this word’s full semantic vector look?

The LLM must only judge applicability of one provided Leuven feature to one provided word.

## 3. Why judgments must be independent

Feature batching is efficient, but it creates a possible dependency problem. If the model sees many features for the same word in one context, earlier or neighboring features may bias later judgments. If it sees many words for the same feature, previous words may anchor the interpretation of the feature. If the prompt becomes long, the model may also rely on context-level patterns instead of making each feature judgment independently.

Therefore, for the main validation and production runs:

```text
No word batches.
No feature batches.
No full-vector prompts.
No previous judgments in context.
No conversational memory.
No chain of feature decisions inside one model call.
```

Each model call should contain exactly one target word and exactly one Leuven feature statement.

Batching is allowed only at the infrastructure level: many independent model calls can be parallelized across workers or submitted as separate cluster jobs. Batching must not mean placing multiple judgments in the same prompt.

## 4. Background and motivation

ISC-CI is currently limited by the small number of items in the Leuven semantic norm dataset. This limits how broadly the model can be applied to DRM stimuli.

The first-pass solution is a schema-preserving feature expansion pipeline:

1. Freeze the existing Leuven feature schema.
2. Validate whether an open-source LLM can reproduce Leuven word × feature judgments on held-out Leuven data.
3. Use the validated pipeline to judge new DRM words against the same Leuven features.
4. Build an expanded item × feature matrix.
5. Quantify how much DRM stimulus coverage improves.
6. Test whether the expanded matrix improves ISC-CI applicability while preserving representational geometry.

This is not semantic enrichment. The LLM cannot create new feature columns.

## 5. Input files

Required:

```text
data/leuven_combined_features_consolidated.csv
```

Optional but useful:

```text
data/leuven_categories.csv
data/leuven_combined_exemplar_data.csv
data/leuven_singular_to_plural.csv
```

The feature file defines the fixed schema. The category file is secondary metadata for stratified validation, error analysis, and coverage summaries.

## 6. Main output artifacts

Each run writes to:

```text
artifacts/leuven_feature_expansion/{job_id}/
```

Required files:

```text
expanded_feature_matrix.csv
feature_votes.csv
feature_adjudication_votes.csv
feature_validation_predictions.csv
feature_validation_metrics.json
feature_level_metrics.csv
word_level_metrics.csv
semantic_geometry_metrics.json
coverage_report.csv
isc_ci_expanded_items.csv
manifest.json
run.log
parse_errors.csv
audit_sample.csv
feature_judge_prompt_A.txt
feature_judge_prompt_B.txt
feature_judge_prompt_C.txt
feature_adjudicator_prompt.txt
feature_schema.json
scratchpad.md
todo.md
```

## 7. Feature schema

Generate the feature schema directly from the column names of:

```text
leuven_combined_features_consolidated.csv
```

The first column is the item name. All remaining columns are frozen Leuven feature labels.

Do not hard-code the feature list. Always read it from the Leuven feature matrix.

Example schema:

```json
{
  "schema_version": "leuven_feature_schema_v1",
  "item_column": "Name",
  "n_original_items": 293,
  "n_features": 1995,
  "feature_columns": [
    "is small",
    "is a bird",
    "is an animal",
    "is big",
    "can fly"
  ],
  "value_scale": {
    "min": 0,
    "max": 4,
    "interpretation": "Leuven-style feature applicability"
  }
}
```

## 8. Atomic judgment target

Each atomic judgment is:

```text
target_word × leuven_feature
```

Example:

```text
target_word = "dog"
feature_text = "is an animal"
```

The LLM returns:

```text
feature_value
confidence
ambiguous
reason
```

Use this scale:

```text
0 = feature does not apply
1 = weakly or rarely applies
2 = moderately applies
3 = strongly applies
4 = highly central or diagnostic
```

Because the original Leuven values may include fractional or aggregated ratings, the model should produce 0–4 raw judgments and the pipeline can average resolved votes when appropriate.

## 9. Independence requirements

### 9.1 Prompt-level independence

Each first-pass prompt must include only:

```text
target_word
feature_id
feature_text
value scale
small fixed instruction block
optional fixed few-shot examples
```

Each prompt must not include:

```text
other target words
other Leuven features
previous feature judgments
the word’s current or predicted vector
nearest neighbors
category label unless explicitly used in a separate ablation
DRM dataset
list_id
role
critical_lure
source_paper
false_memory_rate
MBAS
MGS
ISC-CI prediction
```

### 9.2 Call-level independence

Each model call must be stateless.

Implementation requirements:

* Create a fresh user message for every word × feature pair.
* Do not append to chat history.
* Do not use multi-turn conversations.
* Do not include previous model outputs.
* Do not retry by adding previous invalid output into the prompt.
* On retry, resend the same atomic prompt with stricter JSON instruction if needed.

### 9.3 Infrastructure-level parallelism is allowed

Parallel execution is encouraged.

Allowed:

```text
many independent atomic calls across ThreadPool workers
many independent atomic calls across cluster jobs
many independent atomic calls sent through vLLM
resume by row hash
cache by atomic prompt hash
```

Not allowed:

```text
one prompt containing many features
one prompt containing many words
one prompt asking for a full semantic vector
one prompt asking the model to fill a matrix row
```

## 10. Three-pass judging design

Each word × feature pair receives three independent first-pass judgments.

Use the same open-source model with three rubric-equivalent prompt variants:

```text
Judge A: direct feature applicability
Judge B: conservative decision tree
Judge C: checklist
```

All three judges must see the same atomic word × feature pair but use different prompt variants.

The three judges must not see each other’s outputs.

## 11. First-pass prompt files

Create:

```text
leuven_expansion/prompts/feature_judge_prompt_A_v1.txt
leuven_expansion/prompts/feature_judge_prompt_B_v1.txt
leuven_expansion/prompts/feature_judge_prompt_C_v1.txt
leuven_expansion/prompts/feature_adjudicator_prompt_v1.txt
```

### Prompt A: direct feature applicability

```text
You are reproducing feature judgments from the Leuven semantic norm dataset.

You will see one target word and one existing Leuven feature statement.

Your task is to judge how strongly this single feature applies to this single target word under the Leuven norming convention.

Use only the provided feature. Do not invent new features. Do not infer other features. Do not consider any previous or future judgments.

Assign a value from 0 to 4:

0 = the feature does not apply
1 = the feature weakly or rarely applies
2 = the feature moderately applies
3 = the feature strongly applies
4 = the feature is highly central or diagnostic

Follow Leuven-style common knowledge judgments, not specialized scientific taxonomy unless that is the ordinary interpretation.

You will not see the DRM dataset, list identity, whether the word is a critical lure, behavioral results, MBAS, MGS, or ISC-CI predictions. Do not infer those.

Return valid JSON only.
```

### Prompt B: conservative decision tree

```text
You are judging one target_word × Leuven_feature pair.

For this single pair:

1. Read the target word.
2. Read the feature statement.
3. Decide whether the feature would plausibly be produced or endorsed by typical participants describing the word.
4. If the feature does not apply, assign 0.
5. If it applies only weakly or indirectly, assign 1.
6. If it is a normal but not defining property, assign 2.
7. If it is a strong/common property, assign 3.
8. If it is central or diagnostic, assign 4.

Be conservative. Do not mark a feature positive merely because it is remotely associated with the word.

Judge only this pair. Do not infer a broader semantic vector.

Return valid JSON only.
```

### Prompt C: checklist

```text
Judge this single target_word × feature pair using the fixed Leuven feature schema.

Check:

- Is this feature literally true of the target word?
- Is it a common property of the word?
- Is it central or diagnostic?
- Would a typical person plausibly list this feature when describing the word?
- Is the feature only metaphorically or contextually related?

Assign a value from 0 to 4.

Use 0 when the feature is false, irrelevant, metaphorical, or only remotely associated.

Judge only this one pair. Do not use surrounding context or infer other features.

Return exactly one JSON object. Do not include markdown or text outside JSON.
```

## 12. Atomic judge input format

Use this input format for every first-pass model call:

```text
target_word:
{word_normalized}

feature_id:
{feature_id}

feature_text:
{feature_text}

value_scale:
0 = feature does not apply
1 = weakly or rarely applies
2 = moderately applies
3 = strongly applies
4 = highly central or diagnostic

few_shot_examples:
{small fixed examples, optional}
```

The input must contain exactly one target word and one feature.

## 13. Few-shot examples

Few-shot examples are allowed, but they must be fixed and minimal.

Rules:

* Use the same few-shot examples for all atomic judgments within a frozen prompt version.
* Do not choose examples dynamically based on the target word.
* Do not choose examples based on DRM source dataset.
* Do not include held-out validation labels in few-shot examples.
* Keep examples short to avoid context-window crowding.
* Include both positive and negative examples.

Example:

```text
few_shot_examples:
[
  {"word": "monkey", "feature_text": "is an animal", "feature_value": 4},
  {"word": "monkey", "feature_text": "can fly", "feature_value": 0},
  {"word": "hammer", "feature_text": "is a tool", "feature_value": 4},
  {"word": "hammer", "feature_text": "has feathers", "feature_value": 0}
]
```

For validation, examples must come only from the training split or from manually specified examples outside the held-out set.

## 14. JSON schema

Create:

```text
leuven_expansion/schemas/atomic_feature_judgment_schema_v1.json
```

Schema:

```json
{
  "type": "object",
  "additionalProperties": false,
  "required": [
    "target_word",
    "feature_id",
    "feature_value",
    "confidence",
    "ambiguous",
    "reason"
  ],
  "properties": {
    "target_word": {
      "type": "string"
    },
    "feature_id": {
      "type": "integer"
    },
    "feature_value": {
      "type": "number",
      "minimum": 0,
      "maximum": 4
    },
    "confidence": {
      "type": "number",
      "minimum": 0,
      "maximum": 1
    },
    "ambiguous": {
      "type": "boolean"
    },
    "reason": {
      "type": "string",
      "maxLength": 180
    }
  }
}
```

After parsing, validate:

* returned target word matches requested target word
* returned feature ID matches requested feature ID
* feature value is in [0, 4]
* confidence is in [0, 1]
* no extra fields are present

## 15. Feature votes output

Create:

```text
feature_votes.csv
```

One row per word × feature × judge.

Required columns:

```text
job_id
word_original
word_normalized
row_hash
feature_id
feature_text
judge_id
judge_prompt_variant
judge_model
feature_value
confidence
ambiguous
reason
raw_json
parse_error
prompt_hash
```

The `prompt_hash` must uniquely identify the atomic prompt used for that judgment.

## 16. Adjudication

Adjudication happens at the word × feature level.

Trigger adjudication when:

```text
range(feature_value across judges) >= 2
```

or:

```text
one judge says 0 and another says >= 3
```

or:

```text
two judges disagree and the dissenting judge has confidence >= 0.80
```

or:

```text
any first-pass parse/schema error remains after retry
```

or:

```text
at least two judges mark ambiguous=true
```

### Adjudicator input

Adjudicators may see:

```text
target_word
feature_id
feature_text
three first-pass feature values
three confidence values
three ambiguity flags
three reasons
```

Adjudicators must not see:

```text
other features
other words
DRM metadata
ISC-CI outputs
false-memory outcomes
MBAS
MGS
```

### Adjudication resolution

Use three adjudicator calls.

* If at least two adjudicators agree within a tolerance of 0.5, use their mean.
* If all adjudicators disagree substantially, mark the cell for human audit.
* Preserve all adjudicator votes.
* Preserve the first-pass votes.

## 17. Final feature value

For each word × feature pair:

```text
final_feature_value
```

is determined as:

* exact unanimous agreement: use that value
* small disagreement: use mean of valid first-pass values
* majority exact agreement with low-confidence dissent: use majority value
* large disagreement: use adjudicated value
* unresolved adjudication: use provisional mean and mark `needs_human_audit=true`

Do not round values unless ISC-CI requires integer inputs.

## 18. Expanded feature matrix

Create:

```text
expanded_feature_matrix.csv
```

Rows:

```text
original Leuven items
new DRM items
```

Columns:

```text
word
source
all original Leuven feature columns
```

Required metadata columns:

```text
word_original
word_normalized
source
in_original_leuven
in_llm_expansion
feature_completion_method
mean_feature_confidence
n_positive_features
n_adjudicated_features
n_low_confidence_features
n_ambiguous_features
needs_human_audit
```

Feature columns must match the original Leuven feature columns exactly.

## 19. Validation design

Validation should prioritize feature-vector recovery and semantic-geometry preservation.

### 19.1 Cell-level holdout

Hold out individual Leuven word × feature cells.

Use stratified sampling:

```text
positive cells: value > 0
zero cells: value = 0
high-value cells: value >= 3
low-value cells: value = 1 or 2
```

Because each judgment is atomic, held-out cells should be evaluated one at a time using the same atomic prompt format used in production.

Metrics:

```text
binary_accuracy
positive_precision
positive_recall
positive_f1
AUPRC
ROC_AUC
MAE_0_4
RMSE_0_4
confident_error_rate
parse_error_rate
```

### 19.2 Word-level holdout

Hold out entire Leuven words.

For each held-out word, generate a full feature vector by independently judging every word × feature pair.

Metrics:

```text
cosine_similarity_to_gold
Pearson_correlation_to_gold
Spearman_correlation_to_gold
top_10_feature_recall
top_20_feature_recall
positive_precision
positive_recall
positive_f1
MAE_0_4
```

### 19.3 Semantic geometry preservation

Compute pairwise similarity matrices among held-out Leuven items.

Compare:

```text
human Leuven vectors
LLM-reconstructed vectors
embedding baseline vectors
```

Metrics:

```text
Mantel correlation
Spearman correlation of pairwise similarities
nearest-neighbor overlap
category-cluster preservation
```

### 19.4 ISC-CI sensitivity test

Train or apply ISC-CI using:

```text
original human Leuven vectors
LLM-reconstructed Leuven vectors
human Leuven + LLM-expanded vectors
```

Check whether LLM reconstruction preserves ISC-CI behavior for held-out Leuven items.

## 20. Candidate pruning

Candidate pruning is allowed only after exhaustive atomic validation.

Default MVP:

```text
exhaustive atomic judging
```

That means every new word is judged independently against every Leuven feature.

Candidate-pruned mode may be added later to reduce cost, but it must be validated against exhaustive atomic judgments. If pruning misses positive features, it should not be used for the scientific run.

## 21. Categories are secondary

Categories from `leuven_categories.csv` should be used only for:

* stratified train/test splitting
* error grouping
* few-shot balancing if needed
* sanity checks
* coverage reports

Categories should not be the primary judgment target.

Optional category sanity check:

After generating feature vectors, test whether Category 1 can be recovered from predicted feature vectors using a simple nearest-neighbor or classifier approach.

## 22. Backend

Use vLLM on the cluster through an OpenAI-compatible client.

Default settings:

```text
backend: vllm
base_url: http://localhost:8000/v1
temperature: 0
top_p: 1
max_tokens: 200
max_workers: configurable
retries: 2
```

If vLLM supports guided JSON or JSON-schema constrained decoding, use it.

Each request is an independent atomic call.

## 23. Command-line interface

### Validate on held-out Leuven cells

```bash
python -m leuven_expansion.validate_features \
  --mode cell_holdout \
  --leuven-features data/leuven_combined_features_consolidated.csv \
  --leuven-categories data/leuven_categories.csv \
  --job-id leuven_atomic_cell_validation_qwen \
  --output-dir artifacts/leuven_feature_expansion/leuven_atomic_cell_validation_qwen \
  --model Qwen2.5-72B-Instruct \
  --base-url http://localhost:8000/v1 \
  --test-size 0.20 \
  --seed 42 \
  --max-workers 64 \
  --resume
```

### Validate on held-out Leuven words

```bash
python -m leuven_expansion.validate_features \
  --mode word_holdout \
  --leuven-features data/leuven_combined_features_consolidated.csv \
  --leuven-categories data/leuven_categories.csv \
  --job-id leuven_atomic_word_validation_qwen \
  --output-dir artifacts/leuven_feature_expansion/leuven_atomic_word_validation_qwen \
  --model Qwen2.5-72B-Instruct \
  --base-url http://localhost:8000/v1 \
  --test-size 0.20 \
  --seed 42 \
  --max-workers 64 \
  --resume
```

### Expand DRM items

```bash
python -m leuven_expansion.expand_feature_matrix \
  --items data/drm_items_to_classify.csv \
  --leuven-features data/leuven_combined_features_consolidated.csv \
  --singular-plural data/leuven_singular_to_plural.csv \
  --job-id drm_atomic_leuven_feature_expansion_qwen \
  --output-dir artifacts/leuven_feature_expansion/drm_atomic_leuven_feature_expansion_qwen \
  --model Qwen2.5-72B-Instruct \
  --base-url http://localhost:8000/v1 \
  --max-workers 64 \
  --resume
```

## 24. Implementation modules

Create package:

```text
leuven_expansion/
```

Recommended modules:

```text
leuven_expansion/feature_schema.py
leuven_expansion/feature_prompts.py
leuven_expansion/feature_judge.py
leuven_expansion/feature_adjudicate.py
leuven_expansion/validate_features.py
leuven_expansion/expand_feature_matrix.py
leuven_expansion/compute_feature_coverage.py
leuven_expansion/category_metadata.py
leuven_expansion/normalize.py
leuven_expansion/run_jobs.py
```

## 25. TDD and RED-to-GREEN requirement

Use test-driven development.

For every module:

1. Write failing tests first.
2. Confirm they fail for the expected reason.
3. Implement the minimum code needed to pass.
4. Re-run tests.
5. Refactor only after tests are green.
6. Record progress in `todo.md` and `scratchpad.md`.

Do not implement large modules without tests.

Do not mark a task complete until tests pass.

## 26. Scratchpad and TODO requirements

Create and maintain:

```text
scratchpad.md
todo.md
```

### `scratchpad.md`

Required sections:

```text
# Scratchpad

## Current assumptions

## Open questions

## Implementation notes

## Bugs encountered

## Decisions made

## Validation observations
```

### `todo.md`

Required sections:

```text
# TODO

## Red tests to write

## Implementation

## Green tests passed

## Refactors

## Final checks
```

The coding agent should update these after each meaningful implementation step.

## 27. Testing plan

Create:

```text
tests/test_leuven_feature_schema.py
tests/test_leuven_feature_prompts.py
tests/test_leuven_feature_judge.py
tests/test_leuven_feature_adjudicate.py
tests/test_leuven_validate_features.py
tests/test_leuven_expand_feature_matrix.py
tests/test_leuven_feature_coverage.py
tests/test_leuven_category_metadata.py
```

### 27.1 `test_leuven_feature_schema.py`

Tests:

* Loads feature columns from Leuven matrix.
* First column is treated as item name.
* Feature columns are preserved exactly.
* Valid atomic JSON passes.
* Returned feature ID must match requested feature ID.
* Returned target word must match requested target word.
* Extra JSON fields fail.
* Missing required fields fail.
* Feature values must be in [0, 4].
* Confidence must be in [0, 1].

### 27.2 `test_leuven_feature_prompts.py`

Tests:

* Prompt contains exactly one target word.
* Prompt contains exactly one feature ID.
* Prompt contains exactly one feature text.
* Prompt does not contain additional feature statements.
* Prompt does not contain other words to judge.
* Prompt does not contain DRM metadata.
* Prompt does not ask model to fill a vector or matrix.
* Prompt emphasizes independent atomic judgment.

Forbidden fields:

```text
dataset
list_id
role
critical_lure
source_paper
false_memory_rate
MBAS
MGS
ISC-CI prediction
```

### 27.3 `test_leuven_feature_judge.py`

Tests:

* Three first-pass judgments are produced per word × feature pair.
* Prompt variants A/B/C are used.
* Each call receives only one word × feature pair.
* Invalid JSON triggers one retry.
* Retry does not include previous invalid output.
* Raw JSON is saved.
* Prompt hash is stable.

### 27.4 `test_leuven_feature_adjudicate.py`

Tests:

* Exact 3/3 agreement resolves without adjudication.
* Small numeric disagreement averages.
* 0 vs 3 disagreement triggers adjudication.
* 0 vs 4 disagreement triggers adjudication.
* High-confidence dissent triggers adjudication.
* Adjudication sees only the disputed word × feature pair.
* Adjudication failure marks cell for human audit.

### 27.5 `test_leuven_validate_features.py`

Tests:

* Cell holdout works.
* Word holdout works.
* Positive cells are oversampled for validation.
* Metrics are written.
* Pairwise semantic geometry metrics are written.
* Held-out words are not used in few-shot examples.
* Validation uses atomic prompts only.

### 27.6 `test_leuven_expand_feature_matrix.py`

Tests:

* New DRM words load.
* Existing Leuven words are skipped or preserved based on config.
* Duplicate normalized words are classified once.
* Every new word × feature pair creates independent atomic jobs.
* Expanded matrix has exact original feature columns.
* Metadata columns are added.
* Resume mode works.

## 28. Red-to-green implementation order

1. Create feature-first package skeleton.
2. Create `scratchpad.md` and `todo.md`.
3. Write failing feature-schema tests.
4. Implement atomic feature schema validation.
5. Write failing prompt-independence tests.
6. Implement atomic prompt builder.
7. Write failing mock judge tests.
8. Implement three-pass atomic judging.
9. Write failing adjudication tests.
10. Implement cell-level disagreement logic.
11. Write failing validation tests.
12. Implement cell holdout and word holdout validation.
13. Write failing matrix-expansion tests.
14. Implement production matrix expansion using atomic jobs.
15. Write failing coverage tests.
16. Implement DRM coverage reporting.
17. Run full test suite.
18. Run 20-cell mock test.
19. Run 20-cell vLLM smoke test.
20. Run held-out Leuven cell validation.
21. Run held-out Leuven word validation.
22. Inspect feature-level failures.
23. Freeze prompt/schema versions.
24. Run production DRM feature expansion.
25. Build ISC-CI expanded matrix.
26. Build coverage report.
27. Build human audit sample.

## 29. Success criteria

Minimum viable:

```text
positive feature recall >= 0.80
positive feature precision >= 0.70
word-level cosine similarity with human vector >= 0.75
pairwise semantic similarity correlation >= 0.70
parse/schema failure rate <= 1%
all production judgments are atomic and independent
```

Publication-ready:

```text
positive feature recall >= 0.85
positive feature precision >= 0.75
word-level cosine similarity with human vector >= 0.80
pairwise semantic similarity correlation >= 0.75
confident error rate <= 5%
parse/schema failure rate <= 0.5%
all production judgments are atomic and independent
```

## 30. Methods paragraph draft

We expanded ISC-CI stimulus coverage using a schema-preserving LLM feature-norming procedure. The feature schema was fixed to the original Leuven item-by-feature matrix: the model could assign values to existing Leuven feature columns but could not introduce new features. To avoid context-induced dependencies among judgments, each model call judged exactly one target word and one Leuven feature statement. The model did not see other features, other words, previous judgments, DRM dataset identity, list membership, critical-lure status, associative strength, gist strength, or behavioral false-memory outcomes. Each word × feature pair was judged three times using rubric-equivalent prompt variants, with large or high-confidence disagreements adjudicated at the cell level. We first validated the procedure by holding out Leuven cells and Leuven words, testing whether independently generated feature vectors recovered human Leuven vectors and preserved pairwise semantic geometry. After validation, we applied the frozen pipeline to DRM stimulus sets and quantified the increase in item, lure, and complete-list coverage for ISC-CI analyses.
