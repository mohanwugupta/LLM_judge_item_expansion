# Scratchpad

## Current assumptions

- Leuven feature matrix CSV has a "Name" column as the item column (first column).
- All other columns are feature labels in the original Leuven schema.
- Feature IDs are zero-indexed integers corresponding to column order.
- The vLLM server is accessible at the configured base_url.
- Temperature=0 for deterministic outputs.
- Three-pass judging with A/B/C variants; adjudication is triggered by large disagreement.
- Prompt-level independence is strictly enforced: each call sees exactly one word × feature pair.

## Open questions

- What is the exact column name for word labels in leuven_combined_features_consolidated.csv? Assumed "Name" — need to verify.
- Does vLLM at the cluster support guided JSON decoding (json_schema response_format)?
- What is the DRM items file format (drm_items_to_classify.csv)? Assumed CSV with "word" column.
- Are the leuven_singular_to_plural.csv columns "singular" and "plural"?

## Implementation notes

- The package is at leuven_expansion/ within the LLM_judge_item_expansion workspace.
- Uses existing vllm_client.py (top-level) for LLM calls.
- Tests use pytest with mock clients to avoid requiring actual vLLM server.
- feature_votes.csv is written incrementally (append mode) to support resume.
- Row hash = SHA-256 of "{word_normalized}|{feature_id}" for deduplication.
- Prompt hash = SHA-256 of "{system_prompt}\n---\n{user_message}" for audit.

## Bugs encountered

- None yet (initial implementation).

## Decisions made

- Used integer feature_id (0-indexed column position) rather than string keys to keep the JSON schema simple and the validation cross-check exact.
- Adjudication tolerance of 0.5 for "two agree" check.
- Cell-level holdout uses stratified sampling on positive/zero cells separately.
- Word-level holdout uses stratified sampling by Leuven category.
- Few-shot examples are fixed (monkey/hammer) and do not include held-out validation items.

## Validation observations

- Not yet run. See todo.md for validation plan.
