# TODO

## Red tests to write

- [x] test_leuven_feature_schema.py
- [x] test_leuven_feature_prompts.py
- [x] test_leuven_feature_judge.py
- [x] test_leuven_feature_adjudicate.py
- [x] test_leuven_validate_features.py
- [x] test_leuven_expand_feature_matrix.py
- [x] test_leuven_feature_coverage.py
- [x] test_leuven_category_metadata.py

## Implementation

- [x] leuven_expansion/__init__.py
- [x] leuven_expansion/schemas/atomic_feature_judgment_schema_v1.json
- [x] leuven_expansion/prompts/feature_judge_prompt_A_v1.txt
- [x] leuven_expansion/prompts/feature_judge_prompt_B_v1.txt
- [x] leuven_expansion/prompts/feature_judge_prompt_C_v1.txt
- [x] leuven_expansion/prompts/feature_adjudicator_prompt_v1.txt
- [x] leuven_expansion/normalize.py
- [x] leuven_expansion/feature_schema.py
- [x] leuven_expansion/feature_prompts.py
- [x] leuven_expansion/feature_judge.py
- [x] leuven_expansion/feature_adjudicate.py
- [x] leuven_expansion/category_metadata.py
- [x] leuven_expansion/run_jobs.py
- [x] leuven_expansion/validate_features.py
- [x] leuven_expansion/expand_feature_matrix.py
- [x] leuven_expansion/compute_feature_coverage.py

## Green tests passed

- [x] Run: pytest tests/ -v  → **98 passed, 0 failed**
- [x] Verify all tests GREEN after implementation

## Refactors

- [x] Verify item_column name in leuven_combined_features_consolidated.csv  → `"Unnamed: 0"` (first col); load_leuven_feature_schema now auto-detects
- [x] Confirm singular_to_plural CSV column names  → `singular` / `plural` (already correct)
- [x] Add parse_errors.csv output to run_jobs.py
- [x] Add manifest.json output to run_jobs.py
- [x] Add run.log output to run_jobs.py

## Final checks

- [ ] Run 20-cell mock test (no vLLM)
- [ ] Run 20-cell vLLM smoke test
- [ ] Run held-out Leuven cell validation (--mode cell_holdout)
- [ ] Run held-out Leuven word validation (--mode word_holdout)
- [ ] Inspect feature-level failures in feature_level_metrics.csv
- [ ] Freeze prompt/schema versions
- [ ] Run production DRM feature expansion
- [ ] Build expanded_feature_matrix.csv
- [ ] Build coverage_report.csv
- [ ] Build human audit sample (audit_sample.csv)
- [ ] Verify positive_feature_recall >= 0.80
- [ ] Verify word-level cosine_similarity >= 0.75
- [ ] Verify parse_error_rate <= 1%
