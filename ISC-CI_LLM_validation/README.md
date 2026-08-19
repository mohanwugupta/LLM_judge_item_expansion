# ISC-CI LLM Validation

This directory contains the complete human-v2-v3-v3.1 ISC-CI validation and paper-simulation
reproduction. Prompt variants A, B, and C are separate conditions within each generation
version.

## Run Order

```bash
python consolidate_v3.py
python run_validation.py
python evaluate_validation.py
python run_paper_simulations.py
python audit_consolidation_rules.py
python summarize_results.py
pytest -q tests
```

For the cumulative V3.1 comparison, run the exact commands recorded in
`reports/v3_1/RESULTS.md`. The V3.1 training stage initializes
`artifacts/validation_v3_1/` from the completed V3 validation, verifies and reuses those
checkpoints, and adds only the 12 new V3.1 models.

Each stage writes a manifest with input hashes and is resumable. Use `--force` only when the
relevant source, configuration, or implementation intentionally changed.

## Key Files

- `DECISIONS.md`: methodological decisions and reconstruction caveats.
- `configs/v3_validation.json`: locked consolidation, training, and evaluation parameters.
- `configs/v3_consolidation_manual_review.csv`: required review of retained semantic merges.
- `configs/v3_1_validation.json`: V3.1 protocol with parameters locked equal to V3.
- `configs/v3_1_consolidation_manual_review.csv`: required V3.1 semantic-merge review.
- `artifacts/v3_consolidation/`: phrase assignments, matrices, sensitivity analysis, manifest.
- `artifacts/v3_1_consolidation/`: V3.1 assignments, matrices, sensitivity, and manifest.
- `artifacts/validation_v3/`: 20 retrained checkpoints, fixed-context evaluation, and all paper
  simulation outputs.
- `artifacts/validation_v3_1/`: cumulative V3.1 validation with 32 retrained checkpoints and
  four released checkpoints.
- `reports/RESULTS.md`: scientific interpretation and main tables.
- `reports/v3_1/RESULTS.md`: prompt-matched V3.1 versus V3 results and paper simulations.
- `reports/consolidation_audit/CONSOLIDATION_AUDIT.md`: embedding/profile/lexical rule
  sensitivity and retained-cluster audit.
- `reports/report_manifest.json`: hashes linking the report to every analysis stage.

The directory is intentionally ignored by the parent repository, per project policy. Preserve
or archive it explicitly before cleanup because Git will not recover these files.
