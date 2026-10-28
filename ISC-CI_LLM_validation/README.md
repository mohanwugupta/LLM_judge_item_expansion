# ISC-CI LLM Validation

The latest figures add the investigator-approved four-seed human-sized random-feature
control: `reports/v4_20261028_random_control/figures_final/INDEX.md`.
See `RANDOM_FEATURE_CONTROL.md` for the fixed shuffle, original-code reuse, results
locations, limitations, and one-command replay. The original 13-condition run and
figures remain unchanged.

This directory contains the complete human-v2-v3-v3.1 ISC-CI validation and paper-simulation
reproduction. Prompt variants A, B, and C are separate conditions within each generation
version.

## Finalized V4

The completed October 2 run adds 20 four-seed V4 models: fixed-B and ensemble,
locked and calibrated, plus the investigator-approved ensemble source-only control.
The cumulative comparison has 52 trained models and four released references.
See `reports/v4_20261002/RESULTS.md` for results, uncertainty, and limitations,
`reports/v4_20261002/all_condition_summary.csv` for every condition/task, and
`V4_20261002_RUN.md` for decisions and reproduction commands.

All finalized inputs passed coverage/hash checks, with the disclosed exclusion of
64 columns containing 127 unrecoverable judgments. The fidelity gate passes, but
V4 completion does not uniformly improve human-behavior prediction. Input geometry,
shared semantic activations, and fully trained behavior remain separate results.
The earlier V3/V3.1 and October 1 partial behavior reports are preserved as history.

The reporting-only figure package is in `reports/v4_20261002/figures_final/`.
See `FINAL_FIGURES.md` for the one-command renderer, source-data exports, and
uncertainty definitions. It does not run judgments, training, or model inference.

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

- `HUMAN_BEHAVIOR_VALIDATION.md`: gated external human-behavior protocol, original
  measures, source discrepancies, uncertainty, and run commands.
- `run_human_behavior_validation.py`: shared downstream stage for all available
  models; refuses new training or candidate behavioral tests when fidelity fails.
- `configs/human_behavior_validation.json`: locked downstream configuration.

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

Source code, configurations, tests, and documentation are tracked. Generated artifacts,
reports, and the upstream checkout are excluded by the parent `.gitignore` (apart from
the existing V3.1 consolidation exception). Preserve or archive generated outputs before
cleanup; pushing the implementation does not upload checkpoints, datasets, or figures.
