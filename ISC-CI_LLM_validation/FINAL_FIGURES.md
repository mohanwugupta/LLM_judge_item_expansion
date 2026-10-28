# Frozen Final-Results Figures

For the cumulative 14-condition figures including `random_human`, see
`RANDOM_FEATURE_CONTROL.md` and
`reports/v4_20261028_random_control/figures_final/INDEX.md`.
The default command below intentionally still reproduces the original frozen
13-condition package. `--include-random-control` requires the completed extension
tables and a new output directory; it never performs training or model inference.

Render the completed October 2, 2026 experiment without training, checkpoint
inference, LLM calls, model simulations, or loading the original judgment payload.

From the project root:

```bash
python ISC-CI_LLM_validation/plot_finalized_v4_results.py
```

On this workstation the verified interpreter is
`/Users/mg9965/.pyenv/versions/3.11.8/bin/python3`. Dependencies are `numpy`,
`pandas`, `scipy`, and `matplotlib` (verified with Matplotlib 3.10.5). No Torch,
checkpoint, API credentials, GPU, or cluster job is needed.

## Inputs And Outputs

Defaults read:

- `artifacts/human_behavior_validation_v4_20261002/`: authoritative protocol 1.1.0,
  completed manifest, summary/per-seed tables, lower-level similarity predictions,
  reliability, paired contrasts, and `per_model/*_seed_*/` item/participant CSVs.
- `artifacts/validation_v4_20261002/data_matrices/`: thirteen finalized **retained**
  binary matrices. These are not the giant original judgments.
- `reports/v4_20261002/report_manifest.json` and `RESULTS.md`: frozen report context.
- Original `data/leuven_dataset/leuven_categories.csv` from the adjacent paper
  repository, for performance-independent category/file ordering. Without that
  file, frozen human-matrix order is documented and used instead.

The input directories can be overridden with `--behavior-dir`, `--validation-dir`,
`--report-dir`, and `--categories`. Set `--output-dir` to a new report directory;
the default is `reports/v4_20261002/figures_final/`, **not** an artifact directory.
Inputs and figure sources are hashed before/after rendering. Finalized matrices
are parsed once and compact derived RDM/count caches are saved under output
`cache/`, keyed by the matrix file hash. No original matrix is modified.

Missing core tables or a failed fidelity gate cause immediate failure. Missing
panel-specific input files block the affected group, are recorded in
`figure_manifest.json`, and do not stop available groups from rendering. The
default exit status remains nonzero if any required group is blocked. Explicit
`--allow-missing` permits partial reporting with the gaps still documented. No
historical-protocol result substitutes for missing authoritative results.

## Package Contents

Four main figures are split into readable panels; mandatory all-condition,
representation, contrast, asymmetry, RDM, and coverage supplements accompany them.
Every figure is exported as editable SVG, PDF, and 300-dpi PNG. Every panel has a
CSV in `source_data/`; scatter sources retain human observations, original
stimulus columns, stable item IDs, and all four seed predictions. The complete
all-condition summary, context-effect table, captions, interpretation, and source
manifest are in the same output directory.

The common-finite-pair sensitivity reuses saved predictions and computes only
Pearson correlations on intersected item IDs. It generates no new uncertainty
intervals. The unchanged primary scores and saved intervals remain separate.
Replicate SD, reliability split-half distributions, and model bootstrap intervals
are explicitly different quantities. The shared pretrained encoder appears once
per domain. V3-B is design-matched, not selected on performance.

Run focused tests from the project root:

```bash
python -m pytest -q ISC-CI_LLM_validation/tests/test_final_results_figures.py
```

Visual QA uses rendered PNGs and independent PDF rasterization; see
`figures_final/visual_qa.json` for the completed inspection record. The output
directory remains covered by the project's validation-folder ignore policy.

Independent render checks (optional QA dependencies: `PyMuPDF`, `Pillow`,
`resvg-py==0.5.0`; these are not required by the plotting command):

```bash
python ISC-CI_LLM_validation/qa_final_results_figures.py ISC-CI_LLM_validation/reports/v4_20261002/figures_final
```

This checks PDF text bounds, SVG XML, independent nonblank PDF/SVG rasters, and
300-dpi PNGs, and creates contact sheets under `qa/`. The workstation QA used an
isolated `resvg-py` install in `/private/tmp/iscci-svg-qa`, passed with
`--svg-renderer-path /private/tmp/iscci-svg-qa`; the user's environment was not
changed. Inspect the current images before using
`--record-reviewed`; automated format checks alone cannot establish visual clarity.
