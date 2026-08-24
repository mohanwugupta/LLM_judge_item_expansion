#!/usr/bin/env python3
"""Regenerate the V4 status/results report from immutable manifests and tables."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from leuven_expansion.v4 import sha256_file, write_json


ROOT = Path(__file__).resolve().parent


def read_json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def fmt(value: Any, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "not available"
    if isinstance(value, (float, int)):
        return f"{float(value):.{digits}f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "artifacts" / "v4")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    reports = run_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    discovery_manifest = read_json(run_dir / "discovery" / "candidate_bank_manifest.json")
    judgment_manifest = read_json(run_dir / "judgments" / "judgment_manifest.json")
    judgment_dry_run = read_json(run_dir / "judgments" / "dry_run_cost_report.json")
    threshold = read_json(run_dir / "judgments" / "judgment_threshold.json")
    matrix_manifest = read_json(run_dir / "matrices" / "matrix_manifest.json")
    validation_manifest = read_json(run_dir / "validation" / "manifest.json")
    retrieval = read_json(
        run_dir / "retrieval_efficiency" / "v2_retrospective" / "retrieval_selection.json"
    )
    cascade_path = (
        run_dir
        / "retrieval_efficiency"
        / "v2_retrospective"
        / "cascade_metrics_heldout.csv"
    )
    best_cascade: dict[str, Any] = {}
    if cascade_path.exists():
        cascade = pd.read_csv(cascade_path)
        eligible_cascade = cascade.loc[
            cascade["positive_recall"].ge(0.95)
            & cascade["object_geometry_correlation"].ge(0.99)
        ]
        selected_cascade = (
            eligible_cascade.sort_values("call_reduction", ascending=False).iloc[0]
            if not eligible_cascade.empty
            else cascade.sort_values("positive_recall", ascending=False).iloc[0]
        )
        best_cascade = selected_cascade.to_dict()
    prompt_c_cascade_path = (
        run_dir / "retrieval_efficiency" / "prompt_c_cascade" / "benchmark_summary.csv"
    )
    prompt_c_benchmarks: dict[str, dict[str, Any]] = {}
    if prompt_c_cascade_path.exists():
        prompt_c_frame = pd.read_csv(prompt_c_cascade_path)
        prompt_c_benchmarks = {
            str(row["benchmark"]): row
            for row in prompt_c_frame.to_dict(orient="records")
        }
    merge_review_path = run_dir / "discovery" / "candidate_merge_review_required.csv"
    pending_merges = 0
    if merge_review_path.exists():
        review = pd.read_csv(merge_review_path, dtype=str).fillna("")
        pending_merges = int((~review["verdict"].isin(["pass", "reject"])).sum())
    pending_discovery_sources: list[str] = []
    resolved_config = read_json(run_dir / "resolved_v4_config.json")
    if resolved_config:
        for source in resolved_config["discovery"]["sources"]:
            if not source.get("required_for_primary_freeze", False):
                continue
            source_dir = ROOT / source["path"]
            if not (
                (source_dir / "manifest.json").exists()
                and (source_dir / "generated_features_long.csv").exists()
            ):
                pending_discovery_sources.append(str(source["source_id"]))

    calibration_rows: list[dict[str, Any]] = []
    calibration_summary_path = run_dir / "judgments" / "calibration_summary.csv"
    if calibration_summary_path.exists():
        calibration = pd.read_csv(calibration_summary_path)
        calibration["selected"] = calibration["threshold_id"].eq(
            threshold["selected_rule"]["threshold_id"] if threshold else ""
        )
        calibration.to_csv(reports / "v4_calibration_report.csv", index=False)
        calibration_rows = calibration.to_dict(orient="records")
    else:
        pd.DataFrame().to_csv(reports / "v4_calibration_report.csv", index=False)

    matrix_comparison_path = run_dir / "matrices" / "context_inventory_comparison.csv"
    if matrix_comparison_path.exists():
        pd.read_csv(matrix_comparison_path).to_csv(
            reports / "v4_pruning_completion.csv", index=False
        )
    else:
        pd.DataFrame(
            columns=["matrix", "candidate_count_after_strict_gt_3"]
        ).to_csv(reports / "v4_pruning_completion.csv", index=False)

    validation_eval = run_dir / "validation" / "evaluation"
    representational_path = validation_eval / "input_matrix_rdm_comparisons.csv"
    if representational_path.exists():
        pd.read_csv(representational_path).to_csv(
            reports / "v4_representational_metrics.csv", index=False
        )
    else:
        pd.DataFrame().to_csv(reports / "v4_representational_metrics.csv", index=False)
    simulation_dir = run_dir / "validation" / "paper_simulations"
    behavioral_files = sorted(simulation_dir.glob("*_summary.csv"))
    behavioral = []
    for path in behavioral_files:
        frame = pd.read_csv(path)
        frame.insert(0, "source_table", path.name)
        behavioral.append(frame)
    (pd.concat(behavioral, ignore_index=True) if behavioral else pd.DataFrame()).to_csv(
        reports / "v4_behavioral_metrics.csv", index=False
    )
    source_rarefaction = run_dir / "discovery" / "candidate_rarefaction.csv"
    if source_rarefaction.exists():
        pd.read_csv(source_rarefaction).to_csv(
            reports / "v4_source_marginal_yield.csv", index=False
        )
    else:
        pd.DataFrame().to_csv(reports / "v4_source_marginal_yield.csv", index=False)
    pd.DataFrame(
        columns=["contrast", "estimate", "interval_low", "interval_high", "status"]
    ).to_csv(reports / "v4_primary_contrasts.csv", index=False)

    selected = threshold["selected_cross_validated_metrics"] if threshold else {}
    heldout = retrieval.get("heldout_test_metrics", {}) if retrieval else {}
    if judgment_manifest and judgment_manifest.get("complete"):
        atomic_status = "complete"
    elif judgment_dry_run and judgment_dry_run.get("existing_reusable_cells", 0):
        atomic_status = (
            f"partial: {int(judgment_dry_run['existing_reusable_cells']):,} valid cells "
            "reusable; cascade resume ready"
        )
    else:
        atomic_status = "pending cluster run"
    stage_status = {
        "candidate_bank": (
            "complete"
            if discovery_manifest
            else "blocked: primary generation pending"
            if pending_discovery_sources
            else f"blocked: {pending_merges} merge decisions pending"
        ),
        "atomic_judgments": atomic_status,
        "matrices": "complete" if matrix_manifest else "pending atomic judgments",
        "ISC-CI_training": "complete" if validation_manifest else "pending matrices",
        "paper_simulations": "complete" if (simulation_dir / "manifest.json").exists() else "pending trained models",
    }
    if discovery_manifest:
        discovery_summary = (
            f"Discovery is complete with {int(discovery_manifest['candidate_count']):,} "
            "frozen candidates."
        )
    elif pending_discovery_sources:
        discovery_summary = "The new seven-family discovery run is pending."
    else:
        discovery_summary = (
            f"The pooled candidate bank requires {pending_merges:,} merge decisions."
        )
    if judgment_dry_run and not judgment_manifest:
        judgment_summary = (
            f" The interrupted atomic run has "
            f"{int(judgment_dry_run.get('existing_reusable_cells', 0)):,} valid reusable "
            f"resolutions and {int(judgment_dry_run.get('remaining_cells', 0)):,} cells "
            "remaining."
        )
    else:
        judgment_summary = ""
    v2_prompt_c = prompt_c_benchmarks.get("v2_complete", {})
    v4_prompt_c = prompt_c_benchmarks.get("all_complete_v4_pilot", {})
    report = f"""# V4 Results and Reproduction Status

## 1. Executive interpretation

The primary V4 semantic and ISC-CI comparison is not yet estimable because atomic completion is still pending. {discovery_summary}{judgment_summary}

Calibration is frozen independently of V4 outcomes at `resolved_value >= {threshold['selected_rule']['value'] if threshold else 'pending'}`. On five held-out-word folds, this rule has positive recall {fmt(selected.get('positive_recall_mean'))}, precision {fmt(selected.get('positive_precision_mean'))}, MCC {fmt(selected.get('MCC_mean'))}, density {fmt(selected.get('matrix_density_mean'))}, and human input-object RDM correlation {fmt(selected.get('input_object_RDM_correlation_mean'))}. The executed V2 `resolved_value > 0` rule remains a mandatory control.

The V2 posthoc retrieval benchmark does not support an aggressive `K <= 100` shortlist. The development split selected `K={retrieval.get('selected_K') if retrieval else 'pending'}`; on untouched held-out features it retained {fmt(heldout.get('positive_cell_recall'))} of V2-positive cells, preserved object geometry at {fmt(heldout.get('object_geometry_correlation'))}, and reduced initial retrieval cells by only {fmt(heldout.get('initial_call_reduction'))}. Prompt-C cascading has been approved for the production resume, while embedding retrieval remains a posthoc analysis.

## 2. Stage status

| Stage | Status |
| --- | --- |
""" + "\n".join(f"| {stage} | {status} |" for stage, status in stage_status.items()) + f"""

## 3. Candidate discovery and consolidation

- Existing V3 and V3.1 generations are imported with source manifests.
- Seven configured V4 prompt families requested up to ten liberal candidate propositions per response.
- Valid singleton and rare phrases are retained; no V3 response-frequency cutoff is applied before judging.
- The exact ordered V3.1-B 175-context inventory is frozen as the fixed-vocabulary control.
- At the investigator's direction, consolidation was automated; {int(discovery_manifest.get('merge_passed', 0)) if discovery_manifest else 0:,} proposed merges were accepted. This deviation from manual review remains a sensitivity limitation.
- Frozen candidate inventory hash: `{discovery_manifest.get('candidate_inventory_hash', 'pending') if discovery_manifest else 'pending'}`.
- Pending primary discovery sources: {'none; the bank is frozen' if discovery_manifest else ', '.join(pending_discovery_sources) if pending_discovery_sources else 'none'}.
- Pending merge decisions: {0 if discovery_manifest else pending_merges:,}{'; all proposals were applied automatically' if discovery_manifest else ''}.

## 4. Atomic judgment and calibration

The primary condition remains exhaustive candidate-by-293-word screening with the existing V2 prompts, Qwen2.5-72B model identity, and no source provenance in calls. Every unresolved cell receives prompt C. Positive, ambiguous, confidence-below-.80, and parse-failed responses receive A/B and the frozen V2 resolver; other C negatives resolve to zero. Every final cell must have either the exact A/B/C panel or a valid unrouted C-only negative plus one resolution. Valid completed full-panel cells are reused unchanged.

Threshold selection used only 112,805 completed V2 cells for the 293 words and 385 retained human features. Five-fold splitting occurred by word. The selected threshold maximized mean held-out MCC subject to recall at least .80, with precision and conservatism as tie breakers.

## 5. Retrieval and cascade efficiency

Embedding retrieval is simulated only after the primary run. On the retrospective V2 benchmark, source words force human-positive recall to 1.0 by construction, so V2-positive recall and geometry are the discriminating criteria. `K=100` retained about 74% of V2 positives. The frozen escalation schedule required `K=275` to pass the 95% recall and .99 geometry gates on development features.

The held-out `K=275` result retained {fmt(heldout.get('positive_cell_recall'))} of V2 positives and used {fmt(heldout.get('shortlist_fraction'))} of cells. The expanded prompt-C analysis retained {fmt(v2_prompt_c.get('positive_cell_recall'), 4)} of V2 positives and {fmt(v4_prompt_c.get('positive_cell_recall'), 4)} in the V4 pilot; object geometry correlations were {fmt(v2_prompt_c.get('object_geometry_correlation'), 4)} and {fmt(v4_prompt_c.get('object_geometry_correlation'), 4)}. Matched ISC-CI runs and paper simulations are documented in `artifacts/v4/retrieval_efficiency/prompt_c_cascade/RESULTS.md`. The complete V4 matrix still requires a stratified negative audit.

## 6. Required V4 comparisons

Pending exhaustive judgments, matrices, four-seed training, fixed-probe evaluation, and unchanged paper simulations:

1. `V4_B_CALIBRATED - V3_1_B`: pruning and completion at fixed vocabulary.
2. `V4_ENSEMBLE_CALIBRATED - V4_B_CALIBRATED`: broad discovery.
3. `V4_ENSEMBLE_CALIBRATED - V2_ORIGINAL_SCHEMA`: generated versus human-seeded vocabulary.
4. `V4_ENSEMBLE_CALIBRATED - HUMAN_RETRAIN`: gap to the human self-ceiling.

## 7. Reproduction commands

```bash
python run_v4_judgments.py --candidate-bank artifacts/v4/discovery/candidate_bank.csv --leuven-words data/leuven_combined_features_consolidated.csv --v2-manifest artifacts/leuven_full_labels/leuven_full_v2/manifest.json --output-dir artifacts/v4/judgments --model Qwen2.5-72B-Instruct --shard-count 32 --execution-mode prompt-c-cascade --cascade-confidence-threshold 0.80 --dry-run
sbatch run_leuven_v4_atomic_smoke_test.sh
sbatch run_leuven_v4_atomic.sh
sbatch --dependency=afterok:<array_job_id> run_leuven_v4_atomic_finalize.sh
python calibrate_v4_judgments.py --v2-resolved artifacts/leuven_full_labels/leuven_full_v2/feature_resolutions.csv --human-features data/leuven_combined_features_consolidated.csv --output-dir artifacts/v4/judgments
python build_v4_matrices.py --candidate-bank artifacts/v4/discovery/candidate_bank.csv --resolved-values artifacts/v4/judgments/resolved_feature_values.csv --threshold artifacts/v4/judgments/judgment_threshold.json --output-dir artifacts/v4/matrices
python ISC-CI_LLM_validation/run_validation.py --config configs/v4_validation.json --output-dir artifacts/v4/validation --base-validation-dir ISC-CI_LLM_validation/artifacts/validation_v3_1 --include-v4
python ISC-CI_LLM_validation/evaluate_validation.py --config configs/v4_validation.json --validation-dir artifacts/v4/validation
python ISC-CI_LLM_validation/run_paper_simulations.py --config configs/v4_validation.json --validation-dir artifacts/v4/validation
python analyze_v4_retrieval_efficiency.py --config configs/v4_validation.json --candidate-bank artifacts/v4/discovery/candidate_bank.csv --human-mapping artifacts/v4/discovery/candidate_human_feature_mapping.csv --resolved-values artifacts/v4/judgments/resolved_feature_values.csv --votes artifacts/v4/judgments/feature_votes.csv --gold-rule calibrated --output-dir artifacts/v4/retrieval_efficiency/v4_posthoc
python summarize_v4_results.py --run-dir artifacts/v4
```

## 8. Limitations and DRM decision

DRM expansion is blocked until V4 completes the Leuven validation gates. The V2 efficiency benchmark may not generalize to a broader generated vocabulary, and its human-positive recall is inflated by forced original source words. Per-request tokens and wall time were not retained in V2, so call fractions are explicitly labeled as token, cost, and runtime proxies.
"""
    report_path = reports / "V4_RESULTS.md"
    report_path.write_text(report, encoding="utf-8")

    output_files = sorted(path for path in reports.iterdir() if path.name != "report_manifest.json")
    manifest = {
        "protocol_version": "v4-report-1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "stage_status": stage_status,
        "source_manifest_hashes": {
            "discovery": sha256_file(run_dir / "discovery" / "candidate_bank_manifest.json") if discovery_manifest else None,
            "judgment": sha256_file(run_dir / "judgments" / "judgment_manifest.json") if judgment_manifest else None,
            "judgment_dry_run": sha256_file(run_dir / "judgments" / "dry_run_cost_report.json") if judgment_dry_run else None,
            "threshold": sha256_file(run_dir / "judgments" / "judgment_threshold.json") if threshold else None,
            "matrices": sha256_file(run_dir / "matrices" / "matrix_manifest.json") if matrix_manifest else None,
            "validation": sha256_file(run_dir / "validation" / "manifest.json") if validation_manifest else None,
            "prompt_c_cascade": sha256_file(run_dir / "retrieval_efficiency" / "prompt_c_cascade" / "manifest.json") if (run_dir / "retrieval_efficiency" / "prompt_c_cascade" / "manifest.json").exists() else None,
        },
        "outputs": {path.name: sha256_file(path) for path in output_files},
        "calibration_candidate_count": len(calibration_rows),
    }
    write_json(reports / "report_manifest.json", manifest)
    print(report_path)


if __name__ == "__main__":
    main()
