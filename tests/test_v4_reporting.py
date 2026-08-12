import json
import sys

import pandas as pd

from summarize_v4_results import main as summarize_main


def test_partial_report_is_reproducible_and_labels_blocked_stages(tmp_path, monkeypatch):
    run = tmp_path / "v4"
    (run / "judgments").mkdir(parents=True)
    (run / "discovery").mkdir()
    (run / "retrieval_efficiency" / "v2_retrospective").mkdir(parents=True)
    threshold = {
        "selected_rule": {"threshold_id": "ge_1", "value": 1},
        "selected_cross_validated_metrics": {
            "positive_recall_mean": 0.85,
            "positive_precision_mean": 0.25,
            "MCC_mean": 0.39,
            "matrix_density_mean": 0.25,
            "input_object_RDM_correlation_mean": 0.82,
        },
    }
    (run / "judgments" / "judgment_threshold.json").write_text(json.dumps(threshold))
    pd.DataFrame(
        {"threshold_id": ["ge_1"], "positive_recall_mean": [0.85]}
    ).to_csv(run / "judgments" / "calibration_summary.csv", index=False)
    pd.DataFrame(
        {
            "merge_candidate_id": ["m1"],
            "verdict": [""],
        }
    ).to_csv(run / "discovery" / "candidate_merge_review_required.csv", index=False)
    selection = {
        "selected_K": 275,
        "heldout_test_metrics": {
            "positive_cell_recall": 0.98,
            "object_geometry_correlation": 0.99,
            "initial_call_reduction": 0.05,
            "shortlist_fraction": 0.95,
        },
    }
    (run / "retrieval_efficiency" / "v2_retrospective" / "retrieval_selection.json").write_text(
        json.dumps(selection)
    )
    monkeypatch.setattr(sys, "argv", ["summarize_v4_results.py", "--run-dir", str(run)])
    summarize_main()
    report = (run / "reports" / "V4_RESULTS.md").read_text()
    assert "not yet estimable" in report
    assert "1 merge decisions pending" in report
    assert "K=275" in report
    assert (run / "reports" / "report_manifest.json").exists()

