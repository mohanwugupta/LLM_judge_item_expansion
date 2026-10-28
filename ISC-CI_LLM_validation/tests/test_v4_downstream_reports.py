import json

import numpy as np
import pytest

import resume_finalized_v4_analysis as resume
import summarize_finalized_v4 as summarize
from audit_v4_feature_profiles import profile_summary


def test_profile_audit_is_descriptive_and_uses_frozen_retention():
    matrix = np.array(
        [[1, 1, 1], [1, 1, 1], [1, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=bool
    )
    original = matrix.copy()
    summary = profile_summary("example", matrix)
    assert summary["retained_contexts"] == 2
    assert summary["distinct_object_membership_profiles"] == 1
    assert summary["extra_columns_with_repeated_profile"] == 1
    assert summary["largest_profile_multiplicity"] == 2
    assert summary["positive_cells"] == 8
    np.testing.assert_array_equal(matrix, original)


def test_downstream_wait_rejects_copied_baseline_and_missing_seeds(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(resume, "VALIDATION", tmp_path)
    manifest = {
        "protocol_version": "iscci-v3-1-validation-1.0.0",
        "selected_conditions": list(resume.REQUIRED),
        "selected_seeds": [0, 1, 2, 3],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    assert not resume.training_ready()
    manifest["protocol_version"] = "iscci-v4-validation-1.0.0"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    assert not resume.training_ready()
    for condition in resume.REQUIRED:
        for seed in range(4):
            directory = tmp_path / "models" / condition / f"seed_{seed}"
            directory.mkdir(parents=True)
            (directory / "checkpoint.pt").touch()
    assert resume.training_ready()


def test_final_report_refuses_partial_or_failed_behavior(tmp_path, monkeypatch):
    monkeypatch.setattr(summarize, "BEHAVIOR", tmp_path)
    for complete, passed in [(False, True), (True, False)]:
        (tmp_path / "manifest.json").write_text(
            json.dumps({"complete": complete, "fidelity_gate": {"passed": passed}})
        )
        with pytest.raises(ValueError, match="Final report requires"):
            summarize.main()
