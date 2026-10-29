import json

import pytest

from semantic_retraining.gates import verify_sources, verify_gate
from semantic_retraining.inventory import sha256
from semantic_retraining.provenance import validate_output_directory


def test_preservation_gate_rejects_any_changed_input(tmp_path):
    source = tmp_path / "unchanged.csv"
    source.write_text("original\n")
    records = [{"path": str(source), "sha256": sha256(source)}]
    assert verify_sources(records)[str(source)] == records[0]["sha256"]
    source.write_text("changed\n")
    with pytest.raises(ValueError, match="changed"):
        verify_sources(records)


def test_stale_or_failed_preflight_cannot_release_training(tmp_path):
    path = tmp_path / "gate.json"
    path.write_text(json.dumps({"passed": True, "signature": {"code": "old"}}))
    with pytest.raises(ValueError, match="signature"):
        verify_gate(path, {"code": "new"})
    path.write_text(json.dumps({"passed": False, "signature": {"code": "new"}}))
    with pytest.raises(ValueError, match="passed"):
        verify_gate(path, {"code": "new"})


def test_existing_artifact_paths_are_protected_before_writing():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for path in [
        root / "artifacts/validation_v4_20261002/models/new_model",
        root / "artifacts/random_feature_control_20261028/validation/models/new_model",
        root / "upstream/IntegratedSemanticsControl",
    ]:
        with pytest.raises(ValueError, match="Protected"):
            validate_output_directory(path)


def test_plotting_only_changes_do_not_invalidate_semantic_training_provenance(
    monkeypatch,
):
    from semantic_retraining.provenance import code_hashes
    from semantic_retraining import inventory

    original = inventory.sha256
    training = code_hashes("semantic")
    everything = code_hashes()
    monkeypatch.setattr(
        inventory,
        "sha256",
        lambda path: (
            "changed_plot" if path.name == "report_results.py" else original(path)
        ),
    )
    assert code_hashes("semantic") == training
    assert code_hashes() != everything
