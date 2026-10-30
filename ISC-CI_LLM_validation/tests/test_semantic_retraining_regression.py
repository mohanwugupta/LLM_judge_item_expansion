import json

import pytest

from semantic_retraining.gates import verify_sources, verify_gate, saved_path
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


def test_portable_frozen_paths_prefer_current_checkout(tmp_path):
    original = tmp_path / "old/ISC-CI_LLM_validation/reports/input.csv"
    current = tmp_path / "clone/ISC-CI_LLM_validation"
    original.parent.mkdir(parents=True)
    original.write_text("stale original checkout")
    portable = current / "reports/input.csv"
    portable.parent.mkdir(parents=True)
    portable.write_text("current checkout")
    assert saved_path(str(original), current) == portable


def test_external_reference_prefers_byte_exact_packaged_copy(tmp_path):
    root = tmp_path / "ISC-CI_LLM_validation"
    relative = "IntegratedSemanticsControlContextInference/data/categories.csv"
    reference = root / "inputs/reference_sources" / relative
    reference.parent.mkdir(parents=True)
    reference.write_bytes(b"category\r\n")
    upstream = root / "upstream" / relative
    upstream.parent.mkdir(parents=True)
    upstream.write_bytes(b"category\n")
    assert saved_path("/old/" + relative, root) == reference


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


@pytest.mark.parametrize("failure", ["tests", "preservation", "preservation_error"])
def test_preflight_failure_exposes_output_and_keeps_gate_failed(
    tmp_path, monkeypatch, capsys, failure
):
    from types import SimpleNamespace
    import semantic_retraining.gates as gates

    calls = []

    def history(*args):
        calls.append(True)
        if len(calls) == 2 and failure == "preservation_error":
            raise ValueError("Frozen input changed: frozen.csv")
        return {
            "frozen.csv": (
                "new" if len(calls) == 2 and failure == "preservation" else "original"
            )
        }

    monkeypatch.setattr(gates, "verify_history", history)
    monkeypatch.setattr(gates, "gate_signature", lambda *a: {"code": "fixture"})
    monkeypatch.setattr(gates, "sha256", lambda *a: "fixture")
    monkeypatch.setattr(
        gates.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(
            returncode=1 if failure == "tests" else 0,
            stdout=(
                "FAILED test_example: actionable test explanation"
                if failure == "tests"
                else "tests passed"
            ),
            stderr="dependency details" if failure == "tests" else "",
        ),
    )
    output = tmp_path / "output"
    with pytest.raises(ValueError) as error:
        gates.preflight(
            tmp_path, output, {}, {"frozen_behavior_manifest": "history.json"}
        )
    path = output / "gates/preflight.json"
    assert str(path) in str(error.value)
    record = json.loads(path.read_text())
    assert record["passed"] is False
    assert record["tests_passed"] == (failure != "tests")
    assert record["preservation_passed"] == (failure == "tests")
    logs = capsys.readouterr().err
    if failure == "tests":
        assert "actionable test explanation" in logs and "dependency details" in logs
    elif failure == "preservation":
        assert "frozen.csv" in logs
        assert record["preservation_changes"] == [
            {"path": "frozen.csv", "before": "original", "after": "new"}
        ]
    else:
        assert "Frozen input changed" in logs
        assert record["preservation_error"] == "Frozen input changed: frozen.csv"
