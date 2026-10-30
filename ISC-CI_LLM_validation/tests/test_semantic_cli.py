import json

import pytest

from semantic_retraining.cli import main


def test_dry_run_does_not_make_output_directory_or_model_calls(tmp_path, capsys):
    destination = tmp_path / "not_created"
    assert (
        main(
            [
                "--stage",
                "semantic",
                "--phase",
                "3",
                "--dry-run",
                "--output-dir",
                str(destination),
            ]
        )
        == 0
    )
    plan = json.loads(capsys.readouterr().out)
    assert plan["blocked"] and len(plan["jobs"]) == 48
    assert not destination.exists()


def test_workers_and_real_training_gates_fail_before_starting(tmp_path, capsys):
    with pytest.raises(SystemExit):
        main(["--workers", "0"])
    assert (
        main(
            [
                "--stage",
                "semantic",
                "--phase",
                "3",
                "--output-dir",
                str(tmp_path / "blocked"),
            ]
        )
        == 2
    )
    assert not (tmp_path / "blocked").exists()


def test_synthetic_smoke_is_explicitly_not_a_completed_human_model(tmp_path, capsys):
    assert main(["--stage", "smoke", "--output-dir", str(tmp_path / "smoke")]) == 0
    record = json.loads((tmp_path / "smoke/gates/smoke.json").read_text())
    assert record["passed"] and record["synthetic_only"]
    assert not (tmp_path / "smoke/isc_models/human").exists()


def test_entire_requested_cohort_is_checked_before_any_job_starts(
    tmp_path, monkeypatch, capsys
):
    import semantic_retraining.cli as cli

    jobs = [
        {
            "semantic_source": condition,
            "seed": 0,
            "directory": str(tmp_path / condition),
        }
        for condition in ["human", "v3_1_B"]
    ]
    monkeypatch.setattr(
        cli, "execution_plan", lambda *args, **kwargs: {"blocked": False, "jobs": jobs}
    )
    monkeypatch.setattr(cli, "require_training_gates", lambda *args: None)

    def validate(output, condition, path, taxonomy):
        if condition == "v3_1_B":
            raise ValueError("Missing reviewed V3.1-B mapping")
        return {}, "fixture"

    monkeypatch.setattr(cli, "verify_frozen_mapping", validate)
    started = []

    def run(job, *args):
        started.append(job)
        return {"complete": True, "status": "complete"}

    monkeypatch.setattr(cli, "execute_job", run)
    assert (
        main(
            [
                "--stage",
                "semantic",
                "--phase",
                "2",
                "--output-dir",
                str(tmp_path / "output"),
            ]
        )
        == 2
    )
    assert not started


def test_unconverged_stage_returns_failure_to_cluster_scheduler(
    tmp_path, monkeypatch, capsys
):
    import semantic_retraining.cli as cli

    jobs = [
        {"semantic_source": "human", "seed": 0, "directory": str(tmp_path / "model")}
    ]
    monkeypatch.setattr(
        cli, "execution_plan", lambda *a, **k: {"blocked": False, "jobs": jobs}
    )
    monkeypatch.setattr(cli, "require_training_gates", lambda *a: None)
    monkeypatch.setattr(cli, "verify_frozen_mapping", lambda *a: ({}, "fixture"))
    monkeypatch.setattr(
        cli, "execute_job", lambda *a: {"complete": False, "status": "not_converged"}
    )
    assert (
        main(
            [
                "--stage",
                "semantic",
                "--phase",
                "2",
                "--output-dir",
                str(tmp_path / "out"),
            ]
        )
        == 2
    )


@pytest.mark.parametrize("stage", ["classify", "freeze"])
def test_default_classification_and_freeze_do_not_require_large_ensembles(
    tmp_path, monkeypatch, capsys, stage
):
    import semantic_retraining.cli as cli

    directory = tmp_path / "classification"
    directory.mkdir()
    (directory / "model_fingerprint.json").write_text('{"revision":"fixture"}')
    monkeypatch.setattr(cli, "verify_gate", lambda *a: None)
    monkeypatch.setattr(cli, "gate_signature", lambda *a: {})
    selected = []

    def classify(*args):
        selected.extend(args[-1])
        return {"passed": True}

    def freeze(*args):
        selected.extend(args[-1])
        return {"passed": True}

    monkeypatch.setattr(cli, "classify_catalog", classify)
    monkeypatch.setattr(cli, "freeze_catalog", freeze)
    assert main(["--stage", stage, "--output-dir", str(tmp_path)]) == 0
    assert len(selected) == 12
    assert (
        "v4_ensemble_locked_v2" not in selected
        and "v4_ensemble_calibrated" not in selected
    )
    assert "v4_ensemble_source_only" in selected


def test_phase_five_evaluation_does_not_reintroduce_deferred_models(
    tmp_path, monkeypatch, capsys
):
    import semantic_retraining.cli as cli
    import semantic_retraining.evaluate_crosses as evaluation

    monkeypatch.setattr(cli, "require_training_gates", lambda *a: None)
    selected = []

    def evaluate(root, output, jobs, config, phase):
        selected.extend(jobs)
        return {"complete": True}

    monkeypatch.setattr(evaluation, "evaluate_context_models", evaluate)
    assert (
        main(["--stage", "evaluate", "--phase", "5", "--output-dir", str(tmp_path)])
        == 0
    )
    assert len(selected) == 576
    assert all(
        "v4_ensemble_locked_v2" not in (j["semantic_source"], j["context_source"])
        for j in selected
    )
