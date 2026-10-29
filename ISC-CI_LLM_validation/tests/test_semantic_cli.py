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
    assert plan["blocked"] and len(plan["jobs"]) == 56
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
