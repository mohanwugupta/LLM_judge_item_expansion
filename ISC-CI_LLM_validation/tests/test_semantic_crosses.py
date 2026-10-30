from pathlib import Path

import pytest

from semantic_retraining.run_crosses import (
    all_pairs,
    phase_pairs,
    planned_jobs,
    execution_plan,
)


def test_complete_factorial_and_staged_counts():
    pairs = all_pairs()
    assert len(pairs) == len(set(pairs)) == 196
    assert len(phase_pairs(3)) == 14
    assert len(phase_pairs(4)) == 64
    assert len(phase_pairs(5)) == 132
    assert set(phase_pairs(4)).isdisjoint(phase_pairs(5))
    assert set(phase_pairs(4)) | set(phase_pairs(5)) == set(pairs)
    jobs = planned_jobs(Path("/tmp/fixture"))
    assert len(jobs["semantic"]) == 56 and len(jobs["context"]) == 784
    assert len({r["id"] for r in jobs["context"]}) == 784
    assert all(r["status"] == "planned" for rows in jobs.values() for r in rows)


def test_dry_plan_is_readonly_and_expensive_stages_expose_gate_failures(tmp_path):
    output = tmp_path / "not_created"
    plan = execution_plan(output, stage="semantic", phase=3, include_deferred=True)
    assert not output.exists()
    assert plan["config_sha256"]
    assert plan["blocked"] and plan["blockers"]
    assert len(plan["jobs"]) == 56
    assert max(j["memory"]["estimated_bytes"] for j in plan["jobs"]) > 100_000_000
    assert plan["scientific_artifacts_written"] is False
    with pytest.raises(ValueError, match="source"):
        execution_plan(output, stage="context", phase=5, semantic_source="released")


def test_full_training_configuration_is_canonical_before_checkpoint_signatures(
    tmp_path,
):
    import json
    from semantic_retraining.run_crosses import load_config

    config = load_config()
    config["semantic"]["full"] = {"max_epochs": 1000, "learning_rate": 0.01}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    assert load_config(path)["semantic"]["full"]["chunk_size"] == 4096


def test_context_memory_estimate_uses_actual_cico_architecture_not_isc_output_head(
    tmp_path,
):
    from iscci_validation.modeling import CICOModel

    model = CICOModel(294, 385)
    job = planned_jobs(tmp_path)["context"][0]
    assert job["memory"]["parameters"] == sum(p.numel() for p in model.parameters())


def test_default_cohort_defers_only_large_ensemble_sources(tmp_path):
    from semantic_retraining.run_crosses import load_config, source_cohort

    config = load_config()
    active = source_cohort(config)
    deferred = {"v4_ensemble_locked_v2", "v4_ensemble_calibrated"}
    assert len(active) == 12 and not deferred.intersection(active)
    assert {
        "human",
        "random_human",
        "v2",
        "v4_b_locked_v2",
        "v4_b_calibrated",
        "v4_ensemble_source_only",
    }.issubset(active)
    assert len(execution_plan(tmp_path, "semantic", 3, config=config)["jobs"]) == 48
    phases = [
        execution_plan(tmp_path, "context", phase, config=config) for phase in [4, 5]
    ]
    assert [len(p["jobs"]) for p in phases] == [216, 360]
    assert all(
        j["semantic_source"] in active and j["context_source"] in active
        for p in phases
        for j in p["jobs"]
    )
    registry = planned_jobs(tmp_path, config=config)
    assert sum(j["status"] == "deferred" for j in registry["semantic"]) == 8
    assert sum(j["status"] == "deferred" for j in registry["context"]) == 208


def test_deferred_source_requires_deliberate_opt_in(tmp_path):
    from semantic_retraining.run_crosses import load_config

    config = load_config()
    with pytest.raises(ValueError, match="deferred"):
        execution_plan(
            tmp_path, "semantic", 3, condition="v4_ensemble_locked_v2", config=config
        )
    plan = execution_plan(
        tmp_path,
        "semantic",
        3,
        condition="v4_ensemble_locked_v2",
        config=config,
        include_deferred=True,
    )
    assert len(plan["jobs"]) == 4
    assert (
        len(
            execution_plan(
                tmp_path, "context", 5, config=config, include_deferred=True
            )["jobs"]
        )
        == 528
    )


@pytest.mark.parametrize(
    "deferred", [["human"], ["random_human"], ["v3_1_B"], ["typo"], ["v2", "v2"]]
)
def test_invalid_deferral_cannot_silently_change_reference_design(deferred):
    from semantic_retraining.run_crosses import load_config, source_cohort

    config = load_config()
    config["deferred_conditions"] = deferred
    with pytest.raises(ValueError):
        source_cohort(config)
