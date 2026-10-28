import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from iscci_validation.behavior import (
    cross_validated_mapping,
    item_table,
    load_tasks,
    metrics_for_items,
    notebook_reference,
    paired_condition_contrasts,
    predict_task,
    summarize_behavior,
    trial_statistics,
)
from iscci_validation.behavior_fidelity import fidelity_gate, item_fidelity
from iscci_validation.modeling import CICOModel
from iscci_validation.simulations import (
    _lca_choice_probabilities,
    model_similarity_context_effect,
    model_similarity_context_predictions,
)
from run_human_behavior_validation import input_geometry, registry

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "upstream/IntegratedSemanticsControlContextInference"


def test_original_data_and_reference_are_not_new_benchmarks():
    tasks = load_tasks(UPSTREAM / "data")
    assert len(tasks["induction"]) == 1067
    assert len(tasks["similarity"]) == 54888
    assert "Human" not in tasks["phenomena"]
    references = notebook_reference(UPSTREAM)
    assert len(references) == 17
    assert references.query("task == 'context'")["reference"].iloc[0] == 37 / 38


def test_item_probe_matches_unedited_original():
    names = ["a", "b", "c", "d"]
    model = CICOModel(input_d=5, task_output_d=2)
    arguments = pd.DataFrame(
        {
            "Premise 1": ["a", "a", "a"],
            "Premise 2": [np.nan, "b", "b"],
            "Premise 3": [np.nan, np.nan, "c"],
            "Conclusion": ["d"] * 3,
        }
    )
    result = item_fidelity(
        model, {"induction": arguments, "phenomena": arguments}, names, UPSTREAM, 1e-5
    )
    assert result["passed"].all()


def test_cv_never_splits_related_argument_clusters():
    rows = [
        {
            "task": "induction",
            "dataset": "dataset",
            "cluster_id": str(i // 2),
            "human": float(i),
            "prediction": float(i),
        }
        for i in range(20)
    ]
    frame = cross_validated_mapping(pd.DataFrame(rows), 5)
    assert frame.groupby("cluster_id")["cv_fold"].nunique().max() == 1
    assert np.allclose(frame["cv_prediction"], frame["human"])
    altered = pd.DataFrame(rows)
    held = frame["cv_fold"] == 0
    altered.loc[held, "human"] += 1000
    result = cross_validated_mapping(altered, 5)
    assert np.allclose(
        result.loc[held, "cv_prediction"], frame.loc[held, "cv_prediction"]
    )


def test_fidelity_detects_model_graph_drift(monkeypatch):
    model = CICOModel(input_d=3, task_output_d=2)
    original_output = model.get_ft_output
    monkeypatch.setattr(
        model, "get_ft_output", lambda x, context: original_output(x, context) + 1
    )
    arguments = pd.DataFrame(
        {
            "Premise 1": ["a"],
            "Premise 2": [np.nan],
            "Premise 3": [np.nan],
            "Conclusion": ["b"],
        }
    )
    result = item_fidelity(
        model,
        {"induction": arguments, "phenomena": arguments},
        ["a", "b"],
        UPSTREAM,
        1e-5,
    )
    assert not result["passed"].any()


def test_asymmetry_saves_unselected_original_trials():
    data = pd.DataFrame(
        {
            "Argument 1-Premise 1": ["a"] * 4,
            "Argument 1-Premise 2": [np.nan] * 4,
            "Argument 1-Conclusion": ["b"] * 4,
            "Argument 2-Premise 1": ["b"] * 4,
            "Argument 2-Premise 2": [np.nan] * 4,
            "Argument 2-Conclusion": ["a"] * 4,
            "Argument 2 Chosen": [0, 1, 0, 1],
            "participant_id": [1, 2, 3, 4],
            "source_row": ["0", "1", "2", "3"],
        }
    )
    model = CICOModel(input_d=3, task_output_d=2)
    items, trials = predict_task("asymmetry", data, model, ["a", "b"], 123)
    assert not items["paper_selected"].any()
    assert len(items) == 1 and len(trials) == 4
    assert trials["prediction"].notna().all()


def test_context_detail_preserves_old_effect_and_common_random_numbers():
    model = CICOModel(input_d=6, task_output_d=2)
    row = pd.Series(
        dict(
            zip(
                [
                    "Premise 1",
                    "Conclusion 1",
                    "Conclusion 2",
                    "Distractor 1",
                    "Distractor 2",
                ],
                list("abcde"),
            )
        )
    )
    predictions = model_similarity_context_predictions(row, model, list("abcde"), 9)
    assert predictions["model_effect"] == model_similarity_context_effect(
        row, model, list("abcde"), 9
    )
    assert predictions == model_similarity_context_predictions(
        row, model, list("abcde"), 9
    )


def test_lca_matches_scalar_equations_with_identical_draws():
    drift = np.array([[0.8, 0.1, 0.1], [0.3, 0.5, 0.2]])
    simulations, steps, burn_in = 4, 20, 5
    rng = np.random.default_rng(8)
    states = np.zeros((simulations, 3))
    counts = np.zeros(3, int)
    for step in range(steps):
        contexts = rng.integers(0, len(drift), simulations)
        noise = rng.normal(0, 0.2, (simulations, 3))
        for sim in range(simulations):
            a1, a2, a3 = drift[contexts[sim]]
            i1, i2, i3 = (
                max(0, a1 - a2 - a3),
                max(0, a2 - a1 - a3),
                max(0, a3 - a1 - a2),
            )
            A1, A2, A3 = states[sim]
            A1 = max(0, 0.94 * A1 + 0.06 * (i1 - 0.6 * (A2 + A3) + noise[sim, 0]))
            A2 = max(0, 0.94 * A2 + 0.06 * (i2 - 0.6 * (A1 + A3) + noise[sim, 1]))
            A3 = max(0, 0.94 * A3 + 0.06 * (i3 - 0.6 * (A1 + A2) + noise[sim, 2]))
            states[sim] = [A1, A2, A3]
        if step >= burn_in:
            counts += np.bincount(states.argmax(1), minlength=3)
    expected = counts / counts.sum()
    actual = _lca_choice_probabilities(
        drift, np.random.default_rng(8), simulations, steps, burn_in
    )
    assert np.array_equal(actual, expected)


def predictions_fixture():
    rows = []
    for condition in ("a", "b"):
        for seed in range(4):
            for i in range(8):
                rows.append(
                    {
                        "condition": condition,
                        "seed": seed,
                        "task": "similarity",
                        "dataset": "x",
                        "item_id": str(i),
                        "cluster_id": str(i),
                        "human": float(i),
                        "prediction": float(i),
                        "cv_prediction": float(i),
                        "paper_selected": True,
                    }
                )
    return pd.DataFrame(rows)


def test_paired_bootstrap_zero_difference_for_identical_predictions():
    predictions = predictions_fixture()
    seed_metrics, summary = summarize_behavior(predictions, 10, 1)
    assert len(seed_metrics) == 8
    assert set(summary["seed_count"]) == {4}
    contrasts = paired_condition_contrasts(predictions, [("a", "b")], 10, 1)
    assert (contrasts[["difference_a_minus_b", "ci_low", "ci_high"]] == 0).all().all()


def test_bootstrap_seed_cache_preserves_estimates_and_intervals(monkeypatch):
    import iscci_validation.behavior as behavior

    predictions = predictions_fixture()
    predictions["prediction"] += np.sin(predictions["seed"] + predictions["human"]) / 3
    predictions["cv_prediction"] = predictions["prediction"] + 0.2
    predictions.loc[predictions["condition"].eq("b"), "prediction"] += 0.4
    cached_seed, cached_summary = summarize_behavior(predictions, 30, 71)
    cached_contrasts = paired_condition_contrasts(predictions, [("a", "b")], 30, 71)

    def uncached(task, arrays, indices, seeds):
        return [
            behavior.metrics_for_values(
                task,
                *[values[indices] for values in arrays[int(seed)]],
                include_tests=False
            )
            for seed in seeds
        ]

    monkeypatch.setattr(behavior, "bootstrap_seed_metrics", uncached)
    original_seed, original_summary = summarize_behavior(predictions, 30, 71)
    original_contrasts = paired_condition_contrasts(predictions, [("a", "b")], 30, 71)
    pd.testing.assert_frame_equal(cached_seed, original_seed)
    pd.testing.assert_frame_equal(cached_summary, original_summary)
    pd.testing.assert_frame_equal(cached_contrasts, original_contrasts)


def test_gate_cannot_pass_when_reconstructed_models_or_sources_disagree():
    references = notebook_reference(UPSTREAM)
    rows = []
    for condition, seeds in (("human", range(4)), ("released", [3])):
        for seed in seeds:
            for (task, dataset), group in references.groupby(["task", "dataset"]):
                rows.append(
                    {
                        "condition": condition,
                        "seed": seed,
                        "task": task,
                        "dataset": dataset,
                        **dict(zip(group["metric"], group["reference"])),
                    }
                )
    config = json.loads((ROOT / "configs/human_behavior_validation.json").read_text())[
        "fidelity"
    ]
    config["reconstructed_criterion"] = "absolute_numerical_tolerance"
    config["require_pdf_notebook_context_consistency"] = True
    config["require_exact_stochastic_notebook_reproduction"] = True
    checks, gate = fidelity_gate(
        UPSTREAM, pd.DataFrame(rows), pd.DataFrame({"passed": [True]}), config
    )
    assert not gate["passed"]
    assert gate["failed_required_statistic_checks"] == 1
    config["require_pdf_notebook_context_consistency"] = False
    checks, gate = fidelity_gate(
        UPSTREAM, pd.DataFrame(rows), pd.DataFrame({"passed": [True]}), config
    )
    assert gate["passed"]
    rows[0]["paper_statistic"] += 0.01
    assert not fidelity_gate(
        UPSTREAM, pd.DataFrame(rows), pd.DataFrame({"passed": [True]}), config
    )[1]["passed"]


def test_registry_rejects_fractional_matrix_before_integer_cast(tmp_path):
    directory = tmp_path / "data_matrices"
    directory.mkdir()
    human = pd.DataFrame({"f": [1, 0]}, index=["a", "b"])
    pd.DataFrame({"f": [0.9, 0]}, index=["a", "b"]).to_csv(directory / "v2.csv")
    with pytest.raises(ValueError, match="Nonbinary"):
        registry([tmp_path], ["human", "v2"], human, tmp_path / "models")


def test_statistical_gate_requires_four_reference_runs_and_rejects_outside_envelope():
    references = notebook_reference(UPSTREAM)
    rows = []
    for condition in ("human", "released"):
        for seed in range(4):
            for (task, dataset), group in references.groupby(["task", "dataset"]):
                rows.append(
                    {
                        "condition": condition,
                        "seed": seed,
                        "task": task,
                        "dataset": dataset,
                        **dict(zip(group["metric"], group["reference"])),
                    }
                )
    config = json.loads((ROOT / "configs/human_behavior_validation.json").read_text())[
        "fidelity"
    ]
    data = pd.DataFrame(rows)
    items = pd.DataFrame({"passed": [True]})
    assert fidelity_gate(UPSTREAM, data, items, config)[1]["passed"]
    data.loc[0, "paper_statistic"] += 0.01
    assert not fidelity_gate(UPSTREAM, data, items, config)[1]["passed"]
    data = data[~((data["condition"] == "released") & (data["seed"] == 0))]
    assert not fidelity_gate(UPSTREAM, data, items, config)[1]["passed"]


def test_individual_trial_error_compares_to_actual_responses():
    trials = pd.DataFrame(
        {
            "task": ["thematic"] * 2,
            "dataset": ["thematic"] * 2,
            "observed": [0.0, 1.0],
            "cv_prediction": [0.5, 0.5],
            "model_choice": [False, False],
        }
    )
    result = trial_statistics(trials).iloc[0]
    assert result["individual_choice_agreement"] == 0.5
    assert result["cv_brier"] == 0.25


def test_induction_individual_error_survives_mixed_task_concatenation():
    induction = pd.DataFrame(
        {
            "task": ["induction"] * 2,
            "dataset": ["x"] * 2,
            "Human": [1.0, 3.0],
            "cv_prediction": [2.0, 2.0],
        }
    )
    choice = pd.DataFrame(
        {
            "task": ["thematic"],
            "dataset": ["thematic"],
            "observed": [1.0],
            "cv_prediction": [0.5],
            "model_choice": [True],
        }
    )
    result = trial_statistics(pd.concat([induction, choice], ignore_index=True))
    assert result[result["task"] == "induction"]["cv_trial_rmse"].iloc[0] == 1.0


def test_input_geometry_preserves_original_zero_vector_convention():
    from iscci_validation.evaluation import matrix_rdm_comparisons

    human = pd.DataFrame([[1, 0], [1, 1], [0, 1], [1, 0]], dtype=np.int8)
    candidate = pd.DataFrame([[0, 0], [1, 1], [0, 1], [1, 0]], dtype=np.int8)
    result = input_geometry(candidate, human)
    expected = matrix_rdm_comparisons({"human": human, "candidate": candidate})
    assert np.isclose(
        result["input_object_rdm_spearman"], expected["object_rdm_spearman"].iloc[0]
    )
    assert result["input_objects_without_features"] == 1
    assert result["input_rdm_undefined_pairs"] == 3
