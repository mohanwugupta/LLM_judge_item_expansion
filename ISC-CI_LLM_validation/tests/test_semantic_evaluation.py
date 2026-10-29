import numpy as np
import pandas as pd
import pytest

from semantic_retraining.evaluate_crosses import (
    similarity_predictions,
    rdm_agreement,
    factorial_effects,
    scoring_functions,
)


def test_original_similarity_items_and_undefined_pair_policy_are_retained():
    values = np.array([[1, 0], [1, 1], [0, 0]], dtype=np.float32)
    data = pd.DataFrame(
        {
            "Premise 1": ["a", "a", "b"],
            "Conclusion": ["b", "zero", "zero"],
            "Domain": ["fixture"] * 3,
            "Normalized Similarity": [0.8, 0.2, 0.1],
        }
    )
    result = similarity_predictions(
        values, ["a", "b", "zero"], data, "fixture", 0, "raw_feature_matrix"
    )
    assert result.iloc[0]["prediction"] == pytest.approx(1 / np.sqrt(2))
    assert result["prediction"].isna().sum() == 2
    assert result["item_id"].is_unique
    assert result["prediction_status"].tolist() == [
        "defined",
        "undefined_zero_feature_vector",
        "undefined_zero_feature_vector",
    ]


def test_rdms_report_common_defined_pairs_without_zero_imputation():
    values = np.array([[1, 0], [0, 0], [1, 1], [0, 1]], dtype=float)
    result = rdm_agreement(values, values)
    assert result["valid_pairs"] == 3 and result["total_pairs"] == 6
    assert result["spearman"] == pytest.approx(1)


def test_factorial_decomposition_is_paired_by_seed_not_196_independent_datasets():
    rows = [
        {
            "semantic_source": s,
            "context_source": c,
            "seed": seed,
            "task": "similarity",
            "dataset": "fixture",
            "paper_statistic": value + seed,
        }
        for seed in range(4)
        for (s, c), value in zip(
            [("h", "h"), ("h", "r"), ("r", "h"), ("r", "r")], [1, 3, 5, 7]
        )
    ]
    effects = factorial_effects(pd.DataFrame(rows))
    assert effects["interaction"].abs().max() == pytest.approx(0)
    assert set(effects["semantic_effect"]) == {-2, 2}
    assert set(effects["context_effect"]) == {-1, 1}
    assert set(effects["seed"]) == {0, 1, 2, 3}


def test_behavior_scoring_is_the_existing_implementation():
    import run_human_behavior_validation as original
    from iscci_validation.behavior import summarize_behavior

    assert scoring_functions()["predict"] is original.run_predictions
    assert scoring_functions()["summarize"] is summarize_behavior


def test_modified_behavioral_protocol_is_not_silently_accepted(tmp_path):
    import json
    from semantic_retraining.evaluate_crosses import behavior_config

    (tmp_path / "behavior.json").write_text(
        json.dumps({"analysis": {"bootstrap_repetitions": 20}})
    )
    (tmp_path / "historical.json").write_text(
        json.dumps({"config": {"analysis": {"bootstrap_repetitions": 2000}}})
    )
    with pytest.raises(ValueError, match="frozen"):
        behavior_config(
            tmp_path,
            {
                "behavior_config": "behavior.json",
                "frozen_behavior_manifest": "historical.json",
            },
        )


def test_stage1_export_integration_with_synthetic_sources_only(tmp_path, monkeypatch):
    import torch
    import semantic_retraining.evaluate_crosses as evaluation
    from semantic_retraining.modeling import ISCModel

    torch.set_num_threads(1)
    names = list("abcd")
    matrix = pd.DataFrame(
        [[1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]],
        index=names,
        columns=list("wxyz"),
    )
    path = tmp_path / "toy.csv"
    matrix.to_csv(path)
    data = pd.DataFrame(
        {
            "Premise 1": ["a", "a", "a", "b", "b", "c"],
            "Conclusion": ["b", "c", "d", "c", "d", "d"],
            "Domain": ["fixture"] * 6,
            "Normalized Similarity": [0.1, 0.4, 0.6, 0.5, 0.7, 0.2],
        }
    )
    directory = (
        tmp_path
        / "upstream/IntegratedSemanticsControlContextInference/data/leuven_dataset"
    )
    directory.mkdir(parents=True)
    pd.DataFrame({"Name": names, "Category 1": ["x", "x", "y", "y"]}).to_csv(
        directory / "leuven_categories.csv", index=False
    )
    output = tmp_path / "outputs"
    models = {}
    for condition in ["human", "v2"]:
        for seed in range(4):
            torch.manual_seed(seed)
            models[condition, seed] = ISCModel(4, 4, 32)
            run = output / "isc_models" / condition / f"seed_{seed}"
            run.mkdir(parents=True)
            (run / "checkpoint.pt").write_bytes(
                f"synthetic fixture {condition} {seed}".encode()
            )

    def load_fixture(run):
        condition, seed = run.parent.name, int(run.name.split("_")[-1])
        return (
            models[condition, seed],
            {
                "object_ids": names,
                "feature_ids": list("wxyz"),
                "feature_tasks": [0, 0, 1, 1],
                "signature": {
                    "training": {"fixture": True},
                    "condition": condition,
                    "seed": seed,
                },
                "result": {"converged": True},
            },
            {},
        )

    monkeypatch.setattr(evaluation, "load_run", load_fixture)
    monkeypatch.setattr(evaluation, "verify_history", lambda *a: {})
    monkeypatch.setattr(evaluation, "matrix_path", lambda *a: path)
    monkeypatch.setattr(evaluation, "CONDITIONS", {"human": 4, "v2": 4})
    monkeypatch.setattr(evaluation, "load_tasks", lambda *a: {"similarity": data})
    monkeypatch.setattr(
        evaluation,
        "behavior_config",
        lambda *a: {
            "analysis": {
                "cross_validation_folds": 3,
                "bootstrap_repetitions": 3,
                "seed": 1,
            }
        },
    )
    result = evaluation.evaluate_semantics(
        tmp_path,
        output,
        ["v2"],
        {"semantic": {"schedule_locked": True, "full": {"fixture": True}}},
    )
    assert result["complete"] and result["conditions"] == ["human", "v2"]
    tasks = pd.read_csv(output / "evaluations/semantic/task_geometry.csv")
    assert len(tasks) == 8 * 32
    assert (
        len(pd.read_csv(output / "evaluations/semantic/semantic_similarity_items.csv"))
        == 8 * 6
    )
