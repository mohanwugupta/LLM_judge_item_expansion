import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import plot_finalized_v4_results as figures


def test_exact_condition_registry_and_no_model_imports():
    assert len(figures.MAIN) == 7
    assert len(set(figures.ALL)) == 13
    assert figures.MAIN[2] == "v3_B"
    assert set(figures.LABELS) == set(figures.ALL) | {figures.RANDOM}
    tree = ast.parse(Path(figures.__file__).read_text())
    imports = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    assert not any(
        n and ("training" in n or "simulations" in n or "torch" in n) for n in imports
    )


def test_missing_panel_documented_but_available_work_continues(tmp_path):
    args = figures.parser().parse_args(["--output-dir", str(tmp_path / "output")])
    package = figures.Package(args)
    package.stage("missing", lambda: package.read(tmp_path / "missing.csv"))
    calls = []
    package.stage("available", lambda: calls.append(True))
    assert calls == [True]
    assert str(tmp_path / "missing.csv") in package.blocked[0]["missing_input"]


def test_output_cannot_overwrite_artifact_directories():
    args = figures.parser().parse_args([])
    args.output_dir = args.behavior_dir
    with pytest.raises(ValueError, match="new subdirectory"):
        figures.Package(args)


def test_missing_core_input_is_strict(tmp_path):
    args = figures.parser().parse_args(
        [
            "--behavior-dir",
            str(tmp_path / "absent"),
            "--output-dir",
            str(tmp_path / "output"),
        ]
    )
    with pytest.raises(FileNotFoundError, match="manifest.json"):
        figures.Package(args).initialize()


def test_duplicate_and_missing_seed_rejection():
    with pytest.raises(ValueError, match="Duplicate"):
        figures.check_unique(pd.DataFrame({"id": [1, 1]}), ["id"], "observations")
    rows = pd.DataFrame(
        [(c, s) for c in figures.ALL for s in range(4)], columns=["condition", "seed"]
    )
    figures.check_seeds(rows)
    with pytest.raises(ValueError, match="seeds 0-3"):
        figures.check_seeds(rows.drop(index=0))


def test_common_finite_pairs_stable_ids_no_imputation_no_intervals():
    raw, trained, shared = [], [], []
    for domain in figures.DOMAINS:
        for c in figures.MAIN:
            for i in range(5):
                prediction = float(i)
                if c == "v3_B" and i == 4:
                    prediction = np.nan
                raw.append(
                    dict(
                        condition=c,
                        dataset=domain,
                        item_id=f"{domain}-{i}",
                        prediction=prediction,
                        human=float(i),
                    )
                )
                for seed in range(4):
                    trained.append(
                        dict(
                            condition=c,
                            dataset=domain,
                            seed=seed,
                            item_id=f"{domain}-{i}",
                            prediction=float(i),
                            human=float(i),
                        )
                    )
        for i in range(5):
            shared.append(
                dict(
                    condition="shared_encoder",
                    dataset=domain,
                    item_id=f"{domain}-{i}",
                    prediction=float(i),
                    human=float(i),
                )
            )
    raw = pd.DataFrame(raw).sample(frac=1, random_state=1)
    trained = pd.DataFrame(trained).sample(frac=1, random_state=2)
    scores, membership = figures.common_pair_scores(raw, trained, pd.DataFrame(shared))
    assert scores["items"].eq(4).all()
    assert scores.coverage.eq(0.8).all()
    assert scores.ci_low.isna().all() and scores.ci_high.isna().all()
    assert np.allclose(scores["mean"], 1)
    assert membership.groupby("dataset").common_finite.sum().eq(4).all()
    assert len(scores[scores.condition.eq("shared_encoder")]) == 9
    with pytest.raises(ValueError, match="Duplicate"):
        figures.common_pair_scores(
            pd.concat([raw, raw.iloc[[0]]]), trained, pd.DataFrame(shared)
        )


def test_context_score_orientation_and_three_choice_denominator():
    frame = pd.DataFrame(
        {
            "c1-Distractor 1": [0.4],
            "c2-Distractor 1": [0.2],
            "c1-Distractor 2": [0.1],
            "c2-Distractor 2": [0.5],
            "Conclusion 1 Chosen-Distractor 1": [0.3],
            "Conclusion 2 Chosen-Distractor 1": [0.2],
            "Conclusion 1 Chosen-Distractor 2": [0.2],
            "Conclusion 2 Chosen-Distractor 2": [0.4],
            "human": [0.3],
            "prediction": [0.6],
        }
    )
    figures.verify_effects(frame, "context")
    frame["prediction"] = 0.8
    with pytest.raises(ValueError, match="original definition"):
        figures.verify_effects(frame, "context")


def test_asymmetry_preserves_logit_difference_not_choice_accuracy():
    frame = pd.DataFrame(
        {"argument1_logit": [-1.0], "argument2_logit": [2.0], "prediction": [3.0]}
    )
    figures.verify_effects(frame, "asymmetry")
    frame["prediction"] = 1.0
    with pytest.raises(ValueError, match="original definition"):
        figures.verify_effects(frame, "asymmetry")


def test_saved_source_metadata_and_input_hashes(tmp_path):
    args = figures.parser().parse_args(["--output-dir", str(tmp_path / "output")])
    package = figures.Package(args)
    source = tmp_path / "input.json"
    source.write_text(json.dumps({"value": 1}))
    package.json(source)
    assert package.sources[str(source)] == figures.sha256(source)
    filename = package.source(
        "test", pd.DataFrame({"condition": ["human"], "mean": [0.5]})
    )
    frame = pd.read_csv(package.out / filename)
    assert frame.display_label.iloc[0] == figures.LABELS["human"]
    assert frame.estimate.iloc[0] == 0.5
    assert {
        "condition",
        "level",
        "task",
        "dataset",
        "metric",
        "subset",
        "seed",
        "items",
        "coverage",
        "uncertainty",
    }.issubset(frame.columns)
    source.write_text(json.dumps({"value": 2}))
    with pytest.raises(ValueError, match="Input changed"):
        package.json(source)


def test_mixed_human_model_panel_retains_estimates_and_levels(tmp_path):
    args = figures.parser().parse_args(["--output-dir", str(tmp_path / "output")])
    package = figures.Package(args)
    frame = pd.DataFrame(
        [
            {"condition": "human", "mean": 0.7, "level": np.nan, "estimate": np.nan},
            {"condition": "observed_human", "level": "observed_human", "estimate": 0.8},
        ]
    )
    name = package.source("mixed", frame, level="fully_trained")
    exported = pd.read_csv(package.out / name)
    assert exported.level.tolist() == ["fully_trained", "observed_human"]
    assert exported.estimate.tolist() == [0.7, 0.8]


def test_small_object_rdm_preserves_zero_vector_missingness(tmp_path):
    validation = tmp_path / "validation"
    matrix_dir = validation / "data_matrices"
    matrix_dir.mkdir(parents=True)
    matrix = pd.DataFrame(np.ones((293, 4)), index=[f"word{i}" for i in range(293)])
    matrix.iloc[-1] = 0
    matrix.to_csv(matrix_dir / "human.csv")
    args = figures.parser().parse_args(
        ["--validation-dir", str(validation), "--output-dir", str(tmp_path / "output")]
    )
    package = figures.Package(args)
    package.rep = pd.DataFrame(
        {"condition": ["human"], "features": [4], "input_objects_without_features": [1]}
    )
    result = package.matrix("human")
    assert result["rdm"].shape == (293, 293)
    assert np.isnan(result["rdm"][-1]).all()
    assert np.isnan(result["rdm"][:, -1]).all()
    assert result["rdm"][0, 0] == 0
    assert (result["object_counts"] == 0).sum() == 1
    np.testing.assert_array_equal(
        matrix, pd.read_csv(matrix_dir / "human.csv", index_col=0)
    )


def test_matrix_parser_rejects_nonbinary_values_before_casting(tmp_path):
    validation = tmp_path / "validation"
    matrix_dir = validation / "data_matrices"
    matrix_dir.mkdir(parents=True)
    pd.DataFrame({"feature": [1.5]}, index=["word"]).to_csv(matrix_dir / "human.csv")
    args = figures.parser().parse_args(
        ["--validation-dir", str(validation), "--output-dir", str(tmp_path / "output")]
    )
    with pytest.raises(ValueError, match="Invalid finalized binary matrix"):
        figures.Package(args).matrix("human")
