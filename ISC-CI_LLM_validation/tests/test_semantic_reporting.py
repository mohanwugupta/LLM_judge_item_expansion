import pandas as pd

from semantic_retraining.report_results import report
from semantic_retraining.run_crosses import load_config


def test_unrun_conditions_are_reported_not_fabricated_or_omitted(tmp_path):
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    destination = tmp_path / "report"
    result = report(
        root, tmp_path / "empty_artifacts", destination, load_config(), phase=3
    )
    assert result["report_generated"] and not result["experiment_complete"]
    table = pd.read_csv(destination / "source_data/all_196_conditions.csv")
    assert len(table) == 196 and table["completed_seeds"].sum() == 0
    assert set(table["status"]) == {"planned", "deferred"}
    assert table["status"].eq("deferred").sum() == 52
    models = pd.read_csv(destination / "source_data/all_840_planned_models.csv")
    assert models["status"].eq("deferred").sum() == 216
    assert result["enabled_sources"] == 12 and result["active_source_pairs"] == 144
    assert "32-class" in (destination / "RESULTS.md").read_text()
    assert "not" in (destination / "RESULTS.md").read_text().lower()


def test_partial_cohort_geometry_leaves_unrun_sources_missing(tmp_path, monkeypatch):
    from pathlib import Path
    from semantic_retraining.inventory import sha256
    from semantic_retraining.provenance import atomic_json
    import semantic_retraining.report_results as reporting

    output = tmp_path / "output"
    directory = output / "evaluations/semantic"
    directory.mkdir(parents=True)
    path = directory / "raw_geometry.csv"
    pd.DataFrame([{"condition": "human", "spearman": 1.0}]).to_csv(path, index=False)
    atomic_json(
        directory / "manifest.json", {"outputs": {"raw_geometry.csv": sha256(path)}}
    )
    monkeypatch.setattr(reporting, "make_figures", lambda *a: [])
    destination = tmp_path / "report"
    report(
        Path(__file__).resolve().parents[1], output, destination, load_config(), phase=3
    )
    table = pd.read_csv(destination / "source_data/all_196_conditions.csv")
    assert (
        table.loc[
            table.semantic_source.eq("human"), "semantic_input_leuven_rdm_spearman"
        ]
        .eq(1.0)
        .all()
    )
    assert (
        table.loc[
            table.semantic_source.eq("v4_ensemble_locked_v2"),
            "semantic_input_leuven_rdm_spearman",
        ]
        .isna()
        .all()
    )


def test_saved_behavior_plots_include_prespecified_contrast_uncertainty(tmp_path):
    import numpy as np
    from PIL import Image
    from semantic_retraining.report_results import make_figures

    directory = tmp_path / "artifacts/evaluations/context/phase_3"
    directory.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "condition": "human__human",
                "task": "fixture",
                "dataset": "fixture",
                "metric": "paper_statistic",
                "mean": 0.6,
            },
            {
                "condition": "random_human__random_human",
                "task": "fixture",
                "dataset": "fixture",
                "metric": "paper_statistic",
                "mean": 0.4,
            },
        ]
    ).to_csv(directory / "behavior_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "condition_a": "human__human",
                "condition_b": "random_human__random_human",
                "task": "fixture",
                "dataset": "fixture",
                "metric": "paper_statistic",
                "difference_a_minus_b": 0.2,
                "ci_low": 0.1,
                "ci_high": 0.3,
            }
        ]
    ).to_csv(directory / "paired_contrasts.csv", index=False)
    destination = tmp_path / "report"
    figures = make_figures(
        tmp_path / "artifacts", destination, pd.DataFrame(), None, {"phase": 3}
    )
    assert "matched_pipeline_contrasts_fixture" in figures
    pixels = np.asarray(
        Image.open(
            destination / "figures/matched_pipeline_contrasts_fixture.png"
        ).convert("RGB")
    )
    assert pixels.std() > 5 and pixels.shape[0] > 500
