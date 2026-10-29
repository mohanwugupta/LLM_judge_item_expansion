from pathlib import Path

import pytest

from semantic_retraining.classify_features import (
    classify_descriptions,
    freeze_assignments,
    retained_labels,
)
from semantic_retraining.inventory import load_matrix, matrix_path, CONDITIONS
from semantic_retraining.taxonomy import load_taxonomy

ROOT = Path(__file__).resolve().parents[1]
TAXONOMY = ROOT / "configs/task_taxonomy_32_reconstructed.json"


def test_cached_exact_text_classification_and_resume(tmp_path):
    taxonomy = load_taxonomy(TAXONOMY)
    config = {"model": "fixture", "revision": "fixed-1", "temperature": 0, "seed": 0}
    calls = []

    def classify(payload):
        assert set(payload) == {"features", "taxonomy", "instruction"}
        calls.append(payload)
        return [
            {
                "description": d,
                "task_id": 5,
                "uncertain": False,
                "explanation": "Intrinsic action",
            }
            for d in payload["features"]
        ]

    labels = ["can fly", "can fly", "runs"]
    first = classify_descriptions(
        labels, taxonomy, config, tmp_path, classify, batch_size=2
    )
    assert len(first) == 2 and len(calls) == 1
    assert first == classify_descriptions(labels, taxonomy, config, tmp_path, classify)
    assert len(calls) == 1
    with pytest.raises(ValueError, match="configuration"):
        classify_descriptions(
            labels, taxonomy, {**config, "revision": "changed"}, tmp_path, classify
        )


def test_unresolved_and_invalid_classification_blocks_freeze(tmp_path):
    taxonomy = load_taxonomy(TAXONOMY)
    config = {"model": "fixture", "revision": "fixed-1", "temperature": 0, "seed": 0}

    def classifier(payload):
        return [
            {
                "description": "runs",
                "task_id": 5,
                "uncertain": True,
                "explanation": "needs review",
            }
        ]

    records = classify_descriptions(["runs"], taxonomy, config, tmp_path, classifier)
    with pytest.raises(ValueError, match="unresolved"):
        freeze_assignments({"f": "runs"}, records, taxonomy, [])
    with pytest.raises(ValueError, match="missing"):
        freeze_assignments({"f": "unknown"}, records, taxonomy, [])


def test_all_retained_ids_have_real_descriptions():
    for condition in CONDITIONS:
        matrix = load_matrix(matrix_path(ROOT, condition))
        before = matrix.values.copy()
        labels = retained_labels(ROOT, condition, list(matrix.columns))
        assert list(labels) == list(matrix.columns)
        assert all(isinstance(v, str) and v.strip() for v in labels.values())
        assert (before == matrix.values).all()
    human = load_matrix(matrix_path(ROOT, "human"))
    assert retained_labels(ROOT, "human", list(human.columns)) == retained_labels(
        ROOT, "random_human", list(human.columns)
    )


def test_reviews_cover_each_nonempty_task_stratum_not_just_an_unrelated_label():
    taxonomy = load_taxonomy(TAXONOMY)
    records = {
        "flies": {
            "description": "flies",
            "task_id": 5,
            "uncertain": False,
            "explanation": "behavior",
        },
        "red": {
            "description": "red",
            "task_id": 8,
            "uncertain": False,
            "explanation": "visible",
        },
    }
    with pytest.raises(ValueError, match="stratif"):
        freeze_assignments(
            {"f": "flies", "g": "red"},
            records,
            taxonomy,
            [{"description": "flies", "task_id": 5, "reviewer": "reviewer"}],
        )


def test_separate_adjudication_prompt_is_auditable(tmp_path):
    taxonomy = load_taxonomy(TAXONOMY)
    config = {"model": "fixture", "revision": "fixed-1", "temperature": 0, "seed": 0}

    def classifier(payload):
        assert payload["instruction"] == "Separate ambiguity review"
        return [
            {
                "description": "flies",
                "task_id": 5,
                "uncertain": False,
                "explanation": "behavior",
            }
        ]

    result = classify_descriptions(
        ["flies"],
        taxonomy,
        config,
        tmp_path,
        classifier,
        instruction="Separate ambiguity review",
    )
    assert not result["flies"]["uncertain"]
