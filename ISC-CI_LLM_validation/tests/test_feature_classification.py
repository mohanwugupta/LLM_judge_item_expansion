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


@pytest.mark.parametrize(
    "original,returned",
    [
        ("has big claws ", "has big claws"),
        (" has a tail", "has a tail"),
        ("has bones", " has bones\t"),
    ],
)
def test_outer_whitespace_echo_is_recovered_with_original_key_and_audit(
    tmp_path, original, returned
):
    import copy
    import json

    taxonomy = load_taxonomy(TAXONOMY)
    config = {"model": "fixture", "revision": "fixed-1", "temperature": 0, "seed": 0}
    response = [
        {
            "description": returned,
            "task_id": 8,
            "uncertain": False,
            "explanation": "Original model explanation",
        }
    ]
    before = copy.deepcopy(response)
    result = classify_descriptions(
        [original], taxonomy, config, tmp_path, lambda p: response
    )
    assert response == before
    assert result == {original: {**before[0], "description": original}}
    audit_paths = list((tmp_path / "format_recoveries").glob("*.json"))
    assert len(audit_paths) == 1
    audit = json.loads(audit_paths[0].read_text())
    assert audit["policy"] == "unambiguous-outer-whitespace-only-v1"
    assert audit["received_rows"] == before
    assert audit["accepted_rows"] == [result[original]]

    def must_not_call(payload):
        pytest.fail("Recovered exact cache key must be reused")

    assert (
        classify_descriptions([original], taxonomy, config, tmp_path, must_not_call)
        == result
    )


@pytest.mark.parametrize(
    "returned", ["has big  claws", "Has big claws", "has big claws.", "has claws"]
)
def test_substantive_echo_changes_remain_rejected(tmp_path, returned):
    import sqlite3

    taxonomy = load_taxonomy(TAXONOMY)
    config = {"model": "fixture", "revision": "fixed-1", "temperature": 0, "seed": 0}

    def classifier(payload):
        return [
            {
                "description": returned,
                "task_id": 8,
                "uncertain": False,
                "explanation": "fixture",
            }
        ]

    with pytest.raises(ValueError, match="cover each batch description once"):
        classify_descriptions(
            ["has big claws "], taxonomy, config, tmp_path, classifier
        )
    with sqlite3.connect(tmp_path / "cache.sqlite3") as db:
        assert db.execute("SELECT COUNT(*) FROM labels").fetchone()[0] == 0


def test_ambiguous_trimmed_descriptions_cannot_be_assigned_by_position(tmp_path):
    taxonomy = load_taxonomy(TAXONOMY)
    config = {"model": "fixture", "revision": "fixed-1", "temperature": 0, "seed": 0}

    def classifier(payload):
        return [
            {
                "description": d,
                "task_id": 8,
                "uncertain": False,
                "explanation": "fixture",
            }
            for d in ["has big claws", "has big claws "]
        ]

    with pytest.raises(ValueError, match="[Aa]mbiguous"):
        classify_descriptions(
            [" has big claws", "has big claws "], taxonomy, config, tmp_path, classifier
        )


def test_failed_batch_rolls_back_but_preexisting_cache_and_signature_survive(tmp_path):
    import json
    import sqlite3

    taxonomy = load_taxonomy(TAXONOMY)
    config = {"model": "fixture", "revision": "fixed-1", "temperature": 0, "seed": 0}

    def row(text, task=8):
        return {
            "description": text,
            "task_id": task,
            "uncertain": False,
            "explanation": "fixture",
        }

    classify_descriptions(
        ["already saved"], taxonomy, config, tmp_path, lambda p: [row("already saved")]
    )
    before = (tmp_path / "classifier_manifest.json").read_bytes()
    with pytest.raises(ValueError, match="task ID"):
        classify_descriptions(
            ["already saved", "has big claws ", "has bones"],
            taxonomy,
            config,
            tmp_path,
            lambda p: [row("has big claws"), row("has bones", 999)],
        )
    with sqlite3.connect(tmp_path / "cache.sqlite3") as db:
        assert dict(db.execute("SELECT description,response FROM labels")) == {
            "already saved": json.dumps(row("already saved"), sort_keys=True)
        }
    assert not list((tmp_path / "format_recoveries").glob("*.json"))

    def classifier(payload):
        assert payload["features"] == ["has big claws ", "has bones"]
        return [row("has big claws"), row("has bones")]

    result = classify_descriptions(
        ["already saved", "has big claws ", "has bones"],
        taxonomy,
        config,
        tmp_path,
        classifier,
    )
    assert len(result) == 3
    assert (tmp_path / "classifier_manifest.json").read_bytes() == before


def test_saved_raw_requests_are_archived_before_a_resumed_call_can_replace_them(
    tmp_path,
):
    from semantic_retraining.classify_features import archive_classifier_requests

    path = tmp_path / "requests/initial/request-hash.json"
    path.parent.mkdir(parents=True)
    original = b'{"response":"first attempt"}\n'
    path.write_bytes(original)
    assert archive_classifier_requests(tmp_path) == 1
    copies = list((tmp_path / "request_history/initial").glob("*.json"))
    assert len(copies) == 1 and copies[0].read_bytes() == original
    assert path.read_bytes() == original
    assert archive_classifier_requests(tmp_path) == 0
    path.write_bytes(b'{"response":"second attempt"}\n')
    assert archive_classifier_requests(tmp_path) == 1
    assert copies[0].read_bytes() == original


@pytest.mark.parametrize(
    "descriptions", [None, ["one"], ["one", "one"], ["one", "extra"], [None, "two"]]
)
def test_shape_missing_extra_and_duplicate_rows_never_enter_cache(
    tmp_path, descriptions
):
    import sqlite3

    taxonomy = load_taxonomy(TAXONOMY)
    config = {"model": "fixture", "revision": "fixed-1", "temperature": 0, "seed": 0}
    rows = (
        None
        if descriptions is None
        else [
            {
                "description": d,
                "task_id": 5,
                "uncertain": False,
                "explanation": "fixture",
            }
            for d in descriptions
        ]
    )
    with pytest.raises(ValueError, match="cover each batch description once"):
        classify_descriptions(
            ["one", "two"], taxonomy, config, tmp_path, lambda p: rows
        )
    with sqlite3.connect(tmp_path / "cache.sqlite3") as db:
        assert db.execute("SELECT COUNT(*) FROM labels").fetchone()[0] == 0
    assert not list((tmp_path / "format_recoveries").glob("*.json"))


def test_exact_whitespace_variants_remain_distinct_when_returned_exactly(tmp_path):
    taxonomy = load_taxonomy(TAXONOMY)
    config = {"model": "fixture", "revision": "fixed-1", "temperature": 0, "seed": 0}
    labels = ["has claws", "has claws "]
    result = classify_descriptions(
        labels,
        taxonomy,
        config,
        tmp_path,
        lambda p: [
            {
                "description": d,
                "task_id": 5,
                "uncertain": False,
                "explanation": "fixture",
            }
            for d in labels
        ],
    )
    assert list(result) == labels
    assert not list((tmp_path / "format_recoveries").glob("*.json"))
