import json
import sqlite3
from pathlib import Path

import pytest

from semantic_retraining import classification_r2 as r2
from semantic_retraining.taxonomy import load_taxonomy


ROOT = Path(__file__).resolve().parents[1]


def taxonomy():
    return load_taxonomy(ROOT / "configs/task_taxonomy_32_reconstructed.json")


def config():
    return {
        "model": "Qwen2.5-72B-Instruct",
        "revision": "fixture",
        "temperature": 0,
        "seed": 0,
        "max_tokens": 2048,
        "response_protocol": r2.VERSION,
        "attempts_per_batch": 2,
    }


def reply(payload, task=6):
    return {
        "labels": [
            {
                "feature_id": f["feature_id"],
                "task_id": task,
                "uncertain": False,
                "out_of_inventory": False,
                "explanation": "An external component.",
            }
            for f in payload["features"]
        ]
    }


def test_registry_preserves_exact_text_and_is_order_independent():
    texts = ["has large theeth", "has big claws ", "has big claws"]
    registry = r2.feature_registry({"human": dict(enumerate(texts))})
    assert registry == r2.feature_registry({"other": dict(enumerate(reversed(texts)))})
    assert set(registry.values()) == set(texts)
    assert len(set(registry)) == 3


@pytest.mark.parametrize(
    "fault", ["duplicate", "missing", "unknown", "bool_task", "gap"]
)
def test_response_validation_rejects_identity_or_type_errors(fault):
    features = [
        {"feature_id": "F000001", "description": "has large theeth"},
        {"feature_id": "F000002", "description": "has big claws "},
    ]
    response = reply({"features": features})
    if fault == "duplicate":
        response["labels"][1]["feature_id"] = "F000001"
    elif fault == "missing":
        response["labels"].pop()
    elif fault == "unknown":
        response["labels"][0]["feature_id"] = "F999999"
    elif fault == "bool_task":
        response["labels"][0]["task_id"] = True
    else:
        response["labels"][0]["out_of_inventory"] = True
    with pytest.raises(ValueError):
        r2.validate_response(response, features, taxonomy())


def test_id_response_restores_original_text_without_echo_or_spelling_repair():
    features = [{"feature_id": "F000001", "description": "has large theeth"}]
    rows = r2.validate_response(reply({"features": features}), features, taxonomy())
    assert rows[0]["description"] == "has large theeth"
    schema = r2.response_schema(features, taxonomy())
    assert schema["properties"]["labels"]["items"]["properties"]["feature_id"][
        "enum"
    ] == ["F000001"]


def test_retry_splits_bad_batch_commits_good_leaves_and_resumes_only_missing(tmp_path):
    texts = ["good", "bad", "good too"]
    registry = r2.feature_registry({"human": dict(enumerate(texts))})
    called = []

    def client(payload):
        batch = [f["description"] for f in payload["features"]]
        called.append(batch)
        return {"labels": []} if "bad" in batch else reply(payload)

    with pytest.raises(ValueError, match="incomplete"):
        r2.classify_ids(texts, registry, taxonomy(), config(), tmp_path, client, 8)
    with sqlite3.connect(tmp_path / "cache.sqlite3") as db:
        assert db.execute("SELECT COUNT(*) FROM labels").fetchone()[0] == 2
    assert json.loads((tmp_path / "progress.json").read_text())["missing"] == ["bad"]
    called.clear()

    def fixed(payload):
        called.extend(f["description"] for f in payload["features"])
        return reply(payload)

    rows = r2.classify_ids(texts, registry, taxonomy(), config(), tmp_path, fixed, 8)
    assert called == ["bad"] and set(rows) == set(texts)
    assert len(list((tmp_path / "attempts").glob("*.json"))) >= 5
    with pytest.raises(ValueError, match="configuration"):
        r2.classify_ids(
            texts, registry, taxonomy(), {**config(), "seed": 1}, tmp_path, fixed, 8
        )


def test_prompt_sources_and_known_gap_are_explicit_and_smoke_checks_semantics():
    instruction = r2.instruction()
    assert "Associated object" in instruction and "32" in instruction
    assert "Wu" in instruction and "Table A1" in instruction
    assert "out_of_inventory=true" in instruction
    assert "has large theeth" not in instruction
    anchors = r2.SMOKE_ANCHORS
    assert anchors["can fly"] == 5 and anchors["is eaten"] == 21
    assert anchors["used for cutting"] == 24


def test_review_covers_all_human_features_and_every_gap():
    catalog = {"human": {"h": "wings"}, "v2": {"x": "associated pot"}}
    records = {
        "wings": {"task_id": 6, "uncertain": False},
        "associated pot": {"task_id": 23, "uncertain": True, "out_of_inventory": True},
    }
    rows = r2.review_rows(catalog, records)
    assert {r["description"] for r in rows} == set(records)
    assert (
        "all_human_features"
        in next(r for r in rows if r["description"] == "wings")["strata"]
    )


def test_r2_config_preserves_32_classes_and_training_and_uses_new_directory():
    old = json.loads((ROOT / "configs/semantic_retraining.json").read_text())
    new = json.loads((ROOT / "configs/semantic_retraining_r2.json").read_text())
    for key in [
        "taxonomy",
        "seeds",
        "semantic",
        "downstream_settings",
        "deferred_conditions",
    ]:
        assert old[key] == new[key]
    assert old["output_directory"] != new["output_directory"]
    assert new["classifier"]["response_protocol"] == r2.VERSION


def test_prompt_is_in_preflight_signature_but_not_unrelated_training_code_scope():
    from semantic_retraining.provenance import code_hashes

    key = "configs/semantic_classifier_r2_prompt.txt"
    assert key in code_hashes()
    assert key not in code_hashes("semantic")


def test_cli_defaults_to_r2_and_rejects_unknown_protocol(tmp_path):
    from semantic_retraining.run_crosses import load_config

    current = load_config()
    assert current["classifier"]["response_protocol"] == r2.VERSION
    current["classifier"]["response_protocol"] = "typo-version"
    path = tmp_path / "wrong.json"
    path.write_text(json.dumps(current))
    with pytest.raises(ValueError, match="Unknown classifier"):
        load_config(path)


def test_transport_sends_schema_and_preserves_each_raw_attempt(tmp_path, monkeypatch):
    features = [{"feature_id": "F000001", "description": "has large theeth"}]
    payload = {
        "features": features,
        "instruction": r2.instruction(),
        "taxonomy": taxonomy()["classes"],
        "schema": r2.response_schema(features, taxonomy()),
    }
    sent = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return json.dumps(
                {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {"content": json.dumps(reply(payload))},
                        }
                    ]
                }
            ).encode()

    def open_request(request, **kwargs):
        sent.append(json.loads(request.data))
        return Response()

    monkeypatch.setattr(r2.urllib.request, "urlopen", open_request)
    client = r2.qwen_client("http://localhost:8024/v1", config(), tmp_path)
    assert client(payload) == client(payload) == reply(payload)
    assert len(list(tmp_path.glob("*.json"))) == 2
    assert sent[0]["response_format"]["type"] == "json_schema"
    assert json.loads(sent[0]["messages"][1]["content"])["features"] == features
    assert set(json.loads(sent[0]["messages"][1]["content"])) == {
        "features",
        "taxonomy",
    }


def test_network_failure_is_bounded_without_expanding_into_all_singletons(
    tmp_path, monkeypatch
):
    calls = []
    monkeypatch.setattr(r2.time, "sleep", lambda seconds: None)

    def offline(payload):
        calls.append(payload)
        raise OSError("endpoint unavailable")

    texts = ["a", "b", "c"]
    registry = r2.feature_registry({"human": dict(enumerate(texts))})
    with pytest.raises(ValueError, match="unavailable"):
        r2.classify_ids(texts, registry, taxonomy(), config(), tmp_path, offline, 8)
    assert len(calls) == 2
    assert not json.loads((tmp_path / "progress.json").read_text())["complete"]


def test_smoke_semantic_gate_and_production_are_isolated(tmp_path, monkeypatch):
    (tmp_path / "inventory").mkdir()
    catalog = {"human": {"h1": "has a tail", "h2": "has large theeth"}}
    (tmp_path / "inventory/catalog.json").write_text(json.dumps(catalog))

    def client(endpoint, conf, directory):
        def classify(payload):
            result = reply(payload)
            for feature, row in zip(payload["features"], result["labels"]):
                row["task_id"] = r2.SMOKE_ANCHORS.get(feature["description"], 6)
                if feature["description"] == "cooked in a pot":
                    row.update(uncertain=True, out_of_inventory=True)
            return result

        return classify

    monkeypatch.setattr(r2, "qwen_client", client)
    with pytest.raises(ValueError, match="classify-smoke"):
        r2.classify_catalog(tmp_path, taxonomy(), config(), "fixture")
    result = r2.classification_smoke(tmp_path, taxonomy(), config(), "fixture")
    assert result["passed"] and not (tmp_path / "classification/initial").exists()
    result = r2.classify_catalog(
        tmp_path, taxonomy(), config(), "fixture", conditions=["human"]
    )
    assert result["complete"] and not result["training_authorized"]
    registry = json.loads(
        (tmp_path / "classification/feature_registry.json").read_text()
    )
    assert set(registry.values()) == set(catalog["human"].values())
    review = tmp_path / "classification/human_review.csv"
    with pytest.raises(ValueError, match="every human feature"):
        r2.freeze_catalog(tmp_path, taxonomy(), review, conditions=["human"])
