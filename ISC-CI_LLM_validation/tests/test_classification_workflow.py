import json

import pytest

from semantic_retraining.classification_workflow import (
    model_fingerprint,
    review_sample,
    qwen_client,
)


def test_model_fingerprint_tracks_actual_local_weights(tmp_path):
    (tmp_path / "config.json").write_text('{"model_type":"qwen2"}')
    (tmp_path / "model.safetensors").write_bytes(b"first weights")
    first = model_fingerprint(tmp_path)
    assert first == model_fingerprint(tmp_path)
    (tmp_path / "model.safetensors").write_bytes(b"other weights")
    assert first["revision"] != model_fingerprint(tmp_path)["revision"]


def test_review_sampling_is_stratified_and_repeatable():
    catalog = {
        "human": {"x": "flies", "y": "red"},
        "random_human": {"x": "flies", "y": "red"},
    }
    records = {
        "flies": {"task_id": 5, "uncertain": False},
        "red": {"task_id": 8, "uncertain": False},
    }
    sample = review_sample(catalog, records, 1, 1)
    assert sample == review_sample(catalog, records, 1, 1)
    assert {r["description"] for r in sample} == {"flies", "red"}
    assert all(r["task_id"] == "" and r["reviewer"] == "" for r in sample)


def test_qwen_transport_sends_only_taxonomy_and_descriptions(monkeypatch, tmp_path):
    def open_request(request, timeout):
        body = json.loads(request.data)
        assert body["model"] == "Qwen2.5-72B-Instruct"
        assert body["temperature"] == 0 and body["seed"] == 0
        content = json.loads(body["messages"][1]["content"])
        assert set(content) == {"taxonomy", "features"}
        assert content["features"] == ["flies"]

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
                                "message": {
                                    "content": '[{"description":"flies","task_id":5,"uncertain":false,"explanation":"behavior"}]'
                                }
                            }
                        ]
                    }
                ).encode()

        return Response()

    monkeypatch.setattr("urllib.request.urlopen", open_request)
    client = qwen_client(
        "http://localhost:8024/v1",
        {
            "model": "Qwen2.5-72B-Instruct",
            "temperature": 0,
            "seed": 0,
            "max_tokens": 2048,
        },
        tmp_path,
    )
    assert (
        client({"instruction": "Classify", "features": ["flies"], "taxonomy": []})[0][
            "task_id"
        ]
        == 5
    )


def test_incremental_human_freeze_is_stable_and_produces_coverage_audit(tmp_path):
    import csv
    import pandas as pd
    from semantic_retraining.classification_workflow import freeze_catalog
    from semantic_retraining.inventory import sha256
    from semantic_retraining.taxonomy import load_taxonomy, taxonomy_hash
    from semantic_retraining.provenance import atomic_json
    from pathlib import Path

    taxonomy = load_taxonomy(
        Path(__file__).resolve().parents[1]
        / "configs/task_taxonomy_32_reconstructed.json"
    )
    catalog = {c: {"x": "flies", "y": "red"} for c in ["human", "random_human", "v2"]}
    sources = {}
    for condition in catalog:
        path = tmp_path / f"{condition}.csv"
        pd.DataFrame({"x": [1, 0], "y": [0, 1]}, index=["a", "b"]).to_csv(path)
        sources[condition] = {"path": str(path), "file_sha256": sha256(path)}
    atomic_json(tmp_path / "inventory/catalog.json", catalog)
    atomic_json(tmp_path / "inventory/sources.json", sources)
    records = {
        text: {
            "description": text,
            "task_id": task,
            "uncertain": False,
            "explanation": "fixture",
        }
        for text, task in [("flies", 5), ("red", 8)]
    }
    atomic_json(tmp_path / "classification/classified_descriptions.json", records)
    atomic_json(
        tmp_path / "classification/protocol.json",
        {"taxonomy_sha256": taxonomy_hash(taxonomy), "classifier": "fixture"},
    )
    reviews = tmp_path / "reviews.csv"
    reviews.write_text("description,task_id,reviewer\nflies,5,fixture\nred,8,fixture\n")
    freeze_catalog(tmp_path, taxonomy, reviews, ["human", "random_human"])
    before = sha256(tmp_path / "classification/frozen/human.json")
    freeze_catalog(tmp_path, taxonomy, reviews, ["v2"])
    assert before == sha256(tmp_path / "classification/frozen/human.json")
    audit = pd.read_csv(tmp_path / "classification/frozen/human_task_audit.csv")
    assert len(audit) == 32 and audit["features"].sum() == 2
    assert audit["positive_associations"].sum() == 2
    assert audit["valid_object_task_pairs"].sum() == 2
