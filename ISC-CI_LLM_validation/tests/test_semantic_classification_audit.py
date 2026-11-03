import hashlib
import io
import json
import sqlite3
import tarfile
from pathlib import Path

import pytest

from audit_semantic_classification import inspect_archive
from semantic_retraining.classify_features import INSTRUCTION
from semantic_retraining.provenance import digest
from semantic_retraining.taxonomy import load_taxonomy, taxonomy_hash

ROOT = Path(__file__).resolve().parents[1]


def snapshot(tmp_path, invalid_record=False):
    taxonomy = load_taxonomy(ROOT / "configs/task_taxonomy_32_reconstructed.json")
    row = {
        "description": "has a tail",
        "task_id": 9,
        "uncertain": False,
        "explanation": "An internal part.",
    }
    database = tmp_path / "source.sqlite3"
    with sqlite3.connect(database) as db:
        db.execute(
            "CREATE TABLE labels(description TEXT PRIMARY KEY,response TEXT NOT NULL)"
        )
        db.execute(
            "INSERT INTO labels VALUES (?,?)",
            (
                row["description"],
                json.dumps({**row, "task_id": 99} if invalid_record else row),
            ),
        )
    config = {
        "model": "fixture",
        "revision": "frozen",
        "temperature": 0,
        "seed": 0,
        "max_tokens": 2048,
    }
    request = {
        "model": "fixture",
        "temperature": 0,
        "seed": 0,
        "max_tokens": 2048,
        "messages": [
            {
                "role": "system",
                "content": INSTRUCTION + " Return JSON only, without Markdown.",
            },
            {
                "role": "user",
                "content": json.dumps(
                    {"features": ["has a tail"], "taxonomy": taxonomy["classes"]}
                ),
            },
        ],
    }
    files = {
        "inventory/catalog.json": {"human": {"f1": "has a tail", "f2": "flies"}},
        "inventory/sources.json": {"human": {"file_sha256": "matrix-hash"}},
        "inventory/manifest.json": {
            "signature": {
                "catalog_sha256": digest(
                    {"human": {"f1": "has a tail", "f2": "flies"}}
                ),
                "matrix_sources": {"human": "matrix-hash"},
                "taxonomy_sha256": taxonomy_hash(taxonomy),
            }
        },
        "classification/initial/classifier_manifest.json": {
            "taxonomy": taxonomy_hash(taxonomy),
            "classifier": config,
            "instruction": INSTRUCTION,
        },
        f"classification/requests/initial/{digest(request)}.json": {
            "request": request,
            "response": {
                "choices": [
                    {"finish_reason": "stop", "message": {"content": json.dumps([row])}}
                ]
            },
        },
        "gates/preflight.json": {"passed": True, "returncode": 0},
    }
    archive = tmp_path / "snapshot.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for name, data in files.items():
            content = json.dumps(data).encode()
            member = tarfile.TarInfo(name)
            member.size = len(content)
            tar.addfile(member, io.BytesIO(content))
        content = database.read_bytes()
        member = tarfile.TarInfo("classification/initial/cache.sqlite3")
        member.size = len(content)
        tar.addfile(member, io.BytesIO(content))
    return archive, taxonomy


def test_successful_preflight_is_not_completed_classification(tmp_path):
    archive, taxonomy = snapshot(tmp_path)
    before = hashlib.sha256(archive.read_bytes()).hexdigest()
    result = inspect_archive(archive, taxonomy, ["human"])
    assert result["summary"]["cached_descriptions"] == 1
    assert result["summary"]["expected_descriptions"] == 2
    assert result["summary"]["archived_preflight_passed"]
    assert not result["summary"]["classification_complete"]
    assert not result["summary"]["ready_for_training"]
    assert result["summary"]["missing_final_outputs"]
    assert result["review_flags"][0]["description"] == "has a tail"
    assert result["review_flags"][0]["suggested_task_id"] == 6
    assert result["labels"][0]["task_id"] == 9
    assert before == hashlib.sha256(archive.read_bytes()).hexdigest()


def test_invalid_cached_task_and_missing_raw_evidence_are_not_certified(tmp_path):
    archive, taxonomy = snapshot(tmp_path, invalid_record=True)
    result = inspect_archive(archive, taxonomy, ["human"])
    assert not result["summary"]["technical_integrity_passed"]
    assert any("task ID" in issue for issue in result["summary"]["integrity_issues"])


def test_optional_expected_archive_hash_is_enforced(tmp_path):
    archive, taxonomy = snapshot(tmp_path)
    with pytest.raises(ValueError, match="SHA-256"):
        inspect_archive(archive, taxonomy, ["human"], expected_sha256="0" * 64)
