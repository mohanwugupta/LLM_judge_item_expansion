import csv
import json
import hashlib
from pathlib import Path
import tarfile

import pandas as pd
import pytest

from audit_v4_pretraining import preview_type_only_recovery
from recover_v4_exact_ids import patch_shard
from leuven_expansion.v4 import sha256_file


def test_tracked_recovery_bundle_is_self_contained_and_verified():
    root = Path(__file__).resolve().parents[2]
    with tarfile.open(root / "configs/v4_exact_id_recovery_inputs.tar.gz") as archive:
        expected = {
            "recovery_plan.json",
            "offline_recovery_preview.csv",
            "retry_cells.csv",
            "invalid_cell_adjudication_votes.csv",
            "invalid_resolutions.csv",
        }
        assert set(archive.getnames()) == expected
        assert all(member.isfile() for member in archive.getmembers())
        plan = json.load(archive.extractfile("recovery_plan.json"))
        for name, digest in plan["evidence_sha256"].items():
            assert (
                hashlib.sha256(archive.extractfile(name).read()).hexdigest() == digest
            )
    launcher = (root / "run_leuven_v4_atomic_finalize.sh").read_text()
    assert "configs/v4_exact_id_recovery_inputs.tar.gz" in launcher
    assert launcher.index('tar -xzf "$RECOVERY_INPUTS_BUNDLE"') < launcher.index(
        '--output-dir "$RECOVERY_DIR" --apply'
    )


def test_saved_response_recovery_rejects_wrong_ids_words_and_disagreement(tmp_path):
    rows = []
    for word, identifier, values, returned_word in (
        ("valid", "10", [2, 2, 2], "valid"),
        ("wrong_id", "11", [2, 2, 2], "wrong_id"),
        ("wrong_word", "10", [2, 2, 2], "other"),
        ("disagree", "10", [1, 2, 2], "disagree"),
        ("single", "10", [2], "single"),
    ):
        for index, value in enumerate(values, 1):
            rows.append(
                {
                    "shard_index": 0,
                    "word_normalized": word,
                    "feature_id": 10,
                    "feature_text": "a feature",
                    "adjudicator_idx": index,
                    "raw_json": json.dumps(
                        {
                            "target_word": returned_word,
                            "feature_id": identifier,
                            "feature_value": value,
                            "confidence": 0.9,
                            "ambiguous": False,
                            "reason": "consistent feature",
                        }
                    ),
                }
            )
    pd.DataFrame(rows).to_csv(
        tmp_path / "invalid_cell_adjudication_votes.csv", index=False
    )
    summary = preview_type_only_recovery(tmp_path)
    assert summary["cells_with_unanimous_type_only_recovery"] == 1
    assert summary["cells_still_needing_fresh_judgment_or_review"] == 4


def test_atomic_recovery_preserves_valid_rows_and_is_idempotent(tmp_path):
    path = tmp_path / "feature_resolutions.csv"
    rows = [
        {
            "word_normalized": "dog",
            "feature_id": 1,
            "feature_text": "fur",
            "final_feature_value": "",
            "resolution_method": "adjudicator_failed",
            "needs_human_audit": True,
        },
        {
            "word_normalized": "cat",
            "feature_id": 1,
            "feature_text": "fur",
            "final_feature_value": 3,
            "resolution_method": "unanimous",
            "needs_human_audit": False,
        },
    ]
    pd.DataFrame(rows).to_csv(path, index=False)
    before = path.read_bytes()
    record = {
        "source_sha256": sha256_file(path),
        "expected_cells": 2,
        "invalid_before": 1,
    }
    patches = {("dog", "1"): {"value": 2.0, "feature_text": "fur"}}
    result = patch_shard(path, patches, record, tmp_path / "recovery")
    assert result["recovered_cells"] == 1 and result["remaining_invalid_cells"] == 0
    assert (
        tmp_path / "recovery/original_feature_resolutions.csv"
    ).read_bytes() == before
    actual = list(csv.DictReader(path.open()))
    assert actual[0]["final_feature_value"] == "2.0"
    assert (
        actual[1]["final_feature_value"] == "3"
        and actual[1]["resolution_method"] == "unanimous"
    )
    assert patch_shard(path, patches, record, tmp_path / "recovery") == result


def test_changed_source_or_incomplete_patch_coverage_cannot_commit(tmp_path):
    path = tmp_path / "feature_resolutions.csv"
    pd.DataFrame(
        [
            {
                "word_normalized": "dog",
                "feature_id": 1,
                "feature_text": "fur",
                "final_feature_value": "",
                "resolution_method": "adjudicator_failed",
                "needs_human_audit": True,
            }
        ]
    ).to_csv(path, index=False)
    before = path.read_bytes()
    record = {
        "source_sha256": sha256_file(path),
        "expected_cells": 1,
        "invalid_before": 1,
    }
    with pytest.raises(ValueError, match="coverage"):
        patch_shard(
            path,
            {("cat", "1"): {"value": 2, "feature_text": "fur"}},
            record,
            tmp_path / "recovery",
        )
    assert path.read_bytes() == before
    record["source_sha256"] = "changed"
    with pytest.raises(ValueError, match="audited input"):
        patch_shard(path, {}, record, tmp_path / "another")
