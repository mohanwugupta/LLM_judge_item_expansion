import json

import pandas as pd
import pytest

from leuven_expansion.v4 import candidate_inventory_hash
from run_v4_judgments import (
    CASCADE_RESOLUTION_METHOD,
    build_pairs,
    finalize,
    finalization_preflight,
    protocol_record,
    validate_shard_complete,
)
from leuven_expansion.v4 import stable_json_hash


def _bank(path):
    bank = pd.DataFrame(
        {
            "candidate_index": [0, 1],
            "candidate_id": ["v4_a", "v4_b"],
            "canonical_feature_text": ["has fur", "can fly"],
            "source_words": [json.dumps(["dog"]), json.dumps(["bird"])],
        }
    )
    bank["candidate_inventory_hash"] = candidate_inventory_hash(bank)
    bank.to_csv(path, index=False)


def _words(path):
    pd.DataFrame({"word": ["dog", "bird", "plane"]}).to_csv(path, index=False)


def test_cross_product_and_stable_shards_are_order_independent(tmp_path):
    bank = tmp_path / "bank.csv"
    words = tmp_path / "words.csv"
    _bank(bank)
    _words(words)
    pairs = build_pairs(bank, words, shard_count=3)
    assert len(pairs) == 6
    assignments = {
        (pair["candidate_id"], pair["word_normalized"]): pair["shard_index"]
        for pair in pairs
    }
    reversed_bank = pd.read_csv(bank).iloc[::-1]
    reversed_bank["candidate_index"] = [1, 0]
    reversed_bank.to_csv(bank, index=False)
    reordered = build_pairs(bank, words, shard_count=3)
    assert assignments == {
        (pair["candidate_id"], pair["word_normalized"]): pair["shard_index"]
        for pair in reordered
    }


def test_mock_shards_require_three_votes_and_finalize_exact_cross_product(tmp_path):
    bank = tmp_path / "bank.csv"
    words = tmp_path / "words.csv"
    output = tmp_path / "judgments"
    _bank(bank)
    _words(words)
    protocol = protocol_record(bank, words, "test-model", 2, None)
    for shard_index in range(2):
        pairs = build_pairs(bank, words, 2, shard_index)
        shard = output / "shards" / f"{shard_index:04d}"
        shard.mkdir(parents=True)
        votes = []
        resolutions = []
        for pair in pairs:
            for judge in "ABC":
                votes.append(
                    {
                        "word_normalized": pair["word_normalized"],
                        "feature_id": pair["feature_id"],
                        "judge_id": judge,
                        "prompt_hash": f"hash-{judge}",
                    }
                )
            resolutions.append(
                {
                    "word_normalized": pair["word_normalized"],
                    "feature_id": pair["feature_id"],
                    "feature_text": pair["feature_text"],
                    "final_feature_value": 1,
                    "confidence": 0.9,
                    "ambiguous": False,
                    "resolution_method": "unanimous",
                    "needs_human_audit": False,
                    "adjudicated": False,
                    "adjudication_trigger": "",
                }
            )
        pd.DataFrame(votes).to_csv(shard / "feature_votes.csv", index=False)
        pd.DataFrame(resolutions).to_csv(shard / "feature_resolutions.csv", index=False)
        pd.DataFrame(columns=["empty"]).to_csv(
            shard / "feature_adjudication_votes.csv", index=False
        )
        pd.DataFrame(columns=["empty"]).to_csv(shard / "parse_errors.csv", index=False)
        validate_shard_complete(shard, pairs, protocol, shard_index)
    finalize(bank, words, output, protocol)
    resolved = pd.read_csv(output / "resolved_feature_values.csv")
    assert len(resolved) == 6
    assert resolved[["candidate_id", "target_word"]].duplicated().sum() == 0
    assert set(resolved["resolved_binary_locked_v2"]) == {1}
    assert json.loads((output / "judgment_manifest.json").read_text())["complete"]


def test_finalize_accepts_valid_prompt_c_only_negatives(tmp_path):
    bank = tmp_path / "bank.csv"
    words = tmp_path / "words.csv"
    output = tmp_path / "judgments"
    _bank(bank)
    _words(words)
    protocol = protocol_record(bank, words, "test-model", 1, None)
    pairs = build_pairs(bank, words, 1, 0)
    shard = output / "shards" / "0000"
    shard.mkdir(parents=True)
    votes = []
    resolutions = []
    for index, pair in enumerate(pairs):
        c_only = index % 2 == 0
        for judge in ("C" if c_only else "ABC"):
            votes.append(
                {
                    "word_normalized": pair["word_normalized"],
                    "feature_id": pair["feature_id"],
                    "judge_id": judge,
                    "prompt_hash": f"hash-{judge}",
                    "feature_value": 0 if c_only else 1,
                    "confidence": 0.95,
                    "ambiguous": False,
                    "parse_error": "",
                }
            )
        resolutions.append(
            {
                "word_normalized": pair["word_normalized"],
                "feature_id": pair["feature_id"],
                "feature_text": pair["feature_text"],
                "final_feature_value": 0 if c_only else 1,
                "confidence": 0.95,
                "ambiguous": False,
                "resolution_method": (
                    CASCADE_RESOLUTION_METHOD if c_only else "unanimous"
                ),
                "needs_human_audit": False,
                "adjudicated": False,
                "adjudication_trigger": "",
            }
        )
    pd.DataFrame(votes).to_csv(shard / "feature_votes.csv", index=False)
    pd.DataFrame(resolutions).to_csv(shard / "feature_resolutions.csv", index=False)
    pd.DataFrame(columns=["empty"]).to_csv(
        shard / "feature_adjudication_votes.csv", index=False
    )
    pd.DataFrame(columns=["empty"]).to_csv(shard / "parse_errors.csv", index=False)
    shard_manifest = validate_shard_complete(shard, pairs, protocol, 0)
    assert shard_manifest["prompt_c_only_cells"] == 3
    assert shard_manifest["full_panel_vote_cells"] == 3
    finalize(bank, words, output, protocol)
    manifest = json.loads((output / "judgment_manifest.json").read_text())
    assert manifest["prompt_c_only_cells"] == 3
    assert manifest["full_panel_vote_cells"] == 3


def test_preflight_reports_all_failures_without_rewriting_sidecars(tmp_path):
    protocol = {
        "shard_count": 2,
        "expected_cell_count": 10,
        "candidate_inventory_hash": "bank",
    }
    originals = {}
    for index, invalid in enumerate((2, 3)):
        path = tmp_path / "shards" / f"{index:04d}" / "v4_shard_manifest.json"
        path.parent.mkdir(parents=True)
        text = json.dumps(
            {
                "shard_index": index,
                "candidate_inventory_hash": "bank",
                "protocol_hash": stable_json_hash(protocol),
                "expected_cells": 5,
                "resolved_cells": 5,
                "invalid_resolved_values": invalid,
                "complete": False,
            }
        )
        path.write_text(text)
        originals[path] = text
    report = finalization_preflight(tmp_path, protocol)
    assert report["blocked_shards"] == 2
    assert report["invalid_resolved_values"] == 5
    assert report["manifest_resolved_cells"] == 10
    assert not report["ready_from_manifests"]
    assert all(path.read_text() == text for path, text in originals.items())
    with pytest.raises(ValueError, match="invalid_resolved_values=5"):
        finalize(
            tmp_path / "nonexistent_bank.csv",
            tmp_path / "nonexistent_words.csv",
            tmp_path,
            protocol,
        )
    assert (tmp_path / "finalization_preflight.json").exists()
    assert not (tmp_path / "resolved_feature_values.csv").exists()


def test_preflight_does_not_hide_protocol_or_partition_mismatch(tmp_path):
    protocol = {
        "shard_count": 1,
        "expected_cell_count": 5,
        "candidate_inventory_hash": "bank",
    }
    path = tmp_path / "shards/0000/v4_shard_manifest.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "shard_index": 0,
                "candidate_inventory_hash": "bank",
                "protocol_hash": "other",
                "complete": True,
                "expected_cells": 5,
                "resolved_cells": 5,
            }
        )
    )
    report = finalization_preflight(tmp_path, protocol, [6])
    issues = "; ".join(report["shards"][0]["issues"])
    assert "protocol hash mismatch" in issues and "frozen hash partition" in issues
    assert not report["ready_from_manifests"]


def test_relocated_protocol_recomputes_dependent_legacy_hashes(tmp_path, monkeypatch):
    import run_v4_judgments as runner

    bank, words = tmp_path / "bank.csv", tmp_path / "words.csv"
    _bank(bank)
    _words(words)
    schema = (
        tmp_path / "leuven_expansion/schemas/atomic_feature_judgment_schema_v1.json"
    )
    schema.parent.mkdir(parents=True)
    schema.write_bytes(
        (
            runner.ROOT
            / "leuven_expansion/schemas/atomic_feature_judgment_schema_v1.json"
        ).read_bytes()
    )
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    base = tmp_path / "recorded_root"
    legacy = protocol_record(bank, words, "test-model", 2, None, None, base)
    cascade = protocol_record(
        bank, words, "test-model", 2, None, protocol_project_root=base
    )
    assert legacy["candidate_bank"] == str(base / "bank.csv")
    assert cascade["legacy_full_panel_protocol_hash"] == stable_json_hash(legacy)
    assert cascade["compatible_legacy_protocol_hashes"] == [stable_json_hash(legacy)]
    local = protocol_record(bank, words, "test-model", 2, None)
    assert cascade["candidate_bank_sha256"] == local["candidate_bank_sha256"]
    assert cascade["prompt_sha256"] == local["prompt_sha256"]
    assert stable_json_hash(local) != stable_json_hash(cascade)


def _failed_shard(tmp_path, failed_indices=(0,), failed_value=None):
    bank, words = tmp_path / "bank.csv", tmp_path / "words.csv"
    output = tmp_path / "judgments"
    _bank(bank)
    _words(words)
    protocol = protocol_record(bank, words, "test-model", 1, None)
    pairs = build_pairs(bank, words, 1, 0)
    shard = output / "shards/0000"
    shard.mkdir(parents=True)
    votes, resolutions = [], []
    for index, pair in enumerate(pairs):
        for judge in "ABC":
            votes.append(
                {
                    "word_normalized": pair["word_normalized"],
                    "feature_id": pair["feature_id"],
                    "judge_id": judge,
                    "prompt_hash": judge,
                }
            )
        resolutions.append(
            {
                "word_normalized": pair["word_normalized"],
                "feature_id": pair["feature_id"],
                "feature_text": pair["feature_text"],
                "final_feature_value": failed_value if index in failed_indices else 1,
                "resolution_method": (
                    "adjudicator_failed" if index in failed_indices else "unanimous"
                ),
            }
        )
    pd.DataFrame(votes).to_csv(shard / "feature_votes.csv", index=False)
    pd.DataFrame(resolutions).to_csv(shard / "feature_resolutions.csv", index=False)
    for name in ("feature_adjudication_votes.csv", "parse_errors.csv"):
        pd.DataFrame(columns=["empty"]).to_csv(shard / name, index=False)
    with pytest.raises(ValueError):
        validate_shard_complete(shard, pairs, protocol, 0)
    return bank, words, output, protocol


def test_authorized_failures_finalize_without_zero_fill_and_exclude_whole_context(
    tmp_path,
):
    from build_v4_matrices import load_and_validate_cells

    bank, words, output, protocol = _failed_shard(tmp_path)
    original_sidecar = (output / "shards/0000/v4_shard_manifest.json").read_bytes()
    with pytest.raises(ValueError):
        finalize(bank, words, output, protocol)
    finalize(bank, words, output, protocol, max_unresolved_cells=1)
    assert (
        output / "shards/0000/v4_shard_manifest.json"
    ).read_bytes() == original_sidecar
    manifest_path = output / "judgment_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["complete"] and not manifest["judgments_complete"]
    assert manifest["status"] == "finalized_with_unresolved_exclusions"
    assert manifest["resolved_cells"] == 5 and manifest["unresolved_cells"] == 1
    assert manifest["excluded_candidate_ids"] == ["v4_a"]
    resolved = pd.read_csv(output / "resolved_feature_values.csv")
    assert resolved.resolved_value.notna().all() and set(
        resolved.resolved_binary_locked_v2
    ) == {1}
    failed = pd.read_csv(output / "unresolved_cells.csv")
    assert len(failed) == 1 and failed.candidate_id.iloc[0] == "v4_a"
    _, working, _ = load_and_validate_cells(
        bank,
        output / "resolved_feature_values.csv",
        manifest_path,
        ["dog", "bird", "plane"],
    )
    assert len(working) == 3 and set(working.candidate_id) == {"v4_b"}
    failed.loc[0, "target_word"] = "other"
    failed.to_csv(output / "unresolved_cells.csv", index=False)
    with pytest.raises(ValueError, match="audit differs"):
        load_and_validate_cells(
            bank,
            output / "resolved_feature_values.csv",
            manifest_path,
            ["dog", "bird", "plane"],
        )


def test_authorized_cap_cannot_hide_extra_failed_scores(tmp_path):
    bank, words, output, protocol = _failed_shard(tmp_path, (0, 1))
    with pytest.raises(ValueError, match="exceed authorized cap"):
        finalize(bank, words, output, protocol, max_unresolved_cells=1)
    assert not (output / "judgment_manifest.json").exists()
    assert not (output / "resolved_feature_values.csv").exists()


def test_authorized_missing_scores_do_not_allow_out_of_range_values(tmp_path):
    bank, words, output, protocol = _failed_shard(tmp_path, failed_value=5)
    with pytest.raises(
        ValueError, match="only recorded adjudicator_failed missing scores"
    ):
        finalize(bank, words, output, protocol, max_unresolved_cells=1)
    assert not (output / "judgment_manifest.json").exists()


def test_tolerated_incompleteness_never_bypasses_protocol_mismatch(tmp_path):
    bank, words, output, protocol = _failed_shard(tmp_path)
    path = output / "shards/0000/v4_shard_manifest.json"
    sidecar = json.loads(path.read_text())
    sidecar["protocol_hash"] = "different"
    path.write_text(json.dumps(sidecar))
    with pytest.raises(ValueError, match="protocol hash mismatch"):
        finalize(bank, words, output, protocol, max_unresolved_cells=1)
