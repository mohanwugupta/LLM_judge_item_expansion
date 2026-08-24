import json

import pandas as pd

from leuven_expansion.v4 import candidate_inventory_hash
from run_v4_judgments import (
    CASCADE_RESOLUTION_METHOD,
    build_pairs,
    finalize,
    protocol_record,
    validate_shard_complete,
)


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
