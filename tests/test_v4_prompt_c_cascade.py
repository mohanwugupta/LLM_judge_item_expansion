import hashlib
import json
import re
from unittest.mock import MagicMock

import pandas as pd

from leuven_expansion.cascade_jobs import (
    CASCADE_RESOLUTION_METHOD,
    route_prompt_c,
    run_prompt_c_cascade_jobs,
)
from leuven_expansion.run_jobs import RESOLUTION_COLUMNS, VOTE_COLUMNS
from leuven_expansion.v4 import candidate_inventory_hash
from run_v4_judgments import (
    KNOWN_EXECUTED_V4_LEGACY_PROTOCOL_HASH,
    protocol_record,
    validate_shard_complete,
    validate_shard_resume,
)


PROMPTS = {
    "A": "prompt A",
    "B": "prompt B",
    "C": "prompt C",
    "adjudicator": "adjudicator",
}


def _pair(word, feature_id, feature_text):
    return {
        "word_original": word,
        "word_normalized": word,
        "feature_id": feature_id,
        "feature_text": feature_text,
    }


def _client():
    client = MagicMock()

    def generate(system_prompt, user_prompt, **_):
        word = re.search(r"target_word:\s*(.+)", user_prompt).group(1).strip()
        feature_id = int(re.search(r"feature_id:\s*(\d+)", user_prompt).group(1))
        value = 0 if system_prompt == "prompt C" and feature_id == 0 else 2
        return (
            json.dumps(
                {
                    "target_word": word,
                    "feature_id": feature_id,
                    "feature_value": value,
                    "confidence": 0.95,
                    "ambiguous": False,
                    "reason": "test",
                }
            ),
            {},
        )

    client.generate.side_effect = generate
    return client


def _legacy_vote(job_id, pair, judge, value=1):
    row_hash = hashlib.sha256(
        f"{pair['word_normalized']}|{pair['feature_id']}".encode()
    ).hexdigest()
    return {
        "job_id": job_id,
        "word_original": pair["word_original"],
        "word_normalized": pair["word_normalized"],
        "row_hash": row_hash,
        "feature_id": pair["feature_id"],
        "feature_text": pair["feature_text"],
        "judge_id": judge,
        "judge_prompt_variant": judge,
        "judge_model": "test-model",
        "feature_value": value,
        "confidence": 0.95,
        "ambiguous": False,
        "reason": "legacy",
        "raw_json": "",
        "parse_error": "",
        "prompt_hash": f"hash-{judge}",
    }


def test_prompt_c_routing_rule():
    base = {
        "feature_value": 0,
        "confidence": 0.95,
        "ambiguous": False,
        "parse_error": "",
    }
    assert not route_prompt_c(base)
    assert route_prompt_c(base | {"feature_value": 1})
    assert route_prompt_c(base | {"ambiguous": True})
    assert route_prompt_c(base | {"confidence": 0.79})
    assert route_prompt_c(base | {"parse_error": "invalid JSON"})
    assert route_prompt_c(base | {"feature_value": None})


def test_cascade_screens_every_cell_and_routes_only_nonnegative_c(tmp_path):
    pairs = [_pair("dog", 0, "has fur"), _pair("bird", 1, "can fly")]
    client = _client()
    run_prompt_c_cascade_jobs(
        job_id="test",
        pairs=pairs,
        prompts=PROMPTS,
        client=client,
        model="test-model",
        output_dir=tmp_path,
        max_workers=2,
        resume=True,
    )
    votes = pd.read_csv(tmp_path / "feature_votes.csv")
    resolutions = pd.read_csv(tmp_path / "feature_resolutions.csv")
    assert client.generate.call_count == 4
    assert set(votes.loc[votes["feature_id"].eq(0), "judge_id"]) == {"C"}
    assert set(votes.loc[votes["feature_id"].eq(1), "judge_id"]) == {"A", "B", "C"}
    negative = resolutions.loc[resolutions["feature_id"].eq(0)].iloc[0]
    assert negative["final_feature_value"] == 0
    assert negative["resolution_method"] == CASCADE_RESOLUTION_METHOD
    protocol = {
        "candidate_inventory_hash": "inventory",
        "cascade_confidence_threshold": 0.8,
    }
    manifest = validate_shard_complete(tmp_path, pairs, protocol, 0)
    assert manifest["prompt_c_only_cells"] == 1
    assert manifest["full_panel_vote_cells"] == 1
    assert manifest["complete"]

    resumed_client = _client()
    run_prompt_c_cascade_jobs(
        job_id="test",
        pairs=pairs,
        prompts=PROMPTS,
        client=resumed_client,
        model="test-model",
        output_dir=tmp_path,
        max_workers=2,
        resume=True,
    )
    assert resumed_client.generate.call_count == 0
    assert len(pd.read_csv(tmp_path / "feature_votes.csv")) == 4
    assert len(pd.read_csv(tmp_path / "feature_resolutions.csv")) == 2


def test_cascade_recovers_missing_and_invalid_legacy_resolutions_without_rejudging(tmp_path):
    job_id = "legacy"
    pairs = [_pair("dog", 0, "has fur"), _pair("bird", 1, "can fly")]
    votes = [
        _legacy_vote(job_id, pair, judge)
        for pair in pairs
        for judge in "ABC"
    ]
    pd.DataFrame(votes, columns=VOTE_COLUMNS).to_csv(
        tmp_path / "feature_votes.csv", index=False
    )
    invalid_resolution = {
        "job_id": job_id,
        "word_normalized": "bird",
        "feature_id": 1,
        "feature_text": "can fly",
        "final_feature_value": None,
        "resolution_method": "adjudicator_failed",
        "needs_human_audit": True,
        "adjudicated": True,
        "adjudication_trigger": "high_confidence_dissent",
    }
    pd.DataFrame([invalid_resolution], columns=RESOLUTION_COLUMNS).to_csv(
        tmp_path / "feature_resolutions.csv", index=False
    )
    client = _client()
    run_prompt_c_cascade_jobs(
        job_id=job_id,
        pairs=pairs,
        prompts=PROMPTS,
        client=client,
        model="test-model",
        output_dir=tmp_path,
        max_workers=2,
        resume=True,
    )
    assert client.generate.call_count == 0
    resolutions = pd.read_csv(tmp_path / "feature_resolutions.csv")
    assert len(resolutions) == 2
    assert resolutions["final_feature_value"].notna().all()
    assert set(resolutions["resolution_method"]) == {"unanimous"}
    recovery = pd.read_csv(tmp_path / "feature_resolution_recovery.csv")
    assert len(recovery) == 1
    assert recovery.iloc[0]["resolution_method"] == "adjudicator_failed"
    run_prompt_c_cascade_jobs(
        job_id=job_id,
        pairs=pairs,
        prompts=PROMPTS,
        client=client,
        model="test-model",
        output_dir=tmp_path,
        max_workers=2,
        resume=True,
    )
    assert len(pd.read_csv(tmp_path / "feature_resolution_recovery.csv")) == 1


def test_legacy_shard_manifest_migrates_to_cascade_protocol(tmp_path):
    bank = tmp_path / "bank.csv"
    words = tmp_path / "words.csv"
    frame = pd.DataFrame(
        {
            "candidate_index": [0],
            "candidate_id": ["v4_a"],
            "canonical_feature_text": ["has fur"],
            "source_words": [json.dumps(["dog"])],
        }
    )
    frame["candidate_inventory_hash"] = candidate_inventory_hash(frame)
    frame.to_csv(bank, index=False)
    pd.DataFrame({"word": ["dog"]}).to_csv(words, index=False)
    protocol = protocol_record(bank, words, "test-model", 1, None)
    assert (
        KNOWN_EXECUTED_V4_LEGACY_PROTOCOL_HASH
        not in protocol["compatible_legacy_protocol_hashes"]
    )
    legacy_hash = protocol["legacy_full_panel_protocol_hash"]
    sidecar = tmp_path / "v4_shard_manifest.json"
    sidecar.write_text(
        json.dumps(
            {
                "candidate_inventory_hash": protocol["candidate_inventory_hash"],
                "protocol_hash": legacy_hash,
                "shard_index": 0,
            }
        )
    )
    validate_shard_resume(tmp_path, protocol, 0)
    migrated = json.loads(sidecar.read_text())
    assert migrated["protocol_hash"] != legacy_hash
    assert migrated["migrated_from_protocol_hash"] == legacy_hash
