"""
tests/test_leuven_feature_judge.py

Tests for leuven_expansion/feature_judge.py — three-pass atomic judging.
"""
import json
from unittest.mock import MagicMock, patch, call

import pytest

from leuven_expansion.feature_judge import judge_pair, JUDGE_IDS


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_valid_raw(word="dog", fid=0, value=3.0) -> str:
    return json.dumps({
        "target_word": word,
        "feature_id": fid,
        "feature_value": value,
        "confidence": 0.9,
        "ambiguous": False,
        "reason": "test reason",
    })


def _make_mock_client(raw_output: str):
    client = MagicMock()
    client.generate.return_value = (raw_output, {})
    return client


def _make_prompts():
    return {"A": "system A", "B": "system B", "C": "system C", "adjudicator": "adj"}


# ── Three-pass tests ───────────────────────────────────────────────────────────

def test_three_judgments_produced():
    """Must produce exactly 3 votes (one per judge variant)."""
    client = _make_mock_client(_make_valid_raw())
    votes = judge_pair(
        job_id="test",
        word_original="dog",
        word_normalized="dog",
        feature_id=0,
        feature_text="is an animal",
        prompts=_make_prompts(),
        client=client,
        model="test-model",
    )
    assert len(votes) == 3


def test_judge_ids_are_a_b_c():
    client = _make_mock_client(_make_valid_raw())
    votes = judge_pair(
        job_id="test",
        word_original="dog",
        word_normalized="dog",
        feature_id=0,
        feature_text="is an animal",
        prompts=_make_prompts(),
        client=client,
        model="test-model",
    )
    ids = [v["judge_id"] for v in votes]
    assert set(ids) == {"A", "B", "C"}


def test_each_call_receives_only_one_word_and_feature():
    """
    Each call must use a user message containing exactly one target_word key
    and one feature_id key (the few-shot block may contain example words/features
    but must not contain additional *target* word/feature declarations).
    """
    calls_made = []

    def fake_generate(system_prompt, user_prompt, **kwargs):
        calls_made.append(user_prompt)
        return _make_valid_raw(), {}

    client = MagicMock()
    client.generate.side_effect = fake_generate

    judge_pair(
        job_id="test",
        word_original="dog",
        word_normalized="dog",
        feature_id=0,
        feature_text="is an animal",
        prompts=_make_prompts(),
        client=client,
        model="test-model",
    )

    assert len(calls_made) == 3
    for msg in calls_made:
        # Each message should declare exactly one target_word
        assert msg.count("target_word:") == 1
        # The target word is correct
        assert "dog" in msg
        # Feature ID is correct
        assert "feature_id:" in msg
        assert "is an animal" in msg


def test_invalid_json_triggers_one_retry():
    """On parse failure, judge retries exactly once with the same prompt."""
    # Each judge gets its own per-judge fail/succeed sequence
    # Judge A: fail, retry-fail; Judge B: fail, retry-succeed; Judge C: succeed
    # We want to confirm each failed first attempt causes exactly one retry.
    # Simplest: make every call return invalid JSON; expect 3 judges × 2 = 6 calls.
    client = MagicMock()
    client.generate.return_value = ("{invalid json}", {})

    votes = judge_pair(
        job_id="test",
        word_original="dog",
        word_normalized="dog",
        feature_id=0,
        feature_text="is an animal",
        prompts=_make_prompts(),
        client=client,
        model="test-model",
    )

    # 3 judges × (1 first attempt + 1 retry) = 6 total calls
    assert client.generate.call_count == 6
    # All votes should have parse_error set
    for v in votes:
        assert v["parse_error"] != ""


def test_retry_does_not_include_previous_invalid_output():
    """On retry, the same clean prompt is re-sent — no previous output appended."""
    user_messages_seen = []

    def fake_generate(system_prompt, user_prompt, **kwargs):
        user_messages_seen.append(user_prompt)
        # Always fail
        return "{invalid}", {}

    client = MagicMock()
    client.generate.side_effect = fake_generate

    judge_pair(
        job_id="test",
        word_original="dog",
        word_normalized="dog",
        feature_id=0,
        feature_text="is an animal",
        prompts=_make_prompts(),
        client=client,
        model="test-model",
    )

    # Each pair of retry messages (for same judge) should be identical
    # messages[0] and messages[1] are for judge A: first attempt + retry
    assert user_messages_seen[0] == user_messages_seen[1]  # Judge A retry same msg
    assert "{invalid}" not in user_messages_seen[1]


def test_raw_json_is_saved():
    client = _make_mock_client(_make_valid_raw())
    votes = judge_pair(
        job_id="test",
        word_original="dog",
        word_normalized="dog",
        feature_id=0,
        feature_text="is an animal",
        prompts=_make_prompts(),
        client=client,
        model="test-model",
    )
    # raw_json field exists in each vote (may be empty string if not captured at call)
    for v in votes:
        assert "raw_json" in v


def test_prompt_hash_is_stable_across_same_inputs():
    client = _make_mock_client(_make_valid_raw())
    votes1 = judge_pair(
        job_id="test",
        word_original="dog",
        word_normalized="dog",
        feature_id=0,
        feature_text="is an animal",
        prompts=_make_prompts(),
        client=client,
        model="test-model",
    )
    votes2 = judge_pair(
        job_id="test",
        word_original="dog",
        word_normalized="dog",
        feature_id=0,
        feature_text="is an animal",
        prompts=_make_prompts(),
        client=client,
        model="test-model",
    )
    for v1, v2 in zip(votes1, votes2):
        assert v1["prompt_hash"] == v2["prompt_hash"]


def test_prompt_hash_differs_for_different_features():
    client = _make_mock_client(_make_valid_raw(fid=0))
    votes_a = judge_pair(
        job_id="test",
        word_original="dog",
        word_normalized="dog",
        feature_id=0,
        feature_text="is an animal",
        prompts=_make_prompts(),
        client=client,
        model="test-model",
    )

    client2 = _make_mock_client(_make_valid_raw(fid=1))
    votes_b = judge_pair(
        job_id="test",
        word_original="dog",
        word_normalized="dog",
        feature_id=1,
        feature_text="can fly",
        prompts=_make_prompts(),
        client=client2,
        model="test-model",
    )

    # Hashes for judge A should differ between the two feature pairs
    assert votes_a[0]["prompt_hash"] != votes_b[0]["prompt_hash"]


def test_votes_contain_required_columns():
    client = _make_mock_client(_make_valid_raw())
    votes = judge_pair(
        job_id="test",
        word_original="dog",
        word_normalized="dog",
        feature_id=0,
        feature_text="is an animal",
        prompts=_make_prompts(),
        client=client,
        model="test-model",
    )
    required = {
        "job_id", "word_original", "word_normalized", "row_hash",
        "feature_id", "feature_text", "judge_id", "judge_prompt_variant",
        "judge_model", "feature_value", "confidence", "ambiguous",
        "reason", "raw_json", "parse_error", "prompt_hash",
    }
    for v in votes:
        assert required.issubset(v.keys()), f"Missing keys: {required - v.keys()}"
