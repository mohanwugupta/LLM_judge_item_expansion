"""
tests/test_leuven_feature_prompts.py

Tests for leuven_expansion/feature_prompts.py — prompt independence enforcement.
"""
import pytest

from leuven_expansion.feature_prompts import (
    build_judge_user_message,
    build_adjudicator_user_message,
    assert_no_forbidden_fields,
    load_default_prompts,
    prompt_hash,
    _FORBIDDEN_FIELDS,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_message():
    return build_judge_user_message(
        word_normalized="dog",
        feature_id=0,
        feature_text="is an animal",
    )


@pytest.fixture
def sample_votes():
    return [
        {"feature_value": 3.0, "confidence": 0.9, "ambiguous": False, "reason": "r1"},
        {"feature_value": 3.0, "confidence": 0.85, "ambiguous": False, "reason": "r2"},
        {"feature_value": 2.0, "confidence": 0.7, "ambiguous": False, "reason": "r3"},
    ]


# ── Atomic content tests ───────────────────────────────────────────────────────

def test_prompt_contains_exactly_one_word(sample_message):
    assert "dog" in sample_message
    assert sample_message.count("target_word") == 1


def test_prompt_contains_exactly_one_feature_id(sample_message):
    assert "feature_id" in sample_message
    assert sample_message.count("feature_id") == 1


def test_prompt_contains_exactly_one_feature_text(sample_message):
    assert "is an animal" in sample_message


def test_prompt_does_not_contain_additional_feature_statements():
    # Without few-shot examples, should have exactly one feature_text
    msg = build_judge_user_message("dog", 0, "is an animal", few_shot_examples=[])
    feature_count = msg.lower().count("feature_text")
    assert feature_count == 1


def test_prompt_does_not_ask_to_fill_vector(sample_message):
    lower = sample_message.lower()
    for forbidden in ["fill a vector", "semantic vector", "full vector", "matrix row"]:
        assert forbidden not in lower


def test_no_forbidden_fields_in_prompt(sample_message):
    """assert_no_forbidden_fields must not raise for clean prompt."""
    assert_no_forbidden_fields(sample_message)


def test_forbidden_field_drm_rejected():
    bad_message = "target_word: dog\nfeature_id: 0\nfeature_text: is an animal\ndrm: list1"
    with pytest.raises(AssertionError, match="drm"):
        assert_no_forbidden_fields(bad_message)


def test_forbidden_field_mbas_rejected():
    bad_message = "target_word: dog\nmbas: 0.5"
    with pytest.raises(AssertionError, match="mbas"):
        assert_no_forbidden_fields(bad_message)


def test_forbidden_field_isc_ci_rejected():
    bad_message = "target_word: dog\nisc_ci: 0.7"
    with pytest.raises(AssertionError, match="isc_ci"):
        assert_no_forbidden_fields(bad_message)


def test_forbidden_field_critical_lure_rejected():
    bad_message = "target_word: dog\ncritical_lure: true"
    with pytest.raises(AssertionError, match="critical_lure"):
        assert_no_forbidden_fields(bad_message)


def test_forbidden_field_false_memory_rate_rejected():
    bad_message = "target_word: dog\nfalse_memory_rate: 0.42"
    with pytest.raises(AssertionError):
        assert_no_forbidden_fields(bad_message)


# ── Multiple-word/feature isolation tests ────────────────────────────────────

def test_only_one_word_in_user_message():
    msg = build_judge_user_message("dog", 0, "is an animal", few_shot_examples=[])
    # Message should contain dog but not reference another target word
    assert "cat" not in msg
    assert "hammer" not in msg


def test_only_one_feature_text_in_user_message():
    msg = build_judge_user_message("dog", 0, "is an animal", few_shot_examples=[])
    assert "can fly" not in msg
    assert "is a tool" not in msg


# ── Prompt hash stability ─────────────────────────────────────────────────────

def test_prompt_hash_is_stable():
    msg = build_judge_user_message("dog", 0, "is an animal")
    h1 = prompt_hash("system", msg)
    h2 = prompt_hash("system", msg)
    assert h1 == h2


def test_prompt_hash_differs_for_different_words():
    m1 = build_judge_user_message("dog", 0, "is an animal")
    m2 = build_judge_user_message("cat", 0, "is an animal")
    assert prompt_hash("system", m1) != prompt_hash("system", m2)


def test_prompt_hash_differs_for_different_features():
    m1 = build_judge_user_message("dog", 0, "is an animal")
    m2 = build_judge_user_message("dog", 1, "can fly")
    assert prompt_hash("system", m1) != prompt_hash("system", m2)


# ── Adjudicator message tests ─────────────────────────────────────────────────

def test_adjudicator_message_contains_first_pass_values(sample_votes):
    msg = build_adjudicator_user_message("dog", 0, "is an animal", sample_votes)
    assert "judge_1" in msg
    assert "judge_2" in msg
    assert "judge_3" in msg


def test_adjudicator_message_contains_word_and_feature(sample_votes):
    msg = build_adjudicator_user_message("dog", 0, "is an animal", sample_votes)
    assert "dog" in msg
    assert "is an animal" in msg


def test_adjudicator_message_has_no_drm_metadata(sample_votes):
    msg = build_adjudicator_user_message("dog", 0, "is an animal", sample_votes)
    assert_no_forbidden_fields(msg)


# ── System prompt loading tests ───────────────────────────────────────────────

def test_load_default_prompts_returns_all_keys():
    prompts = load_default_prompts()
    assert set(prompts.keys()) == {"A", "B", "C", "adjudicator"}


def test_all_prompts_nonempty():
    prompts = load_default_prompts()
    for key, text in prompts.items():
        assert len(text) > 50, f"Prompt {key} seems too short"


def test_prompt_A_emphasizes_independence():
    prompts = load_default_prompts()
    lower = prompts["A"].lower()
    assert any(kw in lower for kw in ["independent", "do not consider", "only the provided"])


def test_prompt_B_is_conservative():
    prompts = load_default_prompts()
    lower = prompts["B"].lower()
    assert "conservative" in lower


def test_prompt_C_uses_checklist_style():
    prompts = load_default_prompts()
    lower = prompts["C"].lower()
    assert "check" in lower


def test_no_prompt_contains_drm_fields():
    prompts = load_default_prompts()
    forbidden = ["drm", "list_id", "critical_lure", "mbas", "mgs", "isc-ci", "false_memory"]
    for key, text in prompts.items():
        lower = text.lower()
        for f in forbidden:
            assert f not in lower, f"Prompt {key} contains forbidden field '{f}'"
