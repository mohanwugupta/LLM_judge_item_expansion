"""
tests/test_leuven_feature_prompts.py

Tests for leuven_expansion/feature_prompts.py — prompt independence enforcement.
"""
import pytest

from leuven_expansion.feature_prompts import (
    build_judge_user_message,
    build_adjudicator_user_message,
    build_generation_user_message,
    assert_no_forbidden_fields,
    load_default_prompts,
    load_generation_prompt,
    load_generation_prompts,
    load_prompts_by_version,
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


def test_default_atomic_prompt_is_v2_production_protocol():
    prompts = load_default_prompts()
    assert "spontaneous production" in prompts["A"].lower()
    assert prompts == load_prompts_by_version("v2")


def test_prompt_B_is_conservative():
    prompts = load_default_prompts()
    lower = prompts["B"].lower()
    assert "conservative" in lower


def test_v3_cannot_be_loaded_as_an_atomic_prompt_set():
    with pytest.raises(ValueError, match="free feature-generation"):
        load_prompts_by_version("v3")


def test_no_prompt_contains_drm_fields():
    prompts = load_default_prompts()
    forbidden = ["drm", "list_id", "critical_lure", "mbas", "mgs", "isc-ci", "false_memory"]
    for key, text in prompts.items():
        lower = text.lower()
        for f in forbidden:
            assert f not in lower, f"Prompt {key} contains forbidden field '{f}'"


def test_v3_generation_prompt_recreates_free_listing_task():
    prompts = load_generation_prompts("v3")
    assert set(prompts) == {"A", "B", "C"}
    assert len(set(prompts.values())) == 3
    assert load_generation_prompt("v3") == prompts["A"]
    assert "preferably 10" in prompts["A"].lower()
    for prompt in prompts.values():
        lower = prompt.lower()
        assert "physical" in lower
        assert "function" in lower
        assert "background" in lower or "generally known facts" in lower
        assert '"target_word"' in prompt
        assert '"features"' in prompt
        assert "0 of 4" not in lower
        assert "feature applicability" not in lower
        assert "supplied feature list" in lower


def test_v3_generation_prompt_hashes_are_distinct_for_comparison():
    message = build_generation_user_message("dog")
    hashes = {
        prompt_hash(prompt, message)
        for prompt in load_generation_prompts().values()
    }
    assert len(hashes) == 3


def test_v3_generation_user_message_contains_only_the_stimulus_word():
    message = build_generation_user_message("dog")
    assert message == "stimulus_word:\ndog"
    assert "feature_id" not in message
    assert "feature_text" not in message
    assert "category" not in message
    assert_no_forbidden_fields(message)
