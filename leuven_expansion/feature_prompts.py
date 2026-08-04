"""Build isolated prompts for Leuven atomic judging and free generation."""
from __future__ import annotations

import hashlib
import json
import pathlib
from typing import Dict, List, Optional

# Fields that must NEVER appear in a judge user message
_FORBIDDEN_FIELDS = frozenset([
    "dataset",
    "list_id",
    "role",
    "critical_lure",
    "source_paper",
    "false_memory_rate",
    "mbas",
    "mgs",
    "isc-ci",
    "isc_ci",
    "drm",
    "gold_label",
    "item_id",
    "model_output",
    "benchmark",
])

_PROMPT_DIR = pathlib.Path(__file__).parent / "prompts"
GENERATION_PROMPT_VARIANTS = ("A", "B", "C")
_GENERATION_PROMPT_FILES_BY_VERSION = {
    "v3": {
        "A": "feature_generation_prompt_A_v3_original.txt",
        "B": "feature_generation_prompt_B_v3_concise.txt",
        "C": "feature_generation_prompt_C_v3_structured.txt",
    },
    "v3.1": {
        "A": "feature_generation_prompt_A_v3_1_faithful.txt",
        "B": "feature_generation_prompt_B_v3_1_first_to_mind.txt",
        "C": "feature_generation_prompt_C_v3_1_individual_participant.txt",
    },
}

# Fixed few-shot examples (same for all atomic prompts)
_DEFAULT_FEW_SHOT: List[Dict] = [
    {"word": "monkey", "feature_text": "is an animal", "feature_value": 4},
    {"word": "monkey", "feature_text": "can fly", "feature_value": 0},
    {"word": "hammer", "feature_text": "is a tool", "feature_value": 4},
    {"word": "hammer", "feature_text": "has feathers", "feature_value": 0},
]

_VALUE_SCALE_TEXT = (
    "0 = feature does not apply\n"
    "1 = weakly or rarely applies\n"
    "2 = moderately applies\n"
    "3 = strongly applies\n"
    "4 = highly central or diagnostic"
)


def load_prompt(path: str | pathlib.Path) -> str:
    with open(path) as f:
        return f.read().strip()


def load_default_prompts() -> Dict[str, str]:
    """Return the retained v2 atomic word-by-feature prompt set."""
    return load_prompts_by_version("v2")


def load_prompts_by_version(version: str) -> Dict[str, str]:
    """
    Return the three judge + adjudicator system prompts for a given prompt version.

    Parameters
    ----------
    version : "v2" for the retained atomic word-by-feature experiment

    Returns
    -------
    dict with keys "A", "B", "C", "adjudicator"
    """
    if version == "v2":
        return {
            "A": load_prompt(_PROMPT_DIR / "feature_judge_prompt_A_v2_production.txt"),
            "B": load_prompt(_PROMPT_DIR / "feature_judge_prompt_B_v2_production.txt"),
            "C": load_prompt(_PROMPT_DIR / "feature_judge_prompt_C_v2_production.txt"),
            "adjudicator": load_prompt(_PROMPT_DIR / "feature_adjudicator_prompt_v2_production.txt"),
        }
    if version in _GENERATION_PROMPT_FILES_BY_VERSION:
        raise ValueError(
            f"{version} is a free feature-generation task, not an atomic applicability "
            f"prompt set; use load_generation_prompts({version!r}) and "
            "leuven_expansion.generate_features"
        )
    raise ValueError(f"Unknown prompt version: {version!r}. Expected 'v2'.")


def load_generation_prompts(version: str = "v3") -> Dict[str, str]:
    """Return the three Leuven-style free-generation prompt conditions."""
    if version not in _GENERATION_PROMPT_FILES_BY_VERSION:
        raise ValueError(
            f"Unknown generation prompt version: {version!r}; "
            f"expected one of {sorted(_GENERATION_PROMPT_FILES_BY_VERSION)}"
        )
    return {
        variant: load_prompt(_PROMPT_DIR / filename)
        for variant, filename in _GENERATION_PROMPT_FILES_BY_VERSION[version].items()
    }


def load_generation_prompt(version: str = "v3", variant: str = "A") -> str:
    """Return one generation condition; variant A is the fidelity control."""
    if variant not in GENERATION_PROMPT_VARIANTS:
        raise ValueError(
            f"Unknown {version} generation prompt variant: {variant!r}; "
            f"expected one of {GENERATION_PROMPT_VARIANTS}"
        )
    return load_generation_prompts(version)[variant]


def build_generation_user_message(word_normalized: str) -> str:
    """Build a one-word free-generation trial without feature-schema leakage."""
    message = f"stimulus_word:\n{word_normalized}"
    assert_no_forbidden_fields(message)
    return message


def build_judge_user_message(
    word_normalized: str,
    feature_id: int,
    feature_text: str,
    few_shot_examples: Optional[List[Dict]] = None,
) -> str:
    """
    Build the user-turn content sent to a first-pass judge.
    Contains ONLY the atomic word × feature pair — no metadata.

    Parameters
    ----------
    word_normalized   : the target word (normalized)
    feature_id        : integer index of the feature in the Leuven schema
    feature_text      : the feature statement (e.g. "is an animal")
    few_shot_examples : fixed list of example dicts; uses defaults if None
    """
    examples = few_shot_examples if few_shot_examples is not None else _DEFAULT_FEW_SHOT

    lines = [
        f"target_word:\n{word_normalized}",
        f"\nfeature_id:\n{feature_id}",
        f"\nfeature_text:\n{feature_text}",
        f"\nvalue_scale:\n{_VALUE_SCALE_TEXT}",
    ]

    if examples:
        lines.append(f"\nfew_shot_examples:\n{json.dumps(examples, indent=2)}")

    return "\n".join(lines)


def build_adjudicator_user_message(
    word_normalized: str,
    feature_id: int,
    feature_text: str,
    first_pass_votes: List[Dict],
) -> str:
    """
    Build the user-turn content sent to an adjudicator.
    Includes the word × feature pair and the three first-pass judgments.
    Must not include DRM metadata or ISC-CI context.
    """
    lines = [
        f"target_word:\n{word_normalized}",
        f"\nfeature_id:\n{feature_id}",
        f"\nfeature_text:\n{feature_text}",
        f"\nvalue_scale:\n{_VALUE_SCALE_TEXT}",
        "\nfirst_pass_judgments:",
    ]
    for i, v in enumerate(first_pass_votes, 1):
        lines.append(
            f"  judge_{i}: feature_value={v.get('feature_value', 'N/A')}  "
            f"confidence={v.get('confidence', 0.0):.2f}  "
            f"ambiguous={v.get('ambiguous', False)}  "
            f"reason={v.get('reason', '')}"
        )
    return "\n".join(lines)


def load_verifier_prompt() -> str:
    """Return the system prompt for the positive verifier (v1)."""
    return load_prompt(_PROMPT_DIR / "feature_positive_verifier_prompt_v1.txt")


def build_verifier_user_message(
    word_normalized: str,
    feature_id: int,
    feature_text: str,
    resolved_feature_value: float,
    resolved_confidence: float,
    resolution_method: str,
    first_pass_votes: List[Dict],
) -> str:
    """
    Build the user-turn content sent to the positive verifier.

    Includes the atomic word × feature pair, the resolved value from the
    first-pass pipeline, and the individual judge votes for context.
    Must NOT include DRM metadata or ISC-CI context.
    """
    lines = [
        f"target_word:\n{word_normalized}",
        f"\nfeature_id:\n{feature_id}",
        f"\nfeature_text:\n{feature_text}",
        f"\nvalue_scale:\n{_VALUE_SCALE_TEXT}",
        f"\nresolved_feature_value: {resolved_feature_value}",
        f"resolved_confidence: {resolved_confidence:.2f}",
        f"resolution_method: {resolution_method}",
        "\nfirst_pass_judge_votes:",
    ]
    for v in first_pass_votes:
        judge_label = v.get("judge", v.get("judge_id", "?"))
        lines.append(
            f"  {judge_label}: feature_value={v.get('feature_value', 'N/A')}  "
            f"confidence={float(v.get('confidence', 0.0)):.2f}  "
            f"reason={v.get('reason', '')}"
        )
    return "\n".join(lines)


def prompt_hash(system_prompt: str, user_message: str) -> str:
    """Stable SHA-256 hash uniquely identifying the atomic prompt."""
    content = f"{system_prompt}\n---\n{user_message}"
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def assert_no_forbidden_fields(user_message: str) -> None:
    """
    Raise AssertionError if any forbidden metadata field name appears
    verbatim in the judge user message.
    """
    lower = user_message.lower()
    for field in _FORBIDDEN_FIELDS:
        assert field.lower() not in lower, (
            f"Forbidden field '{field}' found in judge user message."
        )
