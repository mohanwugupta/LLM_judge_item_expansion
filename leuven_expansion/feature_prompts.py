"""
leuven_expansion/feature_prompts.py

Build atomic judge prompts for word × feature judgments.
Enforces that no forbidden metadata enters the judge input.
"""
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
    """Return the three judge system prompts and adjudicator prompt keyed by role."""
    return {
        "A": load_prompt(_PROMPT_DIR / "feature_judge_prompt_A_v1.txt"),
        "B": load_prompt(_PROMPT_DIR / "feature_judge_prompt_B_v1.txt"),
        "C": load_prompt(_PROMPT_DIR / "feature_judge_prompt_C_v1.txt"),
        "adjudicator": load_prompt(_PROMPT_DIR / "feature_adjudicator_prompt_v1.txt"),
    }


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
