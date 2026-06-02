"""
leuven_expansion/positive_verifier.py

Second-stage candidate-positive verification layer (NOVA-style).

After the 3-judge + adjudicator pipeline resolves a final_feature_value,
any pair with final_feature_value >= POSITIVE_THRESHOLD is re-examined by
a single independent verifier LLM call.  If the verifier scores the pair
below VERIFICATION_THRESHOLD the resolution is downgraded to 0.

Public API
----------
POSITIVE_THRESHOLD : float = 1.0
VERIFICATION_THRESHOLD : float = 1.0

route_for_verification(final_feature_value, positive_threshold) -> str
    "candidate" or "not_candidate"

verify_positive(...) -> (record_dict, vote_dict)
    Run one verifier call and return the updated resolution record.

apply_verification_result(resolution_row, verifier_record) -> dict
    Merge verifier output fields into a resolution row.
"""
from __future__ import annotations

import json
import logging
import pathlib
import re
from typing import Any, Dict, List, Optional, Tuple

import jsonschema

from leuven_expansion.feature_schema import load_json_schema
from leuven_expansion.feature_prompts import build_verifier_user_message

logger = logging.getLogger(__name__)

# Defaults (callers can override per call)
POSITIVE_THRESHOLD: float = 1.0
VERIFICATION_THRESHOLD: float = 1.0

_SCHEMA_PATH = (
    pathlib.Path(__file__).parent / "schemas" / "positive_verification_schema_v1.json"
)
_VERIFIER_SCHEMA: Dict[str, Any] = load_json_schema(_SCHEMA_PATH)


# ---------------------------------------------------------------------------
# Routing helper
# ---------------------------------------------------------------------------

def route_for_verification(
    final_feature_value: Optional[float],
    positive_threshold: float = POSITIVE_THRESHOLD,
) -> str:
    """Return 'candidate' if the value warrants verification, else 'not_candidate'."""
    if final_feature_value is None:
        return "not_candidate"
    try:
        v = float(final_feature_value)
    except (TypeError, ValueError):
        return "not_candidate"
    return "candidate" if v >= positive_threshold else "not_candidate"


# ---------------------------------------------------------------------------
# Output validation
# ---------------------------------------------------------------------------

def validate_verifier_output(
    raw: str,
    expected_word: Optional[str] = None,
    expected_feature_id: Optional[int] = None,
    verification_threshold: float = VERIFICATION_THRESHOLD,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Parse *raw* verifier JSON, validate against positive_verification_schema_v1.json,
    and optionally cross-check word/feature_id.

    Returns (record, None) on success or (None, error_message) on failure.
    """
    text = raw.strip()
    if text.startswith("```"):
        lines = [l for l in text.splitlines() if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    # Normalise Python literals (None / True / False)
    text = re.sub(r'(?<!["\w])None(?!["\w])', 'null', text)
    text = re.sub(r'(?<!["\w])True(?!["\w])', 'true', text)
    text = re.sub(r'(?<!["\w])False(?!["\w])', 'false', text)

    try:
        obj = json.loads(text)
    except json.JSONDecodeError as e:
        return None, f"JSON parse error: {e}"

    # Coerce null retain_positive (key present but null) — will be recomputed below
    if "retain_positive" in obj and obj["retain_positive"] is None:
        obj["retain_positive"] = False

    try:
        jsonschema.validate(instance=obj, schema=_VERIFIER_SCHEMA)
    except jsonschema.ValidationError as e:
        return None, f"Schema validation error: {e.message}"

    # Cross-field checks
    if expected_word is not None:
        if obj.get("target_word", "").strip().lower() != expected_word.strip().lower():
            return None, (
                f"Returned target_word '{obj.get('target_word')}' does not match "
                f"expected '{expected_word}'"
            )

    if expected_feature_id is not None:
        if obj.get("feature_id") != expected_feature_id:
            return None, (
                f"Returned feature_id {obj.get('feature_id')} does not match "
                f"expected {expected_feature_id}"
            )

    # Recompute retain_positive from verified_feature_value for consistency
    obj["retain_positive"] = float(obj["verified_feature_value"]) >= verification_threshold

    # Truncate reason
    if len(obj.get("reason", "")) > 300:
        obj["reason"] = obj["reason"][:297] + "..."

    return obj, None


# ---------------------------------------------------------------------------
# Main verifier call
# ---------------------------------------------------------------------------

def verify_positive(
    *,
    job_id: str,
    word_normalized: str,
    feature_id: int,
    feature_text: str,
    resolved_feature_value: float,
    resolved_confidence: float,
    resolution_method: str,
    first_pass_votes: List[Dict],
    prompt: str,
    client,
    model: str,
    verification_threshold: float = VERIFICATION_THRESHOLD,
    temperature: float = 0.0,
    max_tokens: int = 400,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Run one verifier LLM call for a candidate-positive pair.

    Returns
    -------
    (resolution_update_dict, vote_dict)

    resolution_update_dict contains all fields needed to extend the
    resolution row (positive_verification_status, verified_feature_value,
    pre_verification_feature_value, final_feature_value,
    positive_verification_confidence, positive_verification_reason,
    needs_human_audit).
    """
    user_message = build_verifier_user_message(
        word_normalized=word_normalized,
        feature_id=feature_id,
        feature_text=feature_text,
        resolved_feature_value=resolved_feature_value,
        resolved_confidence=resolved_confidence,
        resolution_method=resolution_method,
        first_pass_votes=first_pass_votes,
    )

    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "positive_verification",
            "strict": True,
            "schema": _VERIFIER_SCHEMA,
        },
    }

    raw = _call_verifier(
        prompt=prompt,
        user_message=user_message,
        client=client,
        model=model,
        response_format=response_format,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    record_update, err = validate_verifier_output(
        raw,
        expected_word=word_normalized,
        expected_feature_id=feature_id,
        verification_threshold=verification_threshold,
    )

    if err is not None:
        # One retry
        logger.warning("Verifier parse error (attempt 1): %s — retrying", err)
        raw2 = _call_verifier(
            prompt=prompt,
            user_message=user_message,
            client=client,
            model=model,
            response_format=response_format,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        record_update, err2 = validate_verifier_output(
            raw2,
            expected_word=word_normalized,
            expected_feature_id=feature_id,
            verification_threshold=verification_threshold,
        )
        if err2 is not None:
            logger.error("Verifier repeated parse error: %s", err2)
            resolution_update = {
                "positive_verification_status": "parse_error",
                "verified_feature_value": None,
                "pre_verification_feature_value": resolved_feature_value,
                "final_feature_value": resolved_feature_value,  # keep original
                "positive_verification_confidence": None,
                "positive_verification_reason": err2,
                "needs_human_audit": True,
            }
            vote = {
                "job_id": job_id,
                "word_normalized": word_normalized,
                "feature_id": feature_id,
                "verified_feature_value": None,
                "retain_positive": None,
                "confidence": None,
                "reason": err2,
            }
            return resolution_update, vote

    retain = record_update["retain_positive"]
    status = "retained" if retain else "rejected"
    final_value = float(record_update["verified_feature_value"]) if retain else 0.0

    resolution_update = {
        "positive_verification_status": status,
        "verified_feature_value": record_update["verified_feature_value"],
        "pre_verification_feature_value": resolved_feature_value,
        "final_feature_value": final_value,
        "positive_verification_confidence": record_update["confidence"],
        "positive_verification_reason": record_update.get("reason", ""),
        "needs_human_audit": False,
    }
    vote = {
        "job_id": job_id,
        "word_normalized": word_normalized,
        "feature_id": feature_id,
        "verified_feature_value": record_update["verified_feature_value"],
        "retain_positive": record_update["retain_positive"],
        "confidence": record_update["confidence"],
        "reason": record_update.get("reason", ""),
    }
    return resolution_update, vote


# ---------------------------------------------------------------------------
# Row-level merge helper
# ---------------------------------------------------------------------------

def apply_verification_result(
    resolution_row: Dict[str, Any],
    verifier_record: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Return a new dict merging *verifier_record* fields into *resolution_row*.
    pre_verification_feature_value is set from the original final_feature_value.
    """
    updated = dict(resolution_row)
    updated["pre_verification_feature_value"] = resolution_row.get("final_feature_value")
    for key in (
        "positive_verification_status",
        "verified_feature_value",
        "positive_verification_confidence",
        "positive_verification_reason",
    ):
        updated[key] = verifier_record.get(key)
    # Override final_feature_value only when the verifier ran
    if verifier_record.get("positive_verification_status") not in ("not_candidate", "skipped", None):
        updated["final_feature_value"] = verifier_record.get("final_feature_value",
                                                               resolution_row.get("final_feature_value"))
    return updated


# ---------------------------------------------------------------------------
# Internal LLM call
# ---------------------------------------------------------------------------

def _call_verifier(
    *,
    prompt: str,
    user_message: str,
    client,
    model: str,
    response_format: Dict,
    temperature: float,
    max_tokens: int,
) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        response_format=response_format,
    )
    return response.choices[0].message.content
