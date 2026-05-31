"""
leuven_expansion/feature_judge.py

Run three independent first-pass judgments for each word × feature pair
via the vLLM OpenAI-compatible backend.  One retry per failed call.
Each call is stateless and atomic.
"""
from __future__ import annotations

import hashlib
import logging
from typing import Dict, List, Optional, Tuple

from leuven_expansion.feature_schema import validate_judge_output, load_json_schema
from leuven_expansion.feature_prompts import (
    build_judge_user_message,
    prompt_hash,
)

logger = logging.getLogger(__name__)

JUDGE_IDS = ("A", "B", "C")

_JUDGE_RESPONSE_FORMAT: dict = {
    "type": "json_schema",
    "json_schema": {
        "name": "AtomicFeatureJudgment",
        "strict": True,
        "schema": load_json_schema(),
    },
}


def _row_hash(word_normalized: str, feature_id: int) -> str:
    content = f"{word_normalized}|{feature_id}"
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _call_once(
    client,
    system_prompt: str,
    user_message: str,
    model: str,
    temperature: float,
    max_tokens: int,
    word_normalized: str,
    feature_id: int,
) -> Tuple[Optional[Dict], Optional[str]]:
    """Single LLM call. Returns (parsed_record, parse_error)."""
    try:
        raw, _ = client.generate(
            system_prompt=system_prompt,
            user_prompt=user_message,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=_JUDGE_RESPONSE_FORMAT,
        )
    except Exception as e:
        return None, f"LLM call error: {e}"

    record, err = validate_judge_output(
        raw,
        expected_word=word_normalized,
        expected_feature_id=feature_id,
    )
    return record, err


def judge_pair(
    *,
    job_id: str,
    word_original: str,
    word_normalized: str,
    feature_id: int,
    feature_text: str,
    prompts: Dict[str, str],   # {"A": ..., "B": ..., "C": ...}
    client,
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 400,
) -> List[Dict]:
    """
    Run three independent first-pass judgments for one word × feature pair.

    Returns a list of vote dicts (one per judge variant A, B, C).
    Each vote dict contains all schema fields plus bookkeeping columns.

    IMPORTANT: each call is fully independent and stateless.
    No judge sees another judge's output. No chat history is used.
    """
    row_hash = _row_hash(word_normalized, feature_id)
    user_message = build_judge_user_message(word_normalized, feature_id, feature_text)

    votes: List[Dict] = []
    for judge_id in JUDGE_IDS:
        system_prompt = prompts[judge_id]
        phash = prompt_hash(system_prompt, user_message)

        record, err = _call_once(
            client, system_prompt, user_message, model, temperature, max_tokens,
            word_normalized, feature_id,
        )

        if err and record is None:
            # One retry with same atomic prompt — no previous output included
            logger.warning(
                "Judge %s word=%r feature_id=%d: first attempt failed (%s). Retrying.",
                judge_id, word_normalized, feature_id, err,
            )
            record, err = _call_once(
                client, system_prompt, user_message, model, temperature, max_tokens,
                word_normalized, feature_id,
            )

        vote: Dict = {
            "job_id": job_id,
            "word_original": word_original,
            "word_normalized": word_normalized,
            "row_hash": row_hash,
            "feature_id": feature_id,
            "feature_text": feature_text,
            "judge_id": judge_id,
            "judge_prompt_variant": judge_id,
            "judge_model": model,
            "raw_json": "",
            "parse_error": err or "",
            "prompt_hash": phash,
        }

        if record is not None:
            vote["feature_value"] = record.get("feature_value", 0.0)
            vote["confidence"] = record.get("confidence", 0.0)
            vote["ambiguous"] = record.get("ambiguous", False)
            vote["reason"] = record.get("reason", "")
        else:
            # Sentinel values when parsing failed
            vote["feature_value"] = None
            vote["confidence"] = 0.0
            vote["ambiguous"] = False
            vote["reason"] = ""

        votes.append(vote)

    return votes
