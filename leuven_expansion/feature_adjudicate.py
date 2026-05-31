"""
leuven_expansion/feature_adjudicate.py

Detect disagreement in first-pass votes and run a three-adjudicator
panel when required.  Apply mean-based resolution; mark unresolved
cases for human audit.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from leuven_expansion.feature_schema import validate_judge_output, is_high_confidence
from leuven_expansion.feature_prompts import build_adjudicator_user_message

logger = logging.getLogger(__name__)

HIGH_CONFIDENCE_THRESHOLD = 0.80


def _valid_values(votes: List[Dict]) -> List[float]:
    """Return list of non-None feature_value floats from first-pass votes."""
    return [
        float(v["feature_value"])
        for v in votes
        if v.get("feature_value") is not None and v.get("parse_error", "") == ""
    ]


def _needs_adjudication(votes: List[Dict]) -> Tuple[bool, str]:
    """
    Determine whether this word × feature pair needs adjudication.
    Returns (needs_adjudication, reason_string).
    """
    # Any unresolved parse errors
    if any(v.get("parse_error") for v in votes):
        return True, "parse_error"

    values = _valid_values(votes)
    if len(values) < 3:
        return True, "insufficient_valid_votes"

    rng = max(values) - min(values)
    if rng >= 2:
        return True, f"range_{rng:.1f}"

    # 0 vs >=3 extreme disagreement
    if min(values) == 0 and max(values) >= 3:
        return True, "zero_vs_high"

    # High-confidence dissent: two agree, one dissents with high confidence
    # (two judges same value, one judges differently with conf >= 0.80)
    if len(set(round(v) for v in values)) > 1:
        rounded = [round(v) for v in values]
        majority_val = max(set(rounded), key=rounded.count)
        minority_votes = [
            v for v in votes
            if v.get("feature_value") is not None
            and round(float(v["feature_value"])) != majority_val
        ]
        if minority_votes:
            max_dissent_conf = max(
                float(v.get("confidence", 0.0)) for v in minority_votes
            )
            if is_high_confidence(max_dissent_conf, HIGH_CONFIDENCE_THRESHOLD):
                return True, "high_confidence_dissent"

    # At least two judges mark ambiguous=True
    ambiguous_count = sum(bool(v.get("ambiguous", False)) for v in votes)
    if ambiguous_count >= 2:
        return True, "ambiguous_majority"

    return False, ""


def _resolve_first_pass(votes: List[Dict]) -> Dict:
    """
    Resolve first-pass votes without adjudication (clean agreement case).
    Returns a resolution dict with final_feature_value and metadata.
    """
    values = _valid_values(votes)
    if not values:
        return {
            "final_feature_value": None,
            "resolution_method": "failed",
            "needs_human_audit": True,
        }

    rounded = [round(v) for v in values]
    if len(set(rounded)) == 1:
        return {
            "final_feature_value": values[0],
            "resolution_method": "unanimous",
            "needs_human_audit": False,
        }

    mean_val = sum(values) / len(values)
    return {
        "final_feature_value": mean_val,
        "resolution_method": "mean_small_disagreement",
        "needs_human_audit": False,
    }


def adjudicate_pair(
    *,
    job_id: str,
    word_normalized: str,
    feature_id: int,
    feature_text: str,
    first_pass_votes: List[Dict],
    prompts: Dict[str, str],   # {"adjudicator": ...}
    client,
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 200,
) -> Tuple[Dict, List[Dict]]:
    """
    Run the adjudication panel for one word × feature pair.

    Returns:
      - resolution dict (final_feature_value, resolution_method, needs_human_audit)
      - list of adjudicator vote dicts (for feature_adjudication_votes.csv)
    """
    user_message = build_adjudicator_user_message(
        word_normalized, feature_id, feature_text, first_pass_votes
    )
    system_prompt = prompts["adjudicator"]

    adj_votes: List[Dict] = []
    adj_values: List[float] = []

    for adj_idx in range(1, 4):
        raw_json = ""
        try:
            raw, _ = client.generate(
                system_prompt=system_prompt,
                user_prompt=user_message,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            raw_json = raw
            record, err = validate_judge_output(
                raw,
                expected_word=word_normalized,
                expected_feature_id=feature_id,
            )
        except Exception as e:
            record, err = None, str(e)

        vote: Dict = {
            "job_id": job_id,
            "word_normalized": word_normalized,
            "feature_id": feature_id,
            "feature_text": feature_text,
            "row_hash": first_pass_votes[0]["row_hash"] if first_pass_votes else "",
            "adjudicator_idx": adj_idx,
            "feature_value": record["feature_value"] if record else None,
            "confidence": record["confidence"] if record else 0.0,
            "ambiguous": record["ambiguous"] if record else False,
            "reason": record["reason"] if record else (err or ""),
            "parse_error": err or "",
            "raw_json": raw_json,
        }
        adj_votes.append(vote)
        if record is not None:
            adj_values.append(float(record["feature_value"]))

    # Resolve adjudicator values
    needs_human = False
    if len(adj_values) >= 2:
        # Check if at least two agree within tolerance of 0.5
        pairs_agree = any(
            abs(adj_values[i] - adj_values[j]) <= 0.5
            for i in range(len(adj_values))
            for j in range(i + 1, len(adj_values))
        )
        if pairs_agree:
            # Mean of agreeing pair(s)
            agreeing = []
            for i in range(len(adj_values)):
                for j in range(i + 1, len(adj_values)):
                    if abs(adj_values[i] - adj_values[j]) <= 0.5:
                        agreeing.extend([adj_values[i], adj_values[j]])
            final_val = sum(agreeing) / len(agreeing)
            method = "adjudicator_agree"
        else:
            # All adjudicators substantially disagree — mark for human audit
            final_val = sum(adj_values) / len(adj_values)
            method = "adjudicator_disagree_provisional_mean"
            needs_human = True
    elif len(adj_values) == 1:
        final_val = adj_values[0]
        method = "adjudicator_single"
    else:
        final_val = None
        method = "adjudicator_failed"
        needs_human = True

    resolution = {
        "final_feature_value": final_val,
        "resolution_method": method,
        "needs_human_audit": needs_human,
        "adjudicated": True,
    }
    return resolution, adj_votes


def resolve_first_pass(
    *,
    job_id: str,
    word_normalized: str,
    feature_id: int,
    feature_text: str,
    votes: List[Dict],
    prompts: Dict[str, str],
    client,
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 200,
    _skip_adjudication: bool = False,
) -> Tuple[Dict, List[Dict]]:
    """
    Entry point for resolving one word × feature pair.

    Checks whether adjudication is needed, runs it if so, or
    resolves the first-pass directly.

    Returns (resolution_dict, adjudication_votes).
    """
    needs_adj, reason = _needs_adjudication(votes)

    if not needs_adj or _skip_adjudication:
        resolution = _resolve_first_pass(votes)
        resolution["adjudicated"] = False
        resolution["adjudication_trigger"] = reason if needs_adj else ""
        return resolution, []

    logger.info(
        "Adjudication triggered for word=%r feature_id=%d reason=%s",
        word_normalized, feature_id, reason,
    )
    resolution, adj_votes = adjudicate_pair(
        job_id=job_id,
        word_normalized=word_normalized,
        feature_id=feature_id,
        feature_text=feature_text,
        first_pass_votes=votes,
        prompts=prompts,
        client=client,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    resolution["adjudication_trigger"] = reason
    return resolution, adj_votes
