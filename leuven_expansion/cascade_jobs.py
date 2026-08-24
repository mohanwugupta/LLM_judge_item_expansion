"""Resumable prompt-C cascade runner over legacy and new atomic judgments."""
from __future__ import annotations

import csv
import datetime
import json
import logging
import math
import pathlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from typing import Any, Dict, Iterable, List, Sequence

import pandas as pd

from leuven_expansion.feature_adjudicate import resolve_first_pass
from leuven_expansion.feature_judge import judge_pair
from leuven_expansion.run_jobs import (
    PARSE_ERROR_COLUMNS,
    RESOLUTION_COLUMNS,
    VERIFIER_VOTE_COLUMNS,
    VOTE_COLUMNS,
    _add_file_log_handler,
)


logger = logging.getLogger(__name__)

ADJUDICATION_COLUMNS = [
    "job_id",
    "word_normalized",
    "feature_id",
    "feature_text",
    "row_hash",
    "adjudicator_idx",
    "feature_value",
    "confidence",
    "ambiguous",
    "reason",
    "parse_error",
    "raw_json",
]

RECOVERY_COLUMNS = [*RESOLUTION_COLUMNS, "recovery_reason", "recovered_at"]
CASCADE_RESOLUTION_METHOD = "prompt_c_high_confidence_negative"


def _is_missing(value: object) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if _is_missing(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes"}


def route_prompt_c(vote: Dict[str, Any], confidence_threshold: float = 0.80) -> bool:
    """Route every non-confident-negative prompt-C result to the full panel."""
    value = vote.get("feature_value")
    confidence = vote.get("confidence")
    if vote.get("parse_error") or _is_missing(value) or _is_missing(confidence):
        return True
    return (
        float(value) > 0
        or _as_bool(vote.get("ambiguous"))
        or float(confidence) < confidence_threshold
    )


def _normalize_vote(row: Dict[str, Any]) -> Dict[str, Any]:
    vote = {key: row.get(key, "") for key in VOTE_COLUMNS}
    for key in ["feature_value", "confidence"]:
        if _is_missing(vote[key]):
            vote[key] = None
        else:
            vote[key] = float(vote[key])
    vote["feature_id"] = int(vote["feature_id"])
    vote["ambiguous"] = _as_bool(vote["ambiguous"])
    vote["parse_error"] = "" if _is_missing(vote["parse_error"]) else str(vote["parse_error"])
    return vote


def _valid_resolution_mask(frame: pd.DataFrame) -> pd.Series:
    values = pd.to_numeric(frame["final_feature_value"], errors="coerce")
    return values.notna() & values.between(0, 4)


def _quarantine_invalid_resolutions(
    resolutions_csv: pathlib.Path, recovery_csv: pathlib.Path
) -> pd.DataFrame:
    if not resolutions_csv.exists() or resolutions_csv.stat().st_size == 0:
        return pd.DataFrame(columns=RESOLUTION_COLUMNS)
    frame = pd.read_csv(resolutions_csv, low_memory=False)
    if frame.empty:
        return frame
    if frame.duplicated(["word_normalized", "feature_id"]).any():
        raise ValueError("Cannot resume with duplicate feature resolutions")
    valid = _valid_resolution_mask(frame)
    if valid.all():
        return frame

    invalid = frame.loc[~valid].copy()
    invalid["recovery_reason"] = "missing_or_out_of_range_final_feature_value"
    invalid["recovered_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    recovery_exists = recovery_csv.exists() and recovery_csv.stat().st_size > 0
    to_record = invalid
    if recovery_exists:
        recorded = pd.read_csv(recovery_csv, usecols=["word_normalized", "feature_id"])
        recorded_keys = set(
            zip(
                recorded["word_normalized"].astype(str),
                pd.to_numeric(recorded["feature_id"], errors="raise").astype(int),
            )
        )
        invalid_keys = list(
            zip(
                invalid["word_normalized"].astype(str),
                pd.to_numeric(invalid["feature_id"], errors="raise").astype(int),
            )
        )
        to_record = invalid.loc[
            [key not in recorded_keys for key in invalid_keys]
        ]
    if not to_record.empty:
        to_record.reindex(columns=RECOVERY_COLUMNS).to_csv(
            recovery_csv,
            mode="a",
            header=not recovery_exists,
            index=False,
        )
    retained = frame.loc[valid].copy()
    temporary = resolutions_csv.with_suffix(".csv.tmp")
    retained.to_csv(temporary, index=False)
    temporary.replace(resolutions_csv)
    logger.warning(
        "Quarantined %d invalid resolutions to %s before resume",
        len(invalid),
        recovery_csv,
    )
    return retained


def _load_unresolved_votes(
    votes_csv: pathlib.Path, completed_keys: set[tuple[str, int]]
) -> dict[tuple[str, int], dict[str, Dict[str, Any]]]:
    if not votes_csv.exists() or votes_csv.stat().st_size == 0:
        return {}
    frame = pd.read_csv(votes_csv, low_memory=False)
    frame["feature_id"] = pd.to_numeric(frame["feature_id"], errors="raise").astype(int)
    frame["key"] = list(zip(frame["word_normalized"].astype(str), frame["feature_id"]))
    frame = frame.loc[~frame["key"].isin(completed_keys)]
    if frame.duplicated(["word_normalized", "feature_id", "judge_id"]).any():
        raise ValueError("Cannot resume with duplicate first-pass judge votes")
    result: dict[tuple[str, int], dict[str, Dict[str, Any]]] = {}
    for row in frame.drop(columns="key").to_dict(orient="records"):
        key = (str(row["word_normalized"]), int(row["feature_id"]))
        result.setdefault(key, {})[str(row["judge_id"])] = _normalize_vote(row)
    return result


def _resolution_row(
    job_id: str,
    pair: Dict[str, Any],
    resolution: Dict[str, Any],
) -> Dict[str, Any]:
    row = {
        "job_id": job_id,
        "word_normalized": pair["word_normalized"],
        "feature_id": pair["feature_id"],
        "feature_text": pair["feature_text"],
        **{
            key: resolution.get(key, "")
            for key in RESOLUTION_COLUMNS
            if key
            not in {
                "job_id",
                "word_normalized",
                "feature_id",
                "feature_text",
                "pre_verification_feature_value",
                "verified_feature_value",
                "positive_verification_status",
                "positive_verification_confidence",
                "positive_verification_reason",
            }
        },
        "pre_verification_feature_value": "",
        "verified_feature_value": "",
        "positive_verification_status": "not_candidate",
        "positive_verification_confidence": "",
        "positive_verification_reason": "",
    }
    return row


def _write_manifest(
    output_dir: pathlib.Path,
    *,
    job_id: str,
    model: str,
    total_pairs: int,
    existing_resolutions: int,
    pending_pairs: int,
    confidence_threshold: float,
    started_at: str,
    finished_at: str | None = None,
    counters: Dict[str, int] | None = None,
) -> None:
    record: dict[str, Any] = {
        "job_id": job_id,
        "model": model,
        "execution_mode": "prompt_c_cascade",
        "cascade_confidence_threshold": confidence_threshold,
        "n_total_pairs": total_pairs,
        "n_skipped_resume": existing_resolutions,
        "n_pending": pending_pairs,
        "n_completed_this_run": 0,
        "n_parse_errors": 0,
        "started_at": started_at,
        "finished_at": finished_at,
        "schema_version": "leuven_feature_schema_v1",
    }
    if counters:
        record.update(counters)
        record["n_completed_this_run"] = counters.get("completed_pairs", 0)
        record["n_parse_errors"] = counters.get("new_parse_errors", 0)
    (output_dir / "manifest.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _batches(values: Sequence[Dict[str, Any]], size: int) -> Iterable[Sequence[Dict[str, Any]]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def run_prompt_c_cascade_jobs(
    *,
    job_id: str,
    pairs: List[Dict[str, Any]],
    prompts: Dict[str, str],
    client: Any,
    model: str,
    output_dir: pathlib.Path,
    temperature: float = 0.0,
    max_tokens: int = 400,
    max_workers: int = 16,
    resume: bool = True,
    confidence_threshold: float = 0.80,
) -> pathlib.Path:
    """Resume legacy cells and apply prompt-C cascade scheduling to unresolved cells."""
    if not resume and any(output_dir.glob("*.csv")):
        raise ValueError("Cascade production refuses to overwrite existing CSV outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    votes_csv = output_dir / "feature_votes.csv"
    adj_votes_csv = output_dir / "feature_adjudication_votes.csv"
    resolutions_csv = output_dir / "feature_resolutions.csv"
    parse_errors_csv = output_dir / "parse_errors.csv"
    verifier_votes_csv = output_dir / "positive_verification_votes.csv"
    recovery_csv = output_dir / "feature_resolution_recovery.csv"
    file_handler = _add_file_log_handler(output_dir, job_id)
    started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

    resolutions = _quarantine_invalid_resolutions(resolutions_csv, recovery_csv)
    completed_keys = (
        set(
            zip(
                resolutions["word_normalized"].astype(str),
                pd.to_numeric(resolutions["feature_id"], errors="raise").astype(int),
            )
        )
        if not resolutions.empty
        else set()
    )
    pending = [
        pair
        for pair in pairs
        if (str(pair["word_normalized"]), int(pair["feature_id"])) not in completed_keys
    ]
    existing_votes = _load_unresolved_votes(votes_csv, completed_keys)
    logger.info(
        "Cascade job %s: %d total, %d resolved, %d pending, %d pending cells with votes",
        job_id,
        len(pairs),
        len(completed_keys),
        len(pending),
        len(existing_votes),
    )
    _write_manifest(
        output_dir,
        job_id=job_id,
        model=model,
        total_pairs=len(pairs),
        existing_resolutions=len(completed_keys),
        pending_pairs=len(pending),
        confidence_threshold=confidence_threshold,
        started_at=started_at,
    )

    counters = {
        "completed_pairs": 0,
        "prompt_c_only_resolutions": 0,
        "routed_full_panel_resolutions": 0,
        "recovered_existing_vote_cells": 0,
        "new_first_pass_calls": 0,
        "new_adjudication_calls": 0,
        "new_parse_errors": 0,
    }

    def judge_variants(pair: Dict[str, Any], variants: Sequence[str]) -> List[Dict[str, Any]]:
        if not variants:
            return []
        return judge_pair(
            job_id=job_id,
            word_original=pair["word_original"],
            word_normalized=pair["word_normalized"],
            feature_id=pair["feature_id"],
            feature_text=pair["feature_text"],
            prompts=prompts,
            client=client,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            judge_ids=variants,
        )

    def process(pair: Dict[str, Any]) -> Dict[str, Any]:
        key = (str(pair["word_normalized"]), int(pair["feature_id"]))
        votes_by_judge = dict(existing_votes.get(key, {}))
        had_existing_votes = bool(votes_by_judge)
        new_votes: list[Dict[str, Any]] = []

        if "C" not in votes_by_judge:
            c_vote = judge_variants(pair, ["C"])[0]
            votes_by_judge["C"] = c_vote
            new_votes.append(c_vote)

        c_vote = votes_by_judge["C"]
        has_legacy_panel_votes = "A" in votes_by_judge or "B" in votes_by_judge
        routed = has_legacy_panel_votes or route_prompt_c(c_vote, confidence_threshold)
        if not routed:
            resolution = {
                "final_feature_value": 0.0,
                "resolution_method": CASCADE_RESOLUTION_METHOD,
                "needs_human_audit": False,
                "adjudicated": False,
                "adjudication_trigger": "",
                "confidence": c_vote.get("confidence", ""),
                "ambiguous": c_vote.get("ambiguous", False),
            }
            return {
                "new_votes": new_votes,
                "adjudication_votes": [],
                "resolution": _resolution_row(job_id, pair, resolution),
                "routed": False,
                "recovered": had_existing_votes,
            }

        missing_panel = [judge for judge in ("A", "B") if judge not in votes_by_judge]
        for vote in judge_variants(pair, missing_panel):
            votes_by_judge[str(vote["judge_id"])] = vote
            new_votes.append(vote)
        votes = [votes_by_judge[judge] for judge in ("A", "B", "C")]
        resolution, adjudication_votes = resolve_first_pass(
            job_id=job_id,
            word_normalized=pair["word_normalized"],
            feature_id=pair["feature_id"],
            feature_text=pair["feature_text"],
            votes=votes,
            prompts=prompts,
            client=client,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return {
            "new_votes": new_votes,
            "adjudication_votes": adjudication_votes,
            "resolution": _resolution_row(job_id, pair, resolution),
            "routed": True,
            "recovered": had_existing_votes,
        }

    paths_and_columns = [
        (votes_csv, VOTE_COLUMNS),
        (adj_votes_csv, ADJUDICATION_COLUMNS),
        (resolutions_csv, RESOLUTION_COLUMNS),
        (parse_errors_csv, PARSE_ERROR_COLUMNS),
        (verifier_votes_csv, VERIFIER_VOTE_COLUMNS),
    ]
    try:
        with ExitStack() as stack:
            writers: dict[pathlib.Path, tuple[Any, csv.DictWriter]] = {}
            for path, columns in paths_and_columns:
                exists = path.exists() and path.stat().st_size > 0
                handle = stack.enter_context(path.open("a", newline="", encoding="utf-8"))
                writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
                if not exists:
                    writer.writeheader()
                    handle.flush()
                writers[path] = (handle, writer)

            batch_size = max(max_workers * 4, max_workers)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for batch in _batches(pending, batch_size):
                    futures = {executor.submit(process, pair): pair for pair in batch}
                    for future in as_completed(futures):
                        pair = futures[future]
                        try:
                            result = future.result()
                        except Exception:
                            logger.exception(
                                "Cascade cell failed word=%r feature_id=%s",
                                pair["word_normalized"],
                                pair["feature_id"],
                            )
                            continue
                        for vote in result["new_votes"]:
                            writers[votes_csv][1].writerow(vote)
                            counters["new_first_pass_calls"] += 1
                            if vote.get("parse_error"):
                                writers[parse_errors_csv][1].writerow(vote)
                                counters["new_parse_errors"] += 1
                        for vote in result["adjudication_votes"]:
                            writers[adj_votes_csv][1].writerow(vote)
                            counters["new_adjudication_calls"] += 1
                        for path in [votes_csv, adj_votes_csv, parse_errors_csv]:
                            writers[path][0].flush()
                        writers[resolutions_csv][1].writerow(result["resolution"])
                        writers[resolutions_csv][0].flush()
                        counters["completed_pairs"] += 1
                        if result["routed"]:
                            counters["routed_full_panel_resolutions"] += 1
                        else:
                            counters["prompt_c_only_resolutions"] += 1
                        if result["recovered"]:
                            counters["recovered_existing_vote_cells"] += 1
                        if counters["completed_pairs"] % 1000 == 0:
                            logger.info(
                                "Cascade progress: %d / %d pending cells",
                                counters["completed_pairs"],
                                len(pending),
                            )
    finally:
        logging.getLogger().removeHandler(file_handler)
        file_handler.close()

    finished_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    _write_manifest(
        output_dir,
        job_id=job_id,
        model=model,
        total_pairs=len(pairs),
        existing_resolutions=len(completed_keys),
        pending_pairs=len(pending),
        confidence_threshold=confidence_threshold,
        started_at=started_at,
        finished_at=finished_at,
        counters=counters,
    )
    logger.info("Cascade job %s completed %d cells", job_id, counters["completed_pairs"])
    return votes_csv
