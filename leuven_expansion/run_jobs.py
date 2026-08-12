"""
leuven_expansion/run_jobs.py

Parallel job runner for atomic word × feature judgments.
Submits independent atomic jobs via ThreadPoolExecutor.
Supports resume-by-row-hash and prompt-hash caching.
"""
from __future__ import annotations

import csv
import datetime
import hashlib
import json
import logging
import pathlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Set

import pandas as pd

from leuven_expansion.feature_judge import judge_pair
from leuven_expansion.feature_adjudicate import resolve_first_pass
from leuven_expansion.feature_schema import get_feature_text
from leuven_expansion.positive_verifier import (
    route_for_verification,
    verify_positive,
    POSITIVE_THRESHOLD,
    VERIFICATION_THRESHOLD,
)

logger = logging.getLogger(__name__)

# Columns for feature_votes.csv
VOTE_COLUMNS = [
    "job_id", "word_original", "word_normalized", "row_hash",
    "feature_id", "feature_text",
    "judge_id", "judge_prompt_variant", "judge_model",
    "feature_value", "confidence", "ambiguous", "reason",
    "raw_json", "parse_error", "prompt_hash",
]

RESOLUTION_COLUMNS = [
    "job_id", "word_normalized", "feature_id", "feature_text",
    "final_feature_value", "resolution_method", "needs_human_audit",
    "adjudicated", "adjudication_trigger", "confidence", "ambiguous",
    # positive verification columns (populated only when verification is enabled)
    "pre_verification_feature_value",
    "verified_feature_value",
    "positive_verification_status",
    "positive_verification_confidence",
    "positive_verification_reason",
]

VERIFIER_VOTE_COLUMNS = [
    "job_id", "word_normalized", "feature_id",
    "verified_feature_value", "retain_positive", "confidence", "reason",
]

PARSE_ERROR_COLUMNS = [
    "job_id", "word_original", "word_normalized", "row_hash",
    "feature_id", "feature_text",
    "judge_id", "judge_prompt_variant", "judge_model",
    "raw_json", "parse_error", "prompt_hash",
]


def _pair_hash(word_normalized: str, feature_id: int) -> str:
    return hashlib.sha256(f"{word_normalized}|{feature_id}".encode()).hexdigest()


def _load_completed_pairs(votes_csv: pathlib.Path) -> Set[str]:
    """Return set of row_hash values that already have 3 complete first-pass votes."""
    if not votes_csv.exists():
        return set()
    try:
        df = pd.read_csv(votes_csv)
        counts = df.groupby("row_hash")["judge_id"].count()
        return set(counts[counts >= 3].index.tolist())
    except Exception:
        return set()


def _add_file_log_handler(output_dir: pathlib.Path, job_id: str) -> logging.FileHandler:
    """Attach a per-job file handler to the root logger; return it for later removal."""
    log_path = output_dir / "run.log"
    fh = logging.FileHandler(log_path, mode="a")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(fh)
    return fh


def _write_manifest(
    output_dir: pathlib.Path,
    job_id: str,
    model: str,
    n_total: int,
    n_skipped: int,
    n_pending: int,
    started_at: str,
    finished_at: Optional[str] = None,
    n_completed: int = 0,
    n_parse_errors: int = 0,
) -> None:
    manifest = {
        "job_id": job_id,
        "model": model,
        "n_total_pairs": n_total,
        "n_skipped_resume": n_skipped,
        "n_pending": n_pending,
        "n_completed_this_run": n_completed,
        "n_parse_errors": n_parse_errors,
        "started_at": started_at,
        "finished_at": finished_at,
        "schema_version": "leuven_feature_schema_v1",
    }
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)


def run_atomic_jobs(
    *,
    job_id: str,
    pairs: List[Dict],   # [{"word_original", "word_normalized", "feature_id", "feature_text"}, ...]
    prompts: Dict[str, str],
    client,
    model: str,
    output_dir: pathlib.Path,
    temperature: float = 0.0,
    max_tokens: int = 400,
    max_workers: int = 16,
    resume: bool = True,
    # positive verification options
    enable_positive_verification: bool = False,
    positive_threshold: float = POSITIVE_THRESHOLD,
    verification_threshold: float = VERIFICATION_THRESHOLD,
    verifier_prompt: Optional[str] = None,
    verifier_model: Optional[str] = None,
) -> pathlib.Path:
    """
    Run independent atomic judging jobs for all word × feature pairs.

    Writes incrementally to output_dir:
        feature_votes.csv               — one row per judge call
        feature_adjudication_votes.csv  — one row per adjudicator call
        feature_resolutions.csv         — one row per pair (final value)
        parse_errors.csv                — subset of votes where parse_error=True
        manifest.json                   — job metadata (updated on finish)
        run.log                         — full logging output for this job

    Returns path to the votes CSV.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    votes_csv        = output_dir / "feature_votes.csv"
    adj_votes_csv    = output_dir / "feature_adjudication_votes.csv"
    resolutions_csv  = output_dir / "feature_resolutions.csv"
    parse_errors_csv = output_dir / "parse_errors.csv"
    verifier_votes_csv = output_dir / "positive_verification_votes.csv"

    # Attach per-job file logger
    fh = _add_file_log_handler(output_dir, job_id)

    started_at = datetime.datetime.utcnow().isoformat() + "Z"

    completed = _load_completed_pairs(votes_csv) if resume else set()
    logger.info(
        "Job %s: %d total pairs, %d already completed (resume=%s)",
        job_id, len(pairs), len(completed), resume,
    )

    pending = [
        p for p in pairs
        if _pair_hash(p["word_normalized"], p["feature_id"]) not in completed
    ]
    logger.info("Job %s: %d pairs to judge", job_id, len(pending))

    # Write initial manifest so the file exists even if the job crashes
    _write_manifest(
        output_dir=output_dir, job_id=job_id, model=model,
        n_total=len(pairs), n_skipped=len(completed), n_pending=len(pending),
        started_at=started_at,
    )

    # When not resuming, truncate all output files so they don't accumulate
    # rows from previous runs (which would inflate metrics on re-runs).
    if not resume:
        for path in (votes_csv, adj_votes_csv, resolutions_csv, parse_errors_csv,
                     verifier_votes_csv):
            if path.exists():
                path.unlink()

    votes_file_exists    = votes_csv.exists()
    parse_errors_exists  = parse_errors_csv.exists()
    verifier_votes_exists = verifier_votes_csv.exists()

    # Resolve the effective verifier model and prompt once
    _verifier_model  = verifier_model or model
    _verifier_prompt = verifier_prompt  # None means caller must supply if enable_positive_verification

    def _judge_one(pair: Dict) -> List[Dict]:
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
        )

    n_completed    = 0
    n_parse_errors = 0

    with (
        open(votes_csv, "a", newline="")          as vf,
        open(adj_votes_csv, "a", newline="")      as af,
        open(resolutions_csv, "a", newline="")    as rf,
        open(parse_errors_csv, "a", newline="")   as pef,
        open(verifier_votes_csv, "a", newline="") as vvf,
    ):
        vwriter  = csv.DictWriter(vf,  fieldnames=VOTE_COLUMNS,        extrasaction="ignore")
        awriter  = csv.DictWriter(af,  fieldnames=[
            "job_id", "word_normalized", "feature_id", "feature_text",
            "row_hash", "adjudicator_idx", "feature_value", "confidence",
            "ambiguous", "reason", "parse_error", "raw_json",
        ], extrasaction="ignore")
        rwriter  = csv.DictWriter(rf,  fieldnames=RESOLUTION_COLUMNS,  extrasaction="ignore")
        pewriter = csv.DictWriter(pef, fieldnames=PARSE_ERROR_COLUMNS, extrasaction="ignore")
        vvwriter = csv.DictWriter(vvf, fieldnames=VERIFIER_VOTE_COLUMNS, extrasaction="ignore")

        if not votes_file_exists:
            vwriter.writeheader()
            awriter.writeheader()
            rwriter.writeheader()
        if not parse_errors_exists:
            pewriter.writeheader()
        if not verifier_votes_exists:
            vvwriter.writeheader()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_judge_one, p): p for p in pending}
            for future in as_completed(futures):
                pair = futures[future]
                try:
                    votes = future.result()
                except Exception as e:
                    logger.error(
                        "Pair word=%r feature_id=%d failed: %s",
                        pair["word_normalized"], pair["feature_id"], e,
                    )
                    continue

                for v in votes:
                    vwriter.writerow({k: v.get(k, "") for k in VOTE_COLUMNS})
                    if v.get("parse_error"):
                        pewriter.writerow({k: v.get(k, "") for k in PARSE_ERROR_COLUMNS})
                        n_parse_errors += 1
                vf.flush()
                pef.flush()

                # Resolve
                resolution, adj_votes = resolve_first_pass(
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
                for av in adj_votes:
                    awriter.writerow(av)
                af.flush()

                # Build base resolution row
                res_row = {
                    "job_id": job_id,
                    "word_normalized": pair["word_normalized"],
                    "feature_id": pair["feature_id"],
                    "feature_text": pair["feature_text"],
                    **{k: resolution.get(k, "") for k in RESOLUTION_COLUMNS if k not in
                       ("job_id", "word_normalized", "feature_id", "feature_text",
                        "pre_verification_feature_value", "verified_feature_value",
                        "positive_verification_status", "positive_verification_confidence",
                        "positive_verification_reason")},
                    # verification columns default to empty
                    "pre_verification_feature_value": "",
                    "verified_feature_value": "",
                    "positive_verification_status": "not_candidate",
                    "positive_verification_confidence": "",
                    "positive_verification_reason": "",
                }

                # Positive verification pass
                if enable_positive_verification and _verifier_prompt:
                    fv = resolution.get("final_feature_value")
                    routing = route_for_verification(fv, positive_threshold=positive_threshold)
                    if routing == "candidate":
                        ver_update, ver_vote = verify_positive(
                            job_id=job_id,
                            word_normalized=pair["word_normalized"],
                            feature_id=pair["feature_id"],
                            feature_text=pair["feature_text"],
                            resolved_feature_value=float(fv),
                            resolved_confidence=float(resolution.get("confidence") or 0.0),
                            resolution_method=str(resolution.get("resolution_method", "")),
                            first_pass_votes=votes,
                            prompt=_verifier_prompt,
                            client=client,
                            model=_verifier_model,
                            verification_threshold=verification_threshold,
                            temperature=temperature,
                            max_tokens=max_tokens,
                        )
                        res_row["pre_verification_feature_value"] = fv
                        res_row["final_feature_value"] = ver_update.get("final_feature_value", fv)
                        res_row["verified_feature_value"] = ver_update.get("verified_feature_value", "")
                        res_row["positive_verification_status"] = ver_update.get("positive_verification_status", "")
                        res_row["positive_verification_confidence"] = ver_update.get("positive_verification_confidence", "")
                        res_row["positive_verification_reason"] = ver_update.get("positive_verification_reason", "")
                        if ver_update.get("needs_human_audit"):
                            res_row["needs_human_audit"] = True
                        vvwriter.writerow({k: ver_vote.get(k, "") for k in VERIFIER_VOTE_COLUMNS})
                        vvf.flush()

                rwriter.writerow(res_row)
                rf.flush()

                n_completed += 1
                if n_completed % 100 == 0:
                    logger.info("Progress: %d / %d pairs judged", n_completed, len(pending))

    finished_at = datetime.datetime.utcnow().isoformat() + "Z"
    _write_manifest(
        output_dir=output_dir, job_id=job_id, model=model,
        n_total=len(pairs), n_skipped=len(completed), n_pending=len(pending),
        started_at=started_at, finished_at=finished_at,
        n_completed=n_completed, n_parse_errors=n_parse_errors,
    )
    logger.info(
        "Job %s complete. %d pairs judged, %d parse errors. Votes: %s",
        job_id, n_completed, n_parse_errors, votes_csv,
    )

    logging.getLogger().removeHandler(fh)
    fh.close()

    return votes_csv
