#!/usr/bin/env python3
"""Dry-run, shard, resume, and finalize V4 judgments using the V2 atomic runner."""
from __future__ import annotations

import argparse
import csv
import importlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from leuven_expansion.feature_prompts import load_prompts_by_version
from leuven_expansion.feature_schema import load_candidate_feature_schema
from leuven_expansion.normalize import normalize_word
from leuven_expansion.cascade_jobs import (
    CASCADE_RESOLUTION_METHOD,
    run_prompt_c_cascade_jobs,
)
from leuven_expansion.run_jobs import run_atomic_jobs
from leuven_expansion.v4 import sha256_file, stable_json_hash, stable_shard, write_json


ROOT = Path(__file__).resolve().parent
PROMPT_DIR = ROOT / "leuven_expansion" / "prompts"
V2_RESOLVED = (
    ROOT
    / "artifacts"
    / "leuven_full_labels"
    / "leuven_full_v2"
    / "feature_resolutions.csv"
)
# All 32 production shards from the interrupted exhaustive run use this hash. It is
# accepted only alongside the frozen candidate inventory and matching shard index.
KNOWN_EXECUTED_V4_LEGACY_PROTOCOL_HASH = (
    "723516354f8df921164d122fbaee0226dd02dc36728d4cc2835034a5e165b86c"
)
KNOWN_EXECUTED_V4_CANDIDATE_INVENTORY_HASH = (
    "cf0af7b1c8126e2a06ad162e916685b75a298c3cf8da9a89b8ecd9e9652da3fd"
)


def load_words(path: Path) -> list[dict[str, str]]:
    frame = pd.read_csv(path)
    item_column = frame.columns[0]
    words = [
        {"word_original": str(value), "word_normalized": normalize_word(str(value))}
        for value in frame[item_column]
    ]
    if len({row["word_normalized"] for row in words}) != len(words):
        raise ValueError("Normalized Leuven words must be unique")
    return words


def protocol_record(
    candidate_bank: Path,
    leuven_words: Path,
    model: str,
    shard_count: int,
    v2_manifest: Path | None,
    cascade_confidence_threshold: float | None = 0.80,
    protocol_project_root: Path | None = None,
) -> dict[str, Any]:
    schema = load_candidate_feature_schema(candidate_bank)
    prompt_paths = {
        "A": PROMPT_DIR / "feature_judge_prompt_A_v2_production.txt",
        "B": PROMPT_DIR / "feature_judge_prompt_B_v2_production.txt",
        "C": PROMPT_DIR / "feature_judge_prompt_C_v2_production.txt",
        "adjudicator": PROMPT_DIR / "feature_adjudicator_prompt_v2_production.txt",
    }
    v2 = None
    if v2_manifest is not None:
        v2 = json.loads(v2_manifest.read_text(encoding="utf-8"))
        if not v2.get("finished_at"):
            raise ValueError("V2 reference manifest is incomplete")
        if v2.get("model") != model:
            raise ValueError(
                f"V4 model {model!r} does not match executed V2 model {v2.get('model')!r}"
            )
    legacy = {
        "protocol_version": "v4-atomic-v2-exhaustive-0.1.0",
        "candidate_bank": str(candidate_bank.resolve()),
        "candidate_bank_sha256": sha256_file(candidate_bank),
        "candidate_inventory_hash": schema["candidate_inventory_hash"],
        "leuven_words": str(leuven_words.resolve()),
        "leuven_words_sha256": sha256_file(leuven_words),
        "word_count": len(load_words(leuven_words)),
        "candidate_count": schema["n_features"],
        "expected_cell_count": schema["n_features"] * len(load_words(leuven_words)),
        "judge_model": model,
        "judge_model_revision": model,
        "temperature": 0.0,
        "max_tokens": 400,
        "first_pass_judges": 3,
        "prompt_sha256": {key: sha256_file(path) for key, path in prompt_paths.items()},
        "schema_sha256": sha256_file(
            ROOT
            / "leuven_expansion"
            / "schemas"
            / "atomic_feature_judgment_schema_v1.json"
        ),
        "shard_count": shard_count,
        "shard_function": "sha256(candidate_id NUL normalized_word) first 8 bytes modulo shard_count",
        "v2_manifest": str(v2_manifest.resolve()) if v2_manifest else None,
        "v2_manifest_sha256": sha256_file(v2_manifest) if v2_manifest else None,
        "v2_reference": v2,
    }
    if protocol_project_root is not None:
        # Replay recorded path metadata, but hash the real local input contents.
        for field, path in (
            ("candidate_bank", candidate_bank),
            ("leuven_words", leuven_words),
            ("v2_manifest", v2_manifest),
        ):
            if path is not None:
                legacy[field] = str(
                    protocol_project_root.resolve() / path.resolve().relative_to(ROOT)
                )
    if cascade_confidence_threshold is None:
        return legacy
    legacy_hash = stable_json_hash(legacy)
    compatible_legacy_hashes = {legacy_hash}
    if schema["candidate_inventory_hash"] == KNOWN_EXECUTED_V4_CANDIDATE_INVENTORY_HASH:
        compatible_legacy_hashes.add(KNOWN_EXECUTED_V4_LEGACY_PROTOCOL_HASH)
    return legacy | {
        "protocol_version": "v4-atomic-v2-prompt-c-cascade-0.2.0",
        "execution_mode": "prompt_c_cascade",
        "cascade_screen_prompt": "C",
        "cascade_confidence_threshold": cascade_confidence_threshold,
        "cascade_routing_rule": (
            "C value > 0 OR ambiguous OR confidence < threshold OR parse/schema failure"
        ),
        "cascade_routed_action": "request missing A/B votes, then use frozen V2 resolver",
        "cascade_unrouted_value": 0,
        "legacy_full_panel_protocol_hash": legacy_hash,
        "compatible_legacy_protocol_hashes": sorted(compatible_legacy_hashes),
        "legacy_completed_cells_reused": True,
    }


def build_pairs(
    candidate_bank: Path,
    leuven_words: Path,
    shard_count: int,
    shard_index: int | None = None,
) -> list[dict[str, object]]:
    schema = load_candidate_feature_schema(candidate_bank)
    words = load_words(leuven_words)
    pairs: list[dict[str, object]] = []
    for feature_id, (candidate_id, feature_text) in enumerate(
        zip(schema["candidate_ids"], schema["feature_columns"])
    ):
        for word in words:
            shard = stable_shard(candidate_id, word["word_normalized"], shard_count)
            if shard_index is None or shard == shard_index:
                pairs.append(
                    {
                        **word,
                        "feature_id": feature_id,
                        "feature_text": feature_text,
                        "candidate_id": candidate_id,
                        "shard_index": shard,
                    }
                )
    return pairs


def read_csv_or_empty(
    path: Path, required_columns: set[str] | None = None
) -> pd.DataFrame:
    required_columns = required_columns or set()
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=sorted(required_columns))
    with path.open("rb") as handle:
        prefix = handle.read(128)
    if prefix.startswith(b"version https://git-lfs.github.com/spec/v1"):
        try:
            display_path = path.resolve().relative_to(ROOT)
        except ValueError:
            display_path = path
        raise ValueError(
            f"Checkpoint CSV is a Git LFS pointer, not materialized data: {path}. "
            f"Run git lfs pull --include='{display_path}'."
        )
    try:
        frame = pd.read_csv(path, low_memory=False)
    except pd.errors.EmptyDataError:
        frame = pd.DataFrame()
    missing = required_columns - set(frame.columns)
    if missing:
        raise ValueError(
            f"Checkpoint CSV has the wrong schema: {path}; missing columns "
            f"{sorted(missing)}; available columns {list(frame.columns)}"
        )
    return frame


def validate_shard_resume(
    shard_dir: Path, protocol: dict[str, Any], shard_index: int
) -> None:
    sidecar = shard_dir / "v4_shard_manifest.json"
    expected = {
        "protocol_hash": stable_json_hash(protocol),
        "shard_index": shard_index,
    }
    if sidecar.exists():
        actual = json.loads(sidecar.read_text(encoding="utf-8"))
        compatible_hashes = {
            expected["protocol_hash"],
            protocol.get("legacy_full_panel_protocol_hash"),
            *protocol.get("compatible_legacy_protocol_hashes", []),
        }
        compatible_hashes.discard(None)
        if (
            actual.get("shard_index") != shard_index
            or actual.get("protocol_hash") not in compatible_hashes
            or actual.get(
                "candidate_inventory_hash", protocol["candidate_inventory_hash"]
            )
            != protocol["candidate_inventory_hash"]
        ):
            raise ValueError("Cannot resume V4 shard under a changed protocol")
    votes = read_csv_or_empty(
        shard_dir / "feature_votes.csv",
        {"word_normalized", "feature_id", "judge_id"},
    )
    resolutions = read_csv_or_empty(
        shard_dir / "feature_resolutions.csv",
        {"word_normalized", "feature_id", "final_feature_value"},
    )
    if (
        not votes.empty
        and votes.duplicated(["word_normalized", "feature_id", "judge_id"]).any()
    ):
        raise ValueError("Cannot resume V4 shard with duplicate first-pass votes")
    if (
        not resolutions.empty
        and resolutions.duplicated(["word_normalized", "feature_id"]).any()
    ):
        raise ValueError("Cannot resume V4 shard with duplicate resolutions")
    prior_hash = actual.get("protocol_hash") if sidecar.exists() else None
    migrated = prior_hash is not None and prior_hash != expected["protocol_hash"]
    preserved = actual if sidecar.exists() else {}
    write_json(
        sidecar,
        preserved
        | expected
        | {
            "candidate_inventory_hash": protocol["candidate_inventory_hash"],
            "migrated_from_protocol_hash": (
                prior_hash if migrated else preserved.get("migrated_from_protocol_hash")
            ),
        },
    )


def v2_adjudication_rate() -> float:
    if not V2_RESOLVED.exists():
        return 0.10
    resolved = pd.read_csv(V2_RESOLVED, usecols=["adjudicated"])
    return float(resolved["adjudicated"].fillna(False).astype(bool).mean())


def dry_run_report(protocol: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    existing_resolved = 0
    for path in (output_dir / "shards").glob("*/feature_resolutions.csv"):
        resolved = pd.read_csv(path, usecols=["feature_id", "final_feature_value"])
        values = pd.to_numeric(resolved["final_feature_value"], errors="coerce")
        existing_resolved += int((values.notna() & values.between(0, 4)).sum())
    cells = int(protocol["expected_cell_count"])
    adjudication_rate = v2_adjudication_rate()
    remaining = max(0, cells - existing_resolved)
    v4_pilot_route_rate = 0.01950302802514564
    v2_route_rate = 0.25904693474300083
    v4_pilot_calls_per_cell = 2647269 / 2474385
    v2_calls_per_cell = 1125225 / 584535
    report = {
        **{
            key: protocol[key]
            for key in [
                "candidate_count",
                "word_count",
                "expected_cell_count",
                "shard_count",
            ]
        },
        "execution_mode": protocol.get("execution_mode", "full_panel"),
        "legacy_full_panel_calls": cells * 3,
        "planned_prompt_c_screen_calls_remaining": remaining,
        "legacy_v2_full_panel_adjudication_rate": adjudication_rate,
        "observed_cascade_route_rate_v4_pilot": v4_pilot_route_rate,
        "observed_cascade_route_rate_v2": v2_route_rate,
        "estimated_remaining_calls_v4_pilot_rate": round(
            remaining * v4_pilot_calls_per_cell
        ),
        "estimated_remaining_calls_conservative_v2_rate": round(
            remaining * v2_calls_per_cell
        ),
        "estimated_remaining_call_reduction_v4_pilot_rate": (
            1 - v4_pilot_calls_per_cell / 3
        ),
        "estimated_remaining_call_reduction_conservative_v2_rate": (
            1 - v2_calls_per_cell / 3
        ),
        "existing_reusable_cells": existing_resolved,
        "remaining_cells": remaining,
        "primary_execution": (
            "exhaustive candidate-by-word prompt-C screening; routed cells receive A/B "
            "and the frozen V2 resolver; no retrieval pruning"
        ),
        "cascade_confidence_threshold": protocol.get("cascade_confidence_threshold"),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "dry_run_cost_report.json", report)
    return report


def _as_bool_series(values: pd.Series) -> pd.Series:
    if isinstance(values.dtype, pd.BooleanDtype) or values.dtype == bool:
        return values.fillna(False).astype(bool)
    return values.fillna("").astype(str).str.lower().isin({"true", "1", "yes"})


def _vote_coverage(
    resolutions: pd.DataFrame,
    votes: pd.DataFrame,
    confidence_threshold: float,
) -> dict[str, int]:
    """Validate either legacy A/B/C votes or an unrouted prompt-C-only cell."""
    key_columns = ["word_normalized", "feature_id"]
    resolutions = resolutions.copy()
    votes = votes.copy()
    for frame in [resolutions, votes]:
        if not frame.empty:
            frame["word_normalized"] = frame["word_normalized"].astype(str)
            frame["feature_id"] = pd.to_numeric(
                frame["feature_id"], errors="raise"
            ).astype(int)
    duplicate_vote_rows = int(votes.duplicated([*key_columns, "judge_id"]).sum())
    resolution_keys = pd.MultiIndex.from_frame(resolutions[key_columns])
    if votes.empty:
        return {
            "full_panel_vote_cells": 0,
            "prompt_c_only_cells": 0,
            "missing_first_pass_vote_cells": len(resolutions),
            "invalid_first_pass_vote_counts": len(resolutions),
            "duplicate_first_pass_votes": 0,
            "extra_first_pass_vote_cells": 0,
            "cascade_route_violations": 0,
        }
    summary = votes.groupby(key_columns, observed=True)["judge_id"].agg(
        vote_rows="size",
        judge_signature=lambda values: "".join(sorted(set(map(str, values)))),
    )
    aligned = summary.reindex(resolution_keys)
    full_panel = aligned["vote_rows"].eq(3) & aligned["judge_signature"].eq("ABC")

    c_votes = votes.loc[votes["judge_id"].astype(str).eq("C")].copy()
    c_votes = c_votes.drop_duplicates(key_columns).set_index(key_columns)
    required_route_columns = {"feature_value", "confidence", "ambiguous", "parse_error"}
    if required_route_columns.issubset(c_votes.columns):
        value = pd.to_numeric(c_votes["feature_value"], errors="coerce")
        confidence = pd.to_numeric(c_votes["confidence"], errors="coerce")
        parse_error = c_votes["parse_error"].fillna("").astype(str).str.strip().ne("")
        c_routes = (
            value.gt(0)
            | _as_bool_series(c_votes["ambiguous"])
            | confidence.lt(confidence_threshold)
            | value.isna()
            | confidence.isna()
            | parse_error
        ).reindex(resolution_keys, fill_value=True)
    else:
        c_routes = pd.Series(True, index=resolution_keys)
    methods = (
        resolutions.get("resolution_method", pd.Series("", index=resolutions.index))
        .fillna("")
        .astype(str)
        .to_numpy()
    )
    final_values = pd.to_numeric(
        resolutions.get(
            "final_feature_value", pd.Series(np.nan, index=resolutions.index)
        ),
        errors="coerce",
    ).to_numpy()
    c_only_signature = aligned["vote_rows"].eq(1) & aligned["judge_signature"].eq("C")
    c_only_method = methods == CASCADE_RESOLUTION_METHOD
    c_only_value = final_values == 0
    c_only = (
        c_only_signature.to_numpy()
        & c_only_method
        & c_only_value
        & ~c_routes.to_numpy()
    )
    valid = full_panel.to_numpy() | c_only
    vote_keys = set(votes[key_columns].itertuples(index=False, name=None))
    resolved_key_set = set(resolutions[key_columns].itertuples(index=False, name=None))
    return {
        "full_panel_vote_cells": int(full_panel.sum()),
        "prompt_c_only_cells": int(c_only.sum()),
        "missing_first_pass_vote_cells": int(aligned["vote_rows"].isna().sum()),
        "invalid_first_pass_vote_counts": int((~valid).sum()),
        "duplicate_first_pass_votes": duplicate_vote_rows,
        "extra_first_pass_vote_cells": len(vote_keys - resolved_key_set),
        "cascade_route_violations": int(
            (c_only_signature.to_numpy() & c_routes.to_numpy()).sum()
        ),
    }


def validate_shard_complete(
    shard_dir: Path,
    pairs: list[dict[str, object]],
    protocol: dict[str, Any],
    shard_index: int,
) -> dict[str, Any]:
    resolutions = read_csv_or_empty(shard_dir / "feature_resolutions.csv")
    votes = read_csv_or_empty(shard_dir / "feature_votes.csv")
    expected = {
        (str(pair["word_normalized"]), int(pair["feature_id"])) for pair in pairs
    }
    actual = (
        set(
            zip(
                resolutions["word_normalized"].astype(str),
                resolutions["feature_id"].astype(int),
            )
        )
        if not resolutions.empty
        else set()
    )
    duplicates = (
        int(resolutions.duplicated(["word_normalized", "feature_id"]).sum())
        if not resolutions.empty
        else 0
    )
    resolved_values = pd.to_numeric(
        resolutions.get(
            "final_feature_value", pd.Series(np.nan, index=resolutions.index)
        ),
        errors="coerce",
    )
    invalid_resolved_values = int(
        (resolved_values.isna() | ~resolved_values.between(0, 4)).sum()
    )
    coverage = _vote_coverage(
        resolutions,
        votes,
        float(protocol.get("cascade_confidence_threshold", 0.80)),
    )
    complete = (
        actual == expected
        and duplicates == 0
        and invalid_resolved_values == 0
        and coverage["invalid_first_pass_vote_counts"] == 0
        and coverage["missing_first_pass_vote_cells"] == 0
        and coverage["duplicate_first_pass_votes"] == 0
        and coverage["extra_first_pass_vote_cells"] == 0
    )
    record = {
        "protocol_hash": stable_json_hash(protocol),
        "candidate_inventory_hash": protocol["candidate_inventory_hash"],
        "shard_index": shard_index,
        "expected_cells": len(expected),
        "resolved_cells": len(actual),
        "missing_cells": len(expected - actual),
        "extra_cells": len(actual - expected),
        "duplicate_cells": duplicates,
        "invalid_resolved_values": invalid_resolved_values,
        **coverage,
        "complete": complete,
    }
    write_json(shard_dir / "v4_shard_manifest.json", record)
    if not complete:
        raise ValueError(f"V4 shard {shard_index} is incomplete: {record}")
    return record


def _concatenate_csv_files(paths: list[Path], target: Path) -> int:
    """Concatenate same-schema CSVs without retaining them in memory."""
    expected_header: str | None = None
    row_count = 0
    with target.open("w", encoding="utf-8", newline="") as output:
        for path in paths:
            with path.open("r", encoding="utf-8", newline="") as source:
                header = source.readline()
                if not header:
                    continue
                if expected_header is None:
                    expected_header = header
                    output.write(header)
                elif header != expected_header:
                    raise ValueError(
                        f"Shard CSV headers differ for {target.name}: {path}"
                    )
                for line in source:
                    output.write(line)
                    row_count += 1
        if expected_header is None:
            output.write("")
    return row_count


def _concatenate_logs(paths: list[Path], target: Path) -> None:
    with target.open("w", encoding="utf-8") as output:
        for path in paths:
            output.write(f"\n===== {path.parent.name} =====\n")
            if path.exists():
                with path.open("r", encoding="utf-8", errors="replace") as source:
                    for chunk in iter(lambda: source.read(1024 * 1024), ""):
                        output.write(chunk)


def finalization_preflight(
    output_dir: Path,
    protocol: dict[str, Any],
    expected_by_shard: list[int] | None = None,
    max_unresolved_cells: int = 0,
) -> dict[str, Any]:
    """Inspect every sidecar without changing judgments or declaring CSV integrity."""
    protocol_hash = stable_json_hash(protocol)
    shard_count = int(protocol["shard_count"])
    counters = (
        "missing_cells",
        "extra_cells",
        "duplicate_cells",
        "invalid_resolved_values",
        "invalid_first_pass_vote_counts",
        "missing_first_pass_vote_cells",
        "duplicate_first_pass_votes",
        "extra_first_pass_vote_cells",
        "cascade_route_violations",
    )
    shards = []
    for index in range(shard_count):
        path = output_dir / "shards" / f"{index:04d}" / "v4_shard_manifest.json"
        issues = []
        manifest: dict[str, Any] = {}
        if not path.exists():
            issues.append("missing shard manifest")
        else:
            try:
                manifest = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(manifest, dict):
                    raise ValueError("expected a JSON object")
            except (ValueError, OSError) as error:
                issues.append(f"unreadable shard manifest: {error}")
                manifest = {}
        if manifest:
            if manifest.get("shard_index") != index:
                issues.append(f"shard_index mismatch: {manifest.get('shard_index')!r}")
            if manifest.get("protocol_hash") != protocol_hash:
                issues.append(
                    f"protocol hash mismatch: recorded={manifest.get('protocol_hash')}, "
                    f"requested={protocol_hash}"
                )
            if (
                manifest.get("candidate_inventory_hash")
                != protocol["candidate_inventory_hash"]
            ):
                issues.append("candidate inventory hash mismatch")
            if manifest.get("complete") is not True and not max_unresolved_cells:
                issues.append("complete=false (or absent)")
            for name in ("expected_cells", "resolved_cells"):
                value = manifest.get(name)
                if type(value) is not int or value < 0:
                    issues.append(f"invalid {name}: {value!r}")
            if manifest.get("resolved_cells") != manifest.get("expected_cells"):
                issues.append("resolved cell count differs from expected cell count")
            if (
                expected_by_shard is not None
                and manifest.get("expected_cells") != expected_by_shard[index]
            ):
                issues.append(
                    f"expected cell count differs from frozen hash partition ({expected_by_shard[index]})"
                )
            for name in counters:
                value = manifest.get(name, 0)
                if type(value) is not int or value < 0:
                    issues.append(f"invalid {name}: {value!r}")
                elif value and not (
                    name == "invalid_resolved_values" and max_unresolved_cells
                ):
                    issues.append(f"{name}={value}")
        elif not issues:
            issues.append("empty shard manifest")
        shards.append(
            {
                "shard_index": index,
                "manifest_path": str(path),
                "recorded_protocol_hash": manifest.get("protocol_hash"),
                "complete": manifest.get("complete", False),
                "expected_cells": manifest.get("expected_cells"),
                "resolved_cells": manifest.get("resolved_cells"),
                **{name: manifest.get(name, 0) for name in counters},
                "issues": issues,
            }
        )

    def total(name: str) -> int:
        return sum(
            row[name] for row in shards if type(row[name]) is int and row[name] >= 0
        )

    total_expected = total("expected_cells")
    global_issues = []
    if total_expected != int(protocol["expected_cell_count"]):
        global_issues.append(
            "shard expected counts do not sum to the frozen cross-product"
        )
    blocked = sum(bool(row["issues"]) for row in shards)
    return {
        "requested_protocol_hash": protocol_hash,
        "requested_protocol": protocol,
        "shard_count": shard_count,
        "blocked_shards": blocked,
        "expected_cells": int(protocol["expected_cell_count"]),
        "manifest_expected_cells": total_expected,
        "manifest_resolved_cells": total("resolved_cells"),
        **{name: total(name) for name in counters},
        "global_issues": global_issues,
        "ready_from_manifests": blocked == 0 and not global_issues,
        "max_unresolved_cells": max_unresolved_cells,
        "unresolved_note": (
            "Incomplete/invalid-score sidecar flags may be stale after offline recovery. Actual CSV failures must be adjudicator_failed missing scores, within the authorized cap, and are revalidated during finalization."
            if max_unresolved_cells
            else None
        ),
        "scope": "Read-only manifest checks, not revalidation of raw CSV contents. Finalization must still validate every cell and vote.",
        "shards": shards,
    }


def _require_finalization_ready(output_dir: Path, report: dict[str, Any]) -> None:
    report_path = output_dir / "finalization_preflight.json"
    write_json(report_path, report)
    if report["ready_from_manifests"]:
        return
    failures = [row for row in report["shards"] if row["issues"]]
    first = (
        f"Shard {failures[0]['shard_index']}: " + "; ".join(failures[0]["issues"])
        if failures
        else "; ".join(report["global_issues"])
    )
    raise ValueError(
        f"V4 finalization blocked: {report['blocked_shards']}/{report['shard_count']} "
        f"shards fail manifest checks; invalid_resolved_values={report['invalid_resolved_values']}; "
        f"missing_cells={report['missing_cells']}. {first}. "
        f"Full diagnostics: {report_path}. Repair/retry invalid cells and revalidate shards; "
        "do not set complete=true or rewrite protocol hashes to bypass validation."
    )


def finalize(
    candidate_bank: Path,
    leuven_words: Path,
    output_dir: Path,
    protocol: dict[str, Any],
    max_unresolved_cells: int = 0,
) -> None:
    if max_unresolved_cells < 0:
        raise ValueError("max-unresolved-cells must be nonnegative")
    _require_finalization_ready(
        output_dir,
        finalization_preflight(
            output_dir, protocol, max_unresolved_cells=max_unresolved_cells
        ),
    )
    schema = load_candidate_feature_schema(candidate_bank)
    words = load_words(leuven_words)
    word_set = {row["word_normalized"] for row in words}
    shard_count = int(protocol["shard_count"])
    shard_dirs = [
        output_dir / "shards" / f"{index:04d}" for index in range(shard_count)
    ]
    expected_by_shard = [0] * shard_count
    for candidate_id in schema["candidate_ids"]:
        for word in words:
            expected_by_shard[
                stable_shard(candidate_id, word["word_normalized"], shard_count)
            ] += 1

    protocol_hash = stable_json_hash(protocol)
    _require_finalization_ready(
        output_dir,
        finalization_preflight(
            output_dir, protocol, expected_by_shard, max_unresolved_cells
        ),
    )
    completed_shard_manifests = [
        json.loads((shard / "v4_shard_manifest.json").read_text(encoding="utf-8"))
        for shard in shard_dirs
    ]

    raw_names = [
        "feature_votes.csv",
        "feature_adjudication_votes.csv",
        "parse_errors.csv",
    ]
    raw_counts: dict[str, int] = {}
    for name in raw_names:
        raw_counts[name] = _concatenate_csv_files(
            [shard / name for shard in shard_dirs], output_dir / name
        )
    _concatenate_logs(
        [shard / "run.log" for shard in shard_dirs], output_dir / "run.log"
    )

    required_columns = [
        "candidate_id",
        "target_word",
        "resolved_value",
        "resolved_binary_locked_v2",
        "confidence",
        "ambiguous",
        "resolution_method",
        "needs_human_audit",
        "adjudicated",
        "adjudication_trigger",
        "vote_prompt_hashes",
        "judge_model_revision",
        "feature_id",
        "feature_text",
    ]
    resolved_path = output_dir / "resolved_feature_values.csv"
    temporary_resolved = output_dir / "resolved_feature_values.csv.tmp"
    resolved_cells = 0
    unresolved_frames = []
    with temporary_resolved.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=required_columns)
        writer.writeheader()
        for shard_index, shard_dir in enumerate(shard_dirs):
            resolutions = pd.read_csv(shard_dir / "feature_resolutions.csv")
            vote_columns = {
                "word_normalized",
                "feature_id",
                "judge_id",
                "prompt_hash",
                "feature_value",
                "confidence",
                "ambiguous",
                "parse_error",
            }
            votes = pd.read_csv(
                shard_dir / "feature_votes.csv",
                usecols=lambda column: column in vote_columns,
                low_memory=False,
            )
            if resolutions.duplicated(["word_normalized", "feature_id"]).any():
                raise ValueError(f"Duplicate resolutions in shard {shard_index}")
            resolutions["feature_id"] = pd.to_numeric(
                resolutions["feature_id"], errors="raise"
            ).astype(int)
            invalid_id = ~resolutions["feature_id"].between(0, schema["n_features"] - 1)
            invalid_word = ~resolutions["word_normalized"].astype(str).isin(word_set)
            values = pd.to_numeric(resolutions["final_feature_value"], errors="coerce")
            invalid_value = values.isna() | ~values.between(0, 4)
            candidate_ids = resolutions["feature_id"].map(
                schema["candidate_id_by_feature_id"]
            )
            wrong_shard = np.fromiter(
                (
                    stable_shard(candidate, str(word), shard_count) != shard_index
                    for candidate, word in zip(
                        candidate_ids, resolutions["word_normalized"]
                    )
                ),
                dtype=bool,
                count=len(resolutions),
            )
            if invalid_id.any() or invalid_word.any() or wrong_shard.any():
                raise ValueError(f"Invalid cell content in shard {shard_index}")
            if invalid_value.any():
                allowed = resolutions["final_feature_value"].isna() & resolutions[
                    "resolution_method"
                ].eq("adjudicator_failed")
                if not max_unresolved_cells or (invalid_value & ~allowed).any():
                    raise ValueError(
                        f"Invalid cell content in shard {shard_index}; only recorded adjudicator_failed missing scores may be excluded"
                    )
                failed = resolutions.loc[
                    invalid_value,
                    [
                        "word_normalized",
                        "feature_id",
                        "feature_text",
                        "resolution_method",
                    ],
                ].copy()
                failed["candidate_id"] = candidate_ids.loc[invalid_value]
                failed = failed.rename(columns={"word_normalized": "target_word"})
                failed["shard_index"] = shard_index
                failed["reason"] = "adjudicator_failed_missing_final_value"
                unresolved_frames.append(failed)
                if (
                    sum(len(frame) for frame in unresolved_frames)
                    > max_unresolved_cells
                ):
                    raise ValueError(
                        f"Actual unresolved cells exceed authorized cap {max_unresolved_cells}; apply the saved-response recovery before finalization"
                    )
            if len(resolutions) != expected_by_shard[shard_index]:
                raise ValueError(f"Unexpected resolution count in shard {shard_index}")
            coverage = _vote_coverage(
                resolutions,
                votes,
                float(protocol.get("cascade_confidence_threshold", 0.80)),
            )
            if (
                coverage["invalid_first_pass_vote_counts"]
                or coverage["missing_first_pass_vote_cells"]
                or coverage["duplicate_first_pass_votes"]
                or coverage["extra_first_pass_vote_cells"]
            ):
                raise ValueError(
                    f"Shard {shard_index} has invalid cascade vote coverage: {coverage}"
                )
            prompt_hashes = (
                votes.groupby(["word_normalized", "feature_id"], observed=True)[
                    "prompt_hash"
                ]
                .agg(lambda values: json.dumps(sorted(set(map(str, values)))))
                .rename("vote_prompt_hashes")
                .reset_index()
            )
            resolved = resolutions.merge(
                prompt_hashes,
                on=["word_normalized", "feature_id"],
                how="left",
                validate="one_to_one",
            )
            resolved["candidate_id"] = candidate_ids.to_numpy()
            resolved["target_word"] = resolved.pop("word_normalized")
            resolved["resolved_value"] = values.to_numpy()
            resolved["resolved_binary_locked_v2"] = (
                resolved["resolved_value"].gt(0).astype(int)
            )
            resolved["judge_model_revision"] = protocol["judge_model_revision"]
            for column in required_columns:
                if column not in resolved:
                    resolved[column] = ""
            resolved = resolved.loc[~invalid_value.to_numpy()]
            writer.writerows(resolved[required_columns].to_dict(orient="records"))
            resolved_cells += len(resolved)

    unresolved = (
        pd.concat(unresolved_frames, ignore_index=True)
        if unresolved_frames
        else pd.DataFrame(columns=["candidate_id", "target_word", "reason"])
    )
    if resolved_cells + len(unresolved) != int(protocol["expected_cell_count"]):
        raise ValueError(
            "Final resolved cell count differs from the frozen cross-product"
        )
    unresolved.to_csv(output_dir / "unresolved_cells.csv", index=False)
    temporary_resolved.replace(resolved_path)
    excluded_set = set(unresolved["candidate_id"])
    excluded_candidates = [c for c in schema["candidate_ids"] if c in excluded_set]
    manifest = protocol | {
        "protocol_hash": protocol_hash,
        "resolved_cells": resolved_cells,
        "unresolved_cells": len(unresolved),
        "judgments_complete": len(unresolved) == 0,
        "status": (
            "finalized_with_unresolved_exclusions" if len(unresolved) else "complete"
        ),
        "unresolved_policy": (
            "exclude_incomplete_candidates" if len(unresolved) else "none"
        ),
        "max_unresolved_cells": max_unresolved_cells,
        "excluded_candidate_ids": excluded_candidates,
        "excluded_candidate_count": len(excluded_candidates),
        "unresolved_cells_sha256": sha256_file(output_dir / "unresolved_cells.csv"),
        "duplicate_cells": 0,
        "feature_votes": raw_counts["feature_votes.csv"],
        "adjudication_votes": raw_counts["feature_adjudication_votes.csv"],
        "parse_errors": raw_counts["parse_errors.csv"],
        "full_panel_vote_cells": sum(
            int(record.get("full_panel_vote_cells", 0))
            for record in completed_shard_manifests
        ),
        "prompt_c_only_cells": sum(
            int(record.get("prompt_c_only_cells", 0))
            for record in completed_shard_manifests
        ),
        "resolved_values_sha256": sha256_file(
            output_dir / "resolved_feature_values.csv"
        ),
        "run_log_sha256": sha256_file(output_dir / "run.log"),
        "complete": True,
    }
    write_json(output_dir / "judgment_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-bank", type=Path, required=True)
    parser.add_argument("--leuven-words", type=Path, required=True)
    parser.add_argument("--v2-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="Qwen2.5-72B-Instruct")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--max-workers", type=int, default=64)
    parser.add_argument("--shard-count", type=int, default=256)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preflight-shard", action="store_true")
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument(
        "--max-unresolved-cells",
        type=int,
        default=0,
        help="Explicit finalization-only cap for recorded failed adjudications; exclude affected candidate columns downstream, never fill missing cells with zero",
    )
    parser.add_argument(
        "--check-finalization",
        action="store_true",
        help="Read-only all-shard manifest diagnostics; exit 2 when blocked, without loading a model",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--protocol-project-root",
        type=Path,
        help="Replay an original project's absolute path metadata for local finalization/checks; input-content hashes are still verified",
    )
    parser.add_argument(
        "--execution-mode",
        choices=["prompt-c-cascade", "full-panel"],
        default="prompt-c-cascade",
    )
    parser.add_argument("--cascade-confidence-threshold", type=float, default=0.80)
    args = parser.parse_args()
    if args.max_unresolved_cells < 0 or (
        args.max_unresolved_cells and not (args.finalize or args.check_finalization)
    ):
        raise ValueError(
            "--max-unresolved-cells must be nonnegative and is only allowed for finalization/checks"
        )
    if args.protocol_project_root and not (args.finalize or args.check_finalization):
        raise ValueError("--protocol-project-root is restricted to finalization/checks")
    if args.shard_index is not None and not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard-index must be in [0, shard-count)")
    protocol = protocol_record(
        args.candidate_bank.resolve(),
        args.leuven_words.resolve(),
        args.model,
        args.shard_count,
        args.v2_manifest.resolve() if args.v2_manifest else None,
        (
            args.cascade_confidence_threshold
            if args.execution_mode == "prompt-c-cascade"
            else None
        ),
        protocol_project_root=args.protocol_project_root,
    )
    output = args.output_dir.resolve()
    if args.check_finalization:
        report = finalization_preflight(
            output, protocol, max_unresolved_cells=args.max_unresolved_cells
        )
        print(json.dumps(report, indent=2))
        raise SystemExit(0 if report["ready_from_manifests"] else 2)
    if args.dry_run:
        print(json.dumps(dry_run_report(protocol, output), indent=2))
        return
    if args.finalize:
        finalize(
            args.candidate_bank,
            args.leuven_words,
            output,
            protocol,
            args.max_unresolved_cells,
        )
        return
    if args.shard_index is None:
        raise ValueError("A production run requires --shard-index or --finalize")
    shard_dir = output / "shards" / f"{args.shard_index:04d}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    validate_shard_resume(shard_dir, protocol, args.shard_index)
    if args.preflight_shard:
        print(f"V4 shard {args.shard_index} checkpoint preflight passed: {shard_dir}")
        return
    pairs = build_pairs(
        args.candidate_bank,
        args.leuven_words,
        args.shard_count,
        args.shard_index,
    )
    client_class = importlib.import_module("vllm_client").VLLMClient
    client = client_class(model_name=args.model, base_url=args.base_url)
    runner = (
        run_prompt_c_cascade_jobs
        if args.execution_mode == "prompt-c-cascade"
        else run_atomic_jobs
    )
    runner_args = {
        "job_id": f"v4_atomic_shard_{args.shard_index:04d}",
        "pairs": pairs,
        "prompts": load_prompts_by_version("v2"),
        "client": client,
        "model": args.model,
        "output_dir": shard_dir,
        "temperature": 0.0,
        "max_tokens": 400,
        "max_workers": args.max_workers,
        "resume": args.resume,
    }
    if args.execution_mode == "prompt-c-cascade":
        runner_args["confidence_threshold"] = args.cascade_confidence_threshold
    runner(**runner_args)
    validate_shard_complete(shard_dir, pairs, protocol, args.shard_index)


if __name__ == "__main__":
    main()
