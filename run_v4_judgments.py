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
from leuven_expansion.run_jobs import run_atomic_jobs
from leuven_expansion.v4 import sha256_file, stable_json_hash, stable_shard, write_json


ROOT = Path(__file__).resolve().parent
PROMPT_DIR = ROOT / "leuven_expansion" / "prompts"
V2_RESOLVED = ROOT / "artifacts" / "leuven_full_labels" / "leuven_full_v2" / "feature_resolutions.csv"


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
    return {
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
            ROOT / "leuven_expansion" / "schemas" / "atomic_feature_judgment_schema_v1.json"
        ),
        "shard_count": shard_count,
        "shard_function": "sha256(candidate_id NUL normalized_word) first 8 bytes modulo shard_count",
        "v2_manifest": str(v2_manifest.resolve()) if v2_manifest else None,
        "v2_manifest_sha256": sha256_file(v2_manifest) if v2_manifest else None,
        "v2_reference": v2,
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


def read_csv_or_empty(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() and path.stat().st_size else pd.DataFrame()


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
        if any(actual.get(key) != value for key, value in expected.items()):
            raise ValueError("Cannot resume V4 shard under a changed protocol")
    votes = read_csv_or_empty(shard_dir / "feature_votes.csv")
    resolutions = read_csv_or_empty(shard_dir / "feature_resolutions.csv")
    if not votes.empty:
        complete_vote_keys = set(
            votes.groupby(["word_normalized", "feature_id"])["judge_id"]
            .nunique()
            .loc[lambda values: values >= 3]
            .index
        )
        resolution_keys = (
            set(zip(resolutions["word_normalized"], resolutions["feature_id"]))
            if not resolutions.empty
            else set()
        )
        missing_resolution = complete_vote_keys - resolution_keys
        if missing_resolution:
            raise ValueError(
                "Completed votes lack resolutions; preserve files and audit before resume: "
                f"{list(missing_resolution)[:5]}"
            )
    write_json(sidecar, expected | {"candidate_inventory_hash": protocol["candidate_inventory_hash"]})


def v2_adjudication_rate() -> float:
    if not V2_RESOLVED.exists():
        return 0.10
    resolved = pd.read_csv(V2_RESOLVED, usecols=["adjudicated"])
    return float(resolved["adjudicated"].fillna(False).astype(bool).mean())


def dry_run_report(
    protocol: dict[str, Any], output_dir: Path
) -> dict[str, Any]:
    existing_resolved = 0
    for path in (output_dir / "shards").glob("*/feature_resolutions.csv"):
        existing_resolved += len(pd.read_csv(path, usecols=["feature_id"]))
    cells = int(protocol["expected_cell_count"])
    adjudication_rate = v2_adjudication_rate()
    remaining = max(0, cells - existing_resolved)
    report = {
        **{key: protocol[key] for key in ["candidate_count", "word_count", "expected_cell_count", "shard_count"]},
        "planned_first_pass_calls": cells * 3,
        "v2_observed_adjudication_rate": adjudication_rate,
        "estimated_adjudicated_cells_low": int(remaining * max(0.0, adjudication_rate * 0.8)),
        "estimated_adjudicated_cells_high": int(remaining * min(1.0, adjudication_rate * 1.2)),
        "estimated_adjudication_calls_low": int(remaining * max(0.0, adjudication_rate * 0.8) * 3),
        "estimated_adjudication_calls_high": int(remaining * min(1.0, adjudication_rate * 1.2) * 3),
        "existing_reusable_cells": existing_resolved,
        "remaining_cells": remaining,
        "primary_execution": "exhaustive candidate-by-word judging; no retrieval pruning",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "dry_run_cost_report.json", report)
    return report


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
        set(zip(resolutions["word_normalized"].astype(str), resolutions["feature_id"].astype(int)))
        if not resolutions.empty
        else set()
    )
    duplicates = (
        int(resolutions.duplicated(["word_normalized", "feature_id"]).sum())
        if not resolutions.empty
        else 0
    )
    vote_counts = (
        votes.groupby(["word_normalized", "feature_id"])["judge_id"].nunique()
        if not votes.empty
        else pd.Series(dtype=int)
    )
    invalid_vote_counts = int((vote_counts != 3).sum())
    missing_vote_keys = len(expected - set(vote_counts.index))
    complete = (
        actual == expected
        and duplicates == 0
        and invalid_vote_counts == 0
        and missing_vote_keys == 0
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
        "invalid_first_pass_vote_counts": invalid_vote_counts,
        "missing_first_pass_vote_cells": missing_vote_keys,
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
                    raise ValueError(f"Shard CSV headers differ for {target.name}: {path}")
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


def finalize(
    candidate_bank: Path,
    leuven_words: Path,
    output_dir: Path,
    protocol: dict[str, Any],
) -> None:
    schema = load_candidate_feature_schema(candidate_bank)
    words = load_words(leuven_words)
    word_set = {row["word_normalized"] for row in words}
    shard_count = int(protocol["shard_count"])
    shard_dirs = [output_dir / "shards" / f"{index:04d}" for index in range(shard_count)]
    expected_by_shard = [0] * shard_count
    for candidate_id in schema["candidate_ids"]:
        for word in words:
            expected_by_shard[
                stable_shard(candidate_id, word["word_normalized"], shard_count)
            ] += 1

    protocol_hash = stable_json_hash(protocol)
    for index, shard_dir in enumerate(shard_dirs):
        sidecar = shard_dir / "v4_shard_manifest.json"
        if not sidecar.exists():
            raise ValueError(f"Missing V4 shard manifest: {sidecar}")
        shard_manifest = json.loads(sidecar.read_text(encoding="utf-8"))
        if (
            shard_manifest.get("protocol_hash") != protocol_hash
            or shard_manifest.get("expected_cells") != expected_by_shard[index]
            or not shard_manifest.get("complete")
        ):
            raise ValueError(f"V4 shard {index} is incomplete or belongs to another protocol")

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
    _concatenate_logs([shard / "run.log" for shard in shard_dirs], output_dir / "run.log")

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
    resolved_cells = 0
    with resolved_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=required_columns)
        writer.writeheader()
        for shard_index, shard_dir in enumerate(shard_dirs):
            resolutions = pd.read_csv(shard_dir / "feature_resolutions.csv")
            votes = pd.read_csv(
                shard_dir / "feature_votes.csv",
                usecols=["word_normalized", "feature_id", "judge_id", "prompt_hash"],
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
                    for candidate, word in zip(candidate_ids, resolutions["word_normalized"])
                ),
                dtype=bool,
                count=len(resolutions),
            )
            if invalid_id.any() or invalid_word.any() or invalid_value.any() or wrong_shard.any():
                raise ValueError(f"Invalid cell content in shard {shard_index}")
            if len(resolutions) != expected_by_shard[shard_index]:
                raise ValueError(f"Unexpected resolution count in shard {shard_index}")
            vote_counts = votes.groupby(["word_normalized", "feature_id"])["judge_id"].nunique()
            if len(vote_counts) != len(resolutions) or (vote_counts != 3).any():
                raise ValueError(f"Shard {shard_index} does not have exactly three first-pass votes per cell")
            prompt_hashes = (
                votes.groupby(["word_normalized", "feature_id"], observed=True)["prompt_hash"]
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
            resolved["resolved_binary_locked_v2"] = resolved["resolved_value"].gt(0).astype(int)
            resolved["judge_model_revision"] = protocol["judge_model_revision"]
            for column in required_columns:
                if column not in resolved:
                    resolved[column] = ""
            writer.writerows(resolved[required_columns].to_dict(orient="records"))
            resolved_cells += len(resolved)

    if resolved_cells != int(protocol["expected_cell_count"]):
        raise ValueError("Final resolved cell count differs from the frozen cross-product")
    pd.DataFrame(columns=["candidate_id", "target_word", "reason"]).to_csv(
        output_dir / "unresolved_cells.csv", index=False
    )
    manifest = protocol | {
        "protocol_hash": protocol_hash,
        "resolved_cells": resolved_cells,
        "unresolved_cells": 0,
        "duplicate_cells": 0,
        "feature_votes": raw_counts["feature_votes.csv"],
        "adjudication_votes": raw_counts["feature_adjudication_votes.csv"],
        "parse_errors": raw_counts["parse_errors.csv"],
        "resolved_values_sha256": sha256_file(output_dir / "resolved_feature_values.csv"),
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
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.shard_index is not None and not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard-index must be in [0, shard-count)")
    protocol = protocol_record(
        args.candidate_bank.resolve(),
        args.leuven_words.resolve(),
        args.model,
        args.shard_count,
        args.v2_manifest.resolve() if args.v2_manifest else None,
    )
    output = args.output_dir.resolve()
    if args.dry_run:
        print(json.dumps(dry_run_report(protocol, output), indent=2))
        return
    if args.finalize:
        finalize(args.candidate_bank, args.leuven_words, output, protocol)
        return
    if args.shard_index is None:
        raise ValueError("A production run requires --shard-index or --finalize")
    shard_dir = output / "shards" / f"{args.shard_index:04d}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    validate_shard_resume(shard_dir, protocol, args.shard_index)
    pairs = build_pairs(
        args.candidate_bank,
        args.leuven_words,
        args.shard_count,
        args.shard_index,
    )
    client_class = importlib.import_module("vllm_client").VLLMClient
    client = client_class(model_name=args.model, base_url=args.base_url)
    run_atomic_jobs(
        job_id=f"v4_atomic_shard_{args.shard_index:04d}",
        pairs=pairs,
        prompts=load_prompts_by_version("v2"),
        client=client,
        model=args.model,
        output_dir=shard_dir,
        temperature=0.0,
        max_tokens=400,
        max_workers=args.max_workers,
        resume=args.resume,
    )
    validate_shard_complete(shard_dir, pairs, protocol, args.shard_index)


if __name__ == "__main__":
    main()
