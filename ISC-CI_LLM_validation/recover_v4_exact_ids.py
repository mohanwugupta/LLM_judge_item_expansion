#!/usr/bin/env python3
"""Recover audited V4 exact-ID formatting failures; never infer semantic scores."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from leuven_expansion.v4 import sha256_file, write_json


def source_file(path: Path) -> Path:
    """Use a materialized CSV or an immutable, already-fetched LFS object."""
    with path.open("rb") as handle:
        prefix = handle.read(128)
    if not prefix.startswith(b"version https://git-lfs.github.com/spec/v1"):
        return path
    fields = dict(line.split(" ", 1) for line in path.read_text().splitlines())
    oid = fields["oid"].removeprefix("sha256:")
    cached = ROOT / ".git/lfs/objects" / oid[:2] / oid[2:4] / oid
    if not cached.exists() or cached.stat().st_size != int(fields["size"]):
        raise ValueError(f"Materialize the Git LFS input before recovery: {path}")
    return cached


def prepare(audit: Path, judgments: Path, output: Path) -> dict:
    """Reuse the read-only audit's conservative parser and recorded source hashes."""
    from audit_v4_pretraining import preview_type_only_recovery

    output.mkdir(parents=True, exist_ok=True)
    if (output / "recovery_plan.json").exists():
        raise ValueError("Recovery plan already exists; use --apply to resume it")
    original = json.loads((audit / "audit_summary.json").read_text())
    for name in ("invalid_cell_adjudication_votes.csv", "invalid_resolutions.csv"):
        shutil.copy2(audit / name, output / name)
    summary = preview_type_only_recovery(output)
    preview = pd.read_csv(output / "offline_recovery_preview.csv")
    failed = pd.read_csv(output / "invalid_resolutions.csv", keep_default_na=False)
    keys = ["shard_index", "word_normalized", "feature_id"]
    if failed.duplicated(keys).any() or preview.duplicated(keys).any():
        raise ValueError("Duplicate cells in recovery evidence")
    joined = failed.merge(
        preview, on=keys, how="outer", validate="one_to_one", indicator=True
    )
    if not joined["_merge"].eq("both").all():
        raise ValueError(
            "Every failed cell must have corresponding raw-response evidence"
        )
    if not joined["feature_text_x"].eq(joined["feature_text_y"]).all():
        raise ValueError("Feature text differs between failure and response evidence")
    retry = joined.loc[
        ~joined["unanimous_type_only_recovery"],
        keys + ["candidate_id", "feature_text_x"],
    ]
    retry = retry.rename(columns={"feature_text_x": "feature_text"})
    retry.to_csv(output / "retry_cells.csv", index=False)
    records = {}
    for relative, digest in original["resolution_file_sha256"].items():
        index = int(Path(relative).parent.name)
        sidecar = judgments / "shards" / f"{index:04d}" / "v4_shard_manifest.json"
        manifest = json.loads(sidecar.read_text())
        failed_count = int(failed["shard_index"].eq(index).sum())
        if (
            manifest["shard_index"] != index
            or manifest["invalid_resolved_values"] != failed_count
        ):
            raise ValueError(f"Shard {index} no longer matches the audited failures")
        records[str(index)] = {
            "source_sha256": digest,
            "sidecar_sha256": sha256_file(sidecar),
            "protocol_hash": manifest["protocol_hash"],
            "expected_cells": manifest["expected_cells"],
            "invalid_before": failed_count,
        }
    plan = {
        "policy": "Exact matching decimal-string ID -> integer only; existing schema/word checks; >=2 recovered responses with identical scores. No changes to scores, prompts, thresholds, or protocol hashes.",
        "audit_summary_sha256": sha256_file(audit / "audit_summary.json"),
        "evidence_sha256": {
            name: sha256_file(output / name)
            for name in (
                "invalid_cell_adjudication_votes.csv",
                "invalid_resolutions.csv",
                "offline_recovery_preview.csv",
                "retry_cells.csv",
            )
        },
        "summary": summary,
        "shards": records,
        "implementation_sha256": {
            "recovery": sha256_file(Path(__file__)),
            "audit_parser": sha256_file(
                Path(__file__).with_name("audit_v4_pretraining.py")
            ),
            "original_schema_parser": sha256_file(
                ROOT / "leuven_expansion/feature_schema.py"
            ),
        },
        "applied": False,
    }
    write_json(output / "recovery_plan.json", plan)
    return plan


def patch_shard(target: Path, patches: dict, record: dict, destination: Path) -> dict:
    """Atomically replace only failed scores, keeping original rows and source bytes."""
    destination.mkdir(parents=True, exist_ok=True)
    journal_path = destination / "application.json"
    if journal_path.exists():
        previous = json.loads(journal_path.read_text())
        if sha256_file(source_file(target)) == previous["after_sha256"]:
            if previous["status"] != "applied":
                previous["status"] = "applied"
                write_json(journal_path, previous)
            return previous
    source = source_file(target)
    before = sha256_file(source)
    if before != record["source_sha256"]:
        raise ValueError(f"Resolution source differs from the audited input: {target}")
    backup = destination / "original_feature_resolutions.csv"
    if not backup.exists():
        try:
            os.link(source, backup)
        except OSError:
            shutil.copy2(source, backup)
    elif sha256_file(backup) != before:
        raise ValueError(f"Original backup changed: {backup}")
    temporary = target.with_suffix(".csv.exact_id_recovery.tmp")
    seen, row_count = set(), 0
    try:
        with source.open(newline="") as input_handle, temporary.open(
            "w", newline=""
        ) as output_handle, (destination / "original_failed_rows.csv").open(
            "w", newline=""
        ) as failed_handle:
            reader = csv.reader(input_handle)
            header = next(reader)
            columns = {
                name: header.index(name)
                for name in (
                    "word_normalized",
                    "feature_id",
                    "feature_text",
                    "final_feature_value",
                    "resolution_method",
                    "needs_human_audit",
                )
            }
            writer = csv.writer(output_handle, lineterminator="\n")
            originals = csv.writer(failed_handle, lineterminator="\n")
            writer.writerow(header)
            originals.writerow(header)
            for row in reader:
                if len(row) != len(header):
                    raise ValueError(f"Malformed resolution CSV row in {target}")
                row_count += 1
                key = (row[columns["word_normalized"]], row[columns["feature_id"]])
                if key in patches:
                    patch = patches[key]
                    if (
                        key in seen
                        or row[columns["resolution_method"]] != "adjudicator_failed"
                    ):
                        raise ValueError(
                            f"Recovery target is duplicated or not failed: {key}"
                        )
                    if (
                        row[columns["feature_text"]] != patch["feature_text"]
                        or row[columns["final_feature_value"]].strip()
                    ):
                        raise ValueError(
                            f"Recovery target differs from saved failure: {key}"
                        )
                    originals.writerow(row)
                    row[columns["final_feature_value"]] = str(patch["value"])
                    row[columns["resolution_method"]] = (
                        "adjudicator_agree_exact_id_type_recovery"
                    )
                    row[columns["needs_human_audit"]] = "False"
                    seen.add(key)
                writer.writerow(row)
        if seen != set(patches) or row_count != record["expected_cells"]:
            raise ValueError(
                f"Recovery target coverage/count differs from the audit: {target}"
            )
        result = {
            "status": "prepared_for_atomic_replace",
            "before_sha256": before,
            "after_sha256": sha256_file(temporary),
            "original_backup": str(backup),
            "recovered_cells": len(seen),
            "remaining_invalid_cells": record["invalid_before"] - len(seen),
            "rows": row_count,
            "sidecar_changed": False,
        }
        write_json(journal_path, result)
        temporary.replace(target)
        result["status"] = "applied"
        write_json(journal_path, result)
        return result
    finally:
        if temporary.exists():
            temporary.unlink()


def apply(judgments: Path, output: Path) -> dict:
    plan = json.loads((output / "recovery_plan.json").read_text())
    for name, digest in plan["evidence_sha256"].items():
        if sha256_file(output / name) != digest:
            raise ValueError(f"Recovery evidence changed: {name}")
    preview = pd.read_csv(output / "offline_recovery_preview.csv")
    preview = preview[preview["unanimous_type_only_recovery"]]
    summary = {
        "applied": True,
        "shards": {},
        "recovered_cells": 0,
        "remaining_invalid_cells": 0,
    }
    # No shard writers may run concurrently with this offline operation.
    lock_path = output / ".apply.lock"
    with lock_path.open("x"):
        pass
    try:
        for index, record in plan["shards"].items():
            shard = judgments / "shards" / f"{int(index):04d}"
            sidecar = shard / "v4_shard_manifest.json"
            if sha256_file(sidecar) != record["sidecar_sha256"]:
                raise ValueError(
                    f"Shard manifest changed after audit/recovery planning: {sidecar}"
                )
            patches = {
                (row.word_normalized, str(int(row.feature_id))): {
                    "feature_text": row.feature_text,
                    "value": float(row.proposed_final_value),
                }
                for row in preview[preview["shard_index"].eq(int(index))].itertuples(
                    index=False
                )
            }
            result = patch_shard(
                shard / "feature_resolutions.csv",
                patches,
                record,
                output / "shards" / f"{int(index):04d}",
            )
            summary["shards"][index] = result
            summary["recovered_cells"] += result["recovered_cells"]
            summary["remaining_invalid_cells"] += result["remaining_invalid_cells"]
            write_json(output / "application_manifest.json", summary)
            print(
                f"Shard {int(index):02d}: recovered {result['recovered_cells']}; remaining {result['remaining_invalid_cells']}",
                flush=True,
            )
        summary["plan_sha256"] = sha256_file(output / "recovery_plan.json")
        summary["note"] = (
            "All original sidecars remain unchanged and stale. Revalidate CSVs under the recorded execution protocol after retrying remaining cells; do not mark shards complete from recovery counts alone."
        )
        write_json(output / "application_manifest.json", summary)
        return summary
    finally:
        lock_path.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit-dir",
        type=Path,
        default=Path(__file__).parent / "reports/v4_pretraining_audit_20261001",
    )
    parser.add_argument(
        "--judgments-dir", type=Path, default=ROOT / "artifacts/v4/judgments"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "reports/v4_exact_id_recovery_20261001",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply an existing, hash-verified plan; no LLM calls",
    )
    args = parser.parse_args()
    result = (
        apply(args.judgments_dir, args.output_dir)
        if args.apply
        else prepare(args.audit_dir, args.judgments_dir, args.output_dir)
    )
    print(json.dumps({k: v for k, v in result.items() if k != "shards"}, indent=2))


if __name__ == "__main__":
    main()
