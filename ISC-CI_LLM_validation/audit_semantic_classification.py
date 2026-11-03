#!/usr/bin/env python3
"""Read-only inspection of a retrieved semantic-classification snapshot."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path, PurePosixPath
import sqlite3
import tarfile
from zoneinfo import ZoneInfo

from semantic_retraining.classify_features import (
    INSTRUCTION,
    reconcile_batch_response,
    retained_labels,
    validate_record,
)
from semantic_retraining.gates import verify_sources
from semantic_retraining.inventory import inventory, sha256
from semantic_retraining.provenance import (
    atomic_json,
    digest,
    validate_output_directory,
)
from semantic_retraining.run_crosses import ROOT, load_config, source_cohort
from semantic_retraining.taxonomy import load_taxonomy, taxonomy_hash


# Targeted assistant-review flags, not gold annotations or an accuracy estimate.
REVIEW_NOTES = {
    "has a tail": (9, 6, "A tail is an exterior component, not an internal component."),
    "has claws": (9, 6, "Claws are exterior components, not internal components."),
    **{
        description: (
            21,
            5,
            "The entity performs the action; associated action concerns an agent acting on or involving the entity.",
        )
        for description in [
            "builds nests",
            "drinks water",
            "eats fish",
            "eats insects",
            "eats mice",
            "eats plants",
            "eats small animals",
            "eats worms",
        ]
    },
}


def inspect_archive(
    archive_path,
    taxonomy,
    enabled,
    expected_sha256=None,
    expected_sources=None,
    expected_catalog=None,
):
    archive_path = Path(archive_path)
    archive_sha = sha256(archive_path)
    if expected_sha256 and archive_sha != expected_sha256:
        raise ValueError("Archive SHA-256 does not match expected Git LFS object")
    issues, requests, repeated = [], [], []
    classes = {row["id"]: row for row in taxonomy["classes"]}
    with tarfile.open(archive_path, "r:gz") as archive:
        members = {}
        for member in archive.getmembers():
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or "\\" in member.name:
                raise ValueError(f"Unsafe archive member: {member.name}")
            if member.isdir():
                continue
            if not member.isfile() or member.name in members:
                raise ValueError(
                    f"Nonregular or duplicate archive member: {member.name}"
                )
            members[member.name] = member

        def raw(name):
            return archive.extractfile(members[name]).read()

        def data(name):
            return json.loads(raw(name))

        catalog = data("inventory/catalog.json")
        sources = data("inventory/sources.json")
        manifest = data("inventory/manifest.json")
        classifier = data("classification/initial/classifier_manifest.json")
        if classifier["classifier"].get("response_protocol") is not None:
            raise ValueError(
                "This archive audit supports legacy description-echo V1 only, not R2 ID responses"
            )
        preflight = data("gates/preflight.json")
        signature = manifest["signature"]
        if signature["catalog_sha256"] != digest(catalog):
            issues.append("Archived catalog digest mismatch")
        if signature["matrix_sources"] != {
            c: r["file_sha256"] for c, r in sources.items()
        }:
            issues.append("Archived matrix-source signature mismatch")
        if signature["taxonomy_sha256"] != taxonomy_hash(taxonomy) or classifier[
            "taxonomy"
        ] != taxonomy_hash(taxonomy):
            issues.append("Taxonomy hash mismatch")
        if classifier["instruction"] != INSTRUCTION:
            issues.append("Initial classifier instruction mismatch")
        if expected_sources is not None:
            for condition, source in expected_sources.items():
                if {
                    k: v for k, v in sources.get(condition, {}).items() if k != "path"
                } != {k: v for k, v in source.items() if k != "path"}:
                    issues.append(f"Local frozen matrix metadata mismatch: {condition}")
        if expected_catalog is not None and catalog != expected_catalog:
            issues.append("Local exact-description catalog mismatch")
        expected = {text for c in enabled for text in catalog[c].values()}
        all_descriptions = {
            text for labels in catalog.values() for text in labels.values()
        }

        with sqlite3.connect(":memory:") as db:
            db.deserialize(raw("classification/initial/cache.sqlite3"))
            db.execute("PRAGMA query_only=ON")
            integrity = [row[0] for row in db.execute("PRAGMA integrity_check")]
            if integrity != ["ok"]:
                issues.append("SQLite integrity check failed: " + repr(integrity))
            cached = {
                description: json.loads(response)
                for description, response in db.execute(
                    "SELECT description,response FROM labels ORDER BY description"
                )
            }
        for description, row in cached.items():
            try:
                validate_record(row, {description}, set(classes))
            except (ValueError, TypeError, AttributeError) as error:
                issues.append(f"Invalid cached label {description!r}: {error}")
        if set(cached) - all_descriptions:
            issues.append(
                "Cache contains descriptions absent from the full source catalog"
            )

        evidence = {}
        current_requests = {}
        for name in sorted(members):
            if not name.startswith(
                "classification/requests/initial/"
            ) or not name.endswith(".json"):
                continue
            record = data(name)
            body = record["request"]
            request_sha = digest(body)
            if Path(name).stem != request_sha:
                issues.append(f"Request hash mismatch: {name}")
            payload = json.loads(body["messages"][1]["content"])
            if payload["taxonomy"] != taxonomy["classes"]:
                issues.append(f"Request taxonomy differs: {name}")
            if any(
                body.get(key) != classifier["classifier"][key]
                for key in ["model", "temperature", "seed", "max_tokens"]
            ):
                issues.append(f"Request model/sampling differs: {name}")
            if (
                body["messages"][0]["content"]
                != INSTRUCTION + " Return JSON only, without Markdown."
            ):
                issues.append(f"Request system instruction differs: {name}")
            choice = record["response"]["choices"][0]
            rows = json.loads(choice["message"]["content"])
            state = {
                "path": name,
                "request_sha256": request_sha,
                "timestamp": datetime.fromtimestamp(
                    members[name].mtime, ZoneInfo("America/New_York")
                ).isoformat(),
                "finish_reason": choice.get("finish_reason"),
                "requested": payload["features"],
                "received_rows": rows,
            }
            try:
                accepted, repairs = reconcile_batch_response(
                    rows, payload["features"], set(classes)
                )
                state.update(valid=True, repairs=repairs)
                for row in accepted:
                    if (
                        row["description"] in evidence
                        and evidence[row["description"]] != row
                    ):
                        issues.append(
                            f"Conflicting current request evidence: {row['description']!r}"
                        )
                    evidence[row["description"]] = row
            except (ValueError, TypeError, AttributeError) as error:
                returned = (
                    [row.get("description") for row in rows if isinstance(row, dict)]
                    if isinstance(rows, list)
                    else []
                )
                state.update(
                    valid=False,
                    error=str(error),
                    missing=sorted(
                        set(payload["features"])
                        - {d for d in returned if isinstance(d, str)}
                    ),
                    unexpected=sorted(
                        {d for d in returned if isinstance(d, str)}
                        - set(payload["features"])
                    ),
                )
            requests.append(state)
            current_requests[request_sha] = (name, record, payload)
        for description, row in cached.items():
            if evidence.get(description) != row:
                issues.append(
                    f"Cached label lacks matching accepted raw evidence: {description!r}"
                )

        repairs = []
        for name in sorted(members):
            if "/format_recoveries/" in name and name.endswith(".json"):
                record = data(name)
                accepted, changes = reconcile_batch_response(
                    record["received_rows"],
                    record["requested_descriptions"],
                    set(classes),
                )
                if (
                    digest(record) != Path(name).stem
                    or accepted != record["accepted_rows"]
                    or changes != record["repairs"]
                    or record["classifier_signature_sha256"] != digest(classifier)
                ):
                    issues.append(f"Format-recovery audit mismatch: {name}")
                repairs.append({"path": name, "repairs": record["repairs"]})
            if name.startswith(
                "classification/request_history/initial/"
            ) and name.endswith(".json"):
                record = data(name)
                request_sha = digest(record["request"])
                parts = Path(name).stem.split(".")
                import hashlib

                if parts != [request_sha, hashlib.sha256(raw(name)).hexdigest()]:
                    issues.append(f"Historical raw evidence hash mismatch: {name}")
                if request_sha not in current_requests:
                    continue
                current_name, current_record, payload = current_requests[request_sha]
                if record == current_record:
                    continue
                try:
                    old_rows, _ = reconcile_batch_response(
                        json.loads(
                            record["response"]["choices"][0]["message"]["content"]
                        ),
                        payload["features"],
                        set(classes),
                    )
                    new_rows, _ = reconcile_batch_response(
                        json.loads(
                            current_record["response"]["choices"][0]["message"][
                                "content"
                            ]
                        ),
                        payload["features"],
                        set(classes),
                    )
                    old = {r["description"]: r for r in old_rows}
                    for row in new_rows:
                        if old[row["description"]]["task_id"] != row["task_id"]:
                            repeated.append(
                                {
                                    "description": row["description"],
                                    "old_task_id": old[row["description"]]["task_id"],
                                    "new_task_id": row["task_id"],
                                    "historical_request": name,
                                    "current_request": current_name,
                                }
                            )
                except (ValueError, TypeError, AttributeError):
                    pass
        fingerprint = (
            data("classification/model_fingerprint.json")
            if "classification/model_fingerprint.json" in members
            else None
        )
        if fingerprint and (
            fingerprint["revision"] != "local-sha256:" + digest(fingerprint["files"])
            or fingerprint["revision"] != classifier["classifier"]["revision"]
        ):
            issues.append("Recorded model fingerprint is internally inconsistent")

    coverage = [
        {
            "condition": c,
            "enabled": c in enabled,
            "features": len(labels),
            "unique_descriptions": len(set(labels.values())),
            "classified_features": sum(text in cached for text in labels.values()),
            "classified_unique_descriptions": len(set(labels.values()) & set(cached)),
        }
        for c, labels in catalog.items()
    ]
    flags, labels = [], []
    for description, row in cached.items():
        task = classes.get(row["task_id"], {"name": "INVALID", "family": "INVALID"})
        labeled = {
            **row,
            "description_repr": repr(description),
            "task_name": task["name"],
            "family": task["family"],
        }
        labels.append(labeled)
        if (
            description in REVIEW_NOTES
            and row["task_id"] == REVIEW_NOTES[description][0]
        ):
            _, suggestion, reason = REVIEW_NOTES[description]
            flags.append(
                {
                    **labeled,
                    "priority": "high",
                    "suggested_task_id": suggestion,
                    "suggested_task_name": classes[suggestion]["name"],
                    "reason": reason,
                    "reviewer_type": "assistant_qualitative_flag_not_human_review",
                }
            )
        elif row["uncertain"]:
            flags.append(
                {
                    **labeled,
                    "priority": "uncertain",
                    "suggested_task_id": "",
                    "suggested_task_name": "",
                    "reason": "The model flagged this description as uncertain; adjudication/review remain pending.",
                    "reviewer_type": "model_uncertainty_not_human_review",
                }
            )
    final_paths = [
        "classification/classified_descriptions.json",
        "classification/protocol.json",
        "classification/human_review.csv",
        "classification/adjudication/cache.sqlite3",
    ]
    missing_final = [path for path in final_paths if path not in members]
    summary = {
        "archive": str(archive_path.resolve()),
        "archive_sha256": archive_sha,
        "technical_integrity_passed": not issues,
        "integrity_issues": issues,
        "expected_descriptions": len(expected),
        "cached_descriptions": len(cached),
        "cached_active_descriptions": len(expected & set(cached)),
        "remaining_descriptions": len(expected - set(cached)),
        "completion_fraction": len(expected & set(cached)) / len(expected),
        "classification_complete": expected.issubset(cached) and not missing_final,
        "ready_for_training": False,
        "training_readiness_note": "This audit does not authorize training or substitute for reviewed freezes and a locked training schedule.",
        "missing_final_outputs": missing_final,
        "uncertain_cached_descriptions": sum(r["uncertain"] for r in cached.values()),
        "high_priority_review_flags": sum(r["priority"] == "high" for r in flags),
        "quality_flag_caveat": "Targeted assistant flags are not independent human gold labels, an accuracy estimate or an exhaustive review.",
        "archived_preflight_passed": preflight.get("passed") is True,
        "valid_current_requests": sum(r["valid"] for r in requests),
        "failed_current_requests": sum(not r["valid"] for r in requests),
        "format_recovery_audits": len(repairs),
        "repeated_request_task_changes": len(repeated),
        "local_sources_compared": expected_sources is not None,
        "local_catalog_compared": expected_catalog is not None,
        "classifier": classifier,
        "recorded_model_fingerprint": fingerprint,
        "actual_weight_files_rehashed": False,
    }
    return {
        "summary": summary,
        "coverage": coverage,
        "labels": labels,
        "review_flags": flags,
        "requests": requests,
        "format_recoveries": repairs,
        "repeated_request_changes": repeated,
        "task_distribution": [
            {
                "task_id": i,
                "task_name": c["name"],
                "family": c["family"],
                "cached_descriptions": sum(r["task_id"] == i for r in cached.values()),
            }
            for i, c in classes.items()
        ],
    }


def write_csv(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(dict.fromkeys(k for row in rows for k in row))
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-sha256")
    args = parser.parse_args()
    validate_output_directory(args.output_dir)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            "Use a new audit directory; existing reports are not overwritten"
        )
    config = load_config()
    taxonomy = load_taxonomy(ROOT / config["taxonomy"])
    current = inventory(ROOT)
    catalog = {
        c: retained_labels(ROOT, c, row["feature_ids"]) for c, row in current.items()
    }
    result = inspect_archive(
        args.archive,
        taxonomy,
        source_cohort(config),
        args.expected_sha256,
        current,
        catalog,
    )
    with tarfile.open(args.archive) as archive:
        record = json.load(archive.extractfile("gates/preflight.json"))
    preserved = verify_sources(
        [
            {"path": p, "sha256": h}
            for p, h in record.get("preserved_sources", {}).items()
        ],
        ROOT,
    )
    result["summary"]["archived_preserved_sources_verified_locally"] = len(preserved)
    if sha256(args.archive) != result["summary"]["archive_sha256"]:
        raise ValueError("Archive changed during audit")
    args.output_dir.mkdir(parents=True)
    atomic_json(args.output_dir / "audit.json", result)
    for name in [
        "coverage",
        "labels",
        "review_flags",
        "task_distribution",
        "repeated_request_changes",
    ]:
        write_csv(args.output_dir / f"{name}.csv", result[name])
    s = result["summary"]
    failed = sorted(
        (r for r in result["requests"] if not r["valid"]), key=lambda r: r["timestamp"]
    )
    text = f"""# Retrieved Semantic Classification Audit

## Verdict

Not ready for training. This is an incomplete snapshot, not the completed cohort.
Cached active labels: {s['cached_active_descriptions']} / {s['expected_descriptions']} ({s['completion_fraction']:.2%}); remaining: {s['remaining_descriptions']}.
Technical archive/cache/evidence checks passed: {s['technical_integrity_passed']}.
Archived preflight passed: {s['archived_preflight_passed']}. This verifies startup gates, not completion of classification.
Missing final outputs: {', '.join(s['missing_final_outputs'])}.

## Technical Checks

Archive SHA-256: `{s['archive_sha256']}`.
Compared all 14 archived source matrices and the exact-description catalog with local frozen inputs.
Verified {len(preserved)} archived preservation-source hashes locally. The input archive and caches were read only; SQLite was deserialized into memory.
Current requests: {s['valid_current_requests']} valid, {s['failed_current_requests']} rejected.
Whitespace-recovery audit records: {s['format_recovery_audits']}.
Actual Qwen weight files were not available locally for rehashing; only the saved fingerprint's internal consistency was checked.
Deferred inventories can share cached descriptions with enabled sources; that overlap does not mean their full inventories were classified.

## Latest Rejected Response

{json.dumps(failed[-1] if failed else None, indent=2)}

Do not silently accept spelling edits, fuzzily match feature descriptions, or rename matrix columns.
Stable request IDs would remove dependence on exact echoed text, but changing the classifier protocol requires a separately recorded decision.

## Label Plausibility

{s['high_priority_review_flags']} high-priority assistant review flags; {s['uncertain_cached_descriptions']} model-uncertain cached descriptions.
The main clear concerns are exterior parts called internal parts, and actions performed by the entity called associated actions.
See review_flags.csv for unchanged original assignments, suggested review targets and reasons.
These are targeted qualitative concerns, not an independent human accuracy estimate or a representative sample of the full cohort.
No task ID, uncertainty flag, matrix membership or official human-review record was changed.

## Repeated Requests

Saved repeated requests have {s['repeated_request_task_changes']} task-label changes between attempts; see repeated_request_changes.csv.
Byte-identical backups are not counted as independent repeat runs. Retried failed batches are not a random stability sample.
Current accepted cache records match current raw responses; rejected historical attempts are retained, not merged into the cache.

## Next Steps

Confirm whether a newer completed export exists. Otherwise resolve the stopped batch and classification robustness before continuing.
Review category boundaries and the confident errors, not only the model's uncertain rows, before freezing assignments.
Preserve this snapshot and document any revised prompt/protocol separately. Do not train from this partial cache.
"""
    (args.output_dir / "AUDIT.md").write_text(text)
    print(
        json.dumps(
            {
                k: v
                for k, v in s.items()
                if k
                not in {
                    "classifier",
                    "recorded_model_fingerprint",
                    "missing_final_outputs",
                }
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
