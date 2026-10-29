from __future__ import annotations

import csv
import json
import sqlite3
import logging
from pathlib import Path

from .provenance import atomic_json, digest
from .taxonomy import taxonomy_hash, validate_taxonomy

INSTRUCTION = (
    "Classify each exact semantic feature description into one taxonomy class. "
    "Return a JSON array with description, integer task_id, boolean uncertain, "
    "and explanation. Flag ambiguity rather than forcing a decision. "
    "Do not judge applicability or use object memberships or behavioral outcomes."
)


def retained_labels(root: Path, condition: str, feature_ids: list[str]) -> dict:
    if condition in {"human", "random_human", "v2"}:
        return {f: f for f in feature_ids}
    if condition.startswith("v4_"):
        path = root.parent / "artifacts/v4/discovery/candidate_bank.csv"
        key, value = "candidate_id", "canonical_feature_text"
    else:
        generation = "v3_1" if condition.startswith("v3_1_") else "v3"
        path = (
            root
            / f"artifacts/{generation}_consolidation/{condition}_cluster_labels.csv"
        )
        key, value = "cluster_id", "canonical_feature"
    wanted = set(feature_ids)
    labels = {}
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not {key, value}.issubset(reader.fieldnames):
            raise ValueError(
                f"Missing feature-description columns in {path}; materialize Git LFS metadata before classification"
            )
        for row in reader:
            if row[key] in wanted:
                if row[key] in labels:
                    raise ValueError(f"Duplicate feature metadata: {row[key]}")
                labels[row[key]] = row[value]
    if set(labels) != wanted or any(not text.strip() for text in labels.values()):
        raise ValueError(f"Missing retained feature descriptions for {condition}")
    return {f: labels[f] for f in feature_ids}


def validate_record(row, descriptions, task_ids):
    if row.get("description") not in descriptions:
        raise ValueError("Unknown feature description in classifier response")
    if type(row.get("task_id")) is not int or row["task_id"] not in task_ids:
        raise ValueError("Invalid taxonomy task ID")
    if type(row.get("uncertain")) is not bool or not row.get("explanation", "").strip():
        raise ValueError("Classification requires explicit uncertainty and evidence")


def classify_descriptions(
    descriptions,
    taxonomy,
    config,
    directory,
    classifier,
    batch_size=16,
    instruction=INSTRUCTION,
):
    validate_taxonomy(taxonomy)
    if (
        not config.get("model")
        or not config.get("revision")
        or config.get("temperature") != 0
        or type(config.get("seed")) is not int
    ):
        raise ValueError("Freeze classifier model, revision, temperature=0, and seed")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    signature = {
        "taxonomy": taxonomy_hash(taxonomy),
        "classifier": config,
        "instruction": instruction,
    }
    directory = Path(directory)
    manifest = directory / "classifier_manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != signature:
        raise ValueError("Cached classifier configuration differs")
    atomic_json(manifest, signature)
    unique = list(dict.fromkeys(descriptions))
    if any(not isinstance(d, str) or not d.strip() for d in unique):
        raise ValueError("Feature descriptions must be nonempty strings")
    with sqlite3.connect(directory / "cache.sqlite3") as db:
        db.execute(
            "CREATE TABLE IF NOT EXISTS labels (description TEXT PRIMARY KEY, response TEXT NOT NULL)"
        )
        records = {
            d: json.loads(response)
            for d, response in db.execute("SELECT description,response FROM labels")
        }
        missing = [d for d in unique if d not in records]
        for start in range(0, len(missing), batch_size):
            batch = missing[start : start + batch_size]
            payload = {
                "features": batch,
                "taxonomy": taxonomy["classes"],
                "instruction": instruction,
            }
            rows = classifier(payload)
            if (
                not isinstance(rows, list)
                or len(rows) != len(batch)
                or {r.get("description") for r in rows} != set(batch)
            ):
                raise ValueError(
                    "Classifier response must cover each batch description once"
                )
            for row in rows:
                validate_record(row, batch, set(range(len(taxonomy["classes"]))))
            with db:
                for row in rows:
                    db.execute(
                        "INSERT INTO labels VALUES (?,?)",
                        (row["description"], json.dumps(row, sort_keys=True)),
                    )
                    records[row["description"]] = row
            if (start // batch_size) % 10 == 0:
                logging.getLogger(__name__).warning(
                    "Classified %s/%s new exact descriptions; cache committed",
                    min(start + batch_size, len(missing)),
                    len(missing),
                )
        return {d: records[d] for d in unique}


def freeze_assignments(labels, records, taxonomy, reviews):
    validate_taxonomy(taxonomy)
    tasks = set(range(len(taxonomy["classes"])))
    reviewed = {}
    for review in reviews:
        if (
            not review.get("reviewer")
            or type(review.get("task_id")) is not int
            or review["task_id"] not in tasks
        ):
            raise ValueError("Human review requires reviewer and valid task_id")
        if review["description"] in reviewed:
            raise ValueError("Duplicate reviewed description")
        reviewed[review["description"]] = review
    assignments = {}
    for feature_id, text in labels.items():
        if text not in records:
            raise ValueError(f"missing classification: {feature_id}")
        row = records[text]
        validate_record(row, {text}, tasks)
        if row["uncertain"] and text not in reviewed:
            raise ValueError(f"unresolved classification: {feature_id}")
        assignments[feature_id] = (
            reviewed[text]["task_id"] if text in reviewed else row["task_id"]
        )
    if not reviewed:
        raise ValueError(
            "A stratified human review is required before freezing assignments"
        )
    active = {records[text]["task_id"] for text in labels.values()}
    covered = {
        records[text]["task_id"] for text in set(labels.values()) & set(reviewed)
    }
    if active != covered:
        raise ValueError(
            f"Human review is not stratified across nonempty task classes: {sorted(active - covered)}"
        )
    return {
        "assignments": assignments,
        "taxonomy_sha256": taxonomy_hash(taxonomy),
        "descriptions_sha256": digest(labels),
        "classification_sha256": digest(records),
        "review_sha256": digest(reviews),
        "human_mapping_origin": "reconstructed, not original annotations",
    }
