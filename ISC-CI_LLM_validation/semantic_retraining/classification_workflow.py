from __future__ import annotations

import csv
import json
import os
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .classify_features import (
    INSTRUCTION,
    classify_descriptions,
    freeze_assignments,
    retained_labels,
)
from .inventory import inventory, sha256, load_matrix
from .provenance import atomic_json, digest, validate_output_directory
from .taxonomy import taxonomy_hash

ADJUDICATION = INSTRUCTION + (
    " This is a separate ambiguity adjudication pass. Explicitly compare the most "
    "plausible neighboring classes and explain the distinction. Keep uncertain=true "
    "when the description does not support a unique decision. Return only the JSON array."
)


def model_fingerprint(directory):
    directory = Path(directory).resolve()
    weights = sorted(directory.glob("*.safetensors")) or sorted(
        directory.glob("pytorch_model*.bin")
    )
    if not weights:
        raise ValueError(
            "No Qwen weight shards found for a reproducible model fingerprint"
        )
    files = sorted(
        set(weights + list(directory.glob("*.json")) + list(directory.glob("*.model")))
    )
    hashes = {p.name: sha256(p) for p in files}
    return {
        "revision": "local-sha256:" + digest(hashes),
        "files": hashes,
        "model": "Qwen2.5-72B-Instruct",
    }


def qwen_client(endpoint, config, log_dir):
    log_dir = Path(log_dir)

    def request(payload):
        body = {
            "model": config["model"],
            "temperature": config["temperature"],
            "seed": config["seed"],
            "max_tokens": config["max_tokens"],
            "messages": [
                {
                    "role": "system",
                    "content": payload["instruction"]
                    + " Return JSON only, without Markdown.",
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "taxonomy": payload["taxonomy"],
                            "features": payload["features"],
                        }
                    ),
                },
            ],
        }
        token = os.environ.get("VLLM_API_KEY", "")
        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        req = urllib.request.Request(
            endpoint.rstrip("/") + "/chat/completions",
            data=json.dumps(body).encode(),
            headers=headers,
        )
        with urllib.request.urlopen(
            req, timeout=config.get("timeout_seconds", 300)
        ) as response:
            result = json.loads(response.read())
        request_hash = digest(body)
        atomic_json(
            log_dir / f"{request_hash}.json", {"request": body, "response": result}
        )
        return json.loads(result["choices"][0]["message"]["content"])

    return request


def prepare(root, output, taxonomy):
    validate_output_directory(output)
    sources = inventory(root)
    catalog = {
        condition: retained_labels(root, condition, data["feature_ids"])
        for condition, data in sources.items()
    }
    signature = {
        "matrix_sources": {c: d["file_sha256"] for c, d in sources.items()},
        "catalog_sha256": digest(catalog),
        "taxonomy_sha256": taxonomy_hash(taxonomy),
    }
    manifest = output / "inventory/manifest.json"
    if manifest.exists() and json.loads(manifest.read_text())["signature"] != signature:
        raise ValueError("Inventory changed; use a new experiment namespace")
    atomic_json(output / "inventory/sources.json", sources)
    atomic_json(output / "inventory/catalog.json", catalog)
    atomic_json(manifest, {"complete": True, "signature": signature})
    return sources, catalog


def review_sample(catalog, records, per_stratum=3, seed=20261028):
    rng = np.random.default_rng(seed)
    selected = {}
    for condition, labels in catalog.items():
        grouped = {}
        for text in sorted(set(labels.values())):
            grouped.setdefault(records[text]["task_id"], []).append(text)
        for task, texts in sorted(grouped.items()):
            chosen = rng.choice(texts, min(per_stratum, len(texts)), replace=False)
            for text in chosen:
                row = selected.setdefault(
                    text,
                    {
                        "description": str(text),
                        "predicted_task_id": task,
                        "task_id": "",
                        "reviewer": "",
                        "strata": [],
                    },
                )
                row["strata"].append(f"{condition}:{task}")
    for text, record in records.items():
        if record["uncertain"]:
            selected.setdefault(
                text,
                {
                    "description": text,
                    "predicted_task_id": record["task_id"],
                    "task_id": "",
                    "reviewer": "",
                    "strata": ["unresolved_ambiguity"],
                },
            )
    return [selected[text] for text in sorted(selected)]


def classify_catalog(
    output, taxonomy, classifier_config, endpoint, batch_size=8, conditions=None
):
    catalog = json.loads((output / "inventory/catalog.json").read_text())
    if conditions:
        catalog = {c: catalog[c] for c in conditions}
    texts = list(
        dict.fromkeys(text for labels in catalog.values() for text in labels.values())
    )
    directory = output / "classification"
    protocol = {
        "taxonomy_sha256": taxonomy_hash(taxonomy),
        "classifier": classifier_config,
        "initial_instruction": INSTRUCTION,
        "adjudication_instruction": ADJUDICATION,
    }
    if (directory / "protocol.json").exists() and json.loads(
        (directory / "protocol.json").read_text()
    ) != protocol:
        raise ValueError(
            "Classification protocol changed; use a new experiment namespace"
        )
    client = qwen_client(endpoint, classifier_config, directory / "requests/initial")
    initial = classify_descriptions(
        texts, taxonomy, classifier_config, directory / "initial", client, batch_size
    )
    ambiguous = [text for text, row in initial.items() if row["uncertain"]]
    adjudicated = classify_descriptions(
        ambiguous,
        taxonomy,
        classifier_config,
        directory / "adjudication",
        qwen_client(endpoint, classifier_config, directory / "requests/adjudication"),
        batch_size,
        instruction=ADJUDICATION,
    )
    records = {
        text: {
            **adjudicated.get(text, row),
            "initial_task_id": row["task_id"],
            "initial_uncertain": row["uncertain"],
            "adjudicated": text in adjudicated,
        }
        for text, row in initial.items()
    }
    if (directory / "classified_descriptions.json").exists():
        previous = json.loads((directory / "classified_descriptions.json").read_text())
        if any(
            text in previous and previous[text] != row for text, row in records.items()
        ):
            raise ValueError("A previously classified exact description changed")
        records = {**previous, **records}
    atomic_json(directory / "classified_descriptions.json", records)
    atomic_json(directory / "protocol.json", protocol)
    path = directory / "human_review.csv"
    existing = set()
    if path.exists():
        with path.open(newline="") as stream:
            existing = {r["description"] for r in csv.DictReader(stream)}
    rows = [
        row
        for row in review_sample(catalog, records)
        if row["description"] not in existing
    ]
    with path.open("a", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "description",
                "predicted_task_id",
                "task_id",
                "reviewer",
                "strata",
            ],
        )
        if not existing:
            writer.writeheader()
        writer.writerows({**row, "strata": json.dumps(row["strata"])} for row in rows)
    return {
        "unique_descriptions": len(records),
        "adjudicated": len(ambiguous),
        "unresolved": sum(r["uncertain"] for r in records.values()),
        "human_review": str(path),
    }


def classification_smoke(output, taxonomy, classifier_config, endpoint):
    catalog = json.loads((output / "inventory/catalog.json").read_text())
    texts = list(dict.fromkeys(catalog["human"].values()))[:8]
    directory = output / "classification/smoke"
    records = classify_descriptions(
        texts,
        taxonomy,
        classifier_config,
        directory,
        qwen_client(endpoint, classifier_config, directory / "requests"),
        batch_size=8,
    )
    result = {
        "passed": len(records) == len(texts),
        "smoke_only": True,
        "features": len(records),
        "production_labels_modified": False,
    }
    atomic_json(directory / "smoke_result.json", result)
    return result


def freeze_catalog(output, taxonomy, review_path, conditions=None):
    catalog = json.loads((output / "inventory/catalog.json").read_text())
    sources = json.loads((output / "inventory/sources.json").read_text())
    records = json.loads(
        (output / "classification/classified_descriptions.json").read_text()
    )
    if conditions:
        catalog = {c: catalog[c] for c in conditions}
    protocol = json.loads((output / "classification/protocol.json").read_text())
    if protocol["taxonomy_sha256"] != taxonomy_hash(taxonomy):
        raise ValueError("Classification taxonomy changed")
    with Path(review_path).open(newline="") as stream:
        reviews = list(csv.DictReader(stream))
    for row in reviews:
        if not row.get("task_id") and not row.get("reviewer"):
            continue
        if (
            not row.get("task_id", "").isdigit()
            or not row.get("reviewer", "").strip()
            or row["description"] not in records
        ):
            raise ValueError(
                "Human review has missing or unknown descriptions, reviewer, or task IDs"
            )
        row["task_id"] = int(row["task_id"])
    reviews = [
        {
            "description": r["description"],
            "task_id": r["task_id"],
            "reviewer": r["reviewer"],
        }
        for r in reviews
        if type(r.get("task_id")) is int
    ]
    mappings = {
        condition: freeze_assignments(
            labels,
            {text: records[text] for text in set(labels.values()) if text in records},
            taxonomy,
            [r for r in reviews if r["description"] in set(labels.values())],
        )
        for condition, labels in catalog.items()
    }
    if (
        "human" in mappings
        and "random_human" in mappings
        and mappings["human"]["assignments"] != mappings["random_human"]["assignments"]
    ):
        raise ValueError("Human/random task mappings differ")
    signature = {
        "taxonomy_sha256": taxonomy_hash(taxonomy),
        "catalog_sha256": digest(catalog),
        "protocol_sha256": digest(protocol),
        "records_sha256": digest(records),
        "reviews_sha256": digest(reviews),
        "sources": {c: d["file_sha256"] for c, d in sources.items()},
    }
    manifest_path = output / "classification/frozen/manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if previous:
        if any(
            previous["signature"][key] != signature[key]
            for key in ["taxonomy_sha256", "protocol_sha256"]
        ):
            raise ValueError(
                "Frozen classification protocol changed; use a new namespace"
            )
        for name, expected in previous["outputs"].items():
            if sha256(manifest_path.parent / name) != expected:
                raise ValueError("Frozen classification hash mismatch")
    outputs = dict(previous["outputs"]) if previous else {}
    for condition, mapping in mappings.items():
        if (
            sha256(Path(sources[condition]["path"]))
            != sources[condition]["file_sha256"]
        ):
            raise ValueError("Matrix changed since inventory")
        mapping.update(
            matrix_file_sha256=sources[condition]["file_sha256"],
            protocol_sha256=digest(protocol),
        )
        relevant = set(catalog[condition].values())
        mapping["classification_evidence"] = {
            text: records[text] for text in sorted(relevant)
        }
        mapping["human_reviews"] = [r for r in reviews if r["description"] in relevant]
        mapping["review_agreement"] = float(
            np.mean(
                [
                    records[r["description"]]["task_id"] == r["task_id"]
                    for r in mapping["human_reviews"]
                ]
            )
        )
        mapping_path = manifest_path.parent / f"{condition}.json"
        if mapping_path.exists() and json.loads(mapping_path.read_text()) != mapping:
            raise ValueError(
                "Frozen assignments cannot be overwritten; use a new output namespace"
            )
        atomic_json(mapping_path, mapping)
        outputs[f"{condition}.json"] = sha256(
            manifest_path.parent / f"{condition}.json"
        )
        from .training_adapter import TaskDataset

        matrix = load_matrix(Path(sources[condition]["path"]))
        dataset = TaskDataset(matrix, mapping["assignments"], len(taxonomy["classes"]))
        exposure = dataset.exposure()
        audit = []
        for task in taxonomy["classes"]:
            index = task["id"]
            texts = {
                text
                for feature, text in catalog[condition].items()
                if mapping["assignments"][feature] == index
            }
            reviewed = [
                row for row in mapping["human_reviews"] if row["description"] in texts
            ]
            audit.append(
                {
                    "task_id": index,
                    "class_name": task["name"],
                    "family": task["family"],
                    "features": exposure["features_per_task"][index],
                    "positive_associations": exposure["positive_associations_per_task"][
                        index
                    ],
                    "valid_object_task_pairs": exposure["patterns_per_task"][index],
                    "inactive": index in exposure["inactive_tasks"],
                    "sparse_feature_count_le_3": exposure["features_per_task"][index]
                    <= 3,
                    "initial_uncertain_descriptions": sum(
                        records[text].get(
                            "initial_uncertain", records[text]["uncertain"]
                        )
                        for text in texts
                    ),
                    "human_reviewed_descriptions": len(reviewed),
                    "human_review_agreement": (
                        float(
                            np.mean(
                                [
                                    records[row["description"]]["task_id"]
                                    == row["task_id"]
                                    for row in reviewed
                                ]
                            )
                        )
                        if reviewed
                        else None
                    ),
                    "unresolved_after_freeze": 0,
                }
            )
        frames = {
            f"{condition}_task_audit.csv": pd.DataFrame(audit),
            f"{condition}_object_coverage.csv": pd.DataFrame(
                {
                    "object_id": matrix.index,
                    "positive_features": matrix.sum(axis=1).values,
                    "valid_tasks": exposure["patterns_per_object"],
                }
            ),
            f"{condition}_feature_coverage.csv": pd.DataFrame(
                {
                    "feature_id": matrix.columns,
                    "task_id": [mapping["assignments"][f] for f in matrix.columns],
                    "positive_objects": matrix.sum(axis=0).values,
                }
            ),
        }
        for name, frame in frames.items():
            frame.to_csv(manifest_path.parent / name, index=False)
            outputs[name] = sha256(manifest_path.parent / name)
    conditions_done = sorted(
        set(previous["conditions"] if previous else []) | set(mappings)
    )
    manifest = {
        "complete": True,
        "signature": signature,
        "conditions": conditions_done,
        "outputs": outputs,
    }
    atomic_json(manifest_path, manifest)
    return manifest
