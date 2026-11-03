"""Source-grounded classification with immutable IDs; legacy caches stay separate."""

from __future__ import annotations

import csv
import json
import logging
import os
import sqlite3
import time
import urllib.error
import urllib.request
from pathlib import Path
from uuid import uuid4

from .classification_workflow import freeze_catalog as legacy_freeze, review_sample
from .provenance import atomic_json, digest
from .taxonomy import taxonomy_hash, validate_taxonomy

VERSION = "stable-feature-id-v2"
PROMPT = (
    Path(__file__).resolve().parents[1] / "configs/semantic_classifier_r2_prompt.txt"
)
PROMPT_TEXT = PROMPT.read_text()
SMOKE_ANCHORS = {
    "can fly": 5,
    "has wings": 6,
    "has a heart": 9,
    "has four legs": 13,
    "is eaten": 21,
    "used for cutting": 24,
}


def instruction(adjudication=False):
    text = PROMPT_TEXT
    if adjudication:
        text += (
            "\nSeparate ambiguity pass: compare the plausible neighboring classes. "
            "Retain uncertainty when the text or inventory cannot resolve them.\n"
        )
    return text


def feature_registry(catalog):
    texts = sorted({text for labels in catalog.values() for text in labels.values()})
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("Nonempty exact feature descriptions required")
    return {f"F{index:06d}": text for index, text in enumerate(texts, 1)}


def response_schema(features, taxonomy):
    fields = {
        "feature_id": {"type": "string", "enum": [f["feature_id"] for f in features]},
        "task_id": {"type": "integer", "enum": [c["id"] for c in taxonomy["classes"]]},
        "uncertain": {"type": "boolean"},
        "out_of_inventory": {"type": "boolean"},
        "explanation": {"type": "string", "minLength": 1},
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["labels"],
        "properties": {
            "labels": {
                "type": "array",
                "minItems": len(features),
                "maxItems": len(features),
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": list(fields),
                    "properties": fields,
                },
            }
        },
    }


def validate_response(response, features, taxonomy):
    expected = {f["feature_id"]: f["description"] for f in features}
    rows = response.get("labels") if isinstance(response, dict) else None
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise ValueError("Response must cover each requested feature ID once")
    accepted, seen = [], set()
    fields = {"feature_id", "task_id", "uncertain", "out_of_inventory", "explanation"}
    tasks = {c["id"] for c in taxonomy["classes"]}
    for row in rows:
        if not isinstance(row, dict) or set(row) != fields:
            raise ValueError("Invalid response fields")
        key = row["feature_id"]
        if not isinstance(key, str) or key not in expected or key in seen:
            raise ValueError("Unknown or duplicate feature ID")
        if type(row["task_id"]) is not int or row["task_id"] not in tasks:
            raise ValueError("Invalid taxonomy task ID")
        if (
            type(row["uncertain"]) is not bool
            or type(row["out_of_inventory"]) is not bool
            or not isinstance(row["explanation"], str)
            or not row["explanation"].strip()
            or (row["out_of_inventory"] and not row["uncertain"])
        ):
            raise ValueError("Explicit uncertainty and inventory-gap review required")
        seen.add(key)
        accepted.append({**row, "description": expected[key]})
    return accepted


def qwen_client(endpoint, config, log_dir):
    def request(payload):
        body = {
            key: config[key] for key in ["model", "temperature", "seed", "max_tokens"]
        }
        body.update(
            messages=[
                {"role": "system", "content": payload["instruction"]},
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
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "isc_task_labels", "schema": payload["schema"]},
            },
        )
        evidence = {"request": body, "protocol": VERSION}
        headers = {"Content-Type": "application/json"}
        if os.environ.get("VLLM_API_KEY"):
            headers["Authorization"] = "Bearer " + os.environ["VLLM_API_KEY"]
        req = urllib.request.Request(
            endpoint.rstrip("/") + "/chat/completions",
            data=json.dumps(body).encode(),
            headers=headers,
        )
        try:
            with urllib.request.urlopen(
                req, timeout=config.get("timeout_seconds", 300)
            ) as response:
                raw = response.read().decode()
            evidence["raw_response"] = raw
            result = json.loads(raw)
            evidence["response"] = result
            choice = result["choices"][0]
            if choice.get("finish_reason") != "stop":
                raise ValueError("Model response did not finish normally")
            return json.loads(choice["message"]["content"])
        except urllib.error.HTTPError as exc:
            evidence.update(
                error=str(exc),
                http_status=exc.code,
                error_body=exc.read().decode(errors="replace"),
            )
            if 400 <= exc.code < 500 and exc.code not in {408, 429}:
                raise ConfigurationError(
                    "vLLM rejected the request; inspect request log"
                ) from exc
            raise
        except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
            evidence["error"] = str(exc)
            raise
        finally:
            # Distinct attempt IDs retain every raw response, including retries/resumes.
            atomic_json(Path(log_dir) / f"{digest(body)}.{uuid4().hex}.json", evidence)

    return request


class ConfigurationError(ValueError):
    pass


def classify_ids(
    texts,
    registry,
    taxonomy,
    config,
    directory,
    client,
    batch_size=8,
    adjudication=False,
):
    validate_taxonomy(taxonomy)
    if (
        config.get("response_protocol") != VERSION
        or not config.get("model")
        or not config.get("revision")
        or config.get("temperature") != 0
        or type(config.get("seed")) is not int
    ):
        raise ValueError("Freeze R2 model, revision, temperature=0 and seed")
    attempts = config.get("attempts_per_batch", 2)
    if type(attempts) is not int or not 1 <= attempts <= 3 or batch_size < 1:
        raise ValueError("Use positive batch size and 1..3 attempts per batch")
    unique = list(dict.fromkeys(texts))
    by_text = {text: key for key, text in registry.items()}
    if len(by_text) != len(registry) or not set(unique).issubset(by_text):
        raise ValueError("Registry must cover all exact descriptions uniquely")
    directory = Path(directory)
    signature = {
        "taxonomy": taxonomy_hash(taxonomy),
        "classifier": config,
        "registry_sha256": digest(registry),
        "instruction": instruction(adjudication),
    }
    manifest = directory / "classifier_manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != signature:
        raise ValueError("Cached classifier configuration differs; use a new namespace")
    if not manifest.exists() and (directory / "cache.sqlite3").exists():
        raise ValueError("Unversioned cache cannot be reused")
    atomic_json(manifest, signature)
    with sqlite3.connect(directory / "cache.sqlite3") as db:
        db.execute(
            "CREATE TABLE IF NOT EXISTS labels (description TEXT PRIMARY KEY, response TEXT NOT NULL)"
        )
        records = {
            text: json.loads(row)
            for text, row in db.execute("SELECT description,response FROM labels")
        }
        for text, row in records.items():
            if text not in by_text or row.get("description") != text:
                raise ValueError("Cached description differs from registry")
            wire = {key: value for key, value in row.items() if key != "description"}
            validate_response(
                {"labels": [wire]},
                [{"feature_id": by_text[text], "description": text}],
                taxonomy,
            )

        def progress():
            missing = [text for text in unique if text not in records]
            atomic_json(
                directory / "progress.json",
                {
                    "complete": not missing,
                    "requested": len(unique),
                    "completed": len(unique) - len(missing),
                    "missing": missing,
                    "signature_sha256": digest(signature),
                },
            )
            return missing

        def run_batch(batch):
            features = [
                {"feature_id": by_text[text], "description": text} for text in batch
            ]
            payload = {
                "instruction": instruction(adjudication),
                "features": features,
                "taxonomy": taxonomy["classes"],
                "schema": response_schema(features, taxonomy),
            }
            for attempt in range(attempts):
                evidence = {"features": features, "attempt": attempt + 1}
                try:
                    received = client(payload)
                    evidence["response"] = received
                    accepted = validate_response(received, features, taxonomy)
                    with db:
                        for row in accepted:
                            db.execute(
                                "INSERT INTO labels VALUES (?,?)",
                                (row["description"], json.dumps(row, sort_keys=True)),
                            )
                    records.update({row["description"]: row for row in accepted})
                    evidence["accepted"] = True
                    progress()
                    return
                except ConfigurationError:
                    raise
                except (ValueError, OSError, KeyError, TypeError, IndexError) as exc:
                    evidence.update(accepted=False, error=str(exc))
                    if isinstance(exc, OSError):
                        if attempt + 1 == attempts:
                            raise ValueError(
                                "vLLM unavailable after bounded retries; resume after restoring endpoint"
                            ) from exc
                        time.sleep(2**attempt)
                finally:
                    atomic_json(
                        directory / "attempts" / f"{uuid4().hex}.json", evidence
                    )
            if len(batch) > 1:
                middle = len(batch) // 2
                run_batch(batch[:middle])
                run_batch(batch[middle:])

        missing = progress()
        for start in range(0, len(missing), batch_size):
            run_batch(missing[start : start + batch_size])
            logging.getLogger(__name__).warning(
                "R2 labels committed: %s/%s", len(unique) - len(progress()), len(unique)
            )
        if progress():
            raise ValueError(
                f"Classification incomplete; see {directory / 'progress.json'}; resubmit to retry missing IDs"
            )
    return {text: records[text] for text in unique}


def review_rows(catalog, records):
    rows = {row["description"]: row for row in review_sample(catalog, records)}
    for text in set(catalog.get("human", {}).values()):
        row = rows.setdefault(
            text,
            {
                "description": text,
                "predicted_task_id": records[text]["task_id"],
                "task_id": "",
                "reviewer": "",
                "strata": [],
            },
        )
        row["strata"].append("all_human_features")
    return [rows[text] for text in sorted(rows)]


def classify_catalog(
    output, taxonomy, classifier_config, endpoint, batch_size=8, conditions=None
):
    smoke = output / "classification/smoke/smoke_result.json"
    smoke_result = json.loads(smoke.read_text()) if smoke.exists() else {}
    if (
        not smoke_result.get("passed")
        or smoke_result.get("classifier_sha256") != digest(classifier_config)
        or smoke_result.get("prompt_sha256") != digest(instruction())
    ):
        raise ValueError(
            "Run a passing R2 classify-smoke with this model/protocol before production"
        )
    all_catalog = json.loads((output / "inventory/catalog.json").read_text())
    registry = feature_registry(all_catalog)
    catalog = {c: all_catalog[c] for c in conditions} if conditions else all_catalog
    directory = output / "classification"
    protocol = {
        "taxonomy_sha256": taxonomy_hash(taxonomy),
        "classifier": classifier_config,
        "registry_sha256": digest(registry),
        "initial_instruction": instruction(),
        "adjudication_instruction": instruction(True),
        "review_policy": "all-human-plus-stratified-and-uncertain-v2",
    }
    for name, value in [
        ("protocol.json", protocol),
        ("feature_registry.json", registry),
    ]:
        path = directory / name
        if path.exists() and json.loads(path.read_text()) != value:
            raise ValueError(
                "Classification protocol changed; use a new experiment namespace"
            )
        atomic_json(path, value)
    atomic_json(
        directory / "completion.json",
        {
            "complete": False,
            "requested_conditions": list(catalog),
            "training_authorized": False,
        },
    )
    texts = sorted({text for labels in catalog.values() for text in labels.values()})
    initial = classify_ids(
        texts,
        registry,
        taxonomy,
        classifier_config,
        directory / "initial",
        qwen_client(endpoint, classifier_config, directory / "requests/initial"),
        batch_size,
    )
    ambiguous = [text for text, row in initial.items() if row["uncertain"]]
    adjudicated = classify_ids(
        ambiguous,
        registry,
        taxonomy,
        classifier_config,
        directory / "adjudication",
        qwen_client(endpoint, classifier_config, directory / "requests/adjudication"),
        batch_size,
        adjudication=True,
    )
    records = {}
    for text, first in initial.items():
        row = {
            **adjudicated.get(text, first),
            "initial_task_id": first["task_id"],
            "initial_uncertain": first["uncertain"],
            "initial_explanation": first["explanation"],
            "initial_out_of_inventory": first["out_of_inventory"],
            "adjudicated": text in adjudicated,
        }
        if first["out_of_inventory"]:
            row.update(uncertain=True, out_of_inventory=True)
        records[text] = row
    path = directory / "classified_descriptions.json"
    previous = json.loads(path.read_text()) if path.exists() else {}
    if any(text in previous and previous[text] != row for text, row in records.items()):
        raise ValueError("Previously classified description changed")
    atomic_json(path, {**previous, **records})
    review = directory / "human_review.csv"
    existing = set()
    if review.exists():
        with review.open(newline="") as stream:
            existing = {r["description"] for r in csv.DictReader(stream)}
    with review.open("a", newline="") as stream:
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
        for row in review_rows(catalog, records):
            if row["description"] not in existing:
                writer.writerow({**row, "strata": json.dumps(row["strata"])})
    result = {
        "complete": True,
        "classification_complete": True,
        "training_authorized": False,
        "requested_conditions": list(catalog),
        "unique_descriptions": len(records),
        "adjudicated": len(ambiguous),
        "unresolved": sum(r["uncertain"] for r in records.values()),
        "inventory_gaps": sum(r["out_of_inventory"] for r in records.values()),
        "human_review": str(review),
    }
    atomic_json(directory / "completion.json", result)
    return result


def classification_smoke(output, taxonomy, classifier_config, endpoint):
    texts = [*SMOKE_ANCHORS, "cooked in a pot", "has large theeth"]
    directory = output / "classification/smoke"
    registry = feature_registry({"smoke": dict(enumerate(texts))})
    rows = classify_ids(
        texts,
        registry,
        taxonomy,
        classifier_config,
        directory,
        qwen_client(endpoint, classifier_config, directory / "requests"),
        8,
    )
    failures = [
        text
        for text, task in SMOKE_ANCHORS.items()
        if rows[text]["task_id"] != task
        or rows[text]["uncertain"]
        or rows[text]["out_of_inventory"]
    ]
    if not rows["cooked in a pot"]["out_of_inventory"]:
        failures.append("cooked in a pot")
    result = {
        "passed": not failures,
        "smoke_only": True,
        "features": len(rows),
        "production_labels_modified": False,
        "anchor_failures": failures,
        "interpretation": "Source-rule conformance, not independent classification accuracy",
        "classifier_sha256": digest(classifier_config),
        "prompt_sha256": digest(instruction()),
    }
    atomic_json(directory / "smoke_result.json", result)
    return result


def freeze_catalog(output, taxonomy, review_path, conditions=None):
    catalog = json.loads((output / "inventory/catalog.json").read_text())
    if not conditions or "human" in conditions or "random_human" in conditions:
        with Path(review_path).open(newline="") as stream:
            reviewed = {
                r["description"]
                for r in csv.DictReader(stream)
                if r.get("task_id", "").isdigit() and r.get("reviewer", "").strip()
            }
        if not set(catalog["human"].values()).issubset(reviewed):
            raise ValueError("R2 requires review of every human feature before freeze")
    return legacy_freeze(output, taxonomy, review_path, conditions)
