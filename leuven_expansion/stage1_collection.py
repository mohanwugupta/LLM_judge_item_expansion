"""Isolated full-domain V2/V3 collection. No model training or behavioral tests."""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
import csv
from functools import lru_cache
import hashlib
import io
import json
import math
from pathlib import Path
import sqlite3
import tarfile
import uuid

from leuven_expansion.feature_prompts import (
    load_generation_prompts,
    load_prompts_by_version,
)
from leuven_expansion.generate_features import (
    build_generation_jobs,
    normalize_generated_feature,
    validate_generation_output,
    _RESPONSE_FORMAT,
)
from leuven_expansion.normalize import normalize_word

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "configs/stage1_collection"
CONFIG = ROOT / "configs/stage1_collection.json"


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=True).encode()
    ).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def freeze(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"Changed protocol at {path}; use a new output version")
    else:
        atomic_json(path, value)


def read_csv(path):
    with Path(path).open(newline="") as f:
        if f.readline().startswith("version https://git-lfs.github.com/spec/v1"):
            raise ValueError(f"Materialize Git LFS input: {path}")
        f.seek(0)
        return list(csv.DictReader(f))


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = iter(rows)
    first = next(rows, None)
    fields = fields or (list(first) if first is not None else [])
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        if first is not None:
            writer.writerow(first)
        writer.writerows(rows)
    temporary.replace(path)


@lru_cache(maxsize=1)
def _load_config():
    value = json.loads(CONFIG.read_text())
    if value["stage1_objects"] != 350 or value["v2_original_features"] != 2541:
        raise ValueError("Stage 1 domain changed")
    if value["generation_versions"] != ["v3", "v3.1"] or value["training_authorized"]:
        raise ValueError("Frozen collection scope changed")
    return value


def config():
    return dict(_load_config())


def prepare(output):
    output = Path(output)
    freeze(output / "collection_config.json", config())
    for source, target in [
        ("objects_template.csv", "objects.csv"),
        ("v2_features_template.csv", "v2_features.csv"),
    ]:
        path = output / "inputs" / target
        if not path.exists():
            write_csv(path, read_csv(ASSETS / source))
    freeze(
        output / "source_manifest.json",
        json.loads((ASSETS / "source_manifest.json").read_text()),
    )
    return output


def load_objects(output, require_reviewed=False):
    rows = read_csv(Path(output) / "inputs/objects.csv")
    original = read_csv(ASSETS / "objects_template.csv")
    if len(rows) != 350:
        raise ValueError("Exactly 350 original objects required")
    for row, base in zip(rows, original):
        for key in (
            "object_id",
            "object_index",
            "original_name",
            "category",
            "size",
            "legacy_current_object",
        ):
            if row.get(key) != base[key]:
                raise ValueError(f"Canonical object metadata changed: {key}")
        if row.get("identity_reviewed") not in {"true", "false"}:
            raise ValueError("identity_reviewed must be true or false")
        if not row.get("display_name", "").strip():
            raise ValueError("Empty display name")
        if (
            row["identity_reviewed"] == "true"
            and not row.get("identity_evidence", "").strip()
        ):
            raise ValueError("Reviewed identities require evidence")
    names = [normalize_word(r["display_name"]) for r in rows]
    if len(set(names)) != 350:
        raise ValueError(
            "Object display names must remain distinct; no concept merging"
        )
    if require_reviewed and any(r["identity_reviewed"] != "true" for r in rows):
        raise ValueError(
            "Review all 26 ambiguous original object identities before production"
        )
    return rows


def load_v2_features(output, require_verified=False):
    rows = read_csv(Path(output) / "inputs/v2_features.csv")
    original = read_csv(ASSETS / "v2_features_template.csv")
    if len(rows) != 2541:
        raise ValueError(
            "Exactly 2541 original V2 feature IDs required; not the 1995 Stage 2 inventory"
        )
    for row, base in zip(rows, original):
        if (
            row.get("feature_id") != base["feature_id"]
            or row.get("original_task_id") != base["original_task_id"]
        ):
            raise ValueError("Original feature IDs/task assignments changed")
        if row.get("verified") not in {"true", "false"}:
            raise ValueError("Feature verified must be true or false")
        verified = row["verified"] == "true"
        if verified and (
            not row.get("feature_text", "").strip()
            or not row.get("evidence", "").strip()
        ):
            raise ValueError("Verified feature mappings require text and evidence")
        if require_verified and not verified:
            raise ValueError(
                "Complete verified original feature-text dictionary required for V2"
            )
    return rows


def collection_taxonomy():
    value = json.loads((ASSETS / "taxonomy_33_collection.json").read_text())
    if [c["id"] for c in value["classes"]] != list(range(33)) or value[
        "historical_numeric_correspondence"
    ] is not False:
        raise ValueError(
            "Invalid collection taxonomy; do not claim original numeric correspondence"
        )
    return value


def classifier_instruction():
    return (ASSETS / "classifier_prompt.txt").read_text()


def preflight(output, task, shard_count=1):
    if shard_count < 1:
        raise ValueError("Positive shard count required")
    output = Path(output)
    if json.loads((output / "collection_config.json").read_text()) != config():
        raise ValueError("Collection configuration changed; use a new output version")
    objects = load_objects(output)
    features = load_v2_features(output)
    unreviewed = sum(r["identity_reviewed"] != "true" for r in objects)
    unverified = sum(r["verified"] != "true" for r in features)
    blockers = []
    if task != "smoke" and unreviewed:
        blockers.append(f"Review {unreviewed} object identities in inputs/objects.csv")
    if task == "judge" and unverified:
        blockers.append(
            f"Verify descriptions/evidence for {unverified} original V2 features in inputs/v2_features.csv"
        )
    if (
        task == "classify"
        and not (output / "finalized/generate/candidate_features.csv").exists()
    ):
        blockers.append(
            "Finalize complete V3/V3.1 generation before classifying candidates"
        )
    elif task == "classify":
        folder = output / "finalized/generate"
        manifest = json.loads((folder / "manifest.json").read_text())
        if not manifest["complete"] or not read_csv(folder / "candidate_features.csv"):
            blockers.append(
                "Generation must be finalized with nonempty candidate features"
            )
        for name, expected in manifest["files"].items():
            if sha(folder / name) != expected:
                raise ValueError("Finalized generation output was changed")
    expected = {
        "generate": 350 * 6 * config()["responses_per_word"],
        "judge": 350 * 2541,
        "smoke": 1,
    }.get(task)
    return dict(
        ready=not blockers,
        task=task,
        blockers=blockers,
        objects=350,
        unreviewed_objects=unreviewed,
        unverified_v2_features=unverified,
        expected_units=expected,
        shard_count=shard_count,
        training_authorized=False,
        automatic_legacy_reuse=False,
        taxonomy_status=config()["task_taxonomy_status"],
    )


def shard(key, count):
    if count < 1:
        raise ValueError("Positive shard count required")
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") % count


@contextmanager
def lock(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    import fcntl

    with (directory / ".lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Another process owns this shard") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def run_units(directory, stage, signature, units, worker, workers=8):
    """Bounded submission; durable per-unit transactions; retry errors on resume."""
    directory = Path(directory)
    if workers < 1:
        raise ValueError("Positive worker count required")
    with lock(directory):
        freeze(directory / "protocol.json", signature)
        if (directory / "status.json").exists():
            old = json.loads((directory / "status.json").read_text())
            if (
                old["complete"]
                and sha(directory / "responses.sqlite3") != old["database_sha256"]
            ):
                raise ValueError("Completed database changed; refusing resume")
        with sqlite3.connect(directory / "responses.sqlite3") as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS units (key TEXT PRIMARY KEY, stage TEXT, payload TEXT, result TEXT, status TEXT, error TEXT)"
            )
            pending = iter(units)
            total = completed = skipped = failures = 0
            atomic_json(
                directory / "status.json",
                dict(stage=stage, complete=False, scan_finished=False),
            )

            def next_unit():
                nonlocal total, completed, skipped
                for unit in pending:
                    total += 1
                    found = db.execute(
                        "SELECT payload,result,status FROM units WHERE key=?",
                        (unit["key"],),
                    ).fetchone()
                    payload = json.dumps(unit, sort_keys=True)
                    if found and found[0] != payload:
                        raise ValueError("Unit payload changed under the same key")
                    if found and found[2] == "done":
                        json.loads(found[1])
                        completed += 1
                        skipped += 1
                        continue
                    return unit
                return None

            with ThreadPoolExecutor(max_workers=workers) as pool:
                running = {}
                while True:
                    while len(running) < workers * 2:
                        unit = next_unit()
                        if unit is None:
                            break
                        running[pool.submit(worker, unit)] = unit
                    if not running:
                        break
                    done, _ = wait(running, return_when=FIRST_COMPLETED)
                    for future in done:
                        unit = running.pop(future)
                        try:
                            result, status, error = future.result(), "done", ""
                            completed += 1
                        except Exception as exc:
                            result, status, error = None, "error", str(exc)
                            failures += 1
                        with db:
                            db.execute(
                                "INSERT OR REPLACE INTO units VALUES (?,?,?,?,?,?)",
                                (
                                    unit["key"],
                                    stage,
                                    json.dumps(unit, sort_keys=True),
                                    json.dumps(result),
                                    status,
                                    error,
                                ),
                            )
                        if (completed + failures) % 100 == 0:
                            progress = dict(
                                stage=stage,
                                complete=False,
                                scan_finished=False,
                                units_seen=total,
                                completed=completed,
                                failures_this_attempt=failures,
                                skipped=skipped,
                            )
                            atomic_json(directory / "status.json", progress)
                            print(json.dumps(progress), flush=True)
            record = dict(
                stage=stage,
                expected=total,
                completed=completed,
                pending=total - completed,
                complete=completed == total,
                skipped=skipped,
                failures_this_attempt=failures,
                protocol_sha256=digest(signature),
                training_authorized=False,
                scan_finished=True,
            )
        record["database_sha256"] = sha(directory / "responses.sqlite3")
        atomic_json(directory / "status.json", record)
        return record


def collection_status(output):
    report = {}
    for task in ("generate", "judge", "classify"):
        counts = {
            "done": 0,
            "error": 0,
            "shards_present": 0,
            "training_authorized": False,
        }
        for path in (Path(output) / "production" / task).glob("*/responses.sqlite3"):
            with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as db:
                for state, n in db.execute(
                    "SELECT status,COUNT(*) FROM units GROUP BY status"
                ):
                    counts[state] = counts.get(state, 0) + n
            counts["shards_present"] += 1
        report[task] = counts
    return report


def generation_units(objects, responses=20, versions=("v3", "v3.1")):
    for obj in objects:
        words = [
            {
                "word_original": obj["display_name"],
                "word_normalized": normalize_word(obj["display_name"]),
            }
        ]
        for version in versions:
            jobs = build_generation_jobs(
                words,
                responses,
                config()["model"],
                config()["base_seed"],
                load_generation_prompts(version),
            )
            for job in jobs:
                yield dict(
                    job,
                    key=f"{obj['object_id']}:{version}:{job['prompt_variant']}:{job['replicate_id']}",
                    object_id=obj["object_id"],
                    version=version,
                )


def generate_one(unit, client):
    raw, meta = client.generate(
        unit["system_prompt"],
        unit["user_message"],
        model=config()["model"],
        temperature=config()["generation_temperature"],
        max_tokens=config()["generation_max_tokens"],
        seed=unit["sampling_seed"],
        response_format=_RESPONSE_FORMAT,
    )
    if meta.get("finish_reason") != "stop":
        raise ValueError("Generation finish reason was not stop")
    parsed, error = validate_generation_output(raw, unit["word_normalized"])
    if error:
        raise ValueError(error)
    return dict(parsed, raw_json=raw, metadata=meta)


def judge_units(objects, features):
    for obj in objects:
        for feature in features:
            yield dict(
                key=f"{obj['object_id']}:{feature['feature_id']}",
                object_id=obj["object_id"],
                word_normalized=normalize_word(obj["display_name"]),
                feature_id=int(feature["feature_id"]),
                feature_text=feature["feature_text"],
            )


def judge_one(unit, client):
    from leuven_expansion.feature_judge import judge_pair
    from leuven_expansion.feature_adjudicate import resolve_first_pass

    kwargs = dict(
        job_id="isc-stage1-v2",
        word_normalized=unit["word_normalized"],
        feature_id=unit["feature_id"],
        feature_text=unit["feature_text"],
        prompts=load_prompts_by_version("v2"),
        client=client,
        model=config()["model"],
        temperature=0,
        max_tokens=config()["judge_max_tokens"],
    )
    votes = judge_pair(word_original=unit["word_normalized"], **kwargs)
    if any(v["parse_error"] for v in votes):
        raise ValueError(
            "Invalid first-pass votes; retry cell, never encode missing as zero"
        )
    resolution, adjudicators = resolve_first_pass(votes=votes, **kwargs)
    if adjudicators and any(v["parse_error"] for v in adjudicators):
        raise ValueError("Incomplete adjudication panel; retry cell")
    value = resolution["final_feature_value"]
    if value is None or not math.isfinite(float(value)) or not 0 <= float(value) <= 4:
        raise ValueError("Unresolved applicability value")
    return dict(
        resolution=resolution,
        first_pass=votes,
        adjudicators=adjudicators,
        binary_value=int(float(value) > 0),
        binary_policy="final_feature_value > 0; no rater-frequency cutoff",
    )


def classification_units(catalog, batch_size=8):
    # One label per exact description; memberships are deliberately not sent.
    unique = {r["feature_id"]: r["description"] for r in catalog}
    rows = [dict(feature_id=k, description=v) for k, v in sorted(unique.items())]
    for start in range(0, len(rows), batch_size):
        features = rows[start : start + batch_size]
        yield dict(key=digest(features), features=features)


def classify_one(unit, client):
    import jsonschema

    taxonomy = collection_taxonomy()
    ids = [r["feature_id"] for r in unit["features"]]
    item = dict(
        type="object",
        additionalProperties=False,
        required=[
            "feature_id",
            "task_id",
            "uncertain",
            "out_of_inventory",
            "explanation",
        ],
        properties=dict(
            feature_id=dict(type="string", enum=ids),
            task_id=dict(type="integer", minimum=0, maximum=32),
            uncertain=dict(type="boolean"),
            out_of_inventory=dict(type="boolean"),
            explanation=dict(type="string", minLength=1),
        ),
    )
    schema = dict(
        type="object",
        additionalProperties=False,
        required=["labels"],
        properties=dict(
            labels=dict(type="array", minItems=len(ids), maxItems=len(ids), items=item)
        ),
    )
    raw, meta = client.generate(
        classifier_instruction(),
        json.dumps(dict(taxonomy=taxonomy["classes"], features=unit["features"])),
        model=config()["model"],
        temperature=0,
        seed=0,
        max_tokens=config()["classification_max_tokens"],
        response_format=dict(
            type="json_schema",
            json_schema=dict(name="stage1_tasks", strict=True, schema=schema),
        ),
    )
    if meta.get("finish_reason") != "stop":
        raise ValueError("Classification finish reason was not stop")
    result = json.loads(raw)
    jsonschema.validate(result, schema)
    if len({r["feature_id"] for r in result["labels"]}) != len(ids):
        raise ValueError("Duplicate or missing classified feature IDs")
    if any(r["out_of_inventory"] and not r["uncertain"] for r in result["labels"]):
        raise ValueError("Out-of-inventory label requires uncertainty")
    return dict(
        result, metadata=meta, raw_json=raw, original_numeric_correspondence=False
    )


class JournalClient:
    """Preserve every raw call, including calls rejected by existing validators."""

    def __init__(self, client, directory):
        self.client, self.directory = client, Path(directory)

    def generate(self, *args, **kwargs):
        record = dict(args=args, kwargs=kwargs)
        try:
            raw, metadata = self.client.generate(*args, **kwargs)
            record.update(raw_response=raw, metadata=metadata)
            if metadata.get("finish_reason") != "stop":
                raise ValueError("LLM finish reason was not stop")
            return raw, metadata
        except Exception as exc:
            record["error"] = str(exc)
            raise
        finally:
            atomic_json(self.directory / (uuid.uuid4().hex + ".json"), record)


def fingerprint(output, model_path):
    # Reuse the existing local-weight fingerprint, not a model-name revision claim.
    import sys

    sys.path.insert(0, str(ROOT / "ISC-CI_LLM_validation"))
    from semantic_retraining.classification_workflow import model_fingerprint

    value = model_fingerprint(model_path)
    freeze(Path(output) / "model_fingerprint.json", value)
    return value


def signature(output, task, shard_count, smoke=False):
    output = Path(output)
    inputs = [output / "inputs/objects.csv", CONFIG, Path(__file__)]
    inputs += sorted(ASSETS.glob("*"))
    inputs += [
        Path(__file__).parent / name
        for name in [
            "feature_prompts.py",
            "generate_features.py",
            "feature_judge.py",
            "feature_adjudicate.py",
            "feature_schema.py",
        ]
    ]
    inputs += [ROOT / "vllm_client.py", Path(__file__).parent / "normalize.py"]
    inputs += sorted((Path(__file__).parent / "schemas").glob("*.json"))
    inputs += sorted((Path(__file__).parent / "prompts").glob("*v[23]*.txt"))
    if task == "judge":
        inputs.append(output / "inputs/v2_features.csv")
    if task == "classify":
        inputs.append(output / "finalized/generate/candidate_features.csv")
        inputs.append(output / "finalized/generate/manifest.json")
    return dict(
        protocol=config()["protocol"],
        task=task,
        shard_count=shard_count,
        smoke=smoke,
        inputs={
            (
                str(p.relative_to(output))
                if p.is_relative_to(output)
                else str(p.relative_to(ROOT))
            ): sha(p)
            for p in inputs
            if p.is_file()
        },
        model=json.loads((output / "model_fingerprint.json").read_text()),
        training_authorized=False,
    )


def execute(output, task, shard_index=0, shard_count=1, client=None, workers=8):
    output = Path(output)
    if not 0 <= shard_index < shard_count:
        raise ValueError("shard-index must be in [0, shard-count)")
    plan = preflight(output, task, shard_count)
    if not plan["ready"]:
        raise ValueError("; ".join(plan["blockers"]))
    objects = load_objects(output, require_reviewed=task != "smoke")
    if task == "smoke":
        objects = [
            next(o for o in objects if o["original_name"] == n)
            for n in ["bee", "monkey", "apple"]
        ]
        directory = output / "smoke"
        stages = [
            ("generate", generation_units(objects, responses=1), generate_one),
            (
                "judge",
                judge_units(
                    objects,
                    [
                        dict(feature_id=0, feature_text="is an animal"),
                        dict(feature_id=1, feature_text="has wings"),
                    ],
                ),
                judge_one,
            ),
            (
                "classify",
                classification_units(
                    [
                        dict(feature_id="anchor-wings", description="has wings"),
                        dict(feature_id="anchor-pot", description="is cooked in a pot"),
                    ]
                ),
                classify_one,
            ),
        ]
    else:
        directory = output / "production" / task / f"{shard_index:04d}"
        if task == "generate":
            units, worker = (
                generation_units(objects, responses=config()["responses_per_word"]),
                generate_one,
            )
        elif task == "judge":
            units, worker = (
                judge_units(objects, load_v2_features(output, True)),
                judge_one,
            )
        elif task == "classify":
            units, worker = (
                classification_units(
                    read_csv(output / "finalized/generate/candidate_features.csv"),
                    config()["classification_batch_size"],
                ),
                classify_one,
            )
        else:
            raise ValueError("Unknown collection task")
        stages = [
            (
                task,
                (u for u in units if shard(u["key"], shard_count) == shard_index),
                worker,
            )
        ]
    record = signature(output, task, shard_count, smoke=task == "smoke")
    if task != "smoke":
        freeze(output / "protocols" / (task + ".json"), record)
    if client is None:
        raise ValueError("A client is required after preflight")
    client = JournalClient(client, directory / "requests")
    results = []
    for name, units, worker in stages:
        results.append(
            run_units(
                directory / name if task == "smoke" else directory,
                name,
                record,
                units,
                lambda u, w=worker: w(u, client),
                workers,
            )
        )
    return dict(
        passed=all(r["complete"] for r in results),
        stages=results,
        smoke_only=task == "smoke",
        training_authorized=False,
    )


def finalized_rows(output, task):
    output = Path(output)
    protocol_path = output / "protocols" / (task + ".json")
    record = json.loads(protocol_path.read_text())
    if record != signature(output, task, record["shard_count"]):
        raise ValueError("Finalization inputs/model/protocol changed")
    for index in range(record["shard_count"]):
        folder = output / "production" / task / f"{index:04d}"
        if json.loads((folder / "protocol.json").read_text()) != record:
            raise ValueError("Shard protocol mismatch")
        status = json.loads((folder / "status.json").read_text())
        if not status["complete"]:
            raise ValueError(f"Incomplete {task} shard {index}; resubmit it")
        if sha(folder / "responses.sqlite3") != status["database_sha256"]:
            raise ValueError("Shard database changed after completion")
        db = sqlite3.connect(f"file:{folder / 'responses.sqlite3'}?mode=ro", uri=True)
        try:
            for key, payload, result, state in db.execute(
                "SELECT key,payload,result,status FROM units ORDER BY key"
            ):
                if state != "done" or shard(key, record["shard_count"]) != index:
                    raise ValueError("Incomplete/misassigned unit in shard")
                yield json.loads(payload), json.loads(result)
        finally:
            db.close()


def finalize(output, task):
    output = Path(output)
    folder = output / "finalized" / task
    record = json.loads((output / "protocols" / (task + ".json")).read_text())
    if record != signature(output, task, record["shard_count"]):
        raise ValueError("Finalization protocol/input hashes changed")
    if (folder / "manifest.json").exists():
        manifest = json.loads((folder / "manifest.json").read_text())
        for name, expected in manifest["files"].items():
            if sha(folder / name) != expected:
                raise ValueError("Finalized output hash mismatch")
        return manifest
    rows = finalized_rows(output, task)
    seen = set()
    records = []
    candidates, memberships = {}, []
    if task == "generate":
        for unit, result in rows:
            if unit["key"] in seen:
                raise ValueError("Duplicate generation unit")
            seen.add(unit["key"])
            records.append(
                dict(
                    unit_id=unit["key"],
                    object_id=unit["object_id"],
                    display_name=unit["word_normalized"],
                    version=unit["version"],
                    prompt_variant=unit["prompt_variant"],
                    replicate_id=unit["replicate_id"],
                    sampling_seed=unit["sampling_seed"],
                    features_json=json.dumps(result["features"]),
                )
            )
            for text in result["features"]:
                # Exact transparent phrase normalization only. No semantic merging or retention cutoff.
                normalized = normalize_generated_feature(text)
                key = "F" + hashlib.sha256(normalized.encode()).hexdigest()
                if key not in candidates or text < candidates[key]["description"]:
                    candidates[key] = dict(
                        feature_id=key, description=text, normalized_text=normalized
                    )
                memberships.append(
                    dict(
                        unit_id=unit["key"],
                        object_id=unit["object_id"],
                        version=unit["version"],
                        prompt_variant=unit["prompt_variant"],
                        feature_id=key,
                        raw_feature_text=text,
                    )
                )
        if len(seen) != 350 * 6 * config()["responses_per_word"]:
            raise ValueError(
                "Generation does not cover all 350 objects, both versions and A/B/C"
            )
        write_csv(folder / "feature_generations.csv", records)
        write_csv(
            folder / "candidate_features.csv",
            [candidates[k] for k in sorted(candidates)],
        )
        write_csv(folder / "generated_features_long.csv", memberships)
    elif task == "judge":

        def judgments():
            for unit, result in rows:
                if unit["key"] in seen:
                    raise ValueError("Duplicate V2 cell")
                seen.add(unit["key"])
                resolution = result["resolution"]
                yield dict(
                    object_id=unit["object_id"],
                    feature_id=unit["feature_id"],
                    feature_text=unit["feature_text"],
                    final_feature_value=resolution["final_feature_value"],
                    binary_value=result["binary_value"],
                    needs_human_audit=resolution["needs_human_audit"],
                    resolution_method=resolution["resolution_method"],
                )

        write_csv(folder / "feature_resolutions.csv", judgments())
        if len(seen) != 350 * 2541:
            raise ValueError("V2 does not cover original 350 x 2541 domain")
    elif task == "classify":
        for unit, result in rows:
            for row in result["labels"]:
                if row["feature_id"] in seen:
                    raise ValueError("Duplicate task label")
                seen.add(row["feature_id"])
                records.append(dict(row, original_numeric_correspondence=False))
        if seen != {
            r["feature_id"]
            for r in read_csv(output / "finalized/generate/candidate_features.csv")
        }:
            raise ValueError("Missing candidate task labels")
        write_csv(folder / "task_labels.csv", records)
    else:
        raise ValueError("Unknown finalization task")
    manifest = dict(
        complete=True,
        task=task,
        completed_units=len(seen),
        protocol_sha256=digest(record),
        training_authorized=False,
        original_numeric_task_correspondence=task == "judge",
        consolidation="Exact phrase candidates only; final semantic consolidation/review remains separate",
        files={p.name: sha(p) for p in folder.glob("*.csv")},
    )
    atomic_json(folder / "manifest.json", manifest)
    return manifest


def export_bundle(output, destination):
    output, destination = Path(output).resolve(), Path(destination).resolve()
    if destination.is_relative_to(output):
        raise ValueError("Export destination must be outside the collection directory")
    if destination.exists():
        raise FileExistsError("Export exists; choose a new filename")
    files = sorted(p for p in output.rglob("*") if p.is_file() or p.is_symlink())
    if any(p.is_symlink() for p in files):
        raise ValueError("Refusing symlink in scientific export")
    hashes = {str(p.relative_to(output)): sha(p) for p in files if p.name != ".lock"}
    report = dict(
        files=hashes,
        collection_complete={
            t: (output / "finalized" / t / "manifest.json").exists()
            for t in ("generate", "judge", "classify")
        },
        training_authorized=False,
    )
    with tarfile.open(destination, "w:gz") as archive:
        for name in hashes:
            archive.add(output / name, arcname=name, recursive=False)
        payload = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
        info = tarfile.TarInfo("EXPORT_MANIFEST.json")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    for name, expected in hashes.items():
        if sha(output / name) != expected:
            raise ValueError("Export input changed; stop all jobs before exporting")
    return dict(path=str(destination), sha256=sha(destination), **report)


def verify_bundle(path):
    with tarfile.open(path) as archive:
        members = archive.getmembers()
        names = [m.name for m in members]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate archive member")
        if any(
            not m.isfile() or Path(m.name).is_absolute() or ".." in Path(m.name).parts
            for m in members
        ):
            raise ValueError("Unsafe archive member")
        manifest = json.load(archive.extractfile("EXPORT_MANIFEST.json"))
        if set(names) != set(manifest["files"]) | {"EXPORT_MANIFEST.json"}:
            raise ValueError("Export member set mismatch")
        for name, expected in manifest["files"].items():
            h = hashlib.sha256()
            with archive.extractfile(name) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    h.update(block)
            if h.hexdigest() != expected:
                raise ValueError("Export checksum mismatch: " + name)
    return dict(
        passed=True,
        files_verified=len(manifest["files"]),
        collection_complete=manifest["collection_complete"],
        training_authorized=False,
    )


def require_smoke(output):
    output = Path(output)
    smoke = json.loads((output / "smoke_result.json").read_text())
    if not smoke["passed"] or smoke["model"] != json.loads(
        (output / "model_fingerprint.json").read_text()
    ):
        raise ValueError("Pass the isolated smoke test with the same model first")
    if smoke.get("validation_sources") != smoke_sources(output):
        raise ValueError(
            "Collection code/prompts changed since smoke; use a new collection version"
        )


def smoke_sources(output):
    values = signature(output, "smoke", 1, smoke=True)["inputs"]
    return {k: v for k, v in values.items() if k != "inputs/objects.csv"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        required=True,
        choices=[
            "prepare",
            "preflight",
            "fingerprint",
            "smoke",
            "generate",
            "judge",
            "classify",
            "finalize",
            "export",
            "verify-export",
            "status",
        ],
    )
    parser.add_argument(
        "--task", choices=["smoke", "generate", "judge", "classify"], default="generate"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / config()["output_directory"]
    )
    parser.add_argument("--endpoint", default="http://localhost:8025/v1")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--require-smoke", action="store_true")
    args = parser.parse_args(argv)
    if args.output_dir.resolve() == ROOT or not args.output_dir.name.startswith(
        "isc_stage1_collection_"
    ):
        parser.error("Use a dedicated isc_stage1_collection_<version> directory")
    try:
        if not 0 <= args.shard_index < args.shard_count:
            raise ValueError("shard-index must be in [0, shard-count)")
        if args.stage == "prepare":
            prepare(args.output_dir)
            result = preflight(args.output_dir, args.task, args.shard_count)
        elif args.stage == "preflight":
            result = preflight(args.output_dir, args.task, args.shard_count)
            if args.require_smoke:
                require_smoke(args.output_dir)
        elif args.stage == "status":
            result = collection_status(args.output_dir)
        elif args.stage == "fingerprint":
            if args.model_path is None:
                raise ValueError("--model-path required")
            result = fingerprint(args.output_dir, args.model_path)
        elif args.stage == "finalize":
            result = finalize(args.output_dir, args.task)
        elif args.stage == "export":
            if args.archive is None:
                raise ValueError("--archive required")
            result = export_bundle(args.output_dir, args.archive)
        elif args.stage == "verify-export":
            if args.archive is None:
                raise ValueError("--archive required")
            result = verify_bundle(args.archive)
        else:
            if not 0 <= args.shard_index < args.shard_count:
                raise ValueError("shard-index must be in [0, shard-count)")
            plan = preflight(args.output_dir, args.stage, args.shard_count)
            if not plan["ready"]:
                raise ValueError("; ".join(plan["blockers"]))
            if args.stage != "smoke":
                require_smoke(args.output_dir)
            from vllm_client import VLLMClient

            client = VLLMClient(
                model_name=config()["model"], base_url=args.endpoint, enable_cache=False
            )
            result = execute(
                args.output_dir,
                args.stage,
                args.shard_index,
                args.shard_count,
                client,
                args.workers,
            )
            if args.stage == "smoke":
                atomic_json(
                    args.output_dir / "smoke_result.json",
                    dict(
                        result,
                        model=json.loads(
                            (args.output_dir / "model_fingerprint.json").read_text()
                        ),
                        validation_sources=smoke_sources(args.output_dir),
                    ),
                )
        print(json.dumps(result, indent=2, sort_keys=True))
        return (
            0
            if result.get("ready", result.get("passed", True))
            or args.stage == "prepare"
            else 2
        )
    except (ValueError, OSError, KeyError, sqlite3.Error) as exc:
        print(
            json.dumps(
                dict(error=str(exc), stage=args.stage, training_authorized=False)
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
