"""Run a versioned Leuven-style free feature-generation experiment.

The experimental unit is one stimulus word by one simulated participant
response. No existing Leuven feature is shown to the model.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import importlib
import json
import logging
import pathlib
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional

import jsonschema
import pandas as pd

from leuven_expansion.feature_prompts import (
    GENERATION_PROMPT_VARIANTS,
    build_generation_user_message,
    load_generation_prompts,
    prompt_hash,
)
from leuven_expansion.normalize import normalize_word


logger = logging.getLogger(__name__)

_SCHEMA_PATH = (
    pathlib.Path(__file__).parent
    / "schemas"
    / "feature_generation_response_v3.json"
)
_GENERATION_SCHEMA = json.loads(_SCHEMA_PATH.read_text())
_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "feature_generation_response_v3",
        "strict": True,
        "schema": _GENERATION_SCHEMA,
    },
}

GENERATION_COLUMNS = [
    "job_id",
    "response_id",
    "word_original",
    "word_normalized",
    "prompt_variant",
    "replicate_id",
    "model",
    "temperature",
    "sampling_seed",
    "features_json",
    "n_features",
    "raw_json",
    "parse_error",
    "prompt_hash",
    "finish_reason",
    "prompt_tokens",
    "completion_tokens",
]

LONG_COLUMNS = [
    "job_id",
    "response_id",
    "word_original",
    "word_normalized",
    "prompt_variant",
    "replicate_id",
    "model",
    "sampling_seed",
    "feature_rank",
    "feature_text",
    "feature_text_normalized",
]

FREQUENCY_COLUMNS = [
    "prompt_variant",
    "word_normalized",
    "feature_text_normalized",
    "feature_text_example",
    "response_frequency",
    "valid_response_count",
    "response_proportion",
]


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _normalize_json_literals(text: str) -> str:
    text = re.sub(r'(?<!["\w])None(?!["\w])', "null", text)
    text = re.sub(r'(?<!["\w])True(?!["\w])', "true", text)
    return re.sub(r'(?<!["\w])False(?!["\w])', "false", text)


def _normalize_target_word_for_match(word: str) -> str:
    """Normalize harmless parenthesis formatting without dropping its content."""
    normalized = normalize_word(word)
    return " ".join(re.sub(r"[()]", " ", normalized).split())


def validate_generation_output(
    raw: str,
    expected_word: Optional[str] = None,
) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """Parse and validate one free-generation JSON response."""
    text = raw.strip()
    if text.startswith("```"):
        text = "\n".join(
            line for line in text.splitlines() if not line.strip().startswith("```")
        ).strip()
    text = _normalize_json_literals(text)
    try:
        record = json.loads(text)
    except json.JSONDecodeError as error:
        return None, f"JSON parse error: {error}"
    try:
        jsonschema.validate(record, _GENERATION_SCHEMA)
    except jsonschema.ValidationError as error:
        return None, f"Schema validation error: {error.message}"

    if expected_word is not None:
        returned = _normalize_target_word_for_match(record["target_word"])
        expected = _normalize_target_word_for_match(expected_word)
        if returned != expected:
            return None, (
                f"Returned target_word {record['target_word']!r} does not match "
                f"expected {expected_word!r}"
            )

    features = []
    seen = set()
    for feature in record["features"]:
        cleaned = " ".join(feature.strip().split())
        key = cleaned.casefold()
        if cleaned and key not in seen:
            seen.add(key)
            features.append(cleaned)
    record["features"] = features
    return record, None


def normalize_generated_feature(feature: str) -> str:
    """Apply only transparent normalization; semantic merging is downstream."""
    return " ".join(feature.casefold().strip().rstrip(".").split())


def load_stimulus_words(
    input_csv: str | pathlib.Path,
    item_column: Optional[str] = None,
) -> list[dict[str, str]]:
    """Read only the item column from a Leuven matrix or item CSV."""
    path = pathlib.Path(input_csv)
    first_line = path.read_text(errors="replace").splitlines()[0]
    if first_line.startswith("version https://git-lfs.github.com/spec/v1"):
        raise ValueError(
            f"{path} is a Git LFS pointer, not the data file; run git lfs pull on the cluster"
        )
    columns = pd.read_csv(path, nrows=0).columns.tolist()
    selected = item_column or columns[0]
    if selected not in columns:
        raise ValueError(f"Item column {selected!r} not found in {path}")
    values = pd.read_csv(path, usecols=[selected])[selected]

    rows = []
    seen = set()
    for value in values.dropna():
        original = str(value).strip()
        normalized = normalize_word(original)
        if normalized and normalized not in seen:
            seen.add(normalized)
            rows.append(
                {"word_original": original, "word_normalized": normalized}
            )
    if not rows:
        raise ValueError(f"No stimulus words found in {path}/{selected}")
    return rows


def select_stimulus_words(
    words: list[dict[str, str]],
    max_words: Optional[int] = None,
) -> list[dict[str, str]]:
    """Select an evenly spaced deterministic subset for smoke testing."""
    if max_words is None or max_words >= len(words):
        return words
    if max_words < 1:
        raise ValueError("max_words must be at least 1")
    if max_words == 1:
        return [words[0]]

    last_index = len(words) - 1
    indices = [
        round(position * last_index / (max_words - 1))
        for position in range(max_words)
    ]
    return [words[index] for index in indices]


def _sampling_seed(base_seed: int, word_normalized: str, replicate_id: int) -> int:
    digest = hashlib.sha256(
        f"{base_seed}|{word_normalized}|{replicate_id}".encode()
    ).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFFFFFFFFFF


def _response_id(
    word_normalized: str,
    prompt_variant: str,
    replicate_id: int,
    model: str,
    sampling_seed: int,
    response_prompt_hash: str,
) -> str:
    content = (
        f"{word_normalized}|{prompt_variant}|{replicate_id}|{model}|{sampling_seed}|"
        f"{response_prompt_hash}"
    )
    return hashlib.sha256(content.encode()).hexdigest()


def build_generation_jobs(
    words: list[dict[str, str]],
    responses_per_word: int,
    model: str,
    base_seed: int,
    system_prompts: dict[str, str],
) -> list[dict[str, Any]]:
    if responses_per_word < 1:
        raise ValueError("responses_per_word must be at least 1")
    jobs = []
    for word in words:
        user_message = build_generation_user_message(word["word_normalized"])
        for prompt_variant in GENERATION_PROMPT_VARIANTS:
            system_prompt = system_prompts[prompt_variant]
            response_prompt_hash = prompt_hash(system_prompt, user_message)
            for replicate_id in range(responses_per_word):
                seed = _sampling_seed(
                    base_seed, word["word_normalized"], replicate_id
                )
                jobs.append(
                    {
                        **word,
                        "prompt_variant": prompt_variant,
                        "replicate_id": replicate_id,
                        "sampling_seed": seed,
                        "system_prompt": system_prompt,
                        "user_message": user_message,
                        "prompt_hash": response_prompt_hash,
                        "response_id": _response_id(
                            word["word_normalized"],
                            prompt_variant,
                            replicate_id,
                            model,
                            seed,
                            response_prompt_hash,
                        ),
                    }
                )
    return jobs


def _load_completed_responses(path: pathlib.Path) -> set[str]:
    if not path.exists():
        return set()
    frame = pd.read_csv(path, dtype=str).fillna("")
    valid = frame[frame["parse_error"] == ""]
    return set(valid["response_id"])


def _manifest_config(
    *,
    job_id: str,
    model: str,
    prompt_version: str,
    prompts: dict[str, str],
    input_csv: pathlib.Path,
    item_column: Optional[str],
    responses_per_word: int,
    temperature: float,
    max_tokens: int,
    base_seed: int,
    source_word_count: int,
    word_count: int,
    max_words: Optional[int],
    selected_words: list[dict[str, str]],
) -> dict[str, Any]:
    protocol_versions = {
        "v3": "leuven_free_generation_v3_three_prompt_comparison",
        "v3.1": "leuven_free_generation_v3_1_three_prompt_comparison",
    }
    manifest = {
        "protocol_version": protocol_versions[prompt_version],
        "experimental_unit": (
            "one stimulus word x one prompt condition x one simulated participant"
        ),
        "job_id": job_id,
        "model": model,
        "prompt_variants": list(GENERATION_PROMPT_VARIANTS),
        "prompt_sha256_by_variant": {
            variant: hashlib.sha256(prompts[variant].encode()).hexdigest()
            for variant in GENERATION_PROMPT_VARIANTS
        },
        "prompt_text_by_variant": {
            variant: prompts[variant]
            for variant in GENERATION_PROMPT_VARIANTS
        },
        "response_schema_sha256": hashlib.sha256(
            _SCHEMA_PATH.read_bytes()
        ).hexdigest(),
        "input_csv": str(input_csv.resolve()),
        "input_sha256": hashlib.sha256(input_csv.read_bytes()).hexdigest(),
        "item_column": item_column,
        "source_word_count": source_word_count,
        "word_count": word_count,
        "max_words": max_words,
        "selected_words": [
            word["word_normalized"] for word in selected_words
        ],
        "selected_words_sha256": hashlib.sha256(
            json.dumps(
                [word["word_normalized"] for word in selected_words],
                ensure_ascii=True,
            ).encode()
        ).hexdigest(),
        "responses_per_word_per_prompt": responses_per_word,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "base_seed": base_seed,
        "existing_feature_schema_shown_to_model": False,
        "semantic_feature_merging_performed": False,
        "sampling_design": "paired seeds across prompt variants",
    }
    # Preserve byte-for-byte V3 resume compatibility with manifests created before
    # prompt versions were selectable.
    if prompt_version != "v3":
        manifest["prompt_version"] = prompt_version
    return manifest


def _validate_resume_manifest(
    manifest_path: pathlib.Path, config: dict[str, Any]
) -> None:
    if not manifest_path.exists():
        return
    existing = json.loads(manifest_path.read_text())
    mismatches = {
        key: (existing.get(key), value)
        for key, value in config.items()
        if existing.get(key) != value
    }
    if mismatches:
        raise ValueError(
            "Cannot resume with a changed v3 protocol: "
            + json.dumps(mismatches, sort_keys=True)
        )


def _write_csv_header(path: pathlib.Path, columns: list[str]) -> None:
    if not path.exists():
        with path.open("w", newline="") as handle:
            csv.DictWriter(handle, fieldnames=columns).writeheader()


def _write_derived_outputs(
    generations_csv: pathlib.Path,
    long_csv: pathlib.Path,
    frequency_csv: pathlib.Path,
) -> tuple[int, int]:
    generations = pd.read_csv(generations_csv, dtype=str).fillna("")
    valid = generations[generations["parse_error"] == ""].copy()
    valid = valid.drop_duplicates("response_id", keep="last")

    long_rows = []
    for _, row in valid.iterrows():
        for rank, feature in enumerate(json.loads(row["features_json"]), start=1):
            long_rows.append(
                {
                    key: row[key]
                    for key in [
                        "job_id",
                        "response_id",
                        "word_original",
                        "word_normalized",
                        "prompt_variant",
                        "replicate_id",
                        "model",
                        "sampling_seed",
                    ]
                }
                | {
                    "feature_rank": rank,
                    "feature_text": feature,
                    "feature_text_normalized": normalize_generated_feature(feature),
                }
            )
    long_frame = pd.DataFrame(long_rows, columns=LONG_COLUMNS)
    long_frame.to_csv(long_csv, index=False)

    frequency_rows = []
    valid_counts = valid.groupby(
        ["prompt_variant", "word_normalized"]
    )["response_id"].nunique()
    if not long_frame.empty:
        unique_mentions = long_frame.drop_duplicates(
            ["response_id", "feature_text_normalized"]
        )
        grouped = unique_mentions.groupby(
            ["prompt_variant", "word_normalized", "feature_text_normalized"],
            sort=True,
        )
        for (prompt_variant, word, normalized), rows in grouped:
            count = rows["response_id"].nunique()
            denominator = int(valid_counts[(prompt_variant, word)])
            frequency_rows.append(
                {
                    "prompt_variant": prompt_variant,
                    "word_normalized": word,
                    "feature_text_normalized": normalized,
                    "feature_text_example": rows.iloc[0]["feature_text"],
                    "response_frequency": count,
                    "valid_response_count": denominator,
                    "response_proportion": count / denominator,
                }
            )
    pd.DataFrame(frequency_rows, columns=FREQUENCY_COLUMNS).to_csv(
        frequency_csv, index=False
    )
    return len(valid), len(long_frame)


def revalidate_generation_outputs(
    output_dir: str | pathlib.Path,
) -> dict[str, Any]:
    """Revalidate preserved raw responses and rebuild outputs without model calls."""
    output_path = pathlib.Path(output_dir)
    generations_csv = output_path / "feature_generations.csv"
    manifest_path = output_path / "manifest.json"
    long_csv = output_path / "generated_features_long.csv"
    frequency_csv = output_path / "generated_feature_frequencies.csv"

    for required in [generations_csv, manifest_path]:
        if not required.exists():
            raise FileNotFoundError(f"Required generation artifact not found: {required}")

    generations = pd.read_csv(generations_csv, dtype=str).fillna("")
    latest = generations.drop_duplicates("response_id", keep="last")
    candidates = latest[latest["parse_error"] != ""]
    repaired_rows: list[dict[str, Any]] = []

    for _, row in candidates.iterrows():
        record, parse_error = validate_generation_output(
            row["raw_json"], expected_word=row["word_normalized"]
        )
        if record is None:
            continue
        repaired = {
            column: row.get(column, "") for column in GENERATION_COLUMNS
        }
        repaired["features_json"] = json.dumps(
            record["features"], ensure_ascii=True
        )
        repaired["n_features"] = len(record["features"])
        repaired["parse_error"] = parse_error or ""
        repaired_rows.append(repaired)

    if repaired_rows:
        with generations_csv.open("a", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=GENERATION_COLUMNS,
                extrasaction="ignore",
            )
            writer.writerows(repaired_rows)

    valid_total, feature_total = _write_derived_outputs(
        generations_csv, long_csv, frequency_csv
    )
    all_rows = pd.read_csv(generations_csv, dtype=str).fillna("")
    latest = all_rows.drop_duplicates("response_id", keep="last")
    valid_latest = latest[latest["parse_error"] == ""]
    unresolved_total = int((latest["parse_error"] != "").sum())
    valid_by_prompt = {
        variant: int((valid_latest["prompt_variant"] == variant).sum())
        for variant in GENERATION_PROMPT_VARIANTS
    }

    manifest = json.loads(manifest_path.read_text())
    revalidated_at = _utc_now()
    backup_path = output_path / "manifest.pre_revalidation.json"
    if repaired_rows and not backup_path.exists():
        backup_path.write_text(manifest_path.read_text())
    history = manifest.setdefault("revalidation_history", [])
    history.append(
        {
            "revalidated_at": revalidated_at,
            "candidate_errors": len(candidates),
            "repaired_responses": len(repaired_rows),
            "unresolved_responses": unresolved_total,
            "method": "reparsed preserved raw_json; no model calls",
        }
    )
    manifest.setdefault("generation_finished_at", manifest.get("finished_at"))
    manifest |= {
        "valid_responses_total": valid_total,
        "valid_responses_by_prompt": valid_by_prompt,
        "parse_errors_total": unresolved_total,
        "generated_feature_tokens_total": feature_total,
        "pending_after_run": int(manifest["total_planned_responses"]) - valid_total,
        "revalidated_at": revalidated_at,
        "revalidated_responses_total": sum(
            int(entry.get("repaired_responses", 0)) for entry in history
        ),
        "finished_at": revalidated_at,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    return {
        "candidate_errors": len(candidates),
        "repaired_responses": len(repaired_rows),
        "unresolved_responses": unresolved_total,
        "valid_responses_total": valid_total,
        "valid_responses_by_prompt": valid_by_prompt,
        "pending_after_run": manifest["pending_after_run"],
        "model_calls": 0,
    }


def preflight_feature_generation(
    *,
    job_id: str,
    input_csv: str | pathlib.Path,
    output_dir: str | pathlib.Path,
    model: str,
    prompt_version: str = "v3",
    item_column: Optional[str] = None,
    responses_per_word: int = 20,
    temperature: float = 0.8,
    max_tokens: int = 500,
    base_seed: int = 20260801,
    max_words: Optional[int] = None,
    resume: bool = True,
) -> dict[str, Any]:
    """Validate and summarize a generation plan without files or model calls."""
    if responses_per_word > 1 and temperature <= 0:
        raise ValueError(
            "Multiple simulated participants require temperature > 0 to avoid "
            "deterministic duplicate responses"
        )
    input_path = pathlib.Path(input_csv)
    columns = pd.read_csv(input_path, nrows=0).columns.tolist()
    resolved_item_column = item_column or columns[0]
    source_words = load_stimulus_words(input_path, resolved_item_column)
    words = select_stimulus_words(source_words, max_words)
    system_prompts = load_generation_prompts(prompt_version)
    config = _manifest_config(
        job_id=job_id,
        model=model,
        prompt_version=prompt_version,
        prompts=system_prompts,
        input_csv=input_path,
        item_column=resolved_item_column,
        responses_per_word=responses_per_word,
        temperature=temperature,
        max_tokens=max_tokens,
        base_seed=base_seed,
        source_word_count=len(source_words),
        word_count=len(words),
        max_words=max_words,
        selected_words=words,
    )
    manifest_path = pathlib.Path(output_dir) / "manifest.json"
    if resume:
        _validate_resume_manifest(manifest_path, config)
    jobs = build_generation_jobs(
        words,
        responses_per_word,
        model,
        base_seed,
        system_prompts,
    )
    unique_seeds = len({job["sampling_seed"] for job in jobs})
    expected_unique_seeds = len(words) * responses_per_word
    if unique_seeds != expected_unique_seeds:
        raise RuntimeError("Sampling-seed collision in feature-generation plan")
    return config | {
        "total_planned_responses": len(jobs),
        "planned_responses_by_prompt": {
            variant: len(words) * responses_per_word
            for variant in GENERATION_PROMPT_VARIANTS
        },
        "unique_response_ids": len({job["response_id"] for job in jobs}),
        "unique_sampling_seeds": unique_seeds,
    }


def run_feature_generation(
    *,
    job_id: str,
    input_csv: str | pathlib.Path,
    output_dir: str | pathlib.Path,
    client: Any,
    model: str,
    prompt_version: str = "v3",
    item_column: Optional[str] = None,
    responses_per_word: int = 20,
    temperature: float = 0.8,
    max_tokens: int = 500,
    max_workers: int = 32,
    base_seed: int = 20260801,
    max_words: Optional[int] = None,
    resume: bool = True,
) -> pathlib.Path:
    """Collect and preserve independent free-generation responses."""
    input_path = pathlib.Path(input_csv)
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    generations_csv = output_path / "feature_generations.csv"
    parse_errors_csv = output_path / "parse_errors.csv"
    long_csv = output_path / "generated_features_long.csv"
    frequency_csv = output_path / "generated_feature_frequencies.csv"
    manifest_path = output_path / "manifest.json"

    if not resume and any(
        path.exists()
        for path in [generations_csv, parse_errors_csv, manifest_path]
    ):
        raise FileExistsError(
            f"Output exists in {output_path}; use --resume or a new output directory"
        )

    plan = preflight_feature_generation(
        job_id=job_id,
        input_csv=input_path,
        output_dir=output_path,
        model=model,
        prompt_version=prompt_version,
        item_column=item_column,
        responses_per_word=responses_per_word,
        temperature=temperature,
        max_tokens=max_tokens,
        base_seed=base_seed,
        max_words=max_words,
        resume=resume,
    )
    words = select_stimulus_words(
        load_stimulus_words(input_path, item_column), max_words
    )
    system_prompts = load_generation_prompts(prompt_version)
    config = plan.copy()
    for summary_key in [
        "total_planned_responses",
        "planned_responses_by_prompt",
        "unique_response_ids",
        "unique_sampling_seeds",
    ]:
        config.pop(summary_key)

    jobs = build_generation_jobs(
        words,
        responses_per_word,
        model,
        base_seed,
        system_prompts,
    )
    completed = _load_completed_responses(generations_csv) if resume else set()
    pending = [job for job in jobs if job["response_id"] not in completed]

    started_at = _utc_now()
    manifest = config | {
        "total_planned_responses": len(jobs),
        "completed_before_run": len(completed),
        "pending_at_start": len(pending),
        "started_at": started_at,
        "finished_at": None,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    file_handler = logging.FileHandler(output_path / "run.log", mode="a")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    logging.getLogger().addHandler(file_handler)
    logger.info(
        "%s generation: %d words x %d prompts x %d responses = %d; pending=%d",
        prompt_version,
        len(words),
        len(GENERATION_PROMPT_VARIANTS),
        responses_per_word,
        len(jobs),
        len(pending),
    )

    _write_csv_header(generations_csv, GENERATION_COLUMNS)
    _write_csv_header(parse_errors_csv, GENERATION_COLUMNS)

    def generate_one(job: dict[str, Any]) -> dict[str, Any]:
        raw = ""
        metadata: dict[str, Any] = {}
        record = None
        parse_error = ""
        effective_seed = int(job["sampling_seed"])
        for attempt in range(2):
            effective_seed = (
                int(job["sampling_seed"]) + attempt * 1_000_000_007
            ) & 0x7FFFFFFFFFFFFFFF
            raw, metadata = client.generate(
                job["system_prompt"],
                job["user_message"],
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=_RESPONSE_FORMAT,
                seed=effective_seed,
            )
            record, parse_error = validate_generation_output(
                raw, expected_word=job["word_normalized"]
            )
            if record is not None:
                break
        features = record["features"] if record is not None else []
        return {
            "job_id": job_id,
            "response_id": job["response_id"],
            "word_original": job["word_original"],
            "word_normalized": job["word_normalized"],
            "prompt_variant": job["prompt_variant"],
            "replicate_id": job["replicate_id"],
            "model": model,
            "temperature": temperature,
            "sampling_seed": effective_seed,
            "features_json": json.dumps(features, ensure_ascii=True),
            "n_features": len(features),
            "raw_json": raw,
            "parse_error": parse_error or "",
            "prompt_hash": job["prompt_hash"],
            "finish_reason": metadata.get("finish_reason", ""),
            "prompt_tokens": metadata.get("prompt_tokens", ""),
            "completion_tokens": metadata.get("completion_tokens", ""),
        }

    completed_this_run = 0
    request_errors = 0
    with (
        generations_csv.open("a", newline="") as generations_handle,
        parse_errors_csv.open("a", newline="") as errors_handle,
    ):
        generation_writer = csv.DictWriter(
            generations_handle,
            fieldnames=GENERATION_COLUMNS,
            extrasaction="ignore",
        )
        error_writer = csv.DictWriter(
            errors_handle,
            fieldnames=GENERATION_COLUMNS,
            extrasaction="ignore",
        )
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(generate_one, job): job for job in pending}
            for future in as_completed(futures):
                job = futures[future]
                try:
                    row = future.result()
                except Exception as error:
                    request_errors += 1
                    row = {
                        "job_id": job_id,
                        "response_id": job["response_id"],
                        "word_original": job["word_original"],
                        "word_normalized": job["word_normalized"],
                        "prompt_variant": job["prompt_variant"],
                        "replicate_id": job["replicate_id"],
                        "model": model,
                        "temperature": temperature,
                        "sampling_seed": job["sampling_seed"],
                        "features_json": "[]",
                        "n_features": 0,
                        "raw_json": "",
                        "parse_error": f"Request error: {error}",
                        "prompt_hash": job["prompt_hash"],
                    }
                generation_writer.writerow(row)
                generations_handle.flush()
                if row.get("parse_error"):
                    error_writer.writerow(row)
                    errors_handle.flush()
                else:
                    completed_this_run += 1
                if (completed_this_run + request_errors) % 100 == 0:
                    logger.info(
                        "Progress: %d/%d pending responses processed",
                        completed_this_run + request_errors,
                        len(pending),
                    )

    valid_total, feature_total = _write_derived_outputs(
        generations_csv, long_csv, frequency_csv
    )
    all_rows = pd.read_csv(generations_csv, dtype=str).fillna("")
    latest = all_rows.drop_duplicates("response_id", keep="last")
    parse_error_total = int((latest["parse_error"] != "").sum())
    valid_latest = latest[latest["parse_error"] == ""]
    valid_by_prompt = {
        variant: int(
            (valid_latest["prompt_variant"] == variant).sum()
        )
        for variant in GENERATION_PROMPT_VARIANTS
    }
    manifest |= {
        "completed_this_run": completed_this_run,
        "request_errors_this_run": request_errors,
        "valid_responses_total": valid_total,
        "valid_responses_by_prompt": valid_by_prompt,
        "parse_errors_total": parse_error_total,
        "generated_feature_tokens_total": feature_total,
        "pending_after_run": len(jobs) - valid_total,
        "finished_at": _utc_now(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    logger.info(
        "%s generation complete: valid=%d/%d, features=%d, errors=%d",
        prompt_version,
        valid_total,
        len(jobs),
        feature_total,
        parse_error_total,
    )
    logging.getLogger().removeHandler(file_handler)
    file_handler.close()
    return generations_csv


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--item-column", default=None)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="Qwen2.5-72B-Instruct")
    parser.add_argument(
        "--prompt-version",
        choices=("v3", "v3.1"),
        default="v3",
        help="free-generation prompt set; V3 remains the backward-compatible default",
    )
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument(
        "--responses-per-word",
        type=int,
        default=20,
        help="simulated participant responses per word for each prompt variant",
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=None,
        help="deterministic evenly spaced word subset; intended for smoke tests",
    )
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--max-workers", type=int, default=32)
    parser.add_argument("--base-seed", type=int, default=20260801)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="validate and print the run plan without creating files or model calls",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _parse_args(argv)
    if args.preflight_only:
        plan = preflight_feature_generation(
            job_id=args.job_id,
            input_csv=args.input_csv,
            item_column=args.item_column,
            output_dir=args.output_dir,
            model=args.model,
            prompt_version=args.prompt_version,
            responses_per_word=args.responses_per_word,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            base_seed=args.base_seed,
            max_words=args.max_words,
            resume=args.resume,
        )
        print(json.dumps(plan, indent=2))
        return
    client_class = importlib.import_module("vllm_client").VLLMClient
    client = client_class(model_name=args.model, base_url=args.base_url)
    run_feature_generation(
        job_id=args.job_id,
        input_csv=args.input_csv,
        item_column=args.item_column,
        output_dir=args.output_dir,
        client=client,
        model=args.model,
        prompt_version=args.prompt_version,
        responses_per_word=args.responses_per_word,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_workers=args.max_workers,
        base_seed=args.base_seed,
        max_words=args.max_words,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
