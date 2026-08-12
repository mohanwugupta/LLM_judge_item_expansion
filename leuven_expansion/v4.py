"""Shared V4 adapters and stable identifiers over existing Leuven pipelines."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_hash(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, ensure_ascii=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def stable_candidate_id(
    member_phrases: Iterable[str], normalization_version: str, namespace: str = "ensemble"
) -> str:
    canonical = {
        "namespace": namespace,
        "normalization_version": normalization_version,
        "member_phrases": sorted(set(map(str, member_phrases))),
    }
    return f"v4_{stable_json_hash(canonical)[:16]}"


def candidate_inventory_hash(frame: pd.DataFrame) -> str:
    required = ["candidate_id", "canonical_feature_text"]
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"Candidate inventory is missing columns: {sorted(missing)}")
    rows = frame[required].astype(str).to_dict(orient="records")
    return stable_json_hash(rows)


def stable_shard(candidate_id: str, word: str, shard_count: int) -> int:
    if shard_count < 1:
        raise ValueError("shard_count must be positive")
    digest = hashlib.sha256(f"{candidate_id}\0{word}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % shard_count


def write_json(path: str | Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def json_list(values: Sequence[object]) -> str:
    return json.dumps(sorted(set(map(str, values))), ensure_ascii=True)
