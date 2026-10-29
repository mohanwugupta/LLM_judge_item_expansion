from __future__ import annotations

import hashlib
import json
from pathlib import Path


def taxonomy_hash(taxonomy: dict) -> str:
    return hashlib.sha256(
        json.dumps(taxonomy, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def validate_taxonomy(taxonomy: dict) -> None:
    if taxonomy.get("version") != "isc-table-a1-reconstruction-32-v1":
        raise ValueError("Unknown taxonomy version; use a separately reviewed protocol")
    if taxonomy.get("investigator_approved_reconstruction") is not True:
        raise ValueError("32-class reconstruction must be explicitly approved")
    classes = taxonomy.get("classes", [])
    if len(classes) != 32:
        raise ValueError("Reconstructed taxonomy count must be 32")
    ids = [c["id"] for c in classes]
    if ids != list(range(32)):
        raise ValueError("Task IDs must be unique and ordered 0..31")
    if (
        len({c["name"] for c in classes}) != 32
        or len({c["family"] for c in classes}) != 5
    ):
        raise ValueError("Expected distinct class names under five families")
    for c in classes:
        if not all(
            c.get(k)
            for k in ["name", "family", "description", "examples", "confusable"]
        ):
            raise ValueError(f"Incomplete class definition: {c}")
        if not set(c["confusable"]).issubset(set(ids) - {c["id"]}):
            raise ValueError(f"Invalid confusable neighbor for {c['name']}")
    if taxonomy.get("historical_numeric_correspondence") is not False:
        raise ValueError("Original numeric task correspondence has not been verified")


def load_taxonomy(path: Path) -> dict:
    taxonomy = json.loads(path.read_text())
    validate_taxonomy(taxonomy)
    return taxonomy
