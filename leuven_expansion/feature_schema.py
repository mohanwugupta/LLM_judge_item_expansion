"""
leuven_expansion/feature_schema.py

Load the frozen Leuven feature schema from the feature matrix CSV,
validate atomic judge JSON outputs against the JSON schema, and
provide helper utilities for schema-level checks.
"""
from __future__ import annotations

import json
import pathlib
from typing import Any, Dict, List, Optional, Tuple

import jsonschema
import pandas as pd

_SCHEMA_PATH = pathlib.Path(__file__).parent / "schemas" / "atomic_feature_judgment_schema_v1.json"


def load_json_schema(path: str | pathlib.Path = _SCHEMA_PATH) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


_JSON_SCHEMA: Dict[str, Any] = load_json_schema()


# ---------------------------------------------------------------------------
# Leuven feature schema (derived from the CSV)
# ---------------------------------------------------------------------------

def load_leuven_feature_schema(
    features_csv: str | pathlib.Path,
    item_column: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Read the Leuven feature matrix CSV and return a schema dict describing
    the frozen feature set.

    Parameters
    ----------
    features_csv : path to leuven_combined_features_consolidated.csv
    item_column  : name of the word-label column.  When ``None`` (default)
                   the first column in the CSV is used automatically.

    Returns
    -------
    dict with keys:
        schema_version, item_column, n_original_items, n_features,
        feature_columns, feature_id_map, value_scale
    """
    df = pd.read_csv(features_csv)

    if item_column is None:
        item_column = df.columns[0]

    if item_column not in df.columns:
        raise ValueError(
            f"Expected item column '{item_column}' not found in {features_csv}. "
            f"Available columns: {list(df.columns[:5])}"
        )

    feature_columns: List[str] = [c for c in df.columns if c != item_column]
    feature_id_map: Dict[str, int] = {feat: idx for idx, feat in enumerate(feature_columns)}

    return {
        "schema_version": "leuven_feature_schema_v1",
        "item_column": item_column,
        "n_original_items": len(df),
        "n_features": len(feature_columns),
        "feature_columns": feature_columns,
        "feature_id_map": feature_id_map,
        "value_scale": {
            "min": 0,
            "max": 4,
            "interpretation": "Leuven-style feature applicability",
        },
    }


def get_feature_text(schema: Dict[str, Any], feature_id: int) -> str:
    """Return the feature text for a given feature_id integer index."""
    cols = schema["feature_columns"]
    if feature_id < 0 or feature_id >= len(cols):
        raise IndexError(f"feature_id {feature_id} out of range [0, {len(cols)-1}]")
    return cols[feature_id]


def get_feature_id(schema: Dict[str, Any], feature_text: str) -> int:
    """Return the integer feature_id for a given feature text."""
    fmap = schema["feature_id_map"]
    if feature_text not in fmap:
        raise KeyError(f"Feature text not found in schema: {feature_text!r}")
    return fmap[feature_text]


# ---------------------------------------------------------------------------
# Atomic judgment validation
# ---------------------------------------------------------------------------

def validate_judge_output(
    raw: str,
    expected_word: Optional[str] = None,
    expected_feature_id: Optional[int] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Parse *raw* as JSON, validate against the atomic judgment schema, and
    optionally verify that returned word and feature_id match expectations.

    Returns (record, None) on success or (None, error_message) on failure.
    """
    text = raw.strip()
    if text.startswith("```"):
        lines = [l for l in text.splitlines() if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    try:
        obj = json.loads(text)
    except json.JSONDecodeError as e:
        return None, f"JSON parse error: {e}"

    try:
        jsonschema.validate(instance=obj, schema=_JSON_SCHEMA)
    except jsonschema.ValidationError as e:
        return None, f"Schema validation error: {e.message}"

    # Cross-field checks
    if expected_word is not None:
        if obj.get("target_word", "").strip().lower() != expected_word.strip().lower():
            return None, (
                f"Returned target_word '{obj.get('target_word')}' does not match "
                f"expected '{expected_word}'"
            )

    if expected_feature_id is not None:
        if obj.get("feature_id") != expected_feature_id:
            return None, (
                f"Returned feature_id {obj.get('feature_id')} does not match "
                f"expected {expected_feature_id}"
            )

    # Truncate reason if needed
    if len(obj.get("reason", "")) > 180:
        obj["reason"] = obj["reason"][:177] + "..."

    return obj, None


def is_high_confidence(confidence: float, threshold: float = 0.80) -> bool:
    return confidence >= threshold
