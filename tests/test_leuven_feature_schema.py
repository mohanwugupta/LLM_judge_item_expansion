"""
tests/test_leuven_feature_schema.py

Tests for leuven_expansion/feature_schema.py
"""
import json
import pathlib
import tempfile

import pandas as pd
import pytest

from leuven_expansion.feature_schema import (
    load_leuven_feature_schema,
    validate_judge_output,
    get_feature_id,
    get_feature_text,
    load_json_schema,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_features_csv(tmp_path):
    """Create a minimal Leuven feature matrix CSV."""
    df = pd.DataFrame({
        "Name": ["dog", "cat", "hammer"],
        "is an animal": [4.0, 4.0, 0.0],
        "is a tool": [0.0, 0.0, 4.0],
        "can fly": [0.0, 0.0, 0.0],
        "has legs": [3.0, 3.0, 0.0],
    })
    p = tmp_path / "features.csv"
    df.to_csv(p, index=False)
    return p


@pytest.fixture
def schema(sample_features_csv):
    return load_leuven_feature_schema(sample_features_csv)


# ── Schema loading tests ───────────────────────────────────────────────────────

def test_loads_feature_columns(schema):
    """Feature columns are preserved exactly (minus item column)."""
    assert schema["feature_columns"] == ["is an animal", "is a tool", "can fly", "has legs"]


def test_first_column_is_item_name(sample_features_csv):
    """First column treated as item name; must be 'Name'."""
    schema = load_leuven_feature_schema(sample_features_csv, item_column="Name")
    assert schema["item_column"] == "Name"


def test_n_original_items(schema):
    assert schema["n_original_items"] == 3


def test_n_features(schema):
    assert schema["n_features"] == 4


def test_feature_id_map(schema):
    assert schema["feature_id_map"]["is an animal"] == 0
    assert schema["feature_id_map"]["is a tool"] == 1


def test_get_feature_text(schema):
    assert get_feature_text(schema, 0) == "is an animal"
    assert get_feature_text(schema, 1) == "is a tool"


def test_get_feature_id(schema):
    assert get_feature_id(schema, "is an animal") == 0


def test_feature_columns_preserved_exactly(sample_features_csv):
    """Feature columns are not reordered or renamed."""
    schema = load_leuven_feature_schema(sample_features_csv)
    df = pd.read_csv(sample_features_csv)
    expected = [c for c in df.columns if c != "Name"]
    assert schema["feature_columns"] == expected


def test_missing_item_column_raises(tmp_path):
    df = pd.DataFrame({"Word": ["dog"], "is an animal": [4.0]})
    p = tmp_path / "bad.csv"
    df.to_csv(p, index=False)
    with pytest.raises(ValueError, match="item column"):
        load_leuven_feature_schema(p, item_column="Name")


# ── JSON validation tests ──────────────────────────────────────────────────────

def _make_valid_output(
    word="dog", fid=0, value=3.0, confidence=0.9, ambiguous=False
) -> str:
    return json.dumps({
        "target_word": word,
        "feature_id": fid,
        "feature_value": value,
        "confidence": confidence,
        "ambiguous": ambiguous,
        "reason": "Dogs are animals.",
    })


def test_valid_atomic_json_passes():
    record, err = validate_judge_output(_make_valid_output())
    assert err is None
    assert record["feature_value"] == 3.0


def test_returned_feature_id_must_match():
    raw = _make_valid_output(fid=0)
    record, err = validate_judge_output(raw, expected_feature_id=5)
    assert err is not None
    assert "feature_id" in err


def test_returned_target_word_must_match():
    raw = _make_valid_output(word="dog")
    record, err = validate_judge_output(raw, expected_word="cat")
    assert err is not None
    assert "target_word" in err.lower()


def test_extra_json_fields_fail():
    obj = json.loads(_make_valid_output())
    obj["extra_field"] = "bad"
    _, err = validate_judge_output(json.dumps(obj))
    assert err is not None
    assert "additionalProperties" in err or "extra_field" in err.lower() or "Additional" in err


def test_missing_required_field_fails():
    obj = json.loads(_make_valid_output())
    del obj["confidence"]
    _, err = validate_judge_output(json.dumps(obj))
    assert err is not None


def test_feature_value_out_of_range_fails():
    obj = json.loads(_make_valid_output())
    obj["feature_value"] = 5.0
    _, err = validate_judge_output(json.dumps(obj))
    assert err is not None


def test_feature_value_lower_bound():
    obj = json.loads(_make_valid_output())
    obj["feature_value"] = -1.0
    _, err = validate_judge_output(json.dumps(obj))
    assert err is not None


def test_confidence_out_of_range_fails():
    obj = json.loads(_make_valid_output())
    obj["confidence"] = 1.5
    _, err = validate_judge_output(json.dumps(obj))
    assert err is not None


def test_feature_value_boundary_zero():
    _, err = validate_judge_output(_make_valid_output(value=0.0))
    assert err is None


def test_feature_value_boundary_four():
    _, err = validate_judge_output(_make_valid_output(value=4.0))
    assert err is None


def test_strips_markdown_fences():
    raw = "```json\n" + _make_valid_output() + "\n```"
    record, err = validate_judge_output(raw)
    assert err is None
    assert record is not None


def test_invalid_json_returns_error():
    _, err = validate_judge_output("{not valid json}")
    assert err is not None
    assert "parse error" in err.lower()
