import pandas as pd
import pytest

from leuven_expansion.feature_schema import load_candidate_feature_schema
from leuven_expansion.v4 import candidate_inventory_hash


def write_bank(path):
    bank = pd.DataFrame(
        {
            "candidate_index": [0, 1],
            "candidate_id": ["v4_a", "v4_b"],
            "canonical_feature_text": ["has fur", "can fly"],
        }
    )
    bank["candidate_inventory_hash"] = candidate_inventory_hash(bank)
    bank.to_csv(path, index=False)


def test_candidate_schema_maps_stable_ids_to_existing_integer_interface(tmp_path):
    path = tmp_path / "bank.csv"
    write_bank(path)
    schema = load_candidate_feature_schema(path)
    assert schema["feature_columns"] == ["has fur", "can fly"]
    assert schema["candidate_id_by_feature_id"] == {0: "v4_a", 1: "v4_b"}
    assert schema["feature_id_by_candidate_id"] == {"v4_a": 0, "v4_b": 1}


def test_candidate_schema_fails_closed_on_duplicate_ids(tmp_path):
    path = tmp_path / "bank.csv"
    write_bank(path)
    bank = pd.read_csv(path)
    bank.loc[1, "candidate_id"] = "v4_a"
    bank.to_csv(path, index=False)
    with pytest.raises(ValueError, match="unique"):
        load_candidate_feature_schema(path)
