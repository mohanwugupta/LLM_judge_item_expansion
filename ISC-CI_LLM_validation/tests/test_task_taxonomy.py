import copy
import json
from pathlib import Path

import pytest

from semantic_retraining.taxonomy import load_taxonomy, taxonomy_hash, validate_taxonomy

ROOT = Path(__file__).resolve().parents[1]


def test_investigator_approved_32_class_reconstruction():
    taxonomy = load_taxonomy(ROOT / "configs/task_taxonomy_32_reconstructed.json")
    assert len(taxonomy["classes"]) == 32
    assert taxonomy["historical_numeric_correspondence"] is False
    assert taxonomy["original_claimed_count"] == 33
    assert taxonomy["investigator_approved_reconstruction"] is True
    assert len({c["family"] for c in taxonomy["classes"]}) == 5
    assert taxonomy_hash(taxonomy) == taxonomy_hash(copy.deepcopy(taxonomy))


def test_missing_unknown_and_duplicate_classes_fail():
    taxonomy = load_taxonomy(ROOT / "configs/task_taxonomy_32_reconstructed.json")
    with pytest.raises(ValueError, match="count"):
        validate_taxonomy({**taxonomy, "classes": taxonomy["classes"][:-1]})
    wrong = copy.deepcopy(taxonomy)
    wrong["classes"][0]["id"] = wrong["classes"][1]["id"]
    with pytest.raises(ValueError, match="IDs"):
        validate_taxonomy(wrong)
    wrong = copy.deepcopy(taxonomy)
    wrong["classes"][0]["confusable"] = [99]
    with pytest.raises(ValueError, match="neighbor"):
        validate_taxonomy(wrong)
    with pytest.raises(ValueError, match="approved"):
        validate_taxonomy({**taxonomy, "investigator_approved_reconstruction": False})
