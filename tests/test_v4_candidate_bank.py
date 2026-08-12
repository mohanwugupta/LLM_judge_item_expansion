import json

import pandas as pd

from build_v4_candidate_bank import apply_review, build_bank
from leuven_expansion.v4 import stable_candidate_id


def test_candidate_id_is_stable_under_member_reordering():
    left = stable_candidate_id(["has fur", "is furry"], "test-v1")
    right = stable_candidate_id(["is furry", "has fur"], "test-v1")
    assert left == right


def test_rejected_merge_splits_and_valid_singletons_survive(tmp_path):
    candidates = pd.DataFrame(
        [
            {
                "merge_candidate_id": "m1",
                "member_phrases": json.dumps(["driven on highways", "driven on roads"]),
                "merge_basis": "semantic_profile",
                "verdict": "",
                "review_note": "",
                "reviewer": "",
                "reviewed_at": "",
            }
        ]
    )
    review = candidates.copy()
    review["verdict"] = "reject"
    path = tmp_path / "review.csv"
    review.to_csv(path, index=False)
    clusters, merged = apply_review(
        [["driven on highways", "driven on roads"], ["has fur"]],
        candidates,
        path,
    )
    assert clusters == [["driven on highways"], ["driven on roads"], ["has fur"]]
    assert merged.loc[0, "verdict"] == "reject"


def test_bank_retains_singletons_and_source_provenance():
    long_data = pd.DataFrame(
        {
            "feature_text_normalized": ["has fur", "has fur", "barks"],
            "word_normalized": ["dog", "cat", "dog"],
            "response_id": ["s:r1", "s:r2", "s:r1"],
            "source_response_id": ["r1", "r2", "r1"],
            "source_id": ["s", "s", "s"],
            "source_version": ["v4", "v4", "v4"],
            "generation_round": [1, 1, 1],
            "source_model": ["model", "model", "model"],
            "prompt_variant": ["broad", "broad", "broad"],
        }
    )
    fixed = pd.DataFrame(
        columns=[
            "candidate_id",
            "canonical_feature_text",
            "member_phrases",
            "fixed_v3_1_b_cluster_id",
            "fixed_v3_1_b_order",
            "normalization_version",
            "merge_review_status",
        ]
    )
    bank, assignments = build_bank(
        long_data,
        [["barks"], ["has fur"]],
        fixed,
        {},
        "test-v1",
    )
    assert set(bank["canonical_feature_text"]) == {"barks", "has fur"}
    fur = bank.loc[bank["canonical_feature_text"].eq("has fur")].iloc[0]
    assert json.loads(fur["source_words"]) == ["cat", "dog"]
    assert fur["n_independent_responses"] == 2
    assert assignments["candidate_id"].notna().all()
