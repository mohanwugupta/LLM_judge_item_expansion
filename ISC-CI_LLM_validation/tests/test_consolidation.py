import numpy as np
import pandas as pd

from consolidate_v3 import split_rejected_clusters

from iscci_validation.consolidation import (
    adds_substantive_qualifier,
    consolidate_phrase_types,
    embedding_merge_eligible,
    lexical_signature,
)


def test_lexical_signature_merges_scaffolding_and_morphology() -> None:
    assert lexical_signature("is a mammal") == lexical_signature("mammal")
    assert lexical_signature("used for cutting") == lexical_signature("can cut")
    assert lexical_signature("has four legs") == lexical_signature("four-legged")
    assert lexical_signature("furry") == lexical_signature("has fur")
    assert lexical_signature("made of metal and wood") != lexical_signature(
        "made of metal or wood"
    )


def test_substantive_qualifier_is_not_erased() -> None:
    assert adds_substantive_qualifier("has a tail", "has a long tail")
    assert adds_substantive_qualifier(
        "often made of wool", "typically made of wool or silk"
    )
    assert not adds_substantive_qualifier("is large", "large size")


def test_non_english_phrase_is_not_embedding_merged() -> None:
    assert embedding_merge_eligible("has a soft body")
    assert not embedding_merge_eligible("身体柔软")


def test_semantic_merge_requires_complete_link() -> None:
    phrases = ["alpha", "beta", "gamma"]
    embeddings = np.asarray(
        [
            [1.0, 0.0],
            [0.9, np.sqrt(1 - 0.9**2)],
            [0.62, np.sqrt(1 - 0.62**2)],
        ],
        dtype=np.float32,
    )
    profiles = embeddings.copy()
    clusters = consolidate_phrase_types(
        phrases,
        embeddings,
        profiles,
        embedding_threshold=0.8,
        profile_threshold=0.5,
        nearest_neighbors=3,
    )
    assert clusters == [[0, 1], [2]]


def test_rejected_semantic_cluster_splits_to_lexical_groups() -> None:
    phrases = ["driven on highways", "driven on roads", "used on roads"]
    assignments = pd.DataFrame(
        {
            "feature_text_normalized": phrases,
            "cluster_id": ["C_rejected", "C_rejected", "C_rejected"],
            "lexical_signature": ["highway", "road", "road"],
        }
    )
    revised = split_rejected_clusters(
        [[0, 1, 2]], phrases, assignments, {"C_rejected"}
    )
    assert revised == [[0], [1, 2]]
