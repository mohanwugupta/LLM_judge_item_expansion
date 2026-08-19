from __future__ import annotations

import hashlib
import importlib.util
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer


TOKEN_RE = re.compile(r"[a-z0-9]+")
CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
LEADING_SCAFFOLD_RE = re.compile(
    r"^(?:is\s+used\s+(?:for|to)|used\s+(?:for|to)|can\s+be|can|could\s+be|"
    r"could|is|are|has|have|serves\s+as)\s+"
)
NEGATION_TOKENS = frozenset({"cannot", "never", "no", "not", "without"})
GENERIC_QUALIFIERS = frozenset(
    {"ability", "appearance", "capability", "characteristic", "color", "colour", "size"}
)
STEM_REWRITES = {
    "colore": "color",
    "colour": "color",
    "furri": "fur",
    "gray": "grey",
}
SIGNATURE_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "be",
        "commonly",
        "frequently",
        "generally",
        "often",
        "the",
        "to",
        "typically",
        "usually",
    }
)
REQUIRED_COLUMNS = {
    "response_id",
    "word_normalized",
    "prompt_variant",
    "feature_text_normalized",
}


@dataclass(frozen=True)
class ConsolidationParameters:
    embedding_model: str
    embedding_model_revision: str
    embedding_similarity_threshold: float
    profile_similarity_threshold: float
    nearest_neighbors: int
    rater_cutoff: int
    object_cutoff: int


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = np.arange(size, dtype=np.int32)
        self.members: dict[int, list[int]] = {index: [index] for index in range(size)}

    def find(self, value: int) -> int:
        parent = int(self.parent[value])
        if parent != value:
            self.parent[value] = self.find(parent)
        return int(self.parent[value])

    def union(self, left: int, right: int) -> int:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return left_root
        keep, drop = sorted((left_root, right_root))
        self.parent[drop] = keep
        self.members[keep].extend(self.members.pop(drop))
        self.members[keep].sort()
        return keep


def normalize_phrase(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def lexical_signature(value: str) -> tuple[str, ...]:
    phrase = normalize_phrase(value).replace("-", " ")
    previous = None
    while phrase != previous:
        previous = phrase
        phrase = LEADING_SCAFFOLD_RE.sub("", phrase).strip()
        phrase = re.sub(r"^(?:a|an|the)\s+", "", phrase)

    stemmer = PorterStemmer()
    tokens: list[str] = []
    for token in TOKEN_RE.findall(phrase):
        if token in SIGNATURE_STOPWORDS:
            continue
        stem = stemmer.stem(token)
        tokens.append(STEM_REWRITES.get(stem, stem))
    return tuple(sorted(tokens))


def _has_negation_mismatch(left: str, right: str) -> bool:
    left_tokens = set(TOKEN_RE.findall(normalize_phrase(left)))
    right_tokens = set(TOKEN_RE.findall(normalize_phrase(right)))
    return bool(left_tokens & NEGATION_TOKENS) != bool(right_tokens & NEGATION_TOKENS)


def adds_substantive_qualifier(left: str, right: str) -> bool:
    left_tokens = set(lexical_signature(left))
    right_tokens = set(lexical_signature(right))
    if not left_tokens or not right_tokens or left_tokens == right_tokens:
        return False
    if left_tokens < right_tokens:
        return bool((right_tokens - left_tokens) - GENERIC_QUALIFIERS)
    if right_tokens < left_tokens:
        return bool((left_tokens - right_tokens) - GENERIC_QUALIFIERS)
    return False


def embedding_merge_eligible(value: str) -> bool:
    return bool(TOKEN_RE.search(normalize_phrase(value))) and not bool(
        CJK_RE.search(str(value))
    )


def _load_sentence_transformer(model_name: str, revision: str):
    # The installed torchvision build is unrelated to text embeddings and can be
    # incompatible with torch. Hiding that optional package keeps transformers on
    # its text-only import path; this does not alter the sentence model.
    original_find_spec = importlib.util.find_spec

    def text_only_find_spec(name: str, *args, **kwargs):
        if name == "torchvision" or name.startswith("torchvision."):
            return None
        return original_find_spec(name, *args, **kwargs)

    importlib.util.find_spec = text_only_find_spec
    try:
        from sentence_transformers import SentenceTransformer
    finally:
        importlib.util.find_spec = original_find_spec
    return SentenceTransformer(
        model_name,
        revision=revision,
        local_files_only=True,
        device="cpu",
    )


def encode_phrases(
    phrases: Sequence[str], model_name: str, revision: str, batch_size: int = 256
) -> np.ndarray:
    model = _load_sentence_transformer(model_name, revision)
    embeddings = model.encode(
        list(phrases),
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return np.asarray(embeddings, dtype=np.float32)


def build_phrase_profiles(
    long_data: pd.DataFrame, phrases: Sequence[str], words: Sequence[str]
) -> np.ndarray:
    phrase_to_index = {phrase: index for index, phrase in enumerate(phrases)}
    word_to_index = {word: index for index, word in enumerate(words)}
    counts = (
        long_data.drop_duplicates(
            ["response_id", "word_normalized", "feature_text_normalized"]
        )
        .groupby(["feature_text_normalized", "word_normalized"], observed=True)
        .size()
    )
    profiles = np.zeros((len(phrases), len(words)), dtype=np.float32)
    for (phrase, word), count in counts.items():
        profiles[phrase_to_index[str(phrase)], word_to_index[str(word)]] = float(count)
    norms = np.linalg.norm(profiles, axis=1, keepdims=True)
    profiles /= np.where(norms == 0, 1.0, norms)
    return profiles


def nearest_neighbor_edges(
    embeddings: np.ndarray, threshold: float, nearest_neighbors: int
) -> list[tuple[float, int, int]]:
    import torch

    count = embeddings.shape[0]
    neighbors = min(nearest_neighbors + 1, count)
    edges: dict[tuple[int, int], float] = {}
    matrix = torch.from_numpy(np.ascontiguousarray(embeddings))
    for start in range(0, count, 256):
        stop = min(start + 256, count)
        similarities = matrix[start:stop] @ matrix.T
        values, indices = torch.topk(similarities, k=neighbors, dim=1)
        for offset in range(stop - start):
            left = start + offset
            for similarity, right_value in zip(values[offset], indices[offset]):
                right = int(right_value)
                score = float(similarity)
                if left == right or score < threshold:
                    continue
                pair = (left, right) if left < right else (right, left)
                edges[pair] = max(edges.get(pair, -math.inf), score)
    return sorted(
        ((score, left, right) for (left, right), score in edges.items()),
        key=lambda edge: (-edge[0], edge[1], edge[2]),
    )


def _all_cross_pairs_pass(
    left_members: Sequence[int],
    right_members: Sequence[int],
    phrases: Sequence[str],
    embeddings: np.ndarray,
    profiles: np.ndarray,
    embedding_threshold: float,
    profile_threshold: float,
) -> bool:
    for left in left_members:
        for right in right_members:
            if not embedding_merge_eligible(phrases[left]) or not embedding_merge_eligible(
                phrases[right]
            ):
                return False
            if _has_negation_mismatch(phrases[left], phrases[right]):
                return False
            if adds_substantive_qualifier(phrases[left], phrases[right]):
                return False
            if float(embeddings[left] @ embeddings[right]) < embedding_threshold:
                return False
            if float(profiles[left] @ profiles[right]) < profile_threshold:
                return False
    return True


def consolidate_phrase_types(
    phrases: Sequence[str],
    embeddings: np.ndarray,
    profiles: np.ndarray,
    embedding_threshold: float,
    profile_threshold: float,
    nearest_neighbors: int,
) -> list[list[int]]:
    union_find = UnionFind(len(phrases))

    signatures: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for index, phrase in enumerate(phrases):
        signature = lexical_signature(phrase)
        if signature:
            signatures[signature].append(index)
    for members in signatures.values():
        for member in members[1:]:
            union_find.union(members[0], member)

    for _, left, right in nearest_neighbor_edges(
        embeddings, embedding_threshold, nearest_neighbors
    ):
        left_root = union_find.find(left)
        right_root = union_find.find(right)
        if left_root == right_root:
            continue
        if _all_cross_pairs_pass(
            union_find.members[left_root],
            union_find.members[right_root],
            phrases,
            embeddings,
            profiles,
            embedding_threshold,
            profile_threshold,
        ):
            union_find.union(left_root, right_root)

    return sorted(
        (sorted(members) for members in union_find.members.values()),
        key=lambda members: members[0],
    )


def _cluster_id(prompt: str, members: Iterable[str]) -> str:
    payload = prompt + "\0" + "\0".join(sorted(members))
    return f"{prompt}_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:12]}"


def make_assignments(
    prompt: str,
    phrases: Sequence[str],
    clusters: Sequence[Sequence[int]],
    embeddings: np.ndarray,
    profiles: np.ndarray,
    global_frequency: pd.Series,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for members in clusters:
        member_phrases = [phrases[index] for index in members]
        cluster_id = _cluster_id(prompt, member_phrases)
        canonical = sorted(
            member_phrases,
            key=lambda phrase: (
                -int(global_frequency.get(phrase, 0)),
                len(phrase),
                phrase,
            ),
        )[0]
        if len(members) == 1:
            embedding_min = embedding_mean = profile_min = profile_mean = 1.0
        else:
            embedding_values: list[float] = []
            profile_values: list[float] = []
            for offset, left in enumerate(members):
                for right in members[offset + 1 :]:
                    embedding_values.append(float(embeddings[left] @ embeddings[right]))
                    profile_values.append(float(profiles[left] @ profiles[right]))
            embedding_min = min(embedding_values)
            embedding_mean = float(np.mean(embedding_values))
            profile_min = min(profile_values)
            profile_mean = float(np.mean(profile_values))
        for index in members:
            signatures = {lexical_signature(phrases[value]) for value in members}
            merge_basis = (
                "singleton"
                if len(members) == 1
                else "lexical_signature"
                if len(signatures) == 1
                else "semantic_profile"
            )
            rows.append(
                {
                    "prompt_variant": prompt,
                    "feature_text_normalized": phrases[index],
                    "cluster_id": cluster_id,
                    "canonical_feature": canonical,
                    "cluster_size": len(members),
                    "cluster_merge_basis": merge_basis,
                    "cluster_signature_count": len(signatures),
                    "global_response_frequency": int(
                        global_frequency.get(phrases[index], 0)
                    ),
                    "within_cluster_embedding_min": embedding_min,
                    "within_cluster_embedding_mean": embedding_mean,
                    "within_cluster_profile_min": profile_min,
                    "within_cluster_profile_mean": profile_mean,
                    "lexical_signature": " ".join(lexical_signature(phrases[index])),
                    "contains_cjk": bool(CJK_RE.search(phrases[index])),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["cluster_id", "feature_text_normalized"], ignore_index=True
    )


def build_cluster_counts(
    prompt_data: pd.DataFrame,
    assignments: pd.DataFrame,
    words: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mapped = prompt_data.merge(
        assignments[["feature_text_normalized", "cluster_id"]],
        on="feature_text_normalized",
        how="left",
        validate="many_to_one",
    )
    if mapped["cluster_id"].isna().any():
        raise ValueError("At least one generated feature was not assigned to a cluster")

    deduplicated = mapped.drop_duplicates(
        ["response_id", "word_normalized", "cluster_id"]
    )
    frequencies = (
        deduplicated.groupby(["word_normalized", "cluster_id"], observed=True)
        .size()
        .rename("response_frequency")
        .reset_index()
    )
    cluster_ids = sorted(assignments["cluster_id"].unique())
    counts = (
        frequencies.pivot(
            index="word_normalized", columns="cluster_id", values="response_frequency"
        )
        .reindex(index=list(words), columns=cluster_ids)
        .fillna(0)
        .astype(np.int16)
    )
    counts.index.name = "word_normalized"
    return counts, frequencies


def retained_training_matrix(
    counts: pd.DataFrame, rater_cutoff: int, object_cutoff: int
) -> pd.DataFrame:
    binary = counts.gt(rater_cutoff).astype(np.int8)
    return binary.loc[:, binary.sum(axis=0).gt(object_cutoff)]


def summarize_clusters(
    prompt: str,
    threshold: float,
    assignments: pd.DataFrame,
    counts: pd.DataFrame,
    training_matrix: pd.DataFrame,
) -> dict[str, object]:
    cluster_sizes = assignments.groupby("cluster_id", observed=True).size()
    return {
        "prompt_variant": prompt,
        "embedding_threshold": threshold,
        "phrase_types": int(len(assignments)),
        "clusters_total": int(cluster_sizes.size),
        "multi_phrase_clusters": int(cluster_sizes.gt(1).sum()),
        "merged_phrase_types": int(len(assignments) - cluster_sizes.size),
        "largest_cluster_size": int(cluster_sizes.max()),
        "positive_word_cluster_pairs": int(counts.gt(3).sum().sum()),
        "retained_training_tasks": int(training_matrix.shape[1]),
        "training_positive_cells": int(training_matrix.sum().sum()),
        "training_density": float(training_matrix.values.mean())
        if training_matrix.shape[1]
        else 0.0,
    }


def validate_long_data(long_data: pd.DataFrame, prompts: Sequence[str]) -> None:
    missing = REQUIRED_COLUMNS - set(long_data.columns)
    if missing:
        raise ValueError(f"Missing V3 columns: {sorted(missing)}")
    actual_prompts = set(long_data["prompt_variant"].astype(str).unique())
    if actual_prompts != set(prompts):
        raise ValueError(
            f"Prompt variants differ: expected {sorted(prompts)}, got {sorted(actual_prompts)}"
        )
    duplicates = long_data.duplicated(
        ["response_id", "feature_text_normalized"], keep=False
    )
    if duplicates.any():
        raise ValueError("A response contains duplicate normalized feature rows")
