from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def set_random_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def matrix_sha256(matrix: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update("\0".join(map(str, matrix.index)).encode("utf-8"))
    digest.update(b"\0")
    digest.update("\0".join(map(str, matrix.columns)).encode("utf-8"))
    digest.update(b"\0")
    digest.update(np.ascontiguousarray(matrix.values).tobytes())
    return digest.hexdigest()


def load_human_matrix(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = pd.read_csv(path, index_col=0, encoding="ISO-8859-1")
    binary = counts.gt(3).astype(np.int8)
    retained = binary.loc[:, binary.sum(axis=0).gt(3)]
    return counts, retained


def load_v2_matrix(
    path: Path, human_counts: pd.DataFrame
) -> tuple[pd.DataFrame, list[tuple[str, int]]]:
    resolutions = pd.read_csv(path)
    if resolutions["final_feature_value"].isna().any():
        raise ValueError("V2 has missing final adjudicator decisions")
    pair_counts = resolutions.groupby(["word_normalized", "feature_id"]).size()
    if pair_counts.max() != 1:
        raise ValueError("V2 contains duplicate word-feature decisions")

    matrix = (
        resolutions.assign(value=resolutions["final_feature_value"].gt(0).astype(np.int8))
        .pivot(index="word_normalized", columns="feature_id", values="value")
        .reindex(index=human_counts.index, columns=range(human_counts.shape[1]))
    )
    missing_mask = matrix.isna()
    missing_pairs = [
        (str(matrix.index[row]), int(matrix.columns[column]))
        for row, column in zip(*np.where(missing_mask.values))
    ]
    matrix = matrix.fillna(0).astype(np.int8)
    matrix.columns = human_counts.columns
    human_task_names = human_counts.columns[
        human_counts.gt(3).sum(axis=0).gt(3)
    ]
    human_schema_matrix = matrix.loc[:, human_task_names]
    # V2 is an applicability judgment over the already selected human schema.
    # Per the study decision, every nonzero finalized value is positive and no
    # second rater/object threshold is applied. Empty tasks cannot be sampled.
    retained = human_schema_matrix.loc[:, human_schema_matrix.sum(axis=0).gt(0)]
    return retained, missing_pairs


def load_v3_matrix(path: Path, object_order: Sequence[str]) -> pd.DataFrame:
    matrix = pd.read_csv(path, index_col=0)
    matrix = matrix.reindex(index=list(object_order))
    if matrix.isna().any().any():
        raise ValueError(f"V3 matrix at {path} does not match the human object order")
    values = set(np.unique(matrix.values).tolist())
    if not values.issubset({0, 1}):
        raise ValueError(f"V3 matrix at {path} is not binary: {sorted(values)}")
    return matrix.astype(np.int8)


def load_v4_matrix(path: Path, object_order: Sequence[str]) -> pd.DataFrame:
    """Load a V4 binary matrix and apply the unchanged strict context rule."""
    matrix = load_v3_matrix(path, object_order)
    retained = matrix.sum(axis=0).gt(3)
    matrix = matrix.loc[:, retained]
    if matrix.empty:
        raise ValueError(f"V4 matrix at {path} has no contexts with >3 positives")
    return matrix


class CICODataset(Dataset):
    """Episode sampler copied from the released ISC-CI data pipeline."""

    def __init__(
        self,
        matrix: pd.DataFrame,
        n_sequences: int = 1024,
        n_shot: Sequence[int] = (1, 2),
    ) -> None:
        self.df = matrix
        self.n_sequences = n_sequences
        self.n = list(n_shot)
        self.max_n = max(self.n)
        self.dense_fts = np.concatenate(
            [self.df.values, np.zeros((1, self.df.shape[1]))], axis=0
        ).astype("float32")
        self.task_to_idxs: dict[int, dict[str, np.ndarray]] = {}
        for task_idx in range(self.df.shape[1]):
            self.task_to_idxs[task_idx] = {
                "positive": np.where(self.dense_fts[:, task_idx] == 1)[0],
                "negative": np.where(self.dense_fts[:, task_idx] == 0)[0],
            }

    def __len__(self) -> int:
        return self.n_sequences

    def __getitem__(self, _: int) -> dict[str, np.ndarray]:
        task_idx = int(np.random.choice(range(self.df.shape[1])))
        n = int(np.random.choice(self.n))
        negative_idxs = np.random.permutation(
            self.task_to_idxs[task_idx]["negative"]
        )
        positive_idxs = np.random.permutation(
            self.task_to_idxs[task_idx]["positive"]
        )
        support_idxs = positive_idxs[:n]
        query_idxs = np.concatenate([negative_idxs, positive_idxs[n:]])
        if n < self.max_n:
            support_idxs = np.concatenate(
                [support_idxs, np.array([len(self.df)] * (self.max_n - n))]
            )
        if n > min(self.n):
            query_idxs = np.concatenate(
                [query_idxs, np.array([len(self.df)] * (n - min(self.n)))]
            )
        return {
            "support_x": support_idxs,
            "support_y": self.dense_fts[support_idxs, task_idx],
            "support_c": np.array([task_idx] * len(support_idxs)),
            "query_x": query_idxs,
            "query_y": self.dense_fts[query_idxs, task_idx],
            "query_c": np.array([task_idx] * len(query_idxs)),
        }
