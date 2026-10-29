from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


def parse_output_ids(line: str) -> list[int]:
    """Read a positive legacy semantic pattern without discarding its last ID."""
    match = re.search(r"\bt:\s*([0-9]+(?:\s+[0-9]+)*)\s*;\s*$", line)
    if match is None or len(re.findall(r"\bt:", line)) != 1:
        raise ValueError("Expected nonnegative output IDs followed by one semicolon")
    return [int(token) for token in match.group(1).split()]


class TaskDataset:
    """Positive object-task pairs and the investigator-approved full targets.

    The released text file preselected positive pairs, but its loader accidentally
    discarded the final output ID. New training uses the retained matrix directly:
    y[o,t,f] = F[o,f] * (feature_task[f] == t), without that parsing error.
    """

    def __init__(
        self,
        matrix: pd.DataFrame,
        assignments: Mapping[str, int],
        task_count: int,
    ) -> None:
        if type(task_count) is not int or task_count < 1:
            raise ValueError("task_count must be a positive integer")
        if matrix.empty or not matrix.index.is_unique or not matrix.columns.is_unique:
            raise ValueError(
                "Matrix must be nonempty with unique object and feature IDs"
            )
        if set(assignments) != set(matrix.columns):
            raise ValueError("Assignments must cover exactly the retained feature IDs")
        if any(
            type(t) is not int or not 0 <= t < task_count for t in assignments.values()
        ):
            raise ValueError(
                "Assignments contain missing, unknown, or noninteger task IDs"
            )
        if not np.isin(matrix.values, [0, 1]).all():
            raise ValueError("Semantic targets require an unchanged binary matrix")
        self.object_ids = list(matrix.index)
        self.feature_ids = list(matrix.columns)
        self.task_count = task_count
        self.values = np.array(matrix.values, dtype=np.int8, copy=True)
        self.feature_tasks = np.array(
            [assignments[f] for f in matrix.columns], dtype=np.int64
        )
        self.values.flags.writeable = False
        self.feature_tasks.flags.writeable = False
        positive_pairs = np.zeros((len(matrix), task_count), dtype=bool)
        for task in range(task_count):
            positive_pairs[:, task] = self.values[:, self.feature_tasks == task].any(
                axis=1
            )
        self.pairs = np.argwhere(positive_pairs)
        self.pairs.flags.writeable = False
        self.inactive_tasks = np.flatnonzero(~positive_pairs.any(axis=0)).tolist()
        if not len(self.pairs):
            raise ValueError("No positive object-task patterns available")

    def __len__(self) -> int:
        return len(self.pairs)

    def _indices(self, indices: Sequence[int]) -> np.ndarray:
        result = np.asarray(indices)
        if result.size == 0:
            return np.empty(0, dtype=np.int64)
        if (
            result.ndim != 1
            or result.dtype.kind not in "iu"
            or (result < 0).any()
            or (result >= len(self)).any()
        ):
            raise ValueError("Invalid object-task pattern indices")
        return result.astype(np.int64)

    def inputs(self, indices: Sequence[int]) -> list[torch.Tensor]:
        pairs = torch.tensor(self.pairs[self._indices(indices)], dtype=torch.long)
        return [
            F.one_hot(pairs[:, 0], len(self.object_ids)).float(),
            F.one_hot(pairs[:, 1], self.task_count).float(),
        ]

    def targets(
        self, indices: Sequence[int], start: int = 0, stop: int | None = None
    ) -> torch.Tensor:
        stop = len(self.feature_ids) if stop is None else stop
        if not 0 <= start <= stop <= len(self.feature_ids):
            raise ValueError("Invalid output-feature slice")
        pairs = self.pairs[self._indices(indices)]
        targets = self.values[pairs[:, 0], start:stop] * (
            self.feature_tasks[None, start:stop] == pairs[:, 1, None]
        )
        return torch.from_numpy(targets.astype(np.float32))

    def exposure(self) -> dict:
        feature_counts = np.bincount(self.feature_tasks, minlength=self.task_count)
        positives = self.values.sum(axis=0)
        task_positives = [
            int(positives[self.feature_tasks == task].sum())
            for task in range(self.task_count)
        ]
        object_counts = np.bincount(self.pairs[:, 0], minlength=len(self.object_ids))
        return {
            "valid_patterns": len(self),
            "pair_policy": "positive object-task pairs before target construction",
            "target_policy": "full-vector F[o,f] * (feature_task[f] == t); no discarded IDs",
            "inactive_task_policy": "preserve task IDs; sample no empty-positive pairs",
            "inactive_tasks": self.inactive_tasks,
            "features_per_task": feature_counts.tolist(),
            "positive_associations_per_task": task_positives,
            "patterns_per_task": np.bincount(
                self.pairs[:, 1], minlength=self.task_count
            ).tolist(),
            "patterns_per_object": object_counts.tolist(),
            "objects_without_patterns": [
                self.object_ids[i] for i in np.flatnonzero(object_counts == 0)
            ],
        }
