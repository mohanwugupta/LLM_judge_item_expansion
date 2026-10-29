from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from iscci_validation.dataio import matrix_sha256

CONDITIONS = {
    "human": 385,
    "random_human": 385,
    "v2": 382,
    "v3_A": 94,
    "v3_B": 115,
    "v3_C": 121,
    "v3_1_A": 88,
    "v3_1_B": 175,
    "v3_1_C": 128,
    "v4_b_locked_v2": 175,
    "v4_b_calibrated": 175,
    "v4_ensemble_locked_v2": 87315,
    "v4_ensemble_calibrated": 80699,
    "v4_ensemble_source_only": 2356,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def matrix_path(root: Path, condition: str) -> Path:
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition: {condition}")
    if condition == "random_human":
        return (
            root
            / "artifacts/random_feature_control_20261028/validation/data_matrices/random_human.csv"
        )
    return root / f"artifacts/validation_v4_20261002/data_matrices/{condition}.csv"


def load_matrix(path: Path, object_order=None) -> pd.DataFrame:
    # Read row-wise: pandas' wide CSV parser is expensive for 87,315 columns.
    with path.open(newline="") as stream:
        rows = csv.reader(stream)
        header = next(rows)
        feature_ids = header[1:]
        if not feature_ids or len(set(feature_ids)) != len(feature_ids):
            raise ValueError(f"Missing or duplicate feature IDs: {path}")
        names, values = [], []
        for row in rows:
            if len(row) != len(header):
                raise ValueError(f"Malformed matrix row: {path}")
            names.append(row[0])
            numeric = np.asarray(row[1:], dtype=np.float32)
            if not np.isin(numeric, [0, 1]).all():
                raise ValueError(f"Matrix must be binary: {path}")
            values.append(numeric.astype(np.int8))
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate object IDs: {path}")
    if object_order is not None and names != list(object_order):
        raise ValueError(f"Object identity/order mismatch: {path}")
    if not names:
        raise ValueError(f"Empty matrix: {path}")
    return pd.DataFrame(np.stack(values), index=names, columns=feature_ids)


def inventory(root: Path) -> dict:
    records, objects, human = {}, None, None
    for condition, expected in CONDITIONS.items():
        path = matrix_path(root, condition)
        before = sha256(path)
        matrix = load_matrix(path, objects)
        if objects is None:
            objects, human = list(matrix.index), matrix
        if matrix.shape != (293, expected):
            raise ValueError(
                f"Wrong retained dimensions for {condition}: {matrix.shape}"
            )
        if condition == "random_human":
            if (
                list(matrix.columns) != list(human.columns)
                or not np.array_equal(matrix.sum(axis=0), human.sum(axis=0))
                or int(matrix.values.sum()) != 8056
            ):
                raise ValueError("Fixed random control identity/prevalence mismatch")
        if before != sha256(path):
            raise ValueError(f"Source changed during inventory: {path}")
        records[condition] = {
            "path": str(path.resolve()),
            "file_sha256": before,
            "matrix_sha256": matrix_sha256(matrix),
            "objects": len(matrix),
            "features": matrix.shape[1],
            "positives": int(matrix.values.sum()),
            "object_ids": objects,
            "feature_ids": list(matrix.columns),
            "prevalence": matrix.sum(axis=0).astype(int).tolist(),
        }
    return records
