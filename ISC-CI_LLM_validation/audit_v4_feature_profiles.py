#!/usr/bin/env python3
"""Describe duplicate membership profiles without changing any training matrix."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from iscci_validation.provenance import sha256_file, write_json


ROOT = Path(__file__).resolve().parent
MATRICES = ROOT / "artifacts/v4_matrices"
OUTPUT = ROOT / "reports/v4_20261002"


def profile_summary(name, matrix):
    retained = matrix[:, matrix.sum(axis=0) > 3]
    _, counts = np.unique(np.packbits(retained.T, axis=1), axis=0, return_counts=True)
    return {
        "condition": name,
        "retained_contexts": retained.shape[1],
        "distinct_object_membership_profiles": len(counts),
        "extra_columns_with_repeated_profile": int((counts - 1).sum()),
        "largest_profile_multiplicity": int(counts.max()) if len(counts) else 0,
        "fraction_extra_repeated_columns": (
            float((counts - 1).sum() / retained.shape[1])
            if retained.shape[1]
            else np.nan
        ),
        "positive_cells": int(retained.sum()),
        "zero_feature_objects": int((retained.sum(axis=1) == 0).sum()),
    }


def main():
    provenance_path = MATRICES / "cell_provenance.parquet"
    metadata = json.loads((MATRICES / "matrix_manifest.json").read_text())
    if sha256_file(provenance_path) != metadata["cell_provenance_sha256"]:
        raise ValueError("Cell-provenance hash mismatch")
    reader = pq.ParquetFile(provenance_path)
    fields = [
        "candidate_id",
        "target_word",
        "resolved_binary_locked_v2",
        "resolved_binary_calibrated",
        "resolved_binary_source_only",
    ]
    matrices, candidates, words = None, None, None
    for batch in reader.iter_batches(batch_size=1000000, columns=fields):
        candidate, word = batch.column(0), batch.column(1)
        if matrices is None:
            candidates, words = candidate.dictionary, word.dictionary
            matrices = [
                np.zeros((len(words), len(candidates)), dtype=np.bool_)
                for _ in range(3)
            ]
        if not candidate.dictionary.equals(candidates) or not word.dictionary.equals(
            words
        ):
            raise ValueError("Parquet dictionary order changed between batches")
        rows, columns = word.indices.to_numpy(), candidate.indices.to_numpy()
        for i, matrix in enumerate(matrices):
            matrix[rows, columns] = batch.column(i + 2).to_numpy().astype(bool)
    names = [
        "v4_ensemble_locked_v2",
        "v4_ensemble_calibrated",
        "v4_ensemble_source_only",
    ]
    summaries = [profile_summary(name, matrix) for name, matrix in zip(names, matrices)]
    inventory = pd.read_csv(MATRICES / "context_inventory_comparison.csv").set_index(
        "matrix"
    )
    for summary in summaries:
        if summary["retained_contexts"] != int(
            inventory.loc[summary["condition"], "candidate_count_after_strict_gt_3"]
        ):
            raise ValueError(
                "Profile audit disagrees with the frozen retention inventory"
            )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(summaries)
    table.to_csv(OUTPUT / "feature_profile_redundancy.csv", index=False)
    write_json(
        OUTPUT / "feature_profile_audit_manifest.json",
        {
            "complete": True,
            "cell_provenance_sha256": metadata["cell_provenance_sha256"],
            "implementation_sha256": sha256_file(Path(__file__)),
            "policy": "Descriptive only; no duplicate profiles removed, reweighted, or merged",
            "caveat": "Same membership on these 293 objects does not establish proposition equivalence",
        },
    )
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
