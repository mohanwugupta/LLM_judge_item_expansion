import json
import sys

import numpy as np
import pandas as pd
import pytest

from build_v4_matrices import main as build_matrices_main
from leuven_expansion.v4 import candidate_inventory_hash, sha256_file


def test_matrix_build_preserves_order_rules_completion_and_provenance(tmp_path, monkeypatch):
    pytest.importorskip("pyarrow")
    words = ["dog", "cat", "bird", "plane", "rock"]
    candidate_ids = [f"v4_{index:03d}" for index in range(175)]
    bank = pd.DataFrame(
        {
            "candidate_index": range(175),
            "candidate_id": candidate_ids,
            "canonical_feature_text": [f"feature {index}" for index in range(175)],
            "source_words": [json.dumps(["dog"])] * 175,
            "fixed_v3_1_b_order": range(175),
        }
    )
    bank["candidate_inventory_hash"] = candidate_inventory_hash(bank)
    bank_path = tmp_path / "candidate_bank.csv"
    fixed_path = tmp_path / "candidate_bank_v3_1_b_175.csv"
    bank.to_csv(bank_path, index=False)
    bank.to_csv(fixed_path, index=False)
    human_path = tmp_path / "human.csv"
    pd.DataFrame({"word": words, "human feature": [4, 4, 4, 4, 0]}).to_csv(
        human_path, index=False
    )
    rows = []
    for feature_index, candidate_id in enumerate(candidate_ids):
        for word_index, word in enumerate(words):
            value = 1.0 if word_index < 4 and feature_index == 0 else 0.0
            if feature_index == 1 and word == "cat":
                value = 1.0
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "target_word": word,
                    "resolved_value": value,
                    "confidence": 0.9,
                    "ambiguous": False,
                    "resolution_method": "unanimous",
                    "adjudicated": False,
                    "needs_human_audit": False,
                }
            )
    resolved_path = tmp_path / "resolved_feature_values.csv"
    pd.DataFrame(rows).to_csv(resolved_path, index=False)
    manifest = {
        "complete": True,
        "candidate_inventory_hash": bank["candidate_inventory_hash"].iloc[0],
        "candidate_bank_sha256": sha256_file(bank_path),
        "protocol_hash": "protocol",
    }
    (tmp_path / "judgment_manifest.json").write_text(json.dumps(manifest))
    threshold_path = tmp_path / "judgment_threshold.json"
    threshold_path.write_text(
        json.dumps(
            {
                "selected_rule": {"operator": "ge", "value": 1.0},
                "calibration_hash": "calibration",
            }
        )
    )
    output = tmp_path / "matrices"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_v4_matrices.py",
            "--candidate-bank",
            str(bank_path),
            "--resolved-values",
            str(resolved_path),
            "--threshold",
            str(threshold_path),
            "--output-dir",
            str(output),
            "--human-features",
            str(human_path),
        ],
    )
    build_matrices_main()
    raw = pd.read_csv(output / "v4_ensemble_raw.csv", index_col=0)
    locked = pd.read_csv(output / "v4_ensemble_locked_v2.csv", index_col=0)
    calibrated = pd.read_csv(output / "v4_ensemble_calibrated.csv", index_col=0)
    source_only = pd.read_csv(output / "v4_ensemble_source_only.csv", index_col=0)
    assert raw.index.tolist() == words
    assert raw.columns.tolist() == locked.columns.tolist() == calibrated.columns.tolist()
    assert np.array_equal(locked.to_numpy(), raw.to_numpy() > 0)
    assert (source_only.to_numpy() <= calibrated.to_numpy()).all()
    inventory = pd.read_csv(output / "context_inventory_comparison.csv")
    row = inventory.loc[inventory["matrix"].eq("v4_ensemble_calibrated")].iloc[0]
    assert row["candidate_count_after_strict_gt_3"] == 1
    provenance = pd.read_parquet(output / "cell_provenance.parquet")
    assert len(provenance) == 875
    assert provenance.loc[
        provenance["candidate_id"].eq("v4_001")
        & provenance["target_word"].eq("cat"),
        "source_generated",
    ].item() is False

