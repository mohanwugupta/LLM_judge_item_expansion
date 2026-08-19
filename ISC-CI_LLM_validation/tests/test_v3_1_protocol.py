import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_v3_1_changes_protocol_id_but_not_analysis_parameters() -> None:
    v3 = json.loads((ROOT / "configs" / "v3_validation.json").read_text())
    v3_1 = json.loads((ROOT / "configs" / "v3_1_validation.json").read_text())
    assert v3["protocol_version"] != v3_1["protocol_version"]
    assert {key: value for key, value in v3.items() if key != "protocol_version"} == {
        key: value for key, value in v3_1.items() if key != "protocol_version"
    }


def test_v3_1_manual_review_records_every_verdict() -> None:
    review = pd.read_csv(ROOT / "configs" / "v3_1_consolidation_manual_review.csv")
    assert set(review["verdict"]) == {"pass", "reject"}
    assert review["cluster_id"].is_unique
    assert review["review_note"].notna().all()
