#!/usr/bin/env python3
"""Quick eyeball spot-check of auto-approved V4 merge-review verdicts.

Run on della (login node is fine, this is just pandas over a CSV):

    cd /scratch/gpfs/JORDANAT/mg9965/FalseMemoryISC-CI/LLM_judge_item_expansion
    conda activate PromptControlText
    python spotcheck_v4_merge_review.py \
        --review artifacts/v4/discovery/candidate_merge_review.csv \
        --sample 25 --seed 0
"""
from __future__ import annotations

import argparse
import json

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", required=True)
    parser.add_argument("--sample", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    review = pd.read_csv(args.review, dtype=str).fillna("")
    auto = review.loc[review["reviewer"].eq("automated:embedding_threshold")].copy()
    auto["members"] = auto["member_phrases"].map(json.loads)
    auto["n_members"] = auto["members"].map(len)

    print(f"Total rows: {len(review)}")
    print(f"Auto-approved (pass): {len(auto)}")
    print(f"Human-reviewed (any verdict): {len(review) - len(auto)}")
    print()
    print("Cluster-size distribution among auto-approved merges:")
    print(auto["n_members"].value_counts().sort_index().to_string())
    print()

    print(f"=== Random sample ({min(args.sample, len(auto))} rows) ===")
    sample = auto.sample(n=min(args.sample, len(auto)), random_state=args.seed)
    for _, row in sample.iterrows():
        print(f"[{row['merge_candidate_id']}] ({row['n_members']} members)")
        for phrase in row["members"]:
            print(f"    - {phrase}")
        print()

    print("=== 10 largest auto-approved clusters (highest merge risk) ===")
    largest = auto.sort_values("n_members", ascending=False).head(10)
    for _, row in largest.iterrows():
        print(f"[{row['merge_candidate_id']}] ({row['n_members']} members)")
        for phrase in row["members"]:
            print(f"    - {phrase}")
        print()


if __name__ == "__main__":
    main()
