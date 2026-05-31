"""
leuven_expansion/expand_feature_matrix.py

Production CLI: expand the Leuven feature matrix to cover new DRM items.
Each new word × Leuven-feature pair is judged independently and atomically.

Usage
-----
python -m leuven_expansion.expand_feature_matrix \
  --items data/drm_items_to_classify.csv \
  --leuven-features data/leuven_combined_features_consolidated.csv \
  --singular-plural data/leuven_singular_to_plural.csv \
  --job-id drm_atomic_leuven_feature_expansion_qwen \
  --output-dir artifacts/leuven_feature_expansion/drm_atomic_leuven_feature_expansion_qwen \
  --model Qwen2.5-72B-Instruct \
  --base-url http://localhost:8000/v1 \
  --max-workers 64 \
  --resume
"""
from __future__ import annotations

import argparse
import json
import logging
import pathlib
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

from leuven_expansion.feature_schema import load_leuven_feature_schema
from leuven_expansion.feature_prompts import load_default_prompts
from leuven_expansion.normalize import normalize_word, apply_singular_to_plural
from leuven_expansion.run_jobs import run_atomic_jobs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _load_new_items(
    items_csv: str | pathlib.Path,
    word_col: str = "word",
) -> List[str]:
    """Load DRM items from CSV, return list of original word strings."""
    df = pd.read_csv(items_csv)
    if word_col not in df.columns:
        # Try first column
        word_col = df.columns[0]
    return df[word_col].dropna().tolist()


def _dedup_items(
    new_words: List[str],
    existing_words: set,
) -> List[tuple[str, str]]:
    """
    Deduplicate new words against existing Leuven items.
    Returns list of (word_original, word_normalized) for truly new items.
    Duplicate normalized words are classified only once.
    """
    seen_normalized: set = set()
    result = []
    for w in new_words:
        nw = normalize_word(w)
        if nw in existing_words:
            logger.debug("Skipping existing Leuven item: %r", nw)
            continue
        if nw in seen_normalized:
            logger.debug("Skipping duplicate normalized word: %r", nw)
            continue
        seen_normalized.add(nw)
        result.append((w, nw))
    return result


def build_expanded_matrix(
    leuven_features_csv: str | pathlib.Path,
    resolutions_csv: str | pathlib.Path,
    new_word_pairs: List[tuple[str, str]],  # [(word_original, word_normalized)]
    output_dir: pathlib.Path,
    job_id: str,
) -> pathlib.Path:
    """
    Combine original Leuven matrix with LLM-judged values for new words
    into a single expanded_feature_matrix.csv.
    """
    schema = load_leuven_feature_schema(leuven_features_csv)
    feature_columns = schema["feature_columns"]
    item_column = schema["item_column"]

    leuven_df = pd.read_csv(leuven_features_csv)
    leuven_df["word_normalized"] = leuven_df[item_column].apply(normalize_word)
    leuven_df["word_original"] = leuven_df[item_column]
    leuven_df["source"] = "leuven_original"
    leuven_df["in_original_leuven"] = True
    leuven_df["in_llm_expansion"] = False
    leuven_df["feature_completion_method"] = "human_leuven"

    # Load LLM resolutions for new words
    res_df = pd.read_csv(resolutions_csv)
    new_rows = []
    nw_set = {nw for _, nw in new_word_pairs}

    for word_original, word_normalized in new_word_pairs:
        word_res = res_df[res_df["word_normalized"] == word_normalized]

        feat_vec = {fc: 0.0 for fc in feature_columns}
        confidence_vals = []
        n_adjudicated = 0
        n_ambiguous = 0
        n_low_conf = 0
        needs_audit = False

        for _, row in word_res.iterrows():
            fid = int(row["feature_id"])
            if fid < len(feature_columns):
                fcol = feature_columns[fid]
                val = row.get("final_feature_value")
                if val is not None and not pd.isna(val):
                    feat_vec[fcol] = float(val)
            if row.get("adjudicated"):
                n_adjudicated += 1
            if row.get("needs_human_audit"):
                needs_audit = True

        n_positive = sum(1 for v in feat_vec.values() if v > 0)
        mean_conf = float(np.mean(confidence_vals)) if confidence_vals else 0.0

        row_dict = {
            "word": word_original,
            "word_original": word_original,
            "word_normalized": word_normalized,
            "source": "llm_expansion",
            "in_original_leuven": False,
            "in_llm_expansion": True,
            "feature_completion_method": "llm_atomic",
            "mean_feature_confidence": mean_conf,
            "n_positive_features": n_positive,
            "n_adjudicated_features": n_adjudicated,
            "n_low_confidence_features": n_low_conf,
            "n_ambiguous_features": n_ambiguous,
            "needs_human_audit": needs_audit,
            **feat_vec,
        }
        new_rows.append(row_dict)

    new_df = pd.DataFrame(new_rows)

    # Standardize leuven_df
    meta_cols = [
        "word", "word_original", "word_normalized", "source",
        "in_original_leuven", "in_llm_expansion", "feature_completion_method",
        "mean_feature_confidence", "n_positive_features",
        "n_adjudicated_features", "n_low_confidence_features",
        "n_ambiguous_features", "needs_human_audit",
    ]
    for mc in meta_cols:
        if mc not in leuven_df.columns:
            leuven_df[mc] = None
    if "word" not in leuven_df.columns:
        leuven_df["word"] = leuven_df[item_column]

    all_cols = meta_cols + feature_columns
    leuven_subset = leuven_df.reindex(columns=all_cols, fill_value=None)
    new_subset = new_df.reindex(columns=all_cols, fill_value=0.0)

    expanded = pd.concat([leuven_subset, new_subset], ignore_index=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "expanded_feature_matrix.csv"
    expanded.to_csv(out_path, index=False)
    logger.info("Expanded matrix written to %s (%d rows)", out_path, len(expanded))
    return out_path


def join_occurrences_with_features(
    expanded_matrix_csv: str | pathlib.Path,
    word_occurrences_csv: str | pathlib.Path,
    output_dir: pathlib.Path,
) -> pathlib.Path:
    """
    Left-join drm_word_occurrences_long.csv with the expanded feature matrix
    on word_key (occurrences) ↔ word_normalized (expanded matrix).

    The result is drm_occurrences_with_features.csv — one row per DRM occurrence
    with all Leuven feature columns appended.  Occurrences for words that were
    not in the Leuven norms and could not be expanded will have NaN in every
    feature column and ``source=missing``.
    """
    exp_df = pd.read_csv(expanded_matrix_csv)
    occ_df = pd.read_csv(word_occurrences_csv)

    # Identify the Leuven feature columns in the expanded matrix
    non_feature_cols = {
        "word", "word_original", "word_normalized", "source",
        "in_original_leuven", "in_llm_expansion", "feature_completion_method",
        "mean_feature_confidence", "n_positive_features",
        "n_adjudicated_features", "n_low_confidence_features",
        "n_ambiguous_features", "needs_human_audit",
    }
    feature_cols = [c for c in exp_df.columns if c not in non_feature_cols]
    keep_from_exp = ["word_normalized", "source", "in_original_leuven",
                     "in_llm_expansion", "feature_completion_method",
                     "needs_human_audit"] + feature_cols

    exp_slim = exp_df[keep_from_exp].copy()
    exp_slim = exp_slim.rename(columns={"word_normalized": "word_key"})

    merged = occ_df.merge(exp_slim, on="word_key", how="left")

    # Mark rows whose word had no feature data at all
    merged["source"] = merged["source"].fillna("missing")

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "drm_occurrences_with_features.csv"
    merged.to_csv(out_path, index=False)
    n_covered = merged["word_key"].isin(exp_slim["word_key"]).sum()
    logger.info(
        "Occurrences join written to %s (%d rows, %d/%d occurrences covered)",
        out_path, len(merged), n_covered, len(merged),
    )
    return out_path


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Expand Leuven feature matrix with new DRM items."
    )
    p.add_argument("--items", required=True)
    p.add_argument("--leuven-features", required=True)
    p.add_argument("--singular-plural", default=None)
    p.add_argument("--word-occurrences", default=None,
                   help="drm_word_occurrences_long.csv — joined back with the "
                        "expanded feature matrix on word_key↔word_normalized "
                        "to produce drm_occurrences_with_features.csv")
    p.add_argument("--job-id", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model", default="Qwen2.5-72B-Instruct")
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--max-workers", type=int, default=16)
    p.add_argument("--resume", action="store_true")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()

    import importlib
    vllm_client_mod = importlib.import_module("vllm_client")
    VLLMClient = vllm_client_mod.VLLMClient

    client = VLLMClient(model_name=args.model, base_url=args.base_url)
    prompts = load_default_prompts()
    output_dir = pathlib.Path(args.output_dir)

    schema = load_leuven_feature_schema(args.leuven_features)
    feature_columns = schema["feature_columns"]
    existing_leuven_df = pd.read_csv(args.leuven_features)
    existing_words = set(
        normalize_word(w) for w in existing_leuven_df[schema["item_column"]].tolist()
    )

    new_items = _load_new_items(args.items)

    # Apply singular-to-plural mapping if provided
    if args.singular_plural:
        sp_df = pd.read_csv(args.singular_plural)
        sp_map = apply_singular_to_plural(new_items, sp_df)
        new_items = [sp_map.get(normalize_word(w), w) for w in new_items]

    new_word_pairs = _dedup_items(new_items, existing_words)
    logger.info("New unique words to expand: %d", len(new_word_pairs))

    # Build all word × feature pairs
    pairs: list[Dict] = []
    for word_original, word_normalized in new_word_pairs:
        for fid, fcol in enumerate(feature_columns):
            pairs.append({
                "word_original": word_original,
                "word_normalized": word_normalized,
                "feature_id": fid,
                "feature_text": fcol,
            })

    logger.info("Total atomic pairs to judge: %d", len(pairs))

    run_atomic_jobs(
        job_id=args.job_id,
        pairs=pairs,
        prompts=prompts,
        client=client,
        model=args.model,
        output_dir=output_dir,
        max_workers=args.max_workers,
        resume=args.resume,
    )

    # Build expanded matrix from resolutions
    resolutions_csv = output_dir / "feature_resolutions.csv"
    if resolutions_csv.exists():
        expanded_matrix_csv = build_expanded_matrix(
            leuven_features_csv=args.leuven_features,
            resolutions_csv=resolutions_csv,
            new_word_pairs=new_word_pairs,
            output_dir=output_dir,
            job_id=args.job_id,
        )
        # Join-back: annotate every DRM occurrence with Leuven feature values
        if args.word_occurrences:
            join_occurrences_with_features(
                expanded_matrix_csv=expanded_matrix_csv,
                word_occurrences_csv=args.word_occurrences,
                output_dir=output_dir,
            )
    else:
        logger.error("No feature_resolutions.csv found; cannot build expanded matrix.")
