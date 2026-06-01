"""
leuven_expansion/validate_features.py

Validation CLI and logic for Leuven feature reconstruction.

Modes
-----
cell_holdout  : hold out individual word × feature cells
word_holdout  : hold out entire Leuven words

Usage
-----
python -m leuven_expansion.validate_features \
  --mode cell_holdout \
  --leuven-features data/leuven_combined_features_consolidated.csv \
  --leuven-categories data/leuven_categories.csv \
  --job-id leuven_atomic_cell_validation_qwen \
  --output-dir artifacts/leuven_feature_expansion/leuven_atomic_cell_validation_qwen \
  --model Qwen2.5-72B-Instruct \
  --base-url http://localhost:8000/v1 \
  --test-size 0.20 \
  --seed 42 \
  --max-workers 64 \
  --resume
"""
from __future__ import annotations

import argparse
import json
import logging
import pathlib
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_fscore_support,
)
from scipy.stats import pearsonr, spearmanr

from leuven_expansion.feature_schema import load_leuven_feature_schema, get_feature_text
from leuven_expansion.feature_prompts import load_default_prompts
from leuven_expansion.category_metadata import (
    load_categories,
    get_category_map,
    stratified_word_split,
)
from leuven_expansion.normalize import normalize_word
from leuven_expansion.run_jobs import run_atomic_jobs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _load_leuven_matrix(features_csv: str | pathlib.Path, item_column: str | None = None):
    df = pd.read_csv(features_csv)
    if item_column is None:
        item_column = df.columns[0]
    df["word_normalized"] = df[item_column].apply(normalize_word)
    return df


def _stratified_cell_split(
    df: pd.DataFrame,
    feature_columns: List[str],
    test_fraction: float = 0.20,
    seed: int = 42,
) -> Tuple[List[Tuple], List[Tuple]]:
    """
    Stratified cell-level holdout. Returns (train_cells, test_cells).
    Each cell is (word_normalized, feature_id, feature_text, value).
    """
    random.seed(seed)
    positive_cells = []
    zero_cells = []

    for _, row in df.iterrows():
        wn = row["word_normalized"]
        for fid, fcol in enumerate(feature_columns):
            val = float(row[fcol])
            cell = (wn, fid, fcol, val)
            if val > 0:
                positive_cells.append(cell)
            else:
                zero_cells.append(cell)

    def split_list(lst):
        random.shuffle(lst)
        n = max(1, round(len(lst) * test_fraction))
        return lst[n:], lst[:n]

    train_pos, test_pos = split_list(positive_cells)
    train_zero, test_zero = split_list(zero_cells)
    return train_pos + train_zero, test_pos + test_zero


def _compute_cell_metrics(
    y_true: List[float],
    y_pred: List[float],
) -> Dict:
    """Compute cell-level prediction metrics."""
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    y_true_bin = (y_true_arr > 0).astype(int)
    y_pred_bin = (y_pred_arr > 0).astype(int)

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true_bin, y_pred_bin, average="binary", zero_division=0
    )

    mae = float(np.mean(np.abs(y_true_arr - y_pred_arr)))
    rmse = float(np.sqrt(np.mean((y_true_arr - y_pred_arr) ** 2)))

    metrics: Dict = {
        "binary_accuracy": float(np.mean(y_true_bin == y_pred_bin)),
        "positive_precision": float(prec),
        "positive_recall": float(rec),
        "positive_f1": float(f1),
        "MAE_0_4": round(mae, 4),
        "RMSE_0_4": round(rmse, 4),
        "n_cells": len(y_true),
        "n_positive_cells": int(y_true_bin.sum()),
    }

    # AUPRC / ROC-AUC (only if both classes present)
    if len(set(y_true_bin)) == 2:
        try:
            metrics["AUPRC"] = round(
                float(average_precision_score(y_true_bin, y_pred_arr)), 4
            )
            metrics["ROC_AUC"] = round(
                float(roc_auc_score(y_true_bin, y_pred_arr)), 4
            )
        except Exception:
            pass

    return metrics


def _compute_word_metrics(
    gold_vec: np.ndarray,
    pred_vec: np.ndarray,
    top_k: int = 10,
) -> Dict:
    """Compute word-level vector reconstruction metrics."""
    cos = float(
        np.dot(gold_vec, pred_vec)
        / (np.linalg.norm(gold_vec) * np.linalg.norm(pred_vec) + 1e-12)
    )
    pearson_r, _ = pearsonr(gold_vec, pred_vec)
    spearman_r, _ = spearmanr(gold_vec, pred_vec)

    gold_top = set(np.argsort(gold_vec)[::-1][:top_k])
    pred_top = set(np.argsort(pred_vec)[::-1][:top_k])
    top10_recall = len(gold_top & pred_top) / len(gold_top) if gold_top else 0.0

    return {
        "cosine_similarity": round(float(cos), 4),
        "pearson_r": round(float(pearson_r), 4),
        "spearman_r": round(float(spearman_r), 4),
        f"top_{top_k}_feature_recall": round(float(top10_recall), 4),
    }


def _build_pairs_for_cells(
    cells: List[Tuple],
    df: pd.DataFrame,
    feature_columns: List[str],
    item_column: str = "word_normalized",
) -> List[Dict]:
    """Convert cell tuples to pair dicts for the job runner."""
    pairs = []
    for word_normalized, fid, fcol, _ in cells:
        orig_rows = df[df["word_normalized"] == word_normalized]
        word_original = (
            orig_rows.iloc[0][item_column] if len(orig_rows) > 0 else word_normalized
        )
        pairs.append({
            "word_original": word_original,
            "word_normalized": word_normalized,
            "feature_id": fid,
            "feature_text": fcol,
        })
    return pairs


def run_cell_holdout(
    *,
    features_csv: str | pathlib.Path,
    categories_csv: Optional[str | pathlib.Path],
    job_id: str,
    output_dir: pathlib.Path,
    client,
    model: str,
    prompts: Dict[str, str],
    test_size: float = 0.20,
    seed: int = 42,
    max_workers: int = 16,
    resume: bool = True,
) -> Dict:
    """Run cell-level holdout validation."""
    schema = load_leuven_feature_schema(features_csv)
    df = _load_leuven_matrix(features_csv)

    train_cells, test_cells = _stratified_cell_split(
        df, schema["feature_columns"], test_fraction=test_size, seed=seed
    )
    logger.info(
        "Cell holdout: %d train cells, %d test cells", len(train_cells), len(test_cells)
    )

    pairs = _build_pairs_for_cells(test_cells, df, schema["feature_columns"], item_column=schema["item_column"])
    run_atomic_jobs(
        job_id=job_id,
        pairs=pairs,
        prompts=prompts,
        client=client,
        model=model,
        output_dir=output_dir,
        max_workers=max_workers,
        resume=resume,
    )

    # Load resolutions and compare to gold
    resolutions_csv = output_dir / "feature_resolutions.csv"
    if not resolutions_csv.exists():
        logger.warning("No resolutions found; skipping metric computation.")
        return {}

    res_df = pd.read_csv(resolutions_csv)

    gold_map = {
        (cell[0], cell[1]): cell[3] for cell in test_cells
    }

    y_true, y_pred = [], []
    for _, row in res_df.iterrows():
        key = (str(row["word_normalized"]), int(row["feature_id"]))
        if key in gold_map and pd.notna(row["final_feature_value"]):
            y_true.append(gold_map[key])
            y_pred.append(float(row["final_feature_value"]))

    if not y_true:
        logger.warning("No matched predictions for metric computation.")
        return {}

    metrics = _compute_cell_metrics(y_true, y_pred)
    metrics["mode"] = "cell_holdout"
    metrics["test_size"] = test_size
    metrics["seed"] = seed

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "feature_validation_metrics.json").write_text(
        json.dumps(metrics, indent=2)
    )
    logger.info("Cell holdout metrics: %s", metrics)
    return metrics


def run_word_holdout(
    *,
    features_csv: str | pathlib.Path,
    categories_csv: Optional[str | pathlib.Path],
    job_id: str,
    output_dir: pathlib.Path,
    client,
    model: str,
    prompts: Dict[str, str],
    test_size: float = 0.20,
    seed: int = 42,
    max_workers: int = 16,
    resume: bool = True,
) -> Dict:
    """Run word-level holdout validation."""
    schema = load_leuven_feature_schema(features_csv)
    df = _load_leuven_matrix(features_csv)
    feature_columns = schema["feature_columns"]

    all_words = df["word_normalized"].tolist()

    category_map = {}
    if categories_csv:
        cat_df = load_categories(categories_csv)
        category_map = get_category_map(cat_df)

    _, test_words = stratified_word_split(
        all_words, category_map, test_fraction=test_size, seed=seed
    )
    test_word_set = set(test_words)
    logger.info("Word holdout: %d test words", len(test_word_set))

    test_df = df[df["word_normalized"].isin(test_word_set)]

    pairs: List[Dict] = []
    for _, row in test_df.iterrows():
        for fid, fcol in enumerate(feature_columns):
            pairs.append({
                "word_original": str(row.get(schema["item_column"], row["word_normalized"])),
                "word_normalized": str(row["word_normalized"]),
                "feature_id": fid,
                "feature_text": fcol,
            })

    logger.info("Word holdout: %d total pairs to judge", len(pairs))

    run_atomic_jobs(
        job_id=job_id,
        pairs=pairs,
        prompts=prompts,
        client=client,
        model=model,
        output_dir=output_dir,
        max_workers=max_workers,
        resume=resume,
    )

    # Load resolutions and compute word-level metrics
    resolutions_csv = output_dir / "feature_resolutions.csv"
    if not resolutions_csv.exists():
        logger.warning("No resolutions found; skipping metric computation.")
        return {}

    res_df = pd.read_csv(resolutions_csv)

    word_metrics_rows = []
    all_cos_sims = []

    for wn in test_word_set:
        word_rows = df[df["word_normalized"] == wn]
        if len(word_rows) == 0:
            continue
        gold_vec = np.array([float(word_rows.iloc[0][fcol]) for fcol in feature_columns])

        pred_rows = res_df[res_df["word_normalized"] == wn]
        pred_vec = np.zeros(len(feature_columns))
        for _, row in pred_rows.iterrows():
            fid = int(row["feature_id"])
            if row["final_feature_value"] is not None:
                pred_vec[fid] = float(row["final_feature_value"])

        wm = _compute_word_metrics(gold_vec, pred_vec)
        wm["word_normalized"] = wn
        word_metrics_rows.append(wm)
        all_cos_sims.append(wm["cosine_similarity"])

    if not word_metrics_rows:
        return {}

    word_df = pd.DataFrame(word_metrics_rows)
    word_df.to_csv(output_dir / "word_level_metrics.csv", index=False)

    summary = {
        "mode": "word_holdout",
        "n_test_words": len(test_word_set),
        "mean_cosine_similarity": round(float(np.mean(all_cos_sims)), 4),
        "median_cosine_similarity": round(float(np.median(all_cos_sims)), 4),
        "test_size": test_size,
        "seed": seed,
    }
    (output_dir / "feature_validation_metrics.json").write_text(
        json.dumps(summary, indent=2)
    )
    logger.info("Word holdout summary: %s", summary)
    return summary


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Validate Leuven feature reconstruction."
    )
    p.add_argument("--mode", required=True, choices=["cell_holdout", "word_holdout"])
    p.add_argument("--leuven-features", required=True)
    p.add_argument("--leuven-categories", default=None)
    p.add_argument("--job-id", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model", default="Qwen2.5-72B-Instruct")
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--test-size", type=float, default=0.20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-workers", type=int, default=16)
    p.add_argument("--resume", action="store_true")
    return p.parse_args(argv)


if __name__ == "__main__":
    import sys
    args = _parse_args()

    import importlib
    vllm_client_mod = importlib.import_module("vllm_client")
    VLLMClient = vllm_client_mod.VLLMClient

    client = VLLMClient(model_name=args.model, base_url=args.base_url)
    prompts = load_default_prompts()
    output_dir = pathlib.Path(args.output_dir)

    kwargs = dict(
        features_csv=args.leuven_features,
        categories_csv=args.leuven_categories,
        job_id=args.job_id,
        output_dir=output_dir,
        client=client,
        model=args.model,
        prompts=prompts,
        test_size=args.test_size,
        seed=args.seed,
        max_workers=args.max_workers,
        resume=args.resume,
    )

    if args.mode == "cell_holdout":
        run_cell_holdout(**kwargs)
    else:
        run_word_holdout(**kwargs)
