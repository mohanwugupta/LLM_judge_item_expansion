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


def _compute_verification_delta(
    resolutions: "pd.DataFrame",
    ground_truth: "pd.DataFrame",
    positive_threshold: float = 1.0,
    value_col: str = "feature_value",
) -> Dict:
    """
    Compute pre- and post-verification precision/recall/F1 and verification counts.

    Parameters
    ----------
    resolutions   : DataFrame with columns including
                    final_feature_value, pre_verification_feature_value,
                    positive_verification_status, word_normalized, feature_id
    ground_truth  : DataFrame with columns word_normalized, feature_id, feature_value
    positive_threshold : threshold above which a prediction is "positive"
    value_col     : column name in ground_truth for the gold value

    Returns
    -------
    dict with pre_precision, post_precision, pre_recall, post_recall,
    pre_f1, post_f1, n_candidate_positives, n_verified_retained, n_verified_rejected
    """
    merged = resolutions.merge(
        ground_truth[["word_normalized", "feature_id", value_col]],
        on=["word_normalized", "feature_id"],
        how="inner",
    )

    gold_pos = (merged[value_col] > 0).astype(int)

    # Pre-verification prediction (use pre_verification_feature_value if available)
    pre_col = "pre_verification_feature_value"
    if pre_col in merged.columns:
        pre_values = merged[pre_col].fillna(merged["final_feature_value"])
    else:
        pre_values = merged["final_feature_value"]
    pre_pred = (pre_values >= positive_threshold).astype(int)

    # Post-verification prediction (final_feature_value after verifier may have zeroed)
    post_pred = (merged["final_feature_value"] >= positive_threshold).astype(int)

    def _prf(pred):
        tp = int(((pred == 1) & (gold_pos == 1)).sum())
        fp = int(((pred == 1) & (gold_pos == 0)).sum())
        fn = int(((pred == 0) & (gold_pos == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else float("nan"))
        return precision, recall, f1

    pre_p, pre_r, pre_f = _prf(pre_pred)
    post_p, post_r, post_f = _prf(post_pred)

    status_col = "positive_verification_status"
    n_candidate = int(
        merged[status_col].isin(["retained", "rejected"]).sum()
        if status_col in merged.columns else 0
    )
    n_retained = int(
        (merged[status_col] == "retained").sum()
        if status_col in merged.columns else 0
    )
    n_rejected = int(
        (merged[status_col] == "rejected").sum()
        if status_col in merged.columns else 0
    )

    return {
        "pre_precision": pre_p,
        "post_precision": post_p,
        "pre_recall": pre_r,
        "post_recall": post_r,
        "pre_f1": pre_f,
        "post_f1": post_f,
        "n_candidate_positives": n_candidate,
        "n_verified_retained": n_retained,
        "n_verified_rejected": n_rejected,
    }


def _compute_threshold_sweep(
    y_true: List[float],
    y_pred: List[float],
    thresholds: List[float] = (1.0, 1.5, 2.0, 2.5, 3.0),
) -> List[Dict]:
    """
    For each candidate positivity threshold on final_feature_value, compute
    binary classification metrics.  Gold positivity is always val > 0.

    Returns a list of dicts, one per threshold, sorted by threshold ascending.
    The entry for threshold=1.0 reproduces the default pipeline behaviour
    (predicted positive if final_feature_value > 0, i.e. >= 1).
    """
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    y_true_bin = (y_true_arr > 0).astype(int)

    rows = []
    for t in thresholds:
        y_pred_bin = (y_pred_arr >= t).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true_bin, y_pred_bin, average="binary", zero_division=0
        )
        acc = float(np.mean(y_true_bin == y_pred_bin))
        rows.append({
            "threshold": t,
            "binary_accuracy": round(acc, 4),
            "positive_precision": round(float(prec), 4),
            "positive_recall": round(float(rec), 4),
            "positive_f1": round(float(f1), 4),
            "n_pred_positive": int(y_pred_bin.sum()),
        })
    return sorted(rows, key=lambda r: r["threshold"])


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

    # Post-hoc threshold sweep
    metrics["threshold_sweep"] = _compute_threshold_sweep(y_true, y_pred)

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


def _compute_geometry_metrics(
    gold_matrix: np.ndarray,
    pred_matrix: np.ndarray,
    words: List[str],
    category_map: Optional[Dict[str, str]] = None,
    top_k_neighbors: int = 5,
) -> Dict:
    """
    Compute geometry-level metrics comparing the LLM feature matrix to the
    human Leuven matrix across held-out words.

    Parameters
    ----------
    gold_matrix : (n_words, n_features) human Leuven values
    pred_matrix : (n_words, n_features) LLM predicted values, same ordering
    words       : list of word_normalized strings, same ordering as rows
    category_map: optional {word_normalized: category_label}
    top_k_neighbors: k for nearest-neighbor overlap

    Returns
    -------
    dict with geometry metrics
    """
    n = len(words)
    eps = 1e-12

    def _norm(mat):
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        return mat / (norms + eps)

    gold_n = _norm(gold_matrix)
    pred_n = _norm(pred_matrix)

    # ------------------------------------------------------------------
    # 1. Per-word cosine similarity (already in word_level_metrics.csv,
    #    but summarised here for the geometry dict too)
    # ------------------------------------------------------------------
    per_word_cos = np.einsum("ij,ij->i", gold_n, pred_n)
    mean_cos = float(np.mean(per_word_cos))
    median_cos = float(np.median(per_word_cos))

    # ------------------------------------------------------------------
    # 2. Pairwise similarity matrix correlation
    #    Flatten upper triangle of human pairwise cosine matrix vs LLM's.
    # ------------------------------------------------------------------
    gold_pw = gold_n @ gold_n.T          # (n, n)
    pred_pw = pred_n @ pred_n.T
    triu_idx = np.triu_indices(n, k=1)
    gold_flat = gold_pw[triu_idx]
    pred_flat = pred_pw[triu_idx]
    if len(gold_flat) >= 2:
        pw_pearson, _ = pearsonr(gold_flat, pred_flat)
        pw_spearman, _ = spearmanr(gold_flat, pred_flat)
    else:
        pw_pearson = pw_spearman = float("nan")

    # ------------------------------------------------------------------
    # 3. Nearest-neighbor overlap
    #    For each word, find its top-k gold neighbors and top-k pred
    #    neighbors (excluding itself) and measure Jaccard overlap.
    # ------------------------------------------------------------------
    nn_overlaps = []
    for i in range(n):
        gold_sims = gold_pw[i].copy(); gold_sims[i] = -np.inf
        pred_sims = pred_pw[i].copy(); pred_sims[i] = -np.inf
        gold_nn = set(np.argsort(gold_sims)[::-1][:top_k_neighbors])
        pred_nn = set(np.argsort(pred_sims)[::-1][:top_k_neighbors])
        overlap = len(gold_nn & pred_nn) / len(gold_nn | pred_nn) if gold_nn else 0.0
        nn_overlaps.append(overlap)
    mean_nn_jaccard = float(np.mean(nn_overlaps))

    # ------------------------------------------------------------------
    # 4. Category clustering (silhouette-style: within vs between cosine)
    #    Only computed when category_map is provided.
    # ------------------------------------------------------------------
    category_clustering: Optional[Dict] = None
    if category_map and n >= 4:
        cats = [category_map.get(w, "unknown") for w in words]
        unique_cats = [c for c in set(cats) if cats.count(c) >= 2]
        if len(unique_cats) >= 2:
            within_sims, between_sims = [], []
            for i in range(n):
                for j in range(i + 1, n):
                    sim_g = float(gold_pw[i, j])
                    sim_p = float(pred_pw[i, j])
                    if cats[i] == cats[j] and cats[i] in unique_cats:
                        within_sims.append((sim_g, sim_p))
                    elif cats[i] != cats[j]:
                        between_sims.append((sim_g, sim_p))
            if within_sims and between_sims:
                category_clustering = {
                    "gold_within_mean": round(float(np.mean([s[0] for s in within_sims])), 4),
                    "gold_between_mean": round(float(np.mean([s[0] for s in between_sims])), 4),
                    "pred_within_mean": round(float(np.mean([s[1] for s in within_sims])), 4),
                    "pred_between_mean": round(float(np.mean([s[1] for s in between_sims])), 4),
                    "gold_separation": round(
                        float(np.mean([s[0] for s in within_sims])) -
                        float(np.mean([s[0] for s in between_sims])), 4),
                    "pred_separation": round(
                        float(np.mean([s[1] for s in within_sims])) -
                        float(np.mean([s[1] for s in between_sims])), 4),
                }

    # ------------------------------------------------------------------
    # 5. Feature-density calibration
    #    Compare mean number of positive features per word (threshold > 0).
    # ------------------------------------------------------------------
    gold_density = float(np.mean((gold_matrix > 0).sum(axis=1)))
    pred_density = float(np.mean((pred_matrix > 0).sum(axis=1)))

    # ------------------------------------------------------------------
    # 6. Feature-prevalence calibration
    #    Pearson/Spearman correlation between per-feature means across words.
    # ------------------------------------------------------------------
    gold_feat_means = gold_matrix.mean(axis=0)
    pred_feat_means = pred_matrix.mean(axis=0)
    feat_pearson, _ = pearsonr(gold_feat_means, pred_feat_means)
    feat_spearman, _ = spearmanr(gold_feat_means, pred_feat_means)

    # ------------------------------------------------------------------
    # 7. Human split-half ceiling estimate
    #    Leuven had 4 raters per cell; simulate a 2 vs 2 split.
    #    Even-rater sum vs odd-rater sum is not possible from the aggregate
    #    matrix, so we approximate the ceiling using Spearman-Brown:
    #      r_SB = 2 * r_halfhalf / (1 + r_halfhalf)
    #    We estimate r_halfhalf from the observed inter-word variability
    #    relative to the 0-4 scale range, using a conservative assumption.
    #    NOTE: this is only a rough estimate; report it as such.
    # ------------------------------------------------------------------
    # Since we don't have the raw 4-rater responses, we report NaN and
    # a note for the user to supply if available.
    human_ceiling_note = (
        "Not computable from aggregate matrix. "
        "Supply raw 4-rater data to estimate split-half ceiling."
    )

    geometry: Dict = {
        "n_words": n,
        "mean_cosine_similarity": round(mean_cos, 4),
        "median_cosine_similarity": round(median_cos, 4),
        "pairwise_sim_pearson_r": round(float(pw_pearson), 4),
        "pairwise_sim_spearman_r": round(float(pw_spearman), 4),
        f"mean_nn_jaccard_top{top_k_neighbors}": round(mean_nn_jaccard, 4),
        "gold_mean_positive_features_per_word": round(gold_density, 2),
        "pred_mean_positive_features_per_word": round(pred_density, 2),
        "feature_prevalence_pearson_r": round(float(feat_pearson), 4),
        "feature_prevalence_spearman_r": round(float(feat_spearman), 4),
        "human_ceiling_note": human_ceiling_note,
    }
    if category_clustering:
        geometry["category_clustering"] = category_clustering

    return geometry


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

    # Write threshold sweep to a standalone CSV for easy inspection
    sweep_df = pd.DataFrame(metrics["threshold_sweep"])
    sweep_df.to_csv(output_dir / "threshold_sweep.csv", index=False)
    logger.info("Threshold sweep:\n%s", sweep_df.to_string(index=False))

    logger.info("Cell holdout metrics: %s", {k: v for k, v in metrics.items() if k != "threshold_sweep"})
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
            if pd.notna(row["final_feature_value"]):
                pred_vec[fid] = float(row["final_feature_value"])

        wm = _compute_word_metrics(gold_vec, pred_vec)
        wm["word_normalized"] = wn
        word_metrics_rows.append(wm)
        all_cos_sims.append(wm["cosine_similarity"])

    if not word_metrics_rows:
        return {}

    word_df = pd.DataFrame(word_metrics_rows)
    word_df.to_csv(output_dir / "word_level_metrics.csv", index=False)

    # ------------------------------------------------------------------
    # Geometry metrics: build aligned gold / pred matrices then evaluate
    # ------------------------------------------------------------------
    ordered_words = [r["word_normalized"] for r in word_metrics_rows]
    gold_matrix = np.array([
        [float(df[df["word_normalized"] == wn].iloc[0][fcol]) for fcol in feature_columns]
        for wn in ordered_words
    ])
    pred_matrix = np.array([
        [
            float(
                res_df[
                    (res_df["word_normalized"] == wn) & (res_df["feature_id"] == fid)
                ]["final_feature_value"].dropna().values[0]
            ) if len(
                res_df[
                    (res_df["word_normalized"] == wn) & (res_df["feature_id"] == fid)
                ]["final_feature_value"].dropna()
            ) > 0 else 0.0
            for fid, _ in enumerate(feature_columns)
        ]
        for wn in ordered_words
    ])

    geometry = _compute_geometry_metrics(
        gold_matrix=gold_matrix,
        pred_matrix=pred_matrix,
        words=ordered_words,
        category_map=category_map if category_map else None,
    )
    geometry_path = output_dir / "geometry_metrics.json"
    geometry_path.write_text(json.dumps(geometry, indent=2))
    logger.info("Geometry metrics: %s", {k: v for k, v in geometry.items() if k != "category_clustering"})

    summary = {
        "mode": "word_holdout",
        "n_test_words": len(test_word_set),
        "mean_cosine_similarity": round(float(np.mean(all_cos_sims)), 4),
        "median_cosine_similarity": round(float(np.median(all_cos_sims)), 4),
        "pairwise_sim_pearson_r": geometry.get("pairwise_sim_pearson_r"),
        "pairwise_sim_spearman_r": geometry.get("pairwise_sim_spearman_r"),
        "mean_nn_jaccard_top5": geometry.get("mean_nn_jaccard_top5"),
        "feature_prevalence_pearson_r": geometry.get("feature_prevalence_pearson_r"),
        "feature_prevalence_spearman_r": geometry.get("feature_prevalence_spearman_r"),
        "gold_mean_positive_features_per_word": geometry.get("gold_mean_positive_features_per_word"),
        "pred_mean_positive_features_per_word": geometry.get("pred_mean_positive_features_per_word"),
        "test_size": test_size,
        "seed": seed,
    }
    if "category_clustering" in geometry:
        summary["category_clustering"] = geometry["category_clustering"]

    (output_dir / "feature_validation_metrics.json").write_text(
        json.dumps(summary, indent=2)
    )
    logger.info("Word holdout summary: %s", {k: v for k, v in summary.items() if k != "category_clustering"})
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
