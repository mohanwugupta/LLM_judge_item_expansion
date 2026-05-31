"""
leuven_expansion/compute_feature_coverage.py

Compute DRM coverage statistics for the expanded feature matrix.
Reports which DRM words are now covered, how many lists are complete,
and how coverage improved vs. the original Leuven set.
"""
from __future__ import annotations

import json
import pathlib
from typing import Dict, List, Optional

import pandas as pd

from leuven_expansion.normalize import normalize_word


def compute_coverage(
    expanded_matrix_csv: str | pathlib.Path,
    drm_items_csv: str | pathlib.Path,
    output_dir: str | pathlib.Path,
    word_col: str = "word_normalized",
    list_col: Optional[str] = "list_id",
    lure_col: Optional[str] = "is_critical_lure",
) -> Dict:
    """
    Compute DRM coverage statistics.

    Parameters
    ----------
    expanded_matrix_csv : path to expanded_feature_matrix.csv
    drm_items_csv       : path to DRM items file (must have word column)
    output_dir          : directory to write coverage_report.csv
    word_col            : normalized word column name
    list_col            : optional list-identity column in DRM file
    lure_col            : optional critical-lure indicator column in DRM file

    Returns
    -------
    dict with coverage summary statistics
    """
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    expanded = pd.read_csv(expanded_matrix_csv)
    drm = pd.read_csv(drm_items_csv)

    if "word" not in drm.columns and word_col not in drm.columns:
        raise ValueError(f"DRM items CSV must have a 'word' or '{word_col}' column.")

    drm_word_col = word_col if word_col in drm.columns else "word"
    drm["word_normalized"] = drm[drm_word_col].apply(normalize_word)

    expanded_words = set(expanded[word_col].apply(normalize_word))

    drm["covered"] = drm["word_normalized"].isin(expanded_words)

    n_drm = len(drm)
    n_covered = drm["covered"].sum()
    coverage_rate = n_covered / n_drm if n_drm > 0 else 0.0

    stats: Dict = {
        "n_drm_items": n_drm,
        "n_covered": int(n_covered),
        "coverage_rate": round(float(coverage_rate), 4),
    }

    # Per-list coverage
    if list_col and list_col in drm.columns:
        list_stats = (
            drm.groupby(list_col)["covered"]
            .agg(["sum", "count"])
            .rename(columns={"sum": "n_covered", "count": "n_items"})
        )
        list_stats["list_coverage_rate"] = (
            list_stats["n_covered"] / list_stats["n_items"]
        ).round(4)
        n_complete = int((list_stats["list_coverage_rate"] == 1.0).sum())
        stats["n_complete_lists"] = n_complete
        stats["n_lists"] = int(len(list_stats))
        list_stats.to_csv(output_dir / "list_coverage.csv")

    # Critical lure coverage
    if lure_col and lure_col in drm.columns:
        lure_df = drm[drm[lure_col].astype(str).str.lower().isin(["true", "1", "yes"])]
        n_lures = len(lure_df)
        n_lures_covered = int(lure_df["covered"].sum())
        stats["n_critical_lures"] = n_lures
        stats["n_lures_covered"] = n_lures_covered
        stats["lure_coverage_rate"] = round(
            n_lures_covered / n_lures if n_lures > 0 else 0.0, 4
        )

    # Write coverage report
    drm.to_csv(output_dir / "coverage_report.csv", index=False)
    (output_dir / "coverage_summary.json").write_text(json.dumps(stats, indent=2))

    return stats
