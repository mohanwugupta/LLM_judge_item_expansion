"""
leuven_expansion/category_metadata.py

Load and merge Leuven category metadata for stratified splitting,
error grouping, and coverage reporting.
"""
from __future__ import annotations

import pathlib
from typing import Dict, List, Optional

import pandas as pd

from leuven_expansion.normalize import normalize_word


def load_categories(
    categories_csv: str | pathlib.Path,
    item_col: str = "Name",
    category_col: str = "Category",
) -> pd.DataFrame:
    """
    Load leuven_categories.csv and return a DataFrame with normalized
    word column added.
    """
    df = pd.read_csv(categories_csv)
    if item_col not in df.columns:
        raise ValueError(
            f"Expected item column '{item_col}' not found. "
            f"Available: {list(df.columns)}"
        )
    df["word_normalized"] = df[item_col].apply(normalize_word)
    return df


def get_category_map(
    categories_df: pd.DataFrame,
    item_col: str = "Name",
    category_col: str = "Category",
) -> Dict[str, str]:
    """Return dict {word_normalized: category}."""
    result = {}
    for _, row in categories_df.iterrows():
        nw = normalize_word(str(row[item_col]))
        cat = str(row.get(category_col, "unknown"))
        result[nw] = cat
    return result


def stratified_word_split(
    words: List[str],
    category_map: Dict[str, str],
    test_fraction: float = 0.20,
    seed: int = 42,
) -> tuple[List[str], List[str]]:
    """
    Split a list of words into train/test sets with stratification
    by category.

    Words not found in category_map are treated as category 'unknown'.

    Returns (train_words, test_words).
    """
    import random

    random.seed(seed)

    # Group by category
    from collections import defaultdict
    by_cat: dict[str, list[str]] = defaultdict(list)
    for w in words:
        nw = normalize_word(w)
        cat = category_map.get(nw, "unknown")
        by_cat[cat].append(w)

    train: List[str] = []
    test: List[str] = []

    for cat, cat_words in by_cat.items():
        shuffled = list(cat_words)
        random.shuffle(shuffled)
        n_test = max(1, round(len(shuffled) * test_fraction))
        test.extend(shuffled[:n_test])
        train.extend(shuffled[n_test:])

    return train, test
