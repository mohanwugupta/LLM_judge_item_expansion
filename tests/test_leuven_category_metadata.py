"""
tests/test_leuven_category_metadata.py

Tests for leuven_expansion/category_metadata.py
"""
import pandas as pd
import pytest

from leuven_expansion.category_metadata import (
    load_categories,
    get_category_map,
    stratified_word_split,
)


@pytest.fixture
def categories_csv(tmp_path):
    df = pd.DataFrame({
        "Name": ["dog", "cat", "tiger", "hammer", "saw", "knife", "apple", "pear"],
        "Category": [
            "animals", "animals", "animals",
            "tools", "tools", "tools",
            "fruit", "fruit",
        ],
    })
    p = tmp_path / "categories.csv"
    df.to_csv(p, index=False)
    return p


def test_load_categories_adds_normalized_col(categories_csv):
    df = load_categories(categories_csv)
    assert "word_normalized" in df.columns


def test_load_categories_missing_col_raises(tmp_path):
    df = pd.DataFrame({"Word": ["dog"], "Category": ["animals"]})
    p = tmp_path / "bad.csv"
    df.to_csv(p, index=False)
    with pytest.raises(ValueError, match="item column"):
        load_categories(p, item_col="Name")


def test_get_category_map(categories_csv):
    df = load_categories(categories_csv)
    cmap = get_category_map(df)
    assert cmap["dog"] == "animals"
    assert cmap["hammer"] == "tools"
    assert cmap["apple"] == "fruit"


def test_stratified_split_respects_fraction(categories_csv):
    df = load_categories(categories_csv)
    cmap = get_category_map(df)
    words = list(cmap.keys())
    train, test = stratified_word_split(words, cmap, test_fraction=0.25, seed=42)
    assert len(train) + len(test) == len(words)
    assert len(test) > 0


def test_stratified_split_no_overlap(categories_csv):
    df = load_categories(categories_csv)
    cmap = get_category_map(df)
    words = list(cmap.keys())
    train, test = stratified_word_split(words, cmap, test_fraction=0.25, seed=42)
    assert set(train).isdisjoint(set(test))


def test_stratified_split_all_categories_sampled(categories_csv):
    """Every category should have at least one word in the test set."""
    df = load_categories(categories_csv)
    cmap = get_category_map(df)
    words = list(cmap.keys())
    _, test = stratified_word_split(words, cmap, test_fraction=0.34, seed=42)
    test_cats = {cmap.get(w, "unknown") for w in test}
    all_cats = set(cmap.values())
    assert test_cats == all_cats


def test_unknown_category_handled(categories_csv):
    """Words not in category_map go to 'unknown' category."""
    df = load_categories(categories_csv)
    cmap = get_category_map(df)
    words = list(cmap.keys()) + ["mystery_word"]
    train, test = stratified_word_split(words, cmap, test_fraction=0.25, seed=42)
    assert len(train) + len(test) == len(words)
