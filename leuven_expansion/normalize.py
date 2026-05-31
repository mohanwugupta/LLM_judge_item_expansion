"""
leuven_expansion/normalize.py

Word normalization utilities for the Leuven feature expansion pipeline.
"""
from __future__ import annotations

import re
import unicodedata


def normalize_word(word: str) -> str:
    """
    Normalize a word for deduplication and matching:
    - strip whitespace
    - lowercase
    - remove accents (NFD decomposition + strip combining chars)
    - collapse internal whitespace
    """
    w = word.strip()
    w = unicodedata.normalize("NFD", w)
    w = "".join(c for c in w if unicodedata.category(c) != "Mn")
    w = w.lower()
    w = re.sub(r"\s+", " ", w)
    return w


def apply_singular_to_plural(
    words: list[str],
    mapping_df,
    singular_col: str = "singular",
    plural_col: str = "plural",
) -> dict[str, str]:
    """
    Build a dict mapping normalized singular → normalized plural for any
    words that appear in the mapping CSV.

    Parameters
    ----------
    words        : list of words to look up
    mapping_df   : pandas DataFrame with singular_col and plural_col
    singular_col : column name for singular forms
    plural_col   : column name for plural forms

    Returns
    -------
    dict {normalized_word: canonical_form}  (identity if not in mapping)
    """
    s2p = {
        normalize_word(r[singular_col]): normalize_word(r[plural_col])
        for _, r in mapping_df.iterrows()
    }
    result = {}
    for w in words:
        nw = normalize_word(w)
        result[nw] = s2p.get(nw, nw)
    return result
