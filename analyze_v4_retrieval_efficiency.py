#!/usr/bin/env python3
"""Posthoc retrieval-and-cascade benchmark against complete atomic judgments."""
from __future__ import annotations

import argparse
import json
import sys
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

from calibrate_v4_judgments import apply_threshold
from leuven_expansion.v4 import sha256_file, stable_json_hash, write_json


ROOT = Path(__file__).resolve().parent
VALIDATION_ROOT = ROOT / "ISC-CI_LLM_validation"
if str(VALIDATION_ROOT) not in sys.path:
    sys.path.insert(0, str(VALIDATION_ROOT))

from iscci_validation.consolidation import encode_phrases  # noqa: E402


DEFAULT_V2 = ROOT / "artifacts" / "leuven_full_labels" / "leuven_full_v2"
DEFAULT_HUMAN = ROOT / "data" / "leuven_combined_features_consolidated.csv"
DEFAULT_RELEASED = (
    VALIDATION_ROOT
    / "upstream"
    / "IntegratedSemanticsControlContextInference"
    / "models"
    / "1and2shot_isc-seed3.pt"
)


def _rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    result = spearmanr(left, right)
    return float(result.statistic)


def _geometry_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_rdm = np.nan_to_num(pdist(left, metric="cosine"), nan=0.0)
    right_rdm = np.nan_to_num(pdist(right, metric="cosine"), nan=0.0)
    return _rank_correlation(left_rdm, right_rdm)


def _load_v2_benchmark(
    resolved_path: Path, human_path: Path
) -> tuple[
    list[str], list[str], list[str], np.ndarray, np.ndarray, list[set[str]], pd.DataFrame
]:
    human_counts = pd.read_csv(
        human_path, index_col=0, encoding="ISO-8859-1"
    )
    words = list(map(str, human_counts.index))
    feature_texts = list(map(str, human_counts.columns))
    feature_ids = [f"v2_{index}" for index in range(len(feature_texts))]
    resolved = pd.read_csv(resolved_path)
    matrix = (
        resolved.pivot(
            index="word_normalized", columns="feature_id", values="final_feature_value"
        )
        .reindex(index=words, columns=range(len(feature_texts)))
        .fillna(0.0)
    )
    gold = matrix.to_numpy(dtype=np.float32)
    human_binary = human_counts.gt(3).to_numpy(dtype=np.int8)
    source_words = [
        set(human_counts.index[human_counts.iloc[:, index].gt(0)].astype(str))
        for index in range(len(feature_texts))
    ]
    keys = resolved[["word_normalized", "feature_id", "adjudicated"]].copy()
    keys["feature_key"] = keys["feature_id"].map(lambda value: f"v2_{int(value)}")
    return words, feature_ids, feature_texts, gold, human_binary, source_words, keys


def _load_v4_benchmark(
    resolved_path: Path,
    candidate_bank_path: Path,
    human_path: Path,
    human_mapping_path: Path | None,
) -> tuple[
    list[str], list[str], list[str], np.ndarray, np.ndarray, list[set[str]], pd.DataFrame
]:
    human_counts = pd.read_csv(
        human_path, index_col=0, encoding="ISO-8859-1"
    )
    words = list(map(str, human_counts.index))
    bank = pd.read_csv(candidate_bank_path, dtype=str).fillna("")
    if "candidate_index" in bank:
        bank = bank.assign(
            candidate_index=pd.to_numeric(bank["candidate_index"], errors="raise")
        ).sort_values("candidate_index")
    feature_ids = bank["candidate_id"].tolist()
    feature_texts = bank["canonical_feature_text"].tolist()
    source_words = [set(json.loads(value)) for value in bank["source_words"]]
    resolved = pd.read_csv(resolved_path, dtype={"candidate_id": str})
    value_column = (
        "resolved_value" if "resolved_value" in resolved else "final_feature_value"
    )
    word_column = "target_word" if "target_word" in resolved else "word_normalized"
    matrix = resolved.pivot(
        index=word_column, columns="candidate_id", values=value_column
    ).reindex(index=words, columns=feature_ids)
    if matrix.isna().any().any():
        raise ValueError("The V4 benchmark is not an exhaustive completed matrix")
    human_binary = np.zeros(matrix.shape, dtype=np.int8)
    if human_mapping_path is not None:
        mapping = pd.read_csv(human_mapping_path, dtype=str).fillna("")
        mapped_feature = dict(zip(mapping["candidate_id"], mapping["human_feature"]))
        for column, candidate_id in enumerate(feature_ids):
            human_feature = mapped_feature.get(candidate_id, "")
            if human_feature in human_counts:
                human_binary[:, column] = human_counts[human_feature].gt(3).astype(np.int8)
    keys = resolved[[word_column, "candidate_id", "adjudicated"]].rename(
        columns={word_column: "word_normalized", "candidate_id": "feature_key"}
    )
    return (
        words,
        feature_ids,
        feature_texts,
        matrix.to_numpy(dtype=np.float32),
        human_binary,
        source_words,
        keys,
    )


def _feature_split(
    prevalence: np.ndarray, feature_ids: list[str], fraction: float, seed: int
) -> np.ndarray:
    if not 0 < fraction < 1:
        raise ValueError("development_fraction must be between zero and one")
    ranks = pd.Series(prevalence).rank(method="first")
    bins = pd.qcut(ranks, q=min(10, len(prevalence)), labels=False, duplicates="drop")
    development = np.zeros(len(feature_ids), dtype=bool)
    rng = np.random.default_rng(seed)
    for bin_id in sorted(pd.unique(bins)):
        indices = np.where(np.asarray(bins) == bin_id)[0]
        indices = rng.permutation(indices)
        count = max(1, min(len(indices) - 1, round(len(indices) * fraction)))
        development[indices[:count]] = True
    return development


def _load_or_build_embeddings(
    output_dir: Path,
    words: list[str],
    feature_ids: list[str],
    feature_texts: list[str],
    model: str,
    revision: str,
) -> tuple[np.ndarray, np.ndarray]:
    cache = output_dir / "retrieval_embeddings.npz"
    if cache.exists():
        loaded = np.load(cache, allow_pickle=False)
        if (
            loaded["words"].astype(str).tolist() != words
            or loaded["feature_ids"].astype(str).tolist() != feature_ids
        ):
            raise ValueError("Retrieval embedding cache does not match this benchmark")
        return loaded["word_embeddings"], loaded["feature_embeddings"]
    embeddings = encode_phrases(words + feature_texts, model, revision)
    word_embeddings = embeddings[: len(words)]
    feature_embeddings = embeddings[len(words) :]
    np.savez(
        cache,
        words=np.asarray(words),
        feature_ids=np.asarray(feature_ids),
        word_embeddings=word_embeddings,
        feature_embeddings=feature_embeddings,
    )
    return word_embeddings, feature_embeddings


def _released_neighbors(
    released_path: Path, words: list[str], neighbor_count: int
) -> dict[str, set[str]]:
    checkpoint = torch.load(released_path, map_location="cpu", weights_only=False)
    vectors = checkpoint["state_dict"]["input_to_independent.weight"].detach().numpy()
    vectors = vectors[: len(words)].astype(np.float64)
    vectors /= np.where(
        np.linalg.norm(vectors, axis=1, keepdims=True) == 0,
        1.0,
        np.linalg.norm(vectors, axis=1, keepdims=True),
    )
    similarity = vectors @ vectors.T
    np.fill_diagonal(similarity, -np.inf)
    neighbors: dict[str, set[str]] = {}
    for index, word in enumerate(words):
        nearest = np.argsort(-similarity[index], kind="stable")[:neighbor_count]
        neighbors[word] = {words[value] for value in nearest}
    return neighbors


def _shortlist_masks(
    similarity: np.ndarray,
    words: list[str],
    source_words: list[set[str]],
    semantic_neighbors: dict[str, set[str]],
    k_values: list[int],
) -> tuple[dict[int, np.ndarray], pd.DataFrame]:
    word_to_index = {word: index for index, word in enumerate(words)}
    ranked = np.argsort(-similarity, axis=1, kind="stable")
    masks = {
        k: np.zeros((len(feature_texts := source_words), len(words)), dtype=bool)
        for k in k_values
    }
    force_rows: list[dict[str, Any]] = []
    for feature_index, sources in enumerate(source_words):
        forced = set(sources)
        for source in sources:
            forced.update(semantic_neighbors.get(source, set()))
        forced_indices = [word_to_index[word] for word in forced if word in word_to_index]
        for k in k_values:
            masks[k][feature_index, ranked[feature_index, : min(k, len(words))]] = True
            masks[k][feature_index, forced_indices] = True
        force_rows.append(
            {
                "feature_index": feature_index,
                "source_word_count": len(sources),
                "forced_source_and_neighbor_count": len(forced_indices),
            }
        )
    return masks, pd.DataFrame(force_rows)


def _metrics(
    gold: np.ndarray,
    human: np.ndarray,
    shortlist: np.ndarray,
    feature_mask: np.ndarray,
) -> dict[str, float | int]:
    selected_gold = gold[:, feature_mask]
    selected_human = human[:, feature_mask]
    selected_shortlist = shortlist[feature_mask].T
    pruned = selected_gold * selected_shortlist
    gold_positive = selected_gold.astype(bool)
    human_positive = selected_human.astype(bool)
    retained_gold = int(gold_positive.sum())
    retained_human = int(human_positive.sum())
    return {
        "feature_count": int(feature_mask.sum()),
        "shortlisted_cells": int(selected_shortlist.sum()),
        "total_cells": int(selected_shortlist.size),
        "shortlist_fraction": float(selected_shortlist.mean()),
        "initial_call_reduction": float(1.0 - selected_shortlist.mean()),
        "positive_cell_recall": (
            float((gold_positive & selected_shortlist).sum() / retained_gold)
            if retained_gold
            else float("nan")
        ),
        "positive_cell_precision": 1.0,
        "leuven_positive_cell_recall": (
            float((human_positive & selected_shortlist).sum() / retained_human)
            if retained_human
            else float("nan")
        ),
        "object_geometry_correlation": _geometry_correlation(selected_gold, pruned),
        "feature_geometry_correlation": _geometry_correlation(
            selected_gold.T, pruned.T
        ),
        "full_human_object_RDM_correlation": _geometry_correlation(
            selected_gold, selected_human
        ),
        "pruned_human_object_RDM_correlation": _geometry_correlation(
            pruned, selected_human
        ),
    }


def _audit_excluded(
    gold: np.ndarray,
    shortlist: np.ndarray,
    feature_mask: np.ndarray,
    fraction: float,
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    selected_gold = gold[:, feature_mask].T.astype(bool)
    excluded = ~shortlist[feature_mask]
    excluded_gold = selected_gold[excluded]
    included_positive = int((selected_gold & ~excluded).sum())
    sample_size = max(1, round(len(excluded_gold) * fraction))
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int]] = []
    for repetition in range(repetitions):
        sample = rng.choice(len(excluded_gold), sample_size, replace=False)
        rate = float(excluded_gold[sample].mean())
        estimated_missed = rate * len(excluded_gold)
        rows.append(
            {
                "repetition": repetition,
                "audit_cells": sample_size,
                "audited_positive_rate": rate,
                "estimated_missed_positive_cells": estimated_missed,
                "estimated_positive_recall": (
                    included_positive / (included_positive + estimated_missed)
                    if included_positive + estimated_missed
                    else 1.0
                ),
            }
        )
    return pd.DataFrame(rows)


def _cascade_metrics(
    votes_path: Path,
    resolution_keys: pd.DataFrame,
    feature_ids: list[str],
    words: list[str],
    gold: np.ndarray,
    shortlist: np.ndarray,
    feature_mask: np.ndarray,
) -> pd.DataFrame:
    votes = pd.read_csv(
        votes_path,
        usecols=[
            "word_normalized",
            "feature_id",
            "judge_id",
            "feature_value",
            "confidence",
            "ambiguous",
        ],
    )
    if feature_ids[0].startswith("v2_"):
        votes["feature_key"] = votes["feature_id"].map(lambda value: f"v2_{int(value)}")
    elif "candidate_id" not in votes:
        # V4 final vote tables preserve integer feature_id; recover IDs by bank order.
        votes["feature_key"] = votes["feature_id"].map(
            dict(enumerate(feature_ids))
        )
    else:
        votes["feature_key"] = votes["candidate_id"]
    word_index = {word: index for index, word in enumerate(words)}
    feature_index = {feature: index for index, feature in enumerate(feature_ids)}
    adjudicated = {
        (str(row.word_normalized), str(row.feature_key)): bool(row.adjudicated)
        for row in resolution_keys.itertuples(index=False)
    }
    full_cells = int(gold[:, feature_mask].size)
    full_adjudications = sum(
        value
        for (word, feature), value in adjudicated.items()
        if feature_index.get(feature, -1) < len(feature_mask)
        and feature_index.get(feature, -1) >= 0
        and feature_mask[feature_index[feature]]
    )
    full_calls = 3 * full_cells + 3 * full_adjudications
    rows: list[dict[str, Any]] = []
    for judge in sorted(votes["judge_id"].unique()):
        one = votes.loc[votes["judge_id"].eq(judge)].copy()
        one["word_index"] = one["word_normalized"].map(word_index)
        one["feature_index"] = one["feature_key"].map(feature_index)
        one = one.dropna(subset=["word_index", "feature_index"])
        one["word_index"] = one["word_index"].astype(int)
        one["feature_index"] = one["feature_index"].astype(int)
        one = one.loc[one["feature_index"].map(lambda index: feature_mask[index])]
        in_shortlist = shortlist[
            one["feature_index"].to_numpy(), one["word_index"].to_numpy()
        ]
        one = one.loc[in_shortlist].copy()
        routed = (
            pd.to_numeric(one["feature_value"], errors="coerce").gt(0)
            | one["ambiguous"].fillna(False).astype(bool)
            | pd.to_numeric(one["confidence"], errors="coerce").lt(0.80)
        )
        routed_rows = one.loc[routed]
        reconstructed = np.zeros_like(gold[:, feature_mask], dtype=bool)
        selected_feature_indices = np.where(feature_mask)[0]
        local_feature = {
            global_index: local for local, global_index in enumerate(selected_feature_indices)
        }
        for row in routed_rows.itertuples(index=False):
            reconstructed[
                int(row.word_index), local_feature[int(row.feature_index)]
            ] = bool(gold[int(row.word_index), int(row.feature_index)])
        selected_gold = gold[:, feature_mask].astype(bool)
        positives = int(selected_gold.sum())
        adjudication_calls = 3 * sum(
            adjudicated.get((str(row.word_normalized), str(row.feature_key)), False)
            for row in routed_rows.itertuples(index=False)
        )
        calls = len(one) + 2 * int(routed.sum()) + adjudication_calls
        rows.append(
            {
                "cheap_judge": judge,
                "shortlisted_cells": len(one),
                "routed_cells": int(routed.sum()),
                "cascade_calls": calls,
                "full_panel_calls": full_calls,
                "call_reduction": 1.0 - calls / full_calls,
                "token_cost_runtime_reduction_proxy": 1.0 - calls / full_calls,
                "positive_recall": (
                    float((selected_gold & reconstructed).sum() / positives)
                    if positives
                    else float("nan")
                ),
                "positive_precision": 1.0,
                "object_geometry_correlation": _geometry_correlation(
                    selected_gold, reconstructed
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "v4_validation.json")
    parser.add_argument("--resolved-values", type=Path, default=DEFAULT_V2 / "feature_resolutions.csv")
    parser.add_argument("--votes", type=Path, default=DEFAULT_V2 / "feature_votes.csv")
    parser.add_argument("--candidate-bank", type=Path)
    parser.add_argument("--human-mapping", type=Path)
    parser.add_argument("--human-features", type=Path, default=DEFAULT_HUMAN)
    parser.add_argument("--threshold", type=Path, default=ROOT / "artifacts" / "v4" / "judgments" / "judgment_threshold.json")
    parser.add_argument("--gold-rule", choices=["locked", "calibrated"], default="locked")
    parser.add_argument("--released-checkpoint", type=Path, default=DEFAULT_RELEASED)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "v4" / "retrieval_efficiency" / "v2_retrospective")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))["retrieval_efficiency"]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.candidate_bank:
        loaded = _load_v4_benchmark(
            args.resolved_values.resolve(),
            args.candidate_bank.resolve(),
            args.human_features.resolve(),
            args.human_mapping.resolve() if args.human_mapping else None,
        )
        benchmark = "v4_complete_atomic"
    else:
        loaded = _load_v2_benchmark(
            args.resolved_values.resolve(), args.human_features.resolve()
        )
        benchmark = "v2_complete_atomic"
    words, feature_ids, feature_texts, ordinal, human, source_words, keys = loaded
    if args.gold_rule == "locked":
        gold = ordinal > 0
        rule = {"operator": "gt", "value": 0.0}
    else:
        frozen = json.loads(args.threshold.read_text(encoding="utf-8"))
        rule = frozen["selected_rule"]
        gold = apply_threshold(ordinal, rule).astype(bool)

    word_embeddings, feature_embeddings = _load_or_build_embeddings(
        output,
        words,
        feature_ids,
        feature_texts,
        str(config["embedding_model"]),
        str(config["embedding_model_revision"]),
    )
    similarity = feature_embeddings @ word_embeddings.T
    neighbors = _released_neighbors(
        args.released_checkpoint.resolve(), words, int(config["source_neighbor_count"])
    )
    primary_k_values = list(map(int, config["k_values"]))
    fallback_k_values = list(
        map(int, config.get("fallback_k_values_if_gates_fail", []))
    )
    k_values = sorted(set(primary_k_values + fallback_k_values))
    shortlists, force_summary = _shortlist_masks(
        similarity, words, source_words, neighbors, k_values
    )
    development = _feature_split(
        gold.sum(axis=0),
        feature_ids,
        float(config["development_fraction"]),
        int(config["split_seed"]),
    )
    split_table = pd.DataFrame(
        {
            "feature_id": feature_ids,
            "feature_text": feature_texts,
            "split": np.where(development, "development", "heldout_test"),
            "full_positive_cells": gold.sum(axis=0),
            "human_positive_cells": human.sum(axis=0),
            "source_word_count": [len(values) for values in source_words],
        }
    )
    split_table.to_csv(output / "feature_split.csv", index=False)
    force_summary.insert(0, "feature_id", feature_ids)
    force_summary.to_csv(output / "forced_retrieval_summary.csv", index=False)

    rows: list[dict[str, Any]] = []
    for k in k_values:
        for split, mask in [
            ("development", development),
            ("heldout_test", ~development),
            ("all", np.ones(len(feature_ids), dtype=bool)),
        ]:
            rows.append({"K": k, "split": split, **_metrics(gold, human, shortlists[k], mask)})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output / "retrieval_metrics.csv", index=False)
    development_metrics = metrics.loc[metrics["split"].eq("development")].copy()
    eligible = development_metrics.loc[
        development_metrics["positive_cell_recall"].ge(float(config["positive_recall_gate"]))
        & development_metrics["leuven_positive_cell_recall"].ge(float(config["leuven_positive_recall_gate"]))
        & development_metrics["object_geometry_correlation"].ge(float(config["geometry_correlation_gate"]))
        & development_metrics["feature_geometry_correlation"].ge(float(config["geometry_correlation_gate"]))
    ]
    gate_met = not eligible.empty
    selected_k = int(
        eligible.sort_values("K").iloc[0]["K"]
        if gate_met
        else development_metrics.sort_values("K").iloc[-1]["K"]
    )
    heldout = metrics.loc[
        metrics["split"].eq("heldout_test") & metrics["K"].eq(selected_k)
    ].iloc[0].to_dict()
    audit = _audit_excluded(
        gold,
        shortlists[selected_k],
        ~development,
        float(config["negative_audit_fraction"]),
        int(config["negative_audit_repetitions"]),
        int(config["split_seed"]) + 1,
    )
    audit.to_csv(output / "negative_audit_simulation.csv", index=False)
    cascade = _cascade_metrics(
        args.votes.resolve(),
        keys,
        feature_ids,
        words,
        gold,
        shortlists[selected_k],
        ~development,
    )
    cascade.to_csv(output / "cascade_metrics_heldout.csv", index=False)
    selection = {
        "selection_partition": "development features only",
        "primary_K_values": primary_k_values,
        "automatic_fallback_K_values": fallback_k_values,
        "selected_K": selected_k,
        "development_gates_met": gate_met,
        "gates": {
            "positive_cell_recall": config["positive_recall_gate"],
            "leuven_positive_cell_recall": config["leuven_positive_recall_gate"],
            "object_and_feature_geometry_correlation": config["geometry_correlation_gate"],
        },
        "heldout_test_metrics": heldout,
        "negative_audit_estimated_recall_mean": float(audit["estimated_positive_recall"].mean()),
        "negative_audit_estimated_recall_95_interval": [
            float(audit["estimated_positive_recall"].quantile(0.025)),
            float(audit["estimated_positive_recall"].quantile(0.975)),
        ],
    }
    write_json(output / "retrieval_selection.json", selection)
    manifest = {
        "protocol_version": "v4-posthoc-retrieval-cascade-1.0.0",
        "benchmark": benchmark,
        "primary_v4_execution_unchanged": "exhaustive; retrieval is posthoc only",
        "resolved_values": str(args.resolved_values.resolve()),
        "resolved_values_sha256": sha256_file(args.resolved_values.resolve()),
        "votes": str(args.votes.resolve()),
        "votes_sha256": sha256_file(args.votes.resolve()),
        "human_features_sha256": sha256_file(args.human_features.resolve()),
        "released_checkpoint_sha256": sha256_file(args.released_checkpoint.resolve()),
        "gold_rule": rule,
        "feature_count": len(feature_ids),
        "word_count": len(words),
        "development_feature_count": int(development.sum()),
        "heldout_feature_count": int((~development).sum()),
        "selection": selection,
        "token_cost_runtime_note": (
            "Call fractions are reported as token/cost/runtime proxies because preserved "
            "V2 artifacts do not contain per-request token or wall-time telemetry."
        ),
        "config": config,
    }
    manifest["analysis_hash"] = stable_json_hash(manifest)
    write_json(output / "manifest.json", manifest)
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()
