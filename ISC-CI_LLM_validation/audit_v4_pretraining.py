#!/usr/bin/env python3
"""Read-only audit of V4 shards; save diagnostics, never judgments or models."""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import json
from pathlib import Path
import re
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[1]
WAIT_FOR_LFS = False
sys.path.insert(0, str(ROOT))
from leuven_expansion.normalize import normalize_word
from leuven_expansion.feature_prompts import (
    build_judge_user_message,
    load_prompts_by_version,
    prompt_hash,
)
from leuven_expansion.v4 import candidate_inventory_hash, sha256_file, stable_shard


def actual_file(path: Path) -> Path:
    """Read a materialized file or its already-fetched Git LFS object."""
    if path.stat().st_size > 1024:
        return path
    payload = path.read_bytes()
    if not payload.startswith(b"version https://git-lfs.github.com/spec/v1"):
        return path
    fields = dict(line.split(" ", 1) for line in payload.decode().splitlines())
    oid = fields["oid"].removeprefix("sha256:")
    cached = ROOT / ".git/lfs/objects" / oid[:2] / oid[2:4] / oid
    if WAIT_FOR_LFS and not cached.exists():
        print(
            f"Waiting for an in-progress LFS fetch: {path.relative_to(ROOT)}",
            flush=True,
        )
        deadline = time.monotonic() + 600
        while not cached.exists() and time.monotonic() < deadline:
            time.sleep(1)
    if not cached.exists() or cached.stat().st_size != int(fields["size"]):
        raise FileNotFoundError(f"Unmaterialized Git LFS input: {path}")
    return cached


def bool_values(series: pd.Series) -> np.ndarray:
    return (
        series.fillna("").astype(str).str.lower().isin(["true", "1", "yes"]).to_numpy()
    )


def cosine_ranks(matrix: np.ndarray) -> np.ndarray:
    # A Gram matrix avoids computing 42,778 long distances separately.
    x = matrix.astype(np.float64)
    norms = np.linalg.norm(x, axis=1)
    if np.any(norms == 0):
        raise ValueError("Object geometry is undefined for an all-zero object")
    similarity = (x @ x.T) / np.outer(norms, norms)
    return rankdata((1 - similarity)[np.triu_indices(len(x), 1)])


def preview_type_only_recovery(output: Path) -> dict:
    """Propose repairs only when the echoed ID is an exact decimal string match."""
    from leuven_expansion.feature_schema import validate_judge_output

    votes = pd.read_csv(
        output / "invalid_cell_adjudication_votes.csv", keep_default_na=False
    )
    proposals, errors = [], Counter()
    for row in votes.itertuples(index=False):
        payload = str(row.raw_json).strip()
        if payload.startswith("```"):
            payload = "\n".join(
                line
                for line in payload.splitlines()
                if not line.strip().startswith("```")
            )
        record = None
        try:
            try:
                record = json.loads(payload)
            except json.JSONDecodeError:
                record = ast.literal_eval(payload)
        except (ValueError, SyntaxError):
            errors["unparseable_payload"] += 1
        recovered = None
        if isinstance(record, dict):
            echoed = record.get("feature_id")
            if (
                isinstance(echoed, str)
                and echoed.isdecimal()
                and int(echoed) == int(row.feature_id)
            ):
                record["feature_id"] = int(echoed)
                recovered, error = validate_judge_output(
                    json.dumps(record),
                    expected_word=row.word_normalized,
                    expected_feature_id=int(row.feature_id),
                )
                if recovered is None:
                    errors[str(error)] += 1
            else:
                errors["feature_id_not_an_exact_numeric_string_match"] += 1
        proposals.append(
            {
                "shard_index": row.shard_index,
                "word_normalized": row.word_normalized,
                "feature_id": int(row.feature_id),
                "feature_text": row.feature_text,
                "adjudicator_idx": int(row.adjudicator_idx),
                "type_only_recoverable": recovered is not None,
                "proposed_value": recovered["feature_value"] if recovered else np.nan,
            }
        )
    frame = pd.DataFrame(proposals)
    rows = []
    for (shard, word, feature), group in frame.groupby(
        ["shard_index", "word_normalized", "feature_id"]
    ):
        values = group.loc[group.type_only_recoverable, "proposed_value"]
        agree = len(values) >= 2 and values.nunique() == 1
        rows.append(
            {
                "shard_index": int(shard),
                "word_normalized": word,
                "feature_id": int(feature),
                "feature_text": group.feature_text.iloc[0],
                "recoverable_adjudicator_rows": int(len(values)),
                "unanimous_type_only_recovery": bool(agree),
                "proposed_final_value": float(values.iloc[0]) if agree else np.nan,
                "applied": False,
            }
        )
    preview = pd.DataFrame(rows)
    preview.to_csv(output / "offline_recovery_preview.csv", index=False)
    return {
        "failed_cells": int(len(preview)),
        "cells_with_unanimous_type_only_recovery": int(
            preview.unanimous_type_only_recovery.sum()
        ),
        "cells_still_needing_fresh_judgment_or_review": int(
            (~preview.unanimous_type_only_recovery).sum()
        ),
        "recoverable_adjudicator_rows": int(frame.type_only_recoverable.sum()),
        "remaining_error_counts": dict(errors),
        "repairs_applied": False,
    }


def check_prompt_sample(frame: pd.DataFrame, texts: np.ndarray) -> Counter:
    prompts = load_prompts_by_version("v2")
    counts = Counter()
    for row in frame.iloc[::5000].itertuples(index=False):
        feature = int(row.feature_id)
        expected = prompt_hash(
            prompts[row.judge_id],
            build_judge_user_message(row.word_normalized, feature, texts[feature]),
        )
        counts["sampled_prompt_hash_rows"] += 1
        counts["sampled_prompt_hash_mismatches"] += expected != row.prompt_hash
    return counts


def audit_prompt_samples() -> dict:
    bank = pd.read_csv(ROOT / "artifacts/v4/discovery/candidate_bank.csv").sort_values(
        "candidate_index"
    )
    counts = Counter()
    for si in range(32):
        path = actual_file(
            ROOT / f"artifacts/v4/judgments/shards/{si:04d}/feature_votes.csv"
        )
        for frame in pd.read_csv(
            path,
            usecols=["word_normalized", "feature_id", "judge_id", "prompt_hash"],
            chunksize=150000,
            keep_default_na=False,
        ):
            counts.update(
                check_prompt_sample(frame, bank.canonical_feature_text.to_numpy())
            )
        print(f"Prompt sample {si:02d}/31 checked", flush=True)
    return dict(counts)


def audit_duplicate_geometry(output: Path) -> dict:
    """Diagnostic only: compare column-weighted and distinct-pattern geometry."""
    bank = pd.read_csv(ROOT / "artifacts/v4/discovery/candidate_bank.csv").sort_values(
        "candidate_index"
    )
    human_counts = pd.read_csv(
        actual_file(ROOT / "data/leuven_combined_features_consolidated.csv"),
        index_col=0,
    )
    words = [normalize_word(str(word)) for word in human_counts.index]
    word_map = dict(zip(words, range(len(words))))
    positive = np.zeros((len(bank), len(words)), dtype=bool)
    incomplete = np.zeros(len(bank), dtype=bool)
    for si in range(32):
        path = actual_file(
            ROOT / f"artifacts/v4/judgments/shards/{si:04d}/feature_resolutions.csv"
        )
        for frame in pd.read_csv(
            path,
            usecols=["word_normalized", "feature_id", "final_feature_value"],
            chunksize=300000,
            low_memory=False,
        ):
            f = frame.feature_id.to_numpy(dtype=np.int64)
            w = frame.word_normalized.map(word_map).to_numpy(dtype=np.int64)
            value = pd.to_numeric(frame.final_feature_value, errors="coerce").to_numpy()
            valid = np.isfinite(value) & (value >= 0) & (value <= 4)
            positive[f, w] = valid & (value > 0)
            incomplete[f[~valid]] = True
        print(f"Duplicate diagnostic {si:02d}/31 loaded", flush=True)
    retained = (positive.sum(axis=1) > 3) & ~incomplete
    retained_ids = np.flatnonzero(retained)
    _, indices, counts = np.unique(
        np.packbits(positive[retained], axis=1),
        axis=0,
        return_index=True,
        return_counts=True,
    )
    unique_values = positive[retained_ids[indices]].T
    h = human_counts.gt(3).to_numpy()
    human_ranks = cosine_ranks(h[:, h.sum(axis=0) > 3])
    original_ranks = cosine_ranks(positive[retained].T)
    distinct_ranks = cosine_ranks(unique_values)
    result = {
        "complete_retained_columns": int(retained.sum()),
        "unique_binary_patterns": int(len(indices)),
        "redundant_columns": int(retained.sum() - len(indices)),
        "column_weighted_human_object_rdm_spearman": float(
            spearmanr(human_ranks, original_ranks).statistic
        ),
        "distinct_pattern_human_object_rdm_spearman": float(
            spearmanr(human_ranks, distinct_ranks).statistic
        ),
        "column_weighted_vs_distinct_pattern_rdm_spearman": float(
            spearmanr(original_ranks, distinct_ranks).statistic
        ),
        "primary_matrix_changed": False,
    }
    examples = []
    patterns = np.packbits(positive[retained], axis=1)
    for pi in np.argsort(counts)[-15:][::-1]:
        members = retained_ids[np.all(patterns == patterns[indices[pi]], axis=1)]
        examples.append(
            {
                "duplicate_column_count": int(counts[pi]),
                "positive_objects": json.dumps(
                    [words[i] for i in np.flatnonzero(positive[members[0]])]
                ),
                "example_phrases": json.dumps(
                    bank.iloc[members[:10]].canonical_feature_text.tolist()
                ),
            }
        )
    pd.DataFrame(examples).to_csv(
        output / "duplicate_pattern_examples.csv", index=False
    )
    (output / "duplicate_geometry.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def audit_spotchecks(output: Path) -> None:
    bank = pd.read_csv(ROOT / "artifacts/v4/discovery/candidate_bank.csv")
    index_by_text = dict(zip(bank.canonical_feature_text, bank.candidate_index))
    id_by_index = dict(zip(bank.candidate_index, bank.candidate_id))
    requested = [
        ("plane", "has seats inside for riders"),
        ("plane", "a plane can fly at high altitudes and speeds"),
        ("can opener", "cylindrical shape with an open top"),
        ("screwdriver", "cylindrical shape with an open top"),
        ("bat", "can be handheld or larger for kitchen use"),
        ("mug", "mug may feature logos or decorative designs"),
        ("cap", "mug may feature logos or decorative designs"),
        ("cat", "a cat has fur"),
        ("dog", "a cat has fur"),
        ("monkey", "can be found in different sizes"),
    ]
    targets = {}
    for word, text in requested:
        feature = int(index_by_text[text])
        shard = stable_shard(id_by_index[feature], word, 32)
        targets.setdefault(shard, set()).add((word, feature))
    rows = []
    for si, keys in targets.items():
        path = actual_file(
            ROOT / f"artifacts/v4/judgments/shards/{si:04d}/feature_resolutions.csv"
        )
        for frame in pd.read_csv(
            path, chunksize=150000, keep_default_na=False, low_memory=False
        ):
            mask = np.fromiter(
                (
                    (w, int(f)) in keys
                    for w, f in zip(frame.word_normalized, frame.feature_id)
                ),
                dtype=bool,
                count=len(frame),
            )
            if mask.any():
                selected = frame.loc[mask].copy()
                selected.insert(0, "shard_index", si)
                rows.extend(selected.to_dict("records"))
    pd.DataFrame(rows).to_csv(output / "semantic_spotchecks.csv", index=False)


def main() -> None:
    global WAIT_FOR_LFS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--skip-votes", action="store_true")
    parser.add_argument(
        "--wait-for-lfs",
        action="store_true",
        help="Allow an independent, in-progress LFS fetch to finish missing inputs.",
    )
    args = parser.parse_args()
    WAIT_FOR_LFS = args.wait_for_lfs
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    bank_path = ROOT / "artifacts/v4/discovery/candidate_bank.csv"
    bank = pd.read_csv(actual_file(bank_path), keep_default_na=False)
    bank = bank.sort_values("candidate_index").reset_index(drop=True)
    human_path = ROOT / "data/leuven_combined_features_consolidated.csv"
    human_counts = pd.read_csv(actual_file(human_path), index_col=0)
    words = [normalize_word(str(x)) for x in human_counts.index]
    word_map = dict(zip(words, range(len(words))))
    nc, nw = len(bank), len(words)
    cells = nc * nw
    texts = bank.canonical_feature_text.to_numpy()
    ids = bank.candidate_id.to_numpy()
    raw = np.full(cells, np.nan, dtype=np.float32)
    seen = np.zeros(cells, dtype=np.uint8)
    shard_for_cell = np.full(cells, 255, dtype=np.uint8)
    methods = np.zeros(cells, dtype=np.uint8)
    method_ids: dict[str, int] = {}
    c_route = np.zeros(cells, dtype=bool)
    resolution_adjudicated = np.zeros(cells, dtype=bool)
    source = np.zeros((nc, nw), dtype=bool)
    anchored = np.zeros(nc, dtype=bool)
    unknown_source_words = set()
    for i, row in bank.iterrows():
        for word in json.loads(row.source_words):
            if word in word_map:
                source[i, word_map[word]] = True
                if re.match(
                    r"^(?:(?:a|an|the)\s+)?" + re.escape(word) + r"(?:s|es)?\b",
                    row.canonical_feature_text,
                    re.I,
                ):
                    anchored[i] = True
            else:
                unknown_source_words.add(word)
    manifest = json.loads(
        (bank_path.parent / "candidate_bank_manifest.json").read_text()
    )
    bank_checks = {
        "candidate_count": nc,
        "word_count": nw,
        "unique_ids": int(bank.candidate_id.nunique()),
        "duplicate_canonical_texts": int(
            bank.canonical_feature_text.duplicated().sum()
        ),
        "empty_texts": int(bank.canonical_feature_text.str.strip().eq("").sum()),
        "candidate_index_valid": bool(
            np.array_equal(bank.candidate_index.to_numpy(), np.arange(nc))
        ),
        "file_hash_matches_manifest": sha256_file(actual_file(bank_path))
        == manifest["candidate_bank_sha256"],
        "inventory_hash_matches_manifest": candidate_inventory_hash(bank)
        == manifest["candidate_inventory_hash"],
        "unknown_source_words": sorted(unknown_source_words),
        "source_object_prefix_candidates": int(anchored.sum()),
        "source_object_prefix_detection": "Conservative heuristic: canonical text begins with a source word (optional article/plural suffix).",
    }
    invalid_frames, shard_rows, input_hashes = [], [], {}
    method_counts, value_counts = Counter(), Counter()
    checks = Counter()
    root = ROOT / "artifacts/v4/judgments/shards"
    for si in range(32):
        shard = root / f"{si:04d}"
        path = actual_file(shard / "feature_resolutions.csv")
        input_hashes[str((shard / "feature_resolutions.csv").relative_to(ROOT))] = (
            sha256_file(path)
        )
        record = json.loads((shard / "v4_shard_manifest.json").read_text())
        total, invalid_count, positive_count = 0, 0, 0
        for frame in pd.read_csv(
            path, chunksize=150000, keep_default_na=False, low_memory=False
        ):
            f = pd.to_numeric(frame.feature_id, errors="raise").to_numpy(dtype=np.int64)
            w = frame.word_normalized.map(word_map).to_numpy()
            if np.any(pd.isna(w)) or np.any((f < 0) | (f >= nc)):
                raise ValueError(f"Unknown cell IDs in shard {si}")
            w = w.astype(np.int64)
            keys = f * nw + w
            unique, counts = np.unique(keys, return_counts=True)
            checks["duplicate_resolution_rows"] += int(
                np.sum(counts - 1) + np.sum(seen[unique] > 0)
            )
            seen[unique] += 1
            shard_for_cell[keys] = si
            checks["mismatched_feature_texts"] += int(
                np.sum(frame.feature_text.to_numpy() != texts[f])
            )
            wrong = np.fromiter(
                (stable_shard(ids[fi], words[wi], 32) != si for fi, wi in zip(f, w)),
                dtype=bool,
                count=len(frame),
            )
            checks["wrong_shard_cells"] += int(wrong.sum())
            values = pd.to_numeric(frame.final_feature_value, errors="coerce").to_numpy(
                dtype=np.float32
            )
            valid = np.isfinite(values) & (values >= 0) & (values <= 4)
            raw[keys] = np.where(valid, values, np.nan)
            invalid_count += int((~valid).sum())
            if not valid.all():
                invalid = frame.loc[~valid].copy()
                invalid.insert(0, "shard_index", si)
                invalid.insert(1, "candidate_id", ids[f[~valid]])
                invalid_frames.append(invalid)
            for method in frame.resolution_method.unique():
                if method not in method_ids:
                    method_ids[method] = len(method_ids) + 1
            methods[keys] = frame.resolution_method.map(method_ids).to_numpy(
                dtype=np.uint8
            )
            resolution_adjudicated[keys] = bool_values(frame.adjudicated)
            checks["needs_human_audit_cells"] += int(
                bool_values(frame.needs_human_audit).sum()
            )
            method_counts.update(frame.resolution_method)
            value_counts.update(map(str, values[valid]))
            total += len(frame)
            positive_count += int(np.sum(valid & (values > 0)))
        shard_rows.append(
            {
                "shard_index": si,
                "rows": total,
                "valid_values": total - invalid_count,
                "invalid_values": invalid_count,
                "positive_cells": positive_count,
                "manifest_complete": record["complete"],
                "matches_manifest_row_count": total == record["resolved_cells"],
                "matches_manifest_invalid_count": invalid_count
                == record["invalid_resolved_values"],
                "protocol_hash": record["protocol_hash"],
            }
        )
        print(
            f"Resolutions {si:02d}/31: {total:,} rows; {invalid_count} invalid",
            flush=True,
        )
    invalid = (
        pd.concat(invalid_frames, ignore_index=True)
        if invalid_frames
        else pd.DataFrame()
    )
    invalid.to_csv(output / "invalid_resolutions.csv", index=False)
    pd.DataFrame(shard_rows).to_csv(output / "shard_checks.csv", index=False)
    checks["missing_resolution_cells"] = int(np.sum(seen == 0))
    checks["invalid_resolution_cells"] = int(np.sum(~np.isfinite(raw)))
    signatures = np.zeros(cells, dtype=np.uint8)
    c_value = np.full(cells, np.nan, dtype=np.float32)
    c_confidence = np.full(cells, np.nan, dtype=np.float32)
    vote_counts, parse_errors = Counter(), []
    if not args.skip_votes:
        for si in range(32):
            path = actual_file(root / f"{si:04d}" / "feature_votes.csv")
            columns = [
                "word_normalized",
                "feature_id",
                "judge_id",
                "judge_model",
                "feature_value",
                "confidence",
                "ambiguous",
                "parse_error",
                "prompt_hash",
            ]
            for frame in pd.read_csv(
                path,
                usecols=columns,
                chunksize=150000,
                keep_default_na=False,
                low_memory=False,
            ):
                f = pd.to_numeric(frame.feature_id, errors="raise").to_numpy(
                    dtype=np.int64
                )
                w = frame.word_normalized.map(word_map).to_numpy()
                if np.any(pd.isna(w)) or np.any((f < 0) | (f >= nc)):
                    raise ValueError(f"Unknown vote IDs in shard {si}")
                keys = f * nw + w.astype(np.int64)
                checks["votes_in_wrong_shard"] += int(
                    np.sum(shard_for_cell[keys] != si)
                )
                checks["unknown_judge_ids"] += int(
                    (~frame.judge_id.isin(list("ABC"))).sum()
                )
                checks["wrong_judge_model"] += int(
                    frame.judge_model.ne("Qwen2.5-72B-Instruct").sum()
                )
                checks.update(check_prompt_sample(frame, texts))
                values = pd.to_numeric(frame.feature_value, errors="coerce").to_numpy(
                    dtype=np.float32
                )
                conf = pd.to_numeric(frame.confidence, errors="coerce").to_numpy(
                    dtype=np.float32
                )
                err = frame.parse_error.str.strip().ne("").to_numpy()
                malformed = (
                    ~np.isfinite(values)
                    | (values < 0)
                    | (values > 4)
                    | ~np.isfinite(conf)
                    | (conf < 0)
                    | (conf > 1)
                )
                checks["invalid_vote_without_parse_flag"] += int(
                    np.sum(malformed & ~err)
                )
                checks["first_pass_parse_error_rows"] += int(err.sum())
                for j, bit in [("A", 1), ("B", 2), ("C", 4)]:
                    mask = frame.judge_id.eq(j).to_numpy()
                    jk = keys[mask]
                    unique, counts = np.unique(jk, return_counts=True)
                    checks["duplicate_vote_rows"] += int(
                        np.sum(counts - 1) + np.sum((signatures[unique] & bit) != 0)
                    )
                    signatures[unique] |= bit
                    vote_counts[j] += int(mask.sum())
                    if j == "C":
                        c_value[jk] = values[mask]
                        c_confidence[jk] = conf[mask]
                        c_route[jk] = (
                            (values[mask] > 0)
                            | bool_values(frame.ambiguous)[mask]
                            | (conf[mask] < 0.8)
                            | malformed[mask]
                            | err[mask]
                        )
            parse = pd.read_csv(
                actual_file(root / f"{si:04d}" / "parse_errors.csv"),
                keep_default_na=False,
            )
            if not parse.empty:
                parse.insert(0, "shard_index", si)
                parse_errors.append(parse)
            print(f"Votes {si:02d}/31 checked", flush=True)
        c_only = signatures == 4
        full = signatures == 7
        method = method_ids.get("prompt_c_high_confidence_negative", -1)
        checks["invalid_panel_signatures"] = int(np.sum(~(c_only | full)))
        checks["c_only_route_violations"] = int(np.sum(c_only & c_route))
        checks["c_only_wrong_resolution_method"] = int(
            np.sum(c_only & (methods != method))
        )
        checks["c_only_nonzero_resolutions"] = int(np.sum(c_only & (raw != 0)))
        checks["c_only_adjudicated"] = int(np.sum(c_only & resolution_adjudicated))
        checks["full_panel_cells"] = int(full.sum())
        checks["c_only_cells"] = int(c_only.sum())
        checks["positive_c_resolved_negative"] = int(np.sum((c_value > 0) & (raw == 0)))
        checks["negative_c_resolved_positive"] = int(np.sum((c_value == 0) & (raw > 0)))
        if parse_errors:
            pd.concat(parse_errors, ignore_index=True).to_csv(
                output / "first_pass_parse_errors.csv", index=False
            )
        (output / "prompt_hash_sample.json").write_text(
            json.dumps(
                {
                    key: int(checks[key])
                    for key in [
                        "sampled_prompt_hash_rows",
                        "sampled_prompt_hash_mismatches",
                    ]
                },
                indent=2,
            )
            + "\n"
        )
    positive = (raw > 0).reshape(nc, nw)
    uncertain = ~np.isfinite(raw.reshape(nc, nw))
    counts = positive.sum(axis=1)
    retention = counts > 3
    complete_features = ~uncertain.any(axis=1)
    certain_retention = retention & complete_features
    source_positive = positive & source
    completion_positive = positive & ~source

    def metrics(mask: np.ndarray) -> dict:
        return {
            "candidates": int(mask.sum()),
            "retained_gt3": int(np.sum(mask & retention)),
            "positive_cells": int(positive[mask].sum()),
            "source_positive_cells": int(source_positive[mask].sum()),
            "source_cells": int(source[mask].sum()),
            "completion_positive_cells": int(completion_positive[mask].sum()),
            "median_positive_objects": (
                float(np.median(counts[mask])) if mask.any() else None
            ),
        }

    sanity = {
        "invalid_cells_kept_as_missing": True,
        "positive_cells_valid_only": int(positive.sum()),
        "positive_rate_among_valid_cells": float(
            positive.sum() / np.isfinite(raw).sum()
        ),
        "zero_positive_candidates": int(np.sum(counts == 0)),
        "one_to_three_positive_candidates": int(np.sum((counts >= 1) & (counts <= 3))),
        "definitely_retained_contexts": int(retention.sum()),
        "possible_retained_contexts_after_recovery": int(
            np.sum(counts + uncertain.sum(axis=1) > 3)
        ),
        "complete_retained_contexts_used_for_geometry": int(certain_retention.sum()),
        "retained_density_valid_positives_lower_bound": float(
            positive[retention].mean()
        ),
        "all_positive_features": int(np.sum(counts == nw)),
        "empty_objects_after_retention": int(
            np.sum(positive[retention].sum(axis=0) == 0)
        ),
        "object_prefix_candidates": metrics(anchored),
        "other_candidates": metrics(~anchored),
        "all_candidates": metrics(np.ones(nc, dtype=bool)),
        "invalid_resolution_methods": (
            dict(Counter(invalid.resolution_method)) if not invalid.empty else {}
        ),
        "invalid_adjudication_triggers": (
            dict(Counter(invalid.adjudication_trigger)) if not invalid.empty else {}
        ),
    }
    packed = np.packbits(positive[certain_retention], axis=1)
    _, pattern_counts = np.unique(packed, axis=0, return_counts=True)
    sanity["unique_retained_binary_patterns_complete_features"] = int(
        len(pattern_counts)
    )
    sanity["redundant_retained_columns_complete_features"] = int(
        packed.shape[0] - len(pattern_counts)
    )
    sanity["largest_duplicate_pattern_group"] = int(pattern_counts.max())
    stats = bank[
        [
            "candidate_index",
            "candidate_id",
            "canonical_feature_text",
            "source_words",
            "source_prompt_families",
            "source_ids",
        ]
    ].copy()
    stats["positive_objects"] = counts
    stats["unresolved_objects"] = uncertain.sum(axis=1)
    stats["retained_gt3"] = retention
    stats["source_positive_objects"] = source_positive.sum(axis=1)
    stats["completed_positive_objects"] = completion_positive.sum(axis=1)
    stats["source_object_prefix"] = anchored
    word_stats = pd.DataFrame(
        {
            "word": words,
            "valid_positive_features": positive.sum(axis=0),
            "retained_positive_features": positive[retention].sum(axis=0),
            "source_positive_features": source_positive.sum(axis=0),
            "completed_positive_features": completion_positive.sum(axis=0),
            "unresolved_cells": uncertain.sum(axis=0),
        }
    )
    word_stats.to_csv(output / "word_statistics.csv", index=False)
    pd.DataFrame(
        {
            "positive_object_count": np.arange(nw + 1),
            "candidate_count": np.bincount(counts, minlength=nw + 1),
        }
    ).to_csv(output / "positive_object_count_distribution.csv", index=False)
    h = human_counts.gt(3).to_numpy()
    h = h[:, h.sum(axis=0) > 3]
    human_ranks = cosine_ranks(h)
    geometry = []

    def geom(name: str, mask: np.ndarray, values: np.ndarray = positive) -> None:
        if mask.any() and np.all(values[mask].sum(axis=0) > 0):
            ranks = cosine_ranks(values[mask].T)
            geometry.append(
                {
                    "condition": name,
                    "contexts": int(mask.sum()),
                    "object_rdm_spearman_vs_human": float(
                        spearmanr(human_ranks, ranks).statistic
                    ),
                    "basis": "Only complete candidate columns; no invalid cells imputed.",
                }
            )

    geom("v4_ensemble_locked_v2_complete_columns", certain_retention)
    geom(
        "v4_ensemble_without_source_prefix_complete_columns",
        certain_retention & ~anchored,
    )
    geom("v4_ensemble_source_prefix_complete_columns", certain_retention & anchored)
    source_counts = source_positive.sum(axis=1)
    geom(
        "v4_ensemble_source_only_complete_columns",
        (source_counts > 3) & complete_features,
        source_positive,
    )
    fixed_path = ROOT / "artifacts/v4/discovery/candidate_bank_v3_1_b_175.csv"
    fixed = pd.read_csv(actual_file(fixed_path)).sort_values("fixed_v3_1_b_order")
    fixed_ids = fixed.candidate_id.tolist()
    fixed_mask = bank.candidate_id.isin(fixed_ids).to_numpy()
    sanity["fixed_bank_candidates"] = len(fixed_ids)
    sanity["fixed_bank_all_in_ensemble"] = bool(
        fixed_mask.sum() == len(fixed_ids) == 175
    )
    sanity["fixed_bank_invalid_cells"] = int(uncertain[fixed_mask].sum())
    sanity["fixed_bank_retained"] = int(np.sum(fixed_mask & retention))
    geom("v4_b_locked_v2_complete_columns", fixed_mask & certain_retention)
    geom(
        "v4_b_source_only_complete_columns",
        fixed_mask & (source_counts > 3) & complete_features,
        source_positive,
    )
    threshold_path = ROOT / "artifacts/v4/judgments/judgment_threshold.json"
    threshold = json.loads(threshold_path.read_text())["selected_rule"]
    if threshold["operator"] != "ge" or threshold["value"] != 1:
        raise ValueError("This audit expects the frozen V4 >=1 calibration")
    calibrated = (raw >= 1).reshape(nc, nw)
    calibrated_retention = calibrated.sum(axis=1) > 3
    geom(
        "v4_ensemble_calibrated_complete_columns",
        calibrated_retention & complete_features,
        calibrated,
    )
    geom(
        "v4_b_calibrated_complete_columns",
        fixed_mask & calibrated_retention & complete_features,
        calibrated,
    )
    sanity["calibrated_positive_cells_valid_only"] = int(calibrated.sum())
    sanity["calibrated_retained_contexts"] = int(calibrated_retention.sum())
    sanity["calibrated_retained_density_lower_bound"] = float(
        calibrated[calibrated_retention].mean()
    )
    sanity["fixed_calibrated_retained_contexts"] = int(
        np.sum(fixed_mask & calibrated_retention)
    )
    sanity["fixed_locked_positive_cells"] = int(positive[fixed_mask].sum())
    sanity["fixed_calibrated_positive_cells"] = int(calibrated[fixed_mask].sum())
    stats["calibrated_positive_objects"] = calibrated.sum(axis=1)
    stats["calibrated_retained_gt3"] = calibrated_retention
    stats.to_csv(output / "candidate_statistics.csv", index=False)
    pd.DataFrame(geometry).to_csv(output / "input_geometry.csv", index=False)
    rng = np.random.default_rng(20261001)
    candidates = set(
        rng.choice(
            np.flatnonzero(certain_retention),
            size=min(30, int(certain_retention.sum())),
            replace=False,
        ).tolist()
    )
    candidates.update(np.argsort(counts)[-15:].tolist())
    for phrase in [
        "has fur",
        "has wings",
        "has a handle",
        "is red",
        "can fly",
        "open top",
        "mug may feature logos",
    ]:
        candidates.update(
            np.flatnonzero(
                bank.canonical_feature_text.str.contains(phrase, regex=False).to_numpy()
            )[:3].tolist()
        )
    examples = []
    for i in sorted(candidates):
        examples.append(
            {
                "candidate_id": ids[i],
                "feature": texts[i],
                "source_words": bank.iloc[i].source_words,
                "positive_object_count": int(counts[i]),
                "positive_objects": json.dumps(
                    [words[w] for w in np.flatnonzero(positive[i])]
                ),
                "unresolved_objects": json.dumps(
                    [words[w] for w in np.flatnonzero(uncertain[i])]
                ),
            }
        )
    pd.DataFrame(examples).to_csv(output / "feature_examples.csv", index=False)
    # Inspect failed adjudication responses, preserving all original failures.
    invalid_keys = (
        set(zip(invalid.word_normalized, invalid.feature_id))
        if not invalid.empty
        else set()
    )
    adj_failures = []
    adj_parse_counts = Counter()
    for si in range(32):
        path = actual_file(root / f"{si:04d}" / "feature_adjudication_votes.csv")
        for frame in pd.read_csv(
            path, chunksize=100000, keep_default_na=False, low_memory=False
        ):
            mask = np.fromiter(
                (
                    (w, int(f)) in invalid_keys
                    for w, f in zip(frame.word_normalized, frame.feature_id)
                ),
                dtype=bool,
                count=len(frame),
            )
            if mask.any():
                selected = frame.loc[mask].copy()
                selected.insert(0, "shard_index", si)
                adj_failures.append(selected)
            errors = frame.parse_error.str.strip().ne("")
            adj_parse_counts["rows"] += len(frame)
            adj_parse_counts["parse_error_rows"] += int(errors.sum())
        print(f"Adjudication {si:02d}/31 checked", flush=True)
    if adj_failures:
        failed = pd.concat(adj_failures, ignore_index=True)
        failed.to_csv(output / "invalid_cell_adjudication_votes.csv", index=False)
        sanity["failed_cell_adjudicator_error_counts"] = dict(
            Counter(failed.parse_error)
        )
        from leuven_expansion.feature_schema import validate_judge_output

        recovered = 0
        for row in failed.itertuples(index=False):
            record, _ = validate_judge_output(
                str(row.raw_json),
                expected_word=row.word_normalized,
                expected_feature_id=int(row.feature_id),
            )
            recovered += record is not None
        sanity["failed_cell_adjudicator_rows_recoverable_by_current_parser"] = recovered
        sanity["offline_type_only_recovery_preview"] = preview_type_only_recovery(
            output
        )
        (output / "offline_recovery_summary.json").write_text(
            json.dumps(sanity["offline_type_only_recovery_preview"], indent=2) + "\n"
        )
    sanity["duplicate_geometry_diagnostic"] = audit_duplicate_geometry(output)
    audit_spotchecks(output)
    final_manifest_path = root.parent / "judgment_manifest.json"
    final_manifest_complete = (
        bool(json.loads(final_manifest_path.read_text()).get("complete"))
        if final_manifest_path.exists()
        else False
    )
    required_matrices = [
        ROOT / f"artifacts/v4/matrices/{condition}.csv"
        for condition in [
            "v4_b_locked_v2",
            "v4_b_calibrated",
            "v4_ensemble_locked_v2",
            "v4_ensemble_calibrated",
        ]
    ]
    readiness_checks = [
        "missing_resolution_cells",
        "invalid_resolution_cells",
        "duplicate_resolution_rows",
        "wrong_shard_cells",
        "mismatched_feature_texts",
        "duplicate_vote_rows",
        "invalid_panel_signatures",
        "c_only_route_violations",
        "c_only_wrong_resolution_method",
        "c_only_nonzero_resolutions",
        "votes_in_wrong_shard",
        "wrong_judge_model",
        "sampled_prompt_hash_mismatches",
    ]
    structurally_ready = (
        not args.skip_votes
        and all(checks[key] == 0 for key in readiness_checks)
        and all(row["manifest_complete"] for row in shard_rows)
        and final_manifest_complete
        and (root.parent / "resolved_feature_values.csv").exists()
    )
    summary = {
        "audit_date": "2026-10-01",
        "expected_cells": cells,
        "bank_checks": bank_checks,
        "resolution_checks": dict(checks),
        "resolution_method_counts": dict(method_counts),
        "resolved_value_counts": dict(value_counts),
        "first_pass_vote_counts": dict(vote_counts),
        "adjudicator_rows": dict(adj_parse_counts),
        "sanity": sanity,
        "geometry": geometry,
        "shards": shard_rows,
        "resolution_file_sha256": input_hashes,
        "votes_verified": not args.skip_votes,
        "training_ready": structurally_ready
        and all(path.exists() for path in required_matrices),
        "finalized_judgment_manifest_exists": (
            root.parent / "judgment_manifest.json"
        ).exists(),
        "finalized_resolved_values_exists": (
            root.parent / "resolved_feature_values.csv"
        ).exists(),
    }
    (output / "audit_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {"checks": dict(checks), "sanity": sanity, "geometry": geometry}, indent=2
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
