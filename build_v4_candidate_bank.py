#!/usr/bin/env python3
"""Build the reviewed global V4 candidate inventory from generation manifests."""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from leuven_expansion.v4 import (
    candidate_inventory_hash,
    json_list,
    sha256_file,
    stable_candidate_id,
    stable_json_hash,
    write_json,
)


ROOT = Path(__file__).resolve().parent
VALIDATION_ROOT = ROOT / "ISC-CI_LLM_validation"
if str(VALIDATION_ROOT) not in sys.path:
    sys.path.insert(0, str(VALIDATION_ROOT))

from iscci_validation.consolidation import (  # noqa: E402
    build_phrase_profiles,
    consolidate_phrase_types,
    encode_phrases,
    make_assignments,
    normalize_phrase,
)


REVIEW_COLUMNS = [
    "merge_candidate_id",
    "member_phrases",
    "merge_basis",
    "verdict",
    "review_note",
    "reviewer",
    "reviewed_at",
]


def load_generation_sources(config: dict[str, object]) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    inventory_rows: list[dict[str, object]] = []
    human_path = ROOT / "data" / "leuven_combined_features_consolidated.csv"
    human_words = pd.read_csv(human_path).iloc[:, 0].map(str).tolist()
    human_word_set = set(human_words)
    human_sha256 = sha256_file(human_path)
    for source in config["sources"]:
        source = dict(source)
        source_dir = (ROOT / str(source["path"])).resolve()
        required = bool(source.get("required", False))
        manifest_path = source_dir / "manifest.json"
        long_path = source_dir / "generated_features_long.csv"
        if not manifest_path.exists() or not long_path.exists():
            if required:
                raise FileNotFoundError(f"Required V4 source is incomplete: {source_dir}")
            inventory_rows.append(
                {**source, "status": "pending", "source_dir": str(source_dir)}
            )
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not manifest.get("finished_at"):
            raise ValueError(f"Generation source manifest is unfinished: {source_dir}")
        pending = manifest.get("pending_after_run")
        if pending not in (None, 0):
            raise ValueError(f"Generation source has pending responses: {source_dir}")
        if manifest.get("parse_errors_total") not in (None, 0):
            raise ValueError(f"Generation source has unresolved parse errors: {source_dir}")
        if manifest.get("input_sha256") != human_sha256:
            raise ValueError(f"Generation source used a different Leuven input: {source_dir}")
        if int(manifest.get("word_count", -1)) != len(human_words):
            raise ValueError(f"Generation source word count is not 293: {source_dir}")
        frame = pd.read_csv(long_path, dtype=str).fillna("")
        required_columns = {
            "response_id",
            "word_normalized",
            "prompt_variant",
            "model",
            "feature_text_normalized",
        }
        missing = required_columns - set(frame.columns)
        if missing:
            raise ValueError(f"{long_path} is missing {sorted(missing)}")
        frame["source_id"] = str(source["source_id"])
        frame["source_version"] = str(source["source_version"])
        frame["generation_round"] = int(source["generation_round"])
        frame["source_model"] = str(manifest.get("model", frame["model"].iloc[0]))
        frame["source_response_id"] = frame["response_id"]
        frame["response_id"] = frame["source_id"] + ":" + frame["response_id"]
        frame["feature_text_normalized"] = frame["feature_text_normalized"].map(
            normalize_phrase
        )
        malformed = frame["feature_text_normalized"].eq("")
        frame = frame.loc[~malformed].copy()
        response_count = frame["response_id"].nunique()
        if response_count != int(manifest.get("valid_responses_total", -1)):
            raise ValueError(f"Generation response count disagrees with manifest: {source_dir}")
        if set(frame["word_normalized"]) != human_word_set:
            raise ValueError(f"Generation source word inventory differs from Leuven: {source_dir}")
        prompt_variants = set(map(str, manifest.get("prompt_variants", [])))
        if set(frame["prompt_variant"]) != prompt_variants:
            raise ValueError(f"Generation prompt inventory disagrees with manifest: {source_dir}")
        response_prompt_counts = (
            frame[["source_response_id", "prompt_variant"]]
            .drop_duplicates()
            .groupby("prompt_variant", observed=True)
            .size()
            .to_dict()
        )
        expected_prompt_counts = {
            str(key): int(value)
            for key, value in manifest.get("valid_responses_by_prompt", {}).items()
        }
        if response_prompt_counts != expected_prompt_counts:
            raise ValueError(f"Generation prompt counts disagree with manifest: {source_dir}")
        frames.append(frame)
        inventory_rows.append(
            {
                **source,
                "status": "loaded",
                "source_dir": str(source_dir),
                "manifest_sha256": sha256_file(manifest_path),
                "long_csv_sha256": sha256_file(long_path),
                "rows": len(frame),
                "responses": response_count,
                "phrase_types": frame["feature_text_normalized"].nunique(),
                "model": manifest.get("model"),
                "model_revision": manifest.get("model_revision", manifest.get("model")),
                "model_source_path": manifest.get("model_source_path"),
                "prompt_variants": json.dumps(manifest.get("prompt_variants", [])),
                "prompt_sha256_by_variant": json.dumps(
                    manifest.get("prompt_sha256_by_variant", {}), sort_keys=True
                ),
                "base_seed": manifest.get("base_seed"),
                "parse_errors_total": manifest.get("parse_errors_total"),
            }
        )
    if not frames:
        raise ValueError("No complete generation source is available")
    return pd.concat(frames, ignore_index=True), pd.DataFrame(inventory_rows)


def load_fixed_v3_1_b(
    normalization_version: str,
) -> tuple[pd.DataFrame, dict[str, str], str]:
    root = VALIDATION_ROOT / "artifacts" / "v3_1_consolidation"
    labels = pd.read_csv(root / "v3_1_B_cluster_labels.csv")
    assignments = pd.read_csv(root / "v3_1_B_cluster_assignments.csv")
    labels = labels.loc[labels["retained_for_training"].astype(bool)].copy()
    phrase_to_candidate: dict[str, str] = {}
    rows: list[dict[str, object]] = []
    for order, label in enumerate(labels.itertuples(index=False)):
        members = sorted(
            assignments.loc[
                assignments["cluster_id"].eq(label.cluster_id),
                "feature_text_normalized",
            ].map(normalize_phrase)
        )
        candidate_id = stable_candidate_id(
            members,
            normalization_version,
            namespace=f"fixed-v3.1-b:{label.cluster_id}",
        )
        phrase_to_candidate.update({member: candidate_id for member in members})
        rows.append(
            {
                "candidate_id": candidate_id,
                "canonical_feature_text": label.canonical_feature,
                "member_phrases": json.dumps(members, ensure_ascii=True),
                "fixed_v3_1_b_cluster_id": label.cluster_id,
                "fixed_v3_1_b_order": order,
                "normalization_version": normalization_version,
                "merge_review_status": "imported_locked_v3_1_b",
            }
        )
    fixed = pd.DataFrame(rows)
    inventory_hash = stable_json_hash(
        fixed[
            ["fixed_v3_1_b_order", "fixed_v3_1_b_cluster_id", "canonical_feature_text"]
        ].to_dict(orient="records")
    )
    fixed["fixed_v3_1_b_inventory_hash"] = inventory_hash
    if len(fixed) != 175:
        raise ValueError(f"Expected 175 fixed V3.1-B contexts, found {len(fixed)}")
    return fixed, phrase_to_candidate, inventory_hash


def proposed_clusters(
    long_data: pd.DataFrame,
    excluded_phrases: set[str],
    config: dict[str, object],
    output_dir: Path,
    force_embeddings: bool,
) -> tuple[list[list[str]], pd.DataFrame]:
    step_start = time.monotonic()

    def step(label: str) -> None:
        nonlocal step_start
        now = time.monotonic()
        print(f"[timing] proposed_clusters/{label}: {now - step_start:.1f}s", file=sys.stderr, flush=True)
        step_start = now

    consolidation = dict(config["consolidation"])
    phrases = sorted(
        set(long_data["feature_text_normalized"].unique()) - excluded_phrases
    )
    if not phrases:
        return [], pd.DataFrame(columns=REVIEW_COLUMNS)
    words = sorted(long_data["word_normalized"].unique())
    print(
        f"[timing] proposed_clusters: {len(phrases)} unique phrases, {len(words)} words",
        file=sys.stderr,
        flush=True,
    )
    embedding_path = output_dir / "candidate_phrase_embeddings.npz"
    if embedding_path.exists() and not force_embeddings:
        cached = np.load(embedding_path, allow_pickle=False)
        if cached["phrases"].astype(str).tolist() == phrases:
            embeddings = cached["embeddings"].astype(np.float32)
        else:
            embeddings = encode_phrases(
                phrases,
                str(consolidation["embedding_model"]),
                str(consolidation["embedding_model_revision"]),
            )
            np.savez(
                embedding_path,
                phrases=np.asarray(phrases),
                embeddings=embeddings,
            )
    else:
        embeddings = encode_phrases(
            phrases,
            str(consolidation["embedding_model"]),
            str(consolidation["embedding_model_revision"]),
        )
        np.savez(
            embedding_path,
            phrases=np.asarray(phrases),
            embeddings=embeddings,
        )
    step("embeddings (load cache or encode)")
    subset = long_data.loc[long_data["feature_text_normalized"].isin(phrases)]
    profiles = build_phrase_profiles(subset, phrases, words)
    step("build_phrase_profiles")
    clusters = consolidate_phrase_types(
        phrases,
        embeddings,
        profiles,
        embedding_threshold=float(consolidation["embedding_similarity_threshold"]),
        profile_threshold=float(consolidation["profile_similarity_threshold"]),
        nearest_neighbors=int(consolidation["nearest_neighbors"]),
    )
    step("consolidate_phrase_types (nearest-neighbor + union-find)")
    frequencies = subset.groupby("feature_text_normalized").size()
    assignments = make_assignments(
        "v4", phrases, clusters, embeddings, profiles, frequencies
    )
    step("make_assignments")
    proposed: list[list[str]] = []
    review_rows: list[dict[str, object]] = []
    for cluster_id, group in assignments.groupby("cluster_id", sort=True):
        members = sorted(group["feature_text_normalized"].tolist())
        proposed.append(members)
        if len(members) > 1:
            review_rows.append(
                {
                    "merge_candidate_id": stable_json_hash(
                        {"normalization_version": consolidation["normalization_version"], "members": members}
                    )[:20],
                    "member_phrases": json.dumps(members, ensure_ascii=True),
                    "merge_basis": group["cluster_merge_basis"].iloc[0],
                    "verdict": "",
                    "review_note": "",
                    "reviewer": "",
                    "reviewed_at": "",
                }
            )
    step("build proposed/review rows")
    return proposed, pd.DataFrame(review_rows, columns=REVIEW_COLUMNS)


def auto_approve_pending_verdicts(
    merged_review: pd.DataFrame, consolidation: dict[str, object]
) -> pd.DataFrame:
    """Auto-pass proposed merges left blank by a human reviewer.

    Cluster membership is already fully determined by
    ``embedding_similarity_threshold`` / ``profile_similarity_threshold``
    (the same automated-threshold consolidation used, with no per-cluster
    manual check, for the frozen V3.1-B condition; see
    ``ISC-CI_LLM_validation/artifacts/v3_1_consolidation/threshold_sensitivity.csv``).
    This keeps V4 consistent with that precedent: a merge is accepted
    whenever the configured thresholds already grouped the phrases
    together, and the automated decision is still recorded (not silently
    applied) so it remains auditable.
    """
    merged_review = merged_review.copy()
    pending = merged_review["verdict"].eq("")
    if not pending.any():
        return merged_review
    note = (
        "auto-approved: cluster formed by embedding_similarity_threshold="
        f"{consolidation['embedding_similarity_threshold']} and "
        f"profile_similarity_threshold={consolidation['profile_similarity_threshold']}; "
        "consistent with V3.1-B automated consolidation precedent"
    )
    now = datetime.now(timezone.utc).isoformat()
    merged_review.loc[pending, "verdict"] = "pass"
    merged_review.loc[pending, "reviewer"] = "automated:embedding_threshold"
    merged_review.loc[pending, "reviewed_at"] = now
    merged_review.loc[pending, "review_note"] = note
    return merged_review


def clusters_from_verdicts(
    proposed: list[list[str]], merged_review: pd.DataFrame
) -> list[list[str]]:
    verdict_by_members = {
        tuple(json.loads(row.member_phrases)): row.verdict
        for row in merged_review.itertuples(index=False)
    }
    final: list[list[str]] = []
    for members in proposed:
        if len(members) == 1:
            final.append(members)
            continue
        verdict = verdict_by_members.get(tuple(members), "")
        if verdict == "pass":
            final.append(members)
        elif verdict == "reject":
            final.extend([[member] for member in members])
    return final


def apply_review(
    proposed: list[list[str]], review_candidates: pd.DataFrame, review_path: Path
) -> tuple[list[list[str]], pd.DataFrame]:
    if review_path.exists():
        review = pd.read_csv(review_path, dtype=str).fillna("")
    else:
        review = pd.DataFrame(columns=REVIEW_COLUMNS)
    missing_columns = set(REVIEW_COLUMNS) - set(review.columns)
    if missing_columns:
        raise ValueError(f"Merge review is missing columns: {sorted(missing_columns)}")
    if review["merge_candidate_id"].duplicated().any():
        raise ValueError("Merge-review IDs must be unique")
    merged = review_candidates.drop(columns=["verdict", "review_note", "reviewer", "reviewed_at"]).merge(
        review,
        on=["merge_candidate_id", "member_phrases", "merge_basis"],
        how="left",
        validate="one_to_one",
    ).fillna("")
    unknown_ids = set(review["merge_candidate_id"]) - set(
        review_candidates["merge_candidate_id"]
    )
    if unknown_ids:
        raise ValueError(f"Merge review contains obsolete IDs: {sorted(unknown_ids)[:10]}")
    final = clusters_from_verdicts(proposed, merged)
    return final, merged


def build_bank(
    long_data: pd.DataFrame,
    clusters: list[list[str]],
    fixed: pd.DataFrame,
    fixed_phrase_ids: dict[str, str],
    normalization_version: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    phrase_to_candidate = dict(fixed_phrase_ids)
    candidate_rows = fixed.to_dict(orient="records")
    for members in clusters:
        candidate_id = stable_candidate_id(members, normalization_version)
        phrase_to_candidate.update({member: candidate_id for member in members})
        frequencies = long_data.loc[
            long_data["feature_text_normalized"].isin(members)
        ].groupby("feature_text_normalized").size()
        canonical = sorted(members, key=lambda value: (-int(frequencies.get(value, 0)), len(value), value))[0]
        candidate_rows.append(
            {
                "candidate_id": candidate_id,
                "canonical_feature_text": canonical,
                "member_phrases": json.dumps(sorted(members), ensure_ascii=True),
                "fixed_v3_1_b_cluster_id": "",
                "fixed_v3_1_b_order": "",
                "normalization_version": normalization_version,
                "merge_review_status": "reviewed_pass" if len(members) > 1 else "singleton",
            }
        )
    bank = pd.DataFrame(candidate_rows).drop_duplicates("candidate_id")
    mapped = long_data.copy()
    mapped["candidate_id"] = mapped["feature_text_normalized"].map(phrase_to_candidate)
    if mapped["candidate_id"].isna().any():
        examples = mapped.loc[mapped["candidate_id"].isna(), "feature_text_normalized"].unique()[:10]
        raise ValueError(f"Generated phrases lack candidate assignments: {examples}")
    provenance_rows: list[dict[str, object]] = []
    for candidate_id, group in mapped.groupby("candidate_id", sort=False):
        provenance_rows.append(
            {
                "candidate_id": candidate_id,
                "source_words": json_list(group["word_normalized"].tolist()),
                "source_prompt_families": json_list(group["prompt_variant"].tolist()),
                "source_models": json_list(group["source_model"].tolist()),
                "source_rounds": json_list(group["generation_round"].tolist()),
                "source_ids": json_list(group["source_id"].tolist()),
                "n_independent_responses": group["response_id"].nunique(),
                "n_source_words": group["word_normalized"].nunique(),
                "n_source_models": group["source_model"].nunique(),
            }
        )
    bank = bank.merge(pd.DataFrame(provenance_rows), on="candidate_id", validate="one_to_one")
    bank = bank.sort_values(["canonical_feature_text", "candidate_id"]).reset_index(drop=True)
    bank.insert(0, "candidate_index", range(len(bank)))
    inventory_hash = candidate_inventory_hash(bank)
    bank["candidate_inventory_hash"] = inventory_hash
    assignment_columns = [
        "candidate_id",
        "feature_text_normalized",
        "source_id",
        "source_version",
        "generation_round",
        "source_model",
        "prompt_variant",
        "word_normalized",
        "response_id",
        "source_response_id",
    ]
    return bank, mapped[assignment_columns].drop_duplicates().sort_values(assignment_columns[:2])


def candidate_rarefaction(mapped: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for source_id, source in mapped.groupby("source_id", sort=True):
        for prompt, group in source.groupby("prompt_variant", sort=True):
            replicate = pd.to_numeric(group.get("replicate_id", 0), errors="coerce").fillna(0)
            for responses in (5, 10, 15, 20):
                selected = group.loc[replicate < responses]
                rows.append(
                    {
                        "source_id": source_id,
                        "prompt_family": prompt,
                        "responses_per_word": responses,
                        "phrase_types": selected["feature_text_normalized"].nunique(),
                        "candidate_count": selected["candidate_id"].nunique(),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manual-review", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--force-embeddings", action="store_true")
    parser.add_argument(
        "--auto-approve-merges",
        action="store_true",
        help=(
            "Auto-pass any proposed merge left blank in --manual-review, using the "
            "same embedding_similarity_threshold/profile_similarity_threshold that "
            "already determined cluster membership. This matches the automated, "
            "no-per-cluster-review consolidation precedent used for the frozen "
            "V3.1-B condition (see "
            "ISC-CI_LLM_validation/artifacts/v3_1_consolidation/threshold_sensitivity.csv). "
            "Decisions are still written back to --manual-review with "
            "reviewer=automated:embedding_threshold for auditability; explicit "
            "'reject' verdicts already present are never overridden."
        ),
    )
    args = parser.parse_args()

    def checkpoint(label: str, _last: list[float] = [time.monotonic()]) -> None:
        now = time.monotonic()
        print(f"[timing] {label}: {now - _last[0]:.1f}s", file=sys.stderr, flush=True)
        _last[0] = now

    config = json.loads(args.config.read_text(encoding="utf-8"))
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    long_data, source_inventory = load_generation_sources(config)
    checkpoint("load_generation_sources")
    source_inventory.to_csv(output / "source_inventory.csv", index=False)
    normalization_version = str(config["consolidation"]["normalization_version"])
    fixed, fixed_phrase_ids, fixed_hash = load_fixed_v3_1_b(normalization_version)
    checkpoint("load_fixed_v3_1_b")
    fixed_source = long_data.loc[
        long_data["feature_text_normalized"].isin(fixed_phrase_ids)
    ].copy()
    fixed_bank, _ = build_bank(
        fixed_source,
        [],
        fixed,
        fixed_phrase_ids,
        normalization_version,
    )
    checkpoint("build_bank (fixed 175 subset)")
    fixed_bank = fixed_bank.sort_values("fixed_v3_1_b_order")
    fixed_bank["candidate_index"] = range(len(fixed_bank))
    fixed_bank["candidate_inventory_hash"] = fixed_hash
    fixed_bank.to_csv(output / "candidate_bank_v3_1_b_175.csv", index=False)
    proposed, review_candidates = proposed_clusters(
        long_data,
        set(fixed_phrase_ids),
        config,
        output,
        args.force_embeddings,
    )
    checkpoint("proposed_clusters (embeddings + nearest-neighbor merge)")
    review_candidates.to_csv(output / "candidate_merge_candidates.csv", index=False)
    final_clusters, merged_review = apply_review(
        proposed, review_candidates, args.manual_review.resolve()
    )
    checkpoint("apply_review")
    if args.auto_approve_merges:
        merged_review = auto_approve_pending_verdicts(
            merged_review, config["consolidation"]
        )
        final_clusters = clusters_from_verdicts(proposed, merged_review)
        # Persist the automated decisions back to the configured manual-review
        # file so they remain a documented, auditable record rather than a
        # silent runtime override.
        merged_review.to_csv(args.manual_review.resolve(), index=False)
        checkpoint("auto_approve_pending_verdicts")
    merged_review.to_csv(output / "candidate_merge_review.csv", index=False)
    pending_review = merged_review.loc[~merged_review["verdict"].isin(["pass", "reject"])]
    if not pending_review.empty:
        template = output / "candidate_merge_review_required.csv"
        merged_review.to_csv(template, index=False)
        primary_pending = source_inventory.loc[
            source_inventory.get("required_for_primary_freeze", False)
            .astype(str)
            .str.lower()
            .isin(["true", "1"])
            & ~source_inventory["status"].eq("loaded")
        ]
        if not primary_pending.empty:
            raise ValueError(
                "The merge list is preliminary because primary discovery sources are "
                f"pending: {primary_pending['source_id'].tolist()}"
            )
        configured_review = pd.read_csv(args.manual_review, dtype=str).fillna("")
        if configured_review.empty:
            merged_review.to_csv(args.manual_review, index=False)
        raise ValueError(
            f"{len(pending_review)} proposed merges require review; complete {template}"
        )

    bank, assignments = build_bank(
        long_data,
        final_clusters,
        fixed,
        fixed_phrase_ids,
        normalization_version,
    )
    fixed_ids = set(fixed["candidate_id"])
    fixed_bank = bank.loc[bank["candidate_id"].isin(fixed_ids)].copy()
    fixed_bank = fixed_bank.sort_values("fixed_v3_1_b_order")
    fixed_bank["candidate_index"] = range(len(fixed_bank))
    fixed_bank["candidate_inventory_hash"] = fixed_hash

    bank.to_csv(output / "candidate_bank.csv", index=False)
    fixed_bank.to_csv(output / "candidate_bank_v3_1_b_175.csv", index=False)
    assignments.to_csv(output / "candidate_phrase_assignments.csv", index=False)
    candidate_rarefaction(assignments.merge(
        long_data[["response_id", "replicate_id"]].drop_duplicates("response_id"),
        on="response_id",
        how="left",
        validate="many_to_one",
    )).to_csv(output / "candidate_rarefaction.csv", index=False)

    source_summary = (
        assignments.groupby(["source_id", "source_version", "generation_round", "source_model", "prompt_variant"], observed=True)
        .agg(rows=("response_id", "size"), phrase_types=("feature_text_normalized", "nunique"), candidates=("candidate_id", "nunique"))
        .reset_index()
    )
    source_summary.to_csv(output / "candidate_source_summary.csv", index=False)

    human = pd.read_csv(ROOT / "data" / "leuven_combined_features_consolidated.csv", index_col=0, encoding="ISO-8859-1")
    retained = human.gt(3).sum(axis=0).gt(3)
    human_normalized = {normalize_phrase(value): value for value in human.columns}
    mapping = bank[["candidate_id", "canonical_feature_text"]].copy()
    mapping["human_feature"] = mapping["canonical_feature_text"].map(
        lambda value: human_normalized.get(normalize_phrase(value), "")
    )
    mapping["human_feature_retained"] = mapping["human_feature"].map(retained).fillna(False)
    mapping["mapping_method"] = np.where(mapping["human_feature"].ne(""), "exact", "")
    mapping.to_csv(output / "candidate_human_feature_mapping.csv", index=False)

    manifest = {
        "protocol_version": config["protocol_version"],
        "config": str(args.config.resolve()),
        "config_sha256": sha256_file(args.config),
        "manual_review": str(args.manual_review.resolve()),
        "manual_review_sha256": sha256_file(args.manual_review),
        "candidate_count": len(bank),
        "candidate_inventory_hash": bank["candidate_inventory_hash"].iloc[0],
        "fixed_v3_1_b_count": len(fixed_bank),
        "fixed_v3_1_b_inventory_hash": fixed_hash,
        "source_count": int(source_inventory["status"].eq("loaded").sum()),
        "source_inventory_sha256": sha256_file(output / "source_inventory.csv"),
        "candidate_bank_sha256": sha256_file(output / "candidate_bank.csv"),
        "merge_candidates": len(review_candidates),
        "merge_passed": int(merged_review["verdict"].eq("pass").sum()),
        "merge_rejected": int(merged_review["verdict"].eq("reject").sum()),
        "pre_atomic_filter": "none; valid singleton and rare candidates retained",
        "human_mapping_used_for_bank": False,
    }
    write_json(output / "candidate_bank_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
