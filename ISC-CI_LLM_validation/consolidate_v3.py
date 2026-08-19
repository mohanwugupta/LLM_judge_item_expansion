#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from iscci_validation.consolidation import (
    ConsolidationParameters,
    build_cluster_counts,
    build_phrase_profiles,
    consolidate_phrase_types,
    encode_phrases,
    make_assignments,
    retained_training_matrix,
    summarize_clusters,
    validate_long_data,
)
from iscci_validation.provenance import environment_record, sha256_file, write_json


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent


def split_rejected_clusters(
    clusters: list[list[int]],
    phrases: list[str],
    assignments: pd.DataFrame,
    rejected_cluster_ids: set[str],
) -> list[list[int]]:
    """Undo rejected semantic merges while retaining lexical equivalence groups."""
    phrase_details = assignments.set_index("feature_text_normalized")
    revised: list[list[int]] = []
    for cluster in clusters:
        cluster_id = str(
            phrase_details.loc[phrases[cluster[0]], "cluster_id"]
        )
        if cluster_id not in rejected_cluster_ids:
            revised.append(cluster)
            continue
        by_signature: dict[str, list[int]] = {}
        for phrase_index in cluster:
            signature = str(
                phrase_details.loc[phrases[phrase_index], "lexical_signature"]
            )
            by_signature.setdefault(signature, []).append(phrase_index)
        revised.extend(by_signature.values())
    return revised


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Consolidate V3 Leuven feature generations into ISC-CI matrices."
    )
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs" / "v3_validation.json"
    )
    parser.add_argument(
        "--long-csv",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts"
        / "leuven_feature_generation"
        / "leuven_v3_qwen2_5_72b"
        / "generated_features_long.csv",
    )
    parser.add_argument(
        "--human-csv",
        type=Path,
        default=ROOT
        / "upstream"
        / "IntegratedSemanticsControlContextInference"
        / "data"
        / "leuven_dataset"
        / "leuven_combined_features_consolidated.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts" / "v3_consolidation",
    )
    parser.add_argument(
        "--artifact-prefix",
        default="v3",
        help="Prefix for prompt-specific matrix and assignment files.",
    )
    parser.add_argument(
        "--manual-review-csv",
        type=Path,
        default=ROOT / "configs" / "v3_consolidation_manual_review.csv",
    )
    parser.add_argument("--force-embeddings", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    consolidation_config = config["consolidation"]
    prompts = [str(value) for value in config["prompts"]]
    parameters = ConsolidationParameters(
        embedding_model=consolidation_config["embedding_model"],
        embedding_model_revision=consolidation_config["embedding_model_revision"],
        embedding_similarity_threshold=float(
            consolidation_config["embedding_similarity_threshold"]
        ),
        profile_similarity_threshold=float(
            consolidation_config["profile_similarity_threshold"]
        ),
        nearest_neighbors=int(consolidation_config["nearest_neighbors"]),
        rater_cutoff=int(
            consolidation_config["rater_cutoff_strictly_greater_than"]
        ),
        object_cutoff=int(
            consolidation_config["object_cutoff_strictly_greater_than"]
        ),
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manual_review_path = args.manual_review_csv.resolve()
    manual_review = (
        pd.read_csv(manual_review_path) if manual_review_path.exists() else None
    )
    rejected_cluster_ids = (
        set(
            manual_review.loc[
                manual_review["verdict"].eq("reject"), "cluster_id"
            ]
        )
        if manual_review is not None
        else set()
    )
    long_data = pd.read_csv(args.long_csv)
    long_data["prompt_variant"] = long_data["prompt_variant"].astype(str)
    long_data["feature_text_normalized"] = long_data[
        "feature_text_normalized"
    ].astype(str)
    validate_long_data(long_data, prompts)
    human = pd.read_csv(args.human_csv, index_col=0, encoding="ISO-8859-1")
    words = [str(value) for value in human.index]
    if set(long_data["word_normalized"].unique()) != set(words):
        raise ValueError("V3 and human matrices do not contain the same 293 objects")

    all_phrases = sorted(long_data["feature_text_normalized"].unique())
    embeddings_path = output_dir / "phrase_embeddings.npz"
    if embeddings_path.exists() and not args.force_embeddings:
        cached = np.load(embeddings_path, allow_pickle=False)
        cached_phrases = cached["phrases"].astype(str).tolist()
        if cached_phrases != all_phrases:
            raise ValueError("Cached embeddings do not match current phrase types")
        all_embeddings = cached["embeddings"].astype(np.float32)
    else:
        all_embeddings = encode_phrases(
            all_phrases,
            parameters.embedding_model,
            parameters.embedding_model_revision,
        )
        np.savez(
            embeddings_path,
            phrases=np.asarray(all_phrases),
            embeddings=all_embeddings,
        )
    embedding_by_phrase = {
        phrase: all_embeddings[index] for index, phrase in enumerate(all_phrases)
    }

    primary_assignments: list[pd.DataFrame] = []
    primary_labels: list[pd.DataFrame] = []
    review_candidate_rows: list[pd.DataFrame] = []
    review_candidate_assignments: list[pd.DataFrame] = []
    sensitivity_rows: list[dict[str, object]] = []
    thresholds = [
        float(value)
        for value in consolidation_config["sensitivity_embedding_thresholds"]
    ]
    for prompt in prompts:
        prompt_data = long_data[long_data["prompt_variant"] == prompt].copy()
        phrases = sorted(prompt_data["feature_text_normalized"].unique())
        embeddings = np.stack([embedding_by_phrase[phrase] for phrase in phrases])
        profiles = build_phrase_profiles(prompt_data, phrases, words)
        global_frequency = prompt_data.groupby("feature_text_normalized").size()

        for threshold in thresholds:
            clusters = consolidate_phrase_types(
                phrases,
                embeddings,
                profiles,
                embedding_threshold=threshold,
                profile_threshold=parameters.profile_similarity_threshold,
                nearest_neighbors=parameters.nearest_neighbors,
            )
            assignments = make_assignments(
                prompt,
                phrases,
                clusters,
                embeddings,
                profiles,
                global_frequency,
            )
            counts, _ = build_cluster_counts(prompt_data, assignments, words)
            training_matrix = retained_training_matrix(
                counts, parameters.rater_cutoff, parameters.object_cutoff
            )
            if threshold == parameters.embedding_similarity_threshold:
                proposed_review = (
                    assignments.loc[
                        assignments["cluster_id"].isin(training_matrix.columns)
                        & assignments["cluster_merge_basis"].eq("semantic_profile"),
                        ["cluster_id", "prompt_variant"],
                    ]
                    .drop_duplicates()
                    .copy()
                )
                review_candidate_rows.append(proposed_review)
                review_candidate_assignments.append(
                    assignments.loc[
                        assignments["cluster_id"].isin(
                            proposed_review["cluster_id"]
                        )
                    ].copy()
                )
                if rejected_cluster_ids.intersection(proposed_review["cluster_id"]):
                    clusters = split_rejected_clusters(
                        clusters,
                        phrases,
                        assignments,
                        rejected_cluster_ids,
                    )
                    assignments = make_assignments(
                        prompt,
                        phrases,
                        clusters,
                        embeddings,
                        profiles,
                        global_frequency,
                    )
                    counts, _ = build_cluster_counts(prompt_data, assignments, words)
                    training_matrix = retained_training_matrix(
                        counts, parameters.rater_cutoff, parameters.object_cutoff
                    )
            sensitivity_rows.append(
                summarize_clusters(
                    prompt, threshold, assignments, counts, training_matrix
                )
            )

            if threshold == parameters.embedding_similarity_threshold:
                assignments.to_csv(
                    output_dir
                    / f"{args.artifact_prefix}_{prompt}_cluster_assignments.csv",
                    index=False,
                )
                counts.to_csv(
                    output_dir
                    / f"{args.artifact_prefix}_{prompt}_consolidated_counts.csv"
                )
                training_matrix.to_csv(
                    output_dir / f"{args.artifact_prefix}_{prompt}_training_binary.csv"
                )
                labels = (
                    assignments[
                        [
                            "cluster_id",
                            "canonical_feature",
                            "cluster_size",
                            "cluster_merge_basis",
                            "cluster_signature_count",
                            "within_cluster_embedding_min",
                            "within_cluster_embedding_mean",
                            "within_cluster_profile_min",
                            "within_cluster_profile_mean",
                        ]
                    ]
                    .drop_duplicates("cluster_id")
                    .assign(
                        retained_for_training=lambda frame: frame["cluster_id"].isin(
                            training_matrix.columns
                        )
                    )
                    .sort_values("cluster_id")
                )
                labels.to_csv(
                    output_dir / f"{args.artifact_prefix}_{prompt}_cluster_labels.csv",
                    index=False,
                )
                labels.insert(0, "prompt_variant", prompt)
                primary_labels.append(labels)
                primary_assignments.append(assignments)

    sensitivity = pd.DataFrame(sensitivity_rows).sort_values(
        ["prompt_variant", "embedding_threshold"]
    )
    sensitivity.to_csv(output_dir / "threshold_sensitivity.csv", index=False)
    combined_assignments = pd.concat(primary_assignments, ignore_index=True)
    combined_assignments.to_csv(output_dir / "all_cluster_assignments.csv", index=False)
    combined_labels = pd.concat(primary_labels, ignore_index=True)
    combined_labels.to_csv(output_dir / "all_cluster_labels.csv", index=False)

    review_columns = [
        "cluster_id",
        "prompt_variant",
        "cluster_members",
        "verdict",
        "review_note",
        "reviewed_at",
        "reviewer",
    ]
    retained_semantic = combined_labels.loc[
        combined_labels["retained_for_training"]
        & combined_labels["cluster_merge_basis"].eq("semantic_profile")
    ]
    retained_semantic_ids = set(retained_semantic["cluster_id"])
    review_candidates = pd.concat(review_candidate_rows, ignore_index=True)
    review_candidate_details = pd.concat(
        review_candidate_assignments, ignore_index=True
    )
    proposed_review_ids = set(review_candidates["cluster_id"])
    if not manual_review_path.exists():
        members = (
            review_candidate_details.groupby("cluster_id", observed=True)[
                "feature_text_normalized"
            ]
            .agg(lambda values: " | ".join(sorted(values)))
            .to_dict()
        )
        review_template = review_candidates[
            ["cluster_id", "prompt_variant"]
        ].drop_duplicates()
        review_template["cluster_members"] = review_template["cluster_id"].map(members)
        for column in review_columns[3:]:
            review_template[column] = ""
        template_path = output_dir / "manual_review_required.csv"
        review_template[review_columns].to_csv(template_path, index=False)
        raise ValueError(
            "Retained semantic clusters require manual review. Complete the template at "
            f"{template_path} and pass it with --manual-review-csv."
        )
    assert manual_review is not None
    missing_review_columns = set(review_columns) - set(manual_review.columns)
    if missing_review_columns:
        raise ValueError(
            f"Manual review is missing columns: {sorted(missing_review_columns)}"
        )
    invalid_verdicts = set(manual_review["verdict"]) - {"pass", "reject"}
    if invalid_verdicts:
        raise ValueError(f"Invalid manual-review verdicts: {sorted(invalid_verdicts)}")
    reviewed_ids = set(manual_review["cluster_id"])
    if proposed_review_ids != reviewed_ids:
        raise ValueError(
            "Retained semantic clusters changed and require a new manual review: "
            f"unreviewed={sorted(proposed_review_ids - reviewed_ids)}, "
            f"obsolete={sorted(reviewed_ids - proposed_review_ids)}"
        )
    reviewed_pass_ids = set(
        manual_review.loc[manual_review["verdict"].eq("pass"), "cluster_id"]
    )
    if retained_semantic_ids != reviewed_pass_ids:
        raise ValueError(
            "Manual-review splits do not match the final retained semantic clusters: "
            f"unexpected={sorted(retained_semantic_ids - reviewed_pass_ids)}, "
            f"missing={sorted(reviewed_pass_ids - retained_semantic_ids)}"
        )
    manual_review.to_csv(output_dir / "manual_review.csv", index=False)

    model_snapshot = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--sentence-transformers--all-MiniLM-L6-v2"
        / "snapshots"
        / parameters.embedding_model_revision
        / "model.safetensors"
    )
    manifest = {
        "protocol_version": config["protocol_version"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "long_csv": str(args.long_csv.resolve()),
            "long_csv_sha256": sha256_file(args.long_csv.resolve()),
            "human_csv": str(args.human_csv.resolve()),
            "human_csv_sha256": sha256_file(args.human_csv.resolve()),
            "config": str(args.config.resolve()),
            "config_sha256": sha256_file(args.config.resolve()),
            "manual_review_csv": str(manual_review_path.resolve()),
            "manual_review_csv_sha256": sha256_file(manual_review_path.resolve()),
        },
        "parameters": parameters.__dict__,
        "artifact_prefix": args.artifact_prefix,
        "prompt_handling": "separate conditions; no pooling across A/B/C",
        "response_counting": (
            "deduplicate cluster within response, then count unique response IDs"
        ),
        "threshold_semantics": (
            "strict response_frequency > rater_cutoff, then strict positive_objects "
            "> object_cutoff"
        ),
        "embedding_model_file": str(model_snapshot),
        "embedding_model_file_sha256": sha256_file(model_snapshot),
        "source_rows": int(len(long_data)),
        "source_responses": int(long_data["response_id"].nunique()),
        "source_words": int(long_data["word_normalized"].nunique()),
        "source_phrase_types": int(len(all_phrases)),
        "retained_semantic_clusters_manually_reviewed": len(retained_semantic_ids),
        "proposed_retained_semantic_clusters_reviewed": len(proposed_review_ids),
        "retained_semantic_clusters_rejected_and_split": len(
            rejected_cluster_ids & proposed_review_ids
        ),
        "source_cjk_rows": int(
            long_data["feature_text_normalized"]
            .str.contains(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]", regex=True)
            .sum()
        ),
        "source_cjk_responses": int(
            long_data.loc[
                long_data["feature_text_normalized"].str.contains(
                    r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]", regex=True
                ),
                "response_id",
            ].nunique()
        ),
        "primary_summary": sensitivity[
            sensitivity["embedding_threshold"]
            == parameters.embedding_similarity_threshold
        ].to_dict(orient="records"),
        "environment": environment_record(),
    }
    write_json(output_dir / "manifest.json", manifest)
    print(sensitivity.to_string(index=False))
    print(f"\nWrote {args.artifact_prefix} consolidation artifacts to {output_dir}")


if __name__ == "__main__":
    main()
