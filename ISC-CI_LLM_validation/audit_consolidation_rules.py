#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from iscci_validation.consolidation import (
    TOKEN_RE,
    UnionFind,
    _all_cross_pairs_pass,
    build_cluster_counts,
    build_phrase_profiles,
    lexical_signature,
    make_assignments,
    nearest_neighbor_edges,
    retained_training_matrix,
)
from iscci_validation.dataio import load_human_matrix
from iscci_validation.evaluation import _pearson_from_ranks, _ranked_cosine_rdm
from iscci_validation.provenance import sha256_file, write_json


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
CONSOLIDATION = ROOT / "artifacts" / "v3_consolidation"
OUTPUT = ROOT / "reports" / "consolidation_audit"
MODAL_TOKENS = frozenset({"can", "could"})
FREQUENCY_TOKENS = frozenset(
    {"commonly", "frequently", "generally", "often", "typically", "usually"}
)


CONFIGURATIONS = (
    {
        "configuration": "exact_only",
        "lexical_mode": "none",
        "embedding_threshold": None,
        "profile_threshold": None,
    },
    {
        "configuration": "conservative_lexical_only",
        "lexical_mode": "conservative",
        "embedding_threshold": None,
        "profile_threshold": None,
    },
    {
        "configuration": "current_lexical_only",
        "lexical_mode": "current",
        "embedding_threshold": None,
        "profile_threshold": None,
    },
    {
        "configuration": "stricter_0.90_profile_0.75",
        "lexical_mode": "current",
        "embedding_threshold": 0.90,
        "profile_threshold": 0.75,
    },
    {
        "configuration": "primary_0.85_profile_0.50",
        "lexical_mode": "current",
        "embedding_threshold": 0.85,
        "profile_threshold": 0.50,
    },
    {
        "configuration": "lower_embedding_0.80_profile_0.50",
        "lexical_mode": "current",
        "embedding_threshold": 0.80,
        "profile_threshold": 0.50,
    },
    {
        "configuration": "lower_profile_0.85_profile_0.25",
        "lexical_mode": "current",
        "embedding_threshold": 0.85,
        "profile_threshold": 0.25,
    },
    {
        "configuration": "relaxed_0.80_profile_0.25",
        "lexical_mode": "current",
        "embedding_threshold": 0.80,
        "profile_threshold": 0.25,
    },
    {
        "configuration": "very_relaxed_0.75_profile_0.10",
        "lexical_mode": "current",
        "embedding_threshold": 0.75,
        "profile_threshold": 0.10,
    },
    {
        "configuration": "embedding_only_0.85",
        "lexical_mode": "current",
        "embedding_threshold": 0.85,
        "profile_threshold": 0.0,
    },
)


def phrase_markers(phrase: str) -> tuple[bool, bool]:
    tokens = set(TOKEN_RE.findall(str(phrase).lower()))
    return bool(tokens & MODAL_TOKENS), bool(tokens & FREQUENCY_TOKENS)


def initialize_clusters(phrases: list[str], lexical_mode: str) -> UnionFind:
    union_find = UnionFind(len(phrases))
    if lexical_mode == "none":
        return union_find

    groups: dict[tuple[object, ...], list[int]] = defaultdict(list)
    for index, phrase in enumerate(phrases):
        signature = lexical_signature(phrase)
        if not signature:
            key: tuple[object, ...] = ("empty", index)
        elif lexical_mode == "conservative":
            modal, frequency = phrase_markers(phrase)
            key = ("signature", signature, modal, frequency)
        else:
            key = ("signature", signature)
        groups[key].append(index)
    for members in groups.values():
        for member in members[1:]:
            union_find.union(members[0], member)
    return union_find


def consolidate_with_cached_edges(
    phrases: list[str],
    embeddings: np.ndarray,
    profiles: np.ndarray,
    edges: list[tuple[float, int, int]],
    lexical_mode: str,
    embedding_threshold: float | None,
    profile_threshold: float | None,
) -> list[list[int]]:
    union_find = initialize_clusters(phrases, lexical_mode)
    if embedding_threshold is not None and profile_threshold is not None:
        for score, left, right in edges:
            if score < embedding_threshold:
                break
            left_root = union_find.find(left)
            right_root = union_find.find(right)
            if left_root == right_root:
                continue
            if _all_cross_pairs_pass(
                union_find.members[left_root],
                union_find.members[right_root],
                phrases,
                embeddings,
                profiles,
                embedding_threshold,
                profile_threshold,
            ):
                union_find.union(left_root, right_root)
    return sorted(
        (sorted(members) for members in union_find.members.values()),
        key=lambda members: members[0],
    )


def audit_primary_clusters(
    prompt: str,
    prompt_data: pd.DataFrame,
    assignments: pd.DataFrame,
    counts: pd.DataFrame,
    retained_ids: set[str],
) -> pd.DataFrame:
    phrase_to_cluster = assignments.set_index("feature_text_normalized")[
        "cluster_id"
    ]
    conservative = prompt_data.copy()
    conservative["cluster_id"] = conservative["feature_text_normalized"].map(
        phrase_to_cluster
    )
    conservative["marker"] = conservative["feature_text_normalized"].map(
        phrase_markers
    )
    conservative_positive_objects = (
        conservative.drop_duplicates(
            ["response_id", "word_normalized", "cluster_id", "marker"]
        )
        .groupby(["cluster_id", "marker", "word_normalized"], observed=True)
        .size()
        .gt(3)
        .groupby(["cluster_id", "marker"], observed=True)
        .sum()
    )
    conservative_max = conservative_positive_objects.groupby(
        "cluster_id", observed=True
    ).max()
    conservative_group_count = conservative_positive_objects.groupby(
        "cluster_id", observed=True
    ).size()
    exact_counts = (
        prompt_data.drop_duplicates(
            ["response_id", "word_normalized", "feature_text_normalized"]
        )
        .groupby(["feature_text_normalized", "word_normalized"], observed=True)
        .size()
    )
    rows: list[dict[str, object]] = []
    for cluster_id, members_frame in assignments.groupby("cluster_id", observed=True):
        if cluster_id not in retained_ids:
            continue
        members = members_frame["feature_text_normalized"].astype(str).tolist()
        member_positive_objects = []
        max_member_counts = pd.Series(0, index=counts.index, dtype=np.int64)
        for phrase in members:
            phrase_counts = exact_counts.loc[phrase] if phrase in exact_counts.index else None
            aligned = (
                phrase_counts.reindex(counts.index, fill_value=0)
                if phrase_counts is not None
                else pd.Series(0, index=counts.index)
            )
            member_positive_objects.append(int(aligned.gt(3).sum()))
            max_member_counts = np.maximum(max_member_counts, aligned)
        cluster_positive = counts[cluster_id].gt(3)
        modal_values = {phrase_markers(phrase)[0] for phrase in members}
        frequency_values = {phrase_markers(phrase)[1] for phrase in members}
        rows.append(
            {
                "prompt_variant": prompt,
                "cluster_id": cluster_id,
                "canonical_feature": members_frame["canonical_feature"].iloc[0],
                "merge_basis": members_frame["cluster_merge_basis"].iloc[0],
                "cluster_size": len(members),
                "members": " | ".join(members),
                "positive_objects": int(cluster_positive.sum()),
                "positive_objects_created_by_merge": int(
                    (cluster_positive & pd.Series(max_member_counts, index=counts.index).le(3)).sum()
                ),
                "max_exact_member_positive_objects": max(member_positive_objects),
                "task_depends_on_merge": max(member_positive_objects) <= 3,
                "conservative_subclusters": int(
                    conservative_group_count.get(cluster_id, 0)
                ),
                "max_conservative_positive_objects": int(
                    conservative_max.get(cluster_id, 0)
                ),
                "survives_conservative_split": bool(
                    conservative_max.get(cluster_id, 0) > 3
                ),
                "modal_mismatch": len(modal_values) > 1,
                "frequency_mismatch": len(frequency_values) > 1,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    config_path = ROOT / "configs" / "v3_validation.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    source_path = (
        PROJECT_ROOT
        / "artifacts"
        / "leuven_feature_generation"
        / "leuven_v3_qwen2_5_72b"
        / "generated_features_long.csv"
    )
    human_path = (
        ROOT
        / "upstream"
        / "IntegratedSemanticsControlContextInference"
        / "data"
        / "leuven_dataset"
        / "leuven_combined_features_consolidated.csv"
    )
    long_data = pd.read_csv(source_path)
    _, human = load_human_matrix(human_path)
    words = human.index.astype(str).tolist()
    human_ranks = _ranked_cosine_rdm(human.values.astype(np.float32))
    cached = np.load(CONSOLIDATION / "phrase_embeddings.npz", allow_pickle=False)
    all_phrases = cached["phrases"].astype(str).tolist()
    all_embeddings = cached["embeddings"].astype(np.float32)
    embedding_by_phrase = dict(zip(all_phrases, all_embeddings))

    summary_rows: list[dict[str, object]] = []
    primary_audits: list[pd.DataFrame] = []
    relaxed_cluster_rows: list[pd.DataFrame] = []
    for prompt in config["prompts"]:
        print(f"Auditing prompt {prompt}...")
        prompt_data = long_data[long_data["prompt_variant"].eq(prompt)].copy()
        phrases = sorted(prompt_data["feature_text_normalized"].astype(str).unique())
        embeddings = np.stack([embedding_by_phrase[phrase] for phrase in phrases])
        profiles = build_phrase_profiles(prompt_data, phrases, words)
        frequencies = prompt_data.groupby("feature_text_normalized").size()
        edges = nearest_neighbor_edges(embeddings, 0.75, 128)

        for specification in CONFIGURATIONS:
            clusters = consolidate_with_cached_edges(
                phrases,
                embeddings,
                profiles,
                edges,
                lexical_mode=str(specification["lexical_mode"]),
                embedding_threshold=specification["embedding_threshold"],
                profile_threshold=specification["profile_threshold"],
            )
            assignments = make_assignments(
                prompt, phrases, clusters, embeddings, profiles, frequencies
            )
            counts, _ = build_cluster_counts(prompt_data, assignments, words)
            matrix = retained_training_matrix(counts, 3, 3)
            matrix_ranks = _ranked_cosine_rdm(matrix.values.astype(np.float32))
            cluster_sizes = assignments.groupby("cluster_id", observed=True).size()
            retained_assignments = assignments[
                assignments["cluster_id"].isin(matrix.columns)
            ]
            retained_sizes = retained_assignments.groupby(
                "cluster_id", observed=True
            ).size()
            summary_rows.append(
                {
                    "prompt_variant": prompt,
                    **specification,
                    "clusters_total": int(cluster_sizes.size),
                    "merged_phrase_types": int(len(phrases) - cluster_sizes.size),
                    "multi_phrase_clusters": int(cluster_sizes.gt(1).sum()),
                    "retained_tasks": int(matrix.shape[1]),
                    "retained_positive_cells": int(matrix.values.sum()),
                    "retained_multi_phrase_tasks": int(retained_sizes.gt(1).sum()),
                    "object_rdm_spearman_vs_human": _pearson_from_ranks(
                        matrix_ranks, human_ranks
                    ),
                }
            )

            if specification["configuration"] == "primary_0.85_profile_0.50":
                primary_audits.append(
                    audit_primary_clusters(
                        prompt,
                        prompt_data,
                        assignments,
                        counts,
                        set(matrix.columns),
                    )
                )
            if specification["configuration"] in {
                "lower_profile_0.85_profile_0.25",
                "relaxed_0.80_profile_0.25",
                "very_relaxed_0.75_profile_0.10",
                "embedding_only_0.85",
            }:
                labels = (
                    retained_assignments.groupby(
                        [
                            "cluster_id",
                            "canonical_feature",
                            "cluster_merge_basis",
                        ],
                        observed=True,
                    )["feature_text_normalized"]
                    .agg(cluster_size="size", members=lambda values: " | ".join(values))
                    .reset_index()
                )
                labels.insert(0, "configuration", specification["configuration"])
                labels.insert(0, "prompt_variant", prompt)
                relaxed_cluster_rows.append(labels)

    summary = pd.DataFrame(summary_rows)
    primary_audit = pd.concat(primary_audits, ignore_index=True)
    relaxed_clusters = pd.concat(relaxed_cluster_rows, ignore_index=True)
    summary.to_csv(OUTPUT / "rule_sensitivity.csv", index=False)
    primary_audit.to_csv(OUTPUT / "primary_retained_cluster_audit.csv", index=False)
    relaxed_clusters.to_csv(OUTPUT / "relaxed_retained_clusters.csv", index=False)
    primary_audit["modal_or_frequency_mismatch"] = (
        primary_audit["modal_mismatch"] | primary_audit["frequency_mismatch"]
    )
    lexical_summary = (
        primary_audit.groupby("prompt_variant", observed=True)
        .agg(
            retained_tasks=("cluster_id", "size"),
            tasks_dependent_on_any_merge=("task_depends_on_merge", "sum"),
            modal_or_frequency_mismatch=("modal_or_frequency_mismatch", "sum"),
            tasks_lost_after_conservative_split=(
                "survives_conservative_split",
                lambda values: int((~values).sum()),
            ),
            positive_objects_created_by_merge=(
                "positive_objects_created_by_merge",
                "sum",
            ),
        )
        .reset_index()
    )
    selected_names = {
        "exact_only",
        "conservative_lexical_only",
        "current_lexical_only",
        "stricter_0.90_profile_0.75",
        "primary_0.85_profile_0.50",
        "lower_profile_0.85_profile_0.25",
        "relaxed_0.80_profile_0.25",
        "embedding_only_0.85",
    }
    selected = summary[summary["configuration"].isin(selected_names)][
        [
            "prompt_variant",
            "configuration",
            "retained_tasks",
            "retained_positive_cells",
            "object_rdm_spearman_vs_human",
        ]
    ].copy()
    selected["object_rdm_spearman_vs_human"] = selected[
        "object_rdm_spearman_vs_human"
    ].map(lambda value: f"{value:.3f}")
    report = f"""# V3 Consolidation Rule Audit

This is a diagnostic sensitivity analysis. It does not modify the locked primary V3
consolidation or any trained model.

## Verdict

The `0.85` embedding threshold is **not causing over-merging and is not the reason V3 has
only 94-121 retained tasks**. Removing semantic embedding merges entirely leaves the task
counts unchanged at A=94, B=115, and C=121. Moving the embedding threshold from `0.80` to
`0.90` with the profile guard also leaves all task counts unchanged.

The `0.50` object-profile guard is useful. Lowering it to `0.25` adds only one task per prompt,
and some newly allowed pairs are not equivalent. Removing it adds 10-19 tasks per prompt but
creates clear errors, including complete/incomplete metamorphosis, migratory/non-migratory,
warm/cold weather, and moves quickly/slowly.

The rule that merits revision is the **lexical signature**, not the embedding threshold. It
strips modality and frequency scaffolding before grouping and applies Porter stemming without
an embedding or profile check. Most resulting merges are defensible wording variants, but a
small number change the proposition. The clearest primary error is C cluster
`C_63e10f2951f0`, which pools `can be washed` with `washed frequently` and becomes a retained
four-object task only through that pooling. B cluster `B_bd4be8d2c69e` pools `musical
instrument` with `used for musical instruments`; that error does not currently change a
positive object or task because the second phrase occurs only once.

## Rule Sensitivity

{selected.to_markdown(index=False)}

`object_rdm_spearman_vs_human` compares object geometry in each diagnostic matrix with the
human matrix. The current lexical rule improves this descriptive alignment, but that does not
prove every merge is valid. Generic shared features can increase geometric correlation while
still collapsing distinct propositions.

## Primary Lexical Audit

{lexical_summary.to_markdown(index=False)}

The conservative split preserves whether a phrase is modal (`can`/`could`) and whether it has
a frequency qualifier. Its task count is A=84, B=112, C=114. Some current tasks split into
multiple conservative tasks, so the number of current clusters that fail to survive is larger
than the net task-count change.

## Recommendation

1. Keep embedding threshold `0.85`, profile threshold `0.50`, complete-link clustering,
   negation checks, substantive-qualifier checks, and the English/CJK separation.
2. Replace unrestricted lexical signatures with a safe tier for inflection, pluralization,
   punctuation/hyphenation, and conjunction-order variants.
3. Require manual adjudication when a lexical merge changes modality, frequency, or
   derivational role. At minimum, review the retained clusters flagged in
   `primary_retained_cluster_audit.csv` and all clusters that do not survive the conservative
   split.
4. Build conservative/adjudicated V3 matrices and retrain them as a sensitivity condition
   before treating the current V3 model comparison as final. Do not replace the existing
   primary artifacts; report both.

## Files

- `rule_sensitivity.csv`: all diagnostic rule combinations.
- `primary_retained_cluster_audit.csv`: every retained primary cluster, merge dependence,
  modality/frequency flags, and conservative survival.
- `relaxed_retained_clusters.csv`: members retained under relaxed semantic rules.
- `manifest.json`: hashes of all audit inputs and this script.
"""
    report_path = OUTPUT / "CONSOLIDATION_AUDIT.md"
    report_path.write_text(report, encoding="utf-8")
    write_json(
        OUTPUT / "manifest.json",
        {
            "source_csv_sha256": sha256_file(source_path),
            "human_csv_sha256": sha256_file(human_path),
            "embedding_cache_sha256": sha256_file(
                CONSOLIDATION / "phrase_embeddings.npz"
            ),
            "config_sha256": sha256_file(config_path),
            "audit_script_sha256": sha256_file(Path(__file__)),
            "audit_report_sha256": sha256_file(report_path),
            "configurations": list(CONFIGURATIONS),
            "note": "Diagnostic only; the locked primary consolidation was not modified.",
        },
    )
    print(summary.to_string(index=False))
    print(f"\nWrote consolidation audit to {OUTPUT}")


if __name__ == "__main__":
    main()
