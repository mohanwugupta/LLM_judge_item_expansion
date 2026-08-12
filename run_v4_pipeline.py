#!/usr/bin/env python3
"""Resume all locally executable V4 stages around the external cluster jobs."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from leuven_expansion.v4 import sha256_file, write_json


ROOT = Path(__file__).resolve().parent


def run_command(command: list[str], log_path: Path) -> None:
    rendered = " ".join(command)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{datetime.now(timezone.utc).isoformat()}] $ {rendered}\n")
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def is_complete_manifest(path: Path) -> bool:
    if not path.exists():
        return False
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return bool(manifest.get("complete", manifest.get("finished_at")))


def pending_review_count(path: Path) -> int:
    if not path.exists():
        return 0
    review = pd.read_csv(path, dtype=str).fillna("")
    return int((~review["verdict"].isin(["pass", "reject"])).sum())


def discovery_inventory_is_stale(
    inventory_path: Path, discovery_config: dict[str, Any]
) -> bool:
    if not inventory_path.exists():
        return True
    inventory = pd.read_csv(inventory_path, dtype=str).fillna("")
    status_by_id = dict(zip(inventory["source_id"], inventory["status"]))
    for source in discovery_config["sources"]:
        source_dir = ROOT / source["path"]
        now_complete = (source_dir / "manifest.json").exists() and (
            source_dir / "generated_features_long.csv"
        ).exists()
        if now_complete and status_by_id.get(source["source_id"]) != "loaded":
            return True
    return False


def pending_primary_discovery_sources(discovery_config: dict[str, Any]) -> list[str]:
    pending: list[str] = []
    for source in discovery_config["sources"]:
        if not source.get("required_for_primary_freeze", False):
            continue
        source_dir = ROOT / source["path"]
        if not (
            (source_dir / "manifest.json").exists()
            and (source_dir / "generated_features_long.csv").exists()
        ):
            pending.append(str(source["source_id"]))
    return pending


def source_hashes() -> dict[str, Any]:
    paths = {
        "v3_manifest": ROOT / "artifacts/leuven_feature_generation/leuven_v3_qwen2_5_72b/manifest.json",
        "v3_long": ROOT / "artifacts/leuven_feature_generation/leuven_v3_qwen2_5_72b/generated_features_long.csv",
        "v3_1_manifest": ROOT / "artifacts/leuven_feature_generation/v3.1/leuven_v3.1_qwen2_5_72b/manifest.json",
        "v3_1_long": ROOT / "artifacts/leuven_feature_generation/v3.1/leuven_v3.1_qwen2_5_72b/generated_features_long.csv",
        "v2_manifest": ROOT / "artifacts/leuven_full_labels/leuven_full_v2/manifest.json",
        "v2_resolved": ROOT / "artifacts/leuven_full_labels/leuven_full_v2/feature_resolutions.csv",
        "human_features": ROOT / "data/leuven_combined_features_consolidated.csv",
        "released_model": ROOT / "ISC-CI_LLM_validation/upstream/IntegratedSemanticsControlContextInference/models/1and2shot_isc-seed3.pt",
    }
    return {
        name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
        for name, path in paths.items()
        if path.exists()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/v4_validation.json")
    parser.add_argument("--discovery-config", type=Path, default=ROOT / "configs/v4_discovery.json")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    run_dir = ROOT / "artifacts" / "v4"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"
    resolved_config = {
        "validation": json.loads(args.config.read_text(encoding="utf-8")),
        "discovery": json.loads(args.discovery_config.read_text(encoding="utf-8")),
        "validation_config_path": str(args.config.resolve()),
        "discovery_config_path": str(args.discovery_config.resolve()),
    }
    write_json(run_dir / "resolved_v4_config.json", resolved_config)
    write_json(run_dir / "source_hashes.json", source_hashes())
    stages: dict[str, str] = {}

    threshold = run_dir / "judgments" / "judgment_threshold.json"
    if args.force or not threshold.exists():
        run_command(
            [
                sys.executable,
                "calibrate_v4_judgments.py",
                "--v2-resolved",
                "artifacts/leuven_full_labels/leuven_full_v2/feature_resolutions.csv",
                "--human-features",
                "data/leuven_combined_features_consolidated.csv",
                "--output-dir",
                "artifacts/v4/judgments",
            ],
            log_path,
        )
    stages["calibration"] = "complete"

    bank = run_dir / "discovery" / "candidate_bank.csv"
    review_required = run_dir / "discovery" / "candidate_merge_review_required.csv"
    configured_review = ROOT / "configs" / "v4_candidate_merge_review.csv"
    pending = pending_review_count(review_required)
    configured_pending = pending_review_count(configured_review)
    stale_discovery = discovery_inventory_is_stale(
        run_dir / "discovery" / "source_inventory.csv", resolved_config["discovery"]
    )
    pending_sources = pending_primary_discovery_sources(resolved_config["discovery"])
    if args.force or not bank.exists():
        if pending_sources:
            stages["candidate_bank"] = "blocked_generation:" + ",".join(pending_sources)
        elif (
            pending
            and configured_pending == 0
            and len(pd.read_csv(configured_review)) == 0
            and not stale_discovery
        ):
            stages["candidate_bank"] = f"blocked_review:{pending}"
        else:
            try:
                run_command(
                    [
                        sys.executable,
                        "build_v4_candidate_bank.py",
                        "--config",
                        str(args.discovery_config),
                        "--manual-review",
                        str(configured_review),
                        "--output-dir",
                        "artifacts/v4/discovery",
                    ],
                    log_path,
                )
            except subprocess.CalledProcessError:
                pending = pending_review_count(review_required)
                if pending:
                    stages["candidate_bank"] = f"blocked_review:{pending}"
                else:
                    raise
    if bank.exists():
        stages["candidate_bank"] = "complete"
        run_command(
            [
                sys.executable,
                "run_v4_judgments.py",
                "--candidate-bank",
                str(bank),
                "--leuven-words",
                "data/leuven_combined_features_consolidated.csv",
                "--v2-manifest",
                "artifacts/leuven_full_labels/leuven_full_v2/manifest.json",
                "--output-dir",
                "artifacts/v4/judgments",
                "--shard-count",
                "32",
                "--dry-run",
            ],
            log_path,
        )

    judgment_manifest = run_dir / "judgments" / "judgment_manifest.json"
    if not is_complete_manifest(judgment_manifest):
        stages["atomic_judgments"] = "pending_external_cluster"
    else:
        stages["atomic_judgments"] = "complete"
        matrices = run_dir / "matrices" / "matrix_manifest.json"
        if args.force or not matrices.exists():
            run_command(
                [
                    sys.executable,
                    "build_v4_matrices.py",
                    "--candidate-bank",
                    str(bank),
                    "--resolved-values",
                    "artifacts/v4/judgments/resolved_feature_values.csv",
                    "--threshold",
                    str(threshold),
                    "--output-dir",
                    "artifacts/v4/matrices",
                ],
                log_path,
            )
        stages["matrices"] = "complete"

        validation = run_dir / "validation" / "manifest.json"
        if args.force or not validation.exists():
            command = [
                sys.executable,
                "ISC-CI_LLM_validation/run_validation.py",
                "--config",
                str(args.config),
                "--output-dir",
                "artifacts/v4/validation",
                "--base-validation-dir",
                "ISC-CI_LLM_validation/artifacts/validation_v3_1",
                "--include-v4",
            ]
            if args.force:
                command.append("--force")
            run_command(command, log_path)
        stages["training"] = "complete"

        for script, stage in [
            ("ISC-CI_LLM_validation/evaluate_validation.py", "evaluation"),
            ("ISC-CI_LLM_validation/run_paper_simulations.py", "paper_simulations"),
        ]:
            run_command(
                [
                    sys.executable,
                    script,
                    "--config",
                    str(args.config),
                    "--validation-dir",
                    "artifacts/v4/validation",
                    *( ["--force"] if args.force else [] ),
                ],
                log_path,
            )
            stages[stage] = "complete"

        retrieval_manifest = run_dir / "retrieval_efficiency" / "v4_posthoc" / "manifest.json"
        if args.force or not retrieval_manifest.exists():
            run_command(
                [
                    sys.executable,
                    "analyze_v4_retrieval_efficiency.py",
                    "--config",
                    str(args.config),
                    "--candidate-bank",
                    str(bank),
                    "--human-mapping",
                    "artifacts/v4/discovery/candidate_human_feature_mapping.csv",
                    "--resolved-values",
                    "artifacts/v4/judgments/resolved_feature_values.csv",
                    "--votes",
                    "artifacts/v4/judgments/feature_votes.csv",
                    "--gold-rule",
                    "calibrated",
                    "--output-dir",
                    "artifacts/v4/retrieval_efficiency/v4_posthoc",
                ],
                log_path,
            )
        stages["v4_retrieval_efficiency"] = "complete"

    run_command(
        [sys.executable, "summarize_v4_results.py", "--run-dir", "artifacts/v4"],
        log_path,
    )
    manifest = {
        "protocol_version": "v4-pipeline-1.0.0",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "resume": args.resume,
        "force": args.force,
        "stages": stages,
        "external_commands": {
            "generation": "sbatch run_leuven_v4_generation.sh",
            "atomic_smoke": "sbatch run_leuven_v4_atomic_smoke_test.sh",
            "atomic_production": "sbatch run_leuven_v4_atomic.sh",
            "atomic_finalize": "sbatch run_leuven_v4_atomic_finalize.sh",
        },
    }
    write_json(run_dir / "run_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
