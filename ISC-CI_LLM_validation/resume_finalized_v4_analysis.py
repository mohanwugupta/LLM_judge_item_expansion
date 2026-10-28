#!/usr/bin/env python3
"""Run existing downstream stages once the finalized V4 cohort is complete."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
VALIDATION = ROOT / "artifacts/validation_v4_20261002"
REQUIRED = {
    "v4_b_locked_v2",
    "v4_b_calibrated",
    "v4_ensemble_locked_v2",
    "v4_ensemble_calibrated",
    "v4_ensemble_source_only",
}


def training_ready():
    manifest_path = VALIDATION / "manifest.json"
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError:
        return False
    if manifest.get("protocol_version") != "iscci-v4-validation-1.0.0":
        return False
    if not REQUIRED.issubset(manifest.get("selected_conditions", [])) or set(
        manifest.get("selected_seeds", [])
    ) != {0, 1, 2, 3}:
        return False
    return all(
        (VALIDATION / "models" / condition / f"seed_{seed}/checkpoint.pt").exists()
        for condition in REQUIRED
        for seed in range(4)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-for-training", action="store_true")
    args = parser.parse_args()
    if args.wait_for_training:
        print("Waiting for the completed 20-model V4 training manifest", flush=True)
        while not training_ready():
            time.sleep(5)
    if not training_ready():
        raise ValueError(
            "Train all five V4 conditions with four frozen-protocol seeds first"
        )
    commands = [
        [
            sys.executable,
            str(ROOT / name),
            "--config",
            str(PROJECT / "configs/v4_validation.json"),
            "--validation-dir",
            str(VALIDATION),
        ]
        for name in ("evaluate_validation.py", "run_paper_simulations.py")
    ] + [
        [
            sys.executable,
            str(ROOT / "run_human_behavior_validation.py"),
            "--config",
            str(ROOT / "configs/v4_human_behavior_validation.json"),
            "--evaluate-only",
        ],
        [sys.executable, str(ROOT / "summarize_finalized_v4.py")],
    ]
    commands.insert(-1, [sys.executable, str(ROOT / "audit_v4_feature_profiles.py")])
    commands.insert(-1, [sys.executable, str(ROOT / "audit_finalized_v4_fidelity.py")])
    report = ROOT / "reports/v4_20261002"
    report.mkdir(parents=True, exist_ok=True)
    environment = dict(
        os.environ,
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        PYTHONUNBUFFERED="1",
    )
    with (report / "downstream_run.log").open("a") as log:
        for command in commands:
            message = "Running: " + " ".join(command)
            print(message, flush=True)
            log.write(message + "\n")
            log.flush()
            process = subprocess.Popen(
                command,
                cwd=PROJECT,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            status = process.wait()
            if status:
                raise subprocess.CalledProcessError(status, command)
    print("All finalized V4 analysis stages completed", flush=True)


if __name__ == "__main__":
    main()
