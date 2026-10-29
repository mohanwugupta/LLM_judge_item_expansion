from __future__ import annotations

import argparse
import json
import multiprocessing
import platform
import resource
import signal
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .classification_workflow import (
    prepare,
    model_fingerprint,
    classify_catalog,
    freeze_catalog,
    classification_smoke,
)
from .extract_representations import extract, load_run, train_run
from .gates import (
    gate_signature,
    preflight,
    verify_gate,
    verify_frozen_mapping,
    verify_sources,
)
from .inventory import load_matrix, matrix_path, sha256
from .modeling import ISCModel
from .provenance import atomic_json, code_hashes, digest, validate_output_directory
from .run_crosses import ROOT, execution_plan, load_config, planned_jobs
from .taxonomy import load_taxonomy
from .train_isc import TrainingConfig, backward_bce, estimate_memory
from .training_adapter import TaskDataset
from .transfer_embeddings import train_context


def benchmark(output, taxonomy, device="cpu", steps=1):
    verify_gate(output / "gates/preflight.json", gate_signature(output, taxonomy))
    torch.set_num_threads(1)
    rows = []
    for features in [88, 385, 87315]:
        torch.manual_seed(0)
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        model = ISCModel(293, features, 32, device=device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        inputs = [
            torch.eye(293, device=device)[:64],
            torch.eye(32, device=device)[torch.arange(64, device=device) % 32],
        ]
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        started = time.monotonic()
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            loss = backward_bce(
                model,
                inputs,
                lambda start, stop: (
                    (
                        torch.arange(start, stop)[None, :] % 32
                        == torch.arange(64)[:, None] % 32
                    )
                    & (
                        torch.arange(start, stop)[None, :] % 7
                        == torch.arange(64)[:, None] % 7
                    )
                ).float(),
                4096,
            )
            optimizer.step()
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        seconds = time.monotonic() - started
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
            1 if platform.system() == "Darwin" else 1024
        )
        rows.append(
            {
                "features": features,
                "steps": steps,
                "seconds": seconds,
                "seconds_per_update": seconds / steps,
                "device": device,
                "finite_loss": bool(np.isfinite(loss)),
                "loss": loss,
                "process_peak_rss_bytes": rss,
                "cuda_peak_allocated_bytes": (
                    torch.cuda.max_memory_allocated()
                    if device.startswith("cuda")
                    else None
                ),
                "memory_estimate": estimate_memory(features),
                "synthetic_only": True,
            }
        )
        del model, optimizer
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    record = {
        "signature": gate_signature(output, taxonomy),
        "passed": all(
            r["finite_loss"] and r["process_peak_rss_bytes"] < 8 * 1024**3 for r in rows
        ),
        "device": device,
        "measurements": rows,
        "note": "Process RSS is an absolute high-water mark, not incremental allocation",
    }
    atomic_json(output / "gates/benchmark.json", record)
    return record


def synthetic_smoke(output):
    torch.set_num_threads(1)
    matrix = pd.DataFrame(
        [[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]],
        index=list("abcde"),
        columns=list("wxyz"),
        dtype=np.int8,
    )
    dataset = TaskDataset(matrix, dict(zip("wxyz", [0, 0, 1, 1])), 2)
    version = digest(code_hashes())[:12]
    smoke_root = output / "smoke" / version
    directory = smoke_root / "synthetic_semantic"
    result = train_run(
        dataset,
        "synthetic",
        0,
        TrainingConfig(
            max_epochs=800, batch_size=10, learning_rate=0.01, max_seconds=30
        ),
        directory,
        "synthetic-taxonomy",
        "synthetic-mapping",
        resume=directory.exists(),
    )
    if not result["complete"]:
        raise ValueError(
            "Synthetic semantic learning failed; no real training authorized"
        )
    model, _, _ = load_run(directory)
    context_dir = smoke_root / "synthetic_context"
    context = train_context(
        matrix,
        extract(model, dataset.object_ids),
        dataset.object_ids,
        "synthetic",
        "synthetic",
        0,
        context_dir,
        {"semantic_checkpoint_sha256": sha256(directory / "checkpoint.pt")},
        diagnostic=True,
        resume=context_dir.exists(),
    )
    record = {
        "passed": result["complete"] and context["status"] == "diagnostic_complete",
        "synthetic_only": True,
        "real_models_trained": 0,
        "code": code_hashes(),
    }
    atomic_json(output / "gates/smoke.json", record)
    return record


def require_training_gates(output, taxonomy, config, production):
    signature = gate_signature(output, taxonomy)
    gate = verify_gate(output / "gates/preflight.json", signature)
    verify_sources(
        [{"path": p, "sha256": h} for p, h in gate["preserved_sources"].items()]
    )
    if not production:
        return
    semantic = config["semantic"]
    if (
        not semantic["schedule_locked"]
        or not semantic["full"]
        or not semantic.get("schedule_rationale")
    ):
        raise ValueError(
            "A training-only pilot and explicit schedule rationale must precede a locked full schedule"
        )
    full = TrainingConfig(**semantic["full"])
    bench = verify_gate(output / "gates/benchmark.json", signature)
    if bench["device"] != full.device:
        raise ValueError(
            "Resource benchmark device differs from locked training device"
        )
    pilot, checkpoint, manifest = load_run(output / "isc_models/pilot/human/seed_0")
    if not checkpoint["result"]["converged"]:
        raise ValueError("Human-reference training-only pilot did not converge")
    lock = {
        "training": semantic["full"],
        "taxonomy": taxonomy,
        "rationale": semantic["schedule_rationale"],
        "selection_basis": semantic["schedule_selection_basis"],
        "behavior_config_sha256": sha256(ROOT / config["behavior_config"]),
    }
    path = output / "gates/experiment_lock.json"
    if path.exists() and json.loads(path.read_text()) != lock:
        raise ValueError(
            "Training/evaluation lock changed; use a new experiment namespace"
        )
    atomic_json(path, lock)


def execute_job(
    job, output, config, taxonomy, pilot=False, diagnostic=False, resume=False
):
    torch.set_num_threads(1)

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Signal {signum}; preserving resumable state")

    signal.signal(signal.SIGTERM, interrupted)
    source = job["semantic_source"]
    matrix = load_matrix(matrix_path(ROOT, source))
    mapping, mapping_hash = verify_frozen_mapping(
        output, source, matrix_path(ROOT, source), taxonomy
    )
    if "context_source" not in job:
        dataset = TaskDataset(matrix, mapping["assignments"], len(taxonomy["classes"]))
        settings = config["semantic"]["pilot" if pilot else "full"]
        directory = (
            output / "isc_models/pilot/human/seed_0"
            if pilot
            else Path(job["directory"])
        )
        return train_run(
            dataset,
            source,
            job["seed"],
            TrainingConfig(**settings),
            directory,
            mapping["taxonomy_sha256"],
            mapping_hash,
            resume=resume,
        )
    semantic_dir = (
        output / "isc_models/pilot/human/seed_0"
        if diagnostic
        else output / "isc_models" / source / f"seed_{job['seed']}"
    )
    model, checkpoint, manifest = load_run(semantic_dir)
    if (
        checkpoint["signature"]["condition"] != source
        or checkpoint["signature"]["seed"] != job["seed"]
    ):
        raise ValueError("Semantic source/seed identity mismatch")
    if manifest["signature"]["classification_sha256"] != mapping_hash:
        raise ValueError(
            "Semantic checkpoint does not use the current frozen task mapping"
        )
    context_source = job["context_source"]
    context = load_matrix(matrix_path(ROOT, context_source), checkpoint["object_ids"])
    directory = (
        output / "diagnostics/context/human_seed_0"
        if diagnostic
        else Path(job["directory"])
    )
    return train_context(
        context,
        extract(model, checkpoint["object_ids"]),
        checkpoint["object_ids"],
        source,
        context_source,
        job["seed"],
        directory,
        {
            "semantic_checkpoint_sha256": sha256(semantic_dir / "checkpoint.pt"),
            "semantic_manifest_sha256": sha256(semantic_dir / "manifest.json"),
        },
        diagnostic=diagnostic,
        resume=resume,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Staged, gated independent ISC semantic retraining"
    )
    parser.add_argument(
        "--stage",
        choices=[
            "plan",
            "prepare",
            "fingerprint",
            "classify",
            "classify-smoke",
            "freeze",
            "preflight",
            "benchmark",
            "smoke",
            "pilot",
            "semantic",
            "semantic-evaluate",
            "context",
            "evaluate",
            "report",
        ],
        default="plan",
    )
    parser.add_argument("--phase", type=int, choices=range(1, 6), default=1)
    parser.add_argument("--condition")
    parser.add_argument("--semantic-source")
    parser.add_argument("--context-source")
    parser.add_argument("--seed", type=int, choices=range(4))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--endpoint")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--benchmark-steps", type=int, default=1)
    parser.add_argument("--reviews", type=Path)
    args = parser.parse_args(argv)
    if args.workers < 1 or args.benchmark_steps < 1:
        parser.error("workers and benchmark-steps must be positive")
    try:
        config = load_config(args.config)
        output = (args.output_dir or ROOT / config["output_directory"]).resolve()
        validate_output_directory(output)
        taxonomy = load_taxonomy(ROOT / config["taxonomy"])
        plan = execution_plan(
            output,
            args.stage,
            args.phase,
            args.condition,
            args.semantic_source,
            args.context_source,
            args.seed,
            config,
        )
        if args.dry_run or args.stage == "plan":
            print(json.dumps(plan, indent=2))
            return 0
        if args.stage in {"semantic", "context", "pilot"} and plan["blocked"]:
            print(json.dumps(plan, indent=2))
            return 2
        if args.stage == "prepare":
            sources, catalog = prepare(ROOT, output, taxonomy)
            result = {
                "conditions": len(sources),
                "unique_descriptions": len(
                    {t for labels in catalog.values() for t in labels.values()}
                ),
            }
        elif args.stage == "fingerprint":
            if args.model_path is None:
                raise ValueError(
                    "--model-path must identify the actual Qwen checkpoint"
                )
            result = model_fingerprint(args.model_path)
            atomic_json(output / "classification/model_fingerprint.json", result)
        elif args.stage in {"classify", "classify-smoke"}:
            verify_gate(
                output / "gates/preflight.json", gate_signature(output, taxonomy)
            )
            if args.model_path is not None:
                fingerprint = model_fingerprint(args.model_path)
                atomic_json(
                    output / "classification/model_fingerprint.json", fingerprint
                )
            else:
                fingerprint = json.loads(
                    (output / "classification/model_fingerprint.json").read_text()
                )
            import importlib.metadata

            versions = {}
            for package in ["torch", "vllm", "transformers"]:
                try:
                    versions[package] = importlib.metadata.version(package)
                except importlib.metadata.PackageNotFoundError:
                    versions[package] = "not installed in client environment"
            classifier = {
                **config["classifier"],
                "revision": fingerprint["revision"],
                "endpoint": args.endpoint or config["classifier"]["endpoint"],
                "runtime_versions": versions,
                "client_implementation_sha256": sha256(
                    Path(__file__).with_name("classification_workflow.py")
                ),
            }
            conditions = [args.condition] if args.condition else None
            from .provenance import run_lock

            with run_lock(output / "classification"):
                result = (
                    classification_smoke(
                        output, taxonomy, classifier, classifier["endpoint"]
                    )
                    if args.stage == "classify-smoke"
                    else classify_catalog(
                        output,
                        taxonomy,
                        classifier,
                        classifier["endpoint"],
                        classifier["batch_size"],
                        conditions,
                    )
                )
        elif args.stage == "freeze":
            conditions = (
                ["human", "random_human"]
                if args.condition in {"human", "random_human"}
                else [args.condition] if args.condition else None
            )
            result = freeze_catalog(
                output,
                taxonomy,
                args.reviews or output / "classification/human_review.csv",
                conditions,
            )
        elif args.stage == "preflight":
            result = preflight(ROOT, output, taxonomy, config)
        elif args.stage == "benchmark":
            result = benchmark(output, taxonomy, args.device, args.benchmark_steps)
        elif args.stage == "smoke":
            result = synthetic_smoke(output)
        elif args.stage in {"pilot", "semantic", "context"}:
            if args.stage == "pilot" and args.phase != 1:
                raise ValueError("The human-reference pilot is phase 1 only")
            diagnostic = args.stage == "context" and args.phase == 1
            require_training_gates(
                output, taxonomy, config, args.stage != "pilot" and not diagnostic
            )
            jobs = plan["jobs"]
            if not jobs:
                raise ValueError(
                    "No jobs match the requested source/phase/seed selection"
                )
            # Verify the whole requested cohort before any worker can start an expensive fit.
            for source in sorted({job["semantic_source"] for job in jobs}):
                verify_frozen_mapping(
                    output, source, matrix_path(ROOT, source), taxonomy
                )
            if args.stage == "context" and not diagnostic:
                stage1 = json.loads(
                    (output / "evaluations/semantic/manifest.json").read_text()
                )
                if (
                    not stage1.get("complete")
                    or stage1["signature"]["config_sha256"] != digest(config)
                    or any(
                        j["semantic_source"] not in stage1["conditions"] for j in jobs
                    )
                ):
                    raise ValueError(
                        "Complete current Stage 1 evaluation is required for each semantic source"
                    )
                for name, expected in stage1["outputs"].items():
                    if sha256(output / "evaluations/semantic" / name) != expected:
                        raise ValueError("Stage 1 evaluation hash mismatch")
                for source, seed in {
                    (job["semantic_source"], job["seed"]) for job in jobs
                }:
                    checkpoint = (
                        output / "isc_models" / source / f"seed_{seed}/checkpoint.pt"
                    )
                    if (
                        sha256(checkpoint)
                        != stage1["signature"]["checkpoints"][f"{source}_seed_{seed}"]
                    ):
                        raise ValueError(
                            "Semantic checkpoint changed after Stage 1 evaluation"
                        )
            if args.workers == 1:
                results = [
                    execute_job(
                        job,
                        output,
                        config,
                        taxonomy,
                        args.stage == "pilot",
                        diagnostic,
                        args.resume,
                    )
                    for job in jobs
                ]
            else:
                with ProcessPoolExecutor(
                    args.workers, mp_context=multiprocessing.get_context("spawn")
                ) as pool:
                    futures = [
                        pool.submit(
                            execute_job,
                            job,
                            output,
                            config,
                            taxonomy,
                            args.stage == "pilot",
                            diagnostic,
                            args.resume,
                        )
                        for job in jobs
                    ]
                    results = [future.result() for future in futures]
            result = {
                "jobs": len(results),
                "passed": all(
                    (
                        r.get("status") == "diagnostic_complete"
                        if diagnostic
                        else r.get("complete") is True
                    )
                    for r in results
                ),
                "completed": sum(r["complete"] for r in results),
                "diagnostic": diagnostic,
                "statuses": [r["status"] for r in results],
            }
        elif args.stage == "semantic-evaluate":
            from .evaluate_crosses import evaluate_semantics

            require_training_gates(output, taxonomy, config, True)
            jobs = execution_plan(
                output,
                "semantic",
                args.phase,
                args.condition,
                args.semantic_source,
                config=config,
            )["jobs"]
            result = evaluate_semantics(
                ROOT, output, sorted({j["semantic_source"] for j in jobs}), config
            )
        elif args.stage == "evaluate":
            from .evaluate_crosses import evaluate_context_models

            require_training_gates(output, taxonomy, config, True)
            jobs = execution_plan(
                output,
                "context",
                args.phase,
                args.condition,
                args.semantic_source,
                args.context_source,
                args.seed,
                config,
            )["jobs"]
            if args.phase == 5:
                jobs = [
                    job
                    for job in planned_jobs(output)["context"]
                    if (
                        args.semantic_source is None
                        or job["semantic_source"] == args.semantic_source
                    )
                    and (
                        args.context_source is None
                        or job["context_source"] == args.context_source
                    )
                    and (
                        args.condition is None
                        or job["semantic_source"] == args.condition
                    )
                    and (args.seed is None or job["seed"] == args.seed)
                ]
            result = evaluate_context_models(ROOT, output, jobs, config, args.phase)
        else:
            from .report_results import report

            result = report(
                ROOT,
                output,
                args.report_dir or ROOT / config["report_directory"],
                config,
                args.phase,
            )
        display = {
            key: value
            for key, value in result.items()
            if key
            not in {
                "inputs",
                "report_outputs",
                "preserved_sources",
                "signature",
                "outputs",
                "files",
            }
        }
        if "signature" in result:
            display["signature_sha256"] = digest(result["signature"])
        for key in ["preserved_sources", "outputs", "files"]:
            if key in result:
                display[f"{key}_count"] = len(result[key])
        print(json.dumps(display, indent=2, default=str))
        return 0 if result.get("passed", True) and result.get("complete", True) else 2
    except (ValueError, FileNotFoundError, FileExistsError, KeyError) as exc:
        print(
            json.dumps(
                {
                    "error": str(exc),
                    "stage": args.stage,
                    "scientific_training_not_authorized_by_failure": True,
                }
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
