from __future__ import annotations

import json
from dataclasses import asdict
from itertools import product
from pathlib import Path

from .inventory import CONDITIONS
from .provenance import digest
from .train_isc import estimate_memory, TrainingConfig

ROOT = Path(__file__).resolve().parents[1]


def load_config(path=None):
    config = json.loads(
        Path(path or ROOT / "configs/semantic_retraining.json").read_text()
    )
    if (
        config["protocol"] != "isc-independent-32-corrected-targets-v1"
        or config["seeds"] != [0, 1, 2, 3]
        or config["auxiliary_tasks"]
    ):
        raise ValueError(
            "Primary protocol requires paired seeds, approved 32 classes, and no auxiliary supervision"
        )
    if config["pilot_llm_condition"] not in set(CONDITIONS) - {"human", "random_human"}:
        raise ValueError("Unknown prespecified pilot source")
    source_cohort(config)
    from .transfer_embeddings import SETTINGS

    if config["downstream_settings"] != SETTINGS:
        raise ValueError("Downstream architecture/training settings must remain frozen")
    config["semantic"]["pilot"] = asdict(TrainingConfig(**config["semantic"]["pilot"]))
    if config["semantic"]["full"] is not None:
        config["semantic"]["full"] = asdict(
            TrainingConfig(**config["semantic"]["full"])
        )
    return config


def source_cohort(config, include_deferred=False):
    deferred = config.get("deferred_conditions", [])
    if (
        not isinstance(deferred, list)
        or any(not isinstance(c, str) or c not in CONDITIONS for c in deferred)
        or len(set(deferred)) != len(deferred)
        or {"human", "random_human", config["pilot_llm_condition"]}.intersection(
            deferred
        )
    ):
        raise ValueError(
            "Invalid deferred conditions: preserve human/random/pilot references and use unique known sources"
        )
    return [c for c in CONDITIONS if include_deferred or c not in deferred]


def all_pairs():
    return list(product(CONDITIONS, repeat=2))


def phase_pairs(phase, pilot_llm="v3_1_B", sources=None):
    sources = list(CONDITIONS if sources is None else sources)
    if phase == 1:
        return [("human", "human")]
    if phase == 2:
        return [(c, c) for c in ("human", "random_human", pilot_llm)]
    if phase == 3:
        return [(c, c) for c in sources]
    controls = {"human", "random_human"}
    first64 = [
        (s, c)
        for s, c in product(sources, repeat=2)
        if s == c or s in controls or c in controls
    ]
    if phase == 4:
        return first64
    if phase == 5:
        return [(s, c) for s, c in product(sources, repeat=2) if (s, c) not in first64]
    raise ValueError("phase must be 1..5")


def planned_jobs(output, config=None, include_deferred=False):
    output = Path(output)
    semantic = [
        {
            "id": f"{source}_seed_{seed}",
            "semantic_source": source,
            "seed": seed,
            "directory": str(output / "isc_models" / source / f"seed_{seed}"),
            "status": "planned",
            "memory": estimate_memory(CONDITIONS[source]),
        }
        for source, seed in product(CONDITIONS, range(4))
    ]
    context = [
        {
            "id": f"{s}__{c}_seed_{seed}",
            "semantic_source": s,
            "context_source": c,
            "seed": seed,
            "directory": str(output / "iscci_models" / f"{s}__{c}" / f"seed_{seed}"),
            "status": "planned",
            "memory": context_memory(CONDITIONS[c]),
        }
        for (s, c), seed in product(all_pairs(), range(4))
    ]
    if config is not None:
        active = set(source_cohort(config, include_deferred))
        for job in semantic + context:
            if (
                job["semantic_source"] not in active
                or job.get("context_source", "human") not in active
            ):
                job["status"] = "deferred"
                job["deferral_reason"] = config.get(
                    "deferral_reason", "Explicitly deferred in configuration"
                )
    return {"semantic": semantic, "context": context}


def context_memory(features):
    embedding = 294 * 64
    trainable = 33280 + 129 * features
    pools = 294 * features * 12 + features * 512
    activations = 128 * 293 * 128 * 4 * 6 + 128 * features * 4 * 4
    return {
        "parameters": embedding + trainable,
        "sampler_and_dense_target_bytes": pools,
        "estimated_bytes": 4 * embedding
        + 16 * trainable
        + pools
        + activations
        + 293 * features,
        "note": "CICO optimizer, native sampler pools and batch activations; excludes restoring the semantic checkpoint and Python allocator overhead",
    }


def execution_plan(
    output,
    stage="plan",
    phase=1,
    condition=None,
    semantic_source=None,
    context_source=None,
    seed=None,
    config=None,
    include_deferred=False,
):
    config = config or load_config()
    active = source_cohort(config, include_deferred)
    for value in [condition, semantic_source, context_source]:
        if value is not None and value not in CONDITIONS:
            raise ValueError(f"Unknown source: {value}")
        if value is not None and value not in active:
            raise ValueError(
                f"Source {value} is deferred; use --include-deferred for deliberate opt-in"
            )
    if seed is not None and seed not in range(4):
        raise ValueError("seed must be 0..3")
    if phase not in range(1, 6):
        raise ValueError("phase must be 1..5")
    output = Path(output)
    jobs = planned_jobs(output, config, include_deferred)
    pilot_sources = {"human", "random_human", config["pilot_llm_condition"]}
    semantic_sources = (
        {"human"} if phase == 1 else pilot_sources if phase == 2 else set(active)
    )
    pairs = set(phase_pairs(phase, config["pilot_llm_condition"], active))
    selected_semantic = [
        j for j in jobs["semantic"] if j["semantic_source"] in semantic_sources
    ]
    selected_context = [
        j
        for j in jobs["context"]
        if (j["semantic_source"], j["context_source"]) in pairs
    ]
    if phase == 1:
        selected_semantic = [j for j in selected_semantic if j["seed"] == 0]
        selected_context = [j for j in selected_context if j["seed"] == 0]
    selected = (
        selected_context
        if stage == "context"
        else (
            selected_semantic
            if stage in {"semantic", "pilot"}
            else selected_semantic + selected_context
        )
    )
    selected = [
        j
        for j in selected
        if (condition is None or j["semantic_source"] == condition)
        and (semantic_source is None or j["semantic_source"] == semantic_source)
        and (context_source is None or j.get("context_source") == context_source)
        and (seed is None or j["seed"] == seed)
    ]
    training = (
        config["semantic"]["pilot"] if stage == "pilot" else config["semantic"]["full"]
    )
    for job in selected:
        job["expected_output_files"] = (
            ["checkpoint.pt", "training_metrics.csv", "manifest.json"]
            if "context_source" in job
            else [
                "checkpoint.pt",
                "training_state.pt",
                "training_metrics.json",
                "exposure.json",
                "representations.npz",
                "manifest.json",
            ]
        )
        if "context_source" not in job and training:
            job["memory"] = estimate_memory(
                CONDITIONS[job["semantic_source"]],
                batch_size=training["batch_size"],
                chunk_size=training["chunk_size"],
            )
    blockers = []
    if stage in {"semantic", "context", "pilot"}:
        for file, description in [
            (
                "classification/frozen/manifest.json",
                "Complete, reviewed frozen task assignments",
            ),
            ("gates/preflight.json", "Current test/fidelity/preservation preflight"),
        ]:
            if not (output / file).exists():
                blockers.append(description)
        production = stage != "pilot" and not (stage == "context" and phase == 1)
        if production and not config["semantic"]["schedule_locked"]:
            blockers.append(
                "Training-only pilot reviewed and full semantic schedule locked"
            )
        if production and not (output / "gates/benchmark.json").exists():
            blockers.append("Small/medium/ensemble resource benchmarks")
        if (
            stage == "context"
            and phase > 1
            and not (output / "evaluations/semantic/manifest.json").exists()
        ):
            blockers.append("Stage 1 semantic evaluation before downstream testing")
    return {
        "protocol": config["protocol"],
        "config_sha256": digest(config),
        "stage": stage,
        "phase": phase,
        "design": {
            "sources": len(active),
            "source_pairs": len(active) ** 2,
            "semantic_models": 4 * len(active),
            "context_models": 4 * len(active) ** 2,
            "seeds": config["seeds"],
            "randomization_realizations": 1,
        },
        "full_design": {
            "sources": 14,
            "source_pairs": 196,
            "semantic_models": 56,
            "context_models": 784,
            "seeds": config["seeds"],
            "randomization_realizations": 1,
        },
        "enabled_conditions": active,
        "deferred_conditions": [c for c in CONDITIONS if c not in active],
        "jobs": selected,
        "semantic_training": training,
        "schedule_locked": config["semantic"]["schedule_locked"],
        "blocked": bool(blockers),
        "blockers": blockers,
        "output_directory": str(output),
        "scientific_artifacts_written": False,
    }
