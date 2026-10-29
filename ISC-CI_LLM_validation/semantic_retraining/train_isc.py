from __future__ import annotations

import time
import logging
import platform
import resource
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .modeling import ISCModel
from .provenance import digest


@dataclass(frozen=True)
class TrainingConfig:
    max_epochs: int
    batch_size: int = 64
    learning_rate: float = 0.05
    max_seconds: float = 3600
    max_absolute_error: float = 0.1
    stopping: str = "convergence"
    device: str = "cpu"
    chunk_size: int = 4096

    def __post_init__(self):
        if (
            self.max_epochs < 1
            or self.batch_size < 1
            or self.learning_rate <= 0
            or self.max_seconds <= 0
        ):
            raise ValueError("Training budget and learning rate must be positive")
        if (
            self.stopping not in {"convergence", "fixed_budget"}
            or not 0 < self.max_absolute_error < 1
        ):
            raise ValueError("Unknown stopping rule or invalid maximum error")
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be positive")


def output_chunks(model, inputs, chunk_size):
    hub = model.get_context_independent_rep(inputs)
    task = model.get_task_context_rep(inputs)
    dependent = torch.sigmoid(
        model.task_context_rep_to_context_dependent_rep_weights(task)
        + model.hub_to_context_dependent_rep_weights(hub)
    )
    for start in range(0, model.num_output, chunk_size):
        stop = min(start + chunk_size, model.num_output)
        direct, controlled = (
            model.hub_to_output_weights,
            model.context_dependent_rep_to_output_weights,
        )
        logits = F.linear(
            hub, direct.weight[start:stop], direct.bias[start:stop]
        ) + F.linear(
            dependent,
            controlled.weight[start:stop],
            None if controlled.bias is None else controlled.bias[start:stop],
        )
        yield start, stop, logits


def backward_bce(model, inputs, target_fn, chunk_size):
    """Exact dense BCE, streamed over outputs; no negative sampling or reweighting."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    loss_value = 0.0
    for start, stop, logits in output_chunks(model, inputs, chunk_size):
        targets = target_fn(start, stop).to(device=logits.device, dtype=logits.dtype)
        if targets.shape != logits.shape or not torch.isfinite(logits).all():
            raise ValueError("Invalid target shape or nonfinite semantic logits")
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="sum") / (
            len(inputs[0]) * model.num_output
        )
        loss.backward(retain_graph=stop < model.num_output)
        loss_value += loss.item()
    return loss_value


def estimate_memory(features, objects=293, tasks=32, batch_size=64, chunk_size=4096):
    parameters = objects * 64 + tasks * 16 + 16 * 128 + 64 * 128 + features * 193
    optimizer_and_gradients = parameters * 16
    matrix_bytes = objects * features
    activation_allowance = batch_size * min(features, chunk_size) * 4 * 8
    return {
        "parameters": parameters,
        "parameter_optimizer_gradient_bytes": optimizer_and_gradients,
        "matrix_bytes": matrix_bytes,
        "activation_allowance_bytes": activation_allowance,
        "estimated_bytes": optimizer_and_gradients
        + matrix_bytes
        + activation_allowance,
        "note": "Lower-bound working-set estimate, not an allocation guarantee; benchmark on execution device",
    }


@torch.no_grad()
def prediction_metrics(model, dataset, batch_size, chunk_size=4096):
    total_loss, maximum, positives, true_positives, correct, cells = (
        0.0,
        0.0,
        0,
        0,
        0,
        0,
    )
    for start in range(0, len(dataset), batch_size):
        indices = np.arange(start, min(start + batch_size, len(dataset)))
        inputs = [x.to(model.device) for x in dataset.inputs(indices)]
        for column_start, stop, logits in output_chunks(model, inputs, chunk_size):
            targets = dataset.targets(indices, column_start, stop).to(model.device)
            probabilities = logits.sigmoid()
            if not torch.isfinite(logits).all():
                raise ValueError("Nonfinite semantic predictions")
            total_loss += F.binary_cross_entropy_with_logits(
                logits, targets, reduction="sum"
            ).item()
            maximum = max(maximum, (probabilities - targets).abs().max().item())
            positives += int(targets.sum().item())
            true_positives += int(((probabilities >= 0.5) & (targets > 0)).sum().item())
            correct += int(((probabilities >= 0.5) == (targets > 0)).sum().item())
            cells += targets.numel()
    density = positives / cells
    return {
        "bce": total_loss / cells,
        "max_absolute_error": maximum,
        "positive_recall": true_positives / positives,
        "binary_accuracy": correct / cells,
        "target_positive_density": density,
        "all_negative_accuracy": 1 - density,
        "all_negative_bce": float(
            -density * np.log(1e-6) - (1 - density) * np.log1p(-1e-6)
        ),
        "all_negative_probability": 1e-6,
        "constant_probability": float(np.clip(density, 1e-6, 1 - 1e-6)),
        "constant_probability_bce": float(
            -density * np.log(np.clip(density, 1e-6, 1 - 1e-6))
            - (1 - density) * np.log1p(-np.clip(density, 1e-6, 1 - 1e-6))
        ),
        "evaluated_cells": cells,
    }


def fit(
    dataset,
    seed,
    config,
    resume_state=None,
    checkpoint_callback=None,
    checkpoint_every_steps=0,
):
    """Train from scratch, or resume an exactly matching optimizer/RNG cursor."""
    from iscci_validation.dataio import matrix_sha256
    import pandas as pd

    signature = digest(
        {
            "config": asdict(config),
            "seed": seed,
            "matrix": matrix_sha256(
                pd.DataFrame(
                    dataset.values,
                    index=dataset.object_ids,
                    columns=dataset.feature_ids,
                )
            ),
            "feature_tasks": dataset.feature_tasks.tolist(),
            "task_count": dataset.task_count,
        }
    )
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    model = ISCModel(
        len(dataset.object_ids),
        len(dataset.feature_ids),
        dataset.task_count,
        device=config.device,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    epoch, updates, cursor, order, history = 0, 0, 0, None, []
    exposure = np.zeros(len(dataset), dtype=np.int64)
    previous_seconds = 0.0
    if resume_state is not None:
        if resume_state["signature"] != signature:
            raise ValueError("Resume source/training signature mismatch")
        model.load_state_dict(resume_state["state_dict"])
        optimizer.load_state_dict(resume_state["optimizer"])
        rng.set_state(resume_state["numpy_rng"])
        torch.set_rng_state(resume_state["torch_rng"])
        epoch, updates, cursor = (
            resume_state["epoch"],
            resume_state["updates"],
            resume_state["cursor"],
        )
        order, history = resume_state["order"], list(resume_state["history"])
        exposure = np.array(resume_state["exposure"], dtype=np.int64)
        previous_seconds = resume_state["elapsed_seconds"]
    started = time.monotonic()

    def snapshot():
        return {
            "signature": signature,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "numpy_rng": rng.get_state(),
            "torch_rng": torch.get_rng_state(),
            "epoch": epoch,
            "updates": updates,
            "cursor": cursor,
            "order": order,
            "history": list(history),
            "exposure": exposure.copy(),
            "elapsed_seconds": previous_seconds + time.monotonic() - started,
        }

    metrics = prediction_metrics(model, dataset, config.batch_size, config.chunk_size)
    converged = metrics["max_absolute_error"] < config.max_absolute_error
    resource_limit = False
    try:
        while epoch < config.max_epochs and not (
            converged and config.stopping == "convergence"
        ):
            if previous_seconds + time.monotonic() - started >= config.max_seconds:
                resource_limit = True
                break
            if order is None:
                order = rng.permutation(len(dataset))
            indices = order[cursor : cursor + config.batch_size]
            inputs = [x.to(config.device) for x in dataset.inputs(indices)]
            optimizer.zero_grad(set_to_none=True)
            backward_bce(
                model,
                inputs,
                lambda start, stop: dataset.targets(indices, start, stop),
                config.chunk_size,
            )
            optimizer.step()
            exposure[indices] += 1
            updates += 1
            cursor += len(indices)
            if cursor == len(dataset):
                epoch += 1
                order, cursor = None, 0
                metrics = prediction_metrics(
                    model, dataset, config.batch_size, config.chunk_size
                )
                converged = metrics["max_absolute_error"] < config.max_absolute_error
                history.append(
                    {
                        "epoch": epoch,
                        "updates": updates,
                        "examples_seen": int(exposure.sum()),
                        **metrics,
                    }
                )
                if epoch % 25 == 0 or converged:
                    logging.getLogger(__name__).warning(
                        "ISC seed=%s epoch=%s updates=%s BCE=%.6f max_error=%.4f positive_recall=%.4f",
                        seed,
                        epoch,
                        updates,
                        metrics["bce"],
                        metrics["max_absolute_error"],
                        metrics["positive_recall"],
                    )
            if checkpoint_callback and (
                order is None
                or (checkpoint_every_steps and updates % checkpoint_every_steps == 0)
            ):
                checkpoint_callback(snapshot())
    except BaseException:
        if checkpoint_callback:
            checkpoint_callback(snapshot())
        raise
    metrics = prediction_metrics(model, dataset, config.batch_size, config.chunk_size)
    converged = metrics["max_absolute_error"] < config.max_absolute_error
    complete = not resource_limit and (
        converged or (config.stopping == "fixed_budget" and epoch == config.max_epochs)
    )
    result = {
        "complete": complete,
        "status": (
            "complete"
            if complete
            else "resource_limit" if resource_limit else "not_converged"
        ),
        "converged": converged,
        "epochs": epoch,
        "updates": updates,
        "examples_seen": int(exposure.sum()),
        "pattern_exposure_min": int(exposure.min()),
        "pattern_exposure_max": int(exposure.max()),
        "metrics": metrics,
        "initialization": "independent_random_no_pretrained_weights",
        "training_state": snapshot(),
        "config": asdict(config),
        "elapsed_seconds": previous_seconds + time.monotonic() - started,
        "resources": {
            "cpu_threads": torch.get_num_threads(),
            "device": config.device,
            "platform": platform.platform(),
            "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * (1 if platform.system() == "Darwin" else 1024),
            "cuda_device_name": (
                torch.cuda.get_device_name()
                if config.device.startswith("cuda")
                else None
            ),
        },
    }
    return model.eval(), history, result
