from __future__ import annotations

import importlib.util
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from .behavior import notebook_reference
from .simulations import score_arguments


@contextmanager
def original_analysis(upstream: Path):
    """Load the unedited released helpers without polluting project imports."""
    names = ("models", "data", "utils", "analysis")
    previous = {name: sys.modules.get(name) for name in names}
    try:
        for name in names:
            spec = importlib.util.spec_from_file_location(name, upstream / f"{name}.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        yield sys.modules["analysis"]
    finally:
        for name, value in previous.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


def item_fidelity(model, tasks, object_names, upstream, tolerance):
    rows = []
    loader = SimpleNamespace(
        dataset=SimpleNamespace(df=pd.DataFrame(index=object_names))
    )
    with original_analysis(upstream) as analysis:
        original_model = analysis.CICOModel(
            input_d=model.input_to_independent.num_embeddings,
            embedding_d=model.input_to_independent.embedding_dim,
            context_d=model.independent_to_context.out_features,
            context_dependent_d=model.context_to_dependent.out_features,
            task_output_d=model.context_to_task_output.out_features,
        )
        original_model.load_state_dict(model.state_dict())
        original_model.eval()
        for task in ("induction", "phenomena"):
            data = tasks[task].copy()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", pd.errors.SettingWithCopyWarning)
                original = analysis.calc_argument_strength(
                    data, original_model, loader
                )["ISC-CI"].values
            reconstructed = score_arguments(model, data, object_names)
            error = np.abs(original - reconstructed)
            rows.append(
                {
                    "task": task,
                    "queries": len(data),
                    "max_absolute_error": float(error.max()),
                    "tolerance": tolerance,
                    "passed": bool(np.all(error <= tolerance)),
                    "reference": "unedited upstream models.CICOModel and analysis.calc_argument_strength",
                }
            )
    return pd.DataFrame(rows)


def fidelity_gate(upstream, per_seed_metrics, item_checks, config):
    references = notebook_reference(upstream)
    checks = []
    for _, reference in references.iterrows():
        matching = per_seed_metrics[
            (per_seed_metrics["task"] == reference["task"])
            & (per_seed_metrics["dataset"] == reference["dataset"])
        ]
        for condition in ("released", "human"):
            selected = matching[matching["condition"] == condition]
            if condition == "released":
                selected = selected[selected["seed"] == 3]
            observed = (
                float(selected[reference["metric"]].mean()) if len(selected) else np.nan
            )
            tolerance = float(
                config["statistic_absolute_tolerance"]
                if condition == "released"
                else config["reconstructed_human_statistic_absolute_tolerance"]
            )
            interval = None
            if (
                condition == "human"
                and config.get("reconstructed_criterion")
                == "released_seed_prediction_interval"
            ):
                reference_runs = matching[matching["condition"] == "released"]
                values = reference_runs[reference["metric"]].dropna().values
                if len(values) != 4 or set(reference_runs["seed"]) != {0, 1, 2, 3}:
                    interval = (np.nan, np.nan)
                else:
                    center = float(np.mean(values))
                    spread = (
                        student_t.ppf(
                            (1 + config["reference_prediction_interval_level"]) / 2, 3
                        )
                        * np.std(values, ddof=1)
                        * np.sqrt(1 + 1 / 4)
                    )
                    interval = (center - spread, center + spread)
                tolerance = np.nan
            stochastic_required = reference["task"] != "context" or config.get(
                "require_exact_stochastic_notebook_reproduction", True
            )
            passes = (
                bool(
                    np.isfinite(observed)
                    and (interval[0] - 1e-12 <= observed <= interval[1] + 1e-12)
                )
                if interval
                else bool(
                    np.isfinite(observed)
                    and abs(observed - reference["reference"]) <= tolerance
                )
            )
            checks.append(
                {
                    **reference.to_dict(),
                    "condition": condition,
                    "observed": observed,
                    "seed_count": selected["seed"].nunique(),
                    "absolute_error": abs(observed - reference["reference"]),
                    "tolerance": tolerance,
                    "criterion": (
                        "reference_seed_prediction_interval_consistency"
                        if interval
                        else "absolute_numerical_tolerance"
                    ),
                    "reference_interval_low": interval[0] if interval else np.nan,
                    "reference_interval_high": interval[1] if interval else np.nan,
                    "passed": passes,
                    "required": (
                        stochastic_required
                        if condition == "released"
                        else config["require_reconstructed_human_reproduction"]
                    ),
                }
            )
    # The PDF and saved notebook disagree by a whole item, not floating-point rounding.
    context = references[references["task"] == "context"].iloc[0]
    difference = abs(float(context["reference"]) - 36 / 38)
    checks.append(
        {
            "task": "context",
            "dataset": "context",
            "metric": "source_consistency",
            "condition": "source_audit",
            "reference": 36 / 38,
            "observed": context["reference"],
            "absolute_error": difference,
            "tolerance": config["statistic_absolute_tolerance"],
            "source": "ISCCI_2026Jan_ArXiv.pdf page 73 versus saved notebook cell 11",
            "passed": difference <= config["statistic_absolute_tolerance"],
            "required": config["require_pdf_notebook_context_consistency"],
        }
    )
    table = pd.DataFrame(checks)
    required = table[table["required"]]
    human_seeds = set(
        per_seed_metrics.loc[per_seed_metrics["condition"] == "human", "seed"]
    )
    passed = bool(
        required["passed"].all()
        and item_checks["passed"].all()
        and human_seeds == {0, 1, 2, 3}
    )
    return table, {
        "passed": passed,
        "required_statistic_checks": len(required),
        "failed_required_statistic_checks": int((~required["passed"]).sum()),
        "original_item_probe_checks_passed": bool(item_checks["passed"].all()),
        "complete_human_seeds": human_seeds == {0, 1, 2, 3},
        "policy": "No LLM-condition behavioral evaluation or new training before all required fidelity checks pass.",
        "reconstructed_criterion": config.get(
            "reconstructed_criterion", "absolute_numerical_tolerance"
        ),
        "caveat": "The reference-seed prediction interval is a replication-consistency envelope, not a formal equivalence test or proof of behavioral validity. Four reference seeds provide limited precision. The criterion was specified after inspecting the initial human-only numerical gate, before LLM-condition behavior.",
    }
