from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from .inventory import sha256
from .provenance import atomic_json, code_hashes, digest
from .taxonomy import taxonomy_hash


def saved_path(path, root):
    candidate = Path(path)
    for marker, parent in [
        ("/ISC-CI_LLM_validation/", root),
        ("/LLM_judge_item_expansion/", root.parent),
        (
            "/IntegratedSemanticsControlContextInference/",
            root
            / "inputs/reference_sources/IntegratedSemanticsControlContextInference",
        ),
    ]:
        if marker in str(path):
            translated = parent / str(path).split(marker, 1)[1]
            if translated.exists():
                return translated
    if candidate.exists():
        return candidate
    marker = "/IntegratedSemanticsControlContextInference/"
    if marker in str(path):
        translated = (
            root
            / "upstream/IntegratedSemanticsControlContextInference"
            / str(path).split(marker, 1)[1]
        )
        if translated.exists():
            return translated
    raise FileNotFoundError(f"Materialize frozen input before training: {path}")


def verify_sources(records, root=None):
    verified = {}
    for row in records:
        path = saved_path(row["path"], root) if root is not None else Path(row["path"])
        actual = sha256(path)
        if actual != row["sha256"]:
            raise ValueError(f"Frozen input changed: {path}")
        verified[str(path)] = actual
    return verified


def verify_history(root, config):
    rows = []
    for name in config["baseline_figure_manifests"]:
        manifest = root / name
        rows.extend(json.loads(manifest.read_text())["sources"])
        rows.append({"path": str(manifest), "sha256": sha256(manifest)})
    verified = verify_sources(rows, root)
    historical = root / config["frozen_behavior_manifest"]
    record = json.loads(historical.read_text())
    if not record.get("complete") or not record.get("fidelity_gate", {}).get("passed"):
        raise ValueError(
            "Frozen released exact / reconstructed-human statistical fidelity gate did not pass"
        )
    for name, expected in record["implementation_sha256"].items():
        path = (
            root / name
            if name == "run_human_behavior_validation.py"
            else root / "iscci_validation" / name
        )
        if sha256(path) != expected:
            raise ValueError(f"Frozen scientific implementation changed: {path}")
        verified[str(path)] = expected
    for dataset in record["human_datasets"].values():
        verified.update(verify_sources([dataset], root))
    for seed, expected in record.get("released_checkpoint_sha256", {}).items():
        path = (
            root
            / "upstream/IntegratedSemanticsControlContextInference/models"
            / f"1and2shot_isc-seed{seed}.pt"
        )
        verified.update(verify_sources([{"path": str(path), "sha256": expected}]))
    # Checkpoint protection supplements figure manifests, which primarily track inputs and predictions.
    for base in [
        root / "artifacts/validation_v4_20261002/models",
        root / "artifacts/random_feature_control_20261028/validation/models",
    ]:
        for path in sorted(base.glob("**/checkpoint.pt")):
            verified[str(path)] = sha256(path)
    return verified


def gate_signature(output, taxonomy):
    inventory = json.loads((output / "inventory/manifest.json").read_text())
    return {
        "code": code_hashes(),
        "taxonomy_sha256": taxonomy_hash(taxonomy),
        "inventory": inventory["signature"],
    }


def verify_gate(path, signature):
    gate = json.loads(Path(path).read_text())
    if gate.get("signature") != signature:
        raise ValueError(
            f"Gate signature mismatch: {path}; rerun the current preflight"
        )
    if gate.get("passed") is not True:
        raise ValueError(f"Required gate has not passed: {path}")
    return gate


def verify_frozen_mapping(output, condition, matrix_path, taxonomy):
    directory = output / "classification/frozen"
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("complete") is not True or manifest["signature"][
        "taxonomy_sha256"
    ] != taxonomy_hash(taxonomy):
        raise ValueError("Incomplete or mismatched task-assignment freeze")
    name = f"{condition}.json"
    if sha256(directory / name) != manifest["outputs"][name]:
        raise ValueError("Frozen mapping hash mismatch")
    mapping = json.loads((directory / name).read_text())
    if mapping["matrix_file_sha256"] != sha256(matrix_path):
        raise ValueError("Frozen mapping belongs to another matrix")
    return mapping, sha256(directory / name)


def preflight(root, output, taxonomy, config):
    before = verify_history(root, config)
    command = [sys.executable, "-m", "pytest", "-q", str(root / "tests")]
    result = subprocess.run(command, cwd=root.parent, capture_output=True, text=True)
    preservation_error = None
    try:
        after = verify_history(root, config)
    except (ValueError, FileNotFoundError) as error:
        after = None
        preservation_error = str(error)
    changes = (
        [
            {"path": path, "before": before.get(path), "after": after.get(path)}
            for path in sorted(set(before) | set(after))
            if before.get(path) != after.get(path)
        ]
        if after is not None
        else []
    )
    tests_passed = result.returncode == 0
    preservation_passed = preservation_error is None and before == after
    passed = tests_passed and preservation_passed
    record = {
        "signature": gate_signature(output, taxonomy),
        "passed": passed,
        "tests_passed": tests_passed,
        "preservation_passed": preservation_passed,
        "preservation_changes": changes,
        "preservation_error": preservation_error,
        "python_executable": sys.executable,
        "python_version": sys.version,
        "test_command": command,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "preserved_sources": before,
        "historical_fidelity_manifest_sha256": sha256(
            root / config["frozen_behavior_manifest"]
        ),
    }
    path = output / "gates/preflight.json"
    atomic_json(path, record)
    if not passed:
        print(f"Preflight failed. Full record: {path}", file=sys.stderr, flush=True)
        if not tests_passed:
            print(result.stdout, file=sys.stderr, flush=True)
            print(result.stderr, file=sys.stderr, flush=True)
        if not preservation_passed:
            print(
                "Preservation failure: " + (preservation_error or json.dumps(changes)),
                file=sys.stderr,
                flush=True,
            )
        raise ValueError(
            f"Preflight failed (pytest exit {result.returncode}; "
            f"preservation passed={preservation_passed}); see {path}"
        )
    return record
