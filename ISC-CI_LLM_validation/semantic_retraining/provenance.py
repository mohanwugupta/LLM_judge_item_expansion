from __future__ import annotations

import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".")
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def code_hashes(scope="all") -> dict:
    from .inventory import sha256

    root = Path(__file__).resolve().parent
    critical = {
        "semantic": {
            "modeling.py",
            "training_adapter.py",
            "train_isc.py",
            "extract_representations.py",
            "provenance.py",
            "inventory.py",
            "cli.py",
            "run_crosses.py",
            "taxonomy.py",
            "gates.py",
        },
        "context": {
            "transfer_embeddings.py",
            "extract_representations.py",
            "modeling.py",
            "provenance.py",
            "inventory.py",
            "cli.py",
            "run_crosses.py",
            "gates.py",
            "taxonomy.py",
        },
    }
    if scope != "all" and scope not in critical:
        raise ValueError(f"Unknown code provenance scope: {scope}")
    paths = [
        p for p in root.glob("*.py") if scope == "all" or p.name in critical[scope]
    ] + [
        root.parent / "iscci_validation" / name
        for name in ["dataio.py", "modeling.py", "training.py"]
    ]
    return {str(p.relative_to(root.parent)): sha256(p) for p in sorted(paths)}


def atomic_file(path: Path, writer):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".")
    try:
        with os.fdopen(fd, "wb") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def validate_output_directory(path: Path):
    root = Path(__file__).resolve().parents[1]
    project = root.parent
    resolved = path.resolve()
    if resolved == project or project in resolved.parents:
        allowed = False
        for parent in (root / "artifacts", root / "reports"):
            if parent in resolved.parents:
                relative = resolved.relative_to(parent)
                allowed = relative.parts[0].startswith("semantic_retraining_")
                break
        if not allowed:
            raise ValueError(
                "Protected existing workspace: use a new semantic_retraining_ artifact/report directory"
            )


@contextmanager
def run_lock(directory: Path):
    import fcntl

    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".run.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f"Another process owns run: {directory}") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)
