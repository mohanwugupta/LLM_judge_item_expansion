"""Portable, checksum-verified installation of frozen scientific inputs."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import tempfile


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def target_path(root, name):
    relative = PurePosixPath(name)
    if (
        not name
        or not relative.parts
        or relative.is_absolute()
        or str(relative) != name
        or any(p in {"..", ".git"} for p in relative.parts)
        or "\\" in name
    ):
        raise ValueError(f"Unsafe input path: {name}")
    root = Path(root).resolve()
    target = root.joinpath(*relative.parts)
    for parent in [target, *target.parents]:
        if parent == root:
            break
        if parent.is_symlink():
            raise ValueError(f"Symlink input destination is not allowed: {parent}")
    return target


def load_manifest(path):
    record = json.loads(Path(path).read_text())
    if record.get("version") != 1 or not record.get("files"):
        raise ValueError("Unsupported or empty frozen-input manifest")
    seen = set()
    for row in record["files"]:
        target_path(Path.cwd(), row["path"])
        if row["path"] in seen:
            raise ValueError("Duplicate frozen-input manifest path")
        seen.add(row["path"])
        if (
            type(row.get("size")) is not int
            or row["size"] < 0
            or not re.fullmatch(r"[0-9a-f]{64}", row.get("sha256", ""))
        ):
            raise ValueError("Invalid frozen-input hash/size")
    if not re.fullmatch(r"[0-9a-f]{64}", record.get("bundle_sha256", "")):
        raise ValueError("Invalid frozen-input bundle hash")
    return record


def matches(path, row):
    return (
        path.is_file()
        and path.stat().st_size == row["size"]
        and file_hash(path) == row["sha256"]
    )


def matching_pointer(path, row):
    if not path.is_file() or path.stat().st_size > 1024:
        return False
    expected = (
        "version https://git-lfs.github.com/spec/v1\n"
        f"oid sha256:{row['sha256']}\nsize {row['size']}\n"
    ).encode()
    return path.read_bytes() == expected


def check_inputs(root, manifest):
    record = load_manifest(manifest)
    missing, changed = [], []
    for row in record["files"]:
        path = target_path(root, row["path"])
        if not path.exists():
            missing.append(row["path"])
        elif not matches(path, row):
            changed.append(row["path"])
    return {
        "passed": not missing and not changed,
        "files": len(record["files"]),
        "missing": missing,
        "changed": changed,
    }


def write_bundle(root, names, bundle, manifest):
    root, bundle, manifest = Path(root).resolve(), Path(bundle), Path(manifest)
    aliases = names if isinstance(names, dict) else {}
    names = list(names)
    if not names or len(set(names)) != len(names):
        raise ValueError("Frozen input paths must be nonempty and unique")
    rows = []
    for name in sorted(names):
        path = target_path(root, name)
        path = Path(aliases.get(name, path))
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size < 1024 and path.read_bytes().startswith(
            b"version https://git-lfs.github.com/spec/v1\n"
        ):
            raise ValueError(f"Materialize LFS input before packaging: {name}")
        rows.append(
            {"path": name, "size": path.stat().st_size, "sha256": file_hash(path)}
        )
    bundle.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=bundle.parent, prefix=".frozen-inputs-")
    try:
        with os.fdopen(fd, "wb") as raw:
            with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as zipped:
                with tarfile.open(fileobj=zipped, mode="w") as archive:
                    for row in rows:
                        member = tarfile.TarInfo(row["path"])
                        member.size, member.mode, member.mtime = row["size"], 0o644, 0
                        path = Path(
                            aliases.get(row["path"], target_path(root, row["path"]))
                        )
                        with path.open("rb") as stream:
                            archive.addfile(member, stream)
                        if not matches(path, row):
                            raise ValueError(f"Source changed while packaging: {path}")
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(temporary, bundle)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    record = {"version": 1, "bundle_sha256": file_hash(bundle), "files": rows}
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def install_inputs(root, bundle, manifest):
    root, bundle = Path(root).resolve(), Path(bundle)
    record = load_manifest(manifest)
    needed, conflicts = [], []
    for row in record["files"]:
        path = target_path(root, row["path"])
        if matches(path, row):
            continue
        if path.exists() and not matching_pointer(path, row):
            conflicts.append(row["path"])
        else:
            needed.append(row)
    if conflicts:
        raise ValueError(
            "Existing frozen input conflict; nothing replaced: " + ", ".join(conflicts)
        )
    if not needed:
        return {"passed": True, "verified": len(record["files"]), "installed": 0}
    if not bundle.is_file() or file_hash(bundle) != record["bundle_sha256"]:
        raise ValueError(
            "Frozen input bundle missing or hash mismatch. Run git lfs pull "
            "--include='ISC-CI_LLM_validation/inputs/semantic_frozen_inputs.tar.gz' --exclude=''"
        )
    expected = {row["path"]: row for row in record["files"]}
    wanted = {row["path"] for row in needed}
    # Validate and stage every payload before creating any scientific input.
    with tempfile.TemporaryDirectory(prefix="iscci-frozen-inputs-") as temporary:
        stage = Path(temporary)
        seen = set()
        with tarfile.open(bundle, "r:gz") as archive:
            for member in archive:
                if (
                    member.name not in expected
                    or member.name in seen
                    or not member.isfile()
                ):
                    raise ValueError(
                        f"Unsafe, duplicate or unexpected archive member: {member.name}"
                    )
                seen.add(member.name)
                row = expected[member.name]
                if member.size != row["size"]:
                    raise ValueError(f"Archive payload size mismatch: {member.name}")
                digest = hashlib.sha256()
                staged = stage / str(len(seen))
                with archive.extractfile(member) as stream, staged.open("wb") as output:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
                        if member.name in wanted:
                            output.write(block)
                if digest.hexdigest() != row["sha256"]:
                    raise ValueError(f"Archive payload hash mismatch: {member.name}")
                if member.name in wanted:
                    row["staged"] = str(staged)
        if seen != set(expected):
            raise ValueError("Archive is missing frozen input payloads")
        installed = 0
        for row in needed:
            path = target_path(root, row["path"])
            if matches(path, row):
                continue
            if path.exists() and not matching_pointer(path, row):
                raise ValueError(f"Concurrent frozen input conflict: {row['path']}")
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, pending = tempfile.mkstemp(dir=path.parent, prefix=".frozen-input-")
            try:
                with os.fdopen(fd, "wb") as output, Path(row["staged"]).open(
                    "rb"
                ) as stream:
                    shutil.copyfileobj(stream, output)
                    output.flush()
                    os.fsync(output.fileno())
                if path.exists():
                    if not matching_pointer(path, row):
                        raise ValueError(
                            f"Concurrent frozen input conflict: {row['path']}"
                        )
                    os.replace(pending, path)
                else:
                    os.link(pending, path)
                installed += 1
            finally:
                if os.path.exists(pending):
                    os.unlink(pending)
    result = check_inputs(root, manifest)
    if not result["passed"]:
        raise ValueError(f"Frozen input installation verification failed: {result}")
    return {"passed": True, "verified": result["files"], "installed": installed}


def collect_frozen_inputs(project):
    from semantic_retraining.gates import verify_history
    from semantic_retraining.inventory import inventory
    from semantic_retraining.classify_features import retained_labels
    from semantic_retraining.run_crosses import load_config

    project = Path(project).resolve()
    root = project / "ISC-CI_LLM_validation"
    config = load_config(root / "configs/semantic_retraining.json")
    inputs = {}
    for name, expected in verify_history(root, config).items():
        path = Path(name).resolve()
        if not path.is_relative_to(project):
            marker = "/IntegratedSemanticsControlContextInference/"
            if marker not in str(path):
                raise ValueError(f"Unresolved external frozen input: {path}")
            relative = (
                "ISC-CI_LLM_validation/inputs/reference_sources/IntegratedSemanticsControlContextInference/"
                + str(path).split(marker, 1)[1]
            )
            if file_hash(path) != expected:
                raise ValueError(f"Frozen external input changed: {path}")
        else:
            relative = str(path.relative_to(project))
        inputs[relative] = path
    sources = inventory(root)
    for condition, record in sources.items():
        inputs[str(Path(record["path"]).relative_to(project))] = Path(record["path"])
        retained_labels(root, condition, record["feature_ids"])
    for generation, prefix in [("v3", "v3"), ("v3_1", "v3_1")]:
        for prompt in "ABC":
            name = f"ISC-CI_LLM_validation/artifacts/{generation}_consolidation/{prefix}_{prompt}_cluster_labels.csv"
            inputs[name] = project / name
    name = "artifacts/v4/discovery/candidate_bank.csv"
    inputs[name] = project / name
    path = root / config["frozen_behavior_manifest"]
    inputs[str(path.relative_to(project))] = path
    for name in [
        "human_reliability.csv",
        "human_participant_reliability.csv",
        "paper_human_choice_statistics.csv",
    ]:
        path = (root / config["frozen_behavior_manifest"]).parent / name
        inputs[str(path.relative_to(project))] = path
    upstream = root / "upstream/IntegratedSemanticsControlContextInference"
    for pattern in ["*.py", "*.ipynb", "data/**/*.csv", "models/*.pt"]:
        for path in upstream.glob(pattern):
            inputs[str(path.relative_to(project))] = path
    original = root / "upstream/IntegratedSemanticsControl"
    for name in ["models.py", "data.py", "utils.py", "data/leuven_rum_nozeroes.ex"]:
        path = original / name
        inputs[str(path.relative_to(project))] = path
    return dict(sorted(inputs.items()))
