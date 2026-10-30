import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from semantic_inputs import check_inputs, install_inputs, write_bundle


def fixture_bundle(tmp_path):
    source = tmp_path / "source"
    (source / "metadata").mkdir(parents=True)
    (source / "metadata/labels.csv").write_text("id,label\n1,has wings\n")
    (source / "frozen.json").write_text('{"complete":true}')
    bundle = tmp_path / "inputs.tar.gz"
    manifest = tmp_path / "manifest.json"
    write_bundle(source, ["metadata/labels.csv", "frozen.json"], bundle, manifest)
    return source, bundle, manifest


def test_self_contained_install_is_exact_and_idempotent(tmp_path):
    source, bundle, manifest = fixture_bundle(tmp_path)
    destination = tmp_path / "checkout"
    result = install_inputs(destination, bundle, manifest)
    assert result["installed"] == 2
    assert check_inputs(destination, manifest)["passed"]
    for relative in ["metadata/labels.csv", "frozen.json"]:
        assert (destination / relative).read_bytes() == (source / relative).read_bytes()
    assert install_inputs(destination, bundle, manifest)["installed"] == 0


def test_external_reference_bytes_are_preserved_without_rewriting_upstream(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    external = tmp_path / "external.csv"
    external.write_bytes(b"object,category\r\na,animal\r\n")
    bundle, manifest = tmp_path / "bundle.tar.gz", tmp_path / "manifest.json"
    names = {"inputs/reference_sources/categories.csv": external}
    write_bundle(source, names, bundle, manifest)
    first_hash = hashlib.sha256(bundle.read_bytes()).hexdigest()
    write_bundle(source, names, bundle, manifest)
    assert hashlib.sha256(bundle.read_bytes()).hexdigest() == first_hash
    destination = tmp_path / "checkout"
    upstream = destination / "upstream/categories.csv"
    upstream.parent.mkdir(parents=True)
    upstream.write_bytes(b"object,category\na,animal\n")
    install_inputs(destination, bundle, manifest)
    assert (destination / next(iter(names))).read_bytes() == external.read_bytes()
    assert upstream.read_bytes() == b"object,category\na,animal\n"


def test_install_command_requires_no_scientific_packages(tmp_path):
    import subprocess
    import sys

    _, bundle, manifest = fixture_bundle(tmp_path)
    script = Path(__file__).resolve().parents[1] / "run_semantic_inputs.py"
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            str(script),
            "--install",
            "--project-root",
            str(tmp_path / "checkout"),
            "--bundle",
            str(bundle),
            "--manifest",
            str(manifest),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["installed"] == 2


def test_published_manifest_covers_complete_scientific_input_inventory():
    from semantic_inputs import collect_frozen_inputs

    project = Path(__file__).resolve().parents[2]
    manifest = project / "ISC-CI_LLM_validation/configs/semantic_frozen_inputs.json"
    record = json.loads(manifest.read_text())
    assert {r["path"] for r in record["files"]} == set(collect_frozen_inputs(project))
    assert check_inputs(project, manifest)["passed"]


def test_all_missing_and_changed_files_are_reported(tmp_path):
    _, _, manifest = fixture_bundle(tmp_path)
    destination = tmp_path / "checkout"
    destination.mkdir()
    (destination / "frozen.json").write_text("different result")
    result = check_inputs(destination, manifest)
    assert not result["passed"]
    assert result["missing"] == ["metadata/labels.csv"]
    assert result["changed"] == ["frozen.json"]


def test_existing_scientific_results_are_never_overwritten(tmp_path):
    _, bundle, manifest = fixture_bundle(tmp_path)
    destination = tmp_path / "checkout"
    destination.mkdir()
    path = destination / "frozen.json"
    path.write_text("cluster result")
    with pytest.raises(ValueError, match="conflict"):
        install_inputs(destination, bundle, manifest)
    assert path.read_text() == "cluster result"
    assert not (destination / "metadata").exists()


def test_matching_lfs_pointer_can_be_materialized(tmp_path):
    source, bundle, manifest = fixture_bundle(tmp_path)
    destination = tmp_path / "checkout"
    destination.mkdir()
    data = (source / "frozen.json").read_bytes()
    (destination / "frozen.json").write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        f"oid sha256:{hashlib.sha256(data).hexdigest()}\nsize {len(data)}\n"
    )
    assert install_inputs(destination, bundle, manifest)["installed"] == 2
    assert (destination / "frozen.json").read_bytes() == data


def test_bad_archive_hash_writes_no_inputs(tmp_path):
    _, bundle, manifest = fixture_bundle(tmp_path)
    with bundle.open("ab") as stream:
        stream.write(b"corrupt")
    destination = tmp_path / "checkout"
    with pytest.raises(ValueError, match="bundle.*hash"):
        install_inputs(destination, bundle, manifest)
    assert not destination.exists()


@pytest.mark.parametrize("name", ["../escape", "/escape", ".git/config", "."])
def test_manifest_rejects_unsafe_paths(tmp_path, name):
    _, bundle, manifest = fixture_bundle(tmp_path)
    record = json.loads(manifest.read_text())
    record["files"][0]["path"] = name
    manifest.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="[Uu]nsafe"):
        install_inputs(tmp_path / "checkout", bundle, manifest)


@pytest.mark.parametrize("kind", ["symlink", "duplicate", "corrupt_payload"])
def test_unsafe_archive_members_write_no_inputs(tmp_path, kind):
    _, bundle, manifest = fixture_bundle(tmp_path)
    record = json.loads(manifest.read_text())
    with tarfile.open(bundle, "w:gz") as archive:
        for index, row in enumerate(record["files"]):
            data = (tmp_path / "source" / row["path"]).read_bytes()
            member = tarfile.TarInfo(row["path"])
            if index == 0 and kind == "symlink":
                member.type = tarfile.SYMTYPE
                member.linkname = "/tmp/outside"
                archive.addfile(member)
                continue
            if index == 0 and kind == "corrupt_payload":
                data = b"x" * len(data)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
            if index == 0 and kind == "duplicate":
                archive.addfile(member, io.BytesIO(data))
    record["bundle_sha256"] = hashlib.sha256(bundle.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(record))
    destination = tmp_path / "checkout"
    with pytest.raises(ValueError):
        install_inputs(destination, bundle, manifest)
    assert not (destination / "frozen.json").exists()
    assert not (destination / "metadata").exists()


def test_symlink_destination_is_rejected(tmp_path):
    _, bundle, manifest = fixture_bundle(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    destination = tmp_path / "checkout"
    destination.mkdir()
    (destination / "metadata").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="[Ss]ymlink"):
        install_inputs(destination, bundle, manifest)
    assert not list(outside.iterdir())


def test_cluster_bootstrap_precedes_scientific_prepare():
    root = Path(__file__).resolve().parents[1]
    launcher = (root / "run_semantic_classification.sh").read_text()
    assert "run_semantic_inputs.py --install" in launcher
    assert launcher.index("run_semantic_inputs.py --install") < launcher.index(
        "--stage prepare"
    )
