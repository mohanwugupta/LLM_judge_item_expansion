import json
from pathlib import Path

import pytest

from leuven_expansion import stage1_collection as sc


def test_prepare_preserves_original_domain_and_never_invents_feature_text(tmp_path):
    sc.prepare(tmp_path)
    objects = sc.read_csv(tmp_path / "inputs/objects.csv")
    features = sc.read_csv(tmp_path / "inputs/v2_features.csv")
    assert len(objects) == 350
    assert [int(o["object_index"]) for o in objects] == list(range(350))
    assert sum(o["identity_reviewed"] == "true" for o in objects) == 324
    assert len(features) == 2541
    assert not any(f["feature_text"] for f in features)
    assert {int(f["original_task_id"]) for f in features} == set(range(33))
    sc.prepare(tmp_path)
    assert sc.read_csv(tmp_path / "inputs/v2_features.csv") == features


def test_preflight_explains_blockers_without_model_calls(tmp_path):
    sc.prepare(tmp_path)
    plan = sc.preflight(tmp_path, "judge", 4)
    assert not plan["ready"]
    assert plan["unreviewed_objects"] == 26
    assert plan["unverified_v2_features"] == 2541
    assert plan["expected_units"] == 350 * 2541
    assert plan["training_authorized"] is False
    assert not (tmp_path / "production").exists()


def test_inputs_cannot_merge_objects_or_substitute_reduced_features(tmp_path):
    sc.prepare(tmp_path)
    path = tmp_path / "inputs/objects.csv"
    rows = sc.read_csv(path)
    rows[0]["display_name"] = rows[1]["display_name"]
    sc.write_csv(path, rows)
    with pytest.raises(ValueError, match="display"):
        sc.load_objects(tmp_path)
    sc.write_csv(path, rows[1:])
    with pytest.raises(ValueError, match="350"):
        sc.load_objects(tmp_path)


def test_full_feature_dictionary_requires_evidence_and_original_task_ids(tmp_path):
    sc.prepare(tmp_path)
    path = tmp_path / "inputs/v2_features.csv"
    rows = sc.read_csv(path)
    for f in rows:
        f.update(
            feature_text="description " + f["feature_id"],
            verified="true",
            evidence="fixture",
        )
    sc.write_csv(path, rows)
    assert len(sc.load_v2_features(tmp_path, require_verified=True)) == 2541
    rows[0]["original_task_id"] = str((int(rows[0]["original_task_id"]) + 1) % 33)
    sc.write_csv(path, rows)
    with pytest.raises(ValueError, match="task"):
        sc.load_v2_features(tmp_path)


def test_resume_retries_errors_and_shards_cover_each_unit_once(tmp_path):
    units = [{"key": str(i), "value": i} for i in range(19)]
    partition = [[u for u in units if sc.shard(u["key"], 4) == i] for i in range(4)]
    assert sorted(u["value"] for part in partition for u in part) == list(range(19))
    calls = []

    def worker(unit):
        calls.append(unit["key"])
        if unit["key"] == "2" and calls.count("2") == 1:
            raise ValueError("failed response")
        return {"value": unit["value"]}

    signature = {"model": "fixture", "scope": "test"}
    status = sc.run_units(tmp_path, "test", signature, units, worker, workers=2)
    assert status["completed"] == 18 and status["pending"] == 1
    status = sc.run_units(tmp_path, "test", signature, units, worker, workers=2)
    assert status["complete"] and calls.count("2") == 2
    assert all(calls.count(str(i)) == 1 for i in range(19) if i != 2)
    with pytest.raises(ValueError, match="protocol"):
        sc.run_units(tmp_path, "test", {"model": "changed"}, units, worker)


def test_generation_reuses_original_prompts_and_paired_sampling():
    obj = {"object_id": "ISC0000", "display_name": "bee"}
    jobs = list(sc.generation_units([obj]))
    assert len(jobs) == 2 * 3 * 20
    assert {j["version"] for j in jobs} == {"v3", "v3.1"}
    assert {j["prompt_variant"] for j in jobs} == {"A", "B", "C"}
    assert len({j["sampling_seed"] for j in jobs if j["version"] == "v3"}) == 20
    for j in jobs:
        assert (
            j["system_prompt"]
            == sc.load_generation_prompts(j["version"])[j["prompt_variant"]]
        )


def test_generation_rejects_truncation_and_keeps_source_only_semantics():
    class Client:
        def generate(self, *args, **kwargs):
            return json.dumps({"target_word": "bee", "features": ["has wings"]}), {
                "finish_reason": "length"
            }

    job = next(sc.generation_units([{"object_id": "ISC0000", "display_name": "bee"}]))
    with pytest.raises(ValueError, match="finish"):
        sc.generate_one(job, Client())


def test_original_taxonomy_untouched_new_labels_are_explicit_reconstruction():
    taxonomy = sc.collection_taxonomy()
    assert len(taxonomy["classes"]) == 33
    assert taxonomy["classes"][32]["name"] == "Associated object"
    assert taxonomy["historical_numeric_correspondence"] is False
    assert "32-class" not in sc.classifier_instruction()


def test_export_rejects_symlink_escape_and_does_not_include_training(tmp_path):
    sc.prepare(tmp_path)
    outside = tmp_path.parent / (tmp_path.name + "-secret")
    outside.write_text("not an artifact")
    (tmp_path / "escape").symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        sc.export_bundle(tmp_path, tmp_path.parent / (tmp_path.name + ".tar.gz"))


def test_slurm_preflights_before_loading_qwen_and_keeps_resources():
    script = Path("run_stage1_collection.sh").read_text()
    assert "#SBATCH --gres=gpu:4" in script
    assert "#SBATCH --constraint=gpu80" in script
    assert "#SBATCH --cpus-per-task=8" in script
    assert "#SBATCH --mem=32G" in script
    assert script.index("--stage preflight") < script.index("api_server")
    assert "--stage train" not in script


class FakeClient:
    def __init__(self):
        self.calls = 0

    def generate(self, system_prompt, user_prompt, **kwargs):
        import re

        self.calls += 1
        if user_prompt.startswith("stimulus_word:"):
            result = {
                "target_word": user_prompt.splitlines()[-1],
                "features": ["has wings"],
            }
        elif "taxonomy" in user_prompt:
            features = json.loads(user_prompt)["features"]
            result = {
                "labels": [
                    dict(
                        feature_id=f["feature_id"],
                        task_id=6,
                        uncertain=False,
                        out_of_inventory=False,
                        explanation="fixture",
                    )
                    for f in features
                ]
            }
        else:
            word = re.search(r"target_word:\s*([^\n]+)", user_prompt).group(1)
            feature = int(re.search(r"feature_id:\s*(\d+)", user_prompt).group(1))
            result = dict(
                target_word=word,
                feature_id=feature,
                feature_value=0,
                confidence=1,
                ambiguous=False,
                reason="fixture",
            )
        return json.dumps(result), {"finish_reason": "stop"}


def test_smoke_is_isolated_and_resumes_without_production_labels(tmp_path):
    sc.prepare(tmp_path)
    sc.atomic_json(tmp_path / "model_fingerprint.json", {"revision": "fixture"})
    client = FakeClient()
    result = sc.execute(tmp_path, "smoke", client=client, workers=2)
    assert result["passed"] and result["smoke_only"]
    assert len(list((tmp_path / "smoke/requests").glob("*.json"))) == client.calls
    count = client.calls
    assert sc.execute(tmp_path, "smoke", client=client, workers=2)["passed"]
    assert client.calls == count
    assert not (tmp_path / "production").exists()
    assert not any(
        r["verified"] == "true"
        for r in sc.read_csv(tmp_path / "inputs/v2_features.csv")
    )


def test_judge_uses_final_resolver_and_never_adds_frequency_filter():
    client = FakeClient()
    result = sc.judge_one(
        dict(word_normalized="bee", feature_id=4, feature_text="has wings"), client
    )
    assert client.calls == 3
    assert result["resolution"]["resolution_method"] == "unanimous"
    assert result["binary_value"] == 0
    assert " > 0" in result["binary_policy"]


def test_journal_preserves_bad_finish_responses(tmp_path):
    class Bad:
        def generate(self, *args, **kwargs):
            return "truncated raw response", {"finish_reason": "length"}

    with pytest.raises(ValueError, match="finish"):
        sc.JournalClient(Bad(), tmp_path).generate("system", "user")
    saved = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert saved["raw_response"] == "truncated raw response"


def test_archive_roundtrip_preserves_inputs_and_reports_partial_status(tmp_path):
    sc.prepare(tmp_path)
    destination = tmp_path.parent / (tmp_path.name + ".tar.gz")
    report = sc.export_bundle(tmp_path, destination)
    checked = sc.verify_bundle(destination)
    assert checked["passed"] and checked["files_verified"] == len(report["files"])
    assert not any(checked["collection_complete"].values())
    assert not checked["training_authorized"]


def test_missing_shards_or_changed_database_cannot_finalize(tmp_path):
    sc.prepare(tmp_path)
    objects = sc.read_csv(tmp_path / "inputs/objects.csv")
    for o in objects:
        o.update(identity_reviewed="true", identity_evidence="fixture reviewer")
    sc.write_csv(tmp_path / "inputs/objects.csv", objects)
    sc.atomic_json(tmp_path / "model_fingerprint.json", {"revision": "fixture"})
    signature = sc.signature(tmp_path, "generate", 2)
    sc.freeze(tmp_path / "protocols/generate.json", signature)
    with pytest.raises(FileNotFoundError):
        sc.finalize(tmp_path, "generate")
    folder = tmp_path / "production/generate/0000"
    units = [dict(key="fixture", value=0)]
    sc.run_units(folder, "generate", signature, units, lambda u: u)
    with (folder / "responses.sqlite3").open("ab") as f:
        f.write(b"corruption")
    with pytest.raises(ValueError, match="database changed"):
        list(sc.finalized_rows(tmp_path, "generate"))


def test_production_generation_and_classification_finalize_across_shards(
    tmp_path, monkeypatch
):
    settings = sc.config()
    settings["responses_per_word"] = 1
    monkeypatch.setattr(sc, "config", lambda: settings)
    sc.prepare(tmp_path)
    objects = sc.read_csv(tmp_path / "inputs/objects.csv")
    for row in objects:
        row.update(identity_reviewed="true", identity_evidence="fixture reviewer")
    sc.write_csv(tmp_path / "inputs/objects.csv", objects)
    sc.atomic_json(tmp_path / "model_fingerprint.json", {"revision": "fixture"})
    for index in range(2):
        assert sc.execute(tmp_path, "generate", index, 2, FakeClient(), 4)["passed"]
    result = sc.finalize(tmp_path, "generate")
    assert result["complete"] and result["completed_units"] == 350 * 6
    rows = sc.read_csv(tmp_path / "finalized/generate/feature_generations.csv")
    assert len({r["object_id"] for r in rows}) == 350
    assert {r["version"] for r in rows} == {"v3", "v3.1"}
    assert len(sc.read_csv(tmp_path / "finalized/generate/candidate_features.csv")) == 1
    for index in range(2):
        assert sc.execute(tmp_path, "classify", index, 2, FakeClient(), 2)["passed"]
    assert sc.finalize(tmp_path, "classify")["completed_units"] == 1
    assert not sc.finalize(tmp_path, "classify")["training_authorized"]


def test_status_exposes_committed_errors_and_successes(tmp_path):
    directory = tmp_path / "production/generate/0000"
    sc.run_units(
        directory,
        "generate",
        {},
        [{"key": "one"}, {"key": "two"}],
        lambda u: {"ok": True} if u["key"] == "one" else 1 / 0,
        workers=1,
    )
    report = sc.collection_status(tmp_path)
    assert report["generate"]["done"] == 1
    assert report["generate"]["error"] == 1
