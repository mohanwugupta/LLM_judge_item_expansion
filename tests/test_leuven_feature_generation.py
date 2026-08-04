import json
from pathlib import Path

import pandas as pd
import pytest

from leuven_expansion.feature_prompts import load_generation_prompts
from leuven_expansion.generate_features import (
    build_generation_jobs,
    load_stimulus_words,
    preflight_feature_generation,
    revalidate_generation_outputs,
    run_feature_generation,
    select_stimulus_words,
    validate_generation_output,
)


class FakeGenerationClient:
    def __init__(self):
        self.calls = []

    def generate(self, system_prompt, user_prompt, **kwargs):
        self.calls.append((system_prompt, user_prompt, kwargs))
        word = user_prompt.splitlines()[-1]
        seed = kwargs["seed"]
        response = {
            "target_word": word,
            "features": [f"describes {word}", f"sample feature {seed}"],
        }
        return json.dumps(response), {
            "finish_reason": "stop",
            "prompt_tokens": 100,
            "completion_tokens": 20,
        }


@pytest.fixture
def item_csv(tmp_path):
    path = tmp_path / "items.csv"
    pd.DataFrame(
        {
            "Name": ["Dog", "Chair"],
            "existing feature never sent": [4, 0],
        }
    ).to_csv(path, index=False)
    return path


def test_generation_output_accepts_fewer_than_ten_features():
    record, error = validate_generation_output(
        '{"target_word":"dog","features":["has fur","is a pet"]}',
        expected_word="dog",
    )
    assert error is None
    assert record["features"] == ["has fur", "is a pet"]


def test_generation_output_deduplicates_features_case_insensitively():
    record, error = validate_generation_output(
        '{"target_word":"dog","features":["Has fur","has fur","is a pet"]}',
        expected_word="dog",
    )
    assert error is None
    assert record["features"] == ["Has fur", "is a pet"]


def test_generation_output_rejects_wrong_word_and_extra_fields():
    _, word_error = validate_generation_output(
        '{"target_word":"cat","features":[]}', expected_word="dog"
    )
    _, schema_error = validate_generation_output(
        '{"target_word":"dog","features":[],"feature_id":1}',
        expected_word="dog",
    )
    assert "does not match" in word_error
    assert "Additional properties" in schema_error


def test_generation_output_accepts_parenthesis_formatting_difference():
    record, error = validate_generation_output(
        '{"target_word":"hot air balloon","features":["can fly"]}',
        expected_word="(hot air) balloon",
    )
    assert error is None
    assert record["features"] == ["can fly"]


def test_generation_output_does_not_drop_parenthetical_content():
    _, error = validate_generation_output(
        '{"target_word":"balloon","features":["can fly"]}',
        expected_word="(hot air) balloon",
    )
    assert "does not match" in error


def test_jobs_pair_reproducible_seeds_across_prompts_without_schema_content(
    item_csv,
):
    words = load_stimulus_words(item_csv)
    jobs = build_generation_jobs(
        words,
        responses_per_word=3,
        model="test-model",
        base_seed=7,
        system_prompts=load_generation_prompts(),
    )
    assert len(jobs) == 18
    assert len({job["response_id"] for job in jobs}) == 18
    assert len({job["sampling_seed"] for job in jobs}) == 6
    paired = {}
    for job in jobs:
        key = (job["word_normalized"], job["replicate_id"])
        paired.setdefault(key, set()).add(job["sampling_seed"])
    assert all(len(seeds) == 1 for seeds in paired.values())
    assert {job["prompt_variant"] for job in jobs} == {"A", "B", "C"}
    assert all("existing feature never sent" not in job["user_message"] for job in jobs)


def test_preflight_plans_full_replication_without_creating_outputs(
    item_csv, tmp_path
):
    output_dir = tmp_path / "not_created"
    plan = preflight_feature_generation(
        job_id="test_v3",
        input_csv=item_csv,
        output_dir=output_dir,
        model="test-model",
        responses_per_word=20,
        temperature=0.8,
        base_seed=7,
    )
    assert plan["word_count"] == 2
    assert plan["total_planned_responses"] == 120
    assert plan["planned_responses_by_prompt"] == {"A": 40, "B": 40, "C": 40}
    assert plan["unique_response_ids"] == 120
    assert plan["unique_sampling_seeds"] == 40
    assert not output_dir.exists()


def test_smoke_subset_is_deterministic_and_spans_the_word_list():
    words = [
        {"word_original": str(index), "word_normalized": str(index)}
        for index in range(7)
    ]
    selected = select_stimulus_words(words, max_words=3)
    assert [word["word_normalized"] for word in selected] == ["0", "3", "6"]


def test_preflight_applies_smoke_word_limit_to_all_three_prompts(
    item_csv, tmp_path
):
    plan = preflight_feature_generation(
        job_id="test_v3_smoke",
        input_csv=item_csv,
        output_dir=tmp_path / "not_created",
        model="test-model",
        responses_per_word=2,
        max_words=1,
    )
    assert plan["source_word_count"] == 2
    assert plan["word_count"] == 1
    assert plan["selected_words"] == ["dog"]
    assert plan["total_planned_responses"] == 6
    assert plan["planned_responses_by_prompt"] == {"A": 2, "B": 2, "C": 2}


def test_v3_1_preflight_records_versioned_protocol_and_prompts(item_csv, tmp_path):
    plan = preflight_feature_generation(
        job_id="test_v3_1_smoke",
        input_csv=item_csv,
        output_dir=tmp_path / "not_created",
        model="newer-test-model",
        prompt_version="v3.1",
        responses_per_word=2,
        max_words=1,
    )
    assert plan["protocol_version"] == (
        "leuven_free_generation_v3_1_three_prompt_comparison"
    )
    assert plan["prompt_version"] == "v3.1"
    assert plan["total_planned_responses"] == 6
    assert set(plan["prompt_text_by_variant"]) == {"A", "B", "C"}
    assert len(set(plan["prompt_sha256_by_variant"].values())) == 3
    assert not (tmp_path / "not_created").exists()


def test_mock_generation_writes_resumable_raw_and_derived_outputs(
    item_csv, tmp_path
):
    output_dir = tmp_path / "output"
    client = FakeGenerationClient()
    run_feature_generation(
        job_id="test_v3",
        input_csv=item_csv,
        output_dir=output_dir,
        client=client,
        model="test-model",
        responses_per_word=3,
        temperature=0.8,
        max_workers=2,
        base_seed=7,
        resume=True,
    )

    generations = pd.read_csv(output_dir / "feature_generations.csv")
    response_schema = client.calls[0][2]["response_format"]["json_schema"]["schema"]
    assert "uniqueItems" not in response_schema["properties"]["features"]

    long_features = pd.read_csv(output_dir / "generated_features_long.csv")
    frequencies = pd.read_csv(output_dir / "generated_feature_frequencies.csv")
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert len(client.calls) == 18
    assert len(generations) == 18
    assert len(long_features) == 36
    assert not frequencies.empty
    assert generations["parse_error"].isna().all()
    assert set(generations["prompt_variant"]) == {"A", "B", "C"}
    assert frequencies.groupby("prompt_variant")["valid_response_count"].max().to_dict() == {
        "A": 3,
        "B": 3,
        "C": 3,
    }
    assert (
        manifest["protocol_version"]
        == "leuven_free_generation_v3_three_prompt_comparison"
    )
    assert set(manifest["prompt_text_by_variant"]) == {"A", "B", "C"}
    assert len(set(manifest["prompt_sha256_by_variant"].values())) == 3
    assert manifest["existing_feature_schema_shown_to_model"] is False
    assert manifest["valid_responses_total"] == 18
    assert manifest["valid_responses_by_prompt"] == {"A": 6, "B": 6, "C": 6}
    assert manifest["pending_after_run"] == 0

    resume_client = FakeGenerationClient()
    run_feature_generation(
        job_id="test_v3",
        input_csv=item_csv,
        output_dir=output_dir,
        client=resume_client,
        model="test-model",
        responses_per_word=3,
        temperature=0.8,
        max_workers=2,
        base_seed=7,
        resume=True,
    )
    assert resume_client.calls == []
    assert len(pd.read_csv(output_dir / "feature_generations.csv")) == 18


def test_resume_refuses_changed_protocol(item_csv, tmp_path):
    output_dir = tmp_path / "output"
    run_feature_generation(
        job_id="test_v3",
        input_csv=item_csv,
        output_dir=output_dir,
        client=FakeGenerationClient(),
        model="test-model",
        responses_per_word=1,
        temperature=0.8,
        base_seed=7,
        resume=True,
    )
    with pytest.raises(ValueError, match="changed v3 protocol"):
        run_feature_generation(
            job_id="test_v3",
            input_csv=item_csv,
            output_dir=output_dir,
            client=FakeGenerationClient(),
            model="test-model",
            responses_per_word=2,
            temperature=0.8,
            base_seed=7,
            resume=True,
        )


def test_revalidation_repairs_preserved_raw_response_without_model_call(
    item_csv, tmp_path
):
    output_dir = tmp_path / "output"
    client = FakeGenerationClient()
    run_feature_generation(
        job_id="test_v3",
        input_csv=item_csv,
        output_dir=output_dir,
        client=client,
        model="test-model",
        responses_per_word=1,
        base_seed=7,
        resume=True,
    )

    generations_path = output_dir / "feature_generations.csv"
    generations = pd.read_csv(generations_path, dtype=str).fillna("")
    target_index = generations.index[0]
    target_response_id = generations.loc[target_index, "response_id"]
    returned_word = generations.loc[target_index, "word_normalized"].title()
    generations.loc[target_index, "raw_json"] = json.dumps(
        {"target_word": returned_word, "features": ["has fur"]}
    )
    generations.loc[target_index, "features_json"] = "[]"
    generations.loc[target_index, "n_features"] = "0"
    generations.loc[target_index, "parse_error"] = "old validator error"
    generations.to_csv(generations_path, index=False)

    summary = revalidate_generation_outputs(output_dir)
    latest = pd.read_csv(generations_path, dtype=str).fillna("").drop_duplicates(
        "response_id", keep="last"
    )
    manifest = json.loads((output_dir / "manifest.json").read_text())

    assert summary["repaired_responses"] == 1
    assert summary["model_calls"] == 0
    repaired = latest[latest["response_id"] == target_response_id].iloc[0]
    assert repaired["parse_error"] == ""
    assert json.loads(repaired["features_json"]) == ["has fur"]
    assert manifest["revalidated_responses_total"] == 1
    assert (output_dir / "manifest.pre_revalidation.json").exists()

    second_summary = revalidate_generation_outputs(output_dir)
    assert second_summary["candidate_errors"] == 0
    assert second_summary["repaired_responses"] == 0


def test_multiple_responses_require_sampling_variation(item_csv, tmp_path):
    with pytest.raises(ValueError, match="temperature > 0"):
        run_feature_generation(
            job_id="test_v3",
            input_csv=item_csv,
            output_dir=tmp_path / "output",
            client=FakeGenerationClient(),
            model="test-model",
            responses_per_word=2,
            temperature=0,
        )


def test_lfs_pointer_is_rejected_with_cluster_instruction(tmp_path):
    path = Path(tmp_path) / "pointer.csv"
    path.write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:abc\nsize 123\n"
    )
    with pytest.raises(ValueError, match="git lfs pull"):
        load_stimulus_words(path)
