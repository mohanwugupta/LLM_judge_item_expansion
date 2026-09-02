from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PRODUCTION = (ROOT / "run_leuven_v4_atomic.sh").read_text()
FINALIZE = (ROOT / "run_leuven_v4_atomic_finalize.sh").read_text()


def test_v4_atomic_resources_and_frozen_shards():
    assert "#SBATCH --cpus-per-task=8" in PRODUCTION
    assert "#SBATCH --mem=32G" in PRODUCTION
    assert "#SBATCH --gres=gpu:4" in PRODUCTION
    assert "#SBATCH --constraint=gpu80" in PRODUCTION
    assert "#SBATCH --array=0-31%4" in PRODUCTION
    assert "#SBATCH --time=24:00:00" in PRODUCTION
    assert "SHARD_COUNT=32" in PRODUCTION
    assert "SHARD_COUNT=32" in FINALIZE


def test_v4_atomic_rejects_invalid_array_indices_before_model_start():
    guard = 'if ! [[ "$SHARD_INDEX" =~ ^[0-9]+$ ]]'
    assert guard in PRODUCTION
    assert PRODUCTION.index(guard) < PRODUCTION.index(
        "python -m vllm.entrypoints.openai.api_server"
    )
    assert "Never represent repeated waves by extending the array beyond index 31" in PRODUCTION


def test_v4_atomic_materialization_preflight_precedes_model_start():
    assert "version https://git-lfs.github.com/spec/v1" in PRODUCTION
    assert "EXPECTED_CANDIDATE_BANK_SHA256=" in PRODUCTION
    assert 'python run_v4_judgments.py "${JUDGMENT_ARGS[@]}" --dry-run' in PRODUCTION
    assert "--preflight-shard" in PRODUCTION
    assert PRODUCTION.index("--dry-run") < PRODUCTION.index(
        "python -m vllm.entrypoints.openai.api_server"
    )
    assert PRODUCTION.index("--preflight-shard") < PRODUCTION.index(
        "python -m vllm.entrypoints.openai.api_server"
    )
