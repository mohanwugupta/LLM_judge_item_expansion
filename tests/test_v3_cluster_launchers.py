from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SMOKE = (ROOT / "run_leuven_v3_smoke_test.sh").read_text()
PRODUCTION = (ROOT / "run_leuven_v3_generation.sh").read_text()


def test_v3_launchers_are_standalone_cluster_jobs():
    for script in [SMOKE, PRODUCTION]:
        assert "run_leuven_full_labels.sh" not in script
        assert "PROJECT_DIR=/scratch/gpfs/JORDANAT/mg9965/FalseMemoryISC-CI/LLM_judge_item_expansion" in script
        assert "CONDA_ENV=PromptControlText" in script
        assert "MODEL_DIR_NAME=Qwen--Qwen2.5-72B-Instruct" in script
        assert "TENSOR_PARALLEL_SIZE=4" in script
        assert "module load anaconda3/2025.6" in script
        assert "python -m vllm.entrypoints.openai.api_server" in script
        assert "python -m leuven_expansion.generate_features" in script


def test_v3_smoke_uses_established_test_partition_and_small_plan():
    assert "#SBATCH --partition=test" in SMOKE
    assert "#SBATCH --time=1:00:00" in SMOKE
    assert "MAX_WORDS=3" in SMOKE
    assert "RESPONSES_PER_WORD=2" in SMOKE
    assert "--max-words" in SMOKE
    assert "expected={'A':6,'B':6,'C':6}" in SMOKE


def test_v3_production_has_fixed_comparable_protocol():
    assert "RESPONSES_PER_WORD=20" in PRODUCTION
    assert "TEMPERATURE=0.8" in PRODUCTION
    assert "BASE_SEED=20260801" in PRODUCTION
    assert "JOB_ID=leuven_v3_qwen2_5_72b" in PRODUCTION
    assert "--max-words" not in PRODUCTION


def test_v3_production_runs_materialization_and_protocol_preflight_first():
    assert 'INPUT_CSV="$PROJECT_DIR/$LEUVEN_FEATURES"' in PRODUCTION
    assert '[ ! -s "$INPUT_CSV" ]' in PRODUCTION
    assert "version https://git-lfs.github.com/spec/v1" in PRODUCTION
    assert '"${GENERATION_ARGS[@]}"' in PRODUCTION
    assert "--preflight-only" in PRODUCTION
    assert PRODUCTION.index("--preflight-only") < PRODUCTION.index(
        "python -m vllm.entrypoints.openai.api_server"
    )
    assert "export VLLM_HOST_IP=127.0.0.1" in PRODUCTION
    assert 'export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"' in PRODUCTION
    assert "TRANSFORMERS_CACHE=" not in PRODUCTION
    assert "VLLM_CACHE_DIR=" not in PRODUCTION
    assert "VLLM_USAGE_STATS_DIR=" not in PRODUCTION
