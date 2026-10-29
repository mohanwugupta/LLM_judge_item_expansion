import subprocess
from pathlib import Path


def test_classification_cluster_job_preserves_established_resource_and_launch_pattern():
    root = Path(__file__).resolve().parents[1]
    path = root / "run_semantic_classification.sh"
    assert path.exists()
    text = path.read_text()
    for required in [
        "#SBATCH --gres=gpu:4",
        "#SBATCH --constraint=gpu80",
        "#SBATCH --cpus-per-task=8",
        "#SBATCH --mem=32G",
        "anaconda3/2025.6",
        "PromptControlText",
        "Qwen--Qwen2.5-72B-Instruct",
        "--tensor-parallel-size",
        "--stage preflight",
        "--stage classify-smoke",
    ]:
        assert required in text
    assert text.index("--stage preflight") < text.index("python -m vllm")
    assert (
        subprocess.run(["bash", "-n", str(path)], capture_output=True).returncode == 0
    )


def test_cpu_stage_launcher_is_explicit_and_does_not_auto_advance():
    root = Path(__file__).resolve().parents[1]
    path = root / "run_semantic_stage.sh"
    assert path.exists()
    text = path.read_text()
    assert "#SBATCH --cpus-per-task=8" in text and "#SBATCH --mem=32G" in text
    assert "#SBATCH --gres" not in text
    assert "PIPELINE_STAGE=${PIPELINE_STAGE:-smoke}" in text
    assert '--stage "$PIPELINE_STAGE"' in text
    assert (
        subprocess.run(["bash", "-n", str(path)], capture_output=True).returncode == 0
    )
