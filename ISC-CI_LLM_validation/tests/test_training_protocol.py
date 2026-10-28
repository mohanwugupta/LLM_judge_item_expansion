import numpy as np
import pandas as pd
import torch
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

from iscci_validation.dataio import CICODataset
from iscci_validation.modeling import initialize_model
from iscci_validation.training import train_model
from run_validation import train_run


def test_episode_shapes_match_released_one_two_shot_sampler() -> None:
    matrix = pd.DataFrame(
        {
            "feature_a": [1, 1, 1, 1, 0],
            "feature_b": [0, 1, 1, 1, 1],
        }
    )
    np.random.seed(3)
    dataset = CICODataset(matrix, n_sequences=4, n_shot=(1, 2))
    for index in range(4):
        episode = dataset[index]
        assert episode["support_x"].shape == (2,)
        assert episode["query_x"].shape == (5,)
        assert episode["support_y"].shape == (2,)
        assert episode["query_y"].shape == (5,)


def test_released_embedding_is_frozen() -> None:
    embedding = torch.randn(6, 64)
    embedding[-1].zero_()
    model = initialize_model(task_count=3, released_embedding=embedding, seed=1)
    assert torch.equal(model.input_to_independent.weight, embedding)
    assert not model.input_to_independent.weight.requires_grad


def test_integer_task_draw_preserves_legacy_episodes_and_training(monkeypatch) -> None:
    matrix = pd.DataFrame({f"f{i}": [1, 1, 1, 1, i % 2, 0] for i in range(17)})
    embedding = torch.arange(7 * 64, dtype=torch.float32).reshape(7, 64) / 1000
    embedding[-1].zero_()
    np.random.seed(42)
    dataset = CICODataset(matrix)
    optimized_episodes = [dataset[i] for i in range(30)]
    optimized_state = np.random.get_state()
    optimized_model, optimized_metrics, _ = train_model(
        matrix, embedding, 3, 2, 16, 8, 0.001, 0.5
    )
    choice = np.random.choice

    def legacy_choice(population, *args, **kwargs):
        if isinstance(population, int):
            population = range(population)
        return choice(population, *args, **kwargs)

    monkeypatch.setattr(np.random, "choice", legacy_choice)
    np.random.seed(42)
    legacy_episodes = [dataset[i] for i in range(30)]
    legacy_state = np.random.get_state()
    for optimized, legacy in zip(optimized_episodes, legacy_episodes):
        for key in optimized:
            np.testing.assert_array_equal(optimized[key], legacy[key])
    for optimized, legacy in zip(optimized_state, legacy_state):
        np.testing.assert_array_equal(optimized, legacy)
    legacy_model, legacy_metrics, _ = train_model(
        matrix, embedding, 3, 2, 16, 8, 0.001, 0.5
    )
    pd.testing.assert_frame_equal(optimized_metrics, legacy_metrics)
    for key, weight in optimized_model.state_dict().items():
        assert torch.equal(weight, legacy_model.state_dict()[key])


def test_spawned_training_matches_serial_checkpoints(tmp_path) -> None:
    matrix = pd.DataFrame({"f0": [1, 1, 1, 1, 0], "f1": [0, 1, 1, 1, 1]})
    embedding = torch.arange(6 * 64, dtype=torch.float32).reshape(6, 64) / 1000
    embedding[-1].zero_()
    parameters = {
        "epochs": 2,
        "episodes_per_epoch": 16,
        "batch_size": 8,
        "support_sizes": [1, 2],
        "learning_rate": 0.001,
        "task_loss_weight": 0.5,
        "freeze_released_semantic_embedding": True,
    }
    with ProcessPoolExecutor(
        max_workers=2, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        futures = [
            pool.submit(
                train_run,
                matrix,
                embedding.numpy(),
                "test",
                seed,
                parameters,
                tmp_path / "parallel" / str(seed),
                {},
            )
            for seed in (0, 1)
        ]
        for future in futures:
            assert not future.result()["reused"]
    for seed in (0, 1):
        serial_dir = tmp_path / "serial" / str(seed)
        parallel_dir = tmp_path / "parallel" / str(seed)
        train_run(matrix, embedding, "test", seed, parameters, serial_dir, {})
        serial = torch.load(serial_dir / "checkpoint.pt", weights_only=False)
        parallel = torch.load(parallel_dir / "checkpoint.pt", weights_only=False)
        for key in serial["state_dict"]:
            assert torch.equal(serial["state_dict"][key], parallel["state_dict"][key])
        pd.testing.assert_frame_equal(
            pd.read_csv(serial_dir / "training_metrics.csv"),
            pd.read_csv(parallel_dir / "training_metrics.csv"),
        )
