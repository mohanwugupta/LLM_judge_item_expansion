import numpy as np
import pandas as pd
import torch

from iscci_validation.dataio import CICODataset
from iscci_validation.modeling import initialize_model


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
