import numpy as np

from iscci_validation.simulations import _lca_choice_probabilities


def test_lca_is_seed_reproducible_and_normalized() -> None:
    drift = np.asarray(
        [[0.8, 0.1, 0.1], [0.7, 0.2, 0.1], [0.6, 0.2, 0.2]], dtype=float
    )
    left = _lca_choice_probabilities(
        drift, np.random.default_rng(5), simulations=8, steps=30, burn_in=5
    )
    right = _lca_choice_probabilities(
        drift, np.random.default_rng(5), simulations=8, steps=30, burn_in=5
    )
    assert np.allclose(left, right)
    assert np.isclose(left.sum(), 1.0)
