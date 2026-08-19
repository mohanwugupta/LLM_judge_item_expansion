import numpy as np

from iscci_validation.evaluation import compare_evaluations


def test_identical_evaluations_reach_identity() -> None:
    rng = np.random.default_rng(4)
    values = {
        "context_rdm_ranks": rng.normal(size=20).astype(np.float32),
        "context_dependent_rdm_ranks": rng.normal(size=(3, 20)).astype(np.float32),
        "membership_logits": rng.normal(size=(4, 5)).astype(np.float32),
        "membership_logit_ranks": rng.permutation(20).astype(np.float32),
    }
    metrics = compare_evaluations(values, values)
    assert np.isclose(metrics["context_rdm_spearman"], 1.0)
    assert np.isclose(metrics["context_dependent_rdm_spearman_median"], 1.0)
    assert np.isclose(metrics["membership_logit_spearman"], 1.0)
    assert metrics["binary_membership_agreement"] == 1.0
    assert metrics["membership_probability_mae"] == 0.0
