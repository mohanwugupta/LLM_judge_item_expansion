import numpy as np
import pandas as pd
import pytest

import plot_finalized_v4_results as figures
from run_random_feature_control import append_frozen_csv, randomize_features


def test_randomization_preserves_prevalence_ids_shape_and_source():
    source = pd.DataFrame(
        np.random.default_rng(7).integers(0, 2, size=(293, 385)),
        index=[f"object{i}" for i in range(293)],
        columns=[f"feature{i}" for i in range(385)],
        dtype=np.int8,
    )
    original = source.copy(deep=True)
    randomized = randomize_features(source, 20261028)
    pd.testing.assert_frame_equal(source, original)
    assert randomized.shape == source.shape
    assert randomized.index.equals(source.index)
    assert randomized.columns.equals(source.columns)
    assert np.isin(randomized.values, [0, 1]).all()
    np.testing.assert_array_equal(randomized.sum(axis=0), source.sum(axis=0))
    assert (randomized.values != source.values).sum() > source.size * 0.4
    assert not np.array_equal(randomized.sum(axis=1), source.sum(axis=1))
    pd.testing.assert_frame_equal(randomized, randomize_features(source, 20261028))
    assert not randomized.equals(randomize_features(source, 20261029))


def test_randomization_independent_columns_not_common_object_relabeling():
    source = pd.DataFrame(np.tile([0] * 140 + [1] * 153, (10, 1)).T, dtype=np.int8)
    randomized = randomize_features(source, 20261028)
    assert randomized.T.drop_duplicates().shape[0] == 10


def test_randomization_rejects_nonbinary_and_duplicate_ids():
    with pytest.raises(ValueError, match="binary"):
        randomize_features(pd.DataFrame({"f": [0, 0.5]}), 0)
    with pytest.raises(ValueError, match="unique"):
        randomize_features(pd.DataFrame([[0], [1]], index=["same", "same"]), 0)


def test_csv_extension_preserves_original_bytes_and_aligns_columns(tmp_path):
    original = tmp_path / "old.csv"
    original.write_bytes(b"condition,seed,mean\nhuman,0,0.1234567890123456789\n")
    original_bytes = original.read_bytes()
    extra = pd.DataFrame({"mean": [0.7], "condition": ["random_human"], "seed": [0]})
    target = tmp_path / "combined.csv"
    result = append_frozen_csv(original, extra, target)
    assert result["baseline_bytes_preserved"]
    assert original.read_bytes() == original_bytes
    assert target.read_bytes().startswith(original_bytes)
    assert pd.read_csv(target).condition.tolist() == ["human", "random_human"]
    with pytest.raises(ValueError, match="overwrite"):
        append_frozen_csv(original, extra, original)
    with pytest.raises(ValueError, match="Unexpected new columns"):
        append_frozen_csv(original, extra.assign(unexpected=1), target)


def test_optional_control_does_not_change_frozen_condition_orders(tmp_path):
    args = figures.parser().parse_args(
        ["--include-random-control", "--output-dir", str(tmp_path / "figures")]
    )
    package = figures.Package(args)
    assert package.main == figures.MAIN + [figures.RANDOM]
    assert package.all == figures.ALL + [figures.RANDOM]
    default = figures.Package(
        figures.parser().parse_args(["--output-dir", str(tmp_path / "old")])
    )
    assert default.main == figures.MAIN and len(default.all) == 13
    assert len(package.main) == 8 and len(package.all) == 14
    assert len(figures.MAIN) == 7 and len(figures.ALL) == 13


def test_added_control_does_not_change_frozen_finite_pair_subset():
    rows, trained, shared = [], [], []
    for domain in figures.DOMAINS:
        for c in figures.MAIN + [figures.RANDOM]:
            for i in range(5):
                value = np.nan if c == figures.RANDOM and i == 4 else float(i)
                rows.append(
                    dict(
                        condition=c,
                        dataset=domain,
                        item_id=str(i),
                        human=float(i),
                        prediction=value,
                    )
                )
                for seed in range(4):
                    trained.append(
                        dict(
                            condition=c,
                            seed=seed,
                            dataset=domain,
                            item_id=str(i),
                            human=float(i),
                            prediction=float(i),
                        )
                    )
        for i in range(5):
            shared.append(
                dict(
                    condition="shared_encoder",
                    dataset=domain,
                    item_id=str(i),
                    human=float(i),
                    prediction=float(i),
                )
            )
    scores, membership = figures.common_pair_scores(
        pd.DataFrame(rows),
        pd.DataFrame(trained),
        pd.DataFrame(shared),
        conditions=figures.MAIN + [figures.RANDOM],
        finite_conditions=figures.MAIN,
    )
    assert membership.common_finite.all()
    assert scores["items"].eq(5).all()
    assert scores.ci_low.isna().all() and scores.ci_high.isna().all()
