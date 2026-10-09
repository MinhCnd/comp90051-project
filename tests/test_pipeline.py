import numpy as np
import pandas as pd
import pytest
from typing import Generator

from src.pipeline import generate_parameters, stratified_k_fold_split, subsample_to_ratio

TARGET_LABEL = "TARGET"

def _make_labels(n0=90, n1=30) -> np.ndarray:
    return np.array([0] * n0 + [1] * n1)


def test_stratified_k_fold_number_of_folds():
    y = _make_labels()
    folds = stratified_k_fold_split(y, k=5, seed=0)
    assert len(folds) == 5


def test_stratified_k_fold_train_val_disjoint_and_complete():
    y = _make_labels()
    for train_idx, val_idx in stratified_k_fold_split(y, k=5, seed=0):
        assert len(np.intersect1d(train_idx, val_idx)) == 0
        assert len(train_idx) + len(val_idx) == len(y)
        assert set(np.concatenate([train_idx, val_idx])) == set(range(len(y)))


def test_stratified_k_fold_validation_sets_partition_data():
    y = _make_labels()
    folds = stratified_k_fold_split(y, k=5, seed=0)
    all_val = np.concatenate([val for _, val in folds])
    assert len(all_val) == len(y)
    assert len(np.unique(all_val)) == len(y)


def test_stratified_k_fold_preserves_class_ratio():
    y = _make_labels(90, 30)
    for _, val_idx in stratified_k_fold_split(y, k=5, seed=0):
        assert len(val_idx) == 24
        assert y[val_idx].sum() == 6
        assert np.isclose(y[val_idx].mean(), 0.25)


def test_stratified_k_fold_uneven_split():
    y = _make_labels(23, 7)
    folds = stratified_k_fold_split(y, k=3, seed=1)
    pos_counts = [y[val].sum() for _, val in folds]
    assert sum(pos_counts) == 7
    assert max(pos_counts) - min(pos_counts) <= 1


def test_stratified_k_fold_reproducible_with_seed():
    y = _make_labels()
    f1 = stratified_k_fold_split(y, k=5, seed=42)
    f2 = stratified_k_fold_split(y, k=5, seed=42)
    for (t1, v1), (t2, v2) in zip(f1, f2):
        assert np.array_equal(t1, t2)
        assert np.array_equal(v1, v2)


def test_stratified_k_fold_different_seeds_differ():
    y = _make_labels()
    f1 = stratified_k_fold_split(y, k=5, seed=1)
    f2 = stratified_k_fold_split(y, k=5, seed=2)
    assert any(not np.array_equal(np.sort(a[1]), np.sort(b[1])) for a, b in zip(f1, f2))


def test_stratified_k_fold_default_k_is_10():
    y = _make_labels(100, 50)
    assert len(stratified_k_fold_split(y, seed=0)) == 10


# Tests for subsample_to_ratio function
def _make_df(n_pos: int = 10, n_neg: int = 90) -> pd.DataFrame:
    return pd.DataFrame({
        "feature": np.arange(n_pos + n_neg),
        TARGET_LABEL: [1] * n_pos + [0] * n_neg,
    })


def test_subsample_to_ratio_achieves_ratio():
    df = _make_df(10, 90)
    out = subsample_to_ratio(df, TARGET_LABEL, 0.25, seed=0)
    assert (out[TARGET_LABEL] == 1).sum() == 10
    assert (out[TARGET_LABEL] == 0).sum() == 30
    assert (out[TARGET_LABEL] == 1).mean() == pytest.approx(0.25)


def test_subsample_to_ratio_keeps_all_positives():
    df = _make_df(10, 90)
    out = subsample_to_ratio(df, TARGET_LABEL, 0.5, seed=0)
    pos_features = set(df[df[TARGET_LABEL] == 1]["feature"])
    assert pos_features.issubset(set(out["feature"]))


def test_subsample_to_ratio_no_duplicates_and_reset_index():
    df = _make_df(10, 90)
    out = subsample_to_ratio(df, TARGET_LABEL, 0.5, seed=0)
    assert out["feature"].is_unique
    assert list(out.index) == list(range(len(out)))


def test_subsample_to_ratio_reproducible_with_seed():
    df = _make_df(10, 90)
    out1 = subsample_to_ratio(df, TARGET_LABEL, 0.25, seed=42)
    out2 = subsample_to_ratio(df, TARGET_LABEL, 0.25, seed=42)
    # Only the negative selection is seeded before the final shuffle,
    # so compare the selected rows irrespective of order.
    pd.testing.assert_frame_equal(out1, out2)


def test_subsample_to_ratio_does_not_modify_input():
    df = _make_df(10, 90)
    original = df.copy()
    subsample_to_ratio(df, TARGET_LABEL, 0.25, seed=0)
    pd.testing.assert_frame_equal(df, original)


def test_subsample_to_ratio_raises_when_not_enough_negatives():
    df = _make_df(10, 20)
    with pytest.raises(ValueError):
        subsample_to_ratio(df, TARGET_LABEL, 0.1, seed=0)


# ---------------------------------------------------------------------------
# Tests for generate_parameters (run with: pytest src/pipeline.py)
# ---------------------------------------------------------------------------

def test_generate_parameters_single_param():
    result = list(generate_parameters({"a": [1, 2, 3]}))
    assert result == [{"a": 1}, {"a": 2}, {"a": 3}]


def test_generate_parameters_multiple_params_count_and_content():
    grid = {"a": [1, 2], "b": ["x", "y", "z"]}
    result = list(generate_parameters(grid))
    assert len(result) == 6
    assert all(set(p.keys()) == {"a", "b"} for p in result)
    assert {(p["a"], p["b"]) for p in result} == {(a, b) for a in [1, 2] for b in ["x", "y", "z"]}


def test_generate_parameters_unique_combinations():
    result = list(generate_parameters({"a": [1, 2], "b": [3, 4]}))
    assert len({tuple(sorted(p.items())) for p in result}) == len(result)


def test_generate_parameters_deterministic_sorted_key_order():
    grid1 = {"b": [1, 2], "a": [3, 4]}
    grid2 = {"a": [3, 4], "b": [1, 2]}
    assert list(generate_parameters(grid1)) == list(generate_parameters(grid2))
    assert list(generate_parameters(grid1))[0] == {"a": 3, "b": 1}


def test_generate_parameters_empty_grid():
    assert list(generate_parameters({})) == [{}]


def test_generate_parameters_empty_value_list():
    assert list(generate_parameters({"a": [], "b": [1]})) == []