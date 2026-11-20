"""Tests of the data pipeline (T5).

The synthetic tests always run. The tests marked `cache` need the real
cache built by `python -m src.prepare_data` and are skipped without it.

Run from the repository root:  python -m pytest -q tests
"""
import numpy as np
import pytest

from src import config
from src.dataset import lead_stats, make_loaders, split_indices

# Class counts from the table of the PTB-XL 1.0.3 page on PhysioNet
# (https://physionet.org/content/ptb-xl/1.0.3/), checked by OF and MM.
PHYSIONET_COUNTS = {"NORM": 9514, "MI": 5469, "STTC": 5235,
                    "CD": 4898, "HYP": 2649}


# ---------------------------------------------------------------- synthetic
def toy_cache(n_per_fold: int = 4, seed: int = 0) -> dict:
    """Small fake cache: lead k has mean k and standard deviation k + 1
    in the training folds, and very different values elsewhere."""
    rng = np.random.default_rng(seed)
    fold = np.repeat(np.arange(1, 11), n_per_fold)
    n = len(fold)
    X = rng.standard_normal((n, 12, 50)).astype(np.float32)
    X = X * (np.arange(12) + 1)[None, :, None] + np.arange(12)[None, :, None]
    X[fold >= 9] += 1000.0          # validation/test values must not leak
    y = (rng.random((n, 5)) > 0.5).astype(np.float32)
    y[0] = 0.0                      # one training record without superclass
    return {"X": X, "y": y, "fold": fold, "patient_id": np.arange(n),
            "has_superclass": y.sum(axis=1) > 0}


def test_split_follows_official_folds():
    cache = toy_cache()
    idx = split_indices(cache, include_no_superclass=True)
    assert set(cache["fold"][idx["train"]]) == set(config.TRAIN_FOLDS)
    assert set(cache["fold"][idx["val"]]) == {config.VAL_FOLD}
    assert set(cache["fold"][idx["test"]]) == {config.TEST_FOLD}


def test_records_without_superclass_left_out_by_default():
    cache = toy_cache()
    assert 0 not in split_indices(cache)["train"]
    assert 0 in split_indices(cache, include_no_superclass=True)["train"]


def test_stats_use_training_records_only():
    cache = toy_cache(n_per_fold=200)
    idx = split_indices(cache)
    mean, std = lead_stats(cache["X"], idx["train"])
    # The +1000 shift of folds 9-10 would show up here if they leaked in.
    np.testing.assert_allclose(mean, np.arange(12), atol=0.1)
    np.testing.assert_allclose(std, np.arange(12) + 1, rtol=0.05)


def test_normalised_training_batch_is_standardised():
    cache = toy_cache(n_per_fold=50)
    loaders, _ = make_loaders(cache, batch_size=10_000)
    x, _ = next(iter(loaders["train"]))
    np.testing.assert_allclose(x.mean(dim=(0, 2)).numpy(), 0, atol=1e-4)
    np.testing.assert_allclose(x.std(dim=(0, 2), unbiased=False).numpy(),
                               1, atol=1e-4)


# --------------------------------------------------------------- real cache
@pytest.fixture(scope="module")
def cache():
    if not config.CACHE_PATH.exists():
        pytest.skip("cache not built: run python -m src.prepare_data")
    from src.data import load_cache
    return load_cache()


@pytest.mark.cache
def test_cache_shapes(cache):
    n = len(cache["y"])
    assert cache["X"].shape == (n, len(config.LEADS), config.N_SAMPLES)
    assert cache["y"].shape == (n, len(config.CLASSES))
    assert list(cache["classes"]) == config.CLASSES
    assert list(cache["leads"]) == config.LEADS


@pytest.mark.cache
def test_class_counts_match_physionet(cache):
    counts = dict(zip(config.CLASSES, cache["y"].sum(axis=0).astype(int)))
    assert counts == PHYSIONET_COUNTS


@pytest.mark.cache
def test_no_patient_in_two_splits(cache):
    idx = split_indices(cache, include_no_superclass=True)
    patients = {s: set(cache["patient_id"][r]) for s, r in idx.items()}
    assert not patients["train"] & patients["val"]
    assert not patients["train"] & patients["test"]
    assert not patients["val"] & patients["test"]


@pytest.mark.cache
def test_batch_shapes(cache):
    loaders, info = make_loaders(cache, batch_size=32)
    for split in ["train", "val", "test"]:
        x, y = next(iter(loaders[split]))
        assert x.shape == (32, 12, 1000) and y.shape == (32, 5)
        assert str(x.dtype) == str(y.dtype) == "torch.float32"
    assert sum(info["n_records"].values()) == int(cache["has_superclass"].sum())
