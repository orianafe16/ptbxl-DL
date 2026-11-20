"""Tests of the evaluation module (T7) on toy cases with known answers.

Run from the repository root:  python -m pytest -q tests
"""
import numpy as np
import pytest

from src import metrics


def toy(n: int = 200, n_classes: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    y = (rng.random((n, n_classes)) > 0.6).astype(np.float32)
    y[0], y[1] = 1, 0                      # every class has both values
    return y


# ------------------------------------------------------------------- AUROC
def test_auroc_perfect_reversed_and_macro():
    y = toy()
    perfect = y + 0.01 * np.arange(len(y))[:, None] / len(y)
    assert np.allclose(metrics.auroc_per_class(y, perfect), 1.0)
    assert np.allclose(metrics.auroc_per_class(y, -perfect), 0.0)
    mixed = np.column_stack([perfect[:, 0], -perfect[:, 1], perfect[:, 2]])
    assert metrics.macro_auroc(y, mixed) == pytest.approx(2 / 3)


def test_auroc_known_value():
    # Positives score 0.9 and 0.4, negatives 0.5 and 0.1: of the 4
    # positive-negative pairs, 3 are ordered correctly -> 0.75.
    y = np.array([[1], [1], [0], [0]])
    s = np.array([[0.9], [0.4], [0.5], [0.1]])
    assert metrics.auroc_per_class(y, s)[0] == pytest.approx(0.75)


def test_auroc_undefined_without_negatives():
    y = np.ones((10, 1))
    assert np.isnan(metrics.auroc_per_class(y, np.random.rand(10, 1))[0])


def test_sigmoid():
    assert metrics.sigmoid(np.array([0.0]))[0] == pytest.approx(0.5)


# --------------------------------------------------------------------- F1
def test_threshold_search_finds_a_separating_threshold():
    y = np.array([[0], [0], [0], [1], [1]])
    s = np.array([[0.10], [0.20], [0.30], [0.70], [0.80]])
    t = metrics.best_thresholds(y, s)
    assert 0.30 < t[0] <= 0.70
    assert metrics.f1_per_class(y, s, t)[0] == pytest.approx(1.0)
    # Ties are broken towards the lowest threshold: 0.31.
    assert t[0] == pytest.approx(0.31)


def test_f1_known_value():
    # Threshold 0.5: predictions 1,1,0,0 against labels 1,0,1,0
    # -> TP 1, FP 1, FN 1 -> precision 0.5, recall 0.5, F1 0.5.
    y = np.array([[1], [0], [1], [0]])
    s = np.array([[0.9], [0.8], [0.2], [0.1]])
    assert metrics.f1_per_class(y, s, np.array([0.5]))[0] == pytest.approx(0.5)


def test_perplexity_of_uniform_guess_is_vocabulary_size():
    assert metrics.perplexity(np.log(256)) == pytest.approx(256)


# --------------------------------------------------------------- bootstrap
def test_bootstrap_keeps_patients_together():
    groups = np.array([7, 7, 7, 3, 3, 9])       # 3 patients, 6 records
    for idx in metrics.patient_bootstrap(groups, n_boot=50, seed=1):
        counts = np.bincount(idx, minlength=len(groups))
        for patient in np.unique(groups):
            rows = groups == patient
            # all records of a patient enter the same number of times
            assert len(set(counts[rows])) == 1
        # as many patient draws as patients
        n_draws = sum(counts[groups == p][0] for p in np.unique(groups))
        assert n_draws == 3


def test_bootstrap_is_reproducible_and_paired():
    groups = np.repeat(np.arange(30), 2)
    a = list(metrics.patient_bootstrap(groups, n_boot=5, seed=42))
    b = list(metrics.patient_bootstrap(groups, n_boot=5, seed=42))
    c = list(metrics.patient_bootstrap(groups, n_boot=5, seed=43))
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert not all(np.array_equal(x, y) for x, y in zip(a, c))


def test_bootstrap_interval_of_a_perfect_model():
    y = toy()
    groups = np.arange(len(y))
    ci = metrics.bootstrap_ci(y, y.astype(float), groups,
                              thresholds=np.full(3, 0.5), n_boot=100)
    assert ci["macro_auroc"][0] == pytest.approx(1.0)
    assert ci["macro_auroc"][1] == pytest.approx(1.0)
    assert ci["macro_f1"][0] == pytest.approx(1.0)


def test_bootstrap_interval_contains_estimate():
    rng = np.random.default_rng(0)
    y = toy(n=400)
    score = np.clip(y * 0.3 + rng.random(y.shape) * 0.7, 0, 1)
    groups = np.repeat(np.arange(200), 2)
    est = metrics.macro_auroc(y, score)
    ci = metrics.bootstrap_ci(y, score, groups, n_boot=200)
    assert ci["macro_auroc"][0] < est < ci["macro_auroc"][1]
    assert ci["auroc"][0].shape == (3,)
    assert ci["n_skipped"] == 0


def test_bootstrap_skips_resamples_without_positives():
    # One single positive record: many resamples lose it.
    y = np.zeros((20, 1))
    y[0] = 1
    ci = metrics.bootstrap_ci(y, np.random.rand(20, 1), np.arange(20),
                              n_boot=100)
    assert ci["n_skipped"] > 0


def test_paired_difference_of_identical_models_is_zero():
    y = toy()
    score = np.random.default_rng(1).random(y.shape)
    d = metrics.paired_bootstrap_diff(y, score, score, np.arange(len(y)),
                                      n_boot=50)
    assert d["diff"] == 0 and d["lower"] == 0 and d["upper"] == 0
