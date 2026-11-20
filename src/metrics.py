"""Evaluation metrics and bootstrap confidence intervals (DECISIONS D4, D14).

Everything here works on NumPy arrays that were already taken out of the
model, so it can be tested without PyTorch:

    y_true  : (n_records, n_classes) multi-hot labels, 0/1
    y_score : (n_records, n_classes) probabilities, sigmoid(logits)
    groups  : (n_records,) patient id of each record

Main metric: macro AUROC = plain mean of the AUROC of each class.
Secondary: AUROC per class; F1 per class and macro F1 at thresholds chosen
on validation. Uncertainty: bootstrap over patients.
"""
import math

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

# Candidate thresholds for F1: 0.01, 0.02, ..., 0.99 (D14).
THRESHOLD_GRID = np.round(np.arange(1, 100) / 100, 2)


# ---------------------------------------------------------------------------
# 1. Point metrics
# ---------------------------------------------------------------------------
def sigmoid(logits: np.ndarray) -> np.ndarray:
    """Logits -> probabilities. The models have no sigmoid in their last
    layer (BCEWithLogitsLoss contains it), so it is applied here."""
    return 1.0 / (1.0 + np.exp(-logits))


def auroc_per_class(y_true: np.ndarray, y_score: np.ndarray) -> np.ndarray:
    """AUROC of each class (column).

    AUROC = probability that a random positive record gets a higher score
    than a random negative one; 0.5 is chance, 1 is a perfect ranking.
    It needs no threshold. It is undefined (NaN) when a class has only
    positives or only negatives in the evaluated records.
    """
    out = np.full(y_true.shape[1], np.nan)
    for c in range(y_true.shape[1]):
        if 0 < y_true[:, c].sum() < len(y_true):
            out[c] = roc_auc_score(y_true[:, c], y_score[:, c])
    return out


def macro_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Plain mean of the per-class AUROC (NaN if any class is undefined)."""
    return float(np.mean(auroc_per_class(y_true, y_score)))


def best_thresholds(y_true: np.ndarray, y_score: np.ndarray,
                    grid: np.ndarray = THRESHOLD_GRID) -> np.ndarray:
    """For each class, the grid threshold that maximises F1.

    Called on VALIDATION predictions only; the thresholds found are then
    applied unchanged to the test predictions. A fixed 0.5 for every class
    would penalise rare classes such as HYP, whose probabilities tend to be
    lower. On ties np.argmax keeps the first, i.e. the lowest, threshold.
    """
    thresholds = np.empty(y_true.shape[1])
    for c in range(y_true.shape[1]):
        f1s = [f1_score(y_true[:, c], y_score[:, c] >= t, zero_division=0)
               for t in grid]
        thresholds[c] = grid[int(np.argmax(f1s))]
    return thresholds


def f1_per_class(y_true: np.ndarray, y_score: np.ndarray,
                 thresholds: np.ndarray) -> np.ndarray:
    """F1 of each class after thresholding its probability."""
    y_pred = y_score >= thresholds[None, :]
    return np.array([f1_score(y_true[:, c], y_pred[:, c], zero_division=0)
                     for c in range(y_true.shape[1])])


def classification_metrics(y_true: np.ndarray, y_score: np.ndarray,
                           thresholds: np.ndarray | None = None) -> dict:
    """Macro and per-class AUROC; F1 too when thresholds are given."""
    auroc = auroc_per_class(y_true, y_score)
    out = {"macro_auroc": float(np.mean(auroc)), "auroc": auroc}
    if thresholds is not None:
        f1 = f1_per_class(y_true, y_score, thresholds)
        out["macro_f1"] = float(np.mean(f1))
        out["f1"] = f1
    return out


def perplexity(mean_token_ce: float) -> float:
    """Perplexity of a next-token model = exp(mean cross-entropy per token).

    The mean must be taken over all tokens of the split, not as a mean of
    batch means. Uniform guessing over V tokens gives exactly V.
    """
    return float(math.exp(mean_token_ce))


# ---------------------------------------------------------------------------
# 2. Bootstrap over patients
# ---------------------------------------------------------------------------
def patient_bootstrap(groups: np.ndarray, n_boot: int = 1000,
                      seed: int = 42):
    """Yield n_boot arrays of record indices, resampling PATIENTS.

    In each resample, as many patients as there are are drawn with
    replacement; every record of a drawn patient enters as many times as
    the patient was drawn. Records of the same patient are correlated, so
    resampling records one by one would give intervals that are too narrow.

    The draws depend only on `groups` and `seed`: two models evaluated on
    the same split get exactly the same resamples (paired comparison).
    """
    patients, inverse = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(seed)
    records = np.arange(len(groups))
    for _ in range(n_boot):
        drawn = rng.integers(0, len(patients), size=len(patients))
        times = np.bincount(drawn, minlength=len(patients))   # per patient
        yield np.repeat(records, times[inverse])               # per record


def _valid(y_true: np.ndarray) -> bool:
    """Every class has at least one positive and one negative record."""
    pos = y_true.sum(axis=0)
    return bool(np.all((pos > 0) & (pos < len(y_true))))


def _interval(values: list, alpha: float) -> tuple[float, float]:
    lo, hi = np.percentile(np.asarray(values), [100 * alpha / 2,
                                                100 * (1 - alpha / 2)], axis=0)
    return lo, hi


def bootstrap_ci(y_true: np.ndarray, y_score: np.ndarray, groups: np.ndarray,
                 thresholds: np.ndarray | None = None, n_boot: int = 1000,
                 seed: int = 42, alpha: float = 0.05) -> dict:
    """Percentile bootstrap intervals of the classification metrics.

    Only the predictions are resampled; the model and the thresholds stay
    fixed. Resamples where a class has no positive or no negative record
    are skipped (AUROC undefined) and counted in `n_skipped`.
    Returns, for each metric, (lower, upper) bounds; per-class metrics give
    arrays of bounds.
    """
    draws = {"macro_auroc": [], "auroc": [], "macro_f1": [], "f1": []}
    skipped = 0
    for idx in patient_bootstrap(groups, n_boot, seed):
        yt, ys = y_true[idx], y_score[idx]
        if not _valid(yt):
            skipped += 1
            continue
        m = classification_metrics(yt, ys, thresholds)
        for key in draws:
            if key in m:
                draws[key].append(m[key])
    out = {"n_boot": n_boot, "n_skipped": skipped, "alpha": alpha,
           "seed": seed}
    for key, values in draws.items():
        if values:
            out[key] = _interval(values, alpha)
    return out


def paired_bootstrap_diff(y_true: np.ndarray, score_a: np.ndarray,
                          score_b: np.ndarray, groups: np.ndarray,
                          n_boot: int = 1000, seed: int = 42,
                          alpha: float = 0.05) -> dict:
    """Interval of macro AUROC(A) - macro AUROC(B) on the same resamples.

    Both models must have been evaluated on the same records in the same
    order. If the interval contains 0, we do not say that one model is
    better than the other (weekly summary of 16/10).
    """
    diffs, skipped = [], 0
    for idx in patient_bootstrap(groups, n_boot, seed):
        yt = y_true[idx]
        if not _valid(yt):
            skipped += 1
            continue
        diffs.append(macro_auroc(yt, score_a[idx]) - macro_auroc(yt, score_b[idx]))
    lo, hi = _interval(diffs, alpha)
    return {"diff": macro_auroc(y_true, score_a) - macro_auroc(y_true, score_b),
            "lower": float(lo), "upper": float(hi),
            "n_boot": n_boot, "n_skipped": skipped, "seed": seed}
