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
