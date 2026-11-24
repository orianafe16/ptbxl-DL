def lead_stats(X: np.ndarray, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and standard deviation of each lead over the records in `idx`.

    Only the training indices must be passed here: using validation or test
    records would leak information about them into the model (data leakage).
    The sums are done in float64, one lead at a time, so that adding about
    17 million float32 values stays exact enough and memory stays small.
    """
    n_leads = X.shape[1]
    mean = np.empty(n_leads, dtype=np.float32)
    std = np.empty(n_leads, dtype=np.float32)
    for lead in range(n_leads):
        values = X[idx, lead, :].astype(np.float64)   # (n_records, n_samples)
        mean[lead] = values.mean()
        std[lead] = values.std()
    return mean, std
