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
