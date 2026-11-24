def superclass_labels(df: pd.DataFrame,
                      raw_dir: Path = config.RAW_DIR) -> np.ndarray:
    """Multi-hot matrix (n_records, 5) of the diagnostic superclasses.

    A record gets superclass c if ANY of its SCP codes is a diagnostic
    statement of class c, whatever its likelihood value. This is the rule
    of example_physionet.py (shipped with PTB-XL) and reproduces the class
    counts of the PhysioNet page (DECISIONS D2, 2026-11-20).
    """
    statements = pd.read_csv(raw_dir / "scp_statements.csv", index_col=0)
    diagnostic = statements[statements["diagnostic"] == 1]["diagnostic_class"]

    y = np.zeros((len(df), len(config.CLASSES)), dtype=np.float32)
    for i, codes in enumerate(df["scp_codes"]):
        for code in codes:                       # the likelihood is ignored
            if code in diagnostic.index:
                y[i, config.CLASSES.index(diagnostic[code])] = 1.0
    return y
