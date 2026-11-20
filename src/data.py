"""PTB-XL: integrity check, superclass labels and the single-file cache.

The Dataset and DataLoader classes are added in the next step of T5.
"""
import ast
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from src import config


# ---------------------------------------------------------------------------
# 1. Integrity check
# ---------------------------------------------------------------------------
def _sha256(path: Path) -> str:
    """SHA-256 of a file, read in 1 MB blocks so memory stays small."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify_raw(raw_dir: Path = config.RAW_DIR) -> list[str]:
    """Compare our files with the checksums published with the dataset.

    SHA256SUMS.txt lists every file of PTB-XL as "<hash> <path>". We check
    the top-level files we use and everything under records100/; the
    500 Hz folder (records500/) was deleted on purpose and is skipped.
    Returns the list of problems; an empty list means the data are intact.
    """
    needed_top = {"ptbxl_database.csv", "scp_statements.csv"}
    problems = []
    with open(raw_dir / "SHA256SUMS.txt", encoding="utf-8") as f:
        for line in f:
            expected, rel = line.strip().split(maxsplit=1)
            if not (rel in needed_top or rel.startswith("records100/")):
                continue
            path = raw_dir / rel
            if not path.exists():
                problems.append(f"missing: {rel}")
            elif _sha256(path) != expected:
                problems.append(f"checksum mismatch: {rel}")
    return problems


# ---------------------------------------------------------------------------
# 2. Metadata and labels
# ---------------------------------------------------------------------------
def load_metadata(raw_dir: Path = config.RAW_DIR) -> pd.DataFrame:
    """One row per record, indexed by ecg_id, with scp_codes as a dict."""
    df = pd.read_csv(raw_dir / "ptbxl_database.csv", index_col="ecg_id")
    # scp_codes is stored as text, e.g. "{'NORM': 100.0, 'SR': 0.0}".
    # literal_eval turns it into a real dict without running any code.
    df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)
    df["patient_id"] = df["patient_id"].astype(int)
    return df


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


# ---------------------------------------------------------------------------
# 3. Signals and cache
# ---------------------------------------------------------------------------
def load_signal(raw_dir: Path, filename_lr: str) -> np.ndarray:
    """Read one 100 Hz record and return it as (12 leads, 1000 samples)."""
    import wfdb  # imported here so the rest of the module works without it

    signal, fields = wfdb.rdsamp(str(raw_dir / filename_lr))
    # Stop at once if a record is not what the whole pipeline assumes.
    if fields["fs"] != config.SAMPLING_RATE:
        raise ValueError(f"{filename_lr}: fs = {fields['fs']}")
    if fields["sig_name"] != config.LEADS:
        raise ValueError(f"{filename_lr}: leads = {fields['sig_name']}")
    if signal.shape != (config.N_SAMPLES, len(config.LEADS)):
        raise ValueError(f"{filename_lr}: shape = {signal.shape}")
    # wfdb returns (time, lead); PyTorch's Conv1d expects (channel, time).
    return signal.T.astype(np.float32)


def build_cache(raw_dir: Path = config.RAW_DIR,
                cache_path: Path = config.CACHE_PATH,
                limit: int | None = None) -> dict:
    """Read every record once and save signals, labels and folds together.

    All records are kept, including those without a diagnostic superclass;
    they are flagged in `has_superclass` and filtered later, in the
    DataLoader (DECISIONS O3, provisional).
    """
    df = load_metadata(raw_dir)
    if limit is not None:                         # small run for smoke tests
        df = df.iloc[:limit]
    y = superclass_labels(df, raw_dir)

    # Pre-allocate the full array: about 1 GB in float32 for all records.
    X = np.empty((len(df), len(config.LEADS), config.N_SAMPLES),
                 dtype=np.float32)
    for i, filename in enumerate(df["filename_lr"]):
        X[i] = load_signal(raw_dir, filename)
        if (i + 1) % 2000 == 0:
            print(f"  read {i + 1}/{len(df)} records")

    arrays = {
        "X": X,
        "y": y,
        "ecg_id": df.index.to_numpy(dtype=np.int64),
        "patient_id": df["patient_id"].to_numpy(dtype=np.int64),
        "fold": df["strat_fold"].to_numpy(dtype=np.int64),
        "has_superclass": y.sum(axis=1) > 0,
        "classes": np.array(config.CLASSES),
        "leads": np.array(config.LEADS),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temporary file and rename it, so an interrupted run never
    # leaves a half-written cache that looks valid.
    tmp = cache_path.with_suffix(".tmp.npz")
    np.savez(tmp, **arrays)
    tmp.replace(cache_path)
    return arrays


def load_cache(cache_path: Path = config.CACHE_PATH) -> dict:
    """Load the cache built by build_cache into a dict of arrays."""
    with np.load(cache_path) as data:
        return {key: data[key] for key in data.files}
