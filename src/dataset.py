"""PyTorch Dataset and DataLoaders for the cached PTB-XL signals.

Split (DECISIONS D3): train = folds 1-8, validation = fold 9, test = fold 10.
Normalisation (DECISIONS D3, 2026-11-20): per-lead z-score whose mean and
standard deviation are computed on the training records only.
Records without a diagnostic superclass are left out unless asked for
(DECISIONS O3, provisional).
"""
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src import config


def split_indices(cache: dict, include_no_superclass: bool = False) -> dict:
    """Row indices of the cache for each split, as sorted integer arrays."""
    keep = np.ones(len(cache["fold"]), dtype=bool)
    if not include_no_superclass:
        keep &= cache["has_superclass"]
    fold = cache["fold"]
    return {
        "train": np.flatnonzero(keep & np.isin(fold, config.TRAIN_FOLDS)),
        "val": np.flatnonzero(keep & (fold == config.VAL_FOLD)),
        "test": np.flatnonzero(keep & (fold == config.TEST_FOLD)),
    }


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


class PTBXLDataset(Dataset):
    """One item = (signal of shape (12, 1000), labels of shape (5,)).

    The dataset keeps a reference to the full cache and a list of row
    indices, so building the three splits does not copy the 1 GB array.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, idx: np.ndarray,
                 mean: np.ndarray | None = None,
                 std: np.ndarray | None = None):
        self.X, self.y, self.idx = X, y, idx
        # Shapes (12, 1) so that they broadcast over the 1000 time samples.
        self.mean = None if mean is None else mean[:, None]
        self.std = None if std is None else std[:, None]

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, i: int):
        row = self.idx[i]
        x = self.X[row]                                   # (12, 1000)
        if self.mean is not None:
            x = (x - self.mean) / self.std
        # Labels as float: BCEWithLogitsLoss needs float targets (D2).
        return (torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)),
                torch.from_numpy(self.y[row].astype(np.float32)))


def make_loaders(cache: dict, batch_size: int = 64, normalize: bool = True,
                 include_no_superclass: bool = False,
                 seed: int = config.SEED) -> tuple[dict, dict]:
    """Build the train/val/test DataLoaders.

    Returns (loaders, info). `info` holds the split sizes and the
    normalisation statistics, to be written into each run's JSON log.
    """
    X, y = cache["X"], cache["y"]
    idx = split_indices(cache, include_no_superclass)
    mean, std = lead_stats(X, idx["train"]) if normalize else (None, None)

    # A seeded generator makes the shuffling order reproducible.
    generator = torch.Generator().manual_seed(seed)
    loaders = {}
    for split, rows in idx.items():
        loaders[split] = DataLoader(
            PTBXLDataset(X, y, rows, mean, std),
            batch_size=batch_size,
            shuffle=(split == "train"),        # only the training set
            generator=generator if split == "train" else None,
            num_workers=0,   # Windows: extra workers would copy the cache
            pin_memory=torch.cuda.is_available(),
        )
    info = {
        "n_records": {s: int(len(r)) for s, r in idx.items()},
        "normalize": normalize,
        "include_no_superclass": include_no_superclass,
        "lead_mean": None if mean is None else mean.tolist(),
        "lead_std": None if std is None else std.tolist(),
    }
    return loaders, info
