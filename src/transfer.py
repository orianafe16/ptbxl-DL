"""Transfer learning from the ECG-GPT (T13, DECISIONS D20, D7, D13, D19).

The ECG-GPT of chat DL 13 was trained without labels to predict the next
sample of lead II (self-supervised pre-training, the "pretext task"). Here
it is reused for the 5-superclass task (the "downstream task") in three
ways (D7):

    scratch   the same architecture with random weights, all trainable
    probe     the pre-trained network frozen; only a new linear head trains
              (linear probe = "frozen feature extractor")
    finetune  the pre-trained network and a new head, all trainable, with a
              smaller learning rate (fine-tuning)

    tokens (batch, 1000)  lead II quantised with the generator's tokenizer
      -> backbone.record_features()  (batch, d): the generator without its
                                     next-token head; mean of its outputs
                                     over the 4 windows of 250 samples (D19)
      -> Dropout(0.5) -> Linear(d, 5) -> logits (batch, 5), no sigmoid (D2)

This module holds the data (tokens of the labelled records, the 10 % label
subset), the models and the feature extraction. The experiments, summary,
tables and figures are in src/transfer_experiments.py.
"""
import math

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from src import config
from src import generative as gx
from src.dataset import split_indices
from src.models.generative import build_generator

LEAD = "II"
FAST_SIZES = {"train": 512, "val": 256, "test": 256}   # --fast, as D14
FEATURE_BATCH = 64                                     # records per batch


# ---------------------------------------------------------------------------
# 1. Labelled records and the label fraction
# ---------------------------------------------------------------------------
def label_subset(rows: np.ndarray, patient_id: np.ndarray, fraction: float,
                 seed: int) -> np.ndarray:
    """Training rows kept when only `fraction` of the labels is used (D20).

    Drawn by PATIENT, as the bootstrap (D14): the training patients are put
    in a random order (generator seeded with `seed`) and taken one after
    another, with all their records, until at least ceil(fraction * n)
    records are reached. Records of one patient are correlated, so drawing
    records one by one would put part of a patient in the subset and hide
    how few independent people it really contains.
    rows: cache rows of the training split; patient_id: of every cache row.
    Returns the kept rows, sorted.
    """
    rows = np.asarray(rows)
    if fraction >= 1.0:
        return rows.copy()
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must be in (0, 1]")
    target = int(math.ceil(fraction * len(rows)))
    patients, counts = np.unique(patient_id[rows], return_counts=True)
    order = np.random.default_rng(seed).permutation(len(patients))
    reached = np.cumsum(counts[order])
    n_take = int(np.searchsorted(reached, target) + 1)
    chosen = patients[order[:n_take]]
    return np.sort(rows[np.isin(patient_id[rows], chosen)])


class TransferData:
    """Lead II, labels and ids of the labelled records of the three splits,
    built once per command. Split rows come from src.dataset.split_indices,
    the function every classifier uses, so the validation and test records
    are in the same order as in every other block: predictions can be paired
    record by record (D14). Records without a superclass are left out, as in
    every classifier (D12). --fast: 512 / 256 / 256 records drawn with
    seed 42 (D14)."""

    def __init__(self, cache: dict, fast: bool = False,
                 seed: int = config.SEED):
        idx = split_indices(cache)
        if fast:
            rng = np.random.default_rng(seed)
            idx = {s: np.sort(rng.choice(r, size=min(FAST_SIZES[s], len(r)),
                                         replace=False))
                   for s, r in idx.items()}
        self.fast = fast
        self.idx = idx
        self.patient_id_all = cache["patient_id"]
        lead = config.LEADS.index(LEAD)
        self.mv = {s: np.ascontiguousarray(cache["X"][r, lead, :],
                                           dtype=np.float32)
                   for s, r in idx.items()}
        self.y = {s: cache["y"][r].astype(np.float32) for s, r in idx.items()}
        self.patient_id = {s: cache["patient_id"][r] for s, r in idx.items()}
        self.ecg_id = {s: cache["ecg_id"][r] for s, r in idx.items()}
        self._tokens: dict = {}

    def tokens(self, tokenizer: gx.Tokenizer, split: str) -> np.ndarray:
        """Tokens of one split with a generator's tokenizer, int16."""
        key = (tokenizer.lo_uv, tokenizer.delta_uv, tokenizer.vocab, split)
        if key not in self._tokens:
            self._tokens[key] = tokenizer.encode(self.mv[split]).astype(np.int16)
        return self._tokens[key]

    def train_positions(self, fraction: float, seed: int) -> np.ndarray:
        """Positions (in this object's training arrays) of the training
        records kept with `fraction` of the labels (label_subset)."""
        rows = label_subset(self.idx["train"], self.patient_id_all,
                            fraction, seed)
        return np.flatnonzero(np.isin(self.idx["train"], rows))

    def describe(self, positions: np.ndarray, fraction: float) -> dict:
        """Facts about the labelled data of one run, for its log."""
        y = self.y["train"][positions]
        return {"lead": LEAD,
                "n_records": {"train": int(len(positions)),
                              "val": int(len(self.idx["val"])),
                              "test": int(len(self.idx["test"]))},
                "n_records_train_all": int(len(self.idx["train"])),
                "label_fraction": float(fraction),
                "n_patients_train": int(len(np.unique(
                    self.patient_id["train"][positions]))),
                "train_class_counts": dict(zip(config.CLASSES,
                                               y.sum(axis=0).astype(int).tolist())),
                "include_no_superclass": False,
                "fast_subset": FAST_SIZES if self.fast else None}


class TokenDataset(Dataset):
    """One item = (tokens of lead II (1000,) int64, labels (5,) float)."""

    def __init__(self, tokens: np.ndarray, y: np.ndarray):
        self.tokens, self.y = tokens, y

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, i: int):
        return (torch.from_numpy(self.tokens[i].astype(np.int64)),
                torch.from_numpy(self.y[i]))


class FeatureDataset(Dataset):
    """One item = (record vector of the frozen backbone (d,), labels (5,))."""

    def __init__(self, features: np.ndarray, y: np.ndarray):
        self.features, self.y = features, y

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, i: int):
        return (torch.from_numpy(self.features[i]),
                torch.from_numpy(self.y[i]))


# ---------------------------------------------------------------------------
# 2. Models
# ---------------------------------------------------------------------------
def feature_dim(backbone: nn.Module) -> int:
    """Size of the record vector: d_model (GPT) or hidden size (LSTM)."""
    return int(getattr(backbone, "d_model", None)
               or getattr(backbone, "hidden_size"))


def load_backbone(model_args: dict, checkpoint=None) -> nn.Module:
    """A generator of D19 without its next-token head.

    With `checkpoint` (path of a generator run's state_dict) the pre-trained
    weights are loaded first, then the head is removed; without it the
    weights are random (the "scratch" strategy, same architecture). The
    head predicts the next token, which the 5-superclass task does not
    need: removing it keeps it out of the parameter count, the optimiser
    and the checkpoint."""
    generator = build_generator(model_args)
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        generator.load_state_dict(state)
    del generator.head
    return generator


class TransferClassifier(nn.Module):
    """Generator backbone + record vector + new classification head.

    freeze=True is the linear probe: the backbone's parameters do not
    require gradients (the optimiser of src/train.py only receives
    trainable parameters, D14) and the backbone stays in eval mode even
    when the loop calls model.train(): its dropout is off, so a record
    always gives the same vector, as a fixed feature extractor should."""

    def __init__(self, backbone: nn.Module, n_classes: int = len(config.CLASSES),
                 head_dropout: float = 0.5, freeze: bool = False):
        super().__init__()
        self.backbone = backbone
        self.freeze = bool(freeze)
        if self.freeze:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
        self.drop = nn.Dropout(head_dropout)
        self.head = nn.Linear(feature_dim(backbone), n_classes)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            self.backbone.eval()
        return self

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.head(self.drop(self.backbone.record_features(tokens)))


class FeatureHead(nn.Module):
    """The head of TransferClassifier alone, trained on record vectors that
    the frozen backbone computed once (the probe, D20). With the same
    weights it gives exactly the output of TransferClassifier(freeze=True),
    because a frozen backbone in eval mode always returns the same vector
    (checked in tests/test_transfer.py)."""

    def __init__(self, d: int, n_classes: int = len(config.CLASSES),
                 head_dropout: float = 0.5):
        super().__init__()
        self.drop = nn.Dropout(head_dropout)
        self.head = nn.Linear(d, n_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(self.drop(features))


@torch.no_grad()
def extract_features(backbone: nn.Module, tokens: np.ndarray,
                     device: torch.device | None = None,
                     batch: int = FEATURE_BATCH) -> np.ndarray:
    """Record vectors of a frozen backbone, (n, d) float32: eval mode (no
    dropout) and no gradients, batch by batch."""
    device = device or torch.device("cpu")
    backbone = backbone.to(device).eval()
    out = []
    for i in range(0, len(tokens), batch):
        x = torch.as_tensor(tokens[i:i + batch].astype(np.int64),
                            device=device)
        out.append(backbone.record_features(x).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, feature_dim(backbone)),
                                                    dtype=np.float32)
