"""Tests of the shared training loop (T8), on synthetic data and the CPU.

Run from the repository root:  python -m pytest -q tests
"""
import json

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src import train
from src.models.trivial import LeadStatsLinear
from src.tasks import MultiLabelTask, NextTokenTask

CPU = torch.device("cpu")
CLASSES = ["A", "B", "C"]


def classification_loaders(n: int = 32, batch_size: int = 8, seed: int = 0):
    """Random signals (n, 12, 50) and random multi-hot labels (n, 3); the
    same records serve as train, val and test, so the model can memorise
    them. Every record is its own patient."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 12, 50, generator=g)
    y = (torch.rand(n, 3, generator=g) > 0.5).float()
    y[0], y[1] = 1, 0                       # both values in every class
    ds = TensorDataset(x, y)
    loaders = {"train": DataLoader(ds, batch_size=batch_size, shuffle=True,
                                   generator=torch.Generator().manual_seed(seed)),
               "val": DataLoader(ds, batch_size=batch_size),
               "test": DataLoader(ds, batch_size=batch_size)}
    groups = {s: np.arange(n) for s in loaders}
    return loaders, groups


class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(12 * 50, 64),
                                 nn.ReLU(), nn.Linear(64, 3))

    def forward(self, x):
        return self.net(x)


def config(**kw):
    base = dict(model="tiny", tag="test", lr=1e-2, max_epochs=60,
                patience=60, n_boot=20)
    base.update(kw)
    return train.TrainConfig(**base)


# --------------------------------------------------------------- the loop
def test_loop_overfits_a_small_batch():
    """A network that can memorise 32 records must drive the training
    loss towards 0. If one of the five steps were missing (e.g. no
    zero_grad or no step) this would fail."""
    loaders, groups = classification_loaders()
    task = MultiLabelTask(CLASSES, groups)
    train.set_seed(0)
    model = TinyMLP()
    out = train.fit(model, loaders, task, config(), CPU, verbose=False)
    assert out["history"]["train_loss"][-1] < 0.05
    assert out["best_value"] == pytest.approx(1.0)      # val AUROC


def test_early_stopping_stops_after_patience():
    """With lr = 0 nothing changes, so validation never improves after
    epoch 1 and training stops after `patience` more epochs."""
    loaders, groups = classification_loaders()
    task = MultiLabelTask(CLASSES, groups)
    model = TinyMLP()
    out = train.fit(model, loaders, task,
                    config(lr=0.0, max_epochs=50, patience=3), CPU,
                    verbose=False)
    assert out["best_epoch"] == 1
    assert out["epochs_run"] == 4
    assert out["stopped_early"]


def test_best_weights_are_restored():
    """After fit, evaluating the model gives the best epoch's value."""
    loaders, groups = classification_loaders()
    task = MultiLabelTask(CLASSES, groups)
    train.set_seed(0)
    model = TinyMLP()
    out = train.fit(model, loaders, task, config(max_epochs=8), CPU,
                    verbose=False)
    loss, state = train.evaluate(model, loaders["val"], task, CPU)
    value = task.epoch_metrics(loss, state)["macro_auroc"]
    assert value == pytest.approx(out["best_value"])


def test_frozen_layers_do_not_change():
    """Frozen parameters (requires_grad=False) stay as they were (D7)."""
    loaders, groups = classification_loaders()
    task = MultiLabelTask(CLASSES, groups)
    model = TinyMLP()
    first = model.net[1]
    for p in first.parameters():
        p.requires_grad = False
    before = first.weight.clone()
    train.fit(model, loaders, task, config(max_epochs=3), CPU, verbose=False)
    assert torch.equal(first.weight, before)
    total, trainable = train.count_parameters(model)
    assert trainable == total - first.weight.numel() - first.bias.numel()


def test_same_seed_same_history():
    loaders, groups = classification_loaders()
    task = MultiLabelTask(CLASSES, groups)
    runs = []
    for _ in range(2):
        loaders, _ = classification_loaders()
        train.set_seed(7)
        model = TinyMLP()
        runs.append(train.fit(model, loaders, task, config(max_epochs=3),
                              CPU, verbose=False)["history"]["train_loss"])
    assert runs[0] == runs[1]


def test_evaluation_loss_is_exact_mean_over_records():
    """The last batch is smaller (30 = 8 + 8 + 8 + 6); the loop weights
    each batch by its size, so the result equals the loss on all records
    at once."""
    loaders, groups = classification_loaders(n=30)
    task = MultiLabelTask(CLASSES, groups)
    model = TinyMLP()
    loss, _ = train.evaluate(model, loaders["val"], task, CPU)
    x, y = loaders["val"].dataset.tensors
    with torch.no_grad():
        expected = task.loss(model(x), y).item()
    assert loss == pytest.approx(expected, rel=1e-5)


# ------------------------------------------------------ next-token task
class Bigram(nn.Module):
    """Predicts the next token from the current one only."""
    def __init__(self, vocab: int):
        super().__init__()
        self.emb = nn.Embedding(vocab, vocab)

    def forward(self, tokens):                   # (B, T) -> (B, T, V)
        return self.emb(tokens)


def test_next_token_task_learns_a_cyclic_sequence():
    """Sequence 0, 1, ..., V-1, 0, 1, ...: the next token is fully
    determined, so perplexity must approach 1. Untrained, with all logits
    equal, it is V (uniform guess)."""
    V, T = 8, 32
    starts = torch.arange(64) % V
    seq = (starts[:, None] + torch.arange(T + 1)[None, :]) % V
    ds = TensorDataset(seq[:, :-1], seq[:, 1:])  # input, target shifted by 1
    loaders = {"train": DataLoader(ds, batch_size=16, shuffle=True),
               "val": DataLoader(ds, batch_size=16)}
    task = NextTokenTask(V)

    model = Bigram(V)
    nn.init.zeros_(model.emb.weight)
    loss, state = train.evaluate(model, loaders["val"], task, CPU)
    assert task.epoch_metrics(loss, state)["perplexity"] == pytest.approx(V)

    out = train.fit(model, loaders, task, config(lr=0.1, max_epochs=40),
                    CPU, verbose=False)
    assert out["history"]["val_perplexity"][-1] < 1.1
    assert task.mode == "min" and out["best_value"] < 0.1


# ----------------------------------------------------------- full run
def test_run_experiment_writes_log_checkpoint_and_predictions(tmp_path):
    loaders, groups = classification_loaders()
    task = MultiLabelTask(CLASSES, groups,
                          record_ids={s: np.arange(32) + 1000 for s in groups},
                          n_boot=20)
    cfg = config(model="trivial", max_epochs=3, final=True)
    path, rec = train.run_experiment(
        lambda: LeadStatsLinear(n_classes=3), loaders, task, cfg,
        {"n_records": {"train": 32}}, logs_dir=tmp_path / "logs",
        checkpoint_dir=tmp_path / "ckpt", predictions_dir=tmp_path / "pred",
        device=CPU, verbose=False)

    with open(path, encoding="utf-8") as f:
        on_disk = json.load(f)                   # valid JSON
    assert on_disk["run_id"] == rec["run_id"] == path.stem
    for key in ("scalars", "config", "model", "data", "selection",
                "history", "results", "checkpoint", "predictions", "meta"):
        assert key in on_disk
    s = on_disk["scalars"]
    assert s["val_macro_auroc_lo"] <= s["val_macro_auroc"] <= s["val_macro_auroc_hi"]
    assert "test_macro_auroc" in s and "threshold_A" in s
    assert all(isinstance(v, (int, float)) for v in s.values())
    assert len(on_disk["history"]["epoch"]) == 3
    assert (tmp_path / "ckpt" / f"{path.stem}.pt").exists()

    pred = np.load(tmp_path / "pred" / f"{path.stem}.npz")
    assert pred["val_prob"].shape == (32, 3)
    assert np.array_equal(pred["test_ecg_id"], np.arange(32) + 1000)


def test_test_fold_untouched_without_final(tmp_path):
    loaders, groups = classification_loaders()
    task = MultiLabelTask(CLASSES, groups, n_boot=20)
    _, rec = train.run_experiment(
        lambda: LeadStatsLinear(n_classes=3), loaders, task,
        config(model="trivial", max_epochs=2), {}, logs_dir=tmp_path,
        checkpoint_dir=tmp_path, predictions_dir=tmp_path, device=CPU,
        verbose=False)
    assert "test" not in rec["results"]
    assert not any(k.startswith("test_") for k in rec["scalars"])


def test_second_final_run_is_refused(tmp_path):
    loaders, groups = classification_loaders()
    task = MultiLabelTask(CLASSES, groups, n_boot=20)
    kw = dict(logs_dir=tmp_path, checkpoint_dir=tmp_path,
              predictions_dir=tmp_path, device=CPU, verbose=False)
    cfg = config(model="trivial", max_epochs=1, final=True)
    build = lambda: LeadStatsLinear(n_classes=3)
    train.run_experiment(build, loaders, task, cfg, {}, **kw)
    with pytest.raises(RuntimeError, match="once per final model"):
        train.run_experiment(build, loaders, task, cfg, {}, **kw)


# --------------------------------------------------------------- PTB-XL
def test_fast_subset_and_task_alignment():
    """--fast keeps 512/256/256 records and the patient ids given to the
    task follow the order of the validation loader."""
    from src.dataset import split_indices
    from tests.test_data import toy_cache
    cache = toy_cache(n_per_fold=400)
    cache["ecg_id"] = np.arange(len(cache["fold"])) + 1
    cfg = train.TrainConfig(fast=True, batch_size=1000, normalize=False)
    loaders, info, task = train.ptbxl_setup(cfg, cache=cache)
    assert info["n_records"] == {"train": 512, "val": 256, "test": 256}
    # Order check: the validation batch and the task's ids come from the
    # same rows of the full cache.
    full_val = split_indices(cache)["val"]
    x_val, _ = next(iter(loaders["val"]))
    assert len(task.groups["val"]) == len(x_val) == 256
    assert set(task.record_ids["val"]) <= set(cache["ecg_id"][full_val])
    rows = np.searchsorted(cache["ecg_id"], task.record_ids["val"])
    np.testing.assert_array_equal(cache["patient_id"][rows],
                                  task.groups["val"])
    np.testing.assert_allclose(x_val.numpy(), cache["X"][rows], rtol=1e-6)
