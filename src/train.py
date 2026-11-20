"""Shared training loop (T8) and command line for PTB-XL runs (D14).

PyTorch has no compile() and no fit(): we write the loop ourselves, with the
five steps of every iteration (weekly summary 13/11):

    1. optimizer.zero_grad()   clear the gradients of the previous batch
    2. output = model(x)       forward pass
    3. loss = task.loss(...)   loss on this batch
    4. loss.backward()         backpropagation: gradients of the loss
    5. optimizer.step()        update the weights

model.train() switches dropout (and batch-norm statistics) on for training;
model.eval() and torch.no_grad() switch them off and stop gradient tracking
during evaluation.

The loop does not know the problem: the loss, the metrics and the metric
used for early stopping come from a task object (src/tasks.py), so the same
code trains the classifiers and the autoregressive generator of D13.

Command line (from the repository root):
    python -m src.train --model trivial --fast            # smoke test
    python -m src.train --model trivial                   # full run
    python -m src.train --model trivial --final           # + test fold
"""
import argparse
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn

from src import config, runlog
from src.tasks import MultiLabelTask, Task


@dataclass
class TrainConfig:
    """Everything that defines a run; logged as `config` in the JSON."""
    model: str = "trivial"
    model_args: dict = field(default_factory=dict)
    tag: str = "base"                 # names the experiment or ablation
    seed: int = config.SEED
    lr: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 64
    max_epochs: int = 50
    patience: int = 5                 # epochs without improvement
    grad_clip: float | None = None    # max gradient norm; None = off
    deterministic: bool = True        # cuDNN deterministic, no benchmark
    normalize: bool = True            # per-lead z-score (D3)
    pos_weight: bool = False          # class weights in the loss (D14)
    n_boot: int = 1000
    fast: bool = False
    final: bool = False               # evaluate the test fold (D3)
    notes: str = ""


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch (CPU and every GPU).

    Called before the model is built, so the initial weights are the same
    for the same seed. cuDNN's deterministic mode picks reproducible
    convolution algorithms; benchmark mode is off because it chooses the
    fastest algorithm by timing, which can change between runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)           # also seeds the GPUs
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def rel(path: Path | None) -> str | None:
    """Path relative to the repository root, with / separators, so logs
    look the same on Windows and Linux."""
    if path is None:
        return None
    path = Path(path).resolve()
    if path.is_relative_to(config.ROOT):
        return path.relative_to(config.ROOT).as_posix()
    return path.as_posix()


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ---------------------------------------------------------------------------
# One epoch of training, one pass of evaluation
# ---------------------------------------------------------------------------
def train_one_epoch(model: nn.Module, loader, task: Task,
                    optimizer: torch.optim.Optimizer, device: torch.device,
                    grad_clip: float | None = None) -> float:
    """One pass over the training set; returns the mean training loss."""
    model.train()                     # dropout and batch norm in training mode
    total, n = 0.0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad()                  # 1. clear old gradients
        output = model(x)                      # 2. forward
        loss = task.loss(output, y)            # 3. loss (mean over the batch)
        loss.backward()                        # 4. backpropagation
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()                       # 5. update the weights
        k = task.n_terms(y)
        total += loss.item() * k               # .item(): a Python number,
        n += k                                 # so no graph is kept alive
    return total / n


@torch.no_grad()                      # no gradients: faster, less memory
def evaluate(model: nn.Module, loader, task: Task,
             device: torch.device) -> tuple[float, object]:
    """Mean loss over the whole split and whatever the task collects."""
    model.eval()                      # dropout off, batch norm uses its
    total, n = 0.0, 0                 # running statistics
    state = task.start()
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        output = model(x)
        # Sum, then divide by the number of terms at the end: the exact mean
        # over the split, even if the last batch is smaller.
        total += task.loss(output, y, reduction="sum").item()
        n += task.n_terms(y)
        state = task.collect(state, output, y)
    return total / n, state


# ---------------------------------------------------------------------------
# Training with early stopping
# ---------------------------------------------------------------------------
def _improves(value: float, best: float | None, mode: str) -> bool:
    if math.isnan(value):
        return False
    if best is None or math.isnan(best):
        return True
    return value > best if mode == "max" else value < best


def fit(model: nn.Module, loaders: dict, task: Task, cfg: TrainConfig,
        device: torch.device, verbose: bool = True) -> dict:
    """Train with early stopping on the task's selection metric.

    After every epoch the model is evaluated on validation. When the
    selection metric improves, a copy of the weights is kept; after
    `patience` epochs without improvement training stops. At the end the
    best weights are loaded back into the model.
    """
    # Only trainable parameters: frozen layers (requires_grad=False, D7)
    # are left out without any special code.
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=cfg.lr,
                                 weight_decay=cfg.weight_decay)

    history = {"epoch": [], "train_loss": [], "lr": [], "time_s": []}
    best, best_epoch, best_state = None, 0, None
    waited, stopped_early = 0, False
    for epoch in range(1, cfg.max_epochs + 1):
        t0 = time.perf_counter()
        train_loss = train_one_epoch(model, loaders["train"], task,
                                     optimizer, device, cfg.grad_clip)
        val_loss, state = evaluate(model, loaders["val"], task, device)
        val = task.epoch_metrics(val_loss, state)
        elapsed = time.perf_counter() - t0

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["lr"].append(optimizer.param_groups[0]["lr"])
        history["time_s"].append(elapsed)
        for k, v in val.items():
            history.setdefault(f"val_{k}", []).append(v)

        value = val[task.selection_metric]
        if _improves(value, best, task.mode):
            best, best_epoch, waited = value, epoch, 0
            # Copy to CPU: a reference to model.state_dict() would keep
            # changing as training goes on.
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            mark = " *"
        else:
            waited += 1
            mark = ""
        if verbose:
            print(f"epoch {epoch:3d}  train_loss {train_loss:.4f}  "
                  f"val_loss {val_loss:.4f}  "
                  f"val_{task.selection_metric} {value:.4f}  "
                  f"({elapsed:.1f} s){mark}")
        if waited >= cfg.patience:
            stopped_early = True
            break

    model.load_state_dict(best_state)
    return {"history": history, "best_epoch": best_epoch,
            "best_value": best, "stopped_early": stopped_early,
            "epochs_run": len(history["epoch"])}


# ---------------------------------------------------------------------------
# A complete run: seed, model, training, final evaluation, logs
# ---------------------------------------------------------------------------
def run_experiment(build_model: Callable[[], nn.Module], loaders: dict,
                   task: Task, cfg: TrainConfig, data_info: dict,
                   logs_dir: Path = config.LOGS_DIR,
                   checkpoint_dir: Path = config.CHECKPOINT_DIR,
                   predictions_dir: Path = config.PREDICTIONS_DIR,
                   device: torch.device | None = None,
                   verbose: bool = True) -> tuple[Path, dict]:
    """Train one model and write its JSON log, checkpoint and predictions.

    build_model is a function that returns a new model: it is called after
    the seed is set, so the initial weights depend only on the seed.
    `loaders` holds "train", "val" and, for final runs, "test".
    Returns (path of the JSON log, the log as a dict).
    """
    if cfg.final and not cfg.fast:
        previous = runlog.previous_final_runs(cfg.model, cfg.tag, cfg.seed,
                                              logs_dir)
        if previous:
            raise RuntimeError(
                f"the test fold was already used for {cfg.model}/{cfg.tag}/"
                f"seed {cfg.seed}: {previous}. D3: once per final model.")

    device = device or torch.device("cuda" if torch.cuda.is_available()
                                    else "cpu")
    set_seed(cfg.seed, cfg.deterministic)
    model = build_model().to(device)
    n_params, n_trainable = count_parameters(model)
    run_id = runlog.make_run_id(cfg.model, cfg.tag, cfg.seed)
    if cfg.fast:                      # smoke tests: ignored by git (D14)
        run_id += "_fast"
    if verbose:
        print(f"run {run_id} on {device}: {n_params:,} parameters "
              f"({n_trainable:,} trainable)")

    t0 = time.perf_counter()
    trained = fit(model, loaders, task, cfg, device, verbose)
    train_time = time.perf_counter() - t0

    # Final evaluation of the best weights.
    results = {"val": evaluate(model, loaders["val"], task, device)}
    if cfg.final:
        results["test"] = evaluate(model, loaders["test"], task, device)
    details, task_scalars, arrays = task.report(results)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_dir / f"{run_id}.pt"
    torch.save(model.state_dict(), checkpoint)
    predictions = None
    if arrays:
        predictions_dir.mkdir(parents=True, exist_ok=True)
        predictions = predictions_dir / f"{run_id}.npz"
        np.savez_compressed(predictions, **arrays)

    scalars = {"best_epoch": trained["best_epoch"],
               "epochs_run": trained["epochs_run"],
               "n_params": n_params, "n_trainable": n_trainable,
               "train_time_s": train_time}
    scalars.update(task_scalars)
    record = {
        "run_id": run_id,
        "schema": runlog.SCHEMA_VERSION,
        "task": task.name,
        "fast": cfg.fast,
        "final": cfg.final,
        "scalars": scalars,
        "config": asdict(cfg),
        "model": {"name": cfg.model, "args": cfg.model_args,
                  "n_params": n_params, "n_trainable": n_trainable},
        "data": data_info,
        "selection": {"metric": task.selection_metric, "mode": task.mode,
                      "best_value": trained["best_value"],
                      "best_epoch": trained["best_epoch"],
                      "stopped_early": trained["stopped_early"]},
        "history": trained["history"],
        "results": details,
        "checkpoint": rel(checkpoint),
        "predictions": rel(predictions),
        "meta": runlog.environment(device),
    }
    path = runlog.write_run(record, logs_dir)
    if verbose:
        print(f"\nbest epoch {trained['best_epoch']} of "
              f"{trained['epochs_run']}; log written to {rel(path)}")
    return path, record


# ---------------------------------------------------------------------------
# PTB-XL: data for the 5-superclass task
# ---------------------------------------------------------------------------
PER_RECORD_KEYS = ("X", "y", "ecg_id", "patient_id", "fold", "has_superclass")
FAST_SIZES = {"train": 512, "val": 256, "test": 256}


def fast_subset(cache: dict, seed: int) -> dict:
    """Small random cache for --fast: 512/256/256 records drawn with the
    seed from the three splits (D14)."""
    from src.dataset import split_indices
    idx = split_indices(cache)
    rng = np.random.default_rng(seed)
    rows = np.sort(np.concatenate([
        rng.choice(idx[s], size=min(n, len(idx[s])), replace=False)
        for s, n in FAST_SIZES.items()]))
    small = {k: v[rows] for k, v in cache.items() if k in PER_RECORD_KEYS}
    small.update({k: v for k, v in cache.items() if k not in PER_RECORD_KEYS})
    return small


def ptbxl_setup(cfg: TrainConfig, cache: dict | None = None):
    """Loaders, data description and task for the 5-superclass problem."""
    from src.data import load_cache
    from src.dataset import make_loaders, split_indices
    cache = load_cache() if cache is None else cache
    if cfg.fast:
        cache = fast_subset(cache, cfg.seed)
    loaders, info = make_loaders(cache, batch_size=cfg.batch_size,
                                 normalize=cfg.normalize, seed=cfg.seed)
    # Same function and arguments as make_loaders, so the order matches the
    # (unshuffled) validation and test loaders record by record.
    idx = split_indices(cache)
    groups = {s: cache["patient_id"][r] for s, r in idx.items()}
    ecg_ids = {s: cache["ecg_id"][r] for s, r in idx.items()}

    pos_weight = None
    if cfg.pos_weight:
        y_train = cache["y"][idx["train"]]
        positives = y_train.sum(axis=0)
        pos_weight = (len(y_train) - positives) / positives
        info["pos_weight"] = pos_weight.tolist()
    info["fast_subset"] = FAST_SIZES if cfg.fast else None

    task = MultiLabelTask(config.CLASSES, groups, ecg_ids, pos_weight,
                          n_boot=cfg.n_boot, seed=config.SEED)
    return loaders, info, task


def parse_args(argv=None) -> TrainConfig:
    from src.models import MODELS
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", default="trivial", choices=sorted(MODELS))
    p.add_argument("--tag", default="base")
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--grad-clip", type=float, default=None)
    p.add_argument("--no-normalize", action="store_true")
    p.add_argument("--pos-weight", action="store_true")
    p.add_argument("--fast", action="store_true")
    p.add_argument("--final", action="store_true")
    p.add_argument("--notes", default="")
    a = p.parse_args(argv)
    cfg = TrainConfig(model=a.model, tag=a.tag, seed=a.seed, lr=a.lr,
                      weight_decay=a.weight_decay, batch_size=a.batch_size,
                      max_epochs=a.max_epochs, patience=a.patience,
                      grad_clip=a.grad_clip, normalize=not a.no_normalize,
                      pos_weight=a.pos_weight, fast=a.fast, final=a.final,
                      notes=a.notes)
    if cfg.fast:                      # D14: 2 epochs, 50 resamples
        cfg.max_epochs, cfg.n_boot = 2, 50
    return cfg


def print_summary(record: dict) -> None:
    """What to paste back into the chat."""
    print("\n== Summary ==")
    for key, value in record["scalars"].items():
        if key.startswith(("val_macro", "test_macro", "best_epoch",
                           "epochs_run", "n_params", "train_time")):
            print(f"  {key}: {value:.4f}" if isinstance(value, float)
                  else f"  {key}: {value}")
    print(f"  bootstrap: {record['results']['val']['bootstrap']}")
    print(f"  git_commit: {record['meta']['git_commit']}  "
          f"git_dirty: {record['meta']['git_dirty']}")
    if not record["fast"]:
        print("\nTo use it in the reports, add an alias to report/runs.json,"
              f' e.g. "myAlias": "{record["run_id"]}"')


def main(argv=None) -> None:
    from src.models import MODELS
    cfg = parse_args(argv)
    loaders, info, task = ptbxl_setup(cfg)
    model_class = MODELS[cfg.model]
    _, record = run_experiment(lambda: model_class(**cfg.model_args),
                               loaders, task, cfg, info)
    print_summary(record)


if __name__ == "__main__":
    main()
