"""RNN block (T11): LSTM / GRU, RNN + attention, ablations, final models.

Decisions: docs/DECISIONS.md D17 (this block), D3, D4 and D14 (protocol).
Same structure and protocol as src/cnn_experiments.py (D16).

    python -m src.rnn_experiments estimate     # time per epoch on this GPU,
                                               # before the long runs
    python -m src.rnn_experiments all --fast   # smoke test of everything
    python -m src.rnn_experiments run          # 12 configurations x 3 seeds,
                                               # validation only
    python -m src.rnn_experiments final        # the two selected models
                                               # -> test fold
    python -m src.rnn_experiments report       # summary log, tables, figures

`run` and `final` can be stopped and started again: a configuration and
seed that already has a (non --fast) log is skipped.

How an ablation works: the baseline is a 2-layer unidirectional LSTM with
128 units that reads the record as 100 frames of 100 ms and classifies its
last state (D6). Each other configuration changes ONE factor and keeps
everything else equal - same data, split, loss, optimiser, learning rate,
batch size, early stopping, gradient clipping and seeds - and is compared
with a reference trained with the same seed: the baseline, or for the
"x" configurations the model without attention (does attention help more
with 1000 steps, with a unidirectional network, with a GRU?), and for
`attn_tanh` the model with additive attention (score function).

Two final models are chosen on validation before the test fold is touched
(D17): the best configuration overall, and the best configuration with
attention (the "RNN + attention" model asked for in the professor's
message of 20/11). When they coincide there is only one.

Outputs of `report`:
    results/logs/rnn_summary.json      numbers for make_numbers.py
                                       (alias "rnn" in report/runs.json)
    report/tables/rnn_ablations.tex    ablation table (LaTeX macros only)
    report/tables/rnn_attention.tex    how concentrated the attention is
    report/tables/rnn_final.tex        test results of the final models,
                                       with the final CNN, Dense and trivial
    report/figures/rnn_curves.*        learning curves (seed 42)
    report/figures/rnn_ablations.*     validation macro AUROC per config
    report/figures/rnn_time_gradient.* which part of the 10 s the output
                                       depends on, per configuration
    report/figures/rnn_attention.*     attention weights over the ECG of
                                       one record per superclass
With --fast every output name ends in _fast and git ignores it.
"""
import os

# PyTorch's nn.LSTM documentation warns of non-deterministic RNN functions
# on some cuDNN / CUDA versions and suggests this variable. It must be set
# before CUDA starts, i.e. before the first model is moved to the GPU (D17).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:2")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from datetime import datetime  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src import cnn_experiments as cx  # noqa: E402   generic helpers
from src import config, runlog  # noqa: E402
from src import dense_experiments as dx  # noqa: E402
from src.dense_experiments import Paths  # noqa: E402
from src.models.rnn import AttentionPooling, RNNClassifier  # noqa: E402
from src.train import (TrainConfig, ptbxl_setup, rel, run_experiment,  # noqa: E402
                       set_seed)

MODEL = "rnn"
SEEDS = (42, 43, 44)                  # D14: seeds of the final models
ALIAS = "rnn"                         # alias of the summary in runs.json
SUMMARY_NAME = "rnn_summary"
GRAD_CLIP = 1.0                       # max total gradient norm (D17)
EARLY_EPOCH = cx.EARLY_EPOCH          # training loss compared at epoch 5

# Baseline (D17). Every ablation starts from these values.
BASE_ARGS = {"cell": "lstm", "hidden_size": 128, "num_layers": 2,
             "bidirectional": False, "front_end": "frames", "frame": 10,
             "readout": "last", "attention": "additive",
             "attention_dim": 64, "dropout": 0.5}

# Interpretation (D17): records and windows of the sensitivity analysis.
N_SENS_RECORDS = 512                  # validation records, drawn with seed 42
EDGE_S = 2.0                          # "first / last 2 s" of the record
BIN_S = 0.1                           # bins of the stored profiles


@dataclass(frozen=True)
class Ablation:
    tag: str                  # run name part; letters, digits, underscores
    factor: str               # family, used to group the table and figure
    label: str                # short English description for the report
    changes: dict = field(default_factory=dict)   # changes to BASE_ARGS
    reference: str = "base"   # configuration it is compared with

    @property
    def model_args(self) -> dict:
        return {**BASE_ARGS, **self.changes}

    @property
    def has_attention(self) -> bool:
        return self.model_args["readout"] == "attention"


ATT = {"readout": "attention"}
ABLATIONS = [
    Ablation("base", "baseline",
             "Baseline: LSTM 2 x 128, 100 frames, last state"),
    Ablation("gru", "cell", "GRU instead of LSTM", {"cell": "gru"}),
    Ablation("layers1", "depth", "1 recurrent layer", {"num_layers": 1}),
    Ablation("bi", "direction", "Bidirectional", {"bidirectional": True}),
    Ablation("mean", "readout", "Mean of the states over time",
             {"readout": "mean"}),
    Ablation("attn", "readout", "Additive attention", ATT),
    Ablation("attn_tanh", "attention score",
             "Notebook score tanh(w.h + b) (vs. additive)",
             {**ATT, "attention": "notebook"}, reference="attn"),
    Ablation("gru_attn", "attention x cell", "GRU + attention (vs. GRU)",
             {"cell": "gru", **ATT}, reference="gru"),
    Ablation("bi_attn", "attention x direction",
             "Bidirectional + attention (vs. bidirectional)",
             {"bidirectional": True, **ATT}, reference="bi"),
    Ablation("conv", "input", "Convolutional front end (125 steps)",
             {"front_end": "conv"}),
    # The two slowest configurations (1000 steps) go last, so that a long
    # `run` has done everything else first.
    Ablation("raw", "sequence length", "Raw samples (1000 steps)",
             {"frame": 1}),
    Ablation("raw_attn", "attention x sequence length",
             "1000 steps + attention (vs. 1000 steps)",
             {"frame": 1, **ATT}, reference="raw"),
]
BY_TAG = {a.tag: a for a in ABLATIONS}
ATTENTION_TAGS = [a.tag for a in ABLATIONS if a.has_attention]


# ---------------------------------------------------------------------------
# 1. Configurations and runs
# ---------------------------------------------------------------------------
def make_config(ab: Ablation, seed: int, fast: bool = False,
                final: bool = False) -> TrainConfig:
    """TrainConfig of one ablation: the D14 defaults (as D15, D16) plus
    gradient clipping at norm GRAD_CLIP for every configuration (RNN slide
    49: exploding gradients; D17). Input z-score (D3) always on."""
    cfg = TrainConfig(model=MODEL, tag=ab.tag, seed=seed,
                      model_args=ab.model_args, normalize=True,
                      grad_clip=GRAD_CLIP, fast=fast, final=final,
                      notes=ab.label)
    if fast:                          # D14: 2 epochs, 50 resamples
        cfg.max_epochs, cfg.n_boot = 2, 50
    return cfg


def build_model(model_args: dict) -> RNNClassifier:
    return RNNClassifier(**model_args)


def find_runs(paths: Paths, fast: bool, final: bool) -> dict:
    """(tag, seed) -> log of the latest matching RNN run (by content)."""
    return cx.find_runs(paths, fast, final, model=MODEL)


def _train(ab: Ablation, seed: int, fast: bool, final: bool, cache: dict,
           paths: Paths) -> dict:
    cfg = make_config(ab, seed, fast, final)
    loaders, info, task = ptbxl_setup(cfg, cache)
    info["cublas_workspace_config"] = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    _, record = run_experiment(lambda: build_model(cfg.model_args), loaders,
                               task, cfg, info, logs_dir=paths.logs,
                               checkpoint_dir=paths.checkpoints,
                               predictions_dir=paths.predictions)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()      # free the GPU before the next model
    return record


def run_ablations(seeds=SEEDS, only=None, fast: bool = False,
                  cache: dict | None = None, paths: Paths = Paths()) -> None:
    """Train every configuration with every seed, on validation only."""
    tags = only or [a.tag for a in ABLATIONS]
    unknown = set(tags) - set(BY_TAG)
    if unknown:
        raise ValueError(f"unknown configurations: {sorted(unknown)}")
    cache = dx._load_cache(cache)
    done = find_runs(paths, fast=False, final=False)
    todo = [(t, s) for t in tags for s in seeds]
    for i, (tag, seed) in enumerate(todo, 1):
        if not fast and (tag, seed) in done:
            print(f"[{i}/{len(todo)}] {tag} seed {seed}: already run "
                  f"({done[(tag, seed)]['run_id']}), skipped")
            continue
        print(f"\n[{i}/{len(todo)}] {tag} seed {seed}: {BY_TAG[tag].label}")
        _train(BY_TAG[tag], seed, fast, False, cache, paths)


# ---------------------------------------------------------------------------
# 2. Selection of the final configurations (validation only, D3)
# ---------------------------------------------------------------------------
def val_scores(runs: dict, seeds=SEEDS) -> dict:
    """tag -> list of validation macro AUROC, one per seed (in seed order).
    Only configurations run with every seed are returned."""
    out = {}
    for ab in ABLATIONS:
        values = [runs[(ab.tag, s)]["scalars"]["val_macro_auroc"]
                  for s in seeds if (ab.tag, s) in runs]
        if len(values) == len(seeds):
            out[ab.tag] = values
    return out


def select_final(runs: dict, seeds=SEEDS, pool=None) -> str:
    """The configuration of `pool` (default: all) with the highest mean
    validation macro AUROC over the seeds (D17, as D15 and D16). The test
    fold plays no part in this choice."""
    pool = list(pool or [a.tag for a in ABLATIONS])
    scores = val_scores(runs, seeds)
    if not scores:
        raise RuntimeError("no configuration has been run with every seed "
                           f"{tuple(seeds)}: run `run` first")
    missing = [a.tag for a in ABLATIONS if a.tag not in scores]
    if missing:
        raise RuntimeError(f"configurations not run with every seed: "
                           f"{missing}. Selecting now would ignore them.")
    return max(pool, key=lambda t: float(np.mean(scores[t])))


def select_finals(runs: dict, seeds=SEEDS) -> tuple[str, str]:
    """(best configuration overall, best configuration with attention)."""
    return (select_final(runs, seeds),
            select_final(runs, seeds, pool=ATTENTION_TAGS))


def run_final(fast: bool = False, cache: dict | None = None,
              paths: Paths = Paths(), seeds=SEEDS) -> list[str]:
    """Train the selected configurations again with --final for each seed:
    same training, plus one evaluation of the test fold (D3, D14)."""
    best, best_attn = select_finals(find_runs(paths, fast=fast, final=False),
                                    seeds)
    print(f"selected on validation: {best} ({BY_TAG[best].label})")
    print(f"best with attention:    {best_attn} ({BY_TAG[best_attn].label})")
    tags = list(dict.fromkeys([best, best_attn]))   # one if they coincide
    done = find_runs(paths, fast=False, final=True)
    cache = dx._load_cache(cache)
    for tag in tags:
        for seed in seeds:
            if not fast and (tag, seed) in done:
                print(f"final {tag} seed {seed}: already run, skipped")
                continue
            print(f"\nfinal {tag} seed {seed}")
            _train(BY_TAG[tag], seed, fast, True, cache, paths)
    return tags


# ---------------------------------------------------------------------------
# 3. Time estimate on this machine (before the long runs)
# ---------------------------------------------------------------------------
def _time_steps(model, x, y, task, train: bool, n_steps: int,
                device) -> float:
    """Mean seconds per batch of a training step (5 steps of the loop) or
    of an evaluation forward pass."""
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    def step():
        if train:
            model.train()
            opt.zero_grad()
            loss = task.loss(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
        else:
            model.eval()
            with torch.no_grad():
                model(x)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    for _ in range(2):                # warm-up (cuDNN picks its kernels)
        step()
    sync()
    t0 = time.perf_counter()
    for _ in range(n_steps):
        step()
    sync()
    return (time.perf_counter() - t0) / n_steps


def estimate(cache: dict | None = None, n_steps: int = 10,
             epochs=(15, 25, 35), seeds=SEEDS, fast: bool = False) -> dict:
    """Seconds per epoch of every configuration on this machine, measured
    on one real batch: training batches x time of a training step +
    validation batches x time of a forward pass + time to read the data
    once. An epoch count cannot be known in advance (early stopping), so
    the totals are given for several counts. Nothing is written."""
    cache = dx._load_cache(cache)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = make_config(BY_TAG["base"], seeds[0], fast=fast)
    loaders, _, task = ptbxl_setup(cfg, cache)
    t0 = time.perf_counter()
    for _ in loaders["train"]:
        pass
    for _ in loaders["val"]:
        pass
    t_data = time.perf_counter() - t0
    n_train, n_val = len(loaders["train"]), len(loaders["val"])
    x, y = next(iter(loaders["train"]))
    x, y = x.to(device), y.to(device)
    print(f"device {device}; {n_train} training and {n_val} validation "
          f"batches; reading the data once: {t_data:.1f} s")
    print(f"\n{'configuration':<11}{'steps':>6}{'train ms':>10}{'eval ms':>9}"
          f"{'s/epoch':>9}  minutes per run at {', '.join(map(str, epochs))}"
          " epochs")
    out = {}
    for ab in ABLATIONS:
        set_seed(seeds[0])
        model = build_model(ab.model_args).to(device)
        t_tr = _time_steps(model, x, y, task, True, n_steps, device)
        t_ev = _time_steps(model, x, y, task, False, n_steps, device)
        per_epoch = n_train * t_tr + n_val * t_ev + t_data
        out[ab.tag] = per_epoch
        mins = "  ".join(f"{per_epoch * e / 60:6.1f}" for e in epochs)
        print(f"{ab.tag:<11}{model.n_steps:>6}{1000 * t_tr:>10.1f}"
              f"{1000 * t_ev:>9.1f}{per_epoch:>9.1f}  {mins}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    total = sum(out.values()) * len(seeds)
    print("\nwhole `run` (" + f"{len(ABLATIONS)} configurations x "
          f"{len(seeds)} seeds): " +
          ", ".join(f"{total * e / 3600:.1f} h at {e} epochs"
                    for e in epochs))
    worst = 2 * len(seeds) * max(out.values())
    print(f"`final` (at most 2 configurations x {len(seeds)} seeds, slowest "
          "case): "
          + ", ".join(f"{worst * e / 3600:.1f} h at {e} epochs"
                      for e in epochs))
    return out


# ---------------------------------------------------------------------------
# 4. Interpretation: input sensitivity over time and attention weights
# ---------------------------------------------------------------------------
def load_model(rec: dict) -> RNNClassifier:
    """The model of a run with its best weights, in eval mode, on the CPU."""
    model = build_model(rec["model"]["args"])
    state = torch.load(config.ROOT / rec["checkpoint"], map_location="cpu",
                       weights_only=True)
    model.load_state_dict(state)
    return model.eval()


def normalised(cache: dict, rows, rec: dict) -> np.ndarray:
    """Signals of the cache rows with the run's own z-score (D3)."""
    X = cache["X"][rows].astype(np.float32)
    mean, std = rec["data"].get("lead_mean"), rec["data"].get("lead_std")
    if mean is not None:
        X = (X - np.asarray(mean, np.float32)[:, None]) / \
            np.asarray(std, np.float32)[:, None]
    return X


def sensitivity_rows(cache: dict, n: int = N_SENS_RECORDS,
                     seed: int = config.SEED) -> np.ndarray:
    """Cache rows of the validation records used for the sensitivity
    analysis: n drawn with the seed (all of them if there are fewer)."""
    from src.dataset import split_indices
    val = split_indices(cache)["val"]
    if len(val) <= n:
        return val
    return np.sort(np.random.default_rng(seed).choice(val, n, replace=False))


def input_sensitivity(model: torch.nn.Module, X: np.ndarray,
                      batch: int = 64) -> np.ndarray:
    """How much the output depends on each input sample, (samples,).

    For every record and class, |d logit_c / d x(lead, t)| summed over the
    leads; then averaged over classes and records, and normalised to sum 1.
    A unidirectional RNN that classifies its last state can only "remember"
    the beginning of the record through 100 (or 1000) recurrent steps; if
    gradients vanish backwards in time (RNN slide 48), the beginning
    barely changes the output and the profile piles up at the end.

    Computed on the CPU in eval mode (dropout off): on the GPU, cuDNN only
    runs the backward pass of an RNN in training mode (D17).
    """
    model = model.cpu().eval()
    total = np.zeros(X.shape[2])
    n_classes = model.classifier.out_features
    for i in range(0, len(X), batch):
        x = torch.from_numpy(X[i:i + batch]).requires_grad_(True)
        logits = model(x)
        for c in range(n_classes):
            # The records of a batch do not interact (eval mode), so the
            # gradient of the sum is the gradient of each record's logit.
            g, = torch.autograd.grad(logits[:, c].sum(), x,
                                     retain_graph=c < n_classes - 1)
            total += g.abs().sum(dim=1).sum(dim=0).numpy()
    return total / total.sum()


def sensitivity_numbers(profile: np.ndarray, tag: str) -> dict:
    """Share of the sensitivity in the first and last EDGE_S seconds (0.2
    each if it were uniform over 10 s), and the length of the shortest
    final window that holds half of it (5 s if uniform)."""
    k = int(round(EDGE_S * config.SAMPLING_RATE))
    tail = np.cumsum(profile[::-1])
    half = int(np.searchsorted(tail, 0.5) + 1)
    return {f"sens_first_{tag}": float(profile[:k].sum()),
            f"sens_last_{tag}": float(profile[-k:].sum()),
            f"sens_half_window_s_{tag}": half / config.SAMPLING_RATE}


def bin_profile(profile: np.ndarray, bin_s: float = BIN_S) -> list:
    """Profile summed into bins of bin_s seconds (stored in the summary)."""
    w = int(round(bin_s * config.SAMPLING_RATE))
    return profile[: len(profile) // w * w].reshape(-1, w).sum(axis=1).tolist()


def step_activity(X: np.ndarray, step_samples: int) -> np.ndarray:
    """How fast the signal changes in each RNN step, (records, T): mean
    absolute sample-to-sample change over the leads and the samples of the
    step. QRS complexes are the steepest part of an ECG, so they give the
    largest values; this is a proxy, not a QRS detector (D17)."""
    d = np.abs(np.diff(X, axis=2, prepend=X[:, :, :1])).mean(axis=1)
    n = d.shape[1] // step_samples * step_samples
    return d[:, :n].reshape(len(X), -1, step_samples).mean(axis=2)


def _row_spearman(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Spearman correlation of each row of a with the same row of b."""
    from scipy.stats import rankdata
    ra, rb = rankdata(a, axis=1), rankdata(b, axis=1)
    ra -= ra.mean(axis=1, keepdims=True)
    rb -= rb.mean(axis=1, keepdims=True)
    den = np.sqrt((ra ** 2).sum(axis=1) * (rb ** 2).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return (ra * rb).sum(axis=1) / den


def attention_weights(model: RNNClassifier, X: np.ndarray,
                      batch: int = 256) -> np.ndarray:
    return np.concatenate([
        model.attention_weights(torch.from_numpy(X[i:i + batch])).numpy()
        for i in range(0, len(X), batch)])


def attention_numbers(model: RNNClassifier, X: np.ndarray, tag: str) -> dict:
    """How concentrated the attention weights are, over the records X.

    Normalised entropy H(alpha) / log T: 1 = uniform weights (the same as
    mean pooling), 0 = all the weight on one step. exp(H) is the
    "effective number of steps" attended. Spearman correlation between the
    weights and the activity of each step (step_activity), per record."""
    alpha = attention_weights(model, X).astype(np.float64)
    T = alpha.shape[1]
    H = -(alpha * np.log(np.clip(alpha, 1e-12, None))).sum(axis=1)
    rho = _row_spearman(alpha, step_activity(X, model.step_samples))
    out = {f"attn_steps_{tag}": int(T),
           f"attn_uniform_{tag}": 1.0 / T,
           f"attn_entropy_norm_{tag}": float(np.mean(H / math.log(T))),
           f"attn_effective_steps_{tag}": float(np.mean(np.exp(H))),
           f"attn_max_weight_{tag}": float(np.mean(alpha.max(axis=1))),
           # NaN only if every record has constant weights (no ranking)
           f"attn_spearman_activity_{tag}": (float(np.nanmean(rho))
                                             if np.isfinite(rho).any()
                                             else math.nan)}
    if model.attention.kind == "notebook":
        out[f"attn_max_bound_{tag}"] = AttentionPooling.max_weight_bound(T)
    return out


# ---------------------------------------------------------------------------
# 5. Summary log for make_numbers.py
# ---------------------------------------------------------------------------
def block_finals(paths: Paths, alias: str, model: str) -> tuple[dict, str | None]:
    """seed -> final (test) run of the configuration that another block's
    summary (alias in runs.json) selected, and the run id of that summary."""
    summary = cx._alias_log(paths, alias)
    if summary is None or not summary.get("selected_config"):
        return {}, None
    tag = summary["selected_config"]
    found = cx.find_runs(paths, fast=False, final=True, model=model)
    return ({seed: r for (t, seed), r in found.items() if t == tag},
            summary["run_id"])


def final_numbers(final_recs: list, prefix: str, s: dict) -> None:
    """Test results of one final configuration over the seeds."""
    s[f"{prefix}_n_params"] = int(final_recs[0]["scalars"]["n_params"])
    s[f"{prefix}_n_seeds"] = len(final_recs)
    for key in ("val_macro_auroc", "test_macro_auroc", "test_macro_f1"):
        vals = [r["scalars"][key] for r in final_recs]
        s[f"{prefix}_{key}_mean"] = float(np.mean(vals))
        s[f"{prefix}_{key}_sd"] = dx._sd(vals)
    for c in config.CLASSES:
        vals = [r["scalars"][f"test_auroc_{c}"] for r in final_recs]
        s[f"{prefix}_test_auroc_{c}_mean"] = float(np.mean(vals))
        s[f"{prefix}_test_auroc_{c}_sd"] = dx._sd(vals)
    first = final_recs[0]["scalars"]
    s[f"{prefix}_first_seed"] = int(final_recs[0]["config"]["seed"])
    for key in ["test_macro_auroc", "test_macro_f1"] + \
            [f"test_auroc_{c}" for c in config.CLASSES]:
        s[f"{prefix}_first_{key}"] = first[key]
        s[f"{prefix}_first_{key}_lo"] = first[f"{key}_lo"]
        s[f"{prefix}_first_{key}_hi"] = first[f"{key}_hi"]
    for c in config.CLASSES:
        s[f"{prefix}_first_test_f1_{c}"] = first[f"test_f1_{c}"]


def build_summary(paths: Paths = Paths(), fast: bool = False,
                  seeds=SEEDS, cache: dict | None = None) -> dict:
    """Collect every number of the block into one log with a flat `scalars`
    block (read by make_numbers.py under the alias "rnn"). With the cache,
    the sensitivity profiles and attention statistics are added too."""
    runs = find_runs(paths, fast=fast, final=False)
    finals = find_runs(paths, fast=fast, final=True)
    n_boot = 50 if fast else 1000
    s: dict = {"grad_clip": GRAD_CLIP}
    sources = []

    # Ablations on validation.
    for ab in ABLATIONS:
        recs = [runs[(ab.tag, seed)] for seed in seeds if (ab.tag, seed) in runs]
        if not recs:
            continue
        sources += [r["run_id"] for r in recs]
        t = ab.tag
        aurocs = [r["scalars"]["val_macro_auroc"] for r in recs]
        s[f"params_{t}"] = int(recs[0]["scalars"]["n_params"])
        s[f"steps_{t}"] = int(build_model(ab.model_args).n_steps)
        s[f"val_auroc_mean_{t}"] = float(np.mean(aurocs))
        s[f"val_auroc_sd_{t}"] = dx._sd(aurocs)
        s[f"best_epoch_mean_{t}"] = float(np.mean(
            [r["scalars"]["best_epoch"] for r in recs]))
        s[f"epoch_time_mean_{t}"] = float(np.mean(
            [np.mean(r["history"]["time_s"]) for r in recs]))
        s[f"train_time_min_mean_{t}"] = float(np.mean(
            [r["scalars"]["train_time_s"] for r in recs])) / 60
        early = [cx.early_train_loss(r) for r in recs]
        if all(v is not None for v in early):
            s[f"train_loss_ep{EARLY_EPOCH}_mean_{t}"] = float(np.mean(early))
        first = runs.get((t, seeds[0]))
        if first is not None:
            sc = first["scalars"]
            s[f"val_auroc_first_{t}"] = sc["val_macro_auroc"]
            s[f"val_auroc_lo_first_{t}"] = sc["val_macro_auroc_lo"]
            s[f"val_auroc_hi_first_{t}"] = sc["val_macro_auroc_hi"]
        if t == "base":
            continue
        diffs, gaps, better, worse = [], [], 0, 0
        for seed in seeds:
            ref = runs.get((ab.reference, seed))
            if (t, seed) not in runs or ref is None:
                continue
            print(f"paired difference {t} - {ab.reference}, seed {seed} ...")
            d = dx.paired_diff(runs[(t, seed)], ref, "val", n_boot)
            diffs.append(d["diff"])
            s[f"diff_s{seed}_{t}"] = d["diff"]
            s[f"diff_lo_s{seed}_{t}"] = d["lower"]
            s[f"diff_hi_s{seed}_{t}"] = d["upper"]
            better += d["lower"] > 0          # whole interval above 0
            worse += d["upper"] < 0           # whole interval below 0
            a, b = cx.early_train_loss(runs[(t, seed)]), cx.early_train_loss(ref)
            if a is not None and b is not None:
                gaps.append(a - b)
        if diffs:
            s[f"diff_mean_{t}"] = float(np.mean(diffs))
            s[f"n_seeds_better_{t}"] = int(better)
            s[f"n_seeds_worse_{t}"] = int(worse)
            s[f"n_seeds_compared_{t}"] = len(diffs)
        if gaps:
            s[f"train_loss_ep{EARLY_EPOCH}_diff_mean_{t}"] = float(np.mean(gaps))
            s[f"n_seeds_train_loss_higher_{t}"] = int(sum(g > 0 for g in gaps))

    # Final models: the configurations selected on validation, on test.
    selected = selected_attn = None
    by_tag: dict = {}
    for (t, seed), r in sorted(finals.items(), key=lambda kv: kv[0][1]):
        if seed in seeds:
            by_tag.setdefault(t, []).append(r)
    if by_tag:
        selected, selected_attn = select_finals(runs, seeds)
        unexpected = set(by_tag) - {selected, selected_attn}
        if unexpected:
            raise RuntimeError(f"final runs of configurations that the "
                               f"selection rule did not choose: {unexpected}")
    trivial = dx._trivial_record(paths)
    cnn, cnn_summary = block_finals(paths, "cnn", "cnn")
    dense, dense_summary = block_finals(paths, "dense", "dense")
    blocks = [("final", selected), ("attnfinal", selected_attn)]
    for prefix, tag in blocks:
        recs = by_tag.get(tag)
        if not recs:
            continue
        sources += [r["run_id"] for r in recs if r["run_id"] not in sources]
        final_numbers(recs, prefix, s)
        s[f"{prefix}_steps"] = int(build_model(BY_TAG[tag].model_args).n_steps)
        # --fast runs see a random subset of the test fold, the other
        # blocks' final runs the whole fold: they cannot be paired then.
        # When both selections coincide the comparisons are made once.
        if fast or (prefix == "attnfinal" and tag == selected):
            continue
        name = "" if prefix == "final" else "attn_"
        if trivial is not None and trivial.get("final"):
            cx.compare_finals(recs, {r["config"]["seed"]: trivial
                                     for r in recs}, name + "trivial",
                              n_boot, s)
        if cnn:
            cx.compare_finals(recs, cnn, name + "cnn", n_boot, s)
        if dense:
            cx.compare_finals(recs, dense, name + "dense", n_boot, s)
    if selected is not None:
        s["attn_final_is_final"] = int(selected == selected_attn)
        if selected != selected_attn and not fast \
                and by_tag.get(selected) and by_tag.get(selected_attn):
            others = {r["config"]["seed"]: r for r in by_tag[selected]}
            cx.compare_finals(by_tag[selected_attn], others, "attn_final",
                              n_boot, s)
    if trivial is not None:
        s["trivial_val_macro_auroc"] = trivial["scalars"]["val_macro_auroc"]

    # Interpretation (first seed): sensitivity of every configuration,
    # attention statistics of every attention configuration.
    profiles = {}
    if cache is not None:
        rows = sensitivity_rows(cache)
        s["sens_n_records"] = int(len(rows))
        for ab in ABLATIONS:
            rec = runs.get((ab.tag, seeds[0]))
            if rec is None:
                continue
            try:
                model = load_model(rec)
            except FileNotFoundError as e:
                print(f"interpretation of {ab.tag} skipped: {e}")
                continue
            print(f"input sensitivity of {ab.tag} ...")
            X = normalised(cache, rows, rec)
            profile = input_sensitivity(model, X)
            profiles[ab.tag] = bin_profile(profile)
            s.update(sensitivity_numbers(profile, ab.tag))
            if ab.has_attention:
                from src.dataset import split_indices
                Xv = normalised(cache, split_indices(cache)["val"], rec)
                s.update(attention_numbers(model, Xv, ab.tag))

    return {
        "run_id": SUMMARY_NAME + ("_fast" if fast else ""),
        "schema": runlog.SCHEMA_VERSION,
        "task": "rnn_block_summary",
        "fast": fast,
        "scalars": s,
        "seeds": list(seeds),
        "selected_config": selected,
        "selected_attention_config": selected_attn,
        "selection_rule": "highest mean validation macro AUROC over the "
                          "seeds; the attention model among "
                          + ", ".join(ATTENTION_TAGS),
        "baseline_args": BASE_ARGS,
        "grad_clip": GRAD_CLIP,
        "ablations": [{"tag": a.tag, "factor": a.factor, "label": a.label,
                       "model_args": a.model_args, "reference": a.reference,
                       "steps": build_model(a.model_args).n_steps}
                      for a in ABLATIONS],
        "sensitivity_profiles": {"bin_s": BIN_S, "profiles": profiles},
        "trivial_run": None if trivial is None else trivial["run_id"],
        "cnn_summary": cnn_summary,
        "cnn_final_runs": sorted(r["run_id"] for r in cnn.values()),
        "dense_summary": dense_summary,
        "dense_final_runs": sorted(r["run_id"] for r in dense.values()),
        "source_runs": sources,
        "meta": runlog.environment(),
    }


# ---------------------------------------------------------------------------
# 6. LaTeX tables (made of macros only: no number is written here)
# ---------------------------------------------------------------------------
def _m(key: str, alias: str = ALIAS) -> str:
    """LaTeX macro of a summary key, e.g. val_auroc_mean_base ->
    \\rnnValAurocMeanBase{}."""
    return "\\" + dx.make_numbers_module().macro_name(alias, key) + "{}"


def _tex(text: str) -> str:
    return text.replace(" x ", " $\\times$ ").replace("tanh(w.h + b)",
                                                      "$\\tanh(w^\\top h+b)$")


def ablation_table(summary: dict) -> str:
    s = summary["scalars"]
    lines = [
        "% Generated by src/rnn_experiments.py from results/logs/"
        f"{summary['run_id']}.json.",
        "% Needs report/numbers.tex (alias \"rnn\" in report/runs.json)"
        " and \\usepackage{booktabs}.",
        "% Delta: paired difference with the reference trained with the same"
        " seed (the baseline unless the label says otherwise).",
        "% Last 2 s: share of the input sensitivity in the last 2 s of the"
        " record (seed 42; 0.2 if uniform).",
        "\\begin{tabular}{llrrcrcc}",
        "\\toprule",
        "Factor & Configuration & Parameters & Steps & "
        "Val.\\ macro AUROC & $\\Delta$ vs.\\ reference & "
        "Seeds better / worse & Last 2\\,s \\\\",
        "\\midrule",
    ]
    previous = None
    for ab in ABLATIONS:
        t = ab.tag
        if f"val_auroc_mean_{t}" not in s:
            continue
        if previous is not None and ab.factor != previous:
            lines.append("\\addlinespace")
        previous = ab.factor
        auroc = f"{_m('val_auroc_mean_' + t)} $\\pm$ {_m('val_auroc_sd_' + t)}"
        if t == "base" or f"diff_mean_{t}" not in s:
            diff, seeds = "---", "---"
        else:
            diff = _m(f"diff_mean_{t}")
            seeds = (f"{_m('n_seeds_better_' + t)} / {_m('n_seeds_worse_' + t)}"
                     f" of {_m('n_seeds_compared_' + t)}")
        last = _m(f"sens_last_{t}") if f"sens_last_{t}" in s else "---"
        lines.append(f"{_tex(ab.factor)} & {_tex(ab.label)} & "
                     f"{_m('params_' + t)} & {_m('steps_' + t)} & {auroc} & "
                     f"{diff} & {seeds} & {last} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(lines)


def attention_table(summary: dict) -> str | None:
    s = summary["scalars"]
    tags = [t for t in ATTENTION_TAGS if f"attn_entropy_norm_{t}" in s]
    if not tags:
        return None
    lines = [
        "% Generated by src/rnn_experiments.py from results/logs/"
        f"{summary['run_id']}.json.",
        "% Attention weights of the seed-42 model of each configuration over"
        " the whole validation fold.",
        "% Entropy: H(alpha) / log T (1 = uniform weights). Spearman:"
        " correlation of the weights with the step's signal activity.",
        "\\begin{tabular}{lrcccc}",
        "\\toprule",
        "Configuration & Steps $T$ & Uniform $1/T$ & Mean largest weight & "
        "Norm.\\ entropy (eff.\\ steps) & Spearman with activity \\\\",
        "\\midrule",
    ]
    for t in tags:
        largest = _m(f"attn_max_weight_{t}")
        if f"attn_max_bound_{t}" in s:
            largest += f" (bound {_m('attn_max_bound_' + t)})"
        lines.append(
            f"{_tex(BY_TAG[t].label)} & {_m('attn_steps_' + t)} & "
            f"{_m('attn_uniform_' + t)} & {largest} & "
            f"{_m('attn_entropy_norm_' + t)} "
            f"({_m('attn_effective_steps_' + t)}) & "
            f"{_m('attn_spearman_activity_' + t)} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(lines)


def final_table(summary: dict) -> str | None:
    s = summary["scalars"]
    if "final_test_macro_auroc_mean" not in s:
        return None
    two = s.get("attn_final_is_final") == 0 and \
        "attnfinal_test_macro_auroc_mean" in s
    with_cnn = summary.get("cnn_summary") is not None
    with_dense = summary.get("dense_summary") is not None
    cols = "lcc" + "c" * (two + with_cnn + with_dense) + "c"
    head = [" & \\multicolumn{2}{c}{RNN (final)}"]
    sub = ["Test AUROC & First seed [95\\% CI] & Mean $\\pm$ SD (seeds)"]
    if two:
        head.append("RNN + attention (final)")
        sub.append("Mean $\\pm$ SD")
    if with_cnn:
        head.append("CNN (final)")
        sub.append("Mean $\\pm$ SD")
    if with_dense:
        head.append("Dense (final)")
        sub.append("Mean $\\pm$ SD")
    head.append("Trivial")
    sub.append("")
    lines = [
        "% Generated by src/rnn_experiments.py from results/logs/"
        f"{summary['run_id']}.json.",
        "% Needs report/numbers.tex (aliases \"rnn\", \"trivial\""
        + (", \"cnn\"" if with_cnn else "")
        + (", \"dense\"" if with_dense else "") + ").",
        f"\\begin{{tabular}}{{{cols}}}",
        "\\toprule",
        " & ".join(head) + " \\\\",
        "\\cmidrule(lr){2-3}",
        " & ".join(sub) + " \\\\",
        "\\midrule",
    ]
    rows = [("Macro", "test_macro_auroc")] + \
           [(c, f"test_auroc_{c}") for c in config.CLASSES]
    for name, key in rows:
        first = (f"{_m('final_first_' + key)} "
                 f"[{_m('final_first_' + key + '_lo')}, "
                 f"{_m('final_first_' + key + '_hi')}]")
        mean_key = ("final_test_macro_auroc" if key == "test_macro_auroc"
                    else f"final_{key}")

        def spread(alias, mk=mean_key):
            return f"{_m(mk + '_mean', alias)} $\\pm$ {_m(mk + '_sd', alias)}"

        cells = [name, first, spread(ALIAS)]
        if two:
            ak = "attn" + mean_key
            cells.append(f"{_m(ak + '_mean')} $\\pm$ {_m(ak + '_sd')}")
        if with_cnn:
            cells.append(spread("cnn"))
        if with_dense:
            cells.append(spread("dense"))
        cells.append(_m(key, "trivial"))
        lines.append(" & ".join(cells) + " \\\\")
        if name == "Macro":
            lines.append("\\midrule")
    # Paired differences in test macro AUROC, one row per seed.
    n_cols = len(cols)
    extra = []
    comparisons = [("cnn", "RNN vs.\\ CNN"), ("dense", "RNN vs.\\ Dense"),
                   ("trivial", "RNN vs.\\ trivial")]
    if two:
        comparisons += [("attn_final", "RNN + att.\\ vs.\\ RNN"),
                        ("attn_cnn", "RNN + att.\\ vs.\\ CNN")]
    for other, title in comparisons:
        for seed in summary["seeds"]:
            key = f"diff_{other}_s{seed}"
            if key not in s:
                continue
            extra.append(
                f"$\\Delta$ macro {title}, seed {seed} & "
                f"{_m(key)} [{_m(f'diff_{other}_lo_s{seed}')}, "
                f"{_m(f'diff_{other}_hi_s{seed}')}]"
                + " &" * (n_cols - 2) + " \\\\")
    if extra:
        lines.append("\\midrule")
        lines += extra
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 7. Figures
# ---------------------------------------------------------------------------
def _short(tag: str) -> str:
    return BY_TAG[tag].label.split(":")[0].split(" (vs.")[0]


def figure_curves(runs: dict, paths: Paths, suffix: str,
                  seed: int = SEEDS[0]) -> bool:
    """(a) baseline losses per epoch; (b) validation macro AUROC per epoch
    of the recurrent cell, depth and direction; (c) TRAINING loss per epoch
    with 100 frames, 125 convolutional steps and 1000 raw steps (how hard
    the network is to optimise as the sequence gets longer)."""
    needed = ["base", "gru", "layers1", "bi", "conv", "raw", "raw_attn"]
    if any((t, seed) not in runs for t in needed):
        print("rnn_curves: some seed-42 runs are missing; figure skipped")
        return False
    import matplotlib.pyplot as plt
    eda = dx._style()
    col, muted = eda.SERIES_COLOURS, eda.INK_MUTED
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))

    h = runs[("base", seed)]["history"]
    best = runs[("base", seed)]["selection"]["best_epoch"]
    axes[0].plot(h["epoch"], h["train_loss"], color=col[0], linewidth=2,
                 label="training loss (training mode)")
    axes[0].plot(h["epoch"], h["val_loss"], color=col[1], linewidth=2,
                 label="validation loss")
    axes[0].axvline(best, color=muted, linewidth=1, linestyle="--")
    axes[0].text(best, axes[0].get_ylim()[1], " best epoch", color=muted,
                 fontsize=8, va="top")
    axes[0].set(title="(a) Baseline (LSTM)", xlabel="epoch", ylabel="BCE loss")
    axes[0].legend(loc="upper right", fontsize=8)

    for i, tag in enumerate(["base", "gru", "layers1", "bi"]):
        hh = runs[(tag, seed)]["history"]
        axes[1].plot(hh["epoch"], hh["val_macro_auroc"], color=col[i],
                     linewidth=2, label=_short(tag))
    axes[1].set(title="(b) Validation AUROC", xlabel="epoch",
                ylabel="validation macro AUROC")
    axes[1].legend(loc="lower right", fontsize=7)

    for i, (tag, ls) in enumerate([("base", "-"), ("conv", "-"),
                                   ("raw", "--"), ("raw_attn", "--")]):
        hh = runs[(tag, seed)]["history"]
        axes[2].plot(hh["epoch"], hh["train_loss"], color=col[i],
                     linewidth=2, linestyle=ls,
                     label=f"{_short(tag)}" if tag != "base"
                     else "100 frames (baseline)")
    axes[2].set(title="(c) Sequence length: training loss", xlabel="epoch",
                ylabel="BCE loss (training mode)")
    axes[2].legend(loc="best", fontsize=7)
    fig.suptitle(f"Recurrent networks, seed {seed}", color=eda.INK,
                 fontsize=10)
    dx._save(fig, paths, "rnn_curves" + suffix)
    return True


def figure_ablations(runs: dict, paths: Paths,
                     suffix: str, seeds=SEEDS) -> bool:
    """Validation macro AUROC of every configuration: one small dot per
    seed, a bar for the mean, the baseline mean as a reference line."""
    scores = {ab.tag: [runs[(ab.tag, s)]["scalars"]["val_macro_auroc"]
                       for s in seeds if (ab.tag, s) in runs]
              for ab in ABLATIONS}
    scores = {t: v for t, v in scores.items() if v}
    if "base" not in scores:
        print("rnn_ablations: no baseline run; figure skipped")
        return False
    import matplotlib.pyplot as plt
    eda = dx._style()
    # One colour per group of factors, in the fixed order of the palette.
    groups = {"cell": 0, "depth": 0, "direction": 0, "readout": 1,
              "attention score": 1, "attention x cell": 1,
              "attention x direction": 1, "input": 2, "sequence length": 3,
              "attention x sequence length": 3}
    colour = {f: eda.SERIES_COLOURS[i] for f, i in groups.items()}
    colour["baseline"] = eda.INK
    order = [ab for ab in ABLATIONS if ab.tag in scores]
    fig, ax = plt.subplots(figsize=(7.5, 0.32 * len(order) + 1.2))
    y = np.arange(len(order))[::-1]
    for yi, ab in zip(y, order):
        v = scores[ab.tag]
        c = colour[ab.factor]
        ax.plot([np.mean(v)] * 2, [yi - 0.28, yi + 0.28], color=c,
                linewidth=2.5, solid_capstyle="round")
        ax.scatter(v, [yi] * len(v), s=22, color=c, alpha=0.8,
                   edgecolors="white", linewidths=0.8, zorder=3)
    base_mean = float(np.mean(scores["base"]))
    ax.axvline(base_mean, color=eda.INK_MUTED, linewidth=1, linestyle="--")
    ax.text(base_mean, y[0] + 0.7, "baseline mean", color=eda.INK_MUTED,
            fontsize=8, ha="center")
    ax.set_yticks(y, [ab.label.split(":")[0] for ab in order], fontsize=8)
    ax.set_ylim(y[-1] - 1.2, y[0] + 1.1)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(f"validation macro AUROC (dots: seeds "
                  f"{', '.join(map(str, seeds))}; bar: mean)")
    dx._save(fig, paths, "rnn_ablations" + suffix)
    return True


def figure_time_gradient(summary: dict, paths: Paths, suffix: str) -> bool:
    """Input sensitivity over the 10 s of the record (seed-42 models), in
    bins of 0.1 s: (a) how the sequence of states is read (last state,
    bidirectional, mean, attention); (b) what the sequence is made of (100
    frames, 125 convolutional steps, 1000 raw samples, with and without
    attention). A uniform profile would be the dashed line."""
    prof = summary.get("sensitivity_profiles", {}).get("profiles", {})
    panels = [("(a) Readout and direction", ["base", "bi", "mean", "attn"]),
              ("(b) Input sequence", ["base", "conv", "raw", "raw_attn"])]
    if not all(t in prof for _, tags in panels for t in tags):
        print("rnn_time_gradient: some profiles are missing; figure skipped")
        return False
    import matplotlib.pyplot as plt
    eda = dx._style()
    col = eda.SERIES_COLOURS
    bin_s = summary["sensitivity_profiles"]["bin_s"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.3))   # own y scales
    for ax, (title, tags) in zip(axes, panels):
        for i, tag in enumerate(tags):
            p = np.asarray(prof[tag])
            t = (np.arange(len(p)) + 0.5) * bin_s
            name = "LSTM, last state (baseline)" if tag == "base" \
                else _short(tag)
            ax.plot(t, p, color=col[i], linewidth=1.6, label=name)
        uniform = 1.0 / len(p)
        ax.axhline(uniform, color=eda.INK_MUTED, linewidth=1, linestyle="--")
        ax.text(0.05, uniform, " uniform", color=eda.INK_MUTED, fontsize=8,
                va="bottom")
        ax.set(title=title, xlabel="time in the record (s)")
        ax.legend(loc="upper left", fontsize=7)
    for ax in axes:
        ax.set_ylabel(f"share of |d logit / d input| per {bin_s:g} s")
    n = summary["scalars"].get("sens_n_records")
    fig.suptitle("Which part of the 10 s the output depends on (seed 42"
                 + (f", {n} validation records" if n else "") + ")",
                 color=eda.INK, fontsize=10)
    dx._save(fig, paths, "rnn_time_gradient" + suffix)
    return True


def figure_attention(runs: dict, finals: dict, summary: dict, paths: Paths,
                     suffix: str, cache: dict | None) -> bool:
    """Attention weights over lead II of one validation record per
    superclass (the same records as the CNN feature maps, D16). Row 1: the
    final RNN + attention model (first seed); row 2: the same records with
    the other score function (seed-42 ablation run), to compare how
    concentrated the weights are. Bars: alpha_t of each step (right axis);
    dashed line: uniform weight 1/T."""
    tag = summary.get("selected_attention_config")
    if cache is None or tag is None:
        print("rnn_attention: no final attention model or no cache; skipped")
        return False
    recs = sorted((r for (t, _), r in finals.items() if t == tag),
                  key=lambda r: r["config"]["seed"])
    other = "attn" if BY_TAG[tag].model_args["attention"] == "notebook" \
        else "attn_tanh"
    if not recs or (other, SEEDS[0]) not in runs:
        print("rnn_attention: runs missing; figure skipped")
        return False
    examples = cx.example_records(cache)
    if not examples:
        print("rnn_attention: no single-class validation record; skipped")
        return False
    classes = list(examples)
    rows_cache = [examples[c] for c in classes]
    models = [(recs[0], f"final {tag}, seed {recs[0]['config']['seed']}"),
              (runs[(other, SEEDS[0])], f"{other}, seed {SEEDS[0]}")]

    import matplotlib.pyplot as plt
    eda = dx._style()
    lead2 = config.LEADS.index("II")
    fig, axes = plt.subplots(len(models), len(classes),
                             figsize=(2.4 * len(classes) + 0.6, 2.3 * len(models) + 0.5),
                             squeeze=False, sharex=True)
    for i, (rec, name) in enumerate(models):
        model = load_model(rec)
        X = normalised(cache, rows_cache, rec)
        alpha = attention_weights(model, X)
        logits = model(torch.from_numpy(X)).detach().numpy()
        prob = 1 / (1 + np.exp(-logits))                   # sigmoid, outside
        step = model.step_samples / config.SAMPLING_RATE
        T = alpha.shape[1]
        top = float(alpha.max()) * 1.1
        t = np.arange(X.shape[2]) / config.SAMPLING_RATE
        for j, c in enumerate(classes):
            ax = axes[i, j]
            ax.plot(t, X[j, lead2], color=eda.INK, linewidth=0.6)
            ax.set_yticks([])
            ax.grid(False)
            ax2 = ax.twinx()
            ax2.bar((np.arange(T) + 0.5) * step, alpha[j], width=step,
                    color=eda.CLASS_COLOURS[c], alpha=0.45, linewidth=0)
            ax2.axhline(1.0 / T, color=eda.INK_MUTED, linewidth=0.8,
                        linestyle="--")
            ax2.set_ylim(0, top)
            ax2.tick_params(axis="y", labelsize=6)
            ax2.grid(False)
            ax2.set_zorder(ax.get_zorder() - 1)   # signal drawn on top
            ax.patch.set_visible(False)
            k = config.CLASSES.index(c)
            thr = rec["scalars"].get(f"threshold_{c}")
            verdict = "" if thr is None else (
                " > " if prob[j, k] > thr else " ≤ ") + f"thr. {thr:.2f}"
            ax.set_title(f"{c} (ecg_id {int(cache['ecg_id'][rows_cache[j]])})\n"
                         f"p({c}) = {prob[j, k]:.2f}{verdict}", fontsize=8)
            if j == len(classes) - 1:
                ax2.set_ylabel("attention weight", fontsize=7)
        axes[i, 0].set_ylabel(f"{name}\nlead II (z-score)", fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("time (s)")
    fig.suptitle("Attention weights over time (bars) on lead II, one "
                 "validation record per superclass; dashed: uniform 1/T",
                 color=eda.INK, fontsize=10)
    dx._save(fig, paths, "rnn_attention" + suffix)
    return True


# ---------------------------------------------------------------------------
# 8. report: summary log + tables + figures
# ---------------------------------------------------------------------------
def write_report(fast: bool = False, paths: Paths = Paths(), seeds=SEEDS,
                 cache: dict | None = None) -> dict:
    suffix = "_fast" if fast else ""
    cache = dx._load_cache(cache)     # signals for the interpretation
    summary = build_summary(paths, fast, seeds, cache)
    path = runlog.write_run(summary, paths.logs)
    print(f"summary: {rel(path)} ({len(summary['scalars'])} numbers)")

    paths.tables.mkdir(parents=True, exist_ok=True)
    (paths.tables / f"rnn_ablations{suffix}.tex").write_text(
        ablation_table(summary), encoding="utf-8")
    for name, text in (("rnn_attention", attention_table(summary)),
                       ("rnn_final", final_table(summary))):
        if text is not None:
            (paths.tables / f"{name}{suffix}.tex").write_text(
                text, encoding="utf-8")
    print(f"tables: {rel(paths.tables)}/rnn_*{suffix}.tex")

    runs = find_runs(paths, fast=fast, final=False)
    finals = find_runs(paths, fast=fast, final=True)
    figure_curves(runs, paths, suffix, seeds[0])
    figure_ablations(runs, paths, suffix, seeds)
    figure_time_gradient(summary, paths, suffix)
    figure_attention(runs, finals, summary, paths, suffix, cache)
    print(f"figures: {rel(paths.figures)}/rnn_*{suffix}.pdf|png")

    if not fast:
        _register_alias(paths)
    return summary


def _register_alias(paths: Paths) -> None:
    """Add "rnn": "rnn_summary" to report/runs.json if it is missing."""
    with open(paths.runs_file, encoding="utf-8") as f:
        runs = json.load(f)
    if runs.get(ALIAS) != SUMMARY_NAME:
        runs[ALIAS] = SUMMARY_NAME
        with open(paths.runs_file, "w", encoding="utf-8") as f:
            json.dump(runs, f, indent=2)
            f.write("\n")
        print(f'added "{ALIAS}": "{SUMMARY_NAME}" to {rel(paths.runs_file)}')
    print("next: python report/make_numbers.py")


# ---------------------------------------------------------------------------
def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("step", choices=["estimate", "run", "final", "report",
                                    "all"])
    p.add_argument("--fast", action="store_true",
                   help="smoke test: small subset, 2 epochs, outputs *_fast")
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    p.add_argument("--only", nargs="+", default=None,
                   help="run only these configurations (step run)")
    a = p.parse_args(argv)
    t0 = datetime.now()
    cache = dx._load_cache(None)
    if a.step == "estimate":
        estimate(cache, seeds=a.seeds, fast=a.fast)
    if a.step in ("run", "all"):
        run_ablations(a.seeds, a.only, a.fast, cache)
    if a.step in ("final", "all"):
        run_final(a.fast, cache, seeds=a.seeds)
    if a.step in ("report", "all"):
        write_report(a.fast, seeds=a.seeds, cache=cache)
    print(f"\ndone in {datetime.now() - t0}")


if __name__ == "__main__":
    main()
