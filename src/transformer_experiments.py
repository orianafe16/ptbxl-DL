"""Self-attention and Transformer block (T12): ablations, final model.

Decisions: docs/DECISIONS.md D18 (this block), D3, D4 and D14 (protocol).
Same structure and protocol as src/rnn_experiments.py (D17).

    python -m src.transformer_experiments estimate   # time per epoch and GPU
                                                     # memory, before the runs
    python -m src.transformer_experiments all --fast # smoke test of everything
    python -m src.transformer_experiments run        # 15 configurations x 3
                                                     # seeds, validation only
    python -m src.transformer_experiments final      # selected model -> test
    python -m src.transformer_experiments report     # summary, tables, figures

`run` and `final` can be stopped and started again: a configuration and
seed that already has a (non --fast) log is skipped.

How an ablation works: the baseline is a Pre-LN Transformer encoder with 2
blocks (d_model 128, 4 heads) over 100 patches of 100 ms, with a sinusoidal
positional encoding and mean pooling. Each other configuration changes ONE
factor and keeps everything else equal - same data, split, loss, optimiser,
learning rate, batch size, early stopping, gradient clipping and seeds - and
is compared with a reference trained with the same seed: the baseline, or,
in the step-by-step study of self-attention, the next step of the study
(sa_single -> mhsa -> no_ffn -> layers1 -> base: one head, several heads,
Add & Norm, FFN, a second block).

One final model is chosen on validation before the test fold is touched
(D18): the best of the fifteen configurations.

Outputs of `report`:
    results/logs/transformer_summary.json  numbers for make_numbers.py
                                           (alias "transformer" in runs.json)
    report/tables/transformer_ablations.tex  ablation table (macros only)
    report/tables/transformer_attention.tex  attention statistics
    report/tables/transformer_final.tex      test results of the final model,
                                             with the final CNN, RNN, RNN +
                                             attention, Dense and trivial
    report/figures/transformer_curves.*        learning curves (seed 42)
    report/figures/transformer_ablations.*     validation AUROC per config
    report/figures/transformer_time_gradient.* which part of the 10 s the
                                               output depends on
    report/figures/transformer_attention.*     attention received by each
                                               patch, one record per class
    report/figures/transformer_attention_maps.* full T x T attention maps
With --fast every output name ends in _fast and git ignores it.
"""
import os

# Attention is made of matrix products, run by cuBLAS on the GPU; this
# variable makes them reproducible (as in D17). It must be set before CUDA
# starts. Importing src.rnn_experiments sets it too.
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
from src import config, metrics, runlog  # noqa: E402
from src import dense_experiments as dx  # noqa: E402
from src import rnn_experiments as rx  # noqa: E402
from src.dense_experiments import Paths  # noqa: E402
from src.models.transformer import TransformerClassifier  # noqa: E402
from src.train import (TrainConfig, ptbxl_setup, rel, run_experiment,  # noqa: E402
                       set_seed)

MODEL = "transformer"
SEEDS = (42, 43, 44)                  # D14: seeds of the final models
ALIAS = "transformer"                 # alias of the summary in runs.json
SUMMARY_NAME = "transformer_summary"
GRAD_CLIP = 1.0                       # max total gradient norm (D18, as D17)
EARLY_EPOCH = cx.EARLY_EPOCH          # training loss compared at epoch 5

# Baseline (D18). Every ablation starts from these values.
BASE_ARGS = {"patch": 10, "d_model": 128, "num_heads": 4, "num_layers": 2,
             "d_ff": 512, "ffn": True, "residual": True, "norm": "pre",
             "out_proj": True, "pos_encoding": "sinusoidal",
             "readout": "mean", "dropout": 0.1, "head_dropout": 0.5}


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


# Self-attention step by step (D18): each step adds one component, in the
# order of the class notebooks 01 -> 02 -> 03/05.
NO_BLOCK = {"num_layers": 1, "norm": "none", "residual": False, "ffn": False}
ABLATIONS = [
    Ablation("base", "baseline",
             "Baseline: Pre-LN, 2 blocks, 4 heads, 100 patches, mean"),
    Ablation("sa_single", "step: heads",
             "One self-attention layer, 1 head (vs. 4 heads + W_O)",
             {**NO_BLOCK, "num_heads": 1, "out_proj": False},
             reference="mhsa"),
    Ablation("mhsa", "step: Add & Norm",
             "Multi-head attention only (vs. + Add & Norm)", NO_BLOCK,
             reference="no_ffn"),
    Ablation("no_ffn", "step: FFN", "Attention + Add & Norm (vs. + FFN)",
             {"num_layers": 1, "ffn": False}, reference="layers1"),
    Ablation("layers1", "depth", "1 Transformer block", {"num_layers": 1}),
    Ablation("layers4", "depth", "4 Transformer blocks", {"num_layers": 4}),
    Ablation("heads1", "heads", "1 head (128 dimensions)", {"num_heads": 1}),
    Ablation("heads8", "heads", "8 heads (16 dimensions)", {"num_heads": 8}),
    Ablation("no_pos", "positional encoding",
             "No positional encoding (bag of patches)",
             {"pos_encoding": "none"}),
    Ablation("learned_pos", "positional encoding",
             "Learned position embeddings", {"pos_encoding": "learned"}),
    Ablation("patch25", "patch length", "Patches of 25 samples (40 tokens)",
             {"patch": 25}),
    Ablation("patch50", "patch length", "Patches of 50 samples (20 tokens)",
             {"patch": 50}),
    Ablation("cls", "readout", "CLS token instead of the mean",
             {"readout": "cls"}),
    Ablation("post_ln", "LayerNorm placement", "Post-LN (original Transformer)",
             {"norm": "post"}),
    # The longest sequence (200 tokens, attention cost x 4) goes last, so a
    # long `run` has done everything else first.
    Ablation("patch5", "patch length", "Patches of 5 samples (200 tokens)",
             {"patch": 5}),
]
BY_TAG = {a.tag: a for a in ABLATIONS}
STEP_TAGS = ["sa_single", "mhsa", "no_ffn", "layers1", "base"]
# Extra paired comparisons beyond each configuration's own reference:
# (configuration, other, key name, description).
EXTRA_PAIRS = [("sa_single", "layers1", "sa_vs_block",
                "One self-attention layer (sa\\_single) vs.\\ one complete "
                "block (layers1)")]

# Interpretation (D18).
ATTN_BATCH = 64                       # records per batch for attention maps


# ---------------------------------------------------------------------------
# 1. Configurations and runs
# ---------------------------------------------------------------------------
def make_config(ab: Ablation, seed: int, fast: bool = False,
                final: bool = False) -> TrainConfig:
    """TrainConfig of one ablation: the D14 defaults (as D15-D17) plus
    gradient clipping at norm GRAD_CLIP for every configuration (D18, as
    D17). Input z-score (D3) always on."""
    cfg = TrainConfig(model=MODEL, tag=ab.tag, seed=seed,
                      model_args=ab.model_args, normalize=True,
                      grad_clip=GRAD_CLIP, fast=fast, final=final,
                      notes=ab.label)
    if fast:                          # D14: 2 epochs, 50 resamples
        cfg.max_epochs, cfg.n_boot = 2, 50
    return cfg


def build_model(model_args: dict) -> TransformerClassifier:
    return TransformerClassifier(**model_args)


def find_runs(paths: Paths, fast: bool, final: bool) -> dict:
    """(tag, seed) -> log of the latest matching Transformer run."""
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
# 2. Selection of the final configuration (validation only, D3)
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


def select_final(runs: dict, seeds=SEEDS) -> str:
    """The configuration with the highest mean validation macro AUROC over
    the seeds, among all fifteen (D18, as D15-D17). The test fold plays no
    part in this choice."""
    scores = val_scores(runs, seeds)
    if not scores:
        raise RuntimeError("no configuration has been run with every seed "
                           f"{tuple(seeds)}: run `run` first")
    missing = [a.tag for a in ABLATIONS if a.tag not in scores]
    if missing:
        raise RuntimeError(f"configurations not run with every seed: "
                           f"{missing}. Selecting now would ignore them.")
    return max(scores, key=lambda t: float(np.mean(scores[t])))


def run_final(fast: bool = False, cache: dict | None = None,
              paths: Paths = Paths(), seeds=SEEDS) -> str:
    """Train the selected configuration again with --final for each seed:
    same training, plus one evaluation of the test fold (D3, D14)."""
    tag = select_final(find_runs(paths, fast=fast, final=False), seeds)
    print(f"selected on validation: {tag} ({BY_TAG[tag].label})")
    done = find_runs(paths, fast=False, final=True)
    cache = dx._load_cache(cache)
    for seed in seeds:
        if not fast and (tag, seed) in done:
            print(f"final {tag} seed {seed}: already run, skipped")
            continue
        print(f"\nfinal {tag} seed {seed}")
        _train(BY_TAG[tag], seed, fast, True, cache, paths)
    return tag


# ---------------------------------------------------------------------------
# 3. Time and memory estimate on this machine (before the long runs)
# ---------------------------------------------------------------------------
def estimate(cache: dict | None = None, n_steps: int = 10,
             epochs=(15, 25, 35), seeds=SEEDS, fast: bool = False) -> dict:
    """Seconds per epoch and peak GPU memory of every configuration,
    measured on one real batch (as rnn_experiments.estimate). The attention
    matrix grows with the square of the number of tokens (Part I slides
    102-103): the memory column shows whether every configuration fits on
    the laptop's GPU before a long run is started. Nothing is written."""
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
          f"batches of {x.shape[0]}; reading the data once: {t_data:.1f} s")
    print(f"\n{'configuration':<13}{'tokens':>7}{'train ms':>10}{'eval ms':>9}"
          f"{'GPU MB':>8}{'s/epoch':>9}  minutes per run at "
          f"{', '.join(map(str, epochs))} epochs")
    out = {}
    for ab in ABLATIONS:
        set_seed(seeds[0])
        model = build_model(ab.model_args).to(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        t_tr = rx._time_steps(model, x, y, task, True, n_steps, device)
        t_ev = rx._time_steps(model, x, y, task, False, n_steps, device)
        mem = (torch.cuda.max_memory_allocated(device) / 2 ** 20
               if device.type == "cuda" else float("nan"))
        per_epoch = n_train * t_tr + n_val * t_ev + t_data
        out[ab.tag] = per_epoch
        mins = "  ".join(f"{per_epoch * e / 60:6.1f}" for e in epochs)
        print(f"{ab.tag:<13}{model.n_steps:>7}{1000 * t_tr:>10.1f}"
              f"{1000 * t_ev:>9.1f}{mem:>8.0f}{per_epoch:>9.1f}  {mins}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    total = sum(out.values()) * len(seeds)
    print("\nwhole `run` (" + f"{len(ABLATIONS)} configurations x "
          f"{len(seeds)} seeds): " +
          ", ".join(f"{total * e / 3600:.1f} h at {e} epochs"
                    for e in epochs))
    worst = len(seeds) * max(out.values())
    print(f"`final` (1 configuration x {len(seeds)} seeds, slowest case): "
          + ", ".join(f"{worst * e / 3600:.1f} h at {e} epochs"
                      for e in epochs))
    return out


# ---------------------------------------------------------------------------
# 4. Interpretation: patch order, attention statistics and maps
# ---------------------------------------------------------------------------
def load_model(rec: dict) -> TransformerClassifier:
    """The model of a run with its best weights, in eval mode, on the CPU."""
    model = build_model(rec["model"]["args"])
    state = torch.load(config.ROOT / rec["checkpoint"], map_location="cpu",
                       weights_only=True)
    model.load_state_dict(state)
    return model.eval()


def shuffle_patches(X: np.ndarray, patch: int,
                    seed: int = config.SEED) -> np.ndarray:
    """The same signals with their patches in a random order, (N, leads,
    samples). Each record gets its own permutation of its T patches, the
    same for all leads, so every patch stays intact; only the order
    changes."""
    n, c, s = X.shape
    t = s // patch
    perm = np.argsort(np.random.default_rng(seed).random((n, t)), axis=1)
    idx = np.broadcast_to(perm[:, None, :, None], (n, c, t, patch))
    return np.take_along_axis(X.reshape(n, c, t, patch), idx,
                              axis=2).reshape(n, c, s)


@torch.no_grad()
def predict(model: torch.nn.Module, X: np.ndarray,
            batch: int = 256) -> np.ndarray:
    """sigmoid(logits) of the records X, (N, 5), on the CPU in eval mode."""
    model = model.cpu().eval()
    logits = [model(torch.from_numpy(np.ascontiguousarray(X[i:i + batch])))
              for i in range(0, len(X), batch)]
    return metrics.sigmoid(torch.cat(logits).numpy())


def order_numbers(model: TransformerClassifier, X: np.ndarray,
                  y: np.ndarray, tag: str) -> dict:
    """Validation macro AUROC with the patches in their order and shuffled
    (D18). Both are computed here, on the CPU, so the difference does not
    mix CPU and GPU arithmetic."""
    normal = metrics.macro_auroc(y, predict(model, X))
    shuffled = metrics.macro_auroc(
        y, predict(model, shuffle_patches(X, model.patch)))
    return {f"order_val_auroc_{tag}": float(normal),
            f"order_shuffled_auroc_{tag}": float(shuffled),
            f"order_drop_{tag}": float(shuffled - normal)}


def patch_maps(model: TransformerClassifier, x: torch.Tensor) -> torch.Tensor:
    """Attention weights between patch tokens, (batch, layers, heads, T, T).
    With a CLS token its row and column are removed and each row is
    renormalised to sum 1 (D18)."""
    maps = model.attention_maps(x).double()
    p = model.n_prefix
    if p:
        maps = maps[..., p:, p:]
        maps = maps / maps.sum(dim=-1, keepdim=True)
    return maps


def received_attention(model: TransformerClassifier, X: np.ndarray,
                       layer: int = -1) -> np.ndarray:
    """Attention received by each patch, (N, T): the weights of `layer`,
    heads averaged, then averaged over the queries. Each row sums to 1.

    For sa_single (one layer, no skip connection, no FFN, mean readout)
    this is exactly the pooling weight of each patch: the record's vector
    is sum_j abar_j v_j (D18; tests/test_transformer.py)."""
    model = model.cpu().eval()
    out = []
    for i in range(0, len(X), ATTN_BATCH):
        m = patch_maps(model, torch.from_numpy(X[i:i + ATTN_BATCH]))
        out.append(m[:, layer].mean(dim=1).mean(dim=1).numpy())
    return np.concatenate(out)


def distance_matrix(T: int, patch: int) -> np.ndarray:
    """|i - j| between patch centres, in seconds, (T, T)."""
    i = np.arange(T)
    return np.abs(i[:, None] - i[None, :]) * patch / config.SAMPLING_RATE


def _entropy(p: torch.Tensor) -> torch.Tensor:
    return -(p * torch.log(p.clamp_min(1e-12))).sum(dim=-1)


def attention_numbers(model: TransformerClassifier, X: np.ndarray,
                      tag: str) -> dict:
    """How the seed-42 model attends, over the records X (D18), for the
    first and the last layer:
      - mean attention distance (s): sum_j alpha_ij |i - j|, averaged over
        queries, heads and records, beside its value for uniform weights;
      - normalised entropy of each query's weights, H / log T (1 = uniform),
        computed per head, then averaged;
      - share of each query's weight on its own patch (1/T if uniform);
      - normalised entropy and largest value of the attention received by
        each patch (heads averaged, mean over the queries)."""
    model = model.cpu().eval()
    T = model.n_steps
    D = torch.from_numpy(distance_matrix(T, model.patch))
    logT = math.log(T)
    sums: dict = {}
    n = 0
    for i in range(0, len(X), ATTN_BATCH):
        maps = patch_maps(model, torch.from_numpy(X[i:i + ATTN_BATCH]))
        n += maps.shape[0]
        for name, layer in (("first", 0), ("last", -1)):
            A = maps[:, layer]                               # (B, H, T, T)
            recv = A.mean(dim=1).mean(dim=1)                  # (B, T)
            values = {
                "dist": (A * D).sum(-1).mean(dim=(1, 2)),
                "qentropy": (_entropy(A) / logT).mean(dim=(1, 2)),
                "diag": A.diagonal(dim1=-2, dim2=-1).mean(dim=(1, 2)),
                "recv_entropy": _entropy(recv) / logT,
                "recv_max": recv.max(dim=-1).values,
            }
            for k, v in values.items():
                key = f"attn_{k}_{name}_{tag}"
                sums[key] = sums.get(key, 0.0) + float(v.sum())
    out = {k: v / n for k, v in sums.items()}
    out.update({f"attn_tokens_{tag}": int(T),
                f"attn_layers_{tag}": len(model.blocks),
                f"attn_uniform_{tag}": 1.0 / T,
                f"attn_dist_uniform_{tag}": float(distance_matrix(
                    T, model.patch).mean())})
    return out


# ---------------------------------------------------------------------------
# 5. Summary log for make_numbers.py
# ---------------------------------------------------------------------------
def rnn_finals(paths: Paths) -> tuple[dict, dict, dict | None]:
    """seed -> final (test) run of the final RNN and of the final RNN +
    attention selected by the RNN summary (D17), and that summary. When the
    two RNN selections coincide, the second dict is empty."""
    summary = cx._alias_log(paths, "rnn")
    if summary is None or not summary.get("selected_config"):
        return {}, {}, None
    found = cx.find_runs(paths, fast=False, final=True, model="rnn")

    def of(tag):
        return {seed: r for (t, seed), r in found.items() if t == tag}

    best, best_attn = (summary["selected_config"],
                       summary.get("selected_attention_config"))
    attn = of(best_attn) if best_attn and best_attn != best else {}
    return of(best), attn, summary


def build_summary(paths: Paths = Paths(), fast: bool = False,
                  seeds=SEEDS, cache: dict | None = None) -> dict:
    """Collect every number of the block into one log with a flat `scalars`
    block (read by make_numbers.py under the alias "transformer"). With the
    cache, the interpretation numbers are added too."""
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
        s[f"tokens_{t}"] = int(build_model(ab.model_args).n_steps)
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

    # Extra paired comparisons (D18: one self-attention layer vs one block).
    for a_tag, b_tag, name, _ in EXTRA_PAIRS:
        diffs, better, worse = [], 0, 0
        for seed in seeds:
            if (a_tag, seed) not in runs or (b_tag, seed) not in runs:
                continue
            d = dx.paired_diff(runs[(a_tag, seed)], runs[(b_tag, seed)],
                               "val", n_boot)
            diffs.append(d["diff"])
            s[f"pair_{name}_s{seed}"] = d["diff"]
            s[f"pair_{name}_lo_s{seed}"] = d["lower"]
            s[f"pair_{name}_hi_s{seed}"] = d["upper"]
            better += d["lower"] > 0
            worse += d["upper"] < 0
        if diffs:
            s[f"pair_{name}_mean"] = float(np.mean(diffs))
            s[f"pair_{name}_n_better"] = int(better)
            s[f"pair_{name}_n_worse"] = int(worse)
            s[f"pair_{name}_n_compared"] = len(diffs)

    # Final model: the configuration selected on validation, on test.
    selected = None
    by_tag: dict = {}
    for (t, seed), r in sorted(finals.items(), key=lambda kv: kv[0][1]):
        if seed in seeds:
            by_tag.setdefault(t, []).append(r)
    if by_tag:
        selected = select_final(runs, seeds)
        unexpected = set(by_tag) - {selected}
        if unexpected:
            raise RuntimeError(f"final runs of configurations that the "
                               f"selection rule did not choose: {unexpected}")
    trivial = dx._trivial_record(paths)
    cnn, cnn_summary = rx.block_finals(paths, "cnn", "cnn")
    dense, dense_summary = rx.block_finals(paths, "dense", "dense")
    rnn, rnn_attn, rnn_summary = rnn_finals(paths)
    recs = by_tag.get(selected) if selected else None
    if recs:
        sources += [r["run_id"] for r in recs if r["run_id"] not in sources]
        rx.final_numbers(recs, "final", s)
        s["final_tokens"] = int(build_model(BY_TAG[selected].model_args).n_steps)
        # --fast runs see a random subset of the test fold, the other
        # blocks' final runs the whole fold: they cannot be paired then.
        if not fast:
            if trivial is not None and trivial.get("final"):
                cx.compare_finals(recs, {r["config"]["seed"]: trivial
                                         for r in recs}, "trivial", n_boot, s)
            for name, others in (("cnn", cnn), ("rnn", rnn),
                                 ("rnn_attn", rnn_attn), ("dense", dense)):
                if others:
                    cx.compare_finals(recs, others, name, n_boot, s)
    if trivial is not None:
        s["trivial_val_macro_auroc"] = trivial["scalars"]["val_macro_auroc"]

    # Interpretation (first seed of every configuration).
    profiles = {}
    if cache is not None:
        from src.dataset import split_indices
        val = split_indices(cache)["val"]
        y_val = cache["y"][val]
        rows = rx.sensitivity_rows(cache)
        s["sens_n_records"] = int(len(rows))
        s["attn_n_records"] = int(len(val))
        for ab in ABLATIONS:
            rec = runs.get((ab.tag, seeds[0]))
            if rec is None:
                continue
            try:
                model = load_model(rec)
            except FileNotFoundError as e:
                print(f"interpretation of {ab.tag} skipped: {e}")
                continue
            print(f"interpretation of {ab.tag} ...")
            Xv = rx.normalised(cache, val, rec)
            s.update(order_numbers(model, Xv, y_val, ab.tag))
            s.update(attention_numbers(model, Xv, ab.tag))
            profile = rx.input_sensitivity(model, rx.normalised(cache, rows,
                                                                rec))
            profiles[ab.tag] = rx.bin_profile(profile)
            s.update(rx.sensitivity_numbers(profile, ab.tag))

    rnn_scalars = (rnn_summary or {}).get("scalars", {})
    return {
        "run_id": SUMMARY_NAME + ("_fast" if fast else ""),
        "schema": runlog.SCHEMA_VERSION,
        "task": "transformer_block_summary",
        "fast": fast,
        "scalars": s,
        "seeds": list(seeds),
        "selected_config": selected,
        "selection_rule": "highest mean validation macro AUROC over the "
                          "seeds, among all configurations",
        "baseline_args": BASE_ARGS,
        "grad_clip": GRAD_CLIP,
        "ablations": [{"tag": a.tag, "factor": a.factor, "label": a.label,
                       "model_args": a.model_args, "reference": a.reference,
                       "tokens": build_model(a.model_args).n_steps}
                      for a in ABLATIONS],
        "extra_pairs": [{"config": a, "other": b, "name": n}
                        for a, b, n, _ in EXTRA_PAIRS],
        "sensitivity_profiles": {"bin_s": rx.BIN_S, "profiles": profiles},
        "trivial_run": None if trivial is None else trivial["run_id"],
        "cnn_summary": cnn_summary,
        "cnn_final_runs": sorted(r["run_id"] for r in cnn.values()),
        "rnn_summary": None if rnn_summary is None else rnn_summary["run_id"],
        "rnn_final_runs": sorted(r["run_id"] for r in rnn.values()),
        "rnn_attention_final_runs": sorted(r["run_id"]
                                           for r in rnn_attn.values()),
        "rnn_attention_column": bool(rnn_attn) and
        "attnfinal_test_macro_auroc_mean" in rnn_scalars,
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
    \\transformerValAurocMeanBase{}."""
    return "\\" + dx.make_numbers_module().macro_name(alias, key) + "{}"


def _tex(text: str) -> str:
    return (text.replace("&", "\\&").replace("W_O", "$W_O$")
            .replace(" x ", " $\\times$ "))


def ablation_table(summary: dict) -> str:
    s = summary["scalars"]
    lines = [
        "% Generated by src/transformer_experiments.py from results/logs/"
        f"{summary['run_id']}.json.",
        "% Needs report/numbers.tex (alias \"transformer\" in report/runs.json)"
        " and \\usepackage{booktabs}.",
        "% Delta: paired difference with the reference trained with the same"
        " seed (the baseline unless the label says otherwise).",
        "% Shuffled: change in validation macro AUROC when the patches of"
        " every record are shuffled (seed 42 model).",
        "\\begin{tabular}{llrrcrcr}",
        "\\toprule",
        "Factor & Configuration & Parameters & Tokens & "
        "Val.\\ macro AUROC & $\\Delta$ vs.\\ reference & "
        "Seeds better / worse & Shuffled \\\\",
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
        order = _m(f"order_drop_{t}") if f"order_drop_{t}" in s else "---"
        lines.append(f"{_tex(ab.factor)} & {_tex(ab.label)} & "
                     f"{_m('params_' + t)} & {_m('tokens_' + t)} & {auroc} & "
                     f"{diff} & {seeds} & {order} \\\\")
    extra = [p for p in EXTRA_PAIRS if f"pair_{p[2]}_mean" in s]
    if extra:
        lines.append("\\midrule")
    for _, _, name, text in extra:
        lines.append(
            f"\\multicolumn{{5}}{{l}}{{{text}}} & {_m(f'pair_{name}_mean')} & "
            f"{_m(f'pair_{name}_n_better')} / {_m(f'pair_{name}_n_worse')} of "
            f"{_m(f'pair_{name}_n_compared')} & \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(lines)


def attention_table(summary: dict) -> str | None:
    s = summary["scalars"]
    tags = [a.tag for a in ABLATIONS if f"attn_dist_last_{a.tag}" in s]
    if not tags:
        return None
    lines = [
        "% Generated by src/transformer_experiments.py from results/logs/"
        f"{summary['run_id']}.json.",
        "% Seed-42 model of each configuration, whole validation fold, patch"
        " tokens only, heads averaged.",
        "% Distance: mean attention distance in seconds (uniform weights in"
        " brackets). Entropy: H / log T (1 = uniform).",
        "% Own patch: weight of each query on its own patch (1/T if uniform)."
        " Received: entropy of the attention each patch receives.",
        "\\begin{tabular}{lrrcccccc}",
        "\\toprule",
        " & & & \\multicolumn{2}{c}{Distance (s)} & "
        "\\multicolumn{2}{c}{Query entropy} & Own patch & Received \\\\",
        "\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}",
        "Configuration & Layers & Tokens & First layer & Last layer "
        "(uniform) & First & Last & Last (1/$T$) & Last \\\\",
        "\\midrule",
    ]
    for t in tags:
        lines.append(
            f"{_tex(BY_TAG[t].label.split(' (vs.')[0])} & "
            f"{_m('attn_layers_' + t)} & {_m('attn_tokens_' + t)} & "
            f"{_m('attn_dist_first_' + t)} & {_m('attn_dist_last_' + t)} "
            f"({_m('attn_dist_uniform_' + t)}) & "
            f"{_m('attn_qentropy_first_' + t)} & "
            f"{_m('attn_qentropy_last_' + t)} & "
            f"{_m('attn_diag_last_' + t)} ({_m('attn_uniform_' + t)}) & "
            f"{_m('attn_recv_entropy_last_' + t)} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(lines)


def final_table(summary: dict) -> str | None:
    s = summary["scalars"]
    if "final_test_macro_auroc_mean" not in s:
        return None
    others = []                       # (alias, mean-key prefix, column title)
    if summary.get("cnn_summary") is not None:
        others.append(("cnn", "", "CNN (final)"))
    if summary.get("rnn_summary") is not None:
        others.append(("rnn", "", "RNN (final)"))
        if summary.get("rnn_attention_column"):
            others.append(("rnn", "attn", "RNN + att.\\ (final)"))
    if summary.get("dense_summary") is not None:
        others.append(("dense", "", "Dense (final)"))
    cols = "lcc" + "c" * len(others) + "c"
    head = [" & \\multicolumn{2}{c}{Transformer (final)}"] + \
        [title for _, _, title in others] + ["Trivial"]
    sub = ["Test AUROC & First seed [95\\% CI] & Mean $\\pm$ SD (seeds)"] + \
        ["Mean $\\pm$ SD"] * len(others) + [""]
    aliases = ["transformer", "trivial"] + list(dict.fromkeys(
        a for a, _, _ in others))
    lines = [
        "% Generated by src/transformer_experiments.py from results/logs/"
        f"{summary['run_id']}.json.",
        "% Needs report/numbers.tex (aliases "
        + ", ".join(f'"{a}"' for a in aliases) + ").",
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
        cells = [name, first,
                 f"{_m(mean_key + '_mean')} $\\pm$ {_m(mean_key + '_sd')}"]
        for alias, prefix, _ in others:
            k = prefix + mean_key
            cells.append(f"{_m(k + '_mean', alias)} $\\pm$ "
                         f"{_m(k + '_sd', alias)}")
        cells.append(_m(key, "trivial"))
        lines.append(" & ".join(cells) + " \\\\")
        if name == "Macro":
            lines.append("\\midrule")
    # Paired differences in test macro AUROC, one row per seed.
    n_cols = len(cols)
    extra = []
    for other, title in (("cnn", "vs.\\ CNN"), ("rnn", "vs.\\ RNN"),
                         ("rnn_attn", "vs.\\ RNN + att."),
                         ("dense", "vs.\\ Dense"),
                         ("trivial", "vs.\\ trivial")):
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
    of the step-by-step configurations; (c) TRAINING loss per epoch with
    Pre-LN and Post-LN (does Pre-LN make optimisation easier without
    warm-up? Part II slide 36)."""
    needed = STEP_TAGS + ["post_ln"]
    if any((t, seed) not in runs for t in needed):
        print("transformer_curves: some seed-42 runs are missing; skipped")
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
    axes[0].set(title="(a) Baseline", xlabel="epoch", ylabel="BCE loss")
    axes[0].legend(loc="upper right", fontsize=8)

    names = {"sa_single": "1 head, attention only", "mhsa": "+ heads, W_O",
             "no_ffn": "+ Add & Norm", "layers1": "+ FFN (1 block)",
             "base": "2 blocks (baseline)"}
    for i, tag in enumerate(STEP_TAGS):
        hh = runs[(tag, seed)]["history"]
        axes[1].plot(hh["epoch"], hh["val_macro_auroc"], color=col[i],
                     linewidth=2, label=names[tag])
    axes[1].set(title="(b) Self-attention step by step", xlabel="epoch",
                ylabel="validation macro AUROC")
    axes[1].legend(loc="lower right", fontsize=7)

    for i, (tag, name) in enumerate([("base", "Pre-LN (baseline)"),
                                     ("post_ln", "Post-LN")]):
        hh = runs[(tag, seed)]["history"]
        axes[2].plot(hh["epoch"], hh["train_loss"], color=col[i],
                     linewidth=2, label=name)
    axes[2].set(title="(c) LayerNorm placement: training loss",
                xlabel="epoch", ylabel="BCE loss (training mode)")
    axes[2].legend(loc="best", fontsize=7)
    fig.suptitle(f"Transformer encoder, seed {seed}", color=eda.INK,
                 fontsize=10)
    dx._save(fig, paths, "transformer_curves" + suffix)
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
        print("transformer_ablations: no baseline run; figure skipped")
        return False
    import matplotlib.pyplot as plt
    eda = dx._style()
    groups = {"step: heads": 0, "step: Add & Norm": 0, "step: FFN": 0,
              "depth": 1, "heads": 1, "positional encoding": 2,
              "patch length": 3, "readout": 4, "LayerNorm placement": 4}
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
    ax.set_yticks(y, [_short(ab.tag) for ab in order], fontsize=8)
    ax.set_ylim(y[-1] - 1.2, y[0] + 1.1)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(f"validation macro AUROC (dots: seeds "
                  f"{', '.join(map(str, seeds))}; bar: mean)")
    dx._save(fig, paths, "transformer_ablations" + suffix)
    return True


def figure_time_gradient(summary: dict, paths: Paths, suffix: str) -> bool:
    """Input sensitivity over the 10 s of the record (seed-42 models), in
    bins of 0.1 s, as rnn_time_gradient: (a) positional encoding, readout
    and the single self-attention layer; (b) patch length."""
    prof = summary.get("sensitivity_profiles", {}).get("profiles", {})
    panels = [("(a) Position, readout, one layer",
               ["base", "no_pos", "cls", "sa_single"]),
              ("(b) Patch length", ["patch5", "base", "patch25", "patch50"])]
    if not all(t in prof for _, tags in panels for t in tags):
        print("transformer_time_gradient: some profiles are missing; skipped")
        return False
    import matplotlib.pyplot as plt
    eda = dx._style()
    col = eda.SERIES_COLOURS
    bin_s = summary["sensitivity_profiles"]["bin_s"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.3))
    for ax, (title, tags) in zip(axes, panels):
        for i, tag in enumerate(tags):
            p = np.asarray(prof[tag])
            t = (np.arange(len(p)) + 0.5) * bin_s
            name = ("Baseline (patches of 10)" if tag == "base"
                    else _short(tag))
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
    dx._save(fig, paths, "transformer_time_gradient" + suffix)
    return True


def _final_first(finals: dict, tag: str | None) -> dict | None:
    recs = sorted((r for (t, _), r in finals.items() if t == tag),
                  key=lambda r: r["config"]["seed"])
    return recs[0] if recs else None


def _probability_title(rec, model, X, j, c, cache, row) -> str:
    logits = model(torch.from_numpy(X[j:j + 1])).detach().numpy()
    k = config.CLASSES.index(c)
    prob = float(metrics.sigmoid(logits)[0, k])
    thr = rec["scalars"].get(f"threshold_{c}")
    verdict = "" if thr is None else (
        " > " if prob > thr else " ≤ ") + f"thr. {thr:.2f}"
    return (f"{c} (ecg_id {int(cache['ecg_id'][row])})\n"
            f"p({c}) = {prob:.2f}{verdict}")


def figure_attention(runs: dict, finals: dict, summary: dict, paths: Paths,
                     suffix: str, cache: dict | None) -> bool:
    """Attention received by each patch (bars, right axis) over lead II of
    one validation record per superclass (the same records as D16 and D17).
    Row 1: the single self-attention layer (sa_single, seed 42), where these
    weights are exactly the pooling weights; row 2: the last layer of the
    final model (first seed), heads averaged. Dashed line: uniform 1/T."""
    tag = summary.get("selected_config")
    final_rec = _final_first(finals, tag)
    if cache is None or final_rec is None or ("sa_single", SEEDS[0]) not in runs:
        print("transformer_attention: runs or cache missing; figure skipped")
        return False
    examples = cx.example_records(cache)
    if not examples:
        print("transformer_attention: no single-class record; skipped")
        return False
    classes = list(examples)
    rows_cache = [examples[c] for c in classes]
    models = [(runs[("sa_single", SEEDS[0])],
               f"sa_single, seed {SEEDS[0]}"),
              (final_rec, f"final {tag}, seed {final_rec['config']['seed']},"
                          " last layer")]
    import matplotlib.pyplot as plt
    eda = dx._style()
    lead2 = config.LEADS.index("II")
    fig, axes = plt.subplots(len(models), len(classes),
                             figsize=(2.4 * len(classes) + 0.6,
                                      2.3 * len(models) + 0.5),
                             squeeze=False, sharex=True)
    for i, (rec, name) in enumerate(models):
        model = load_model(rec)
        X = rx.normalised(cache, rows_cache, rec)
        abar = received_attention(model, X)
        step = model.patch / config.SAMPLING_RATE
        T = abar.shape[1]
        top = float(abar.max()) * 1.1
        t = np.arange(X.shape[2]) / config.SAMPLING_RATE
        for j, c in enumerate(classes):
            ax = axes[i, j]
            ax.plot(t, X[j, lead2], color=eda.INK, linewidth=0.6)
            ax.set_yticks([])
            ax.grid(False)
            ax2 = ax.twinx()
            ax2.bar((np.arange(T) + 0.5) * step, abar[j], width=step,
                    color=eda.CLASS_COLOURS[c], alpha=0.45, linewidth=0)
            ax2.axhline(1.0 / T, color=eda.INK_MUTED, linewidth=0.8,
                        linestyle="--")
            ax2.set_ylim(0, top)
            ax2.tick_params(axis="y", labelsize=6)
            ax2.grid(False)
            ax2.set_zorder(ax.get_zorder() - 1)   # signal drawn on top
            ax.patch.set_visible(False)
            ax.set_title(_probability_title(rec, model, X, j, c, cache,
                                            rows_cache[j]), fontsize=8)
            if j == len(classes) - 1:
                ax2.set_ylabel("attention received", fontsize=7)
        axes[i, 0].set_ylabel(f"{name}\nlead II (z-score)", fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("time (s)")
    fig.suptitle("Attention received by each patch (bars) on lead II, one "
                 "validation record per superclass; dashed: uniform 1/T",
                 color=eda.INK, fontsize=10)
    dx._save(fig, paths, "transformer_attention" + suffix)
    return True


def figure_attention_maps(runs: dict, finals: dict, summary: dict,
                          paths: Paths, suffix: str,
                          cache: dict | None) -> bool:
    """Full attention maps (row: query time, column: key time, seconds) of
    one validation record (the NORM example of D16-D17, or the first class
    available). Row 1: sa_single; the final model's first and last layer,
    heads averaged; lead II of the record. Row 2: each head of the final
    model's last layer (at most four). Each map has its own colour scale."""
    tag = summary.get("selected_config")
    final_rec = _final_first(finals, tag)
    if cache is None or final_rec is None or ("sa_single", SEEDS[0]) not in runs:
        print("transformer_attention_maps: runs or cache missing; skipped")
        return False
    examples = cx.example_records(cache)
    if not examples:
        print("transformer_attention_maps: no single-class record; skipped")
        return False
    c = "NORM" if "NORM" in examples else next(iter(examples))
    row = examples[c]

    def maps_of(rec):
        model = load_model(rec)
        X = rx.normalised(cache, [row], rec)
        return model, patch_maps(model, torch.from_numpy(X))[0].numpy(), X

    sa_model, sa_maps, X = maps_of(runs[("sa_single", SEEDS[0])])
    fm, f_maps, _ = maps_of(final_rec)
    seconds = config.N_SAMPLES / config.SAMPLING_RATE
    panels_top = [(sa_maps[0, 0], "sa_single (1 head)"),
                  (f_maps[0].mean(axis=0), f"final {tag}: layer 1, heads "
                                           "averaged")]
    if f_maps.shape[0] > 1:
        panels_top.append((f_maps[-1].mean(axis=0),
                           f"final {tag}: layer {f_maps.shape[0]}, heads "
                           "averaged"))
    heads = f_maps[-1][:4]

    import matplotlib.pyplot as plt
    eda = dx._style()
    fig, axes = plt.subplots(2, 4, figsize=(12, 6.4))
    ext = [0, seconds, seconds, 0]

    def show(ax, A, title):
        vmax = float(np.quantile(A, 0.995))
        im = ax.imshow(A, cmap=eda.SEQUENTIAL, vmin=0, vmax=vmax, extent=ext,
                       interpolation="nearest")
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("key time (s)", fontsize=7)
        ax.set_ylabel("query time (s)", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=6)

    for k in range(3):
        if k < len(panels_top):
            show(axes[0, k], *panels_top[k])
        else:
            axes[0, k].axis("off")
    lead2 = config.LEADS.index("II")
    t = np.arange(X.shape[2]) / config.SAMPLING_RATE
    axes[0, 3].plot(t, X[0, lead2], color=eda.INK, linewidth=0.7)
    axes[0, 3].set(xlim=(0, seconds), xlabel="time (s)",
                   title=f"lead II, {c} record "
                         f"(ecg_id {int(cache['ecg_id'][row])})")
    axes[0, 3].title.set_fontsize(8)
    axes[0, 3].set_yticks([])
    for k in range(4):
        if k < len(heads):
            show(axes[1, k], heads[k],
                 f"final {tag}: layer {f_maps.shape[0]}, head {k + 1}")
        else:
            axes[1, k].axis("off")
    fig.suptitle("Self-attention maps of one validation record (row: query "
                 "patch, column: key patch)", color=eda.INK, fontsize=10)
    dx._save(fig, paths, "transformer_attention_maps" + suffix)
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
    (paths.tables / f"transformer_ablations{suffix}.tex").write_text(
        ablation_table(summary), encoding="utf-8")
    for name, text in (("transformer_attention", attention_table(summary)),
                       ("transformer_final", final_table(summary))):
        if text is not None:
            (paths.tables / f"{name}{suffix}.tex").write_text(
                text, encoding="utf-8")
    print(f"tables: {rel(paths.tables)}/transformer_*{suffix}.tex")

    runs = find_runs(paths, fast=fast, final=False)
    finals = find_runs(paths, fast=fast, final=True)
    figure_curves(runs, paths, suffix, seeds[0])
    figure_ablations(runs, paths, suffix, seeds)
    figure_time_gradient(summary, paths, suffix)
    figure_attention(runs, finals, summary, paths, suffix, cache)
    figure_attention_maps(runs, finals, summary, paths, suffix, cache)
    print(f"figures: {rel(paths.figures)}/transformer_*{suffix}.pdf|png")

    if not fast:
        _register_alias(paths)
    return summary


def _register_alias(paths: Paths) -> None:
    """Add "transformer": "transformer_summary" to report/runs.json."""
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
