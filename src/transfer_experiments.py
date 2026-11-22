"""Learning strategies (T13): transfer from the self-supervised ECG-GPT.

Decisions: docs/DECISIONS.md D20 (this block), D7 and D13 (design of the
transfer), D19 (the pre-trained model), D3, D4 and D14 (protocol).

    python -m src.transfer_experiments estimate     # time and GPU memory
    python -m src.transfer_experiments all --fast   # smoke test of everything
    python -m src.transfer_experiments run          # every configuration
    python -m src.transfer_experiments report       # summary, tables, figures

`run` can be stopped and started again: a configuration and seed that
already has a (non --fast) log is skipped. `run --group main` runs only the
six configurations of the main comparison.

What is compared (D7, D20). The ECG-GPT `vocab2560` (selected by the rule
of D19 before any transfer experiment) learnt lead II without labels, by
predicting the next sample. Its network, without the next-token head,
turns a record into one vector (record_features). Three ways to use it for
the 5 superclasses, each with 100 % and 10 % of the training labels:

    scratch   same architecture, random weights, all trainable, lr 1e-3
    probe     pre-trained and frozen; only the new head trains, lr 1e-3
    finetune  pre-trained, all trainable, lr 1e-4

The six are fixed in advance and none is selected from the others, so each
is trained once with --final (validation and test in the same run), seeds
42-44; seed s uses the generator pre-trained with seed s.

Ablations on validation only:
    finetune_lr1e3, scratch_lr1e4   learning rate x initialisation (2 x 2)
    probe_<generator>               the same probe on every other generator
                                    of D19: do better generators (fewer bits
                                    per sample) give better features?

Outputs of `report`:
    results/logs/transfer_summary.json       numbers for make_numbers.py
                                             (alias "transfer" in runs.json)
    report/tables/transfer_main.tex          the six configurations
    report/tables/transfer_comparisons.tex   paired differences
    report/tables/transfer_probe_study.tex   probe on every generator
    report/figures/transfer_curves.*         validation AUROC per epoch
    report/figures/transfer_strategies.*     test AUROC per strategy
    report/figures/transfer_probe_study.*    probe AUROC vs bits per sample
With --fast every output name ends in _fast and git ignores it.
"""
import os

# Attention and LSTM are matrix products on cuBLAS; this makes them
# reproducible, as in D17-D19. It must be set before CUDA starts.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:2")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from datetime import datetime  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src import cnn_experiments as cx  # noqa: E402   generic helpers
from src import config, runlog  # noqa: E402
from src import dense_experiments as dx  # noqa: E402
from src import generative as gx  # noqa: E402
from src import rnn_experiments as rx  # noqa: E402
from src import transfer as tf  # noqa: E402
from src.dense_experiments import Paths  # noqa: E402
from src.tasks import MultiLabelTask  # noqa: E402
from src.train import TrainConfig, rel, run_experiment, set_seed  # noqa: E402

MODEL = "transfer"                    # run names: transfer_<tag>_s<seed>_...
SEEDS = (42, 43, 44)                  # D14
ALIAS = "transfer"                    # alias of the summary in runs.json
SUMMARY_NAME = "transfer_summary"
GRAD_CLIP = 1.0                       # as D17-D19
BATCH = 32                            # records (x 4 windows of 250), D20
HEAD_DROPOUT = 0.5                    # as the heads of D16-D18
LR_DEFAULT = 1e-3                     # D14 (scratch, probe)
LR_FINETUNE = 1e-4                    # D20 (class notebook: smaller rate)
FRACTIONS = (1.0, 0.1)                # labelled share of the training set
STRATEGIES = ("scratch", "probe", "finetune")
STRATEGY_LABELS = {"scratch": "From scratch", "probe": "Linear probe (frozen)",
                   "finetune": "Fine-tuning"}
PRETRAINED = "pretrained"             # the generator selected by D19


def pct(fraction: float) -> int:
    return int(round(100 * fraction))


@dataclass(frozen=True)
class Config:
    tag: str                  # run name part; letters, digits, underscores
    strategy: str             # scratch | probe | finetune
    fraction: float           # share of the training labels
    lr: float
    group: str                # main | lr | probe_study
    label: str                # short English description for the report
    generator: str = PRETRAINED   # generator tag of D19 (probe study)

    @property
    def main(self) -> bool:
        """The six configurations of the main comparison go to the test
        fold (D20); the ablations stay on validation."""
        return self.group == "main"


def _grid(probe_generators=()) -> list[Config]:
    grid = []
    for f in FRACTIONS:
        p = pct(f)
        for s in STRATEGIES:
            lr = LR_FINETUNE if s == "finetune" else LR_DEFAULT
            grid.append(Config(f"{s}_f{p}", s, f, lr, "main",
                               f"{STRATEGY_LABELS[s]}, {p}% labels"))
    for f in FRACTIONS:
        p = pct(f)
        grid.append(Config(f"finetune_lr1e3_f{p}", "finetune", f, LR_DEFAULT,
                           "lr", f"Fine-tuning at lr 1e-3, {p}% labels"))
        grid.append(Config(f"scratch_lr1e4_f{p}", "scratch", f, LR_FINETUNE,
                           "lr", f"From scratch at lr 1e-4, {p}% labels"))
    for g in probe_generators:
        for f in FRACTIONS:
            p = pct(f)
            grid.append(Config(f"probe_{g}_f{p}", "probe", f, LR_DEFAULT,
                               "probe_study", f"Probe on generator {g}, "
                               f"{p}% labels", generator=g))
    return grid


def probe_generators(selected: str = "vocab2560") -> list[str]:
    """Every selectable generator of D19 except the selected one (its probe
    is the main `probe` configuration). no_mask is left out, as in D19."""
    from src import generative_experiments as ge
    return [a.tag for a in ge.ABLATIONS if a.selectable and a.tag != selected]


GRID = _grid(probe_generators())
BY_TAG = {c.tag: c for c in GRID}
MAIN_TAGS = [c.tag for c in GRID if c.main]


# ---------------------------------------------------------------------------
# 1. Pre-trained generators (from chat DL 13)
# ---------------------------------------------------------------------------
class Context:
    """The pre-trained model handed over by chat DL 13 (generative summary,
    key pretrained_for_chat_08), the generator runs of D19 for the probe
    study, and the record vectors already computed (one set per generator
    checkpoint, shared by the 100 % and 10 % probes)."""

    def __init__(self, paths: Paths, fast: bool = False):
        summary = cx._alias_log(paths, "generative")
        if fast:                       # smoke tests use the fast summary
            try:
                summary = runlog.load_run("generative_summary_fast", paths.logs)
            except OSError:
                pass
        if summary is None or "pretrained_for_chat_08" not in summary:
            raise RuntimeError("no generative summary with "
                               "pretrained_for_chat_08: run chat DL 13 first")
        self.summary = summary
        self.handoff = summary["pretrained_for_chat_08"]
        self.selected = self.handoff["selected_config"]
        self.generator_runs = cx.find_runs(paths, fast, False, model="generator")
        self.features: dict = {}

    def generator(self, cfg: Config, seed: int) -> dict:
        """Arguments, checkpoint (None for scratch), tokenizer and run id of
        the generator a configuration uses with this seed."""
        if cfg.generator == PRETRAINED:
            h = self.handoff
            tok = h["tokenizer"]
            info = {"tag": self.selected, "args": dict(h["model_args"]),
                    "checkpoint": h["checkpoints"][str(seed)],
                    "run_id": h["run_ids"][str(seed)],
                    "tokenizer": gx.Tokenizer(int(tok["lo_uv"]),
                                              int(tok["delta_uv"]),
                                              int(tok["vocab"]))}
        else:
            rec = self.generator_runs.get((cfg.generator, seed))
            if rec is None:
                raise RuntimeError(f"no generator run {cfg.generator} seed {seed}")
            info = {"tag": cfg.generator, "args": dict(rec["model"]["args"]),
                    "checkpoint": rec["checkpoint"], "run_id": rec["run_id"],
                    "tokenizer": gx.tokenizer_from_log(rec)}
        if cfg.strategy == "scratch":
            info["checkpoint"], info["run_id"] = None, None
        return info

    def bits(self, tag: str) -> float | None:
        """Mean validation bits per sample of a generator (D19 summary)."""
        return self.summary["scalars"].get(f"bits_mean_{tag}")


def _ckpt(path: str | None):
    return None if path is None else config.ROOT / path   # absolute stays


# ---------------------------------------------------------------------------
# 2. Runs
# ---------------------------------------------------------------------------
def make_config(cfg: Config, seed: int, gen: dict, fast: bool = False,
                final: bool = False) -> TrainConfig:
    """TrainConfig of one configuration: the D14 defaults (Adam, patience 5
    on validation macro AUROC, at most 50 epochs) with the configuration's
    learning rate, gradient clipping at 1.0 (D17-D19) and batches of 32
    records (D20). `normalize` does not apply: the input is tokens."""
    args = {"strategy": cfg.strategy, "label_fraction": cfg.fraction,
            "generator": gen["tag"], "generator_args": gen["args"],
            "generator_run": gen["run_id"], "head_dropout": HEAD_DROPOUT}
    t = TrainConfig(model=MODEL, tag=cfg.tag, seed=seed, model_args=args,
                    lr=cfg.lr, batch_size=BATCH, grad_clip=GRAD_CLIP,
                    normalize=False, fast=fast, final=final, notes=cfg.label)
    if fast:                          # D14: 2 epochs, 50 resamples
        t.max_epochs, t.n_boot = 2, 50
    return t


def find_runs(paths: Paths, fast: bool, final: bool) -> dict:
    """(tag, seed) -> log of the latest matching transfer run."""
    return cx.find_runs(paths, fast, final, model=MODEL)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def probe_features(ctx: Context, data: tf.TransferData, gen: dict,
                   device: torch.device) -> dict:
    """Record vectors of the frozen generator for the three splits,
    computed once per checkpoint (D20)."""
    key = gen["run_id"]
    if key not in ctx.features:
        backbone = tf.load_backbone(gen["args"], _ckpt(gen["checkpoint"]))
        t0 = time.perf_counter()
        ctx.features[key] = {
            "X": {s: tf.extract_features(backbone,
                                         data.tokens(gen["tokenizer"], s),
                                         device)
                  for s in ("train", "val", "test")},
            "n_params_backbone": int(sum(p.numel()
                                         for p in backbone.parameters())),
            "seconds": time.perf_counter() - t0}
        del backbone
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return ctx.features[key]


def setup(cfg: Config, seed: int, data: tf.TransferData, ctx: Context,
          tcfg: TrainConfig, gen: dict, device: torch.device):
    """Loaders, data description, task and model builder of one run."""
    tok = gen["tokenizer"]
    pos = data.train_positions(cfg.fraction, seed)
    info = data.describe(pos, cfg.fraction)
    info.update({"tokenizer": tok.to_dict(), "batch_records": tcfg.batch_size,
                 "cublas_workspace_config":
                     os.environ.get("CUBLAS_WORKSPACE_CONFIG")})
    splits = ["val", "test"] if tcfg.final else ["val"]
    task = MultiLabelTask(config.CLASSES,
                          {s: data.patient_id[s] for s in splits},
                          {s: data.ecg_id[s] for s in splits},
                          n_boot=tcfg.n_boot, seed=config.SEED)
    generator = torch.Generator().manual_seed(seed)   # order of the batches
    pin = torch.cuda.is_available()

    if cfg.strategy == "probe":
        feats = probe_features(ctx, data, gen, device)
        X = feats["X"]
        sets = {"train": tf.FeatureDataset(X["train"][pos],
                                           data.y["train"][pos])}
        sets.update({s: tf.FeatureDataset(X[s], data.y[s]) for s in splits})
        d = X["train"].shape[1]
        info["n_params_backbone"] = feats["n_params_backbone"]
        info["features"] = {"frozen": True, "dim": int(d),
                            "precomputed": True,
                            "extraction_s": feats["seconds"]}

        def build():
            return tf.FeatureHead(d, head_dropout=HEAD_DROPOUT)
    else:
        tokens = {s: data.tokens(tok, s) for s in ["train"] + splits}
        sets = {"train": tf.TokenDataset(tokens["train"][pos],
                                         data.y["train"][pos])}
        sets.update({s: tf.TokenDataset(tokens[s], data.y[s]) for s in splits})
        ckpt = _ckpt(gen["checkpoint"]) if cfg.strategy == "finetune" else None
        info["features"] = {"frozen": False, "precomputed": False}

        def build():
            return tf.TransferClassifier(tf.load_backbone(gen["args"], ckpt),
                                         head_dropout=HEAD_DROPOUT)
    loaders = {s: DataLoader(ds, batch_size=tcfg.batch_size,
                             shuffle=(s == "train"),
                             generator=generator if s == "train" else None,
                             num_workers=0, pin_memory=pin)
               for s, ds in sets.items()}
    return loaders, info, task, build


def _train(cfg: Config, seed: int, fast: bool, data: tf.TransferData,
           ctx: Context, paths: Paths) -> dict:
    gen = ctx.generator(cfg, seed)
    tcfg = make_config(cfg, seed, gen, fast, final=cfg.main)
    device = _device()
    loaders, info, task, build = setup(cfg, seed, data, ctx, tcfg, gen, device)
    _, record = run_experiment(build, loaders, task, tcfg, info,
                               logs_dir=paths.logs,
                               checkpoint_dir=paths.checkpoints,
                               predictions_dir=paths.predictions,
                               device=device)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()      # free the GPU before the next model
    return record


def run_order(tags) -> list[str]:
    """Main comparison first (cheap ones first: probes, then 10 %), then
    the learning-rate ablations, then the probe study, so a long `run` has
    the results the report needs most before the rest."""
    def key(t):
        c = BY_TAG[t]
        group = {"main": 0, "lr": 1, "probe_study": 2}[c.group]
        cost = {"probe": 0, "finetune": 1, "scratch": 1}[c.strategy]
        return (group, cost if c.group == "main" else 0,
                c.fraction if c.group != "probe_study" else 0, GRID.index(c))
    return sorted(tags, key=key)


def load_data(cache: dict | None, fast: bool) -> tf.TransferData:
    return tf.TransferData(dx._load_cache(cache), fast=fast)


def run_all(seeds=SEEDS, only=None, group=None, fast: bool = False,
            cache: dict | None = None, paths: Paths = Paths(),
            data: tf.TransferData | None = None,
            ctx: Context | None = None) -> None:
    tags = only or [c.tag for c in GRID if group is None or c.group == group]
    unknown = set(tags) - set(BY_TAG)
    if unknown:
        raise ValueError(f"unknown configurations: {sorted(unknown)}")
    data = data or load_data(cache, fast)
    ctx = ctx or Context(paths, fast)
    done = {**find_runs(paths, fast=False, final=False),
            **find_runs(paths, fast=False, final=True)}
    todo = [(t, s) for t in run_order(tags) for s in seeds]
    for i, (tag, seed) in enumerate(todo, 1):
        c = BY_TAG[tag]
        if not fast and (tag, seed) in done \
                and bool(done[(tag, seed)].get("final")) == c.main:
            print(f"[{i}/{len(todo)}] {tag} seed {seed}: already run "
                  f"({done[(tag, seed)]['run_id']}), skipped")
            continue
        print(f"\n[{i}/{len(todo)}] {tag} seed {seed}: {c.label}"
              + ("  (+ test fold)" if c.main else ""))
        _train(c, seed, fast, data, ctx, paths)


# ---------------------------------------------------------------------------
# 3. Time and memory estimate on this machine (before the long runs)
# ---------------------------------------------------------------------------
def estimate(cache: dict | None = None, n_steps: int = 5,
             epochs=(10, 20, 30), seeds=SEEDS, fast: bool = False,
             paths: Paths = Paths(), data: tf.TransferData | None = None,
             ctx: Context | None = None) -> dict:
    """Seconds per epoch and peak GPU memory of the trainable network
    (scratch and fine-tuning share the architecture) and the time to
    compute the record vectors of one generator, measured on one real batch
    of BATCH records. Nothing is written."""
    data = data or load_data(cache, fast)
    ctx = ctx or Context(paths, fast)
    device = _device()
    cfg = BY_TAG["finetune_f100"]
    gen = ctx.generator(cfg, seeds[0])
    tcfg = make_config(cfg, seeds[0], gen, fast)
    loaders, _, task, build = setup(cfg, seeds[0], data, ctx, tcfg, gen,
                                    device)
    x, y = next(iter(loaders["train"]))
    x, y = x.to(device), y.to(device)
    set_seed(seeds[0])
    model = build().to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t_tr = rx._time_steps(model, x, y, task, True, n_steps, device)
    t_ev = rx._time_steps(model, x, y, task, False, n_steps, device)
    mem = (torch.cuda.max_memory_allocated(device) / 2 ** 20
           if device.type == "cuda" else float("nan"))
    n_val = len(loaders["val"])
    n_all = sum(len(data.idx[s]) for s in ("train", "val", "test"))
    feat_s = n_all / BATCH * t_ev     # forward only, all records once
    print(f"device {device}; {len(data.idx['train'])} training records; "
          f"batch {BATCH} records = {4 * BATCH} windows")
    print(f"training step {1000 * t_tr:.0f} ms, evaluation {1000 * t_ev:.0f} "
          f"ms per batch; peak GPU memory {mem:.0f} MB")
    out = {}
    for f in FRACTIONS:
        n_tr = math.ceil(len(data.train_positions(f, seeds[0])) / BATCH)
        per_epoch = n_tr * t_tr + n_val * t_ev
        out[f"trainable_f{pct(f)}"] = per_epoch
        mins = "  ".join(f"{per_epoch * e / 60:6.1f}" for e in epochs)
        print(f"scratch / finetune, {pct(f):3d}% labels: {per_epoch:6.1f} s "
              f"per epoch; minutes per run at {epochs} epochs: {mins}")
    out["features_s"] = feat_s
    print(f"record vectors of one generator (all splits): about "
          f"{feat_s:.0f} s; probe heads train in seconds per epoch")
    main = 2 * sum(out[f"trainable_f{pct(f)}"] for f in FRACTIONS) * len(seeds)
    print("\nmain grid (scratch + finetune, 2 fractions, "
          f"{len(seeds)} seeds): " + ", ".join(
              f"{main * e / 3600:.1f} h at {e} epochs" for e in epochs))
    print("learning-rate ablations: about the same again; probe study "
          f"({len(probe_generators(ctx.selected))} generators x "
          f"{len(seeds)} seeds): about "
          f"{len(probe_generators(ctx.selected)) * len(seeds) * feat_s / 60:.0f}"
          " min of feature extraction")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


# ---------------------------------------------------------------------------
# 4. Summary log for make_numbers.py
# ---------------------------------------------------------------------------
def all_runs(paths: Paths, fast: bool) -> dict:
    """(tag, seed) -> run: main configurations from their --final runs,
    ablations from their validation runs."""
    finals = find_runs(paths, fast, final=True)
    vals = find_runs(paths, fast, final=False)
    out = {}
    for c in GRID:
        src = finals if c.main else vals
        out.update({k: r for k, r in src.items() if k[0] == c.tag})
    return out


def recs_of(runs: dict, tag: str, seeds) -> list:
    return [runs[(tag, s)] for s in seeds if (tag, s) in runs]


def split_numbers(recs: list, split: str, t: str, s: dict) -> None:
    """Mean and SD over seeds of one split's metrics, per-class AUROC and
    the first seed's bootstrap intervals, under keys ending in _<t>."""
    if not recs or f"{split}_macro_auroc" not in recs[0]["scalars"]:
        return
    for key, name in (("macro_auroc", "auroc"), ("macro_f1", "f1")):
        vals = [r["scalars"][f"{split}_{key}"] for r in recs]
        s[f"{split}_{name}_mean_{t}"] = float(np.mean(vals))
        if len(vals) > 1:
            s[f"{split}_{name}_sd_{t}"] = dx._sd(vals)
    for c in config.CLASSES:
        vals = [r["scalars"][f"{split}_auroc_{c}"] for r in recs]
        s[f"{split}_auroc_{c}_mean_{t}"] = float(np.mean(vals))
    first = recs[0]["scalars"]
    s[f"{split}_auroc_first_{t}"] = first[f"{split}_macro_auroc"]
    s[f"{split}_auroc_lo_first_{t}"] = first[f"{split}_macro_auroc_lo"]
    s[f"{split}_auroc_hi_first_{t}"] = first[f"{split}_macro_auroc_hi"]


def paired_numbers(name: str, a: list, b: list, split: str, n_boot: int,
                   s: dict) -> None:
    """Paired patient-bootstrap difference A - B per seed (D14) on one
    split; mean over seeds and how many seeds are better / worse. a, b:
    runs of the same seeds (matched by seed)."""
    b_by_seed = {r["config"]["seed"]: r for r in b}
    diffs, better, worse = [], 0, 0
    for r in a:
        seed = r["config"]["seed"]
        other = b_by_seed.get(seed)
        if other is None or f"{split}_macro_auroc" not in other["scalars"] \
                or f"{split}_macro_auroc" not in r["scalars"]:
            continue
        print(f"paired difference {r['run_id']} - {other['run_id']} on {split}")
        d = dx.paired_diff(r, other, split, n_boot)
        diffs.append(d["diff"])
        s[f"{split}_diff_{name}_s{seed}"] = d["diff"]
        s[f"{split}_diff_lo_{name}_s{seed}"] = d["lower"]
        s[f"{split}_diff_hi_{name}_s{seed}"] = d["upper"]
        better += d["lower"] > 0
        worse += d["upper"] < 0
    if diffs:
        s[f"{split}_diff_mean_{name}"] = float(np.mean(diffs))
        s[f"{split}_n_better_{name}"] = int(better)
        s[f"{split}_n_worse_{name}"] = int(worse)
        s[f"{split}_n_compared_{name}"] = len(diffs)


def comparisons() -> list[tuple[str, str, str, str]]:
    """(name, A, B, label): the paired differences reported, all fixed in
    advance (D20). Main pairs on validation and test; ablation pairs on
    validation only."""
    out = []
    for f in FRACTIONS:
        p = pct(f)
        out += [(f"finetune_vs_scratch_f{p}", f"finetune_f{p}", f"scratch_f{p}",
                 f"Fine-tuning $-$ scratch, {p}\\%"),
                (f"probe_vs_scratch_f{p}", f"probe_f{p}", f"scratch_f{p}",
                 f"Probe $-$ scratch, {p}\\%"),
                (f"finetune_vs_probe_f{p}", f"finetune_f{p}", f"probe_f{p}",
                 f"Fine-tuning $-$ probe, {p}\\%")]
    for st in STRATEGIES:
        out.append((f"{st}_f100_vs_f10", f"{st}_f100", f"{st}_f10",
                    f"{STRATEGY_LABELS[st]}: 100\\% $-$ 10\\% labels"))
    out.append(("finetune_f10_vs_scratch_f100", "finetune_f10", "scratch_f100",
                "Fine-tuning 10\\% $-$ scratch 100\\%"))
    for f in FRACTIONS:
        p = pct(f)
        out += [(f"lr1e3_vs_finetune_f{p}", f"finetune_lr1e3_f{p}",
                 f"finetune_f{p}", f"Fine-tuning lr 1e-3 $-$ lr 1e-4, {p}\\%"),
                (f"lr1e4_vs_scratch_f{p}", f"scratch_lr1e4_f{p}",
                 f"scratch_f{p}", f"Scratch lr 1e-4 $-$ lr 1e-3, {p}\\%"),
                (f"finetune_vs_scratch_lr1e4_f{p}", f"finetune_f{p}",
                 f"scratch_lr1e4_f{p}",
                 f"Fine-tuning $-$ scratch, both lr 1e-4, {p}\\%")]
    for g in probe_generators():
        for f in FRACTIONS:
            p = pct(f)
            name = g.replace("_", "\\_")
            out.append((f"probe_{g}_vs_probe_f{p}", f"probe_{g}_f{p}",
                        f"probe_f{p}",
                        f"Probe {name} $-$ probe selected, {p}\\%"))
    return out


def probe_study_numbers(runs: dict, ctx: Context, seeds, s: dict) -> dict:
    """Validation AUROC of the probe on every generator with its bits per
    sample, and the Spearman correlation between the two over the
    generators (descriptive: 7 points)."""
    from scipy.stats import spearmanr
    gens = [ctx.selected] + probe_generators(ctx.selected)
    table = {}
    for f in FRACTIONS:
        p = pct(f)
        xs, ys = [], []
        for g in gens:
            tag = f"probe_f{p}" if g == ctx.selected else f"probe_{g}_f{p}"
            recs = recs_of(runs, tag, seeds)
            bits = ctx.bits(g)
            if not recs or bits is None:
                continue
            auroc = float(np.mean([r["scalars"]["val_macro_auroc"]
                                   for r in recs]))
            table.setdefault(g, {"bits": bits})[p] = auroc
            xs.append(bits)
            ys.append(auroc)
        if len(xs) >= 3:
            rho = spearmanr(xs, ys).statistic
            if not math.isnan(rho):
                s[f"probe_spearman_f{p}"] = float(rho)
            s[f"probe_n_generators_f{p}"] = len(xs)
    return table


def build_summary(paths: Paths = Paths(), fast: bool = False, seeds=SEEDS,
                  ctx: Context | None = None) -> dict:
    runs = all_runs(paths, fast)
    ctx = ctx or Context(paths, fast)
    n_boot = 50 if fast else 1000
    s: dict = {}
    sources = []
    for c in GRID:
        recs = recs_of(runs, c.tag, seeds)
        if not recs:
            continue
        t = c.tag
        sources += [r["run_id"] for r in recs]
        first = recs[0]
        s[f"params_{t}"] = int(first["scalars"]["n_trainable"])
        s[f"params_total_{t}"] = int(first["scalars"]["n_params"]
                                     + first["data"].get("n_params_backbone", 0))
        s[f"n_train_{t}"] = int(first["data"]["n_records"]["train"])
        s[f"n_train_patients_{t}"] = int(first["data"]["n_patients_train"])
        s[f"n_seeds_{t}"] = len(recs)
        s[f"best_epoch_mean_{t}"] = float(np.mean(
            [r["scalars"]["best_epoch"] for r in recs]))
        s[f"epoch_time_median_{t}"] = float(np.median(
            [x for r in recs for x in r["history"]["time_s"]]))
        split_numbers(recs, "val", t, s)
        if c.main:
            split_numbers(recs, "test", t, s)
    for name, a, b, _ in comparisons():
        ra, rb = recs_of(runs, a, seeds), recs_of(runs, b, seeds)
        if not ra or not rb:
            continue
        paired_numbers(name, ra, rb, "val", n_boot, s)
        if BY_TAG[a].main and BY_TAG[b].main:
            paired_numbers(name, ra, rb, "test", n_boot, s)
    probe_table = probe_study_numbers(runs, ctx, seeds, s)
    # Size of the full-label training set and of the 10 % subsets
    f100 = recs_of(runs, "scratch_f100", seeds) or recs_of(runs, "probe_f100",
                                                           seeds)
    if f100:
        s["n_train_all"] = int(f100[0]["data"]["n_records_train_all"])
    return {
        "run_id": SUMMARY_NAME + ("_fast" if fast else ""),
        "schema": runlog.SCHEMA_VERSION,
        "task": "transfer_block_summary",
        "fast": fast,
        "scalars": s,
        "seeds": list(seeds),
        "pretrained": {"selected_config": ctx.selected,
                       "run_ids": ctx.handoff["run_ids"],
                       "tokenizer": ctx.handoff["tokenizer"],
                       "model_args": ctx.handoff["model_args"]},
        "configs": [{"tag": c.tag, "strategy": c.strategy,
                     "label_fraction": c.fraction, "lr": c.lr,
                     "group": c.group, "generator": c.generator,
                     "label": c.label} for c in GRID],
        "comparisons": [{"name": n, "a": a, "b": b} for n, a, b, _ in
                        comparisons()],
        "probe_study": probe_table,
        "batch_records": BATCH, "grad_clip": GRAD_CLIP,
        "head_dropout": HEAD_DROPOUT,
        "source_runs": sources,
        "meta": runlog.environment(),
    }


# ---------------------------------------------------------------------------
# 5. LaTeX tables (macros only: no number is written here)
# ---------------------------------------------------------------------------
def _m(key: str, alias: str = ALIAS) -> str:
    """LaTeX macro of a summary key, e.g. test_auroc_mean_probe_f10 ->
    \\transferTestAurocMeanProbeFOneZero{}."""
    return "\\" + dx.make_numbers_module().macro_name(alias, key) + "{}"


def _pm(s: dict, mean_key: str, sd_key: str, alias: str = ALIAS) -> str:
    """mean +- SD as macros; only the mean when there is one seed."""
    if mean_key not in s:
        return "---"
    return _m(mean_key, alias) + (f" $\\pm$ {_m(sd_key, alias)}"
                                  if sd_key in s else "")


def main_table(summary: dict) -> str | None:
    s = summary["scalars"]
    if not any(f"val_auroc_mean_{t}" in s for t in MAIN_TAGS):
        return None
    lines = [
        "% Generated by src/transfer_experiments.py from results/logs/"
        f"{summary['run_id']}.json.",
        "% Needs report/numbers.tex (alias \"transfer\") and booktabs.",
        "\\begin{tabular}{llrrccc}",
        "\\toprule",
        "Labels & Strategy & Train.\\ param. & Train records & "
        "Val.\\ macro AUROC & Test macro AUROC & Test macro F1 \\\\",
        "\\midrule",
    ]
    for i, f in enumerate(FRACTIONS):
        if i:
            lines.append("\\addlinespace")
        for st in STRATEGIES:
            t = f"{st}_f{pct(f)}"
            if f"val_auroc_mean_{t}" not in s:
                continue
            lines.append(
                f"{pct(f)}\\% & {STRATEGY_LABELS[st]} & {_m('params_' + t)} & "
                f"{_m('n_train_' + t)} & "
                f"{_pm(s, 'val_auroc_mean_' + t, 'val_auroc_sd_' + t)} & "
                f"{_pm(s, 'test_auroc_mean_' + t, 'test_auroc_sd_' + t)} & "
                f"{_pm(s, 'test_f1_mean_' + t, 'test_f1_sd_' + t)} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(lines)


def comparison_table(summary: dict) -> str | None:
    s = summary["scalars"]
    rows = []
    for name, a, b, label in comparisons():
        cells = []
        for split in ("test", "val"):
            if f"{split}_diff_mean_{name}" in s:
                cells.append(
                    f"{_m(f'{split}_diff_mean_{name}')} & "
                    f"{_m(f'{split}_n_better_{name}')} / "
                    f"{_m(f'{split}_n_worse_{name}')} of "
                    f"{_m(f'{split}_n_compared_{name}')}")
            else:
                cells.append("--- & ---")
        if all(c == "--- & ---" for c in cells):
            continue
        rows.append(f"{label} & {cells[0]} & {cells[1]} \\\\")
    if not rows:
        return None
    lines = [
        "% Generated by src/transfer_experiments.py from results/logs/"
        f"{summary['run_id']}.json.",
        "% Delta = paired patient-bootstrap difference in macro AUROC, mean "
        "over seeds; better / worse = seeds whose 95% interval excludes 0.",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        " & \\multicolumn{2}{c}{Test} & \\multicolumn{2}{c}{Validation} \\\\",
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
        "Comparison & $\\Delta$ & Seeds better / worse & $\\Delta$ & "
        "Seeds better / worse \\\\",
        "\\midrule",
        *rows,
        "\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(lines)


def probe_table(summary: dict) -> str | None:
    s = summary["scalars"]
    table = summary.get("probe_study") or {}
    if not table:
        return None
    sel = summary["pretrained"]["selected_config"]
    lines = [
        "% Generated by src/transfer_experiments.py from results/logs/"
        f"{summary['run_id']}.json.",
        "% Needs report/numbers.tex (aliases \"transfer\" and \"generative\").",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Generator (D19) & Val.\\ bits per sample & "
        "Probe val.\\ AUROC, 100\\% & Probe val.\\ AUROC, 10\\% \\\\",
        "\\midrule",
    ]
    for g in [sel] + probe_generators(sel):
        if g not in table:
            continue
        cells = []
        for f in FRACTIONS:
            t = f"probe_f{pct(f)}" if g == sel else f"probe_{g}_f{pct(f)}"
            cells.append(_pm(s, f"val_auroc_mean_{t}", f"val_auroc_sd_{t}"))
        bits = _m(f"bits_mean_{g}", "generative")
        name = g.replace("_", "\\_") + (" (selected)" if g == sel else "")
        lines.append(f"{name} & {bits} & {cells[0]} & {cells[1]} \\\\")
    lines.append("\\midrule")
    rho = [(_m(f"probe_spearman_f{pct(f)}") if f"probe_spearman_f{pct(f)}" in s
            else "---") for f in FRACTIONS]
    lines.append(f"Spearman (bits, AUROC) & & {rho[0]} & {rho[1]} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 6. Figures
# ---------------------------------------------------------------------------
def figure_curves(runs: dict, paths: Paths, suffix: str,
                  seed: int = SEEDS[0]) -> bool:
    """Validation macro AUROC per epoch of the three strategies, with 100 %
    and 10 % of the labels (seed 42), and the training loss of each."""
    needed = [(t, seed) for t in MAIN_TAGS]
    if any(k not in runs for k in needed):
        print("transfer_curves: some seed-42 runs are missing; figure skipped")
        return False
    import matplotlib.pyplot as plt
    eda = dx._style()
    col = dict(zip(STRATEGIES, eda.SERIES_COLOURS))
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))
    for ax, f in zip(axes[:2], FRACTIONS):
        for st in STRATEGIES:
            h = runs[(f"{st}_f{pct(f)}", seed)]["history"]
            ax.plot(h["epoch"], h["val_macro_auroc"], color=col[st],
                    linewidth=2, label=STRATEGY_LABELS[st])
        ax.set(title=f"({'ab'[FRACTIONS.index(f)]}) {pct(f)}% of the labels",
               xlabel="epoch", ylabel="validation macro AUROC")
    axes[0].legend(loc="lower right", fontsize=8)
    for st in STRATEGIES:
        for f, ls in zip(FRACTIONS, ("-", "--")):
            h = runs[(f"{st}_f{pct(f)}", seed)]["history"]
            axes[2].plot(h["epoch"], h["train_loss"], color=col[st],
                         linewidth=2, linestyle=ls,
                         label=f"{STRATEGY_LABELS[st]}, {pct(f)}%")
    axes[2].set(title="(c) Training loss", xlabel="epoch",
                ylabel="BCE loss (training mode)")
    axes[2].legend(loc="upper right", fontsize=6)
    fig.suptitle(f"Transfer from the ECG-GPT, lead II, seed {seed}",
                 color=eda.INK, fontsize=10)
    dx._save(fig, paths, "transfer_curves" + suffix)
    return True


def figure_strategies(runs: dict, paths: Paths, suffix: str,
                      seeds=SEEDS) -> bool:
    """Test macro AUROC of each strategy at 10 % and 100 % of the labels:
    one dot per seed, a line through the means."""
    if not any((t, s) in runs for t in MAIN_TAGS for s in seeds):
        print("transfer_strategies: no main run; figure skipped")
        return False
    import matplotlib.pyplot as plt
    eda = dx._style()
    col = dict(zip(STRATEGIES, eda.SERIES_COLOURS))
    split = "test" if any("test_macro_auroc" in runs[k]["scalars"]
                          for k in runs if k[0] in MAIN_TAGS) else "val"
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    xs = {f: i for i, f in enumerate(sorted(FRACTIONS))}
    for j, st in enumerate(STRATEGIES):
        means = []
        for f in sorted(FRACTIONS):
            vals = [r["scalars"][f"{split}_macro_auroc"]
                    for r in recs_of(runs, f"{st}_f{pct(f)}", seeds)]
            if not vals:
                continue
            x = xs[f] + (j - 1) * 0.08
            ax.scatter([x] * len(vals), vals, s=20, color=col[st], alpha=0.8,
                       edgecolors="white", linewidths=0.8, zorder=3)
            means.append((x, float(np.mean(vals))))
        if means:
            ax.plot(*zip(*means), color=col[st], linewidth=2,
                    label=STRATEGY_LABELS[st])
    ax.set_xticks(list(xs.values()), [f"{pct(f)}% labels" for f in xs])
    ax.set_xlim(-0.4, len(xs) - 0.6)
    ax.set_ylabel(f"{'test' if split == 'test' else 'validation'} macro AUROC")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("ECG-GPT reused for the 5 superclasses (lead II)",
                 fontsize=10)
    dx._save(fig, paths, "transfer_strategies" + suffix)
    return True


def figure_probe_study(summary: dict, paths: Paths, suffix: str) -> bool:
    """Probe validation AUROC of every generator against its validation
    bits per sample (D19): fewer bits = better generator."""
    table = summary.get("probe_study") or {}
    if len(table) < 2:
        print("transfer_probe_study: fewer than 2 generators; figure skipped")
        return False
    import matplotlib.pyplot as plt
    eda = dx._style()
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    for i, f in enumerate(FRACTIONS):
        p = pct(f)
        pts = [(g, v["bits"], v[p]) for g, v in table.items() if p in v]
        if not pts:
            continue
        ax.scatter([b for _, b, _ in pts], [a for _, _, a in pts], s=28,
                   color=eda.SERIES_COLOURS[i], label=f"{p}% labels",
                   edgecolors="white", linewidths=0.8, zorder=3)
        for g, b, a in pts:
            ax.annotate(g, (b, a), fontsize=7, color=eda.INK_MUTED,
                        xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("generator: validation bits per sample (lower = better)")
    ax.set_ylabel("linear probe: validation macro AUROC")
    ax.legend(loc="best", fontsize=8)
    ax.set_title("Better generators, better features?", fontsize=10)
    dx._save(fig, paths, "transfer_probe_study" + suffix)
    return True


# ---------------------------------------------------------------------------
# 7. report: summary log + tables + figures
# ---------------------------------------------------------------------------
def write_report(fast: bool = False, paths: Paths = Paths(), seeds=SEEDS,
                 ctx: Context | None = None) -> dict:
    suffix = "_fast" if fast else ""
    summary = build_summary(paths, fast, seeds, ctx)
    path = runlog.write_run(summary, paths.logs)
    print(f"summary: {rel(path)} ({len(summary['scalars'])} numbers)")
    paths.tables.mkdir(parents=True, exist_ok=True)
    for name, text in (("transfer_main", main_table(summary)),
                       ("transfer_comparisons", comparison_table(summary)),
                       ("transfer_probe_study", probe_table(summary))):
        if text is not None:
            (paths.tables / f"{name}{suffix}.tex").write_text(
                text, encoding="utf-8")
    print(f"tables: {rel(paths.tables)}/transfer_*{suffix}.tex")
    runs = all_runs(paths, fast)
    figure_curves(runs, paths, suffix, seeds[0])
    figure_strategies(runs, paths, suffix, seeds)
    figure_probe_study(summary, paths, suffix)
    print(f"figures: {rel(paths.figures)}/transfer_*{suffix}.pdf|png")
    if not fast:
        register_alias(paths, ALIAS, SUMMARY_NAME)
    return summary


def register_alias(paths: Paths, alias: str, name: str) -> None:
    """Add "alias": "name" to report/runs.json if it is missing."""
    with open(paths.runs_file, encoding="utf-8") as f:
        runs = json.load(f)
    if runs.get(alias) != name:
        runs[alias] = name
        with open(paths.runs_file, "w", encoding="utf-8") as f:
            json.dump(runs, f, indent=2)
            f.write("\n")
        print(f'added "{alias}": "{name}" to {rel(paths.runs_file)}')
    print("next: python report/make_numbers.py")


# ---------------------------------------------------------------------------
def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("step", choices=["estimate", "run", "report", "all"])
    p.add_argument("--fast", action="store_true",
                   help="smoke test: small subset, 2 epochs, outputs *_fast")
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    p.add_argument("--only", nargs="+", default=None,
                   help="run only these configurations (step run)")
    p.add_argument("--group", choices=["main", "lr", "probe_study"],
                   default=None, help="run only one group (step run)")
    a = p.parse_args(argv)
    t0 = datetime.now()
    paths = Paths()
    if a.step in ("estimate", "run", "all"):
        data = load_data(None, a.fast)
        ctx = Context(paths, a.fast)
        if a.step == "estimate":
            estimate(seeds=a.seeds, fast=a.fast, paths=paths, data=data,
                     ctx=ctx)
        else:
            run_all(a.seeds, a.only, a.group, a.fast, paths=paths, data=data,
                    ctx=ctx)
    if a.step in ("report", "all"):
        write_report(a.fast, paths, a.seeds)
    print(f"\ndone in {datetime.now() - t0}")


if __name__ == "__main__":
    main()
