"""Dense block (T9): baseline, ablations, final model and report outputs.

Decisions: docs/DECISIONS.md D15 (this block), D3, D4 and D14 (protocol).

    python -m src.dense_experiments all --fast   # smoke test of everything
    python -m src.dense_experiments run          # 12 configurations x 3 seeds,
                                                 # validation only
    python -m src.dense_experiments final        # best configuration on
                                                 # validation -> test fold
    python -m src.dense_experiments report       # summary log, tables, figures

`run` and `final` can be stopped and started again: a configuration and
seed that already has a (non --fast) log is skipped.

How an ablation works: the baseline has every component switched on
(input z-score, batch norm, dropout 0.5, 2 hidden layers of 256 units).
Each other configuration changes ONE factor and keeps everything else equal
- same data, split, loss, optimiser, learning rate, batch size, early
stopping and seeds - so a change in validation macro AUROC can be put down
to that factor. The one exception is `no_norm_no_bn`, which switches off
two factors on purpose to see whether batch norm makes up for the missing
input scaling (weekly summary 30/10 links both ideas).

Outputs of `report`:
    results/logs/dense_summary.json      numbers for make_numbers.py
                                         (alias "dense" in report/runs.json)
    report/tables/dense_ablations.tex    ablation table (LaTeX macros only)
    report/tables/dense_final.tex        test results of the final model
    report/figures/dense_curves.*        learning curves (seed 42)
    report/figures/dense_ablations.*     validation macro AUROC per config
    report/figures/dense_first_layer.*   first-layer weights as 12 x 1000
                                         patterns (final model, seed 42)
With --fast every output name ends in _fast and git ignores it.
"""
import argparse
import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from src import config, metrics, runlog
from src.models.dense import DenseNet
from src.train import TrainConfig, ptbxl_setup, rel, run_experiment

MODEL = "dense"
SEEDS = (42, 43, 44)                  # D14: seeds of the final models
TABLES_DIR = config.REPORT_DIR / "tables"
RUNS_FILE = config.REPORT_DIR / "runs.json"
ALIAS = "dense"                       # alias of the summary in runs.json
SUMMARY_NAME = "dense_summary"

# Baseline (D15). Every ablation starts from these values.
BASE_ARGS = {"hidden_sizes": [256, 256], "batch_norm": True, "dropout": 0.5}


@dataclass(frozen=True)
class Ablation:
    tag: str                  # run name part; letters, digits, underscores
    factor: str               # family, used to group the table and figure
    label: str                # short English description for the report
    changes: dict = field(default_factory=dict)   # changes to BASE_ARGS
    normalize: bool = True    # per-lead z-score of the input (D3)

    @property
    def model_args(self) -> dict:
        return {**BASE_ARGS, **self.changes}


ABLATIONS = [
    Ablation("base", "baseline", "Baseline: z-score, BN, dropout 0.5, 2 x 256"),
    Ablation("no_norm", "input scaling", "No input z-score (raw mV)",
             normalize=False),
    Ablation("no_bn", "batch norm", "No batch norm", {"batch_norm": False}),
    Ablation("no_norm_no_bn", "input scaling x batch norm",
             "No input z-score and no batch norm", {"batch_norm": False},
             normalize=False),
    Ablation("no_dropout", "dropout", "No dropout", {"dropout": 0.0}),
    Ablation("dropout02", "dropout", "Dropout 0.2", {"dropout": 0.2}),
    Ablation("depth1", "depth", "1 hidden layer (256)",
             {"hidden_sizes": [256]}),
    Ablation("depth3", "depth", "3 hidden layers (256)",
             {"hidden_sizes": [256] * 3}),
    Ablation("depth4", "depth", "4 hidden layers (256)",
             {"hidden_sizes": [256] * 4}),
    Ablation("width64", "width", "Width 64 (2 layers)",
             {"hidden_sizes": [64, 64]}),
    Ablation("width1024", "width", "Width 1024 (2 layers)",
             {"hidden_sizes": [1024, 1024]}),
]
BY_TAG = {a.tag: a for a in ABLATIONS}
FACTORS = list(dict.fromkeys(a.factor for a in ABLATIONS))   # in order


@dataclass
class Paths:
    """Where runs and outputs go; tests point these at a temporary folder."""
    logs: Path = config.LOGS_DIR
    predictions: Path = config.PREDICTIONS_DIR
    checkpoints: Path = config.CHECKPOINT_DIR
    figures: Path = config.FIGURES_DIR
    tables: Path = TABLES_DIR
    runs_file: Path = RUNS_FILE


# ---------------------------------------------------------------------------
# 1. Configurations and runs
# ---------------------------------------------------------------------------
def make_config(ab: Ablation, seed: int, fast: bool = False,
                final: bool = False) -> TrainConfig:
    """TrainConfig of one ablation. Optimiser, learning rate, batch size,
    epochs and patience keep the D14 defaults for every configuration."""
    cfg = TrainConfig(model=MODEL, tag=ab.tag, seed=seed,
                      model_args=ab.model_args, normalize=ab.normalize,
                      fast=fast, final=final, notes=ab.label)
    if fast:                          # D14: 2 epochs, 50 resamples
        cfg.max_epochs, cfg.n_boot = 2, 50
    return cfg


def build_model(model_args: dict) -> DenseNet:
    return DenseNet(**model_args)


def find_runs(paths: Paths, fast: bool, final: bool) -> dict:
    """(tag, seed) -> log of the latest matching dense run.

    Logs are matched by their content (model, tag, seed, fast, final), not
    by their file name, so a tag that is the prefix of another one cannot
    be confused with it."""
    found = {}
    for path in sorted(paths.logs.glob(f"{MODEL}_*.json")):
        with open(path, encoding="utf-8") as f:
            rec = json.load(f)
        cfg = rec.get("config", {})
        if (cfg.get("model") != MODEL or bool(rec.get("fast")) != fast
                or bool(rec.get("final")) != final):
            continue
        found[(cfg["tag"], cfg["seed"])] = rec      # sorted: latest wins
    return found


def _train(ab: Ablation, seed: int, fast: bool, final: bool, cache: dict,
           paths: Paths) -> dict:
    cfg = make_config(ab, seed, fast, final)
    loaders, info, task = ptbxl_setup(cfg, cache)
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
    cache = _load_cache(cache)
    done = find_runs(paths, fast=False, final=False)
    todo = [(t, s) for t in tags for s in seeds]
    for i, (tag, seed) in enumerate(todo, 1):
        if not fast and (tag, seed) in done:
            print(f"[{i}/{len(todo)}] {tag} seed {seed}: already run "
                  f"({done[(tag, seed)]['run_id']}), skipped")
            continue
        print(f"\n[{i}/{len(todo)}] {tag} seed {seed}: {BY_TAG[tag].label}")
        _train(BY_TAG[tag], seed, fast, False, cache, paths)


def _load_cache(cache: dict | None) -> dict:
    if cache is not None:
        return cache
    from src.data import load_cache
    print("loading the cache ...")
    return load_cache()


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
    the seeds (D15). The test fold plays no part in this choice."""
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
    cache = _load_cache(cache)
    for seed in seeds:
        if not fast and (tag, seed) in done:
            print(f"final {tag} seed {seed}: already run, skipped")
            continue
        print(f"\nfinal {tag} seed {seed}")
        _train(BY_TAG[tag], seed, fast, True, cache, paths)
    return tag


# ---------------------------------------------------------------------------
# 3. Summary log for make_numbers.py
# ---------------------------------------------------------------------------
def load_predictions(record: dict, split: str) -> dict:
    """Arrays of one split from a run's predictions file."""
    path = config.ROOT / record["predictions"]     # absolute paths stay as is
    with np.load(path) as data:
        return {k[len(split) + 1:]: data[k] for k in data.files
                if k.startswith(f"{split}_")}


def paired_diff(rec_a: dict, rec_b: dict, split: str, n_boot: int) -> dict:
    """Paired patient-bootstrap interval of macro AUROC(A) - macro AUROC(B)
    (D4, D14), checking that both runs saw the same records."""
    a, b = load_predictions(rec_a, split), load_predictions(rec_b, split)
    for key in ("ecg_id", "patient_id", "true"):
        if key in a and key in b and not np.array_equal(a[key], b[key]):
            raise ValueError(f"{rec_a['run_id']} and {rec_b['run_id']} differ "
                             f"in {split}_{key}")
    return metrics.paired_bootstrap_diff(a["true"], a["prob"], b["prob"],
                                         a["patient_id"], n_boot=n_boot,
                                         seed=config.SEED)


def _sd(values) -> float:
    """Sample standard deviation over seeds (NaN with a single seed)."""
    return float(np.std(values, ddof=1)) if len(values) > 1 else math.nan


def _trivial_record(paths: Paths) -> dict | None:
    """The trivial model's final run, listed as "trivial" in runs.json; it is
    the floor of every comparison (D14)."""
    try:
        with open(paths.runs_file, encoding="utf-8") as f:
            name = json.load(f).get("trivial")
        return runlog.load_run(name, paths.logs) if name else None
    except (OSError, ValueError):
        return None


def build_summary(paths: Paths = Paths(), fast: bool = False,
                  seeds=SEEDS) -> dict:
    """Collect every number of the block into one log with a flat `scalars`
    block (read by make_numbers.py under the alias "dense")."""
    runs = find_runs(paths, fast=fast, final=False)
    finals = find_runs(paths, fast=fast, final=True)
    n_boot = 50 if fast else 1000
    s: dict = {}
    sources = []

    # Ablations on validation.
    base = {seed: runs.get(("base", seed)) for seed in seeds}
    for ab in ABLATIONS:
        recs = [runs[(ab.tag, seed)] for seed in seeds if (ab.tag, seed) in runs]
        if not recs:
            continue
        sources += [r["run_id"] for r in recs]
        t = ab.tag
        aurocs = [r["scalars"]["val_macro_auroc"] for r in recs]
        s[f"params_{t}"] = int(recs[0]["scalars"]["n_params"])
        s[f"val_auroc_mean_{t}"] = float(np.mean(aurocs))
        s[f"val_auroc_sd_{t}"] = _sd(aurocs)
        s[f"best_epoch_mean_{t}"] = float(np.mean(
            [r["scalars"]["best_epoch"] for r in recs]))
        s[f"epoch_time_mean_{t}"] = float(np.mean(
            [np.mean(r["history"]["time_s"]) for r in recs]))
        first = runs.get((t, seeds[0]))
        if first is not None:
            sc = first["scalars"]
            s[f"val_auroc_first_{t}"] = sc["val_macro_auroc"]
            s[f"val_auroc_lo_first_{t}"] = sc["val_macro_auroc_lo"]
            s[f"val_auroc_hi_first_{t}"] = sc["val_macro_auroc_hi"]
        if t == "base":
            continue
        diffs, better, worse = [], 0, 0
        for seed in seeds:
            if (t, seed) not in runs or base[seed] is None:
                continue
            print(f"paired difference {t} - base, seed {seed} ...")
            d = paired_diff(runs[(t, seed)], base[seed], "val", n_boot)
            diffs.append(d["diff"])
            s[f"diff_s{seed}_{t}"] = d["diff"]
            s[f"diff_lo_s{seed}_{t}"] = d["lower"]
            s[f"diff_hi_s{seed}_{t}"] = d["upper"]
            better += d["lower"] > 0          # whole interval above 0
            worse += d["upper"] < 0           # whole interval below 0
        if diffs:
            s[f"diff_mean_{t}"] = float(np.mean(diffs))
            s[f"n_seeds_better_{t}"] = int(better)
            s[f"n_seeds_worse_{t}"] = int(worse)
            s[f"n_seeds_compared_{t}"] = len(diffs)

    # Final model: the configuration selected on validation, on test.
    selected = None
    final_recs = [r for (t, seed), r in sorted(finals.items(),
                                                key=lambda kv: kv[0][1])
                  if seed in seeds]
    tags = {r["config"]["tag"] for r in final_recs}
    if len(tags) > 1:
        raise RuntimeError(f"final runs of several configurations: {tags}")
    trivial = _trivial_record(paths)
    if final_recs:
        selected = final_recs[0]["config"]["tag"]
        sources += [r["run_id"] for r in final_recs]
        s["final_n_params"] = int(final_recs[0]["scalars"]["n_params"])
        s["final_n_seeds"] = len(final_recs)
        for key in ("val_macro_auroc", "test_macro_auroc", "test_macro_f1"):
            vals = [r["scalars"][key] for r in final_recs]
            s[f"final_{key}_mean"] = float(np.mean(vals))
            s[f"final_{key}_sd"] = _sd(vals)
        for c in config.CLASSES:
            vals = [r["scalars"][f"test_auroc_{c}"] for r in final_recs]
            s[f"final_test_auroc_{c}_mean"] = float(np.mean(vals))
            s[f"final_test_auroc_{c}_sd"] = _sd(vals)
        first = final_recs[0]["scalars"]
        s["final_first_seed"] = int(final_recs[0]["config"]["seed"])
        for key in ["test_macro_auroc", "test_macro_f1"] + \
                [f"test_auroc_{c}" for c in config.CLASSES]:
            s[f"final_first_{key}"] = first[key]
            s[f"final_first_{key}_lo"] = first[f"{key}_lo"]
            s[f"final_first_{key}_hi"] = first[f"{key}_hi"]
        for c in config.CLASSES:
            s[f"final_first_test_f1_{c}"] = first[f"test_f1_{c}"]
        # --fast runs see a random subset of the test fold, the trivial run
        # the whole fold: they cannot be paired, so no comparison then.
        if trivial is not None and trivial.get("final") and not fast:
            diffs = []
            for r in final_recs:
                print(f"paired difference {r['run_id']} - trivial on test ...")
                d = paired_diff(r, trivial, "test", n_boot)
                seed = r["config"]["seed"]
                diffs.append(d["diff"])
                s[f"diff_trivial_s{seed}"] = d["diff"]
                s[f"diff_trivial_lo_s{seed}"] = d["lower"]
                s[f"diff_trivial_hi_s{seed}"] = d["upper"]
            s["diff_trivial_mean"] = float(np.mean(diffs))
    if trivial is not None:
        s["trivial_val_macro_auroc"] = trivial["scalars"]["val_macro_auroc"]

    return {
        "run_id": SUMMARY_NAME + ("_fast" if fast else ""),
        "schema": runlog.SCHEMA_VERSION,
        "task": "dense_block_summary",
        "fast": fast,
        "scalars": s,
        "seeds": list(seeds),
        "selected_config": selected,
        "selection_rule": "highest mean validation macro AUROC over the seeds",
        "baseline_args": BASE_ARGS,
        "ablations": [{"tag": a.tag, "factor": a.factor, "label": a.label,
                       "model_args": a.model_args, "normalize": a.normalize}
                      for a in ABLATIONS],
        "trivial_run": None if trivial is None else trivial["run_id"],
        "source_runs": sources,
        "meta": runlog.environment(),
    }


# ---------------------------------------------------------------------------
# 4. LaTeX tables (made of macros only: no number is written here)
# ---------------------------------------------------------------------------
_MAKE_NUMBERS = None


def make_numbers_module():
    """report/make_numbers.py is a script, not a package: load it by path so
    the tables use exactly the macro names it will define."""
    global _MAKE_NUMBERS
    if _MAKE_NUMBERS is None:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "make_numbers", config.REPORT_DIR / "make_numbers.py")
        _MAKE_NUMBERS = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_MAKE_NUMBERS)
    return _MAKE_NUMBERS


def _m(key: str) -> str:
    """LaTeX macro of a summary key, e.g. val_auroc_mean_base ->
    \\denseValAurocMeanBase{}."""
    return "\\" + make_numbers_module().macro_name(ALIAS, key) + "{}"


def ablation_table(summary: dict) -> str:
    s = summary["scalars"]
    lines = [
        "% Generated by src/dense_experiments.py from results/logs/"
        f"{summary['run_id']}.json.",
        "% Needs report/numbers.tex (alias \"dense\" in report/runs.json)"
        " and \\usepackage{booktabs}.",
        "\\begin{tabular}{llrcrc}",
        "\\toprule",
        "Factor & Configuration & Parameters & Val.\\ macro AUROC & "
        "$\\Delta$ vs.\\ baseline & Seeds better / worse \\\\",
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
        factor = ab.factor.replace(" x ", " $\\times$ ")
        auroc = f"{_m('val_auroc_mean_' + t)} $\\pm$ {_m('val_auroc_sd_' + t)}"
        if t == "base" or f"diff_mean_{t}" not in s:
            diff, seeds = "---", "---"
        else:
            diff = _m(f"diff_mean_{t}")
            seeds = (f"{_m('n_seeds_better_' + t)} / {_m('n_seeds_worse_' + t)}"
                     f" of {_m('n_seeds_compared_' + t)}")
        label = ab.label.replace(" x ", " $\\times$ ")
        lines.append(f"{factor} & {label} & {_m('params_' + t)} & "
                     f"{auroc} & {diff} & {seeds} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(lines)


def final_table(summary: dict) -> str | None:
    s = summary["scalars"]
    if "final_test_macro_auroc_mean" not in s:
        return None
    lines = [
        "% Generated by src/dense_experiments.py from results/logs/"
        f"{summary['run_id']}.json.",
        "% Needs report/numbers.tex (aliases \"dense\" and \"trivial\").",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        " & \\multicolumn{2}{c}{Dense (final)} & Trivial \\\\",
        "\\cmidrule(lr){2-3}",
        "Test AUROC & First seed [95\\% CI] & Mean $\\pm$ SD (seeds) & \\\\",
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
        spread = f"{_m(mean_key + '_mean')} $\\pm$ {_m(mean_key + '_sd')}"
        trivial = "\\" + make_numbers_module().macro_name("trivial", key)
        lines.append(f"{name} & {first} & {spread} & {trivial}{{}} \\\\")
        if name == "Macro":
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 5. Figures
# ---------------------------------------------------------------------------
def _style():
    import matplotlib
    matplotlib.use("Agg")
    from src import eda               # same palette and style as the EDA
    eda.set_style()
    return eda


def _save(fig, paths: Paths, name: str) -> None:
    import matplotlib.pyplot as plt
    paths.figures.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(paths.figures / f"{name}.{ext}")
    plt.close(fig)


def figure_curves(runs: dict, paths: Paths, suffix: str,
                  seed: int = SEEDS[0]) -> bool:
    """(a) baseline and (b) no-dropout losses per epoch; (c) validation
    macro AUROC per epoch of the 2 x 2 input-scaling x batch-norm grid."""
    needed = ["base", "no_dropout", "no_norm", "no_bn", "no_norm_no_bn"]
    if any((t, seed) not in runs for t in needed):
        print("dense_curves: some seed-42 runs are missing; figure skipped")
        return False
    import matplotlib.pyplot as plt
    eda = _style()
    col, ink, muted = eda.SERIES_COLOURS, eda.INK, eda.INK_MUTED
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))
    panels = [(axes[0], "base", "(a) Baseline"),
              (axes[1], "no_dropout", "(b) No dropout")]
    # Same y range in (a) and (b), so the two gaps can be compared by eye.
    losses = [v for _, tag, _ in panels for k in ("train_loss", "val_loss")
              for v in runs[(tag, seed)]["history"][k]]
    pad = 0.05 * (max(losses) - min(losses))
    ylim = (min(losses) - pad, max(losses) + 3 * pad)
    for ax, tag, title in panels:
        h = runs[(tag, seed)]["history"]
        best = runs[(tag, seed)]["selection"]["best_epoch"]
        ax.plot(h["epoch"], h["train_loss"], color=col[0], linewidth=2,
                label="training loss (training mode)")
        ax.plot(h["epoch"], h["val_loss"], color=col[1], linewidth=2,
                label="validation loss")
        ax.set_ylim(*ylim)
        ax.axvline(best, color=muted, linewidth=1, linestyle="--")
        ax.text(best, ylim[1], " best epoch", color=muted, fontsize=8,
                va="top")
        ax.set(title=title, xlabel="epoch", ylabel="BCE loss")
    axes[0].legend(loc="upper right", fontsize=8)
    grid = [("base", "z-score + BN"), ("no_bn", "z-score, no BN"),
            ("no_norm", "raw mV + BN"), ("no_norm_no_bn", "raw mV, no BN")]
    styles = ["-", "--", "-", "--"]
    for i, ((tag, name), ls) in enumerate(zip(grid, styles)):
        h = runs[(tag, seed)]["history"]
        axes[2].plot(h["epoch"], h["val_macro_auroc"], color=col[i],
                     linewidth=2, linestyle=ls, label=name)
    axes[2].set(title="(c) Input scaling and batch norm", xlabel="epoch",
                ylabel="validation macro AUROC")
    axes[2].legend(loc="lower right", fontsize=8)
    fig.suptitle(f"Dense network, seed {seed}", color=ink, fontsize=10)
    _save(fig, paths, "dense_curves" + suffix)
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
        print("dense_ablations: no baseline run; figure skipped")
        return False
    import matplotlib.pyplot as plt
    eda = _style()
    # One colour per group of factors, in the fixed order of the palette
    # (never cycled); the baseline is drawn in ink.
    groups = {"input scaling": 0, "batch norm": 0,
              "input scaling x batch norm": 0, "dropout": 1, "depth": 2,
              "width": 3}
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
    # The trivial model is not drawn: its much lower AUROC would squeeze
    # the differences between configurations; it is in dense_final.tex.
    ax.set_yticks(y, [ab.label for ab in order], fontsize=8)
    ax.set_ylim(y[-1] - 1.2, y[0] + 1.1)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(f"validation macro AUROC (dots: seeds {', '.join(map(str, seeds))};"
                  " bar: mean)")
    _save(fig, paths, "dense_ablations" + suffix)
    return True


def figure_first_layer(finals: dict, paths: Paths, suffix: str,
                       n_units: int = 3) -> bool:
    """First-layer weights of the final model (first seed) as 12 x 1000
    patterns: each hidden unit compares the whole record with one such
    pattern (FeedForward slides 29-30). The units with the largest weight
    norm are shown, on a diverging scale centred at 0."""
    if not finals:
        print("dense_first_layer: no final run; figure skipped")
        return False
    rec = finals[min(finals, key=lambda k: k[1])]
    model = build_model(rec["model"]["args"])
    state = torch.load(config.ROOT / rec["checkpoint"], map_location="cpu",
                       weights_only=True)
    model.load_state_dict(state)
    W = model.first_layer.weight.detach().numpy()          # (units, 12000)
    n_leads, n_samples = len(config.LEADS), W.shape[1] // len(config.LEADS)
    units = np.argsort(-np.linalg.norm(W, axis=1))[:n_units]

    import matplotlib
    import matplotlib.pyplot as plt
    eda = _style()
    diverging = matplotlib.colors.LinearSegmentedColormap.from_list(
        "blue_grey_orange", ["#0d366b", "#2a78d6", "#f1f0ec", "#eb6834",
                             "#7a2c0b"])
    fig, axes = plt.subplots(n_units, 1, figsize=(9, 2.2 * n_units),
                             sharex=True)
    axes = np.atleast_1d(axes)
    t = np.arange(n_samples) / config.SAMPLING_RATE
    for ax, u in zip(axes, units):
        pattern = W[u].reshape(n_leads, n_samples)
        lim = float(np.abs(pattern).max())
        im = ax.imshow(pattern, aspect="auto", cmap=diverging, vmin=-lim,
                       vmax=lim, extent=[t[0], t[-1] + 1 / config.SAMPLING_RATE,
                                         n_leads - 0.5, -0.5],
                       interpolation="nearest")
        ax.set_yticks(range(n_leads), config.LEADS, fontsize=7)
        ax.set_title(f"hidden unit {u}", fontsize=9, loc="left")
        ax.grid(False)
        fig.colorbar(im, ax=ax, label="weight", pad=0.01)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(f"First-layer weights of the final Dense model "
                 f"({rec['config']['tag']}, seed {rec['config']['seed']})",
                 color=eda.INK, fontsize=10)
    _save(fig, paths, "dense_first_layer" + suffix)
    return True


# ---------------------------------------------------------------------------
# 6. report: summary log + tables + figures
# ---------------------------------------------------------------------------
def write_report(fast: bool = False, paths: Paths = Paths(),
                 seeds=SEEDS) -> dict:
    suffix = "_fast" if fast else ""
    summary = build_summary(paths, fast, seeds)
    path = runlog.write_run(summary, paths.logs)
    print(f"summary: {rel(path)} ({len(summary['scalars'])} numbers)")

    paths.tables.mkdir(parents=True, exist_ok=True)
    (paths.tables / f"dense_ablations{suffix}.tex").write_text(
        ablation_table(summary), encoding="utf-8")
    final = final_table(summary)
    if final is not None:
        (paths.tables / f"dense_final{suffix}.tex").write_text(
            final, encoding="utf-8")
    print(f"tables: {rel(paths.tables)}/dense_*{suffix}.tex")

    runs = find_runs(paths, fast=fast, final=False)
    finals = find_runs(paths, fast=fast, final=True)
    figure_curves(runs, paths, suffix, seeds[0])
    figure_ablations(runs, paths, suffix, seeds)
    figure_first_layer(finals, paths, suffix)
    print(f"figures: {rel(paths.figures)}/dense_*{suffix}.pdf|png")

    if not fast:
        _register_alias(paths)
    return summary


def _register_alias(paths: Paths) -> None:
    """Add "dense": "dense_summary" to report/runs.json if it is missing."""
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
    p.add_argument("step", choices=["run", "final", "report", "all"])
    p.add_argument("--fast", action="store_true",
                   help="smoke test: small subset, 2 epochs, outputs *_fast")
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    p.add_argument("--only", nargs="+", default=None,
                   help="run only these configurations (step run)")
    a = p.parse_args(argv)
    t0 = datetime.now()
    cache = None
    if a.step in ("run", "all"):
        cache = _load_cache(None)
        run_ablations(a.seeds, a.only, a.fast, cache)
    if a.step in ("final", "all"):
        run_final(a.fast, cache, seeds=a.seeds)
    if a.step in ("report", "all"):
        write_report(a.fast, seeds=a.seeds)
    print(f"\ndone in {datetime.now() - t0}")


if __name__ == "__main__":
    main()
