"""CNN block (T10): baseline, ablations, final model and report outputs.

Decisions: docs/DECISIONS.md D16 (this block), D3, D4 and D14 (protocol).
Same structure and protocol as src/dense_experiments.py (D15).

    python -m src.cnn_experiments all --fast   # smoke test of everything
    python -m src.cnn_experiments run          # 10 configurations x 3 seeds,
                                               # validation only
    python -m src.cnn_experiments final        # best configuration on
                                               # validation -> test fold
    python -m src.cnn_experiments report       # summary log, tables, figures

`run` and `final` can be stopped and started again: a configuration and
seed that already has a (non --fast) log is skipped.

How an ablation works: the baseline has every component switched on
(4 residual blocks of two convolutions with kernel 7, batch norm, global
average pooling, dropout 0.5). Each other configuration changes ONE factor
and keeps everything else equal - same data, split, loss, optimiser,
learning rate, batch size, early stopping and seeds. Each configuration is
compared with a reference trained with the same seed: the baseline, except
`depth6_no_residual`, which is compared with `depth6` so that only the
skip connections differ (are they more useful in a deeper network?).

Outputs of `report`:
    results/logs/cnn_summary.json      numbers for make_numbers.py
                                       (alias "cnn" in report/runs.json)
    report/tables/cnn_ablations.tex    ablation table (LaTeX macros only)
    report/tables/cnn_final.tex        test results of the final model,
                                       with the final Dense and the trivial
                                       model
    report/figures/cnn_curves.*        learning curves (seed 42)
    report/figures/cnn_ablations.*     validation macro AUROC per config
    report/figures/cnn_first_layer.*   first-layer filters and their
                                       frequency response (final model)
    report/figures/cnn_feature_maps.*  feature maps and class activation
                                       maps of one record per superclass
With --fast every output name ends in _fast and git ignores it.
"""
import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import torch

from src import config, runlog
from src import dense_experiments as dx       # generic helpers, not copied
from src.dense_experiments import Paths
from src.models.cnn import CNN1D
from src.train import TrainConfig, ptbxl_setup, rel, run_experiment

MODEL = "cnn"
SEEDS = (42, 43, 44)                  # D14: seeds of the final models
ALIAS = "cnn"                         # alias of the summary in runs.json
SUMMARY_NAME = "cnn_summary"

# Baseline (D16). Every ablation starts from these values.
BASE_ARGS = {"channels": [32, 64, 128, 128], "kernel_size": 7,
             "batch_norm": True, "residual": True, "pooling": "gap",
             "dropout": 0.5}
DEEP = [32, 64, 128, 128, 128, 128]   # 6 blocks

# Training loss compared at this epoch (D16, second entry): with patience 5
# every full run lasts at least 6 epochs, so epoch 5 exists in all of them.
# It measures how fast each network fits the training set (optimisation),
# which is what skip connections are meant to help (CNN slides 49-51).
EARLY_EPOCH = 5


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


ABLATIONS = [
    Ablation("base", "baseline",
             "Baseline: 4 residual blocks, kernel 7, BN, GAP, dropout 0.5"),
    Ablation("kernel3", "kernel size", "Kernel 3", {"kernel_size": 3}),
    Ablation("kernel15", "kernel size", "Kernel 15", {"kernel_size": 15}),
    Ablation("depth2", "number of blocks", "2 blocks",
             {"channels": [32, 64]}),
    Ablation("depth6", "number of blocks", "6 blocks", {"channels": DEEP}),
    Ablation("no_bn", "batch norm", "No batch norm", {"batch_norm": False}),
    Ablation("no_residual", "residual connections", "No residual connections",
             {"residual": False}),
    Ablation("depth6_no_residual", "residual connections x depth",
             "6 blocks, no residual (vs. 6 blocks)",
             {"channels": DEEP, "residual": False}, reference="depth6"),
    Ablation("flatten", "global pooling vs flatten",
             "Flatten instead of global average pooling",
             {"pooling": "flatten"}),
    Ablation("no_dropout", "dropout", "No dropout", {"dropout": 0.0}),
]
BY_TAG = {a.tag: a for a in ABLATIONS}
FACTORS = list(dict.fromkeys(a.factor for a in ABLATIONS))   # in order


# ---------------------------------------------------------------------------
# 1. Configurations and runs
# ---------------------------------------------------------------------------
def make_config(ab: Ablation, seed: int, fast: bool = False,
                final: bool = False) -> TrainConfig:
    """TrainConfig of one ablation. Optimiser, learning rate, batch size,
    epochs and patience keep the D14 defaults for every configuration, and
    the input z-score (D3) is always on."""
    cfg = TrainConfig(model=MODEL, tag=ab.tag, seed=seed,
                      model_args=ab.model_args, normalize=True,
                      fast=fast, final=final, notes=ab.label)
    if fast:                          # D14: 2 epochs, 50 resamples
        cfg.max_epochs, cfg.n_boot = 2, 50
    return cfg


def build_model(model_args: dict) -> CNN1D:
    return CNN1D(**model_args)


def find_runs(paths: Paths, fast: bool, final: bool,
              model: str = MODEL) -> dict:
    """(tag, seed) -> log of the latest matching run of `model`.

    Logs are matched by their content (model, tag, seed, fast, final), not
    by their file name, so a tag that is the prefix of another one (depth6,
    depth6_no_residual) cannot be confused with it."""
    found = {}
    for path in sorted(paths.logs.glob(f"{model}_*.json")):
        with open(path, encoding="utf-8") as f:
            rec = json.load(f)
        cfg = rec.get("config", {})
        if (cfg.get("model") != model or bool(rec.get("fast")) != fast
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
    the seeds (D16, as D15). The test fold plays no part in this choice."""
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
# 3. Summary log for make_numbers.py
# ---------------------------------------------------------------------------
def receptive_field(ab: Ablation) -> int:
    """Receptive field (samples) of one position of the last feature map."""
    return build_model(ab.model_args).receptive_field()


def _alias_log(paths: Paths, alias: str) -> dict | None:
    """The log listed under `alias` in runs.json, or None."""
    try:
        with open(paths.runs_file, encoding="utf-8") as f:
            name = json.load(f).get(alias)
        return runlog.load_run(name, paths.logs) if name else None
    except (OSError, ValueError):
        return None


def dense_finals(paths: Paths) -> tuple[dict, str | None]:
    """seed -> final (test) run of the Dense configuration that the Dense
    summary selected (D15), and the run id of that summary."""
    summary = _alias_log(paths, "dense")
    if summary is None or not summary.get("selected_config"):
        return {}, None
    tag = summary["selected_config"]
    found = find_runs(paths, fast=False, final=True, model="dense")
    return ({seed: r for (t, seed), r in found.items() if t == tag},
            summary["run_id"])


def compare_finals(final_recs: list, others: dict, name: str, n_boot: int,
                   s: dict) -> None:
    """Paired test-fold differences, final CNN - other model, one per seed
    (others: seed -> run). Writes diff_<name>_s<seed>, its interval, the
    mean and how many seeds are better / worse into `s`."""
    diffs, better, worse = [], 0, 0
    for r in final_recs:
        seed = r["config"]["seed"]
        other = others.get(seed)
        if other is None:
            continue
        print(f"paired difference {r['run_id']} - {other['run_id']} on test")
        d = dx.paired_diff(r, other, "test", n_boot)
        diffs.append(d["diff"])
        s[f"diff_{name}_s{seed}"] = d["diff"]
        s[f"diff_{name}_lo_s{seed}"] = d["lower"]
        s[f"diff_{name}_hi_s{seed}"] = d["upper"]
        better += d["lower"] > 0
        worse += d["upper"] < 0
    if diffs:
        s[f"diff_{name}_mean"] = float(np.mean(diffs))
        s[f"n_seeds_better_{name}"] = int(better)
        s[f"n_seeds_worse_{name}"] = int(worse)
        s[f"n_seeds_compared_{name}"] = len(diffs)


def early_train_loss(record: dict) -> float | None:
    """Training loss of a run at EARLY_EPOCH (None if it stopped before)."""
    losses = record["history"]["train_loss"]
    return losses[EARLY_EPOCH - 1] if len(losses) >= EARLY_EPOCH else None


def build_summary(paths: Paths = Paths(), fast: bool = False,
                  seeds=SEEDS, cache: dict | None = None) -> dict:
    """Collect every number of the block into one log with a flat `scalars`
    block (read by make_numbers.py under the alias "cnn"). With the cache,
    the probabilities of the example records are added too."""
    runs = find_runs(paths, fast=fast, final=False)
    finals = find_runs(paths, fast=fast, final=True)
    n_boot = 50 if fast else 1000
    s: dict = {}
    sources = []

    # Ablations on validation.
    for ab in ABLATIONS:
        recs = [runs[(ab.tag, seed)] for seed in seeds if (ab.tag, seed) in runs]
        if not recs:
            continue
        sources += [r["run_id"] for r in recs]
        t = ab.tag
        aurocs = [r["scalars"]["val_macro_auroc"] for r in recs]
        rf = receptive_field(ab)
        s[f"params_{t}"] = int(recs[0]["scalars"]["n_params"])
        s[f"rf_samples_{t}"] = int(rf)
        s[f"rf_seconds_{t}"] = rf / config.SAMPLING_RATE
        s[f"val_auroc_mean_{t}"] = float(np.mean(aurocs))
        s[f"val_auroc_sd_{t}"] = dx._sd(aurocs)
        s[f"best_epoch_mean_{t}"] = float(np.mean(
            [r["scalars"]["best_epoch"] for r in recs]))
        s[f"epoch_time_mean_{t}"] = float(np.mean(
            [np.mean(r["history"]["time_s"]) for r in recs]))
        early = [early_train_loss(r) for r in recs]
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
        diffs, better, worse = [], 0, 0
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
        if diffs:
            s[f"diff_mean_{t}"] = float(np.mean(diffs))
            s[f"n_seeds_better_{t}"] = int(better)
            s[f"n_seeds_worse_{t}"] = int(worse)
            s[f"n_seeds_compared_{t}"] = len(diffs)
        # Training loss at EARLY_EPOCH minus that of the reference, same seed.
        gaps = []
        for seed in seeds:
            ref = runs.get((ab.reference, seed))
            if (t, seed) not in runs or ref is None:
                continue
            a, b = early_train_loss(runs[(t, seed)]), early_train_loss(ref)
            if a is not None and b is not None:
                gaps.append(a - b)
        if gaps:
            s[f"train_loss_ep{EARLY_EPOCH}_diff_mean_{t}"] = float(np.mean(gaps))
            s[f"n_seeds_train_loss_higher_{t}"] = int(sum(g > 0 for g in gaps))

    # Final model: the configuration selected on validation, on test.
    selected = None
    final_recs = [r for (t, seed), r in sorted(finals.items(),
                                                key=lambda kv: kv[0][1])
                  if seed in seeds]
    tags = {r["config"]["tag"] for r in final_recs}
    if len(tags) > 1:
        raise RuntimeError(f"final runs of several configurations: {tags}")
    trivial = dx._trivial_record(paths)
    dense, dense_summary = dense_finals(paths)
    if final_recs:
        selected = final_recs[0]["config"]["tag"]
        sources += [r["run_id"] for r in final_recs]
        s["final_n_params"] = int(final_recs[0]["scalars"]["n_params"])
        s["final_n_seeds"] = len(final_recs)
        s["final_rf_samples"] = int(receptive_field(BY_TAG[selected]))
        for key in ("val_macro_auroc", "test_macro_auroc", "test_macro_f1"):
            vals = [r["scalars"][key] for r in final_recs]
            s[f"final_{key}_mean"] = float(np.mean(vals))
            s[f"final_{key}_sd"] = dx._sd(vals)
        for c in config.CLASSES:
            vals = [r["scalars"][f"test_auroc_{c}"] for r in final_recs]
            s[f"final_test_auroc_{c}_mean"] = float(np.mean(vals))
            s[f"final_test_auroc_{c}_sd"] = dx._sd(vals)
        first = final_recs[0]["scalars"]
        s["final_first_seed"] = int(final_recs[0]["config"]["seed"])
        for key in ["test_macro_auroc", "test_macro_f1"] + \
                [f"test_auroc_{c}" for c in config.CLASSES]:
            s[f"final_first_{key}"] = first[key]
            s[f"final_first_{key}_lo"] = first[f"{key}_lo"]
            s[f"final_first_{key}_hi"] = first[f"{key}_hi"]
        for c in config.CLASSES:
            s[f"final_first_test_f1_{c}"] = first[f"test_f1_{c}"]
        # --fast runs see a random subset of the test fold, the trivial and
        # Dense final runs the whole fold: they cannot be paired then.
        if not fast:
            if trivial is not None and trivial.get("final"):
                compare_finals(final_recs, {r["config"]["seed"]: trivial
                                            for r in final_recs},
                               "trivial", n_boot, s)
            if dense:
                compare_finals(final_recs, dense, "dense", n_boot, s)
    if trivial is not None:
        s["trivial_val_macro_auroc"] = trivial["scalars"]["val_macro_auroc"]

    # Interpretation of the final model (first seed): first-layer filters
    # and the example records of the feature-map figure.
    if final_recs:
        first_final = {(selected, final_recs[0]["config"]["seed"]): final_recs[0]}
        s.update(first_layer_numbers(first_final))
        if cache is not None:
            s.update(example_numbers(first_final, cache))

    return {
        "run_id": SUMMARY_NAME + ("_fast" if fast else ""),
        "schema": runlog.SCHEMA_VERSION,
        "task": "cnn_block_summary",
        "fast": fast,
        "scalars": s,
        "seeds": list(seeds),
        "selected_config": selected,
        "selection_rule": "highest mean validation macro AUROC over the seeds",
        "baseline_args": BASE_ARGS,
        "ablations": [{"tag": a.tag, "factor": a.factor, "label": a.label,
                       "model_args": a.model_args, "reference": a.reference,
                       "receptive_field": receptive_field(a)}
                      for a in ABLATIONS],
        "trivial_run": None if trivial is None else trivial["run_id"],
        "dense_summary": dense_summary,
        "dense_final_runs": sorted(r["run_id"] for r in dense.values()),
        "source_runs": sources,
        "meta": runlog.environment(),
    }


# ---------------------------------------------------------------------------
# 4. LaTeX tables (made of macros only: no number is written here)
# ---------------------------------------------------------------------------
def _m(key: str, alias: str = ALIAS) -> str:
    """LaTeX macro of a summary key, e.g. val_auroc_mean_base ->
    \\cnnValAurocMeanBase{}."""
    return "\\" + dx.make_numbers_module().macro_name(alias, key) + "{}"


def ablation_table(summary: dict) -> str:
    s = summary["scalars"]
    lines = [
        "% Generated by src/cnn_experiments.py from results/logs/"
        f"{summary['run_id']}.json.",
        "% Needs report/numbers.tex (alias \"cnn\" in report/runs.json)"
        " and \\usepackage{booktabs}.",
        "% Delta: paired difference with the reference trained with the same"
        " seed (the baseline unless the label says otherwise).",
        "\\begin{tabular}{llrrcrc}",
        "\\toprule",
        "Factor & Configuration & Parameters & RF (samples) & "
        "Val.\\ macro AUROC & $\\Delta$ vs.\\ reference & "
        "Seeds better / worse \\\\",
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
        lines.append(f"{factor} & {ab.label} & {_m('params_' + t)} & "
                     f"{_m('rf_samples_' + t)} & {auroc} & {diff} & "
                     f"{seeds} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(lines)


def final_table(summary: dict) -> str | None:
    s = summary["scalars"]
    if "final_test_macro_auroc_mean" not in s:
        return None
    with_dense = summary.get("dense_summary") is not None
    cols = "lccc" + ("c" if with_dense else "")
    head = " & \\multicolumn{2}{c}{CNN (final)} & " + \
        ("Dense (final) & " if with_dense else "") + "Trivial \\\\"
    sub = "Test AUROC & First seed [95\\% CI] & Mean $\\pm$ SD (seeds) & " + \
        ("Mean $\\pm$ SD & " if with_dense else "") + " \\\\"
    lines = [
        "% Generated by src/cnn_experiments.py from results/logs/"
        f"{summary['run_id']}.json.",
        "% Needs report/numbers.tex (aliases \"cnn\", \"trivial\""
        + (" and \"dense\"" if with_dense else "") + ").",
        f"\\begin{{tabular}}{{{cols}}}",
        "\\toprule",
        head,
        "\\cmidrule(lr){2-3}",
        sub,
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
        cells = [name, first, spread]
        if with_dense:
            cells.append(f"{_m(mean_key + '_mean', 'dense')} $\\pm$ "
                         f"{_m(mean_key + '_sd', 'dense')}")
        cells.append(_m(key, "trivial"))
        lines.append(" & ".join(cells) + " \\\\")
        if name == "Macro":
            lines.append("\\midrule")
    # Paired differences in test macro AUROC, one row per seed.
    n_cols = len(cols)
    extra = []
    for other, title in (("dense", "Dense"), ("trivial", "trivial")):
        for seed in summary["seeds"]:
            key = f"diff_{other}_s{seed}"
            if key not in s:
                continue
            extra.append(
                f"$\\Delta$ macro vs.\\ {title}, seed {seed} & "
                f"{_m(key)} [{_m(f'diff_{other}_lo_s{seed}')}, "
                f"{_m(f'diff_{other}_hi_s{seed}')}]"
                + " &" * (n_cols - 2) + " \\\\")
    if extra:
        lines.append("\\midrule")
        lines += extra
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 5. Figures
# ---------------------------------------------------------------------------
def _diverging():
    import matplotlib
    return matplotlib.colors.LinearSegmentedColormap.from_list(
        "blue_grey_orange", ["#0d366b", "#2a78d6", "#f1f0ec", "#eb6834",
                             "#7a2c0b"])


def figure_curves(runs: dict, paths: Paths, suffix: str,
                  seed: int = SEEDS[0]) -> bool:
    """(a) baseline losses per epoch; (b) validation macro AUROC per epoch
    without batch norm, residuals or global pooling; (c) TRAINING loss with
    and without skip connections at 4 and 6 blocks (the slides' claim is
    about optimisation: deeper plain networks reach a higher training
    error, PRDL_MLLB_CNN.pptx slides 49-51)."""
    needed = ["base", "no_bn", "no_residual", "flatten", "depth6",
              "depth6_no_residual"]
    if any((t, seed) not in runs for t in needed):
        print("cnn_curves: some seed-42 runs are missing; figure skipped")
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

    for i, tag in enumerate(["base", "no_bn", "no_residual", "flatten"]):
        hh = runs[(tag, seed)]["history"]
        axes[1].plot(hh["epoch"], hh["val_macro_auroc"], color=col[i],
                     linewidth=2, label=BY_TAG[tag].label.split(":")[0])
    axes[1].set(title="(b) Validation AUROC", xlabel="epoch",
                ylabel="validation macro AUROC")
    axes[1].legend(loc="lower right", fontsize=7)

    pairs = [("base", "4 blocks, residual", 0, "-"),
             ("no_residual", "4 blocks, plain", 0, "--"),
             ("depth6", "6 blocks, residual", 1, "-"),
             ("depth6_no_residual", "6 blocks, plain", 1, "--")]
    for tag, name, c, ls in pairs:
        hh = runs[(tag, seed)]["history"]
        axes[2].plot(hh["epoch"], hh["train_loss"], color=col[c],
                     linewidth=2, linestyle=ls, label=name)
    axes[2].set(title="(c) Skip connections: training loss", xlabel="epoch",
                ylabel="BCE loss (training mode)")
    axes[2].legend(loc="upper right", fontsize=8)
    fig.suptitle(f"1D CNN, seed {seed}", color=eda.INK, fontsize=10)
    dx._save(fig, paths, "cnn_curves" + suffix)
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
        print("cnn_ablations: no baseline run; figure skipped")
        return False
    import matplotlib.pyplot as plt
    eda = dx._style()
    # One colour per group of factors, in the fixed order of the palette
    # (never cycled); the baseline is drawn in ink.
    groups = {"kernel size": 0, "number of blocks": 1, "batch norm": 2,
              "dropout": 2, "residual connections": 3,
              "residual connections x depth": 3,
              "global pooling vs flatten": 4}
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
    # The Dense and trivial models are not drawn: their much lower AUROC
    # would squeeze the differences between configurations; they are in
    # cnn_final.tex.
    ax.set_yticks(y, [ab.label.split(":")[0] for ab in order], fontsize=8)
    ax.set_ylim(y[-1] - 1.2, y[0] + 1.1)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(f"validation macro AUROC (dots: seeds "
                  f"{', '.join(map(str, seeds))}; bar: mean)")
    dx._save(fig, paths, "cnn_ablations" + suffix)
    return True


def load_final_model(finals: dict) -> tuple[CNN1D, dict]:
    """The final model of the first seed, in eval mode, on the CPU."""
    rec = finals[min(finals, key=lambda k: k[1])]
    model = build_model(rec["model"]["args"])
    state = torch.load(config.ROOT / rec["checkpoint"], map_location="cpu",
                       weights_only=True)
    model.load_state_dict(state)
    return model.eval(), rec


def filter_responses(weight: np.ndarray, n_fft: int = 256):
    """Magnitude frequency response of first-layer filters.

    weight: (filters, leads, k). Each filter applies one k-tap FIR filter
    to every lead and adds the results, so it has one response per lead;
    their RMS over leads summarises how strongly the filter passes each
    frequency. Returns (frequencies in Hz, |H| of shape (filters, bins)).
    """
    H = np.abs(np.fft.rfft(weight, n=n_fft, axis=-1))       # (F, leads, bins)
    freqs = np.fft.rfftfreq(n_fft, d=1 / config.SAMPLING_RATE)
    return freqs, np.sqrt((H ** 2).mean(axis=1))


PEAK_LOW_HZ, PEAK_HIGH_HZ = 5, 20     # bands used to count filter peaks


def first_layer_numbers(finals: dict) -> dict:
    """Peak frequency of every first-layer filter of the final model (the
    frequency where its RMS response is largest): median and how many peak
    below PEAK_LOW_HZ or above PEAK_HIGH_HZ. Empty if the checkpoint is
    missing (checkpoints are not committed)."""
    try:
        model, _ = load_final_model(finals)
    except FileNotFoundError as e:
        print(f"first-layer numbers skipped: {e}")
        return {}
    W = model.first_conv.weight.detach().numpy()
    freqs, R = filter_responses(W)
    peaks = freqs[R.argmax(axis=1)]
    return {"first_layer_n_filters": int(len(W)),
            "first_layer_peak_hz_median": float(np.median(peaks)),
            "first_layer_n_peak_below_5hz": int((peaks < PEAK_LOW_HZ).sum()),
            "first_layer_n_peak_above_20hz": int((peaks > PEAK_HIGH_HZ).sum())}


def example_numbers(finals: dict, cache: dict) -> dict:
    """For the example record of each class (feature-map figure): its
    ecg_id, the probability the final model gives to that class (from the
    run's saved validation predictions) and the class threshold chosen on
    validation (D14). A record above its threshold is classified as
    positive for that class."""
    rec = finals[min(finals, key=lambda k: k[1])]
    val = dx.load_predictions(rec, "val")
    out = {}
    for c, row in example_records(cache).items():
        eid = int(cache["ecg_id"][row])
        i = np.flatnonzero(val["ecg_id"] == eid)
        if len(i) != 1:                # e.g. --fast: not in the subset
            continue
        out[f"example_ecg_id_{c}"] = eid
        out[f"example_prob_{c}"] = float(val["prob"][i[0], config.CLASSES.index(c)])
        out[f"example_threshold_{c}"] = float(rec["scalars"][f"threshold_{c}"])
    return out


def figure_first_layer(finals: dict, paths: Paths, suffix: str,
                       n_filters: int = 8) -> bool:
    """(a) The first-layer filters with the largest weight norm, as
    leads x taps; (b) the frequency response of every first-layer filter,
    each normalised to its maximum and sorted by peak frequency: the first
    layer read as a learnt filter bank (PRDL_MLLB_CNN.pptx slides 99-102)."""
    if not finals:
        print("cnn_first_layer: no final run; figure skipped")
        return False
    model, rec = load_final_model(finals)
    W = model.first_conv.weight.detach().numpy()             # (F, leads, k)
    norms = np.linalg.norm(W.reshape(len(W), -1), axis=1)
    top = np.argsort(-norms)[:n_filters]
    freqs, R = filter_responses(W)
    R = R / R.max(axis=1, keepdims=True)
    order = np.argsort(freqs[R.argmax(axis=1)], kind="stable")

    import matplotlib.pyplot as plt
    eda = dx._style()
    fig = plt.figure(figsize=(11, 4.4))
    left, right = fig.subfigures(1, 2, width_ratios=[1.35, 1])
    lim = float(np.abs(W[top]).max())
    k = W.shape[2]
    n_cols = (n_filters + 1) // 2
    axes = left.subplots(2, n_cols, sharex=True, sharey=True)
    im = None
    for ax, f in zip(axes.flat, top):
        im = ax.imshow(W[f], aspect="auto", cmap=_diverging(), vmin=-lim,
                       vmax=lim, interpolation="nearest")
        ax.set_title(f"filter {f}", fontsize=8)
        ax.grid(False)
    axes[0, 0].set_yticks(range(len(config.LEADS)), config.LEADS)
    axes[0, 0].set_xticks([0, k - 1], ["0", f"{k - 1}"])
    for ax in axes.flat:              # shared axes: same size in every row
        ax.tick_params(axis="y", labelsize=6)
        ax.tick_params(axis="x", labelsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("tap", fontsize=8)
    left.colorbar(im, ax=axes, label="weight", shrink=0.9)
    left.suptitle(f"(a) The {n_filters} filters with the largest norm "
                  "(leads x taps)", fontsize=10, color=eda.INK)
    ax = right.subplots()
    resp = ax.imshow(R[order], aspect="auto", cmap=eda.SEQUENTIAL,
                     extent=[freqs[0], freqs[-1], len(R) - 0.5, -0.5],
                     interpolation="nearest", vmin=0, vmax=1)
    ax.set(xlabel="frequency (Hz)", ylabel="filter (sorted by peak frequency)")
    ax.set_yticks([])
    ax.grid(False)
    right.colorbar(resp, ax=ax, label="|H(f)| / max")
    right.suptitle("(b) Frequency response of every filter, RMS over leads",
                   fontsize=10, color=eda.INK)
    fig.suptitle(f"First convolutional layer of the final CNN "
                 f"({rec['config']['tag']}, seed {rec['config']['seed']})",
                 color=eda.INK, fontsize=10)
    dx._save(fig, paths, "cnn_first_layer" + suffix)
    return True


def example_records(cache: dict, seed: int = config.SEED) -> dict:
    """class -> cache row of one validation record with that superclass
    only, drawn with the seed (D16). Classes without such a record are
    left out."""
    from src.dataset import split_indices
    val = split_indices(cache)["val"]
    y = cache["y"][val]
    single = y.sum(axis=1) == 1
    rng = np.random.default_rng(seed)
    out = {}
    for j, c in enumerate(config.CLASSES):
        rows = val[single & (y[:, j] == 1)]
        if len(rows):
            out[c] = int(rng.choice(rows))
    return out


def capture_maps(model: CNN1D, x: torch.Tensor) -> dict:
    """Outputs of the first and the last block for the batch x, collected
    with forward hooks (functions PyTorch calls after a module's forward),
    plus the class activation maps when the head uses global pooling."""
    maps = {}

    def keep(name):
        def hook(module, inputs, output):
            maps[name] = output.detach().cpu().numpy()
        return hook

    handles = [model.blocks[0].register_forward_hook(keep("first")),
               model.blocks[-1].register_forward_hook(keep("last"))]
    try:
        with torch.no_grad():
            maps["logits"] = model(x).cpu().numpy()
            if model.pooling == "gap":
                maps["cam"] = model.class_activation_map(x).cpu().numpy()
    finally:
        for h in handles:             # hooks stay until removed
            h.remove()
    return maps


def figure_feature_maps(finals: dict, paths: Paths, suffix: str,
                        cache: dict | None) -> bool:
    """One validation record per superclass (columns): lead II of the
    model input; feature maps of the first and the last block (channels x
    time, channels ordered by mean activation); class activation map of the
    record's class over lead II (only with global average pooling)."""
    if not finals or cache is None:
        print("cnn_feature_maps: no final run or no cache; figure skipped")
        return False
    model, rec = load_final_model(finals)
    examples = example_records(cache)
    if not examples:
        print("cnn_feature_maps: no single-class validation record; skipped")
        return False
    classes = list(examples)
    X = cache["X"][[examples[c] for c in classes]].astype(np.float32)
    mean, std = rec["data"].get("lead_mean"), rec["data"].get("lead_std")
    if mean is not None:
        X = (X - np.asarray(mean, np.float32)[:, None]) / \
            np.asarray(std, np.float32)[:, None]
    maps = capture_maps(model, torch.from_numpy(X))
    lead2 = config.LEADS.index("II")
    duration = X.shape[2] / config.SAMPLING_RATE

    import matplotlib.pyplot as plt
    eda = dx._style()
    rows = 4 if "cam" in maps else 3
    fig, axes = plt.subplots(rows, len(classes),
                             figsize=(2.3 * len(classes) + 0.6, 1.6 * rows + 0.6),
                             squeeze=False, sharex=True)
    t = np.arange(X.shape[2]) / config.SAMPLING_RATE
    for name, row in (("first", 1), ("last", 2)):
        A = maps[name]                                     # (n, C, L)
        order = np.argsort(-A.mean(axis=(0, 2)))           # same for all
        vmax = float(np.percentile(A, 99)) or 1.0
        for j in range(len(classes)):
            ax = axes[row, j]
            ax.imshow(A[j][order], aspect="auto", cmap=eda.SEQUENTIAL,
                      vmin=0, vmax=vmax, interpolation="nearest",
                      extent=[0, duration, A.shape[1] - 0.5, -0.5])
            ax.grid(False)
            ax.set_yticks([])
        axes[row, 0].set_ylabel(f"{'block 1' if name == 'first' else 'last block'}"
                                f"\n({A.shape[1]} ch. x {A.shape[2]})",
                                fontsize=8)
    prob = 1 / (1 + np.exp(-maps["logits"]))             # sigmoid, outside
    for j, c in enumerate(classes):
        ax = axes[0, j]
        k = config.CLASSES.index(c)
        thr = rec["scalars"].get(f"threshold_{c}")
        verdict = "" if thr is None else (
            " > " if prob[j, k] > thr else " ≤ ") + f"threshold {thr:.2f}"
        ax.plot(t, X[j, lead2], color=eda.CLASS_COLOURS[c], linewidth=0.8)
        ax.set_title(f"{c} (ecg_id {int(cache['ecg_id'][examples[c]])})\n"
                     f"p({c}) = {prob[j, k]:.2f}{verdict}", fontsize=9)
    axes[0, 0].set_ylabel("lead II\n(z-score)", fontsize=8)
    if "cam" in maps:
        cam = maps["cam"]                                  # (n, classes, L)
        for j, c in enumerate(classes):
            ax = axes[3, j]
            m = cam[j, config.CLASSES.index(c)]
            lim = float(np.abs(m).max()) or 1.0
            lo, hi = X[j, lead2].min(), X[j, lead2].max()
            ax.imshow(m[None, :], aspect="auto", cmap=_diverging(),
                      vmin=-lim, vmax=lim, extent=[0, duration, lo, hi],
                      alpha=0.8, interpolation="nearest")
            ax.plot(t, X[j, lead2], color=eda.INK, linewidth=0.7)
            ax.set_ylim(lo, hi)
            ax.grid(False)
        axes[3, 0].set_ylabel("CAM of the record's class\n"
                              "(orange +, blue −)", fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("time (s)")
    fig.suptitle(f"Feature maps of the final CNN ({rec['config']['tag']}, "
                 f"seed {rec['config']['seed']}), one validation record per "
                 "superclass", color=eda.INK, fontsize=10)
    dx._save(fig, paths, "cnn_feature_maps" + suffix)
    return True


# ---------------------------------------------------------------------------
# 6. report: summary log + tables + figures
# ---------------------------------------------------------------------------
def write_report(fast: bool = False, paths: Paths = Paths(), seeds=SEEDS,
                 cache: dict | None = None) -> dict:
    suffix = "_fast" if fast else ""
    if find_runs(paths, fast=fast, final=True):
        cache = dx._load_cache(cache)   # signals of the example records
    summary = build_summary(paths, fast, seeds, cache)
    path = runlog.write_run(summary, paths.logs)
    print(f"summary: {rel(path)} ({len(summary['scalars'])} numbers)")

    paths.tables.mkdir(parents=True, exist_ok=True)
    (paths.tables / f"cnn_ablations{suffix}.tex").write_text(
        ablation_table(summary), encoding="utf-8")
    final = final_table(summary)
    if final is not None:
        (paths.tables / f"cnn_final{suffix}.tex").write_text(
            final, encoding="utf-8")
    print(f"tables: {rel(paths.tables)}/cnn_*{suffix}.tex")

    runs = find_runs(paths, fast=fast, final=False)
    finals = find_runs(paths, fast=fast, final=True)
    figure_curves(runs, paths, suffix, seeds[0])
    figure_ablations(runs, paths, suffix, seeds)
    figure_first_layer(finals, paths, suffix)
    figure_feature_maps(finals, paths, suffix, cache)
    print(f"figures: {rel(paths.figures)}/cnn_*{suffix}.pdf|png")

    if not fast:
        _register_alias(paths)
    return summary


def _register_alias(paths: Paths) -> None:
    """Add "cnn": "cnn_summary" to report/runs.json if it is missing."""
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
        cache = dx._load_cache(None)
        run_ablations(a.seeds, a.only, a.fast, cache)
    if a.step in ("final", "all"):
        run_final(a.fast, cache, seeds=a.seeds)
    if a.step in ("report", "all"):
        write_report(a.fast, seeds=a.seeds, cache=cache)
    print(f"\ndone in {datetime.now() - t0}")


if __name__ == "__main__":
    main()
