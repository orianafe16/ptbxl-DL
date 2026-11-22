"""Cross-cutting ablation (T14): 12 leads vs lead II alone.

Decisions: docs/DECISIONS.md D20 (this block), D6 (the ablation), D15-D18
(the final model of each family), D3, D4 and D14 (protocol).

    python -m src.leads_experiments all --fast   # smoke test
    python -m src.leads_experiments run          # 5 families x 3 seeds
    python -m src.leads_experiments report       # summary, table, figure

The final model of every family (Dense, CNN, RNN, RNN + attention,
Transformer), as selected on validation by its own block, is trained again
on lead II only (n_leads = 1): same architecture, input scaling, optimiser,
learning rate, gradient clipping, batch, early stopping and seeds as its
12-lead final runs. Nothing is selected here, so each run is a --final run
(validation and test fold in the same run, D20). Run names start with
`onelead_`, so these runs never mix with the 12-lead runs of each block.

Lead II: the lead of the ECG-GPT (D13), so that the one-lead classifiers
are the reference for the transfer experiments of T13.

Outputs of `report`:
    results/logs/leads_summary.json   numbers (alias "leads" in runs.json)
    report/tables/leads.tex           12 leads vs lead II per family
    report/figures/leads_comparison.* the same, one line per seed
With --fast every output name ends in _fast and git ignores it.
"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:2")   # as D17-D19

import argparse  # noqa: E402
from datetime import datetime  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src import cnn_experiments as cx  # noqa: E402
from src import config, runlog  # noqa: E402
from src import dense_experiments as dx  # noqa: E402
from src import transfer_experiments as tx  # noqa: E402   generic helpers
from src.dense_experiments import Paths  # noqa: E402
from src.models import MODELS  # noqa: E402
from src.train import TrainConfig, ptbxl_setup, rel, run_experiment  # noqa: E402

MODEL = "onelead"                     # run names: onelead_<family>_<tag>_...
SEEDS = (42, 43, 44)
ALIAS = "leads"
SUMMARY_NAME = "leads_summary"
LEAD = "II"

# (key, label, model of the block, alias of the block summary, summary key
# holding the selected configuration)
FAMILIES = [
    ("dense", "Dense", "dense", "dense", "selected_config"),
    ("cnn", "CNN", "cnn", "cnn", "selected_config"),
    ("rnn", "RNN", "rnn", "rnn", "selected_config"),
    ("rnnattn", "RNN + attention", "rnn", "rnn", "selected_attention_config"),
    ("transformer", "Transformer", "transformer", "transformer",
     "selected_config"),
]
BY_KEY = {f[0]: f for f in FAMILIES}
# Training settings copied from the 12-lead final run (everything else in
# TrainConfig keeps its default, as in those runs).
COPIED = ("lr", "weight_decay", "batch_size", "max_epochs", "patience",
          "grad_clip", "deterministic", "normalize", "pos_weight")


# ---------------------------------------------------------------------------
# 1. The 12-lead final models
# ---------------------------------------------------------------------------
def selected_tag(paths: Paths, family: str, fast: bool = False) -> str | None:
    """Configuration that the family's own block selected on validation."""
    _, _, _, alias, key = BY_KEY[family]
    summary = cx._alias_log(paths, alias)
    if fast:
        try:
            summary = runlog.load_run(f"{alias}_summary_fast", paths.logs)
        except OSError:
            pass
    if summary is None:
        return None
    return summary.get(key) or (summary.get("selected_config")
                                if family == "rnnattn" else None)


def twelve_lead_finals(paths: Paths, family: str, fast: bool = False) -> dict:
    """seed -> 12-lead final (test) run of the family's selected model."""
    tag = selected_tag(paths, family, fast)
    if tag is None:
        return {}
    model = BY_KEY[family][2]
    found = cx.find_runs(paths, fast, True, model=model)
    return {seed: r for (t, seed), r in found.items() if t == tag}


def lead_cache(cache: dict, lead: str = LEAD) -> dict:
    """The same cache with X restricted to one lead, (N, 1, n_samples).
    Every other array is shared, so the split, labels, patients and order
    of the records are those of the 12-lead runs."""
    j = config.LEADS.index(lead)
    out = dict(cache)
    out["X"] = np.ascontiguousarray(cache["X"][:, j:j + 1, :])
    return out


def make_config(family: str, twelve: dict, seed: int,
                fast: bool = False) -> TrainConfig:
    """TrainConfig of the one-lead run: the 12-lead final run's settings
    with n_leads = 1 (D20)."""
    c = twelve["config"]
    args = {**c["model_args"], "n_leads": 1}
    cfg = TrainConfig(model=MODEL, tag=f"{family}_{c['tag']}", seed=seed,
                      model_args=args, fast=fast, final=True,
                      notes=f"{BY_KEY[family][1]} final model "
                            f"({c['tag']}) on lead {LEAD} only")
    for k in COPIED:
        if k in c:
            setattr(cfg, k, c[k])
    if fast:                          # D14: 2 epochs, 50 resamples
        cfg.max_epochs, cfg.n_boot = 2, 50
    return cfg


def find_runs(paths: Paths, fast: bool) -> dict:
    """(family, seed) -> latest one-lead run."""
    out = {}
    for (tag, seed), rec in cx.find_runs(paths, fast, True, model=MODEL).items():
        out[(tag.split("_", 1)[0], seed)] = rec
    return out


# ---------------------------------------------------------------------------
# 2. Runs
# ---------------------------------------------------------------------------
def _train(family: str, twelve: dict, seed: int, fast: bool, cache: dict,
           paths: Paths) -> dict:
    cfg = make_config(family, twelve, seed, fast)
    loaders, info, task = ptbxl_setup(cfg, cache)
    info["lead"] = LEAD
    info["twelve_lead_run"] = twelve["run_id"]
    info["cublas_workspace_config"] = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    model_class = MODELS[BY_KEY[family][2]]
    _, record = run_experiment(lambda: model_class(**cfg.model_args), loaders,
                               task, cfg, info, logs_dir=paths.logs,
                               checkpoint_dir=paths.checkpoints,
                               predictions_dir=paths.predictions)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return record


def run_all(seeds=SEEDS, only=None, fast: bool = False,
            cache: dict | None = None, paths: Paths = Paths()) -> None:
    families = only or [f[0] for f in FAMILIES]
    unknown = set(families) - set(BY_KEY)
    if unknown:
        raise ValueError(f"unknown families: {sorted(unknown)}")
    cache = lead_cache(dx._load_cache(cache))
    done = find_runs(paths, fast=False)
    todo = [(f, s) for f in families for s in seeds]
    for i, (family, seed) in enumerate(todo, 1):
        twelve = twelve_lead_finals(paths, family, fast).get(seed)
        if twelve is None:
            raise RuntimeError(f"no 12-lead final run of {family} seed {seed}")
        if not fast and (family, seed) in done:
            print(f"[{i}/{len(todo)}] {family} seed {seed}: already run "
                  f"({done[(family, seed)]['run_id']}), skipped")
            continue
        print(f"\n[{i}/{len(todo)}] {family} ({twelve['config']['tag']}) "
              f"seed {seed}, lead {LEAD} only")
        _train(family, twelve, seed, fast, cache, paths)


# ---------------------------------------------------------------------------
# 3. Summary, table, figure
# ---------------------------------------------------------------------------
def build_summary(paths: Paths = Paths(), fast: bool = False,
                  seeds=SEEDS) -> dict:
    runs = find_runs(paths, fast)
    n_boot = 50 if fast else 1000
    s: dict = {}
    sources, selected = [], {}
    for family, *_ in FAMILIES:
        one = [runs[(family, x)] for x in seeds if (family, x) in runs]
        twelve_by_seed = twelve_lead_finals(paths, family, fast)
        twelve = [twelve_by_seed[x] for x in seeds if x in twelve_by_seed]
        if not one:
            continue
        selected[family] = one[0]["config"]["tag"].split("_", 1)[1]
        sources += [r["run_id"] for r in one]
        s[f"params_onelead_{family}"] = int(one[0]["scalars"]["n_params"])
        if twelve:
            s[f"params_twelve_{family}"] = int(twelve[0]["scalars"]["n_params"])
        s[f"best_epoch_mean_onelead_{family}"] = float(np.mean(
            [r["scalars"]["best_epoch"] for r in one]))
        for split in ("val", "test"):
            tx.split_numbers(one, split, f"onelead_{family}", s)
            tx.split_numbers(twelve, split, f"twelve_{family}", s)
            tx.paired_numbers(family, one, twelve, split, n_boot, s)
    return {
        "run_id": SUMMARY_NAME + ("_fast" if fast else ""),
        "schema": runlog.SCHEMA_VERSION,
        "task": "leads_summary",
        "fast": fast,
        "scalars": s,
        "seeds": list(seeds),
        "lead": LEAD,
        "selected_configs": selected,
        "comparison": "paired patient bootstrap, lead II - 12 leads, same "
                      "family, same seed (D20)",
        "source_runs": sources,
        "meta": runlog.environment(),
    }


def _m(key: str) -> str:
    return tx._m(key, ALIAS)


def table(summary: dict) -> str | None:
    s = summary["scalars"]
    rows = []
    for family, label, *_ in FAMILIES:
        if f"test_auroc_mean_onelead_{family}" not in s:
            continue
        tag = summary["selected_configs"][family].replace("_", "\\_")
        diff = (f"{_m('test_diff_mean_' + family)} & "
                f"{_m('test_n_better_' + family)} / "
                f"{_m('test_n_worse_' + family)} of "
                f"{_m('test_n_compared_' + family)}"
                if f"test_diff_mean_{family}" in s else "--- & ---")
        twelve = tx._pm(s, f"test_auroc_mean_twelve_{family}",
                        f"test_auroc_sd_twelve_{family}", ALIAS)
        one = tx._pm(s, f"test_auroc_mean_onelead_{family}",
                     f"test_auroc_sd_onelead_{family}", ALIAS)
        params = (f"{_m('params_twelve_' + family)} / "
                  if f"params_twelve_{family}" in s else "--- / ")
        rows.append(f"{label} & \\texttt{{{tag}}} & {params}"
                    f"{_m('params_onelead_' + family)} & {twelve} & {one} & "
                    f"{diff} \\\\")
    if not rows:
        return None
    return "\n".join([
        "% Generated by src/leads_experiments.py from results/logs/"
        f"{summary['run_id']}.json.",
        "% Needs report/numbers.tex (alias \"leads\") and booktabs.",
        "\\begin{tabular}{llrccrc}",
        "\\toprule",
        "Family & Final config. & Param.\\ (12 / 1) & Test AUROC, 12 leads & "
        "Test AUROC, lead II & $\\Delta$ & Seeds better / worse \\\\",
        "\\midrule", *rows, "\\bottomrule", "\\end{tabular}", ""])


def figure(paths: Paths, suffix: str, fast: bool, seeds=SEEDS) -> bool:
    """Test macro AUROC with 12 leads and with lead II, per family: one
    line per seed joining the two runs of the same seed."""
    runs = find_runs(paths, fast)
    fams = [f for f in FAMILIES if any((f[0], x) in runs for x in seeds)]
    if not fams:
        print("leads_comparison: no one-lead run; figure skipped")
        return False
    import matplotlib.pyplot as plt
    eda = dx._style()
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    for i, (family, label, *_) in enumerate(fams):
        twelve = twelve_lead_finals(paths, family, fast)
        for x in seeds:
            if (family, x) not in runs or x not in twelve:
                continue
            a = twelve[x]["scalars"]["test_macro_auroc"]
            b = runs[(family, x)]["scalars"]["test_macro_auroc"]
            ax.plot([i - 0.15, i + 0.15], [a, b], color=eda.INK_MUTED,
                    linewidth=0.8, zorder=1)
            ax.scatter([i - 0.15], [a], s=22, color=eda.SERIES_COLOURS[0],
                       zorder=3, label="12 leads" if i == 0 and x == seeds[0]
                       else None)
            ax.scatter([i + 0.15], [b], s=22, color=eda.SERIES_COLOURS[1],
                       zorder=3, label="lead II" if i == 0 and x == seeds[0]
                       else None)
    ax.set_xticks(range(len(fams)), [f[1] for f in fams], fontsize=8)
    ax.set_ylabel("test macro AUROC (one line per seed)")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("Final model of each family: 12 leads vs lead II",
                 fontsize=10)
    dx._save(fig, paths, "leads_comparison" + suffix)
    return True


def write_report(fast: bool = False, paths: Paths = Paths(),
                 seeds=SEEDS) -> dict:
    suffix = "_fast" if fast else ""
    summary = build_summary(paths, fast, seeds)
    path = runlog.write_run(summary, paths.logs)
    print(f"summary: {rel(path)} ({len(summary['scalars'])} numbers)")
    text = table(summary)
    if text is not None:
        paths.tables.mkdir(parents=True, exist_ok=True)
        (paths.tables / f"leads{suffix}.tex").write_text(text,
                                                         encoding="utf-8")
    figure(paths, suffix, fast, seeds)
    if not fast:
        tx.register_alias(paths, ALIAS, SUMMARY_NAME)
    return summary


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("step", choices=["run", "report", "all"])
    p.add_argument("--fast", action="store_true")
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    p.add_argument("--only", nargs="+", default=None,
                   help="families: " + " ".join(f[0] for f in FAMILIES))
    a = p.parse_args(argv)
    t0 = datetime.now()
    if a.step in ("run", "all"):
        run_all(a.seeds, a.only, a.fast)
    if a.step in ("report", "all"):
        write_report(a.fast, seeds=a.seeds)
    print(f"\ndone in {datetime.now() - t0}")


if __name__ == "__main__":
    main()
