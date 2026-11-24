"""Final comparison (T15): every final model on the test fold.

Decisions: docs/DECISIONS.md D20 (this summary), D15-D20 (the models).

    python -m src.comparison report [--fast]

Nothing is trained here. The script collects the test results of every
final model already run - trivial, Dense, CNN, RNN, RNN + attention and
Transformer with 12 leads (D15-D18), the same five with lead II only (T14),
the six ECG-GPT transfer configurations (T13) and zero-shot CLIP (T24) -
into one summary, one table, a per-class table and two figures. It adds
the cross-block paired comparisons fixed in D20 and, if chat DL 09 has
filled report/published.json, published results quoted from the
literature (the only numbers typed by hand, each with file, page and
literal quote).

Outputs:
    results/logs/comparison_summary.json  (alias "comparison" in runs.json)
    report/tables/comparison_final.tex, comparison_per_class.tex
    report/figures/comparison_overview.*, comparison_label_efficiency.*
"""
import argparse
import json
import re

import numpy as np

from src import cnn_experiments as cx
from src import config, runlog
from src import dense_experiments as dx
from src import leads_experiments as lx
from src import transfer_experiments as tx
from src.dense_experiments import Paths
from src.train import rel

ALIAS = "comparison"
SUMMARY_NAME = "comparison_summary"
SEEDS = (42, 43, 44)
PUBLISHED_FILE = config.REPORT_DIR / "published.json"


def _alias_log(paths: Paths, alias: str, fast: bool, fast_name: str):
    if fast:
        try:
            return runlog.load_run(fast_name, paths.logs)
        except OSError:
            return None
    return cx._alias_log(paths, alias)


def rows(paths: Paths, fast: bool, seeds=SEEDS) -> list[dict]:
    """Every final model: key, group, label, input, labels and its runs
    (one per seed; one run for the trivial model and zero-shot)."""
    out = []
    trivial = None if fast else dx._trivial_record(paths)
    if trivial is not None and trivial.get("final"):
        out.append({"key": "trivial", "group": "twelve",
                    "label": "Trivial (lead mean and SD)", "leads": "12",
                    "labels": "100\\%", "runs": [trivial]})
    for family, label, *_ in lx.FAMILIES:
        twelve = lx.twelve_lead_finals(paths, family, fast)
        recs = [twelve[x] for x in seeds if x in twelve]
        if recs:
            out.append({"key": f"twelve_{family}", "group": "twelve",
                        "label": label, "leads": "12", "labels": "100\\%",
                        "runs": recs})
    one = lx.find_runs(paths, fast)
    for family, label, *_ in lx.FAMILIES:
        recs = [one[(family, x)] for x in seeds if (family, x) in one]
        if recs:
            out.append({"key": f"onelead_{family}", "group": "onelead",
                        "label": label, "leads": "II", "labels": "100\\%",
                        "runs": recs})
    tr = tx.all_runs(paths, fast)
    for f in tx.FRACTIONS:
        for st in tx.STRATEGIES:
            tag = f"{st}_f{tx.pct(f)}"
            recs = tx.recs_of(tr, tag, seeds)
            if recs:
                out.append({"key": tag, "group": "transfer",
                            "label": f"ECG-GPT, {tx.STRATEGY_LABELS[st].lower()}",
                            "leads": "II", "labels": f"{tx.pct(f)}\\%",
                            "runs": recs})
    zs = _alias_log(paths, "zeroshot", fast, "zeroshot_fast")
    if zs is not None:
        out.append({"key": "zeroshot", "group": "zeroshot",
                    "label": "CLIP ViT-B/32, zero-shot", "leads": "II",
                    "labels": "0\\%", "runs": [zs]})
    return out


def cross_comparisons() -> list[tuple[str, str, str, str]]:
    """(name, row A, row B, label), fixed in D20: each ECG-GPT configuration
    against the lead-II version of the final CNN (the strongest 12-lead
    family, D16), the lead-II CNN against the 12-lead CNN, and zero-shot
    CLIP against the trivial model."""
    out = [(f"{st}_f{tx.pct(f)}_vs_onelead_cnn", f"{st}_f{tx.pct(f)}",
            "onelead_cnn",
            f"ECG-GPT {tx.STRATEGY_LABELS[st].lower()}, {tx.pct(f)}\\% labels"
            " $-$ CNN, lead II, 100\\%")
           for f in tx.FRACTIONS for st in tx.STRATEGIES]
    out.append(("zeroshot_vs_trivial", "zeroshot", "trivial",
                "zero-shot CLIP $-$ trivial"))
    return out


def published(path=PUBLISHED_FILE) -> list[dict]:
    """Published results quoted by chat DL 09 (D20): a list of
    {key, label, value, metric, setting, source, file, page, quote}. The
    file is optional; entries without a literal quote, file and page are
    refused."""
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        entries = json.load(f).get("entries", [])
    for e in entries:
        missing = [k for k in ("key", "label", "value", "source", "file",
                               "page", "quote") if not e.get(k)]
        if missing:
            raise ValueError(f"published entry {e.get('key')}: missing "
                             f"{missing}")
        if not re.fullmatch(r"[A-Za-z0-9_]+", e["key"]):
            raise ValueError(f"published key {e['key']!r}: letters, digits, _")
    return entries


def build_summary(paths: Paths = Paths(), fast: bool = False, seeds=SEEDS,
                  published_path=PUBLISHED_FILE) -> dict:
    table_rows = rows(paths, fast, seeds)
    n_boot = 50 if fast else 1000
    s: dict = {}
    for r in table_rows:
        k, recs = r["key"], r["runs"]
        s[f"n_seeds_{k}"] = len(recs)
        if "n_trainable" in recs[0].get("scalars", {}):
            s[f"params_{k}"] = int(recs[0]["scalars"]["n_trainable"])
        tx.split_numbers(recs, "test", k, s)
    by_key = {r["key"]: r for r in table_rows}
    for name, a, b, _ in cross_comparisons():
        if a in by_key and b in by_key:
            tx.paired_numbers(name, by_key[a]["runs"], by_key[b]["runs"],
                              "test", n_boot, s)
    pubs = published(published_path)
    for e in pubs:
        s[f"published_{e['key']}"] = float(e["value"])
    return {
        "run_id": SUMMARY_NAME + ("_fast" if fast else ""),
        "schema": runlog.SCHEMA_VERSION,
        "task": "comparison_summary",
        "fast": fast,
        "scalars": s,
        "seeds": list(seeds),
        "rows": [{k: v for k, v in r.items() if k != "runs"}
                 | {"run_ids": [x["run_id"] for x in r["runs"]]}
                 for r in table_rows],
        "cross_comparisons": [{"name": n, "a": a, "b": b}
                              for n, a, b, _ in cross_comparisons()],
        "published": pubs,
        "meta": runlog.environment(),
    }


# ---------------------------------------------------------------------------
# Tables and figures
# ---------------------------------------------------------------------------
GROUPS = [("twelve", "12 leads"), ("onelead", "Lead II only (T14)"),
          ("transfer", "Lead II, ECG-GPT transfer (T13)"),
          ("zeroshot", "Lead II, zero-shot (T24)")]


def _m(key: str) -> str:
    return tx._m(key, ALIAS)


def final_table(summary: dict) -> str | None:
    s = summary["scalars"]
    if not summary["rows"]:
        return None
    lines = [
        "% Generated by src/comparison.py from results/logs/"
        f"{summary['run_id']}.json.",
        "% Mean +- SD over seeds (one run: no SD); CI = 95% patient "
        "bootstrap of the first seed. Needs alias \"comparison\".",
        "\\begin{tabular}{lcrrccc}", "\\toprule",
        "Model & Labels & Train.\\ param. & Seeds & Test macro AUROC & "
        "First seed [95\\% CI] & Test macro F1 \\\\", "\\midrule"]
    first = True
    for group, title in GROUPS:
        rs = [r for r in summary["rows"] if r["group"] == group]
        if not rs:
            continue
        if not first:
            lines.append("\\addlinespace")
        first = False
        lines.append(f"\\multicolumn{{7}}{{l}}{{\\textit{{{title}}}}} \\\\")
        for r in rs:
            k = r["key"]
            params = _m(f"params_{k}") if f"params_{k}" in s else "---"
            auroc = tx._pm(s, f"test_auroc_mean_{k}", f"test_auroc_sd_{k}",
                           ALIAS)
            f1 = tx._pm(s, f"test_f1_mean_{k}", f"test_f1_sd_{k}", ALIAS)
            ci = (f"{_m('test_auroc_first_' + k)} [{_m('test_auroc_lo_first_' + k)}"
                  f", {_m('test_auroc_hi_first_' + k)}]")
            lines.append(f"\\quad {r['label']} & {r['labels']} & {params} & "
                         f"{_m('n_seeds_' + k)} & {auroc} & {ci} & {f1} \\\\")
    pubs = summary.get("published") or []
    if pubs:
        lines += ["\\addlinespace",
                  "\\multicolumn{7}{l}{\\textit{Published (quoted, "
                  "different settings: see text)}} \\\\"]
        for e in pubs:
            lines.append(f"\\quad {e['label']}~\\cite{{{e['source']}}} & & & & "
                         f"{_m('published_' + e['key'])} & & \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(lines)


def per_class_table(summary: dict) -> str | None:
    if not summary["rows"]:
        return None
    head = " & ".join(config.CLASSES)
    lines = [
        "% Generated by src/comparison.py: test AUROC per superclass, mean "
        "over seeds.",
        "\\begin{tabular}{l" + "c" * len(config.CLASSES) + "}", "\\toprule",
        f"Model & {head} \\\\", "\\midrule"]
    for group, title in GROUPS:
        rs = [r for r in summary["rows"] if r["group"] == group]
        if not rs:
            continue
        lines.append(f"\\multicolumn{{{len(config.CLASSES) + 1}}}{{l}}"
                     f"{{\\textit{{{title}}}}} \\\\")
        for r in rs:
            k = r["key"]
            cells = " & ".join(_m(f"test_auroc_{c}_mean_{k}")
                               for c in config.CLASSES)
            label = r["label"] + (f", {r['labels']}" if group == "transfer"
                                  else "")
            lines.append(f"\\quad {label} & {cells} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(lines)


def cross_table(summary: dict) -> str | None:
    s = summary["scalars"]
    body = []
    for name, _, _, label in cross_comparisons():
        if f"test_diff_mean_{name}" not in s:
            continue
        body.append(f"{label} & {_m('test_diff_mean_' + name)} & "
                    f"{_m('test_n_better_' + name)} / "
                    f"{_m('test_n_worse_' + name)} of "
                    f"{_m('test_n_compared_' + name)} \\\\")
    if not body:
        return None
    return "\n".join([
        "% Generated by src/comparison.py: paired patient-bootstrap "
        "differences on the test fold (D20).",
        "\\begin{tabular}{lcc}", "\\toprule",
        "Comparison & $\\Delta$ macro AUROC & Seeds better / worse \\\\",
        "\\midrule", *body, "\\bottomrule", "\\end{tabular}", ""])


def figure_overview(paths: Paths, summary: dict, fast: bool, suffix: str,
                    seeds=SEEDS) -> bool:
    """Test macro AUROC of every final model: dots per seed, a bar for the
    mean, grouped as in the table."""
    table_rows = rows(paths, fast, seeds)
    if not table_rows:
        print("comparison_overview: nothing to draw")
        return False
    import matplotlib.pyplot as plt
    eda = dx._style()
    colour = dict(zip([g for g, _ in GROUPS], eda.SERIES_COLOURS))
    order = [r for g, _ in GROUPS for r in table_rows if r["group"] == g]
    fig, ax = plt.subplots(figsize=(7.5, 0.3 * len(order) + 1.3))
    y = np.arange(len(order))[::-1]
    for yi, r in zip(y, order):
        v = [x["scalars"]["test_macro_auroc"] for x in r["runs"]]
        c = colour[r["group"]]
        ax.plot([np.mean(v)] * 2, [yi - 0.28, yi + 0.28], color=c,
                linewidth=2.5, solid_capstyle="round")
        ax.scatter(v, [yi] * len(v), s=20, color=c, alpha=0.8,
                   edgecolors="white", linewidths=0.8, zorder=3)
    labels = [r["label"] + (f" ({r['labels'].replace(chr(92), '')})"
                            if r["group"] == "transfer" else "")
              + ("" if r["group"] in ("transfer", "zeroshot")
                 else f" [{r['leads']}]") for r in order]
    ax.set_yticks(y, labels, fontsize=8)
    ax.axvline(0.5, color=eda.INK_MUTED, linewidth=1, linestyle="--")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("test macro AUROC (dots: seeds; bar: mean; dashed: chance)")
    handles = [plt.Line2D([], [], color=colour[g], linewidth=2.5, label=t)
               for g, t in GROUPS if any(r["group"] == g for r in order)]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5),
              fontsize=7)
    dx._save(fig, paths, "comparison_overview" + suffix)
    return True


def figure_label_efficiency(paths: Paths, fast: bool, suffix: str,
                            seeds=SEEDS) -> bool:
    """Test macro AUROC against the share of training labels: zero-shot
    CLIP at 0 %, the three ECG-GPT strategies at 10 % and 100 %, and the
    final CNN with lead II and with 12 leads (100 %) as references."""
    tr = tx.all_runs(paths, fast)
    if not any(tx.recs_of(tr, t, seeds) for t in tx.MAIN_TAGS):
        print("comparison_label_efficiency: no transfer run; figure skipped")
        return False
    import matplotlib.pyplot as plt
    eda = dx._style()
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    xpos = {0.0: 0, 0.1: 1, 1.0: 2}
    for j, st in enumerate(tx.STRATEGIES):
        pts = []
        for f in sorted(tx.FRACTIONS):
            v = [r["scalars"].get("test_macro_auroc")
                 for r in tx.recs_of(tr, f"{st}_f{tx.pct(f)}", seeds)]
            v = [x for x in v if x is not None]
            if v:
                x = xpos[f] + (j - 1) * 0.07
                ax.scatter([x] * len(v), v, s=18, color=eda.SERIES_COLOURS[j],
                           edgecolors="white", linewidths=0.8, zorder=3)
                pts.append((x, float(np.mean(v))))
        if pts:
            ax.plot(*zip(*pts), color=eda.SERIES_COLOURS[j], linewidth=2,
                    label=f"ECG-GPT, {tx.STRATEGY_LABELS[st].lower()}")
    zs = _alias_log(paths, "zeroshot", fast, "zeroshot_fast")
    if zs is not None:
        ax.scatter([0], [zs["scalars"]["test_macro_auroc"]], marker="D", s=30,
                   color=eda.SERIES_COLOURS[3], label="CLIP zero-shot", zorder=3)
    refs = [("onelead", "CNN, lead II, 100% labels", "--"),
            ("twelve", "CNN, 12 leads, 100% labels", ":")]
    one = lx.find_runs(paths, fast)
    for kind, label, ls in refs:
        if kind == "onelead":
            v = [one[("cnn", x)]["scalars"]["test_macro_auroc"]
                 for x in seeds if ("cnn", x) in one]
        else:
            t = lx.twelve_lead_finals(paths, "cnn", fast)
            v = [t[x]["scalars"]["test_macro_auroc"] for x in seeds if x in t]
        if v:
            ax.axhline(float(np.mean(v)), color=eda.INK_MUTED, linestyle=ls,
                       linewidth=1, label=label)
    ax.axhline(0.5, color=eda.GRID, linewidth=1)
    ax.set_xticks([0, 1, 2], ["0% (zero-shot)", "10%", "100%"])
    ax.set_xlabel("share of training labels")
    ax.set_ylabel("test macro AUROC")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=7)
    dx._save(fig, paths, "comparison_label_efficiency" + suffix)
    return True


def write_report(fast: bool = False, paths: Paths = Paths(), seeds=SEEDS,
                 published_path=PUBLISHED_FILE) -> dict:
    suffix = "_fast" if fast else ""
    summary = build_summary(paths, fast, seeds, published_path)
    path = runlog.write_run(summary, paths.logs)
    print(f"summary: {rel(path)} ({len(summary['scalars'])} numbers, "
          f"{len(summary['rows'])} models)")
    paths.tables.mkdir(parents=True, exist_ok=True)
    for name, text in (("comparison_final", final_table(summary)),
                       ("comparison_per_class", per_class_table(summary)),
                       ("comparison_cross", cross_table(summary))):
        if text is not None:
            (paths.tables / f"{name}{suffix}.tex").write_text(text,
                                                              encoding="utf-8")
    figure_overview(paths, summary, fast, suffix, seeds)
    figure_label_efficiency(paths, fast, suffix, seeds)
    if not fast:
        tx.register_alias(paths, ALIAS, SUMMARY_NAME)
    return summary


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("step", choices=["report"])
    p.add_argument("--fast", action="store_true")
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    a = p.parse_args(argv)
    write_report(a.fast, seeds=a.seeds)


if __name__ == "__main__":
    main()
