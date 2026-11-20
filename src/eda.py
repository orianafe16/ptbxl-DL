"""Exploratory data analysis of PTB-XL (task T6, chat DL 02).

Run from the repository root:

    python -m src.eda

It rewrites `results/logs/eda.json` and the figures `report/figures/eda_*`
every time, so the numbers always come from one complete run.
Decisions: DECISIONS D12 (2026-11-20) and O3.

Layout of eda.json (DECISIONS D12):
    scalars : flat {name: number}, read by report/make_numbers.py
    tables  : small tables behind the figures
    meta    : date, git commit and package versions

Parts so far:
    1. overview of the splits
    2. records without a diagnostic superclass (O3)
    3. classes: balance, co-occurrence, combinations, spread over folds
    4. demographics (age, sex, records per patient) and data quality
       (signal-quality flags, human validation of the labels)
    5. signals: per-lead statistics and example traces (training only)

Every figure is saved twice: PDF (vector, for the LaTeX reports) and PNG
(for the notebook and the repository page).
"""
import json
import platform
import subprocess
from collections import Counter
from datetime import datetime

import matplotlib
matplotlib.use("Agg")          # draw to files only; no window is opened
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.data import load_metadata, superclass_labels
from src.dataset import lead_stats, split_indices

LOG_PATH = config.LOGS_DIR / "eda.json"
FIG_DIR = config.FIGURES_DIR
SPLITS = ["train", "val", "test"]
SPLIT_NAMES = {"train": "Train (folds 1-8)", "val": "Validation (fold 9)",
               "test": "Test (fold 10)"}

# Figure colours. Categorical colours always follow the same order, so a
# split or a class keeps its colour in every figure. Sequential maps (one
# hue, light -> dark) are used for magnitudes such as the heatmaps.
SERIES_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
SPLIT_COLOURS = dict(zip(SPLITS, SERIES_COLOURS))
CLASS_COLOURS = dict(zip(config.CLASSES, SERIES_COLOURS))
# Sex gets two colours used nowhere else, so it is never mistaken for a
# split or a class. The meaning of 0/1 is not documented in the dataset
# files: DECISIONS D12, [verify].
SEX_COLOURS = {0: "#6250d6", 1: "#008300"}
SEX_NAMES = {0: "sex 0", 1: "sex 1"}
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SEQUENTIAL = matplotlib.colors.LinearSegmentedColormap.from_list(
    "blue_seq", ["#f4f8fd", "#86b6ef", "#2a78d6", "#0d366b"])


def set_style() -> None:
    """One plotting style for every EDA figure: thin marks, quiet axes."""
    plt.rcParams.update({
        "figure.dpi": 100,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "figure.constrained_layout.use": True,   # panels never overlap
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.edgecolor": INK_MUTED,
        "axes.labelcolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,         # grid lines behind the bars
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "legend.frameon": False,
    })


def save_figure(fig: plt.Figure, name: str) -> None:
    """Write report/figures/<name>.pdf and .png, then free the memory."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{name}.{ext}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 0. Inputs
# ---------------------------------------------------------------------------
def split_of(fold: int) -> str:
    """Name of the split a fold belongs to (DECISIONS D3)."""
    if fold in config.TRAIN_FOLDS:
        return "train"
    if fold == config.VAL_FOLD:
        return "val"
    if fold == config.TEST_FOLD:
        return "test"
    raise ValueError(f"unexpected fold: {fold}")


def load_table(raw_dir=config.RAW_DIR) -> pd.DataFrame:
    """Metadata of every record plus one 0/1 column per superclass.

    The labels are built with the same function as the cache
    (src.data.superclass_labels), so the EDA describes exactly the labels
    the models are trained on.
    """
    df = load_metadata(raw_dir)
    y = superclass_labels(df, raw_dir)
    for j, name in enumerate(config.CLASSES):
        df[name] = y[:, j].astype(int)
    df["n_labels"] = y.sum(axis=1).astype(int)
    df["has_superclass"] = df["n_labels"] > 0
    df["split"] = df["strat_fold"].map(split_of)
    return df


def check_against_cache(df: pd.DataFrame, cache_path=config.CACHE_PATH) -> None:
    """Stop if the cache and the metadata disagree on records or labels.

    An .npz file is read lazily: asking for two keys loads only those two
    arrays, not the 1 GB of signals.
    """
    with np.load(cache_path) as cache:
        ecg_id, y = cache["ecg_id"], cache["y"]
    if not np.array_equal(ecg_id, df.index.to_numpy()):
        raise ValueError("record order differs between metadata and cache")
    if not np.array_equal(y, df[config.CLASSES].to_numpy(dtype=np.float32)):
        raise ValueError("labels differ between metadata and cache")


# ---------------------------------------------------------------------------
# 1. Overview of the splits
# ---------------------------------------------------------------------------
def part_overview(df: pd.DataFrame) -> tuple[dict, dict]:
    """Records, patients and labelled records (with a superclass) per split."""
    scalars = {
        "n_records_all": len(df),
        "n_patients_all": df["patient_id"].nunique(),
        "n_labelled_all": int(df["has_superclass"].sum()),
    }
    for split in SPLITS:
        sub = df[df["split"] == split]
        scalars[f"n_records_{split}"] = len(sub)
        scalars[f"n_patients_{split}"] = sub["patient_id"].nunique()
        scalars[f"n_labelled_{split}"] = int(sub["has_superclass"].sum())
    return scalars, {}


# ---------------------------------------------------------------------------
# 2. Records without a diagnostic superclass (O3)
# ---------------------------------------------------------------------------
def statement_kind(code: str, statements: pd.DataFrame) -> str:
    """'diagnostic', 'form', 'rhythm' or a combination such as 'diagnostic+form'.

    scp_statements.csv marks each kind with 1.0 and leaves the others empty
    (NaN); NaN == 1 is False, so empty cells simply do not count.
    """
    if code not in statements.index:
        return "unknown"
    kinds = [k for k in ("diagnostic", "form", "rhythm")
             if statements.loc[code, k] == 1]
    return "+".join(kinds) if kinds else "none"


def part_no_superclass(df: pd.DataFrame,
                       statements: pd.DataFrame) -> tuple[dict, dict]:
    """Where the records without superclass are and which codes they carry."""
    none = df[~df["has_superclass"]]
    kinds = none["scp_codes"].apply(
        lambda codes: {statement_kind(c, statements) for c in codes})

    scalars = {
        "n_nosuper_all": len(none),
        # The free-text column `pacemaker` is filled when a pacemaker was seen.
        "n_nosuper_pacemaker_field": int(none["pacemaker"].notna().sum()),
        "n_nosuper_with_rhythm_code": int(kinds.apply(lambda k: "rhythm" in k).sum()),
        "n_nosuper_with_form_code": int(kinds.apply(lambda k: "form" in k).sum()),
        "n_nosuper_rhythm_only": int(kinds.apply(lambda k: k == {"rhythm"}).sum()),
    }
    for split in SPLITS:
        scalars[f"n_nosuper_{split}"] = int((none["split"] == split).sum())

    per_fold = (none["strat_fold"].value_counts()
                .reindex(range(1, 11), fill_value=0))
    codes = Counter(code for record in none["scp_codes"] for code in record)
    code_rows = [
        {"code": code,
         "count": n,
         "kind": statement_kind(code, statements),
         "description": str(statements["description"].get(code, ""))}
        for code, n in codes.most_common()
    ]
    tables = {
        "nosuper_per_fold": {f"fold_{k}": int(v) for k, v in per_fold.items()},
        "nosuper_codes": code_rows,
    }
    return scalars, tables


# ---------------------------------------------------------------------------
# 3. Classes: balance, co-occurrence, combinations, spread over folds
# ---------------------------------------------------------------------------
def combination_name(row: pd.Series) -> str:
    """'MI+CD' for a record with MI and CD, in the fixed class order."""
    return "+".join(c for c in config.CLASSES if row[c] == 1)


def part_classes(df: pd.DataFrame,
                 statements: pd.DataFrame) -> tuple[dict, dict]:
    """Class balance per split, co-occurrence and per-fold prevalence.

    Only labelled records (with at least one superclass) are counted: they
    are the ones the models see (DECISIONS O3, closed in D12). Prevalence
    of a class = records with that class / labelled records; the classes
    overlap, so the five prevalences of a split add up to more than 1.
    """
    lab = df[df["has_superclass"]]
    Y = lab[config.CLASSES].to_numpy()
    scalars, tables = {}, {}

    # Balance per split
    counts, prevalence = {}, {}
    for split in ["all"] + SPLITS:
        sub = lab if split == "all" else lab[lab["split"] == split]
        counts[split] = {c: int(sub[c].sum()) for c in config.CLASSES}
        prevalence[split] = {c: float(sub[c].mean()) for c in config.CLASSES}
        for c in config.CLASSES:
            scalars[f"count_{split}_{c}"] = counts[split][c]
            scalars[f"prev_{split}_{c}"] = prevalence[split][c]
    # Imbalance: most frequent class over least frequent, on training.
    train_counts = counts["train"].values()
    scalars["imbalance_ratio_train"] = max(train_counts) / min(train_counts)

    # Number of labels per record
    n_labels = lab["n_labels"].value_counts().sort_index()
    for k, n in n_labels.items():
        scalars[f"n_with_{k}_labels"] = int(n)
    scalars["frac_multilabel_all"] = float((lab["n_labels"] > 1).mean())
    norm_other = lab[(lab["NORM"] == 1) & (lab["n_labels"] > 1)]
    scalars["n_norm_with_other"] = len(norm_other)
    # Which diagnostic codes of OTHER classes sit next to NORM?
    diag_class = statements.loc[statements["diagnostic"] == 1,
                                "diagnostic_class"]
    other_codes = Counter(
        code for record in norm_other["scp_codes"] for code in record
        if code in diag_class.index and diag_class[code] != "NORM")
    norm_other_rows = [
        {"code": code, "count": n, "class": diag_class[code],
         "description": str(statements.loc[code, "description"])}
        for code, n in other_codes.most_common()]

    # Co-occurrence: C[i, j] = records carrying both class i and class j.
    # The diagonal holds the class counts. Dividing row i by C[i, i] gives
    # P(class j | class i): "of the records with i, the share that also has j".
    cooc = Y.T @ Y
    conditional = cooc / np.diag(cooc)[:, None]

    # Most frequent label combinations
    combos = lab.apply(combination_name, axis=1).value_counts()
    scalars["n_combinations"] = len(combos)

    # Spread over folds: prevalence of each class in each fold, and the
    # largest gap between a fold and the whole dataset (percentage points).
    per_fold = lab.groupby("strat_fold")[config.CLASSES].mean()
    gap = (per_fold - lab[config.CLASSES].mean()).abs().to_numpy().max()
    scalars["max_fold_prev_gap_pp"] = float(100 * gap)

    tables.update({
        "class_counts": counts,
        "class_prevalence": prevalence,
        "cooccurrence_counts": cooc.astype(int).tolist(),
        "cooccurrence_conditional": conditional.round(4).tolist(),
        "label_combinations": [{"labels": k, "count": int(v)}
                               for k, v in combos.items()],
        "prevalence_per_fold": {c: per_fold[c].round(4).tolist()
                                for c in config.CLASSES},
        "norm_with_other_codes": norm_other_rows,
    })
    return scalars, tables


def plot_class_counts(tables: dict) -> None:
    """(a) records per class; (b) prevalence per class in each split."""
    classes = config.CLASSES
    x = np.arange(len(classes))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.2))

    counts = [tables["class_counts"]["all"][c] for c in classes]
    bars = ax1.bar(x, counts, width=0.6, color=SERIES_COLOURS[0])
    ax1.bar_label(bars, padding=2, fontsize=8, color=INK)
    ax1.set_xticks(x, classes)
    ax1.set_ylabel("Records")
    ax1.set_title("(a) Labelled records per superclass", loc="left")
    ax1.grid(axis="x", visible=False)

    width = 0.26
    for k, split in enumerate(SPLITS):
        values = [100 * tables["class_prevalence"][split][c] for c in classes]
        ax2.bar(x + (k - 1) * width, values, width=width * 0.92,
                color=SPLIT_COLOURS[split], label=SPLIT_NAMES[split])
    ax2.set_xticks(x, classes)
    ax2.set_ylabel("Records with the class (%)")
    ax2.set_title("(b) Prevalence per split", loc="left")
    ax2.grid(axis="x", visible=False)
    ax2.legend(fontsize=8)
    save_figure(fig, "eda_class_counts")


def plot_cooccurrence(tables: dict) -> None:
    """(a) P(column class | row class); (b) most frequent label combinations."""
    classes = config.CLASSES
    cond = np.array(tables["cooccurrence_conditional"]) * 100
    np.fill_diagonal(cond, np.nan)          # 100 % by definition: left blank
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8),
                                   gridspec_kw={"width_ratios": [1, 1.1]})

    im = ax1.imshow(cond, cmap=SEQUENTIAL, vmin=0, vmax=np.nanmax(cond))
    for i in range(len(classes)):
        for j in range(len(classes)):
            if i != j:
                light = cond[i, j] > 0.6 * np.nanmax(cond)
                ax1.text(j, i, f"{cond[i, j]:.0f}", ha="center", va="center",
                         fontsize=8, color="white" if light else INK)
    ax1.set_xticks(range(len(classes)), classes)
    ax1.set_yticks(range(len(classes)), classes)
    ax1.set_xlabel("... also has this class")
    ax1.set_ylabel("Records with this class ...")
    ax1.set_title("(a) Co-occurrence (% of row class)", loc="left")
    ax1.grid(False)
    for side in ("top", "right", "left", "bottom"):
        ax1.spines[side].set_visible(False)
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04).outline.set_visible(False)

    top = tables["label_combinations"][:12][::-1]       # largest at the top
    names = [r["labels"] for r in top]
    values = [r["count"] for r in top]
    single = ["+" not in n for n in names]
    bars = ax2.barh(names, values, height=0.65,
                    color=[SERIES_COLOURS[0] if s else SERIES_COLOURS[1]
                           for s in single])
    ax2.bar_label(bars, padding=2, fontsize=7.5, color=INK)
    ax2.set_xlabel("Records")
    ax2.set_title("(b) Most frequent label combinations", loc="left")
    ax2.grid(axis="y", visible=False)
    ax2.legend(handles=[
        matplotlib.patches.Patch(color=SERIES_COLOURS[0], label="one superclass"),
        matplotlib.patches.Patch(color=SERIES_COLOURS[1], label="several")],
        fontsize=8, loc="lower right")
    save_figure(fig, "eda_label_cooccurrence")


def spread_labels(ends: dict, min_gap: float) -> dict:
    """Label heights at least `min_gap` apart, kept as close as possible
    to the line ends (walk upwards from the lowest label)."""
    placed, last = {}, -np.inf
    for name, y in sorted(ends.items(), key=lambda item: item[1]):
        last = max(y, last + min_gap)
        placed[name] = last
    return placed


def plot_fold_distribution(tables: dict) -> None:
    """Prevalence of each class in each of the ten folds."""
    folds = np.arange(1, 11)
    markers = ["o", "s", "^", "D", "v"]     # second cue besides colour
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ends = {}
    for c, marker in zip(config.CLASSES, markers):
        values = 100 * np.array(tables["prevalence_per_fold"][c])
        ax.plot(folds, values, color=CLASS_COLOURS[c], linewidth=2,
                marker=marker, markersize=5, label=c)
        ends[c] = values[-1]
    # Direct labels at the right end, pushed apart when lines end close.
    for c, y in spread_labels(ends, min_gap=1.6).items():
        ax.annotate(c, (folds[-1], y), xytext=(10, 0),
                    textcoords="offset points", va="center", fontsize=8,
                    color=INK)
    ax.axvspan(8.5, 10.5, color=GRID, alpha=0.5, linewidth=0)
    ax.text(9, ax.get_ylim()[1], "val", ha="center", va="top",
            fontsize=8, color=INK_MUTED)
    ax.text(10, ax.get_ylim()[1], "test", ha="center", va="top",
            fontsize=8, color=INK_MUTED)
    ax.set_xticks(folds)
    ax.set_xlim(0.5, 10.5)
    ax.set_xlabel("Fold (strat_fold)")
    ax.set_ylabel("Records with the class (%)")
    ax.set_title("Class prevalence per fold", loc="left")
    ax.legend(ncol=5, fontsize=8, loc="upper center",
              bbox_to_anchor=(0.5, -0.2))
    save_figure(fig, "eda_fold_distribution")


# ---------------------------------------------------------------------------
# 4. Demographics and data quality
# ---------------------------------------------------------------------------
AGE_CODED_OVER_89 = 300     # ptbxl_v102_changelog.txt, line 2 (DECISIONS D12)
QUALITY_COLUMNS = ["baseline_drift", "static_noise", "burst_noise",
                   "electrodes_problems", "extra_beats", "pacemaker"]


def quality_flags(df: pd.DataFrame) -> pd.DataFrame:
    """True where a free-text quality column says something.

    The columns hold notes in German such as ', V1' or 'alles' (all leads);
    an empty cell means no problem was noted. Blank strings count as empty.
    """
    flags = pd.DataFrame(index=df.index)
    for col in QUALITY_COLUMNS:
        text = df[col].fillna("").astype(str).str.strip(" ,")
        flags[col] = text != ""
    return flags


def part_demographics(df: pd.DataFrame) -> tuple[dict, dict]:
    """Age, sex and records per patient.

    Ages coded as 300 (older than 89) are counted apart and left out of
    medians and histograms (DECISIONS D12).
    """
    scalars, tables = {}, {}
    over89 = df["age"] == AGE_CODED_OVER_89
    age = df.loc[~over89, "age"]
    scalars["n_age_over89_all"] = int(over89.sum())
    scalars["n_age_missing_all"] = int(df["age"].isna().sum())
    scalars["age_min_all"] = float(age.min())
    scalars["age_max_all"] = float(age.max())

    for split in ["all"] + SPLITS:
        sub = df if split == "all" else df[df["split"] == split]
        a = sub.loc[sub["age"] != AGE_CODED_OVER_89, "age"]
        scalars[f"age_median_{split}"] = float(a.median())
        scalars[f"age_q1_{split}"] = float(a.quantile(0.25))
        scalars[f"age_q3_{split}"] = float(a.quantile(0.75))
        for code in (0, 1):
            scalars[f"n_sex{code}_{split}"] = int((sub["sex"] == code).sum())
        scalars[f"frac_sex1_{split}"] = float((sub["sex"] == 1).mean())

    # Age and sex per superclass (labelled records only)
    lab = df[df["has_superclass"]]
    for c in config.CLASSES:
        rows = lab[lab[c] == 1]
        a = rows.loc[rows["age"] != AGE_CODED_OVER_89, "age"]
        scalars[f"age_median_{c}"] = float(a.median())
        scalars[f"frac_sex1_{c}"] = float((rows["sex"] == 1).mean())

    # Height and weight: only how many are missing (DECISIONS D12)
    for col in ("height", "weight"):
        scalars[f"n_{col}_missing_all"] = int(df[col].isna().sum())

    # Records per patient: why the split must keep patients together (D3)
    per_patient = df.groupby("patient_id").size()
    scalars["n_patients_one_record"] = int((per_patient == 1).sum())
    scalars["n_patients_several_records"] = int((per_patient > 1).sum())
    scalars["max_records_per_patient"] = int(per_patient.max())

    bins = np.arange(0, 95, 5)                       # 0-4, 5-9, ..., 85-89
    tables["age_hist_bins"] = bins.tolist()
    tables["age_hist_by_sex"] = {
        SEX_NAMES[code]: np.histogram(
            df.loc[(~over89) & (df["sex"] == code), "age"], bins=bins
        )[0].tolist()
        for code in (0, 1)}
    tables["age_by_class"] = {
        c: lab.loc[(lab[c] == 1) & (lab["age"] != AGE_CODED_OVER_89),
                   "age"].astype(int).tolist()
        for c in config.CLASSES}
    return scalars, tables


def part_quality(df: pd.DataFrame) -> tuple[dict, dict]:
    """Signal-quality flags (described, nothing excluded: D12) and how the
    labels were validated."""
    flags = quality_flags(df)
    any_flag = flags.any(axis=1)
    scalars, tables = {}, {}

    per_split = {}
    for split in ["all"] + SPLITS:
        in_split = (np.ones(len(df), dtype=bool) if split == "all"
                    else (df["split"] == split).to_numpy())
        f = flags[in_split]
        per_split[split] = {col: float(f[col].mean()) for col in QUALITY_COLUMNS}
        for col in QUALITY_COLUMNS:
            scalars[f"n_{col}_{split}"] = int(f[col].sum())
        scalars[f"n_any_quality_flag_{split}"] = int(any_flag[in_split].sum())
        scalars[f"frac_any_quality_flag_{split}"] = float(any_flag[in_split].mean())

    # Share of the records of each class that carry each flag
    lab = df["has_superclass"]
    per_class = {
        col: [float(flags.loc[lab & (df[c] == 1), col].mean())
              for c in config.CLASSES]
        for col in QUALITY_COLUMNS}

    # Label quality: report validated by a person, second opinion
    for split in ["all"] + SPLITS:
        sub = df if split == "all" else df[df["split"] == split]
        scalars[f"n_validated_by_human_{split}"] = int(sub["validated_by_human"].sum())
        scalars[f"frac_validated_by_human_{split}"] = float(
            sub["validated_by_human"].mean())
    scalars["n_second_opinion_all"] = int(df["second_opinion"].sum())

    tables["quality_frac_per_split"] = per_split
    tables["quality_frac_per_class"] = per_class
    return scalars, tables


def plot_age_sex(tables: dict, scalars: dict) -> None:
    """(a) age histogram for each sex; (b) age distribution per class."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.3))
    bins = np.array(tables["age_hist_bins"])
    for code in (0, 1):
        name = SEX_NAMES[code]
        ax1.stairs(tables["age_hist_by_sex"][name], bins, linewidth=2,
                   color=SEX_COLOURS[code],
                   label=f"{name} (n = {scalars[f'n_sex{code}_all']})")
    ax1.set_xlabel("Age (years)")
    ax1.set_ylabel("Records")
    ax1.set_title("(a) Age by sex", loc="left")
    ax1.legend(fontsize=8, loc="upper left", title_fontsize=7.5,
               title=f"not shown: > 89 years "
                     f"(coded 300, n = {scalars['n_age_over89_all']})",
               alignment="left")

    data = [tables["age_by_class"][c] for c in config.CLASSES]
    box = ax2.boxplot(data, widths=0.5, patch_artist=True, showfliers=False,
                      medianprops={"color": INK, "linewidth": 1.5},
                      whiskerprops={"color": INK_MUTED},
                      capprops={"color": INK_MUTED})
    for patch, c in zip(box["boxes"], config.CLASSES):
        patch.set_facecolor(CLASS_COLOURS[c])
        patch.set_edgecolor(CLASS_COLOURS[c])
        patch.set_alpha(0.85)
    ax2.set_xticks(range(1, len(config.CLASSES) + 1), config.CLASSES)
    ax2.set_ylabel("Age (years)")
    ax2.set_title("(b) Age per superclass (up to 89 years)", loc="left")
    ax2.grid(axis="x", visible=False)
    save_figure(fig, "eda_age_sex")


def plot_quality(tables: dict) -> None:
    """(a) records with each quality flag per split; (b) per class."""
    cols = QUALITY_COLUMNS
    labels = [c.replace("_", " ") for c in cols]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.4),
                                   gridspec_kw={"width_ratios": [1.1, 1]})

    y = np.arange(len(cols))
    height = 0.26
    for k, split in enumerate(SPLITS):
        values = [100 * tables["quality_frac_per_split"][split][c] for c in cols]
        ax1.barh(y + (k - 1) * height, values, height=height * 0.92,
                 color=SPLIT_COLOURS[split], label=SPLIT_NAMES[split])
    ax1.set_yticks(y, labels)
    ax1.invert_yaxis()
    ax1.set_xlabel("Records with the note (%)")
    ax1.set_title("(a) Quality notes per split", loc="left")
    ax1.grid(axis="y", visible=False)
    ax1.legend(fontsize=8, loc="lower right")

    grid = 100 * np.array([tables["quality_frac_per_class"][c] for c in cols])
    im = ax2.imshow(grid, cmap=SEQUENTIAL, vmin=0, aspect="auto")
    for i in range(len(cols)):
        for j in range(len(config.CLASSES)):
            light = grid[i, j] > 0.6 * grid.max()
            ax2.text(j, i, f"{grid[i, j]:.1f}", ha="center", va="center",
                     fontsize=8, color="white" if light else INK)
    ax2.set_xticks(range(len(config.CLASSES)), config.CLASSES)
    ax2.set_yticks(range(len(cols)), labels)
    ax2.set_title("(b) % of the records of each class", loc="left")
    ax2.grid(False)
    for side in ("top", "right", "left", "bottom"):
        ax2.spines[side].set_visible(False)
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04).outline.set_visible(False)
    save_figure(fig, "eda_signal_quality")


# ---------------------------------------------------------------------------
# 5. Signals: per-lead statistics and example traces (training records only)
# ---------------------------------------------------------------------------
# Signals are in millivolts: the record headers give a gain of 1000 units
# per mV (checked in records100/00000/00001_lr.hea).
LEAD_II = config.LEADS.index("II")


def load_signals(cache_path=config.CACHE_PATH) -> np.ndarray:
    """All signals, (n_records, 12, 1000) float32, about 1 GB in memory."""
    with np.load(cache_path) as cache:
        return cache["X"]


def training_rows(df: pd.DataFrame) -> np.ndarray:
    """Training rows exactly as the DataLoader picks them (D3, O3).

    split_indices is the function used by make_loaders, so the per-lead
    mean and std below are the very values used to normalise the inputs.
    """
    return split_indices({"fold": df["strat_fold"].to_numpy(),
                          "has_superclass": df["has_superclass"].to_numpy()}
                         )["train"]


def example_rows(df: pd.DataFrame) -> dict:
    """One training record per superclass carrying that class only.

    Drawn at random with seed 42 (D12), not hand-picked, so the examples
    are typical rather than the cleanest ones.
    """
    rng = np.random.default_rng(config.SEED)
    single = (df["split"] == "train") & (df["n_labels"] == 1)
    return {c: int(rng.choice(np.flatnonzero((single & (df[c] == 1)).to_numpy())))
            for c in config.CLASSES}


def part_signals(df: pd.DataFrame, X: np.ndarray) -> tuple[dict, dict]:
    """Per-lead statistics over the training records.

    - mean and std per lead: the normalisation statistics (D3);
    - peak-to-peak amplitude of each record (max - min, in mV): its median,
      quartiles and 5th / 95th percentiles per lead;
    - 0.5th and 99.5th percentiles of all sample values (extreme values);
    - integrity: missing values (NaN) and constant (flat) leads.
    One lead at a time, so only about 70 MB is copied at once.
    """
    rows = training_rows(df)
    mean, std = lead_stats(X, rows)
    scalars = {"n_signal_records_train": len(rows)}
    stats = {k: [] for k in ("mean", "std", "ptp_p5", "ptp_q1", "ptp_median",
                             "ptp_q3", "ptp_p95", "value_p0_5", "value_p99_5")}
    n_nan, n_flat, flat = 0, 0, []
    for lead, name in enumerate(config.LEADS):
        values = X[rows, lead, :]                      # (n_records, 1000)
        n_nan += int(np.isnan(values).sum())
        ptp = np.ptp(values, axis=1)                   # one value per record
        n_flat += int((ptp == 0).sum())
        flat += [{"ecg_id": int(df.index[r]), "lead": name}
                 for r in rows[ptp == 0]]
        p5, q1, med, q3, p95 = np.percentile(ptp, [5, 25, 50, 75, 95])
        low, high = np.percentile(values, [0.5, 99.5])
        for key, value in zip(stats, (mean[lead], std[lead], p5, q1, med, q3,
                                      p95, low, high)):
            stats[key].append(float(value))
        scalars[f"lead_mean_{name}"] = float(mean[lead])
        scalars[f"lead_std_{name}"] = float(std[lead])
        scalars[f"lead_ptp_median_{name}"] = float(med)
    scalars["n_nan_values_train"] = n_nan
    scalars["n_flat_leads_train"] = n_flat
    # How different the leads are: largest std over smallest std.
    scalars["lead_std_ratio_max_min"] = float(std.max() / std.min())

    examples = example_rows(df)
    tables = {
        "lead_stats_train": {"leads": config.LEADS, **stats},
        "flat_leads_train": flat,
        "example_records": {c: {"row": r, "ecg_id": int(df.index[r])}
                            for c, r in examples.items()},
    }
    return scalars, tables


def plot_lead_stats(tables: dict) -> None:
    """(a) std per lead; (b) peak-to-peak amplitude per lead."""
    st = tables["lead_stats_train"]
    leads = st["leads"]
    x = np.arange(len(leads))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.3))

    bars = ax1.bar(x, st["std"], width=0.6, color=SERIES_COLOURS[0])
    ax1.bar_label(bars, labels=[f"{v:.2f}" for v in st["std"]], padding=2,
                  fontsize=7, color=INK)
    ax1.set_xticks(x, leads)
    ax1.set_ylabel("Standard deviation (mV)")
    ax1.set_title("(a) Spread of the samples per lead (training)", loc="left")
    ax1.grid(axis="x", visible=False)

    # Boxes from precomputed statistics: whiskers at the 5th and 95th
    # percentiles (stated in the title), no outlier points.
    boxes = [{"label": lead, "whislo": st["ptp_p5"][i], "q1": st["ptp_q1"][i],
              "med": st["ptp_median"][i], "q3": st["ptp_q3"][i],
              "whishi": st["ptp_p95"][i], "fliers": []}
             for i, lead in enumerate(leads)]
    drawn = ax2.bxp(boxes, widths=0.55, patch_artist=True, showfliers=False,
                    medianprops={"color": INK, "linewidth": 1.5},
                    whiskerprops={"color": INK_MUTED},
                    capprops={"color": INK_MUTED})
    for patch in drawn["boxes"]:
        patch.set_facecolor(SERIES_COLOURS[0])
        patch.set_edgecolor(SERIES_COLOURS[0])
        patch.set_alpha(0.85)
    ax2.set_ylabel("Peak-to-peak amplitude (mV)")
    ax2.set_title("(b) Amplitude per record (whiskers: 5th-95th pct.)",
                  loc="left")
    ax2.grid(axis="x", visible=False)
    save_figure(fig, "eda_lead_stats")


def plot_example_traces(X: np.ndarray, tables: dict) -> None:
    """Lead II of one training record per superclass, raw values in mV."""
    t = np.arange(config.N_SAMPLES) / config.SAMPLING_RATE      # seconds
    examples = tables["example_records"]
    fig, axes = plt.subplots(len(config.CLASSES), 1, figsize=(9, 7.5),
                             sharex=True)
    for ax, c in zip(axes, config.CLASSES):
        row, ecg_id = examples[c]["row"], examples[c]["ecg_id"]
        ax.plot(t, X[row, LEAD_II], color=CLASS_COLOURS[c], linewidth=1)
        ax.set_title(f"{c}  (ecg_id {ecg_id}, lead II)", loc="left",
                     fontsize=9)
        ax.set_ylabel("mV")
        ax.grid(axis="x", visible=False)
    axes[-1].set_xlabel("Time (s)")
    save_figure(fig, "eda_example_traces")


def plot_example_12lead(X: np.ndarray, tables: dict) -> None:
    """The 12 leads of the NORM example: what one input (12, 1000) is."""
    t = np.arange(config.N_SAMPLES) / config.SAMPLING_RATE
    row = tables["example_records"]["NORM"]["row"]
    ecg_id = tables["example_records"]["NORM"]["ecg_id"]
    fig, axes = plt.subplots(6, 2, figsize=(10, 7), sharex=True, sharey=True)
    for lead, ax in enumerate(axes.T.flatten()):       # column by column
        ax.plot(t, X[row, lead], color=CLASS_COLOURS["NORM"], linewidth=0.9)
        ax.text(0.01, 0.85, config.LEADS[lead], transform=ax.transAxes,
                fontsize=8.5, color=INK, fontweight="bold")
        ax.grid(axis="x", visible=False)
    for ax in axes[-1]:
        ax.set_xlabel("Time (s)")
    for ax in axes[:, 0]:
        ax.set_ylabel("mV")
    fig.suptitle(f"One input to the models: 12 leads x 1000 samples "
                 f"(NORM, ecg_id {ecg_id})", x=0.01, ha="left", fontsize=10)
    save_figure(fig, "eda_example_12lead")


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def run_meta() -> dict:
    """When and with what the EDA was run, for the reproducibility record."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=config.ROOT,
            capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {
        "created": datetime.now().isoformat(timespec="seconds"),
        "git_commit": commit,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }


def merge(results: list[tuple[dict, dict]]) -> tuple[dict, dict]:
    """Join the (scalars, tables) of every part; a repeated name is a bug."""
    scalars, tables = {}, {}
    for part_scalars, part_tables in results:
        for target, new in ((scalars, part_scalars), (tables, part_tables)):
            repeated = target.keys() & new.keys()
            if repeated:
                raise KeyError(f"name used by two parts: {sorted(repeated)}")
            target.update(new)
    return scalars, tables


def write_json(scalars: dict, tables: dict, path=LOG_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = {"scalars": scalars, "tables": tables, "meta": run_meta()}
    with open(path, "w", encoding="utf-8") as f:
        # numpy integers and floats are not JSON types: .item() converts them.
        json.dump(out, f, indent=2, ensure_ascii=False,
                  default=lambda o: o.item())


def print_summary(scalars: dict, tables: dict) -> None:
    """What to paste back into the chat."""
    print("\n== Overview ==")
    for key, value in scalars.items():
        if (key.startswith(("n_records_", "n_patients_", "n_labelled_"))
                and key.rsplit("_", 1)[-1] in ("all", "train", "val", "test")):
            print(f"  {key}: {value}")
    print("\n== Records without superclass (O3) ==")
    for key, value in scalars.items():
        if key.startswith("n_nosuper"):
            print(f"  {key}: {value}")
    print("  per fold:", tables["nosuper_per_fold"])
    print("  most frequent codes:")
    for row in tables["nosuper_codes"][:15]:
        print(f"    {row['code']:6s} {row['count']:4d}  {row['kind']:18s} "
              f"{row['description']}")

    print("\n== Classes (labelled records) ==")
    print(f"  {'':8s}" + "".join(f"{c:>8s}" for c in config.CLASSES))
    for split in ["all"] + SPLITS:
        print(f"  {split:8s}" + "".join(
            f"{scalars[f'count_{split}_{c}']:8d}" for c in config.CLASSES))
    for split in ["all"] + SPLITS:
        print(f"  {split + ' %':8s}" + "".join(
            f"{100 * scalars[f'prev_{split}_{c}']:8.1f}" for c in config.CLASSES))
    for key in ("imbalance_ratio_train", "frac_multilabel_all",
                "n_norm_with_other", "n_combinations", "max_fold_prev_gap_pp"):
        print(f"  {key}: {scalars[key]:.4g}")
    print("  records by number of labels:", {
        k: v for k, v in scalars.items() if k.startswith("n_with_")})
    print("  top combinations:", [
        (r["labels"], r["count"]) for r in tables["label_combinations"][:8]])
    print("  codes of other classes next to NORM:", [
        (r["code"], r["count"], r["class"])
        for r in tables["norm_with_other_codes"][:6]])

    print("\n== Demographics ==")
    for key, value in scalars.items():
        if key.startswith(("age_", "n_age", "n_sex", "frac_sex", "n_height",
                           "n_weight", "n_patients_one", "n_patients_several",
                           "max_records")):
            print(f"  {key}: {value:.4g}" if isinstance(value, float)
                  else f"  {key}: {value}")

    print("\n== Data quality ==")
    for split in ["all"] + SPLITS:
        print(f"  {split:6s}" + "".join(
            f"  {col}={scalars[f'n_{col}_{split}']}" for col in QUALITY_COLUMNS)
            + f"  any={scalars[f'n_any_quality_flag_{split}']}")
    for split in ["all"] + SPLITS:
        print(f"  {split:6s}  any quality note: "
              f"{100 * scalars[f'frac_any_quality_flag_{split}']:.1f} %   "
              f"validated by human: "
              f"{100 * scalars[f'frac_validated_by_human_{split}']:.1f} %")
    print(f"  n_second_opinion_all: {scalars['n_second_opinion_all']}")

    print("\n== Signals (training records) ==")
    st = tables["lead_stats_train"]
    print(f"  records: {scalars['n_signal_records_train']}   "
          f"NaN values: {scalars['n_nan_values_train']}   "
          f"flat leads: {scalars['n_flat_leads_train']}   "
          f"std ratio max/min: {scalars['lead_std_ratio_max_min']:.3g}")
    print(f"  {'lead':5s}{'mean':>9s}{'std':>8s}{'ptp med':>9s}"
          f"{'p0.5':>8s}{'p99.5':>8s}   (mV)")
    for i, lead in enumerate(st["leads"]):
        print(f"  {lead:5s}{st['mean'][i]:9.4f}{st['std'][i]:8.3f}"
              f"{st['ptp_median'][i]:9.3f}{st['value_p0_5'][i]:8.3f}"
              f"{st['value_p99_5'][i]:8.3f}")
    print("  flat leads (constant signal):", tables["flat_leads_train"])
    print("  example records:", {c: v["ecg_id"]
                                for c, v in tables["example_records"].items()})


def main() -> None:
    statements = pd.read_csv(config.RAW_DIR / "scp_statements.csv", index_col=0)
    df = load_table()
    check_against_cache(df)
    print(f"Metadata and cache agree: {len(df)} records.")

    scalars, tables = merge([
        part_overview(df),
        part_no_superclass(df, statements),
        part_classes(df, statements),
        part_demographics(df),
        part_quality(df),
    ])
    print("Loading the signals (about 1 GB)...")
    X = load_signals()
    signal_scalars, signal_tables = part_signals(df, X)
    scalars, tables = merge([(scalars, tables),
                             (signal_scalars, signal_tables)])
    write_json(scalars, tables)

    set_style()
    plot_class_counts(tables)
    plot_cooccurrence(tables)
    plot_fold_distribution(tables)
    plot_age_sex(tables, scalars)
    plot_quality(tables)
    plot_lead_stats(tables)
    plot_example_traces(X, tables)
    plot_example_12lead(X, tables)
    print_summary(scalars, tables)
    print(f"\nWritten: {LOG_PATH}")
    print(f"Figures: {FIG_DIR}  (eda_class_counts, eda_label_cooccurrence, "
          "eda_fold_distribution, eda_age_sex, eda_signal_quality, "
          "eda_lead_stats, eda_example_traces, eda_example_12lead; "
          ".pdf and .png)")


if __name__ == "__main__":
    main()
