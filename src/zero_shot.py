"""Zero-shot learning (T24): CLIP on CIFAR-100 and on PTB-XL images.

Decisions: docs/DECISIONS.md D20 (this experiment), D7 (why zero-shot is
studied with CLIP and not with our ECG-GPT).

    python -m src.zero_shot run --fast    # smoke test (small subsets)
    python -m src.zero_shot run           # CIFAR-100 + PTB-XL, then report

Zero-shot classification needs a space shared by the inputs and the names
of the classes (weekly summary 9/10: "Multimodal Correspondence"). CLIP
learnt one from (image, text) pairs by contrastive learning, so it can
classify an image among classes it was never trained on: the image is
compared with a sentence per class and the most similar sentence wins
(class notebook Interacting_with_CLIP.ipynb: cosine similarity x 100 as
logits of a softmax). Our ECG-GPT has no text side, so it cannot do this;
this experiment asks how far a general image-text model gets on ECGs.

    Part A  CIFAR-100 test set, the notebook's prompt "This is a photo of a
            {label}" and the bare label: top-1 and top-5 accuracy, where
            zero-shot is expected to work (chance 1 %).
    Part B  PTB-XL: lead II of every validation and test record drawn as a
            224 x 224 ECG-paper image; one sentence per superclass; macro
            AUROC with patient-bootstrap interval (D14), F1 thresholds from
            validation; and a modality check (does CLIP at least see that
            the image is an electrocardiogram?).

Model: CLIP ViT-B/32 with OpenAI's weights (the notebook's model), loaded
with open_clip ("ViT-B-32-quickgelu", pretrained="openai") or, if
open_clip is missing, the class package `clip`. Nothing is trained.

Outputs: results/logs/zeroshot.json (alias "zeroshot" in runs.json),
results/predictions/zeroshot.npz, report/tables/zeroshot.tex,
report/figures/zeroshot_examples.*, zeroshot_ptbxl.*. With --fast every
name ends in _fast and git ignores it.
"""
import argparse
from dataclasses import dataclass
from datetime import datetime
from typing import Callable

import numpy as np
import torch

from src import config, metrics, runlog
from src import dense_experiments as dx
from src.dataset import split_indices
from src.dense_experiments import Paths
from src.train import rel

ALIAS = "zeroshot"
RUN_NAME = "zeroshot"
MODEL_NAME = "ViT-B/32"
IMAGE_SIZE = 224
LEAD = "II"
N_BOOT = 1000
FAST_PTBXL = 128                       # records per split with --fast
FAST_CIFAR = 500                       # test images with --fast
DATA_DIR = config.ROOT / "data" / "external"

# One sentence per superclass, fixed before running (D20).
CLASS_PROMPTS = {
    "NORM": "an electrocardiogram of a healthy heart",
    "MI": "an electrocardiogram showing a myocardial infarction",
    "STTC": "an electrocardiogram showing ST-segment and T-wave changes",
    "CD": "an electrocardiogram showing a conduction disturbance",
    "HYP": "an electrocardiogram showing ventricular hypertrophy",
}
# Modality check: the first sentence is the right one for every ECG image.
MODALITY_PROMPTS = ["an electrocardiogram", "a photo of a cat",
                    "a page of text", "a landscape photo", "a bar chart",
                    "a city map"]
CIFAR_TEMPLATES = {"notebook": "This is a photo of a {}", "label": "{}"}


# ---------------------------------------------------------------------------
# 1. The model
# ---------------------------------------------------------------------------
@dataclass
class Clip:
    model: torch.nn.Module
    preprocess: Callable          # PIL image -> tensor (3, 224, 224)
    tokenize: Callable            # list of str -> token tensor
    backend: str
    device: torch.device

    @torch.no_grad()
    def text_features(self, texts: list[str]) -> torch.Tensor:
        """Unit-norm text embeddings, (n_texts, d), on the CPU like the
        image embeddings (AI Error Bounty B10: they were left on the GPU)."""
        t = self.tokenize(texts).to(self.device)
        f = self.model.encode_text(t).float()
        return (f / f.norm(dim=-1, keepdim=True)).cpu()

    @torch.no_grad()
    def image_features(self, images, batch: int = 64) -> torch.Tensor:
        """Unit-norm image embeddings of a list of PIL images, (n, d)."""
        out = []
        for i in range(0, len(images), batch):
            x = torch.stack([self.preprocess(im) for im in images[i:i + batch]])
            f = self.model.encode_image(x.to(self.device)).float()
            out.append((f / f.norm(dim=-1, keepdim=True)).cpu())
        return torch.cat(out)


def load_clip(device: torch.device | None = None) -> Clip:
    """CLIP ViT-B/32 with OpenAI's weights, in eval mode. open_clip first
    (`pip install open_clip_torch`); the class notebook's `clip` package
    otherwise (it imports pkg_resources, which recent setuptools versions
    may not provide)."""
    device = device or torch.device("cuda" if torch.cuda.is_available()
                                    else "cpu")
    try:
        import open_clip
        # OpenAI trained ViT-B/32 with the QuickGELU activation; open_clip's
        # plain "ViT-B-32" config uses GELU and warns of the mismatch, so the
        # "-quickgelu" config is the model of the class notebook.
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32-quickgelu", pretrained="openai")
        tokenize = open_clip.get_tokenizer("ViT-B-32-quickgelu")
        backend = f"open_clip {open_clip.__version__}"
    except ImportError:
        import clip
        model, preprocess = clip.load(MODEL_NAME, device="cpu")
        tokenize = clip.tokenize
        backend = "clip (OpenAI package)"
    model = model.float().to(device).eval()
    return Clip(model, preprocess, tokenize, backend, device)


def zero_shot_probs(image_feats: torch.Tensor,
                    text_feats: torch.Tensor) -> np.ndarray:
    """Softmax over the sentences of 100 x cosine similarity (the notebook's
    zero-shot rule), (n_images, n_texts)."""
    image_feats, text_feats = image_feats.cpu(), text_feats.cpu()
    return (100.0 * image_feats @ text_feats.T).softmax(dim=-1).numpy()


# ---------------------------------------------------------------------------
# 2. ECG records as images
# ---------------------------------------------------------------------------
class ECGRenderer:
    """Lead II of a 10 s record drawn on a 224 x 224 image, the size CLIP
    expects, so its preprocessing neither crops nor shrinks the strip: two
    rows of 5 s, black trace on a light ECG-paper grid (0.2 s, 0.5 mV),
    fixed vertical scale from -1.5 to 2.0 mV (values outside are clipped).
    One figure is reused for every record (fast)."""

    Y_RANGE = (-1.5, 2.0)

    def __init__(self, fs: int = config.SAMPLING_RATE, size: int = IMAGE_SIZE):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        self.fs, self.size = fs, size
        # layout="none": the axes are placed by hand (the EDA style turns
        # constrained layout on for every other figure)
        self.fig = plt.figure(figsize=(size / 100, size / 100), dpi=100,
                              layout="none")
        self.axes, self.lines = [], []
        for row in range(2):
            ax = self.fig.add_axes([0.02, 0.51 - 0.49 * row, 0.96, 0.47])
            ax.set_xlim(0, 5)
            ax.set_ylim(*self.Y_RANGE)
            for x in np.arange(0, 5.01, 0.2):
                ax.axvline(x, color="#f4b6b6", linewidth=0.4, zorder=0)
            for y in np.arange(-1.5, 2.01, 0.5):
                ax.axhline(y, color="#f4b6b6", linewidth=0.4, zorder=0)
            ax.set_axis_off()
            line, = ax.plot([], [], color="black", linewidth=0.7)
            self.axes.append(ax)
            self.lines.append(line)

    def render(self, x_mv: np.ndarray) -> np.ndarray:
        """(n_samples,) mV -> (size, size, 3) uint8 RGB image."""
        half = len(x_mv) // 2
        x = np.clip(x_mv, *self.Y_RANGE)
        for line, part in zip(self.lines, (x[:half], x[half:])):
            line.set_data(np.arange(len(part)) / self.fs, part)
        self.fig.canvas.draw()
        rgba = np.asarray(self.fig.canvas.buffer_rgba())
        return np.ascontiguousarray(rgba[:, :, :3])

    def images(self, signals: np.ndarray) -> list:
        from PIL import Image
        return [Image.fromarray(self.render(s)) for s in signals]

    def close(self) -> None:
        import matplotlib.pyplot as plt
        plt.close(self.fig)


# ---------------------------------------------------------------------------
# 3. Part B: PTB-XL
# ---------------------------------------------------------------------------
def ptbxl_rows(cache: dict, fast: bool = False) -> dict:
    """Validation and test rows (records with a superclass), in the order
    of every classifier (split_indices), so predictions pair with theirs.
    --fast: FAST_PTBXL rows per split drawn with seed 42."""
    idx = split_indices(cache)
    idx = {s: idx[s] for s in ("val", "test")}
    if fast:
        rng = np.random.default_rng(config.SEED)
        idx = {s: np.sort(rng.choice(r, size=min(FAST_PTBXL, len(r)),
                                     replace=False)) for s, r in idx.items()}
    return idx


def run_ptbxl(clip: Clip, cache: dict, fast: bool = False,
              n_boot: int = N_BOOT) -> dict:
    """Zero-shot class scores, metrics and modality check on PTB-XL."""
    idx = ptbxl_rows(cache, fast)
    lead = config.LEADS.index(LEAD)
    class_txt = clip.text_features([CLASS_PROMPTS[c] for c in config.CLASSES])
    mod_txt = clip.text_features(MODALITY_PROMPTS)
    renderer = ECGRenderer()
    out = {"probs": {}, "true": {}, "patient_id": {}, "ecg_id": {},
           "modality_top1": {}}
    try:
        for split, rows in idx.items():
            img = clip.image_features(renderer.images(cache["X"][rows, lead]))
            out["probs"][split] = zero_shot_probs(img, class_txt)
            out["modality_top1"][split] = zero_shot_probs(
                img, mod_txt).argmax(axis=1)
            out["true"][split] = cache["y"][rows].astype(np.float32)
            out["patient_id"][split] = cache["patient_id"][rows]
            out["ecg_id"][split] = cache["ecg_id"][rows]
    finally:
        renderer.close()
    thresholds = metrics.best_thresholds(out["true"]["val"], out["probs"]["val"])
    s = {}
    for split in idx:
        y, p = out["true"][split], out["probs"][split]
        m = metrics.classification_metrics(y, p, thresholds)
        ci = metrics.bootstrap_ci(y, p, out["patient_id"][split], thresholds,
                                  n_boot=n_boot, seed=config.SEED)
        s[f"{split}_macro_auroc"] = m["macro_auroc"]
        s[f"{split}_macro_auroc_lo"] = float(ci["macro_auroc"][0])
        s[f"{split}_macro_auroc_hi"] = float(ci["macro_auroc"][1])
        s[f"{split}_macro_f1"] = m["macro_f1"]
        s[f"{split}_macro_f1_lo"] = float(ci["macro_f1"][0])
        s[f"{split}_macro_f1_hi"] = float(ci["macro_f1"][1])
        for j, c in enumerate(config.CLASSES):
            s[f"{split}_auroc_{c}"] = float(m["auroc"][j])
            s[f"{split}_auroc_{c}_lo"] = float(ci["auroc"][0][j])
            s[f"{split}_auroc_{c}_hi"] = float(ci["auroc"][1][j])
            s[f"{split}_f1_{c}"] = float(m["f1"][j])
        s[f"frac_modality_ecg_{split}"] = float(
            np.mean(out["modality_top1"][split] == 0))
        s[f"n_records_{split}"] = int(len(y))
        # how often each sentence wins: a model that always picks the same
        # sentence carries no information
        top = p.argmax(axis=1)
        for j, c in enumerate(config.CLASSES):
            s[f"frac_top_{c}_{split}"] = float(np.mean(top == j))
    for c, t in zip(config.CLASSES, thresholds):
        s[f"threshold_{c}"] = float(t)
    out["scalars"] = s
    out["thresholds"] = thresholds
    return out


# ---------------------------------------------------------------------------
# 4. Part A: CIFAR-100
# ---------------------------------------------------------------------------
def cifar_images(fast: bool = False):
    """CIFAR-100 test set as (list of PIL images, labels, class names);
    downloaded by torchvision into data/external/ (not committed)."""
    from torchvision.datasets import CIFAR100
    ds = CIFAR100(str(DATA_DIR / "cifar100"), train=False, download=True)
    n = len(ds)
    rows = (np.sort(np.random.default_rng(config.SEED).choice(
        n, FAST_CIFAR, replace=False)) if fast else np.arange(n))
    images = [ds[int(i)][0] for i in rows]
    labels = np.asarray([ds.targets[int(i)] for i in rows])
    return images, labels, list(ds.classes)


def accuracy_ci(correct: np.ndarray, n_boot: int = N_BOOT,
                seed: int = config.SEED) -> tuple[float, float]:
    """Percentile bootstrap interval of an accuracy, resampling images
    (one image per draw: CIFAR images are independent)."""
    rng = np.random.default_rng(seed)
    n = len(correct)
    vals = [correct[rng.integers(0, n, n)].mean() for _ in range(n_boot)]
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def run_cifar(clip: Clip, images, labels, classes, n_boot: int = N_BOOT) -> dict:
    """Top-1 / top-5 zero-shot accuracy with each prompt template."""
    img = clip.image_features(images)
    s = {"cifar_n_images": int(len(labels)), "cifar_n_classes": len(classes),
         "cifar_chance_top_one": 1.0 / len(classes)}
    top1_examples = {}
    for name, template in CIFAR_TEMPLATES.items():
        probs = zero_shot_probs(img, clip.text_features(
            [template.format(c) for c in classes]))
        top5 = np.argsort(-probs, axis=1)[:, :5]
        for k, ok in (("one", top5[:, 0] == labels),
                      ("five", (top5 == labels[:, None]).any(axis=1))):
            s[f"cifar_top_{k}_{name}"] = float(ok.mean())
            lo, hi = accuracy_ci(ok.astype(float), n_boot)
            s[f"cifar_top_{k}_lo_{name}"] = lo
            s[f"cifar_top_{k}_hi_{name}"] = hi
        top1_examples[name] = top5[:8, 0].tolist()
    return {"scalars": s, "examples": top1_examples}


# ---------------------------------------------------------------------------
# 5. Log, table, figures
# ---------------------------------------------------------------------------
def write_log(ptb: dict, cifar: dict | None, clip: Clip, fast: bool,
              paths: Paths) -> dict:
    name = RUN_NAME + ("_fast" if fast else "")
    paths.predictions.mkdir(parents=True, exist_ok=True)
    pred = paths.predictions / f"{name}.npz"
    arrays = {}
    for split in ptb["probs"]:
        arrays[f"{split}_prob"] = ptb["probs"][split].astype(np.float32)
        arrays[f"{split}_true"] = ptb["true"][split]
        arrays[f"{split}_patient_id"] = np.asarray(ptb["patient_id"][split])
        arrays[f"{split}_ecg_id"] = np.asarray(ptb["ecg_id"][split])
    np.savez_compressed(pred, **arrays)
    s = dict(ptb["scalars"])
    if cifar is not None:
        s.update(cifar["scalars"])
    record = {
        "run_id": name, "schema": runlog.SCHEMA_VERSION,
        "task": "zero_shot", "fast": fast, "final": True,
        "scalars": s,
        # seed only so that paired comparisons can match this run with
        # one-seed runs (the trivial model); nothing here is random
        "config": {"model": "zeroshot", "tag": "clip", "seed": config.SEED},
        "model": {"name": f"CLIP {MODEL_NAME}", "weights": "openai",
                  "backend": clip.backend, "trained_here": False},
        "prompts": {"classes": CLASS_PROMPTS, "modality": MODALITY_PROMPTS,
                    "cifar_templates": CIFAR_TEMPLATES},
        "data": {"lead": LEAD, "image_size": IMAGE_SIZE,
                 "renderer": "two rows of 5 s, grid 0.2 s / 0.5 mV, "
                             "y from -1.5 to 2.0 mV",
                 "n_records": {k: len(v) for k, v in ptb["true"].items()},
                 "fast_subset": FAST_PTBXL if fast else None},
        "thresholds": dict(zip(config.CLASSES,
                               map(float, ptb["thresholds"]))),
        "bootstrap": {"n_boot": 50 if fast else N_BOOT, "unit": "patient",
                      "seed": config.SEED},
        "predictions": rel(pred),
        "meta": runlog.environment(clip.device),
    }
    runlog.write_run(record, paths.logs)
    return record


def _m(key: str) -> str:
    return "\\" + dx.make_numbers_module().macro_name(ALIAS, key) + "{}"


def table(record: dict) -> str:
    s = record["scalars"]
    lines = [
        "% Generated by src/zero_shot.py from results/logs/"
        f"{record['run_id']}.json.",
        "% Needs report/numbers.tex (alias \"zeroshot\") and booktabs.",
        "\\begin{tabular}{llc}", "\\toprule",
        "Data & Measure & Value [95\\% CI] \\\\", "\\midrule"]
    if "cifar_top_one_notebook" in s:
        for name, label in (("notebook", "``This is a photo of a \\{label\\}''"),
                            ("label", "bare label")):
            for k, kl in (("one", "top-1"), ("five", "top-5")):
                lines.append(
                    f"CIFAR-100 & {kl} accuracy, {label} & "
                    f"{_m(f'cifar_top_{k}_{name}')} "
                    f"[{_m(f'cifar_top_{k}_lo_{name}')}, "
                    f"{_m(f'cifar_top_{k}_hi_{name}')}] \\\\")
        lines.append(f"CIFAR-100 & chance (top-1) & "
                     f"{_m('cifar_chance_top_one')} \\\\")
        lines.append("\\midrule")
    lines.append(f"PTB-XL lead II & images seen as ``an electrocardiogram'' "
                 f"& {_m('frac_modality_ecg_test')} \\\\")
    for split, sl in (("val", "validation"), ("test", "test")):
        lines.append(f"PTB-XL lead II & macro AUROC, {sl} & "
                     f"{_m(split + '_macro_auroc')} "
                     f"[{_m(split + '_macro_auroc_lo')}, "
                     f"{_m(split + '_macro_auroc_hi')}] \\\\")
    for c in config.CLASSES:
        lines.append(f"PTB-XL lead II & AUROC {c}, test & "
                     f"{_m(f'test_auroc_{c}')} "
                     f"[{_m(f'test_auroc_{c}_lo')}, {_m(f'test_auroc_{c}_hi')}]"
                     " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(lines)


def figure_examples(cache: dict, ptb: dict, paths: Paths, suffix: str) -> bool:
    """The images CLIP receives for one validation record per superclass
    (the records of D16-D19 when they are in use) and the five sentence
    probabilities of each."""
    from src import cnn_experiments as cx
    val_ids = list(ptb["ecg_id"]["val"])
    ex = {c: r for c, r in cx.example_records(cache).items()
          if cache["ecg_id"][r] in val_ids}
    if not ex:                        # --fast: first record of each class
        y = ptb["true"]["val"]
        ex = {}
        for j, c in enumerate(config.CLASSES):
            hit = np.flatnonzero(y[:, j] == 1)
            if len(hit):
                ex[c] = int(np.flatnonzero(cache["ecg_id"]
                                           == ptb["ecg_id"]["val"][hit[0]])[0])
    if not ex:
        return False
    import matplotlib.pyplot as plt
    eda = dx._style()
    renderer = ECGRenderer()
    lead = config.LEADS.index(LEAD)
    try:
        imgs = {c: renderer.render(cache["X"][r, lead]) for c, r in ex.items()}
    finally:
        renderer.close()
    fig, axes = plt.subplots(2, len(ex), figsize=(2.3 * len(ex), 4.6),
                             gridspec_kw={"height_ratios": [1.3, 1]})
    axes = np.asarray(axes).reshape(2, -1)
    for k, (c, r) in enumerate(ex.items()):
        j = val_ids.index(cache["ecg_id"][r])
        p = ptb["probs"]["val"][j]
        axes[0, k].imshow(imgs[c])
        axes[0, k].set_axis_off()
        axes[0, k].set_title(f"true: {c} (ecg_id {cache['ecg_id'][r]})",
                             fontsize=8)
        colours = [eda.SERIES_COLOURS[1] if cc == c else eda.SERIES_COLOURS[0]
                   for cc in config.CLASSES]
        axes[1, k].barh(range(len(p)), p, color=colours)
        axes[1, k].set_yticks(range(len(p)), config.CLASSES, fontsize=7)
        axes[1, k].invert_yaxis()
        axes[1, k].set_xlim(0, 1)
        axes[1, k].set_xlabel("CLIP probability", fontsize=7)
    fig.suptitle("Zero-shot CLIP on lead II images (validation records)",
                 color=eda.INK, fontsize=10)
    dx._save(fig, paths, "zeroshot_examples" + suffix)
    return True


def figure_ptbxl(record: dict, paths: Paths, suffix: str) -> bool:
    """Test AUROC per superclass with its interval, against chance."""
    s = record["scalars"]
    import matplotlib.pyplot as plt
    eda = dx._style()
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    names = config.CLASSES + ["macro"]
    vals = [s[f"test_auroc_{c}"] for c in config.CLASSES] + \
        [s["test_macro_auroc"]]
    lo = [s[f"test_auroc_{c}_lo"] for c in config.CLASSES] + \
        [s["test_macro_auroc_lo"]]
    hi = [s[f"test_auroc_{c}_hi"] for c in config.CLASSES] + \
        [s["test_macro_auroc_hi"]]
    x = np.arange(len(names))
    ax.bar(x, vals, color=[eda.SERIES_COLOURS[0]] * 5 + [eda.INK], width=0.6)
    ax.errorbar(x, vals, yerr=[np.subtract(vals, lo), np.subtract(hi, vals)],
                fmt="none", ecolor=eda.INK_MUTED, capsize=3, linewidth=1)
    ax.axhline(0.5, color=eda.INK_MUTED, linestyle="--", linewidth=1)
    ax.text(len(names) - 0.5, 0.5, " chance", color=eda.INK_MUTED, fontsize=8,
            va="bottom", ha="right")
    ax.set_xticks(x, names)
    ax.set_ylim(0, 1)
    ax.set_ylabel("test AUROC (95% CI)")
    ax.set_title("Zero-shot CLIP, lead II images, test fold", fontsize=10)
    dx._save(fig, paths, "zeroshot_ptbxl" + suffix)
    return True


def run(fast: bool = False, skip_cifar: bool = False, cache: dict | None = None,
        paths: Paths = Paths(), clip: Clip | None = None,
        cifar_data=None) -> dict:
    suffix = "_fast" if fast else ""
    n_boot = 50 if fast else N_BOOT
    clip = clip or load_clip()
    print(f"CLIP {MODEL_NAME} ({clip.backend}) on {clip.device}")
    cache = dx._load_cache(cache)
    ptb = run_ptbxl(clip, cache, fast, n_boot)
    print(f"PTB-XL test macro AUROC {ptb['scalars']['test_macro_auroc']:.3f}; "
          f"images seen as an electrocardiogram: "
          f"{ptb['scalars']['frac_modality_ecg_test']:.1%}")
    cifar = None
    if not skip_cifar:
        images, labels, classes = cifar_data or cifar_images(fast)
        cifar = run_cifar(clip, images, labels, classes, n_boot)
        print(f"CIFAR-100 top-1 accuracy (notebook prompt): "
              f"{cifar['scalars']['cifar_top_one_notebook']:.3f}")
    record = write_log(ptb, cifar, clip, fast, paths)
    paths.tables.mkdir(parents=True, exist_ok=True)
    (paths.tables / f"zeroshot{suffix}.tex").write_text(table(record),
                                                        encoding="utf-8")
    figure_examples(cache, ptb, paths, suffix)
    figure_ptbxl(record, paths, suffix)
    if not fast:
        from src.transfer_experiments import register_alias
        register_alias(paths, ALIAS, RUN_NAME)
    return record


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("step", choices=["run"])
    p.add_argument("--fast", action="store_true")
    p.add_argument("--skip-cifar", action="store_true",
                   help="PTB-XL part only (no CIFAR-100 download)")
    a = p.parse_args(argv)
    t0 = datetime.now()
    run(a.fast, a.skip_cifar)
    print(f"\ndone in {datetime.now() - t0}")


if __name__ == "__main__":
    main()
