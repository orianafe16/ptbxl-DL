"""Tests of chat DL 08 (DECISIONS D20): transfer from the ECG-GPT (T13),
12 leads vs lead II (T14), zero-shot CLIP (T24) and the final comparison
(T15). Unit tests of each piece, then the whole pipeline on a small
synthetic 12-lead cache, on the CPU, with small generators standing in for
the D19 runs and a fake CLIP model (no download).

Run from the repository root:  python -m pytest -q tests
"""
import json
import math
import re

import numpy as np
import pytest
import torch

from src import cnn_experiments as cx
from src import comparison as cmp
from src import config, runlog
from src import dense_experiments as dx
from src import generative as gx
from src import generative_experiments as ge
from src import leads_experiments as lx
from src import rnn_experiments as rx
from src import transfer as tf
from src import transfer_experiments as tx
from src import transformer_experiments as trx
from src import zero_shot as zs
from src.dataset import split_indices
from src.models.generative import build_generator


# ------------------------------------------------------------ synthetic data
def synthetic_cache(n_per_fold: int = 24, n_samples: int = 1000,
                    seed: int = 0) -> dict:
    """10 folds of 12-lead records on the microvolt grid; lead II is a train
    of beats whose height depends on the first label (so there is something
    to learn); every class is positive in every fold; every 7th record has
    no superclass; two records per patient."""
    rng = np.random.default_rng(seed)
    n = 10 * n_per_fold
    t = np.arange(n_samples)
    X = rng.normal(scale=0.05, size=(n, 12, n_samples)).astype(np.float32)
    y = (rng.random((n, 5)) < 0.25).astype(np.float32)
    y[np.arange(n), np.arange(n) % 5] = 1
    y[::7] = 0
    lead2 = config.LEADS.index("II")
    for i in range(n):
        rr = int(rng.integers(60, 101))
        h = 0.6 + 0.6 * y[i, 0]
        x = 0.02 * rng.normal(size=n_samples)
        for p in range(int(rng.integers(0, rr)), n_samples, rr):
            x += h * np.exp(-0.5 * ((t - p) / 1.5) ** 2)
            x += 0.2 * np.exp(-0.5 * ((t - p - 25) / 6.0) ** 2)
        X[i, lead2] = np.round(x * 1000) / 1000
        X[i] += 0.3 * y[i, 1]
    return {"X": X, "y": y, "ecg_id": np.arange(1, n + 1),
            "patient_id": 1000 + np.arange(n) // 2,
            "fold": np.repeat(np.arange(1, 11), n_per_fold),
            "has_superclass": y.sum(axis=1) > 0}


SMALL_GPT = {"arch": "gpt", "vocab": 64, "context": 100, "d_model": 32,
             "num_heads": 2, "num_layers": 1, "d_ff": 64, "dropout": 0.1,
             "pos_encoding": "sinusoidal", "causal": True}
SMALL_LSTM = {"arch": "lstm", "vocab": 64, "context": 100, "d_embed": 32,
              "hidden_size": 24, "num_layers": 1, "dropout": 0.1}


def fake_generators(paths, seeds, lo_uv: int = -1138) -> None:
    """Small generator "runs" of every selectable D19 configuration and a
    fast generative summary with pretrained_for_chat_08, as chat DL 13
    leaves them (logs, checkpoints, tokenizer, bits per sample)."""
    tok = gx.make_tokenizer(lo_uv, 64)
    paths.checkpoints.mkdir(parents=True, exist_ok=True)
    handoff = {"checkpoints": {}, "run_ids": {}}
    scalars = {}
    for k, ab in enumerate(a for a in ge.ABLATIONS if a.selectable):
        args = dict(SMALL_LSTM if ab.base == "lstm" else SMALL_GPT)
        scalars[f"bits_mean_{ab.tag}"] = 6.0 - 0.1 * k
        for seed in seeds:
            torch.manual_seed(seed + k)
            model = build_generator(args)
            run_id = f"generator_{ab.tag}_s{seed}_20260101-000000_fast"
            ckpt = paths.checkpoints / f"{run_id}.pt"
            torch.save(model.state_dict(), ckpt)
            runlog.write_run({"run_id": run_id, "fast": True, "final": False,
                              "config": {"model": "generator", "tag": ab.tag,
                                         "seed": seed},
                              "model": {"args": args},
                              "checkpoint": str(ckpt),
                              "data": {"tokenizer": tok.to_dict()}},
                             paths.logs)
            if ab.tag == "vocab2560":
                handoff["checkpoints"][str(seed)] = str(ckpt)
                handoff["run_ids"][str(seed)] = run_id
    handoff.update({"selected_config": "vocab2560",
                    "model_args": dict(SMALL_GPT),
                    "tokenizer": tok.to_dict(), "lead": "II"})
    runlog.write_run({"run_id": "generative_summary_fast", "fast": True,
                      "scalars": scalars,
                      "pretrained_for_chat_08": handoff}, paths.logs)


class FakeClipModel(torch.nn.Module):
    """Stands in for CLIP: image features from mean colour per row band,
    text features from the token ids, both through fixed random maps."""

    def __init__(self, d: int = 8):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.img = torch.randn(12, d, generator=g)
        self.txt = torch.randn(64, d, generator=g)

    def encode_image(self, x):                     # (n, 3, H, W)
        bands = x.reshape(x.shape[0], 3, 4, -1).mean(dim=-1).reshape(
            x.shape[0], 12)
        return bands @ self.img

    def encode_text(self, t):                      # (n, L) ids
        return self.txt[t % 64].sum(dim=1)


def fake_clip() -> zs.Clip:
    def preprocess(im):
        a = np.asarray(im.convert("RGB").resize((32, 32)), dtype=np.float32)
        return torch.from_numpy(a / 255.0).permute(2, 0, 1)

    def tokenize(texts):
        return torch.tensor([[ord(c) for c in (s + " " * 16)[:16]]
                             for s in texts])
    return zs.Clip(FakeClipModel(), preprocess, tokenize, "fake",
                   torch.device("cpu"))


# --------------------------------------------------------------- label subset
def test_label_subset_by_patient():
    cache = synthetic_cache()
    rows = split_indices(cache)["train"]
    sub = tf.label_subset(rows, cache["patient_id"], 0.1, 42)
    assert len(sub) >= math.ceil(0.1 * len(rows))
    assert len(sub) <= math.ceil(0.1 * len(rows)) + 2   # at most one patient more
    pats = set(cache["patient_id"][sub])
    whole = rows[np.isin(cache["patient_id"][rows], list(pats))]
    assert np.array_equal(np.sort(whole), sub)           # all their records
    assert np.array_equal(sub, tf.label_subset(rows, cache["patient_id"],
                                               0.1, 42))
    assert not np.array_equal(sub, tf.label_subset(rows, cache["patient_id"],
                                                   0.1, 43))
    assert np.array_equal(tf.label_subset(rows, cache["patient_id"], 1.0, 42),
                          rows)
    with pytest.raises(ValueError):
        tf.label_subset(rows, cache["patient_id"], 0.0, 42)


def test_transfer_data_follows_the_classifier_order():
    cache = synthetic_cache()
    data = tf.TransferData(cache)
    idx = split_indices(cache)
    for s in ("train", "val", "test"):
        assert np.array_equal(data.ecg_id[s], cache["ecg_id"][idx[s]])
        assert np.array_equal(data.y[s], cache["y"][idx[s]])
    tok = gx.make_tokenizer(-1138, 64)
    t = data.tokens(tok, "val")
    assert t.dtype == np.int16 and t.shape == (len(idx["val"]), 1000)
    assert np.array_equal(t, tok.encode(
        cache["X"][idx["val"], config.LEADS.index("II")]))
    pos = data.train_positions(0.1, 42)
    d = data.describe(pos, 0.1)
    assert d["n_records"]["train"] == len(pos) and d["label_fraction"] == 0.1
    assert sum(d["train_class_counts"].values()) == int(
        data.y["train"][pos].sum())


# ---------------------------------------------------------------- models
def test_backbone_keeps_the_pretrained_weights_and_drops_the_head(tmp_path):
    torch.manual_seed(0)
    gen = build_generator(SMALL_GPT)
    ckpt = tmp_path / "g.pt"
    torch.save(gen.state_dict(), ckpt)
    bb = tf.load_backbone(SMALL_GPT, ckpt)
    assert not hasattr(bb, "head")
    tokens = torch.randint(0, 64, (3, 1000))
    gen.eval(), bb.eval()
    assert torch.allclose(gen.record_features(tokens), bb.record_features(tokens))
    scratch = tf.load_backbone(SMALL_GPT, None)
    assert not torch.allclose(scratch.tok_emb.weight, bb.tok_emb.weight)
    n_head = 32 * 64 + 64
    assert sum(p.numel() for p in bb.parameters()) == \
        sum(p.numel() for p in gen.parameters()) - n_head


@pytest.mark.parametrize("args", [SMALL_GPT, SMALL_LSTM])
def test_frozen_probe_trains_only_the_head_and_keeps_the_backbone_in_eval(args):
    model = tf.TransferClassifier(tf.load_backbone(args), freeze=True)
    model.train()
    assert not model.backbone.training and model.drop.training
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable == ["head.weight", "head.bias"]
    d = tf.feature_dim(model.backbone)
    assert model.head.in_features == d
    tokens = torch.randint(0, 64, (4, 1000))
    model.eval()
    with torch.no_grad():
        a = model(tokens)
    head = tf.FeatureHead(d)
    head.load_state_dict({"head.weight": model.head.weight,
                          "head.bias": model.head.bias})
    feats = tf.extract_features(model.backbone, tokens.numpy().astype(np.int16))
    with torch.no_grad():
        b = head.eval()(torch.from_numpy(feats))
    assert torch.allclose(a, b, atol=1e-5)
    # in training mode the frozen backbone gives the same vector every time
    model.train()
    with torch.no_grad():
        f1 = model.backbone.record_features(tokens)
        f2 = model.backbone.record_features(tokens)
    assert torch.equal(f1, f2)


def test_finetune_model_trains_everything():
    model = tf.TransferClassifier(tf.load_backbone(SMALL_GPT), freeze=False)
    assert all(p.requires_grad for p in model.parameters())
    model.train()
    assert model.backbone.training


# ---------------------------------------------------------------- grid
def test_grid_and_settings():
    main = [c for c in tx.GRID if c.main]
    assert [c.tag for c in main] == ["scratch_f100", "probe_f100",
                                     "finetune_f100", "scratch_f10",
                                     "probe_f10", "finetune_f10"]
    assert {c.lr for c in main if c.strategy == "finetune"} == {1e-4}
    assert {c.lr for c in main if c.strategy != "finetune"} == {1e-3}
    assert tx.BY_TAG["finetune_lr1e3_f100"].lr == 1e-3
    assert tx.BY_TAG["scratch_lr1e4_f10"].lr == 1e-4
    study = [c for c in tx.GRID if c.group == "probe_study"]
    gens = {c.generator for c in study}
    assert "vocab2560" not in gens and "no_mask" not in gens
    assert gens == {a.tag for a in ge.ABLATIONS if a.selectable} - {"vocab2560"}
    for name, a, b, _ in tx.comparisons():
        assert a in tx.BY_TAG and b in tx.BY_TAG, name
    order = tx.run_order([c.tag for c in tx.GRID])
    assert order[:6] == ["probe_f10", "probe_f100", "scratch_f10",
                         "finetune_f10", "scratch_f100", "finetune_f100"]
    assert [tx.BY_TAG[t].group for t in order[6:10]] == ["lr"] * 4
    gen = {"tag": "vocab2560", "args": SMALL_GPT, "run_id": "r"}
    cfg = tx.make_config(tx.BY_TAG["finetune_f10"], 42, gen)
    assert (cfg.batch_size, cfg.grad_clip, cfg.lr, cfg.normalize) == \
        (32, 1.0, 1e-4, False)
    assert (cfg.max_epochs, cfg.patience) == (50, 5)


# ------------------------------------------------------------ zero-shot units
def test_ecg_renderer_draws_a_224_square():
    r = zs.ECGRenderer()
    try:
        x = np.sin(np.linspace(0, 60, 1000)).astype(np.float32)
        img = r.render(x)
        img2 = r.render(np.zeros(1000, dtype=np.float32))
    finally:
        r.close()
    assert img.shape == (224, 224, 3) and img.dtype == np.uint8
    assert (img < 60).any()                         # a dark trace
    assert not np.array_equal(img, img2)


def test_zero_shot_probabilities():
    img = torch.nn.functional.normalize(torch.randn(6, 8), dim=-1)
    txt = torch.nn.functional.normalize(torch.randn(5, 8), dim=-1)
    p = zs.zero_shot_probs(img, txt)
    assert p.shape == (6, 5) and np.allclose(p.sum(axis=1), 1, atol=1e-5)
    assert np.array_equal(p.argmax(axis=1), (img @ txt.T).argmax(dim=1).numpy())
    assert list(zs.CLASS_PROMPTS) == config.CLASSES


def test_accuracy_interval():
    ok = np.array([1.0] * 70 + [0.0] * 30)
    lo, hi = zs.accuracy_ci(ok, 200)
    assert lo < 0.7 < hi


# ------------------------------------------------------------ whole pipeline
@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    root = tmp_path_factory.mktemp("strategies")
    paths = dx.Paths(logs=root / "logs", predictions=root / "pred",
                     checkpoints=root / "ckpt", figures=root / "fig",
                     tables=root / "tab", runs_file=root / "runs.json")
    paths.runs_file.write_text('{"trivial": "trivial_full"}', encoding="utf-8")
    seeds = (42, 43)
    cache = synthetic_cache()

    # 12-lead final runs of every family, as their blocks leave them
    blocks = [("dense", dx, "no_norm"), ("cnn", cx, "no_dropout"),
              ("rnn", rx, "conv"), ("rnn", rx, "raw_attn"),
              ("transformer", trx, "no_pos")]
    for _, mod, tag in blocks:
        for seed in seeds:
            mod._train(mod.BY_TAG[tag], seed, True, True, cache, paths)
    for alias, sel in (("dense", "no_norm"), ("cnn", "no_dropout"),
                       ("transformer", "no_pos")):
        runlog.write_run({"run_id": f"{alias}_summary_fast", "fast": True,
                          "scalars": {}, "selected_config": sel}, paths.logs)
    runlog.write_run({"run_id": "rnn_summary_fast", "fast": True,
                      "scalars": {}, "selected_config": "conv",
                      "selected_attention_config": "raw_attn"}, paths.logs)

    # T13
    fake_generators(paths, seeds)
    data = tf.TransferData(cache, fast=True)
    ctx = tx.Context(paths, fast=True)
    estimate = tx.estimate(n_steps=1, seeds=seeds, fast=True, paths=paths,
                           data=data, ctx=ctx)
    tx.run_all(seeds, fast=True, paths=paths, data=data, ctx=ctx)
    transfer = tx.write_report(fast=True, paths=paths, seeds=seeds, ctx=ctx)

    # T14
    lx.run_all(seeds, fast=True, cache=cache, paths=paths)
    leads = lx.write_report(fast=True, paths=paths, seeds=seeds)

    # T24 (fake CLIP, tiny fake CIFAR)
    from PIL import Image
    rng = np.random.default_rng(0)
    imgs = [Image.fromarray(rng.integers(0, 255, (32, 32, 3), dtype=np.uint8))
            for _ in range(40)]
    zero = zs.run(fast=True, cache=cache, paths=paths, clip=fake_clip(),
                  cifar_data=(imgs, rng.integers(0, 4, 40),
                              ["apple", "bee", "car", "dog"]))

    # T15, with a published entry
    pub = root / "published.json"
    pub.write_text(json.dumps({"entries": [
        {"key": "paper_model", "label": "Some model", "value": 0.9,
         "metric": "macro AUROC", "source": "key2021", "file": "a.pdf",
         "page": 5, "quote": "the literal sentence"}]}), encoding="utf-8")
    final = cmp.write_report(fast=True, paths=paths, seeds=seeds,
                             published_path=pub)
    return {"paths": paths, "seeds": seeds, "cache": cache, "data": data,
            "transfer": transfer, "leads": leads, "zero": zero,
            "final": final, "estimate": estimate}


def test_estimate(pipeline):
    e = pipeline["estimate"]
    assert e["trainable_f100"] > 0 and e["features_s"] > 0


def test_transfer_runs(pipeline):
    paths, seeds = pipeline["paths"], pipeline["seeds"]
    runs = tx.all_runs(paths, fast=True)
    assert set(runs) == {(c.tag, s) for c in tx.GRID for s in seeds}
    for (tag, seed), rec in runs.items():
        c = tx.BY_TAG[tag]
        assert rec["final"] == c.main
        assert any(k.startswith("test_") for k in rec["scalars"]) == c.main
        assert rec["config"]["batch_size"] == 32
        assert rec["config"]["grad_clip"] == 1.0
        assert rec["config"]["lr"] == c.lr
        assert rec["data"]["cublas_workspace_config"] == ":4096:2"
        if c.strategy == "probe":
            d = SMALL_LSTM["hidden_size"] if c.generator == "lstm" \
                else SMALL_GPT["d_model"]
            assert rec["scalars"]["n_trainable"] == d * 5 + 5
            assert rec["data"]["features"]["frozen"]
            assert rec["data"]["n_params_backbone"] > 0
        else:
            assert rec["scalars"]["n_trainable"] == rec["scalars"]["n_params"]
        if c.strategy == "scratch":
            assert rec["model"]["args"]["generator_run"] is None
        else:
            assert rec["model"]["args"]["generator_run"].endswith(
                f"_s{seed}_20260101-000000_fast")
            assert f"generator_{c.generator if c.generator != tx.PRETRAINED else 'vocab2560'}_" \
                in rec["model"]["args"]["generator_run"]
    # same 10 % subset for the three strategies of a seed, another per seed
    sub = {s: runs[("scratch_f10", s)]["data"]["train_class_counts"]
           for s in seeds}
    for st in ("probe", "finetune"):
        for s in seeds:
            assert runs[(f"{st}_f10", s)]["data"]["train_class_counts"] == sub[s]
    assert runs[("scratch_f10", 42)]["data"]["n_records"]["train"] < \
        runs[("scratch_f100", 42)]["data"]["n_records"]["train"]


def test_transfer_predictions_pair_with_other_classifiers(pipeline):
    paths = pipeline["paths"]
    runs = tx.all_runs(paths, fast=True)
    a = dx.load_predictions(runs[("probe_f100", 42)], "test")
    cnn = cx.find_runs(paths, True, True, model="cnn")[("no_dropout", 42)]
    b = dx.load_predictions(cnn, "test")
    # fast subsets are drawn differently, but every id must be a test record
    cache = pipeline["cache"]
    test_ids = cache["ecg_id"][split_indices(cache)["test"]]
    assert set(a["ecg_id"]) <= set(test_ids) and set(b["ecg_id"]) <= set(test_ids)


def test_transfer_summary(pipeline):
    s = pipeline["transfer"]["scalars"]
    for t in tx.MAIN_TAGS:
        assert f"test_auroc_mean_{t}" in s and f"val_auroc_mean_{t}" in s
    assert "test_auroc_mean_finetune_lr1e3_f100" not in s
    assert "val_auroc_mean_finetune_lr1e3_f100" in s
    assert s["test_n_compared_finetune_vs_scratch_f100"] == 2
    assert "test_diff_mean_lr1e3_vs_finetune_f100" not in s
    assert "val_diff_mean_lr1e3_vs_finetune_f100" in s
    assert "probe_spearman_f100" in s
    runs = tx.all_runs(pipeline["paths"], fast=True)
    d = dx.paired_diff(runs[("finetune_f10", 42)], runs[("scratch_f10", 42)],
                       "test", 50)
    assert s["test_diff_finetune_vs_scratch_f10_s42"] == pytest.approx(d["diff"])
    assert len(pipeline["transfer"]["probe_study"]) == \
        len([a for a in ge.ABLATIONS if a.selectable])


def test_leads_runs_copy_the_twelve_lead_settings(pipeline):
    paths, seeds = pipeline["paths"], pipeline["seeds"]
    one = lx.find_runs(paths, fast=True)
    assert set(one) == {(f[0], s) for f in lx.FAMILIES for s in seeds}
    for (family, seed), rec in one.items():
        twelve = lx.twelve_lead_finals(paths, family, True)[seed]
        assert rec["model"]["args"] == {**twelve["config"]["model_args"],
                                        "n_leads": 1}
        for k in ("lr", "grad_clip", "normalize", "batch_size", "patience"):
            assert rec["config"][k] == twelve["config"][k]
        assert rec["final"] and rec["data"]["lead"] == "II"
        a, b = dx.load_predictions(rec, "test"), dx.load_predictions(twelve,
                                                                     "test")
        assert np.array_equal(a["ecg_id"], b["ecg_id"])    # paired
    assert pipeline["leads"]["selected_configs"]["rnnattn"] == "raw_attn"
    s = pipeline["leads"]["scalars"]
    assert s["params_onelead_cnn"] < s["params_twelve_cnn"]
    assert s["test_n_compared_transformer"] == 2


def test_zero_shot_log(pipeline):
    rec = pipeline["zero"]
    s = rec["scalars"]
    assert 0 <= s["test_macro_auroc"] <= 1
    assert 0 <= s["frac_modality_ecg_test"] <= 1
    assert sum(s[f"frac_top_{c}_test"] for c in config.CLASSES) == \
        pytest.approx(1)
    assert "cifar_top_one_notebook" in s and "cifar_top_five_label" in s
    p = dx.load_predictions(rec, "test")
    assert p["prob"].shape[1] == 5
    assert np.allclose(p["prob"].sum(axis=1), 1, atol=1e-5)
    assert rec["model"]["trained_here"] is False


def test_comparison(pipeline):
    summary = pipeline["final"]
    keys = {r["key"] for r in summary["rows"]}
    assert {f"twelve_{f[0]}" for f in lx.FAMILIES} <= keys
    assert {f"onelead_{f[0]}" for f in lx.FAMILIES} <= keys
    assert set(tx.MAIN_TAGS) <= keys and "zeroshot" in keys
    s = summary["scalars"]
    assert s["published_paper_model"] == 0.9
    assert "test_diff_mean_finetune_f100_vs_onelead_cnn" not in s or \
        s["test_n_compared_finetune_f100_vs_onelead_cnn"] >= 0


def test_published_entries_need_a_quote(tmp_path):
    p = tmp_path / "pub.json"
    p.write_text(json.dumps({"entries": [{"key": "x", "label": "x",
                                          "value": 0.9}]}), encoding="utf-8")
    with pytest.raises(ValueError):
        cmp.published(p)
    assert cmp.published(tmp_path / "missing.json") == []


def test_tables_use_only_defined_macros(pipeline):
    paths = pipeline["paths"]
    mk = dx.make_numbers_module()
    logs = {"transfer": pipeline["transfer"]["run_id"],
            "leads": pipeline["leads"]["run_id"],
            "zeroshot": pipeline["zero"]["run_id"],
            "comparison": pipeline["final"]["run_id"],
            "generative": "generative_summary_fast"}
    lines, _ = mk.build(logs, logs_dir=paths.logs, allow_fast=True)
    defined = set(re.findall(r"\\newcommand\{\\(\w+)\}", "\n".join(lines)))
    files = list(paths.tables.glob("*_fast.tex"))
    names = {f.name for f in files}
    for n in ("transfer_main_fast.tex", "transfer_comparisons_fast.tex",
              "transfer_probe_study_fast.tex", "leads_fast.tex",
              "zeroshot_fast.tex", "comparison_final_fast.tex",
              "comparison_per_class_fast.tex"):
        assert n in names, n
    for f in files:
        if not f.name.startswith(("transfer", "leads", "zeroshot",
                                  "comparison")):
            continue
        used = set(re.findall(r"\\((?:transfer|leads|zeroshot|comparison|"
                              r"generative)\w+)\{\}", f.read_text("utf-8")))
        assert used and used <= defined, (f.name, used - defined)


def test_figures_written(pipeline):
    paths = pipeline["paths"]
    for name in ("transfer_curves", "transfer_strategies",
                 "transfer_probe_study", "leads_comparison",
                 "zeroshot_examples", "zeroshot_ptbxl",
                 "comparison_overview", "comparison_label_efficiency"):
        for ext in ("pdf", "png"):
            assert (paths.figures / f"{name}_fast.{ext}").exists(), name


def test_fast_summaries_are_not_registered(pipeline):
    runs = json.loads(pipeline["paths"].runs_file.read_text(encoding="utf-8"))
    assert runs == {"trivial": "trivial_full"}


def test_macro_names_are_unique(pipeline):
    mk = dx.make_numbers_module()
    for alias, summ in (("transfer", pipeline["transfer"]),
                        ("leads", pipeline["leads"]),
                        ("zeroshot", pipeline["zero"]),
                        ("comparison", pipeline["final"])):
        names = [mk.macro_name(alias, k) for k in summ["scalars"]]
        assert len(names) == len(set(names)), alias
