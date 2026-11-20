"""Tests of the CNN block (T10): model, receptive field, class activation
map, ablation grid, selection rule, paired comparisons and the whole
pipeline (runs -> final -> summary, tables, figures) on a small synthetic
cache, on the CPU.

Run from the repository root:  python -m pytest -q tests
"""
import json
import re

import numpy as np
import pytest
import torch
from torch import nn

from src import cnn_experiments as cx
from src import config, runlog
from src.models import MODELS
from src.models.cnn import CNN1D, ConvBlock
from tests.test_dense import synthetic_cache


# ------------------------------------------------------------------ model
def test_registered():
    assert MODELS["cnn"] is CNN1D


@pytest.mark.parametrize("ab", cx.ABLATIONS, ids=lambda a: a.tag)
def test_output_shape_every_configuration(ab):
    """(batch, 12, 1000) in, (batch, 5) logits out, for every ablation."""
    model = CNN1D(**ab.model_args)
    assert model(torch.randn(3, 12, 1000)).shape == (3, 5)


def test_parameter_count_of_the_baseline():
    """Conv1d(c_in, c_out, k): c_out * c_in * k weights + c_out biases;
    BatchNorm1d: 2 parameters per channel (running mean/var are buffers);
    projection shortcut where the channels change (12->32, 32->64, 64->128),
    identity in block 4 (128->128); GAP has no parameters."""
    k = 7

    def conv(ci, co, kk=k):
        return co * ci * kk + co

    def bn(c):
        return 2 * c

    expected = 0
    for ci, co in [(12, 32), (32, 64), (64, 128), (128, 128)]:
        expected += conv(ci, co) + bn(co) + conv(co, co) + bn(co)
        if ci != co:
            expected += conv(ci, co, 1) + bn(co)
    expected += 128 * 5 + 5
    model = CNN1D(**cx.BASE_ARGS)
    assert sum(p.numel() for p in model.parameters()) == expected == 468325


def test_logits_not_probabilities():
    """The last layer is Linear: no sigmoid inside the model (D2)."""
    model = CNN1D(n_samples=200)
    assert isinstance(model.classifier, nn.Linear)
    torch.manual_seed(0)
    out = model.eval()(torch.randn(16, 12, 200) * 10)
    assert (out < 0).any() or (out > 1).any()


def test_shortcuts():
    """Identity when the channels do not change, 1x1 projection when they
    do, nothing without residual connections."""
    same = ConvBlock(16, 16, 7)
    proj = ConvBlock(12, 16, 7)
    plain = ConvBlock(12, 16, 7, residual=False)
    assert isinstance(same.shortcut, nn.Identity)
    assert isinstance(proj.shortcut[0], nn.Conv1d)
    assert proj.shortcut[0].kernel_size == (1,)
    assert plain.shortcut is None
    x = torch.randn(4, 12, 50)
    assert proj(x).shape == plain(x).shape == (4, 16, 25)   # pooled by 2


def test_residual_adds_the_input():
    """With the main path's last batch norm set to zero, F(x) = 0 and the
    block returns pool(ReLU(x)): the identity is easy to learn (slide 54)."""
    block = ConvBlock(8, 8, 5).eval()
    nn.init.zeros_(block.bn2.weight)
    nn.init.zeros_(block.bn2.bias)
    x = torch.randn(2, 8, 40)
    expected = nn.functional.max_pool1d(torch.relu(x), 2)
    assert torch.allclose(block(x), expected)


def test_no_batch_norm_leaves_no_batch_norm():
    model = CNN1D(batch_norm=False)
    assert not any(isinstance(m, nn.BatchNorm1d) for m in model.modules())


def test_flatten_head_sees_every_position():
    model = CNN1D(pooling="flatten")
    assert model.classifier.in_features == 128 * 62
    with pytest.raises(ValueError):
        model.class_activation_map(torch.randn(1, 12, 1000))


def test_single_lead_input():
    """Chat 08 trains the best model of each family on lead II only."""
    model = CNN1D(n_leads=1)
    assert model.first_conv.weight.shape == (32, 1, 7)
    assert model(torch.randn(2, 1, 1000)).shape == (2, 5)


def test_dropout_only_in_training_mode():
    torch.manual_seed(0)
    model = CNN1D(n_samples=200)
    x = torch.randn(8, 12, 200)
    model.train()
    assert not torch.equal(model(x), model(x))
    model.eval()
    assert torch.equal(model(x), model(x))


def test_invalid_arguments():
    for bad in ({"channels": []}, {"kernel_size": 4}, {"pooling": "max"},
                {"dropout": 1.0}, {"channels": [8] * 11}):
        with pytest.raises(ValueError):
            CNN1D(**bad)


@pytest.mark.parametrize("args,expected", [
    ({}, 196), ({"kernel_size": 3}, 76), ({"kernel_size": 15}, 436),
    ({"channels": [32, 64]}, 40), ({"channels": cx.DEEP}, 820)])
def test_receptive_field_analytic_and_measured(args, expected):
    """The formula gives the expected number of samples, and so does a
    measurement: in a linear copy of the network (ReLU -> identity,
    max pooling -> average pooling, same receptive field) the gradient of
    one central output position is non-zero exactly on its receptive
    field."""
    model = CNN1D(batch_norm=False, dropout=0.0, **args)
    assert model.receptive_field() == expected
    for block in model.blocks:
        block.relu = nn.Identity()
        block.pool = nn.AvgPool1d(2)
    torch.manual_seed(0)
    x = torch.randn(1, 12, 1000, requires_grad=True)
    a = model.features(x)
    a[0, :, a.shape[2] // 2].sum().backward()
    support = np.flatnonzero(x.grad.abs().sum(dim=1)[0].numpy() > 0)
    assert support[-1] - support[0] + 1 == expected


def test_class_activation_map_gives_the_logits():
    """logit_c = b_c + mean_t CAM_c(t) in eval mode (D16)."""
    torch.manual_seed(0)
    model = CNN1D().eval()
    x = torch.randn(4, 12, 1000)
    with torch.no_grad():
        cam = model.class_activation_map(x)
        logits = model(x)
    assert cam.shape == (4, 5, 62)
    assert torch.allclose(cam.mean(dim=2) + model.classifier.bias, logits,
                          atol=1e-5)


def test_hooks_capture_and_are_removed():
    model = CNN1D().eval()
    maps = cx.capture_maps(model, torch.randn(2, 12, 1000))
    assert maps["first"].shape == (2, 32, 500)
    assert maps["last"].shape == (2, 128, 62)
    assert maps["cam"].shape == (2, 5, 62)
    assert all(len(b._forward_hooks) == 0 for b in model.blocks)


def test_filter_responses():
    """A filter [1, 1] on one lead is a low-pass filter: |H(0)| = 2,
    |H(50 Hz)| = 0 at 100 Hz."""
    w = np.zeros((1, 12, 2))
    w[0, 0] = [1, 1]
    freqs, R = cx.filter_responses(w, n_fft=8)
    assert freqs[0] == 0 and freqs[-1] == 50
    assert R[0, 0] == pytest.approx(2 / np.sqrt(12))
    assert R[0, -1] == pytest.approx(0, abs=1e-12)


# ------------------------------------------------------------------ grid
def _differences(ab, other):
    a, b = ab.model_args, other.model_args
    return {k for k in a if a[k] != b[k]}


def test_ablation_grid_one_factor_at_a_time():
    tags = [a.tag for a in cx.ABLATIONS]
    assert len(tags) == len(set(tags)) == 10
    assert tags[0] == "base"
    assert _differences(cx.ABLATIONS[0], cx.BY_TAG["base"]) == set()
    for ab in cx.ABLATIONS[1:]:
        assert re.fullmatch(r"[a-z0-9_]+", ab.tag)
        assert ab.reference in cx.BY_TAG and ab.reference != ab.tag
        assert len(_differences(ab, cx.BY_TAG[ab.reference])) == 1, ab.tag
    assert cx.BY_TAG["depth6_no_residual"].reference == "depth6"
    assert _differences(cx.BY_TAG["depth6_no_residual"],
                        cx.BY_TAG["depth6"]) == {"residual"}


def test_configs_keep_the_d14_defaults():
    for ab in cx.ABLATIONS:
        cfg = cx.make_config(ab, 42)
        assert (cfg.lr, cfg.batch_size, cfg.max_epochs, cfg.patience,
                cfg.weight_decay, cfg.normalize) == (1e-3, 64, 50, 5, 0.0,
                                                     True)
        assert cfg.model == "cnn" and cfg.tag == ab.tag
    fast = cx.make_config(cx.ABLATIONS[0], 42, fast=True)
    assert (fast.max_epochs, fast.n_boot) == (2, 50)


# ------------------------------------------------------------- selection
def _fake_runs(means: dict, seeds=(42, 43, 44)):
    return {(tag, s): {"scalars": {"val_macro_auroc": m + 0.001 * i}}
            for tag, m in means.items() for i, s in enumerate(seeds)}


def test_selection_takes_the_best_mean():
    means = {a.tag: 0.80 for a in cx.ABLATIONS}
    means["kernel15"] = 0.85
    assert cx.select_final(_fake_runs(means)) == "kernel15"


def test_selection_refuses_incomplete_grids():
    runs = _fake_runs({a.tag: 0.80 for a in cx.ABLATIONS})
    del runs[("flatten", 44)]
    with pytest.raises(RuntimeError, match="flatten"):
        cx.select_final(runs)


# ------------------------------------------------- paired comparisons
def _fake_final(tmp_path, name, seed, prob, true, ids, model="cnn",
                tag="base"):
    path = tmp_path / f"{name}.npz"
    np.savez(path, test_prob=prob, test_true=true, test_patient_id=ids,
             test_ecg_id=ids)
    return {"run_id": name, "final": True, "fast": False,
            "config": {"model": model, "tag": tag, "seed": seed},
            "scalars": {"val_macro_auroc": 0.7},
            "predictions": str(path)}


def test_compare_finals(tmp_path):
    """Perfect scores minus random scores: positive difference whose
    interval excludes 0; every key written for every seed."""
    rng = np.random.default_rng(0)
    n = 200
    true = (rng.random((n, 5)) < 0.4).astype(np.float32)
    ids = np.arange(n)
    finals = [_fake_final(tmp_path, f"cnn{s}", s, true + 0.0, true, ids)
              for s in (42, 43)]
    others = {s: _fake_final(tmp_path, f"dense{s}", s,
                             rng.random((n, 5)).astype(np.float32), true, ids,
                             model="dense") for s in (42, 43)}
    s = {}
    cx.compare_finals(finals, others, "dense", n_boot=50, s=s)
    for seed in (42, 43):
        assert s[f"diff_dense_lo_s{seed}"] > 0
        assert s[f"diff_dense_s{seed}"] == pytest.approx(0.5, abs=0.1)
    assert s["n_seeds_better_dense"] == 2 and s["n_seeds_compared_dense"] == 2
    assert s["diff_dense_mean"] == pytest.approx(
        np.mean([s["diff_dense_s42"], s["diff_dense_s43"]]))


def test_compare_finals_refuses_misaligned_records(tmp_path):
    true = np.zeros((10, 5), np.float32)
    true[::2] = 1
    a = _fake_final(tmp_path, "a", 42, true, true, np.arange(10))
    b = _fake_final(tmp_path, "b", 42, true, true, np.arange(10)[::-1])
    with pytest.raises(ValueError, match="differ"):
        cx.compare_finals([a], {42: b}, "dense", n_boot=10, s={})


def test_dense_finals_follow_the_dense_summary(tmp_path):
    paths = cx.Paths(logs=tmp_path / "logs", runs_file=tmp_path / "runs.json")
    true = np.zeros((10, 5), np.float32)
    for seed in (42, 43):
        for tag in ("base", "no_norm"):
            rec = _fake_final(tmp_path, f"dense_{tag}_s{seed}_x", seed, true,
                              true, np.arange(10), model="dense", tag=tag)
            runlog.write_run(rec, paths.logs)
    runlog.write_run({"run_id": "dense_summary", "selected_config": "no_norm",
                      "scalars": {}}, paths.logs)
    paths.runs_file.write_text('{"dense": "dense_summary"}', encoding="utf-8")
    found, name = cx.dense_finals(paths)
    assert name == "dense_summary"
    assert {s: r["config"]["tag"] for s, r in found.items()} == \
        {42: "no_norm", 43: "no_norm"}


def test_final_table_dense_macros_exist():
    """The Dense column of cnn_final.tex uses macros that make_numbers.py
    defines from the committed Dense summary (skipped if it is missing)."""
    path = config.LOGS_DIR / "dense_summary.json"
    if not path.exists():
        pytest.skip("results/logs/dense_summary.json not present")
    s = {"final_test_macro_auroc_mean": 0.9}
    table = cx.final_table({"run_id": "x", "scalars": s, "seeds": [42],
                            "dense_summary": "dense_summary"})
    mk = cx.dx.make_numbers_module()
    lines, _ = mk.build({"dense": "dense_summary"})
    defined = set(re.findall(r"\\newcommand\{\\(\w+)\}", "\n".join(lines)))
    used = set(re.findall(r"\\(dense\w+)\{\}", table))
    assert used and used <= defined, used - defined


# ------------------------------------------------------ whole pipeline
@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    root = tmp_path_factory.mktemp("cnn")
    paths = cx.Paths(logs=root / "logs", predictions=root / "pred",
                     checkpoints=root / "ckpt", figures=root / "fig",
                     tables=root / "tab", runs_file=root / "runs.json")
    # A trivial final run on the whole (synthetic) test fold, as in the real
    # report/runs.json: the --fast runs only see a subset of that fold.
    paths.predictions.mkdir(parents=True)
    rng = np.random.default_rng(1)
    n = 30
    np.savez(paths.predictions / "trivial.npz",
             test_prob=rng.random((n, 5)).astype(np.float32),
             test_true=(rng.random((n, 5)) < 0.3).astype(np.float32),
             test_patient_id=np.arange(n), test_ecg_id=np.arange(n))
    runlog.write_run({"run_id": "trivial_full", "final": True, "fast": False,
                      "scalars": {"val_macro_auroc": 0.7},
                      "predictions": str(paths.predictions / "trivial.npz")},
                     paths.logs)
    paths.runs_file.write_text('{"trivial": "trivial_full"}', encoding="utf-8")
    seeds = (42, 43)
    cache = synthetic_cache()
    cx.run_ablations(seeds, None, fast=True, cache=cache, paths=paths)
    selected = cx.run_final(fast=True, cache=cache, paths=paths, seeds=seeds)
    summary = cx.write_report(fast=True, paths=paths, seeds=seeds,
                              cache=cache)
    return paths, seeds, selected, summary


def test_pipeline_runs_and_logs(pipeline):
    paths, seeds, selected, _ = pipeline
    runs = cx.find_runs(paths, fast=True, final=False)
    assert set(runs) == {(a.tag, s) for a in cx.ABLATIONS for s in seeds}
    finals = cx.find_runs(paths, fast=True, final=True)
    assert set(finals) == {(selected, s) for s in seeds}
    for rec in runs.values():                  # ablations never see test
        assert "test_macro_auroc" not in rec["scalars"]
        assert rec["data"]["normalize"] is True
    for rec in finals.values():
        assert "test_macro_auroc" in rec["scalars"]
    assert runs[("flatten", 42)]["model"]["args"]["pooling"] == "flatten"


def test_summary_selection_and_references(pipeline):
    paths, seeds, selected, summary = pipeline
    runs = cx.find_runs(paths, fast=True, final=False)
    scores = cx.val_scores(runs, seeds)
    assert selected == max(scores, key=lambda t: np.mean(scores[t]))
    assert summary["selected_config"] == selected
    s = summary["scalars"]
    assert s["val_auroc_mean_base"] == pytest.approx(np.mean(scores["base"]))
    assert s["rf_samples_base"] == 196 and s["rf_samples_depth6"] == 820
    assert isinstance(s["params_base"], int)
    assert s["n_seeds_compared_no_bn"] == len(seeds)
    # depth6_no_residual is paired with depth6, not with the baseline
    d = cx.dx.paired_diff(runs[("depth6_no_residual", 42)],
                          runs[("depth6", 42)], "val", 50)
    assert s["diff_s42_depth6_no_residual"] == pytest.approx(d["diff"])


def test_tables_use_only_defined_macros(pipeline):
    """Every \\cnn... macro in the generated tables is defined by
    make_numbers.py from the summary log."""
    paths, _, _, summary = pipeline
    mk = cx.dx.make_numbers_module()
    lines, _ = mk.build({"cnn": summary["run_id"]}, logs_dir=paths.logs,
                        allow_fast=True)
    defined = set(re.findall(r"\\newcommand\{\\(\w+)\}", "\n".join(lines)))
    for name in ("cnn_ablations_fast.tex", "cnn_final_fast.tex"):
        text = (paths.tables / name).read_text(encoding="utf-8")
        used = set(re.findall(r"\\(cnn\w+)\{\}", text))
        assert used and used <= defined, used - defined


def test_figures_written(pipeline):
    paths = pipeline[0]
    for name in ("cnn_curves_fast", "cnn_ablations_fast",
                 "cnn_first_layer_fast", "cnn_feature_maps_fast"):
        for ext in ("pdf", "png"):
            assert (paths.figures / f"{name}.{ext}").exists(), name


def test_fast_summary_is_not_registered_nor_paired(pipeline):
    paths, _, _, summary = pipeline
    runs = json.loads(paths.runs_file.read_text(encoding="utf-8"))
    assert runs == {"trivial": "trivial_full"}
    assert summary["trivial_run"] == "trivial_full"
    assert not any(k.startswith(("diff_trivial", "diff_dense"))
                   for k in summary["scalars"])
