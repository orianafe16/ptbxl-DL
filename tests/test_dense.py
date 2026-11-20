"""Tests of the Dense block (T9): model, ablation grid, selection rule and
the whole pipeline (runs -> final -> summary, tables, figures) on a small
synthetic cache, on the CPU.

Run from the repository root:  python -m pytest -q tests
"""
import json
import re

import numpy as np
import pytest
import torch


from src import dense_experiments as dx
from src.models import MODELS
from src.models.dense import DenseNet


# ------------------------------------------------------------------ model
def test_registered():
    assert MODELS["dense"] is DenseNet


@pytest.mark.parametrize("ab", dx.ABLATIONS, ids=lambda a: a.tag)
def test_output_shape_every_configuration(ab):
    """(batch, 12, time) in, (batch, 5) logits out, for every ablation."""
    model = DenseNet(n_samples=50, **ab.model_args)
    x = torch.randn(8, 12, 50)
    assert model(x).shape == (8, 5)


def test_parameter_count_of_the_baseline():
    """12,000 inputs -> 256 -> 256 -> 5, with batch norm (2 parameters per
    unit: scale and shift; running mean/var are buffers, not parameters)."""
    model = DenseNet(**dx.BASE_ARGS)
    expected = (12000 * 256 + 256) + 2 * 256 \
        + (256 * 256 + 256) + 2 * 256 + (256 * 5 + 5)
    assert sum(p.numel() for p in model.parameters()) == expected


def test_logits_not_probabilities():
    """The last layer is Linear: no sigmoid inside the model (D2)."""
    model = DenseNet(n_samples=50)
    assert isinstance(model.net[-1], torch.nn.Linear)
    torch.manual_seed(0)
    out = model.eval()(torch.randn(64, 12, 50) * 10)
    assert (out < 0).any() or (out > 1).any()


def test_components_switched_on_and_off():
    names = lambda m: [type(layer).__name__ for layer in m.net]   # noqa: E731
    full = names(DenseNet(n_samples=50, hidden_sizes=[16, 16]))
    assert full == ["Flatten", "Linear", "BatchNorm1d", "ReLU", "Dropout",
                    "Linear", "BatchNorm1d", "ReLU", "Dropout", "Linear"]
    bare = names(DenseNet(n_samples=50, hidden_sizes=[16], batch_norm=False,
                          dropout=0.0))
    assert bare == ["Flatten", "Linear", "ReLU", "Linear"]


def test_dropout_only_in_training_mode():
    """model.train(): two passes differ (different units dropped);
    model.eval(): dropout off and batch norm uses its running statistics,
    so two passes are identical."""
    torch.manual_seed(0)
    model = DenseNet(n_samples=50, hidden_sizes=[64], dropout=0.5)
    x = torch.randn(16, 12, 50)
    model.train()
    assert not torch.equal(model(x), model(x))
    model.eval()
    assert torch.equal(model(x), model(x))


def test_first_layer_sees_the_whole_record():
    model = DenseNet(n_samples=50, hidden_sizes=[32])
    assert model.first_layer.weight.shape == (32, 12 * 50)


def test_invalid_arguments():
    with pytest.raises(ValueError):
        DenseNet(hidden_sizes=[])
    with pytest.raises(ValueError):
        DenseNet(dropout=1.0)


# ------------------------------------------------------------------ grid
def _differences(ab):
    diff = {k for k, v in ab.model_args.items() if dx.BASE_ARGS[k] != v}
    if not ab.normalize:
        diff.add("normalize")
    return diff


def test_ablation_grid_one_factor_at_a_time():
    tags = [a.tag for a in dx.ABLATIONS]
    assert len(tags) == len(set(tags)) == 11
    assert tags[0] == "base" and _differences(dx.ABLATIONS[0]) == set()
    for ab in dx.ABLATIONS[1:]:
        assert re.fullmatch(r"[a-z0-9_]+", ab.tag)
        expected = 2 if ab.tag == "no_norm_no_bn" else 1
        assert len(_differences(ab)) == expected, ab.tag


def test_configs_keep_the_d14_defaults():
    for ab in dx.ABLATIONS:
        cfg = dx.make_config(ab, 42)
        assert (cfg.lr, cfg.batch_size, cfg.max_epochs, cfg.patience,
                cfg.weight_decay) == (1e-3, 64, 50, 5, 0.0)
        assert cfg.model == "dense" and cfg.tag == ab.tag
    fast = dx.make_config(dx.ABLATIONS[0], 42, fast=True)
    assert (fast.max_epochs, fast.n_boot) == (2, 50)


# ------------------------------------------------------------- selection
def _fake_runs(means: dict, seeds=(42, 43, 44)):
    runs = {}
    for tag, m in means.items():
        for i, s in enumerate(seeds):
            runs[(tag, s)] = {"scalars": {"val_macro_auroc": m + 0.001 * i}}
    return runs


def test_selection_takes_the_best_mean():
    means = {a.tag: 0.80 for a in dx.ABLATIONS}
    means["width64"] = 0.85
    assert dx.select_final(_fake_runs(means)) == "width64"


def test_selection_refuses_incomplete_grids():
    means = {a.tag: 0.80 for a in dx.ABLATIONS}
    runs = _fake_runs(means)
    del runs[("depth4", 44)]
    with pytest.raises(RuntimeError, match="depth4"):
        dx.select_final(runs)


# ------------------------------------------------------ whole pipeline
def synthetic_cache(n_per_fold: int = 30, n_samples: int = 1000, seed: int = 0):
    """10 folds of random signals; each class adds a small offset to one
    lead, so the labels can be learnt a little. One patient per record."""
    rng = np.random.default_rng(seed)
    n = 10 * n_per_fold
    y = (rng.random((n, 5)) < 0.35).astype(np.float32)
    y[::7] = 0
    y[::7, 0] = 1
    X = rng.normal(size=(n, 12, n_samples)).astype(np.float32)
    for c in range(5):
        X[:, c, :] += 0.5 * y[:, c:c + 1]
    return {"X": X, "y": y, "ecg_id": np.arange(1, n + 1),
            "patient_id": np.arange(1000, 1000 + n),
            "fold": np.repeat(np.arange(1, 11), n_per_fold),
            "has_superclass": np.ones(n, dtype=bool)}


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    root = tmp_path_factory.mktemp("dense")
    paths = dx.Paths(logs=root / "logs", predictions=root / "pred",
                     checkpoints=root / "ckpt", figures=root / "fig",
                     tables=root / "tab", runs_file=root / "runs.json")
    paths.runs_file.write_text("{}", encoding="utf-8")
    seeds = (42, 43)
    cache = synthetic_cache()
    dx.run_ablations(seeds, None, fast=True, cache=cache, paths=paths)
    selected = dx.run_final(fast=True, cache=cache, paths=paths, seeds=seeds)
    summary = dx.write_report(fast=True, paths=paths, seeds=seeds)
    return paths, seeds, selected, summary


def test_pipeline_runs_and_logs(pipeline):
    paths, seeds, selected, _ = pipeline
    runs = dx.find_runs(paths, fast=True, final=False)
    assert set(runs) == {(a.tag, s) for a in dx.ABLATIONS for s in seeds}
    finals = dx.find_runs(paths, fast=True, final=True)
    assert set(finals) == {(selected, s) for s in seeds}
    for rec in runs.values():                  # ablations never see test
        assert "test_macro_auroc" not in rec["scalars"]
    for rec in finals.values():
        assert "test_macro_auroc" in rec["scalars"]
    # the logged normalisation follows the configuration
    assert runs[("no_norm", 42)]["data"]["normalize"] is False
    assert runs[("base", 42)]["data"]["normalize"] is True


def test_summary_selection_matches_the_rule(pipeline):
    paths, seeds, selected, summary = pipeline
    runs = dx.find_runs(paths, fast=True, final=False)
    scores = dx.val_scores(runs, seeds)
    assert selected == max(scores, key=lambda t: np.mean(scores[t]))
    assert summary["selected_config"] == selected
    s = summary["scalars"]
    assert s["val_auroc_mean_base"] == pytest.approx(np.mean(scores["base"]))
    assert isinstance(s["params_base"], int)
    assert s["n_seeds_compared_no_bn"] == len(seeds)


def test_tables_use_only_defined_macros(pipeline):
    """Every \\dense... macro in the generated tables is defined by
    make_numbers.py from the summary log."""
    paths, _, _, summary = pipeline
    mk = dx.make_numbers_module()
    lines, _ = mk.build({"dense": summary["run_id"]}, logs_dir=paths.logs,
                        allow_fast=True)
    defined = set(re.findall(r"\\newcommand\{\\(\w+)\}", "\n".join(lines)))
    for name in ("dense_ablations_fast.tex", "dense_final_fast.tex"):
        text = (paths.tables / name).read_text(encoding="utf-8")
        used = set(re.findall(r"\\(dense\w+)\{\}", text))
        assert used and used <= defined, used - defined


def test_figures_written(pipeline):
    paths = pipeline[0]
    for name in ("dense_curves_fast", "dense_ablations_fast",
                 "dense_first_layer_fast"):
        for ext in ("pdf", "png"):
            assert (paths.figures / f"{name}.{ext}").exists()


def test_fast_summary_is_not_registered(pipeline):
    paths = pipeline[0]
    assert json.loads(paths.runs_file.read_text(encoding="utf-8")) == {}
