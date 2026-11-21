"""Tests of the RNN block (T11): model (framing, LSTM/GRU states, readouts,
attention), interpretation helpers, ablation grid, selection rules and the
whole pipeline (estimate -> runs -> finals -> summary, tables, figures) on a
small synthetic cache, on the CPU.

Run from the repository root:  python -m pytest -q tests
"""
import json
import math
import re

import numpy as np
import pytest
import torch
from torch import nn

from src import config, runlog
from src import rnn_experiments as rx
from src.models import MODELS
from src.models.rnn import AttentionPooling, RNNClassifier
from tests.test_dense import synthetic_cache


# ------------------------------------------------------------------ model
def test_registered():
    assert MODELS["rnn"] is RNNClassifier


@pytest.mark.parametrize("ab", rx.ABLATIONS, ids=lambda a: a.tag)
def test_output_shape_every_configuration(ab):
    """(batch, 12, 1000) in, (batch, 5) logits out, for every ablation."""
    model = RNNClassifier(**ab.model_args)
    assert model(torch.randn(2, 12, 1000)).shape == (2, 5)


def _lstm_params(d_in, h):
    """4 gates, each with input weights (h x d_in), recurrent weights
    (h x h) and two bias vectors (PyTorch keeps b_ih and b_hh)."""
    return 4 * (h * d_in + h * h + 2 * h)


def test_parameter_count_of_the_baseline():
    """Frames of 10 samples x 12 leads = 120 inputs; 2 layers of 128;
    head 128 -> 5."""
    expected = _lstm_params(120, 128) + _lstm_params(128, 128) + 128 * 5 + 5
    model = RNNClassifier(**rx.BASE_ARGS)
    assert sum(p.numel() for p in model.parameters()) == expected == 260741


def test_gru_has_three_gates():
    model = RNNClassifier(cell="gru", num_layers=1)
    rnn = sum(p.numel() for p in model.rnn.parameters())
    assert rnn == 3 * (128 * 120 + 128 * 128 + 2 * 128)


def test_logits_not_probabilities():
    """The last layer is Linear: no sigmoid inside the model (D2)."""
    model = RNNClassifier(n_samples=100)
    assert isinstance(model.classifier, nn.Linear)
    torch.manual_seed(0)
    out = model.eval()(torch.randn(16, 12, 100) * 10)
    assert (out < 0).any() or (out > 1).any()


def test_frames_are_consecutive_samples_lead_by_lead():
    """Step t of the sequence holds samples 10t .. 10t+9 of lead 0, then of
    lead 1, ...; frame=1 is just the permuted signal (batch, time, leads)."""
    x = torch.arange(2 * 12 * 1000, dtype=torch.float32).reshape(2, 12, 1000)
    seq = RNNClassifier(frame=10).sequence(x)
    assert seq.shape == (2, 100, 120)
    assert torch.equal(seq[1, 7, :10], x[1, 0, 70:80])
    assert torch.equal(seq[1, 7, 10:20], x[1, 1, 70:80])
    assert torch.equal(seq[0, 99, -10:], x[0, 11, 990:1000])
    raw = RNNClassifier(frame=1).sequence(x)
    assert torch.equal(raw, x.permute(0, 2, 1))


def test_conv_front_end():
    model = RNNClassifier(front_end="conv")
    assert model.sequence(torch.randn(2, 12, 1000)).shape == (2, 125, 128)
    assert model.n_steps == 125 and model.step_samples == 8


@pytest.mark.parametrize("layers", [1, 2])
def test_last_state_unidirectional(layers):
    """h_n[-1] (top layer) equals out[:, -1, :] for a unidirectional RNN,
    with one or several layers (weekly summary 13/11)."""
    torch.manual_seed(0)
    model = RNNClassifier(n_samples=100, num_layers=layers).eval()
    out, last = model.encode(torch.randn(3, 12, 100))
    assert torch.allclose(last, out[:, -1, :])


@pytest.mark.parametrize("cell", ["lstm", "gru"])
def test_last_state_bidirectional(cell):
    """Bidirectional: the forward final state is at out[:, -1, :H], the
    backward one at out[:, 0, H:]; out[:, -1, :] would pair the forward
    state with a backward state that has read a single step."""
    torch.manual_seed(0)
    model = RNNClassifier(n_samples=100, cell=cell,
                          bidirectional=True).eval()
    out, last = model.encode(torch.randn(3, 12, 100))
    H = model.hidden_size
    assert last.shape == (3, 2 * H)
    assert torch.allclose(last[:, :H], out[:, -1, :H])
    assert torch.allclose(last[:, H:], out[:, 0, H:])
    assert not torch.allclose(last, out[:, -1, :])


def test_unidirectional_states_do_not_see_the_future():
    """out[:, t] depends only on steps <= t: changing the second half of the
    record leaves the first half of the states unchanged, unless the
    network is bidirectional."""
    torch.manual_seed(0)
    x = torch.randn(2, 12, 100)
    y = x.clone()
    y[:, :, 50:] += 1.0
    for bi, same in ((False, True), (True, False)):
        model = RNNClassifier(n_samples=100, bidirectional=bi).eval()
        a, _ = model.encode(x)
        b, _ = model.encode(y)
        assert torch.allclose(a[:, :5], b[:, :5]) is same


def test_mean_readout():
    torch.manual_seed(0)
    model = RNNClassifier(n_samples=100, readout="mean").eval()
    x = torch.randn(2, 12, 100)
    out, _ = model.encode(x)
    z, alpha = model.summarize(x)
    assert alpha is None and torch.allclose(z, out.mean(dim=1))


@pytest.mark.parametrize("kind", ["additive", "notebook"])
def test_attention_is_a_weighted_mean(kind):
    torch.manual_seed(0)
    model = RNNClassifier(n_samples=100, readout="attention",
                          attention=kind).eval()
    x = torch.randn(4, 12, 100)
    out, _ = model.encode(x)
    z, alpha = model.summarize(x)
    assert alpha.shape == (4, 10)
    assert (alpha >= 0).all()
    assert torch.allclose(alpha.sum(dim=1), torch.ones(4))
    assert torch.allclose(z, (alpha.unsqueeze(-1) * out).sum(dim=1), atol=1e-6)
    assert torch.allclose(model.attention_weights(x), alpha)


def test_notebook_scores_are_bounded():
    """tanh keeps every score in [-1, 1]: whatever the hidden states, no
    weight exceeds 1 / (1 + (T - 1) e^-2); the additive score can."""
    T = 100
    bound = AttentionPooling.max_weight_bound(T)
    assert bound == pytest.approx(0.0695, abs=1e-4)
    h = torch.zeros(1, T, 8)
    h[0, 3] = 100.0                    # one very different state
    nb = AttentionPooling(8, kind="notebook")
    nn.init.constant_(nb.score_layer.weight, 1.0)
    nn.init.constant_(nb.score_layer.bias, -10.0)   # other scores -> -1
    with torch.no_grad():
        _, a = nb(h)
    assert float(a.max()) <= bound + 1e-6
    assert float(a.max()) == pytest.approx(bound, rel=1e-3)
    add = AttentionPooling(8, dim=4, kind="additive")
    nn.init.constant_(add.proj.weight, 1.0)
    nn.init.constant_(add.v.weight, 10.0)
    with torch.no_grad():
        _, a = add(h)
    assert float(a.max()) > 0.9


def test_attention_weights_need_attention():
    with pytest.raises(ValueError):
        RNNClassifier(n_samples=100).attention_weights(torch.randn(1, 12, 100))


def test_dropout_only_in_training_mode():
    torch.manual_seed(0)
    model = RNNClassifier(n_samples=100)
    x = torch.randn(8, 12, 100)
    model.train()
    assert not torch.equal(model(x), model(x))
    model.eval()
    assert torch.equal(model(x), model(x))


def test_single_lead_input():
    """Chat 08 trains the best model of each family on lead II only."""
    model = RNNClassifier(n_leads=1)
    assert model(torch.randn(2, 1, 1000)).shape == (2, 5)


def test_invalid_arguments():
    for bad in ({"cell": "rnn"}, {"front_end": "fft"}, {"readout": "max"},
                {"attention": "dot"}, {"frame": 7}, {"dropout": 1.0},
                {"hidden_size": 0}, {"conv_channels": [], "front_end": "conv"}):
        with pytest.raises(ValueError):
            RNNClassifier(**bad)


# ------------------------------------------------------- interpretation
class _LastSamples(nn.Module):
    """Toy model whose 5 logits are the sum of the last 100 samples of
    lead 0: its sensitivity is uniform there and zero elsewhere."""

    def __init__(self):
        super().__init__()
        self.classifier = nn.Linear(1, 5)

    def forward(self, x):
        return x[:, 0, -100:].sum(dim=1, keepdim=True).repeat(1, 5)


def test_input_sensitivity_toy_model():
    X = np.random.default_rng(0).normal(size=(10, 12, 1000)).astype(np.float32)
    p = rx.input_sensitivity(_LastSamples(), X, batch=4)
    assert p.shape == (1000,) and p.sum() == pytest.approx(1.0)
    assert np.allclose(p[-100:], 0.01) and np.all(p[:-100] == 0)
    s = rx.sensitivity_numbers(p, "toy")
    assert s["sens_last_toy"] == pytest.approx(1.0)
    assert s["sens_first_toy"] == 0.0
    assert s["sens_half_window_s_toy"] == pytest.approx(0.5)


def test_input_sensitivity_of_an_rnn():
    torch.manual_seed(0)
    X = np.random.default_rng(0).normal(size=(4, 12, 100)).astype(np.float32)
    p = rx.input_sensitivity(RNNClassifier(n_samples=100), X)
    assert p.shape == (100,) and p.sum() == pytest.approx(1.0)
    assert (p >= 0).all()


def test_sensitivity_numbers_uniform():
    s = rx.sensitivity_numbers(np.full(1000, 1e-3), "u")
    assert s["sens_first_u"] == pytest.approx(0.2)
    assert s["sens_last_u"] == pytest.approx(0.2)
    assert s["sens_half_window_s_u"] == pytest.approx(5.0)
    assert len(rx.bin_profile(np.full(1000, 1e-3))) == 100
    assert sum(rx.bin_profile(np.full(1000, 1e-3))) == pytest.approx(1.0)


def test_step_activity_and_spearman():
    X = np.zeros((1, 12, 100), np.float32)
    X[0, :, 55] = 1.0                  # a spike inside step 5 (samples 50-59)
    a = rx.step_activity(X, 10)
    assert a.shape == (1, 10) and a[0].argmax() == 5
    r = rx._row_spearman(np.array([[1., 2, 3, 4]]), np.array([[10., 20, 30, 40]]))
    assert r[0] == pytest.approx(1.0)


def test_attention_numbers_of_uniform_weights():
    """Zero scores give uniform weights: normalised entropy 1, effective
    steps T, largest weight 1/T."""
    model = RNNClassifier(n_samples=100, readout="attention").eval()
    nn.init.zeros_(model.attention.v.weight)
    X = np.random.default_rng(0).normal(size=(6, 12, 100)).astype(np.float32)
    s = rx.attention_numbers(model, X, "u")
    assert s["attn_steps_u"] == 10
    assert s["attn_entropy_norm_u"] == pytest.approx(1.0)
    assert s["attn_effective_steps_u"] == pytest.approx(10.0)
    assert s["attn_max_weight_u"] == pytest.approx(0.1)
    nb = RNNClassifier(n_samples=100, readout="attention",
                       attention="notebook").eval()
    s = rx.attention_numbers(nb, X, "n")
    assert s["attn_max_bound_n"] == pytest.approx(
        AttentionPooling.max_weight_bound(10))
    assert s["attn_max_weight_n"] <= s["attn_max_bound_n"] + 1e-9


# ------------------------------------------------------------------ grid
def _differences(ab, other):
    a, b = ab.model_args, other.model_args
    return {k for k in a if a[k] != b[k]}


def test_ablation_grid_one_factor_at_a_time():
    tags = [a.tag for a in rx.ABLATIONS]
    assert len(tags) == len(set(tags)) == 12
    assert tags[0] == "base" and tags[-2:] == ["raw", "raw_attn"]
    for ab in rx.ABLATIONS[1:]:
        assert re.fullmatch(r"[a-z0-9_]+", ab.tag)
        assert ab.reference in rx.BY_TAG and ab.reference != ab.tag
        assert len(_differences(ab, rx.BY_TAG[ab.reference])) == 1, ab.tag
    # every "x attention" configuration adds attention to its reference
    for tag in ("gru_attn", "bi_attn", "raw_attn"):
        assert _differences(rx.BY_TAG[tag], rx.BY_TAG[rx.BY_TAG[tag].reference]) \
            == {"readout"}
    assert _differences(rx.BY_TAG["attn_tanh"], rx.BY_TAG["attn"]) == {"attention"}
    assert rx.ATTENTION_TAGS == ["attn", "attn_tanh", "gru_attn", "bi_attn",
                                 "raw_attn"]


def test_configs_keep_the_d14_defaults_plus_clipping():
    for ab in rx.ABLATIONS:
        cfg = rx.make_config(ab, 42)
        assert (cfg.lr, cfg.batch_size, cfg.max_epochs, cfg.patience,
                cfg.weight_decay, cfg.normalize, cfg.pos_weight) == \
            (1e-3, 64, 50, 5, 0.0, True, False)
        assert cfg.grad_clip == 1.0
        assert cfg.model == "rnn" and cfg.tag == ab.tag
    fast = rx.make_config(rx.ABLATIONS[0], 42, fast=True)
    assert (fast.max_epochs, fast.n_boot) == (2, 50)


# ------------------------------------------------------------- selection
def _fake_runs(means: dict, seeds=(42, 43, 44)):
    return {(tag, s): {"scalars": {"val_macro_auroc": m + 0.001 * i}}
            for tag, m in means.items() for i, s in enumerate(seeds)}


def test_two_selections():
    means = {a.tag: 0.80 for a in rx.ABLATIONS}
    means["conv"], means["bi_attn"] = 0.90, 0.85
    assert rx.select_finals(_fake_runs(means)) == ("conv", "bi_attn")
    means["raw_attn"] = 0.95
    assert rx.select_finals(_fake_runs(means)) == ("raw_attn", "raw_attn")


def test_selection_refuses_incomplete_grids():
    runs = _fake_runs({a.tag: 0.80 for a in rx.ABLATIONS})
    del runs[("raw", 44)]
    with pytest.raises(RuntimeError, match="raw"):
        rx.select_final(runs)


def test_final_table_cnn_and_dense_macros_exist():
    """The CNN and Dense columns of rnn_final.tex use macros that
    make_numbers.py defines from the committed summaries (skipped if they
    are missing)."""
    for name in ("cnn_summary", "dense_summary"):
        if not (config.LOGS_DIR / f"{name}.json").exists():
            pytest.skip(f"results/logs/{name}.json not present")
    s = {"final_test_macro_auroc_mean": 0.9, "attn_final_is_final": 1}
    table = rx.final_table({"run_id": "x", "scalars": s, "seeds": [42],
                            "cnn_summary": "cnn_summary",
                            "dense_summary": "dense_summary"})
    mk = rx.dx.make_numbers_module()
    lines, _ = mk.build({"cnn": "cnn_summary", "dense": "dense_summary"})
    defined = set(re.findall(r"\\newcommand\{\\(\w+)\}", "\n".join(lines)))
    used = set(re.findall(r"\\((?:cnn|dense)\w+)\{\}", table))
    assert used and used <= defined, used - defined


# ------------------------------------------------------ whole pipeline
@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    root = tmp_path_factory.mktemp("rnn")
    paths = rx.Paths(logs=root / "logs", predictions=root / "pred",
                     checkpoints=root / "ckpt", figures=root / "fig",
                     tables=root / "tab", runs_file=root / "runs.json")
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
    estimate = rx.estimate(cache, n_steps=1, seeds=seeds, fast=True)
    rx.run_ablations(seeds, None, fast=True, cache=cache, paths=paths)
    selected = rx.run_final(fast=True, cache=cache, paths=paths, seeds=seeds)
    summary = rx.write_report(fast=True, paths=paths, seeds=seeds,
                              cache=cache)
    return paths, seeds, selected, summary, estimate


def test_estimate_covers_every_configuration(pipeline):
    estimate = pipeline[4]
    assert set(estimate) == set(rx.BY_TAG)
    assert all(v > 0 for v in estimate.values())


def test_pipeline_runs_and_logs(pipeline):
    paths, seeds, selected, _, _ = pipeline
    runs = rx.find_runs(paths, fast=True, final=False)
    assert set(runs) == {(a.tag, s) for a in rx.ABLATIONS for s in seeds}
    finals = rx.find_runs(paths, fast=True, final=True)
    assert set(finals) == {(t, s) for t in selected for s in seeds}
    assert 1 <= len(selected) <= 2
    assert any(rx.BY_TAG[t].has_attention for t in selected)
    for rec in runs.values():                  # ablations never see test
        assert "test_macro_auroc" not in rec["scalars"]
        assert rec["config"]["grad_clip"] == 1.0
        assert rec["data"]["cublas_workspace_config"] == ":4096:2"
    for rec in finals.values():
        assert "test_macro_auroc" in rec["scalars"]
    assert runs[("raw", 42)]["model"]["args"]["frame"] == 1


def test_summary_selection_and_references(pipeline):
    paths, seeds, selected, summary, _ = pipeline
    runs = rx.find_runs(paths, fast=True, final=False)
    scores = rx.val_scores(runs, seeds)
    best = max(scores, key=lambda t: np.mean(scores[t]))
    best_attn = max(rx.ATTENTION_TAGS, key=lambda t: np.mean(scores[t]))
    assert summary["selected_config"] == best == selected[0]
    assert summary["selected_attention_config"] == best_attn
    s = summary["scalars"]
    assert s["attn_final_is_final"] == int(best == best_attn)
    assert "final_test_macro_auroc_mean" in s
    assert "attnfinal_test_macro_auroc_mean" in s
    assert s["steps_base"] == 100 and s["steps_raw"] == 1000
    assert s["steps_conv"] == 125
    assert isinstance(s["params_base"], int)
    assert s["n_seeds_compared_gru"] == len(seeds)
    # attn_tanh is paired with attn, raw_attn with raw
    for tag, ref in (("attn_tanh", "attn"), ("raw_attn", "raw")):
        d = rx.dx.paired_diff(runs[(tag, 42)], runs[(ref, 42)], "val", 50)
        assert s[f"diff_s42_{tag}"] == pytest.approx(d["diff"])
    # --fast: no comparison with other blocks on (a subset of) test
    assert not any(k.startswith(("diff_trivial", "diff_cnn", "diff_attn_"))
                   for k in s)


def test_summary_interpretation(pipeline):
    s, prof = pipeline[3]["scalars"], pipeline[3]["sensitivity_profiles"]
    for ab in rx.ABLATIONS:
        t = ab.tag
        assert 0 <= s[f"sens_last_{t}"] <= 1
        assert sum(prof["profiles"][t]) == pytest.approx(1.0)
        assert len(prof["profiles"][t]) == 100
        if ab.has_attention:
            assert 0 < s[f"attn_entropy_norm_{t}"] <= 1 + 1e-9
            assert s[f"attn_steps_{t}"] == RNNClassifier(**ab.model_args).n_steps
    assert s["attn_max_weight_attn_tanh"] <= s["attn_max_bound_attn_tanh"]


def test_tables_use_only_defined_macros(pipeline):
    """Every \\rnn... macro in the generated tables is defined by
    make_numbers.py from the summary log."""
    paths, _, _, summary, _ = pipeline
    mk = rx.dx.make_numbers_module()
    lines, _ = mk.build({"rnn": summary["run_id"]}, logs_dir=paths.logs,
                        allow_fast=True)
    defined = set(re.findall(r"\\newcommand\{\\(\w+)\}", "\n".join(lines)))
    for name in ("rnn_ablations_fast.tex", "rnn_attention_fast.tex",
                 "rnn_final_fast.tex"):
        text = (paths.tables / name).read_text(encoding="utf-8")
        used = set(re.findall(r"\\(rnn\w+)\{\}", text))
        assert used and used <= defined, (name, used - defined)


def test_figures_written(pipeline):
    paths = pipeline[0]
    for name in ("rnn_curves_fast", "rnn_ablations_fast",
                 "rnn_time_gradient_fast", "rnn_attention_fast"):
        for ext in ("pdf", "png"):
            assert (paths.figures / f"{name}.{ext}").exists(), name


def test_fast_summary_is_not_registered(pipeline):
    paths = pipeline[0]
    runs = json.loads(paths.runs_file.read_text(encoding="utf-8"))
    assert runs == {"trivial": "trivial_full"}


def test_summary_macro_names_are_unique(pipeline):
    """make_numbers.py stops on two keys with the same macro name; the
    summary must never trigger it."""
    s = pipeline[3]["scalars"]
    mk = rx.dx.make_numbers_module()
    names = [mk.macro_name("rnn", k) for k in s]
    assert len(names) == len(set(names))
    assert not any(isinstance(v, float) and math.isnan(v) and
                   not k.endswith("_sd") for k, v in s.items())
