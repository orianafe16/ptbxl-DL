"""Tests of the generative block (T23): tokenizer, windows, the ECG-GPT and
LSTM generators (causality, sizes, features for chat DL 08), the task and
its metrics, counting baselines, sampling and generation, statistics of
generated signals, the ablation grid and selection rule, and the whole
pipeline (estimate -> runs -> summary, tables, figures) on a small
synthetic ECG-like cache, on the CPU.

Run from the repository root:  python -m pytest -q tests
"""
import json
import math
import re

import numpy as np
import pytest
import torch

from src import config
from src import generative as gx
from src import generative_experiments as ge
from src.models.generative import (ECGGPT, ECGLSTM, build_generator,
                                   window_starts)


# ------------------------------------------------------------ synthetic data
def synthetic_ecg_cache(n_per_fold: int = 12, n_samples: int = 600,
                        seed: int = 0) -> dict:
    """10 folds of 12-lead records whose lead II is a train of beats (a
    narrow positive peak every 60-100 samples, a small T wave, noise and a
    slow drift), on the microvolt grid like PTB-XL. One patient per record;
    every 6th record has no superclass."""
    rng = np.random.default_rng(seed)
    n = 10 * n_per_fold
    t = np.arange(n_samples)
    X = rng.normal(scale=0.05, size=(n, 12, n_samples)).astype(np.float32)
    lead2 = config.LEADS.index("II")
    for i in range(n):
        rr = int(rng.integers(60, 101))
        x = 0.02 * rng.normal(size=n_samples) + 0.1 * np.sin(
            2 * np.pi * t / n_samples)
        for p in range(int(rng.integers(0, rr)), n_samples, rr):
            x += 1.0 * np.exp(-0.5 * ((t - p) / 1.5) ** 2)
            x += 0.2 * np.exp(-0.5 * ((t - p - 25) / 6.0) ** 2)
        X[i, lead2] = np.round(x * 1000) / 1000
    y = (rng.random((n, 5)) < 0.35).astype(np.float32)
    y[::7] = 0
    y[::7, 0] = 1
    y[::6] = 0                         # records without superclass
    return {"X": X, "y": y, "ecg_id": np.arange(1, n + 1),
            "patient_id": np.arange(1000, 1000 + n),
            "fold": np.repeat(np.arange(1, 11), n_per_fold),
            "has_superclass": y.sum(axis=1) > 0}


# ---------------------------------------------------------------- tokenizer
def test_tokenizer_on_the_microvolt_grid():
    tok = gx.make_tokenizer(-100, 64, span_uv=640)          # delta 10 uV
    assert tok.delta_uv == 10 and tok.span_uv == 640
    x_uv = np.array([-100, -91, -90, 0, 539, 540, 5000, -5000])
    k = tok.encode(x_uv / 1000.0)
    assert k.tolist() == [0, 0, 1, 10, 63, 63, 63, 0]       # last two clipped
    assert k.dtype == np.int64
    # decode: middle of the 10 integer values of the token
    assert tok.decode([0])[0] == pytest.approx(-0.0955)
    assert tok.centers_uv[1] == pytest.approx(-85.5)


def test_one_microvolt_tokens_are_lossless_inside_the_range():
    tok = gx.make_tokenizer(-1280, 2560)
    assert tok.delta_uv == 1
    u = np.arange(-1280, 1280)
    assert np.array_equal(tok.encode(u / 1000.0), np.arange(2560))
    assert np.allclose(tok.decode(tok.encode(u / 1000.0)), u / 1000.0)


def test_quantisation_error_at_most_half_a_level():
    tok = gx.make_tokenizer(-1280, 256)
    u = np.arange(-1280, 1280)
    err = np.abs(tok.decode(tok.encode(u / 1000.0)) * 1000.0 - u)
    assert err.max() <= (tok.delta_uv - 1) / 2 + 1e-4


def test_every_vocabulary_divides_the_span():
    for v in ge.VOCABS:
        tok = gx.make_tokenizer(0, v)
        assert tok.span_uv == gx.SPAN_UV
    with pytest.raises(ValueError):
        gx.make_tokenizer(0, 300)


def test_range_is_centred_on_the_training_quantiles():
    x = np.linspace(-0.5, 1.5, 100001)                       # mV
    lo = gx.range_start(x)
    q = np.quantile(x, gx.CLIP_QUANTILES) * 1000
    assert lo == round((q[0] + q[1]) / 2 - gx.SPAN_UV / 2)
    assert isinstance(lo, int)


def test_bits_per_sample_of_uniform_guessing_ignores_the_vocabulary():
    for v in ge.VOCABS:
        tok = gx.make_tokenizer(0, v)
        assert gx.bits_per_sample(math.log(v), tok.delta_uv) == \
            pytest.approx(math.log2(gx.SPAN_UV))


# ------------------------------------------------------------------ windows
@pytest.mark.parametrize("n,L", [(1000, 250), (1000, 100), (1000, 500),
                                 (600, 500), (600, 250)])
def test_validation_windows_cover_every_target(n, L):
    starts = window_starts(n, L)
    assert starts[0] == 0 and starts[-1] + L == n - 1
    assert len(starts) == math.ceil((n - 1) / L)
    covered = set()
    for s in starts:
        covered |= set(range(s + 1, s + L + 1))
    assert covered == set(range(1, n))


def test_training_windows_are_shifted_by_one_and_reproducible():
    tokens = np.arange(3 * 50, dtype=np.int16).reshape(3, 50)
    a = gx.TrainWindows(tokens, 10, seed=1)
    b = gx.TrainWindows(tokens, 10, seed=1)
    for i in range(3):
        x, y = a[i]
        x2, _ = b[i]
        assert x.dtype == torch.long and x.shape == (10,)
        assert torch.equal(y[:-1], x[1:])
        assert torch.equal(x, x2)
        assert int(y[-1]) <= tokens[i, -1]


def test_validation_windows_are_record_by_record():
    tokens = np.arange(2 * 21, dtype=np.int16).reshape(2, 21)
    ds = gx.ValWindows(tokens, 10)
    assert ds.n_windows == 2 and len(ds) == 4
    x, y = ds[2]                       # record 1, window 0
    assert int(x[0]) == 21 and int(y[-1]) == 31


def test_splits_use_all_training_records_and_never_the_test_fold():
    cache = synthetic_ecg_cache()
    idx = gx.generative_splits(cache)
    fold, sup = cache["fold"], cache["has_superclass"]
    assert set(fold[idx["train"]]) == set(config.TRAIN_FOLDS)
    assert (~sup[idx["train"]]).any()                # D12 rule c
    assert np.array_equal(idx["train"],
                          np.flatnonzero(np.isin(fold, config.TRAIN_FOLDS)))
    assert (fold[idx["val"]] == config.VAL_FOLD).all() and sup[idx["val"]].all()
    assert "test" not in idx


def test_gen_data_description():
    data = gx.GenData(synthetic_ecg_cache())
    d = data.description()
    assert d["frac_off_uv_grid_train"] == 0.0
    assert d["range_hi_uv"] - d["range_lo_uv"] + 1 == gx.SPAN_UV
    assert 0 <= d["clip_frac_high_train"] < 0.01
    assert data.tokens(256, "val").dtype == np.int16


# ------------------------------------------------------------------- models
def test_parameter_counts_are_about_equal():
    gpt = build_generator(ge.BASE_ARGS)
    lstm = build_generator(ge.LSTM_ARGS)
    n_gpt = sum(p.numel() for p in gpt.parameters())
    n_lstm = sum(p.numel() for p in lstm.parameters())
    assert n_gpt == 461568 and n_lstm == 465664
    assert abs(n_lstm / n_gpt - 1) < 0.02


@pytest.mark.parametrize("args", [ge.BASE_ARGS, ge.LSTM_ARGS,
                                  {**ge.BASE_ARGS, "pos_encoding": "none"}])
def test_generators_do_not_see_the_future(args):
    torch.manual_seed(0)
    model = build_generator({**args, "vocab": 32, "context": 20}).eval()
    x = torch.randint(0, 32, (2, 20))
    x2 = x.clone()
    x2[:, 12:] = torch.randint(0, 32, (2, 8))
    a, b = model(x), model(x2)
    assert a.shape == (2, 20, 32)
    assert torch.allclose(a[:, :12], b[:, :12], atol=1e-6)
    assert not torch.allclose(a[:, 12:], b[:, 12:])


def test_without_the_mask_the_past_sees_the_future():
    torch.manual_seed(0)
    model = ECGGPT(vocab=32, context=20, causal=False).eval()
    x = torch.randint(0, 32, (2, 20))
    x2 = x.clone()
    x2[:, 15] = (x2[:, 15] + 1) % 32
    assert not torch.allclose(model(x)[:, :15], model(x2)[:, :15])


def test_causal_attention_maps_are_lower_triangular():
    model = ECGGPT(vocab=32, context=16).eval()
    maps = model.attention_maps(torch.randint(0, 32, (2, 16)))
    assert maps.shape == (2, 2, 4, 16, 16)
    assert torch.all(maps.triu(diagonal=1) == 0)
    assert torch.allclose(maps.sum(-1), torch.ones(2, 2, 4, 16))


def test_window_longer_than_the_context_is_refused():
    with pytest.raises(ValueError):
        ECGGPT(vocab=8, context=10)(torch.zeros(1, 11, dtype=torch.long))


def test_next_logits_is_the_last_position():
    for model in (ECGGPT(vocab=16, context=12), ECGLSTM(vocab=16, context=12)):
        model.eval()
        x = torch.randint(0, 16, (3, 12))
        assert torch.allclose(model.next_logits(x), model(x)[:, -1],
                              atol=1e-6)


def test_positional_encoding_is_not_in_the_checkpoint():
    model = ECGGPT(vocab=16, context=12)
    assert "pe" not in model.state_dict()
    assert not any(n == "pe" for n, _ in model.named_parameters())


@pytest.mark.parametrize("cls", [ECGGPT, ECGLSTM])
def test_record_features_for_chat_08(cls):
    model = cls(vocab=16, context=50)
    x = torch.randint(0, 16, (3, 120))
    f = model.record_features(x)
    width = model.d_model if cls is ECGGPT else model.hidden_size
    assert f.shape == (3, width)
    f.sum().backward()                         # gradients reach the weights
    assert model.tok_emb.weight.grad is not None


def test_build_generator_rejects_unknown_arch():
    with pytest.raises(ValueError):
        build_generator({"arch": "cnn"})


# --------------------------------------------------------------------- task
def test_task_metrics_match_a_direct_computation():
    torch.manual_seed(0)
    tok = gx.make_tokenizer(0, 64)
    L, n_win, n_rec = 5, 2, 3
    task = gx.GenerativeTask(tok, L, n_win, {"val": np.array([1, 1, 2])},
                             {"val": np.array([10, 11, 12])}, n_boot=20)
    out = torch.randn(n_rec * n_win, L, 64)
    tgt = torch.randint(0, 64, (n_rec * n_win, L))
    state = task.collect(task.start(), out[:4], tgt[:4])
    state = task.collect(state, out[4:], tgt[4:])
    nll = torch.nn.functional.cross_entropy(out.reshape(-1, 64),
                                            tgt.reshape(-1), reduction="none")
    nll = nll.reshape(n_rec * n_win, L).double()
    mean = float(nll.mean())
    assert float(task.loss(out, tgt)) == pytest.approx(mean, rel=1e-5)
    m = task.epoch_metrics(mean, state)
    assert m["ce_last"] == pytest.approx(float(nll[:, -1].mean()), rel=1e-6)
    assert m["bits_per_sample"] == pytest.approx(
        (mean + math.log(tok.delta_uv)) / math.log(2))
    assert m["perplexity"] == pytest.approx(math.exp(mean))
    rec_nll, rec_n = task.record_sums(state)
    assert np.allclose(rec_nll, nll.sum(1).reshape(n_rec, n_win).sum(1).numpy())
    assert rec_n.tolist() == [n_win * L] * n_rec
    probs = torch.softmax(out, -1)
    c = torch.as_tensor(tok.centers_uv, dtype=torch.float32)
    rmse = float(((probs @ c - c[tgt]) ** 2).mean().sqrt())
    assert m["rmse_uv"] == pytest.approx(rmse, rel=1e-4)
    details, scalars, arrays = task.report({"val": (mean, state)})
    assert arrays["val_pos_ce"].shape == (L,)
    assert np.allclose(arrays["val_pos_ce"], nll.mean(0).numpy())
    lo, hi = scalars["val_bits_per_sample_lo"], scalars["val_bits_per_sample_hi"]
    assert lo <= hi
    assert details["val"]["n_targets"] == n_rec * n_win * L


def test_paired_difference_of_a_run_with_itself_is_zero():
    rng = np.random.default_rng(0)
    run = {"nll": rng.random(20) * 100, "ntok": np.full(20, 50.0),
           "delta": 10, "ecg_id": np.arange(20)}
    d = gx.paired_bits_diff(run, run, np.arange(20), n_boot=30)
    assert d["diff"] == 0 and d["lower"] == 0 and d["upper"] == 0
    other = {**run, "ecg_id": np.arange(1, 21)}
    with pytest.raises(ValueError):
        gx.paired_bits_diff(run, other, np.arange(20), n_boot=5)


def test_bits_are_comparable_across_vocabularies():
    """The same per-sample density coded with two vocabularies gives the
    same bits: a coarse token of 4 fine levels has 4 times their
    probability."""
    fine = {"nll": np.array([math.log(400.0)]), "ntok": np.array([1.0]),
            "delta": 1}
    coarse = {"nll": np.array([math.log(100.0)]), "ntok": np.array([1.0]),
              "delta": 4}
    assert gx.record_bits(fine["nll"], fine["ntok"], 1) == pytest.approx(
        gx.record_bits(coarse["nll"], coarse["ntok"], 4))


# ---------------------------------------------------------------- baselines
def test_ngram_baselines():
    tok = gx.make_tokenizer(0, 64)
    periodic = np.tile(np.arange(8), (5, 25))          # 0..7 repeated
    nb = gx.ngram_bits(periodic, periodic, tok, alpha=1e-6)
    assert nb["uniform_bits"] == pytest.approx(math.log2(gx.SPAN_UV))
    assert nb["unigram_ce"] == pytest.approx(math.log(8), rel=1e-3)
    assert nb["bigram_ce"] < 1e-3                      # fully predictable
    assert nb["bigram_bits"] < nb["unigram_bits"] < nb["uniform_bits"]


def test_persistence_and_quantisation_errors():
    x = np.array([[0.0, 0.003, 0.006]])                 # mV
    assert gx.persistence_rmse_uv(x) == pytest.approx(3.0)
    tok = gx.make_tokenizer(-1280, 2560)
    assert gx.quantisation_rmse_uv(x, tok) == pytest.approx(0.0, abs=1e-3)
    coarse = gx.make_tokenizer(-1280, 256)                   # 10 uV levels
    y = np.array([[0.0, 0.004, 5.0]])                        # 5 mV: clipped
    assert gx.quantisation_rmse_uv(y, coarse) > 1000
    assert gx.quantisation_rmse_uv(y, coarse, in_range_only=True) <= 4.5


# ------------------------------------------------------- sampling, generation
def test_greedy_and_low_temperature_pick_the_most_probable_token():
    torch.manual_seed(0)
    logits = torch.randn(50, 30)
    best = logits.argmax(-1)
    assert torch.equal(gx.sample_next(logits, 0.0), best)
    g = torch.Generator().manual_seed(0)
    assert torch.equal(gx.sample_next(logits, 1e-3, generator=g), best)


def test_top_k_never_samples_outside_the_k_best():
    torch.manual_seed(0)
    logits = torch.randn(4, 30)
    top = torch.topk(logits, 5, dim=-1).indices
    g = torch.Generator().manual_seed(0)
    for _ in range(50):
        k = gx.sample_next(logits, 1.0, top_k=5, generator=g)
        assert all(int(k[i]) in top[i].tolist() for i in range(4))


def test_temperature_one_samples_the_model_distribution():
    logits = torch.log(torch.tensor([[0.7, 0.2, 0.1]])).repeat(20000, 1)
    g = torch.Generator().manual_seed(0)
    k = gx.sample_next(logits, 1.0, generator=g)
    freq = torch.bincount(k, minlength=3).float() / len(k)
    assert torch.allclose(freq, torch.tensor([0.7, 0.2, 0.1]), atol=0.02)
    hot = gx.sample_next(logits, 3.0, generator=g)
    assert torch.bincount(hot, minlength=3)[0] / len(hot) < 0.6   # flatter


@pytest.mark.parametrize("cls", [ECGGPT, ECGLSTM])
def test_generation_slides_the_window(cls):
    torch.manual_seed(0)
    model = cls(vocab=16, context=8)
    prompt = np.random.default_rng(0).integers(0, 16, (5, 6))
    out = gx.generate(model, prompt, 20, 1.0, seed=3, batch=2)
    assert out.shape == (5, 26) and out.min() >= 0 and out.max() < 16
    assert np.array_equal(out[:, :6], prompt)
    again = gx.generate(model, prompt, 20, 1.0, seed=3, batch=2)
    assert np.array_equal(out, again)                  # seeded
    g1 = gx.generate(model, prompt, 20, 0.0, seed=1)
    g2 = gx.generate(model, prompt, 20, 0.0, seed=2)
    assert np.array_equal(g1, g2)                      # greedy: no randomness


# --------------------------------------------------------------- statistics
def test_heart_rate_of_regular_peaks():
    x = np.zeros(800)
    x[10::80] = 1.0                                    # every 0.8 s
    assert gx.heart_rate(x, 0.5) == pytest.approx(75.0)
    assert math.isnan(gx.heart_rate(np.zeros(800), 0.5))


def test_generation_stats_of_perfect_and_flat_generators():
    cache = synthetic_ecg_cache()
    x = gx.lead_signals(cache)[:20]
    prompt, real = x[:, :200], x[:, 200:]
    same = gx.generation_stats(prompt, real, real)
    assert same["std_ratio"] == pytest.approx(1.0)
    assert same["w1_mv"] == pytest.approx(0.0, abs=1e-9)
    assert same["frac_flat"] == 0 and same["hr_mae"] == pytest.approx(0.0)
    assert same["hr_defined_frac"] == 1.0
    flat = gx.generation_stats(prompt, real, np.zeros_like(real))
    assert flat["frac_flat"] == 1.0 and flat["std_ratio"] == 0.0
    assert flat["hr_defined_frac"] == 0.0 and math.isnan(flat["hr_mae"])


def test_synthetic_heart_rate_is_found():
    cache = synthetic_ecg_cache()
    x = gx.lead_signals(cache)[:10]
    ref = gx.real_reference_stats(x[:, :200], x[:, 200:], x[:, 200:])
    assert ref["real_hr_defined_frac"] == 1.0
    assert 55 <= ref["real_hr_mean"] <= 105            # RR of 60-100 samples
    assert ref["real_w1_other_mv"] == pytest.approx(0.0, abs=1e-9)


# ----------------------------------------------------- ablations, selection
def test_ablation_grid_one_factor_at_a_time():
    for ab in ge.ABLATIONS:
        start = ge.BASE_ARGS if ab.base == "gpt" else ge.LSTM_ARGS
        changed = {k for k in ab.model_args if ab.model_args[k] != start.get(k)}
        assert len(changed) <= 1, ab.tag
        assert ab.reference in ge.BY_TAG
    assert not ge.BY_TAG["no_mask"].selectable
    assert ge.BY_TAG["lstm"].model_args["vocab"] == ge.BASE_ARGS["vocab"]
    assert ge.BY_TAG["lstm"].context == ge.BY_TAG["base"].context
    assert len({a.tag for a in ge.ABLATIONS}) == len(ge.ABLATIONS)


def test_configs_keep_the_d14_defaults_plus_clipping():
    cfg = ge.make_config(ge.BY_TAG["base"], 43)
    assert (cfg.lr, cfg.batch_size, cfg.max_epochs, cfg.patience,
            cfg.weight_decay) == (1e-3, 64, 50, 5, 0.0)
    assert cfg.grad_clip == 1.0 and not cfg.final and cfg.seed == 43
    fast = ge.make_config(ge.BY_TAG["base"], 42, fast=True)
    assert fast.max_epochs == 2 and fast.n_boot == 50


def _fake_runs(values: dict, seeds=(42, 43)) -> dict:
    return {(t, s): {"scalars": {"val_bits_per_sample": v + 0.01 * i}}
            for t, v in values.items() for i, s in enumerate(seeds)}


def test_selection_ignores_no_mask_and_takes_the_fewest_bits():
    values = {a.tag: 6.0 for a in ge.ABLATIONS}
    values["no_mask"] = 0.1                            # "cheating" loss
    values["ctx500"] = 5.0
    assert ge.select_pretrained(_fake_runs(values), (42, 43)) == "ctx500"
    assert ge.comparison_model("ctx500") == "lstm"
    assert ge.comparison_model("lstm") == "base"


def test_selection_refuses_incomplete_grids():
    values = {a.tag: 6.0 for a in ge.ABLATIONS if a.tag != "lstm"}
    with pytest.raises(RuntimeError):
        ge.select_pretrained(_fake_runs(values), (42, 43))


# ------------------------------------------------------------ whole pipeline
@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    root = tmp_path_factory.mktemp("generative")
    paths = ge.Paths(logs=root / "logs", predictions=root / "pred",
                     checkpoints=root / "ckpt", figures=root / "fig",
                     tables=root / "tab", runs_file=root / "runs.json")
    paths.runs_file.write_text('{"trivial": "trivial_full"}', encoding="utf-8")
    seeds = (42, 43)
    cache = synthetic_ecg_cache()
    data = gx.GenData(cache, fast=True)
    estimate = ge.estimate(n_steps=1, seeds=seeds, fast=True, data=data)
    ge.run_ablations(seeds, None, fast=True, paths=paths, data=data)
    summary = ge.write_report(fast=True, paths=paths, seeds=seeds,
                              cache=cache, data=data)
    return paths, seeds, summary, estimate, data


def test_estimate_covers_every_configuration(pipeline):
    estimate = pipeline[3]
    assert set(estimate) == set(ge.BY_TAG)
    assert all(v > 0 for v in estimate.values())


def test_pipeline_runs_and_logs(pipeline):
    paths, seeds, _, _, data = pipeline
    runs = ge.find_runs(paths, fast=True)
    assert set(runs) == {(a.tag, s) for a in ge.ABLATIONS for s in seeds}
    for (tag, _), rec in runs.items():
        assert rec["task"] == "next_token"
        assert not any(k.startswith("test_") for k in rec["scalars"])
        assert rec["config"]["grad_clip"] == 1.0
        assert rec["data"]["cublas_workspace_config"] == ":4096:2"
        assert rec["data"]["context"] == ge.BY_TAG[tag].context
        assert rec["data"]["tokenizer"]["vocab"] == ge.BY_TAG[tag].vocab
        assert rec["data"]["tokenizer"]["lo_uv"] == data.lo_uv
        assert rec["selection"]["metric"] == "loss"
        assert "val_bits_per_sample" in rec["history"] or \
            "val_bits_per_sample" in rec["scalars"]
        tok = gx.tokenizer_from_log(rec)
        assert tok.vocab == ge.BY_TAG[tag].vocab


def test_summary_selection_and_references(pipeline):
    paths, seeds, summary, _, _ = pipeline
    runs = ge.find_runs(paths, fast=True)
    scores = ge.val_bits(runs, seeds)
    pool = [a.tag for a in ge.ABLATIONS if a.selectable]
    best = min(pool, key=lambda t: np.mean(scores[t]))
    assert summary["selected_config"] == best
    s = summary["scalars"]
    assert s["params_base"] == 461568 and s["params_lstm"] == 465664
    assert s["context_ctx500"] == 500 and s["vocab_vocab2560"] == 2560
    assert s["delta_uv_vocab64"] == 40
    assert "diff_mean_no_mask" not in s and "diff_mean_base" not in s
    assert s["n_seeds_compared_lstm"] == len(seeds)
    d = ge.paired(runs[("lstm", 42)], runs[("base", 42)], 50)
    assert s["diff_s42_lstm"] == pytest.approx(d["diff"])
    assert s["uniform_bits"] == pytest.approx(math.log2(gx.SPAN_UV))
    for v in ge.VOCABS:
        assert s[f"bigram_bits_v{v}"] < s["uniform_bits"]
    h = summary["pretrained_for_chat_08"]
    assert h["selected_config"] == best
    assert set(h["checkpoints"]) == {str(x) for x in seeds}


def test_summary_generation_numbers(pipeline):
    s, summary = pipeline[2]["scalars"], pipeline[2]
    sel, other = summary["selected_config"], summary["comparison_config"]
    for tag in (sel, other):
        for name, _, _ in gx.SETTINGS:
            assert f"gen_{tag}_{name}_std_ratio" in s
            assert 0 <= s[f"gen_{tag}_{name}_frac_flat"] <= 1
    for ab in ge.ABLATIONS:
        assert f"gen_{ab.tag}_t10_w1_mv" in s
    assert s["gen_n_records"] > 0
    assert 0 <= s["real_hr_defined_frac"] <= 1


def test_tables_use_only_defined_macros(pipeline):
    paths, _, summary, _, _ = pipeline
    mk = ge.dx.make_numbers_module()
    lines, _ = mk.build({"generative": summary["run_id"]},
                        logs_dir=paths.logs, allow_fast=True)
    defined = set(re.findall(r"\\newcommand\{\\(\w+)\}", "\n".join(lines)))
    for name in ("generative_ablations_fast.tex",
                 "generative_sampling_fast.tex"):
        text = (paths.tables / name).read_text(encoding="utf-8")
        used = set(re.findall(r"\\(generative\w+)\{\}", text))
        assert used and used <= defined, (name, used - defined)


def test_examples_come_from_the_validation_records_in_use():
    cache = synthetic_ecg_cache()
    data = gx.GenData(cache, fast=True)
    ex = ge._example_rows(data, cache)
    assert ex and all(0 <= i < len(data.idx["val"]) for i in ex.values())
    full = gx.GenData(cache)
    same = ge.cx.example_records(cache)       # D16-D18 records, whole fold
    got = {c: int(full.idx["val"][i])
           for c, i in ge._example_rows(full, cache).items()}
    assert got == same


def test_figures_written(pipeline):
    paths = pipeline[0]
    for name in ("generative_curves_fast", "generative_ablations_fast",
                 "generative_context_fast", "generative_samples_fast",
                 "generative_sampling_fast", "generative_attention_fast"):
        for ext in ("pdf", "png"):
            assert (paths.figures / f"{name}.{ext}").exists(), name


def test_fast_summary_is_not_registered(pipeline):
    paths = pipeline[0]
    runs = json.loads(paths.runs_file.read_text(encoding="utf-8"))
    assert runs == {"trivial": "trivial_full"}


def test_summary_macro_names_are_unique_and_printable(pipeline):
    s = pipeline[2]["scalars"]
    mk = ge.dx.make_numbers_module()
    names = [mk.macro_name("generative", k) for k in s]
    assert len(names) == len(set(names))
    assert not any(isinstance(v, float) and math.isnan(v) and
                   not k.endswith("_sd") for k, v in s.items())
