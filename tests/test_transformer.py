"""Tests of the self-attention and Transformer block (T12): attention module
(against nn.MultiheadAttention, causal mask), Transformer classifier
(patches, positional encoding, readouts, the step-by-step study),
interpretation helpers, ablation grid, selection and the whole pipeline
(estimate -> runs -> final -> summary, tables, figures) on a small synthetic
cache, on the CPU.

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
from src import transformer_experiments as tx
from src.models import MODELS
from src.models.attention import (FeedForward, MultiHeadSelfAttention,
                                  TransformerBlock, attention, causal_mask,
                                  sinusoidal_encoding)
from src.models.transformer import TransformerClassifier
from tests.test_dense import synthetic_cache


# ------------------------------------------------------- attention module
def test_attention_weights_are_a_softmax_over_keys():
    torch.manual_seed(0)
    q, k, v = torch.randn(2, 3, 6, 8), torch.randn(2, 3, 6, 8), \
        torch.randn(2, 3, 6, 5)
    out, w = attention(q, k, v)
    assert out.shape == (2, 3, 6, 5) and w.shape == (2, 3, 6, 6)
    assert (w >= 0).all()
    assert torch.allclose(w.sum(dim=-1), torch.ones(2, 3, 6))
    expected = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(8), dim=-1)
    assert torch.allclose(w, expected)
    assert torch.allclose(out, expected @ v)


def test_identical_tokens_get_uniform_weights():
    """Notebook 01's TO DO: with the same token repeated, every query
    scores every key equally, so the weights are uniform."""
    x = torch.ones(1, 4, 8)
    _, w = attention(x, x, x)
    assert torch.allclose(w, torch.full((1, 4, 4), 0.25))


def _copy_from_torch(ours: MultiHeadSelfAttention, mha: nn.MultiheadAttention):
    """Notebook 02, "optional, advanced" cell: in_proj_weight stacks W_Q,
    W_K and W_V; out_proj is W_O."""
    d = ours.d_model
    with torch.no_grad():
        ours.W_Q.weight.copy_(mha.in_proj_weight[:d])
        ours.W_K.weight.copy_(mha.in_proj_weight[d:2 * d])
        ours.W_V.weight.copy_(mha.in_proj_weight[2 * d:])
        ours.W_O.weight.copy_(mha.out_proj.weight)


@pytest.mark.parametrize("heads", [1, 2, 4])
def test_same_as_nn_multiheadattention(heads):
    torch.manual_seed(0)
    d = 16
    mha = nn.MultiheadAttention(d, heads, bias=False, batch_first=True).eval()
    ours = MultiHeadSelfAttention(d, heads).eval()
    _copy_from_torch(ours, mha)
    x = torch.randn(3, 7, d)
    out, w = ours(x)
    ref, ref_w = mha(x, x, x, need_weights=True, average_attn_weights=False)
    assert w.shape == (3, heads, 7, 7)
    assert torch.allclose(out, ref, atol=1e-6)
    assert torch.allclose(w, ref_w, atol=1e-6)


def test_causal_attention_same_as_torch_and_blind_to_the_future():
    """causal=True: weights above the diagonal are exactly 0, the output at
    position t does not change when tokens after t change, and it matches
    nn.MultiheadAttention with the equivalent mask (True = NOT allowed in
    PyTorch's convention, the opposite of ours)."""
    torch.manual_seed(0)
    d, T = 16, 9
    ours = MultiHeadSelfAttention(d, 4, causal=True).eval()
    mha = nn.MultiheadAttention(d, 4, bias=False, batch_first=True).eval()
    _copy_from_torch(ours, mha)
    x = torch.randn(2, T, d)
    out, w = ours(x)
    assert torch.all(w.triu(diagonal=1) == 0)
    assert torch.allclose(w.sum(-1), torch.ones(2, 4, T))
    ref, _ = mha(x, x, x, attn_mask=~causal_mask(T))
    assert torch.allclose(out, ref, atol=1e-6)
    y = x.clone()
    y[:, 5:] += 3.0
    out2, _ = ours(y)
    assert torch.allclose(out[:, :5], out2[:, :5], atol=1e-6)
    assert not torch.allclose(out[:, 5:], out2[:, 5:])


def test_causal_block_for_chat_13():
    """A stack of causal blocks (the ECG-GPT of D13) never lets a position
    see the future, through the FFN and the residual stream too."""
    torch.manual_seed(0)
    blocks = nn.ModuleList([TransformerBlock(16, 2, 64, causal=True)
                            for _ in range(2)]).eval()
    x = torch.randn(1, 10, 16)
    y = x.clone()
    y[:, 6:] = torch.randn(1, 4, 16)
    a, b = x, y
    for blk in blocks:
        a, _ = blk(a)
        b, _ = blk(b)
    assert torch.allclose(a[:, :6], b[:, :6], atol=1e-6)


def test_no_output_projection_with_one_head_is_notebook_01():
    """out_proj=False and one head: output = softmax(QK^T/sqrt(d)) V."""
    torch.manual_seed(0)
    sa = MultiHeadSelfAttention(8, 1, out_proj=False)
    x = torch.randn(2, 5, 8)
    out, w = sa(x)
    q, k, v = sa.W_Q(x), sa.W_K(x), sa.W_V(x)
    ref = torch.softmax(q @ k.transpose(1, 2) / math.sqrt(8), -1) @ v
    assert torch.allclose(out, ref, atol=1e-6)
    assert isinstance(sa.W_O, nn.Identity)


def test_heads_must_divide_d_model():
    with pytest.raises(ValueError):
        MultiHeadSelfAttention(10, 4)


def test_feed_forward_acts_on_each_token_separately():
    torch.manual_seed(0)
    ffn = FeedForward(8, 32)
    x = torch.randn(2, 5, 8)
    out = ffn(x)
    assert out.shape == x.shape
    assert torch.allclose(out[:, 2], ffn(x[:, 2:3])[:, 0])


@pytest.mark.parametrize("norm", ["pre", "post"])
def test_block_order_of_operations(norm):
    """Pre-LN: x + MHSA(LN1(x)), then + FFN(LN2(.)); Post-LN: LN1(x +
    MHSA(x)), then LN2(. + FFN(.)). Checked by recomputing the block."""
    torch.manual_seed(0)
    blk = TransformerBlock(16, 4, 64, dropout=0.0, norm=norm).eval()
    x = torch.randn(2, 6, 16)
    out, _ = blk(x)
    if norm == "pre":
        h = x + blk.attn(blk.ln1(x))[0]
        ref = h + blk.ffn(blk.ln2(h))
    else:
        h = blk.ln1(x + blk.attn(x)[0])
        ref = blk.ln2(h + blk.ffn(h))
    assert torch.allclose(out, ref, atol=1e-6)


def test_block_without_residual_and_ffn_is_attention_only():
    torch.manual_seed(0)
    blk = TransformerBlock(16, 4, 64, dropout=0.0, norm="none", ffn=False,
                           residual=False).eval()
    x = torch.randn(2, 6, 16)
    out, w = blk(x)
    ref, ref_w = blk.attn(x)
    assert torch.allclose(out, ref) and torch.allclose(w, ref_w)
    assert blk.ffn is None


def test_sinusoidal_encoding():
    pe = sinusoidal_encoding(50, 16)
    assert pe.shape == (50, 16)
    assert torch.allclose(pe[0, 0::2], torch.zeros(8))      # sin(0)
    assert torch.allclose(pe[0, 1::2], torch.ones(8))       # cos(0)
    assert torch.allclose(pe[:, 0], torch.sin(torch.arange(50.)))
    assert pe[7, 2] == pytest.approx(math.sin(7 / 10000 ** (2 / 16)), abs=1e-6)
    assert pe.abs().max() <= 1.0 + 1e-6
    assert len({tuple(r) for r in pe.numpy().round(5)}) == 50   # all distinct


# ------------------------------------------------------- classifier model
def test_registered():
    assert MODELS["transformer"] is TransformerClassifier


@pytest.mark.parametrize("ab", tx.ABLATIONS, ids=lambda a: a.tag)
def test_output_shape_every_configuration(ab):
    model = TransformerClassifier(**ab.model_args)
    assert model(torch.randn(2, 12, 1000)).shape == (2, 5)
    assert model.n_steps == 1000 // ab.model_args["patch"]


def test_parameter_count_of_the_baseline():
    """Patch embedding 120 -> 128; per block 4 projections of 128 x 128
    (no bias), FFN 128 -> 512 -> 128, two LayerNorms; final LayerNorm; head
    128 -> 5."""
    d, f = 128, 512
    block = 4 * d * d + (d * f + f + f * d + d) + 2 * 2 * d
    expected = (120 * d + d) + 2 * block + 2 * d + (d * 5 + 5)
    model = TransformerClassifier(**tx.BASE_ARGS)
    assert sum(p.numel() for p in model.parameters()) == expected == 411909


def test_heads_do_not_change_the_parameter_count():
    counts = {h: sum(p.numel() for p in TransformerClassifier(
        num_heads=h).parameters()) for h in (1, 4, 8)}
    assert len(set(counts.values())) == 1


def test_tokens_are_consecutive_samples_lead_by_lead():
    """Token t holds samples 10t .. 10t+9 of lead 0, then of lead 1, ...:
    the same layout as the RNN frames (D17)."""
    x = torch.arange(2 * 12 * 1000, dtype=torch.float32).reshape(2, 12, 1000)
    tok = TransformerClassifier(patch=10).tokens(x)
    assert tok.shape == (2, 100, 120)
    assert torch.equal(tok[1, 7, :10], x[1, 0, 70:80])
    assert torch.equal(tok[1, 7, 10:20], x[1, 1, 70:80])
    assert torch.equal(tok[0, 99, -10:], x[0, 11, 990:1000])
    from src.models.rnn import RNNClassifier
    assert torch.equal(tok, RNNClassifier(frame=10).sequence(x))


@pytest.mark.parametrize("patch", [5, 10, 50])
def test_patch_embedding_is_a_strided_convolution(patch):
    """Linear on each patch == Conv1d(kernel = stride = patch) with the same
    weights viewed as (d_model, leads, patch)."""
    torch.manual_seed(0)
    model = TransformerClassifier(patch=patch, d_model=16, num_heads=2)
    conv = nn.Conv1d(12, 16, kernel_size=patch, stride=patch)
    with torch.no_grad():
        conv.weight.copy_(model.embed.weight.view(16, 12, patch))
        conv.bias.copy_(model.embed.bias)
    x = torch.randn(3, 12, 1000)
    assert torch.allclose(model.embed(model.tokens(x)),
                          conv(x).permute(0, 2, 1), atol=1e-5)


def _shuffle(x: torch.Tensor, patch: int, perm: torch.Tensor) -> torch.Tensor:
    b, c, n = x.shape
    return x.reshape(b, c, n // patch, patch)[:, :, perm].reshape(b, c, n)


@pytest.mark.parametrize("readout", ["mean", "cls"])
def test_without_positional_encoding_the_order_is_invisible(readout):
    """No positional encoding: self-attention and the FFN treat the tokens
    as a set, so shuffling the patches does not change the output (D18).
    With the sinusoidal encoding it does."""
    torch.manual_seed(0)
    x = torch.randn(2, 12, 1000)
    perm = torch.randperm(100)
    bag = TransformerClassifier(pos_encoding="none", readout=readout).eval()
    assert torch.allclose(bag(x), bag(_shuffle(x, 10, perm)), atol=1e-5)
    ordered = TransformerClassifier(readout=readout).eval()
    assert not torch.allclose(ordered(x), ordered(_shuffle(x, 10, perm)),
                              atol=1e-4)


def test_shuffle_patches_helper():
    X = np.arange(2 * 12 * 1000, dtype=np.float32).reshape(2, 12, 1000)
    S = tx.shuffle_patches(X, 10, seed=0)
    assert S.shape == X.shape and not np.array_equal(S, X)
    # every patch stays intact and keeps its leads aligned
    p = S.reshape(2, 12, 100, 10)
    first = (p[:, 0, :, 0] - X[:, 0, :1]) // 10       # original patch index
    for r in range(2):
        assert sorted(first[r].astype(int)) == list(range(100))
        for t in range(100):
            k = int(first[r, t])
            assert np.array_equal(p[r, :, t], X[r, :, 10 * k:10 * k + 10])


def test_sinusoidal_encoding_is_a_buffer_learned_one_a_parameter():
    sin = TransformerClassifier()
    assert "pe" not in dict(sin.named_parameters())
    assert sin.pe.shape == (100, 128)
    learned = TransformerClassifier(pos_encoding="learned")
    assert learned.pos_embedding.weight.shape == (100, 128)
    assert learned.pos_embedding.weight.requires_grad
    cls = TransformerClassifier(readout="cls")
    assert cls.pe.shape == (101, 128) and cls.n_prefix == 1


def test_cls_readout_uses_the_first_token():
    torch.manual_seed(0)
    model = TransformerClassifier(readout="cls").eval()
    x = torch.randn(2, 12, 1000)
    h, w = model.encode(x)
    assert h.shape == (2, 101, 128) and w[0].shape == (2, 4, 101, 101)
    assert torch.allclose(model.summarize(x), h[:, 0])


def test_mean_readout():
    torch.manual_seed(0)
    model = TransformerClassifier().eval()
    x = torch.randn(2, 12, 1000)
    h, _ = model.encode(x)
    assert torch.allclose(model.summarize(x), h.mean(dim=1))


def test_single_self_attention_layer_is_an_attention_pooling():
    """sa_single (one head, no W_O, no Add & Norm, no FFN, mean readout):
    the record's vector is sum_j abar_j v_j, where abar_j is the attention
    patch j receives averaged over the queries (D18)."""
    torch.manual_seed(0)
    model = TransformerClassifier(**tx.BY_TAG["sa_single"].model_args).eval()
    x = torch.randn(3, 12, 1000)
    v = model.blocks[0].attn.W_V(model.embed_tokens(x))     # (B, T, d)
    abar = torch.from_numpy(tx.received_attention(model, x.numpy())).float()
    assert torch.allclose(abar.sum(dim=1), torch.ones(3), atol=1e-6)
    assert torch.allclose(model.summarize(x),
                          (abar.unsqueeze(-1) * v).sum(dim=1), atol=1e-5)


def test_step_by_step_structure():
    sa = TransformerClassifier(**tx.BY_TAG["sa_single"].model_args)
    assert len(sa.blocks) == 1 and sa.blocks[0].ffn is None
    assert sa.blocks[0].attn.num_heads == 1
    assert isinstance(sa.blocks[0].attn.W_O, nn.Identity)
    assert isinstance(sa.blocks[0].ln1, nn.Identity)
    assert isinstance(sa.ln_final, nn.Identity)
    no_ffn = TransformerClassifier(**tx.BY_TAG["no_ffn"].model_args)
    assert isinstance(no_ffn.blocks[0].ln1, nn.LayerNorm)
    assert no_ffn.blocks[0].ffn is None and no_ffn.blocks[0].residual
    post = TransformerClassifier(**tx.BY_TAG["post_ln"].model_args)
    assert isinstance(post.ln_final, nn.Identity)


def test_attention_maps_shape():
    model = TransformerClassifier(num_layers=3, num_heads=2).eval()
    maps = model.attention_maps(torch.randn(2, 12, 1000))
    assert maps.shape == (2, 3, 2, 100, 100)
    assert torch.allclose(maps.sum(-1), torch.ones(2, 3, 2, 100), atol=1e-5)


def test_logits_not_probabilities():
    model = TransformerClassifier()
    assert isinstance(model.classifier, nn.Linear)
    torch.manual_seed(0)
    out = model.eval()(torch.randn(16, 12, 1000) * 10)
    assert (out < 0).any() or (out > 1).any()


def test_dropout_only_in_training_mode():
    torch.manual_seed(0)
    model = TransformerClassifier()
    x = torch.randn(4, 12, 1000)
    model.train()
    assert not torch.equal(model(x), model(x))
    model.eval()
    assert torch.equal(model(x), model(x))


def test_single_lead_input():
    """Chat 08 trains the best model of each family on lead II only."""
    model = TransformerClassifier(n_leads=1)
    assert model(torch.randn(2, 1, 1000)).shape == (2, 5)


def test_invalid_arguments():
    for bad in ({"patch": 7}, {"pos_encoding": "rope"}, {"readout": "max"},
                {"norm": "batch"}, {"num_heads": 3}, {"num_layers": 0},
                {"head_dropout": 1.0}, {"dropout": 1.0}):
        with pytest.raises(ValueError):
            TransformerClassifier(**bad)


# ------------------------------------------------------- interpretation
def test_attention_numbers_of_uniform_weights():
    """Zero queries give uniform weights: entropies 1, own-patch share 1/T,
    distance equal to its uniform reference."""
    model = TransformerClassifier(patch=50, num_layers=2).eval()
    for blk in model.blocks:
        nn.init.zeros_(blk.attn.W_Q.weight)
    X = np.random.default_rng(0).normal(size=(5, 12, 1000)).astype(np.float32)
    s = tx.attention_numbers(model, X, "u")
    assert s["attn_tokens_u"] == 20 and s["attn_layers_u"] == 2
    for layer in ("first", "last"):
        assert s[f"attn_qentropy_{layer}_u"] == pytest.approx(1.0)
        assert s[f"attn_recv_entropy_{layer}_u"] == pytest.approx(1.0)
        assert s[f"attn_diag_{layer}_u"] == pytest.approx(1 / 20)
        assert s[f"attn_dist_{layer}_u"] == pytest.approx(
            s["attn_dist_uniform_u"])
    # 20 patches of 0.5 s: mean |i - j| = (T^2 - 1) / (3T) patches
    assert s["attn_dist_uniform_u"] == pytest.approx((400 - 1) / 60 * 0.5)


def test_attention_numbers_ignore_the_cls_token():
    model = TransformerClassifier(patch=50, readout="cls").eval()
    X = np.random.default_rng(0).normal(size=(3, 12, 1000)).astype(np.float32)
    s = tx.attention_numbers(model, X, "c")
    assert s["attn_tokens_c"] == 20
    assert tx.received_attention(model, X).shape == (3, 20)


def test_order_numbers_no_pos_is_exactly_zero():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 12, 1000)).astype(np.float32)
    y = (rng.random((40, 5)) < 0.4).astype(np.float32)
    y[0], y[1] = 1, 0
    torch.manual_seed(0)
    s = tx.order_numbers(TransformerClassifier(pos_encoding="none", patch=50),
                         X, y, "bag")
    assert s["order_drop_bag"] == pytest.approx(0.0, abs=1e-6)


def test_distance_matrix():
    D = tx.distance_matrix(4, 25)
    assert D[0, 3] == pytest.approx(0.75) and np.all(np.diag(D) == 0)


# ------------------------------------------------------------------ grid
def _differences(ab, other):
    a, b = ab.model_args, other.model_args
    return {k for k in a if a[k] != b[k]}


def test_ablation_grid_one_factor_at_a_time():
    tags = [a.tag for a in tx.ABLATIONS]
    assert len(tags) == len(set(tags)) == 15
    assert tags[0] == "base" and tags[-1] == "patch5"
    # the step-by-step study: one component per step
    steps = {"sa_single": {"num_heads", "out_proj"},
             "mhsa": {"norm", "residual"},          # "Add & Norm"
             "no_ffn": {"ffn"}}
    for ab in tx.ABLATIONS[1:]:
        assert re.fullmatch(r"[a-z0-9_]+", ab.tag)
        assert ab.reference in tx.BY_TAG and ab.reference != ab.tag
        diff = _differences(ab, tx.BY_TAG[ab.reference])
        assert diff == steps.get(ab.tag, diff) and \
            (ab.tag in steps or len(diff) == 1), (ab.tag, diff)
    chain = [tx.BY_TAG[t].reference for t in tx.STEP_TAGS[:-1]]
    assert chain == tx.STEP_TAGS[1:]
    for a, b, name, _ in tx.EXTRA_PAIRS:
        assert a in tx.BY_TAG and b in tx.BY_TAG and re.fullmatch(r"[a-z_]+", name)


def test_configs_keep_the_d14_defaults_plus_clipping():
    for ab in tx.ABLATIONS:
        cfg = tx.make_config(ab, 42)
        assert (cfg.lr, cfg.batch_size, cfg.max_epochs, cfg.patience,
                cfg.weight_decay, cfg.normalize, cfg.pos_weight) == \
            (1e-3, 64, 50, 5, 0.0, True, False)
        assert cfg.grad_clip == 1.0
        assert cfg.model == "transformer" and cfg.tag == ab.tag
    fast = tx.make_config(tx.ABLATIONS[0], 42, fast=True)
    assert (fast.max_epochs, fast.n_boot) == (2, 50)


# ------------------------------------------------------------- selection
def _fake_runs(means: dict, seeds=(42, 43, 44)):
    return {(tag, s): {"scalars": {"val_macro_auroc": m + 0.001 * i}}
            for tag, m in means.items() for i, s in enumerate(seeds)}


def test_selection():
    means = {a.tag: 0.80 for a in tx.ABLATIONS}
    means["patch5"] = 0.90
    assert tx.select_final(_fake_runs(means)) == "patch5"


def test_selection_refuses_incomplete_grids():
    runs = _fake_runs({a.tag: 0.80 for a in tx.ABLATIONS})
    del runs[("patch5", 44)]
    with pytest.raises(RuntimeError, match="patch5"):
        tx.select_final(runs)


def test_final_table_other_blocks_macros_exist():
    """The CNN, RNN, RNN + attention and Dense columns of
    transformer_final.tex use macros that make_numbers.py defines from the
    committed summaries (skipped if they are missing)."""
    names = {"cnn": "cnn_summary", "rnn": "rnn_summary",
             "dense": "dense_summary"}
    for name in names.values():
        if not (config.LOGS_DIR / f"{name}.json").exists():
            pytest.skip(f"results/logs/{name}.json not present")
    s = {"final_test_macro_auroc_mean": 0.9}
    table = tx.final_table({"run_id": "x", "scalars": s, "seeds": [42],
                            "cnn_summary": "cnn_summary",
                            "rnn_summary": "rnn_summary",
                            "rnn_attention_column": True,
                            "dense_summary": "dense_summary"})
    mk = tx.dx.make_numbers_module()
    lines, _ = mk.build(names)
    defined = set(re.findall(r"\\newcommand\{\\(\w+)\}", "\n".join(lines)))
    used = set(re.findall(r"\\((?:cnn|rnn|dense)\w+)\{\}", table))
    assert used and used <= defined, used - defined
    assert any(u.startswith("rnnAttnfinal") for u in used)


# ------------------------------------------------------ whole pipeline
@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    root = tmp_path_factory.mktemp("transformer")
    paths = tx.Paths(logs=root / "logs", predictions=root / "pred",
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
    estimate = tx.estimate(cache, n_steps=1, seeds=seeds, fast=True)
    tx.run_ablations(seeds, None, fast=True, cache=cache, paths=paths)
    selected = tx.run_final(fast=True, cache=cache, paths=paths, seeds=seeds)
    summary = tx.write_report(fast=True, paths=paths, seeds=seeds,
                              cache=cache)
    return paths, seeds, selected, summary, estimate


def test_estimate_covers_every_configuration(pipeline):
    estimate = pipeline[4]
    assert set(estimate) == set(tx.BY_TAG)
    assert all(v > 0 for v in estimate.values())


def test_pipeline_runs_and_logs(pipeline):
    paths, seeds, selected, _, _ = pipeline
    runs = tx.find_runs(paths, fast=True, final=False)
    assert set(runs) == {(a.tag, s) for a in tx.ABLATIONS for s in seeds}
    finals = tx.find_runs(paths, fast=True, final=True)
    assert set(finals) == {(selected, s) for s in seeds}
    for rec in runs.values():                  # ablations never see test
        assert "test_macro_auroc" not in rec["scalars"]
        assert rec["config"]["grad_clip"] == 1.0
        assert rec["data"]["cublas_workspace_config"] == ":4096:2"
    for rec in finals.values():
        assert "test_macro_auroc" in rec["scalars"]
    assert runs[("patch5", 42)]["model"]["args"]["patch"] == 5


def test_summary_selection_and_references(pipeline):
    paths, seeds, selected, summary, _ = pipeline
    runs = tx.find_runs(paths, fast=True, final=False)
    scores = tx.val_scores(runs, seeds)
    best = max(scores, key=lambda t: np.mean(scores[t]))
    assert summary["selected_config"] == best == selected
    s = summary["scalars"]
    assert "final_test_macro_auroc_mean" in s
    assert s["tokens_base"] == 100 and s["tokens_patch5"] == 200
    assert s["tokens_patch50"] == 20
    assert s["params_base"] == 411909
    assert s["n_seeds_compared_heads8"] == len(seeds)
    # step-by-step rows are paired with the next step, plus the extra pair
    for tag, ref in (("sa_single", "mhsa"), ("no_ffn", "layers1")):
        d = tx.dx.paired_diff(runs[(tag, 42)], runs[(ref, 42)], "val", 50)
        assert s[f"diff_s42_{tag}"] == pytest.approx(d["diff"])
    d = tx.dx.paired_diff(runs[("sa_single", 42)], runs[("layers1", 42)],
                          "val", 50)
    assert s["pair_sa_vs_block_s42"] == pytest.approx(d["diff"])
    assert s["pair_sa_vs_block_n_compared"] == len(seeds)
    # --fast: no comparison with other blocks on (a subset of) test
    assert not any(k.startswith(("diff_trivial", "diff_cnn", "diff_rnn",
                                 "diff_dense")) for k in s)


def test_summary_interpretation(pipeline):
    s, prof = pipeline[3]["scalars"], pipeline[3]["sensitivity_profiles"]
    for ab in tx.ABLATIONS:
        t = ab.tag
        assert sum(prof["profiles"][t]) == pytest.approx(1.0)
        assert len(prof["profiles"][t]) == 100
        assert 0 < s[f"attn_qentropy_last_{t}"] <= 1 + 1e-9
        assert 0 <= s[f"attn_diag_last_{t}"] <= 1
        assert s[f"attn_tokens_{t}"] == s[f"tokens_{t}"]
        assert -1 <= s[f"order_drop_{t}"] <= 1
    assert s["order_drop_no_pos"] == pytest.approx(0.0, abs=1e-6)


def test_tables_use_only_defined_macros(pipeline):
    """Every \\transformer... macro in the generated tables is defined by
    make_numbers.py from the summary log."""
    paths, _, _, summary, _ = pipeline
    mk = tx.dx.make_numbers_module()
    lines, _ = mk.build({"transformer": summary["run_id"]},
                        logs_dir=paths.logs, allow_fast=True)
    defined = set(re.findall(r"\\newcommand\{\\(\w+)\}", "\n".join(lines)))
    for name in ("transformer_ablations_fast.tex",
                 "transformer_attention_fast.tex",
                 "transformer_final_fast.tex"):
        text = (paths.tables / name).read_text(encoding="utf-8")
        used = set(re.findall(r"\\(transformer\w+)\{\}", text))
        assert used and used <= defined, (name, used - defined)


def test_figures_written(pipeline):
    paths = pipeline[0]
    for name in ("transformer_curves_fast", "transformer_ablations_fast",
                 "transformer_time_gradient_fast",
                 "transformer_attention_fast",
                 "transformer_attention_maps_fast"):
        for ext in ("pdf", "png"):
            assert (paths.figures / f"{name}.{ext}").exists(), name


def test_fast_summary_is_not_registered(pipeline):
    paths = pipeline[0]
    runs = json.loads(paths.runs_file.read_text(encoding="utf-8"))
    assert runs == {"trivial": "trivial_full"}


def test_summary_macro_names_are_unique(pipeline):
    s = pipeline[3]["scalars"]
    mk = tx.dx.make_numbers_module()
    names = [mk.macro_name("transformer", k) for k in s]
    assert len(names) == len(set(names))
    assert not any(isinstance(v, float) and math.isnan(v) and
                   not k.endswith("_sd") for k, v in s.items())
