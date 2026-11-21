# Block 07 — Self-attention and the Transformer encoder (T12)

Chat DL 07, 2026-11-21. T12 done by MM, reviewed by OF (to confirm at
closing). Everything executed on Oriana's laptop (RTX 3050 Ti): all runs at
commit `f11e501` (logs with schema 2 and `git_dirty: false`). Decisions:
`docs/DECISIONS.md` D18 (this block), D3, D4, D14 and D17.

Numbers below are copied from `report/numbers.tex` (3 decimals) only to
make this note readable. **Reports must use the macros**
(`\transformer...{}`, `\rnn...{}`, `\cnn...{}`, `\dense...{}`,
`\trivial...{}`), never these figures. Values marked *(log)* come from the
per-run logs and have no macro.

## What the block delivers

| Item | File |
|---|---|
| Attention written by hand: scaled dot-product, multi-head self-attention with W_O, optional causal mask, FFN, Pre/Post-LN block, sinusoidal encoding | `src/models/attention.py` |
| Model | `src/models/transformer.py` (`TransformerClassifier`; registered as `transformer`) |
| Ablations, final model, summary, tables, figures, time estimate | `src/transformer_experiments.py` (generic helpers imported from `src/dense_experiments.py`, `src/cnn_experiments.py` and `src/rnn_experiments.py`) |
| Tests (attention against `nn.MultiheadAttention`, causal mask, Pre/Post-LN order, positional encoding, patch embedding = `Conv1d`, parameter count, order invariance without positions, one-layer model = attention pooling, grid, selection, whole pipeline on synthetic data) | `tests/test_transformer.py` (66 tests) |
| 45 ablation runs + 3 final runs | `results/logs/transformer_*.json`, `results/predictions/transformer_*.npz` |
| Summary of the block (alias `transformer` in `report/runs.json`) | `results/logs/transformer_summary.json` |
| Tables | `report/tables/transformer_ablations.tex`, `transformer_attention.tex`, `transformer_final.tex` |
| Figures | `report/figures/transformer_curves`, `transformer_ablations`, `transformer_time_gradient`, `transformer_attention`, `transformer_attention_maps` (PDF, PNG) |

```
python -m src.transformer_experiments estimate     # time per epoch on this GPU
python -m src.transformer_experiments all --fast   # smoke test
python -m src.transformer_experiments all          # run (45) + final + report
python report/make_numbers.py
```

Run time on the laptop *(log)*: about 1 h 05 min for the 45 runs and the 3
final runs, then about 15 min for the report (interpretation on the CPU).
Time per epoch (`\transformerEpochTimeMean...{}`): 1.4-2.5 s with one
block or less, 4.2 s for the baseline, 7.1 s with 4 blocks, 8.7 s with 200
tokens (patches of 5). Every configuration, including 200 tokens, ran
without memory errors; peak GPU memory was only printed by `estimate` and
is not in the logs.

## Architecture

- **Tokens.** The record (batch, 12, 1000) is cut into 100 non-overlapping
  patches of 10 samples (100 ms), as the RNN frames (D17). Each patch, 12
  leads x 10 samples = 120 numbers, goes through one Linear layer to a
  128-d token. This Linear layer is exactly a `Conv1d` with kernel = stride
  = 10 (tested).
- **Baseline.** Sinusoidal positional encoding added to the tokens; 2
  Pre-LN blocks (LayerNorm → 4-head self-attention with W_O → residual;
  LayerNorm → FFN 128-512-128 with GELU → residual); dropout 0.1 on the
  sublayer outputs; final LayerNorm; mean over the 100 tokens; dropout 0.5
  and a Linear layer with 5 logits. 411,909 parameters
  (`\transformerParamsBase{}`), between the RNN baseline (260,741) and the
  final CNN (468,325).
- **Attention** is written by hand as in course notebooks 01-02:
  softmax(QKᵀ/√d_k)·V per head, heads concatenated, then W_O. It gives the
  same output as `nn.MultiheadAttention` with the same weights (tested,
  also with a causal mask).
- **Step-by-step models.** `sa_single`: one head, no W_O, no residual, no
  LayerNorm, no FFN. With the mean readout its output is Σ_j ā_j v_j, an
  attention pooling of the value vectors (tested); ā_j is the attention
  that patch j receives, averaged over the queries. `mhsa` adds 4 heads and
  W_O, `no_ffn` adds Add & Norm, `layers1` adds the FFN (one full block),
  `base` has two blocks.
- Training as D15-D17: Adam 1e-3, batch 64, up to 50 epochs, patience 5 on
  validation macro AUROC, gradient clipping at total norm 1.0,
  `CUBLAS_WORKSPACE_CONFIG=:4096:2`. The three final runs reproduced their
  ablation runs exactly (same validation AUROC and best epoch per seed:
  8, 17, 12), so the GPU runs were deterministic *(log)*.

## Results

Validation macro AUROC, mean ± SD over seeds 42-44
(`transformer_ablations.tex`). Δ = mean paired difference with the
reference of the same seed; better / worse = seeds whose paired 95 %
interval lies wholly above / below 0. Shuffle = change in validation macro
AUROC when the patches of every record are put in a random order (seed 42,
CPU). Last 2 s = share of the input sensitivity in the last 2 s of the
record (0.2 if uniform), seed 42.

| Configuration | Reference | Tokens | Val. macro AUROC | Δ | better / worse | Shuffle | Last 2 s |
|---|---|---|---|---|---|---|---|
| One self-attention layer, 1 head | mhsa | 100 | 0.822 ± 0.010 | −0.045 | 0 / 3 | +0.000 | 0.195 |
| Multi-head attention (4 heads, W_O) | no_ffn | 100 | 0.867 ± 0.004 | −0.028 | 0 / 3 | +0.000 | 0.204 |
| + Add & Norm | layers1 | 100 | 0.895 ± 0.002 | −0.008 | 0 / 2 | −0.000 | 0.198 |
| + FFN (1 block) | base | 100 | 0.903 ± 0.004 | −0.003 | 0 / 1 | +0.001 | 0.207 |
| Baseline (2 blocks) | — | 100 | 0.905 ± 0.002 | — | — | +0.001 | 0.187 |
| 4 blocks | base | 100 | 0.906 ± 0.002 | +0.001 | 0 / 0 | +0.002 | 0.191 |
| 1 head (128 dimensions) | base | 100 | 0.904 ± 0.003 | −0.001 | 0 / 1 | −0.001 | 0.185 |
| 8 heads (16 dimensions) | base | 100 | 0.906 ± 0.001 | +0.000 | 0 / 0 | −0.001 | 0.200 |
| No positional encoding | base | 100 | 0.911 ± 0.000 | +0.006 | 3 / 0 | 0.000 | 0.196 |
| Learned position embeddings | base | 100 | 0.899 ± 0.004 | −0.006 | 0 / 2 | +0.002 | 0.201 |
| Patches of 25 samples | base | 40 | 0.903 ± 0.002 | −0.002 | 0 / 0 | +0.000 | 0.200 |
| Patches of 50 samples | base | 20 | 0.893 ± 0.002 | −0.013 | 0 / 3 | +0.000 | 0.196 |
| CLS token instead of the mean | base | 100 | 0.904 ± 0.001 | −0.001 | 0 / 0 | −0.001 | 0.198 |
| Post-LN (original Transformer) | base | 100 | 0.904 ± 0.004 | −0.001 | 0 / 1 | −0.000 | 0.239 |
| Patches of 5 samples | base | 200 | 0.910 ± 0.002 | +0.004 | 0 / 0 | −0.002 | 0.183 |

The step-by-step rows are compared with the *next* step (so a negative Δ
means that adding the next piece helped). Extra pair (`sa_vs_block`): one
self-attention layer vs one full block, −0.080 on average, worse in all
three seeds (`\transformerPairSaVsBlockMean{}`).

Macros of the last two columns: `\transformerOrderDrop...{}` (shuffle)
and `\transformerSensLast...{}` (last 2 s).

**Final model** (chosen on validation by the D18 rule, best mean of the 15
configurations, before the test fold was touched; `transformer_final.tex`):
- Final Transformer = `no_pos` (2 Pre-LN blocks, no positional encoding).
  Test macro AUROC 0.903 ± 0.004; seed 42: 0.900 [0.891, 0.908]. Test
  macro F1 0.707 ± 0.007.
- Paired differences on test (mean over seeds; the other model better or
  worse in every seed):
  - vs the final CNN −0.025 (CNN better in all three seeds,
    `\transformerDiffCnnMean{}`);
  - vs the final RNN (`conv`) −0.015 (RNN better in all three);
  - vs the final RNN + attention (`raw_attn`) −0.014 (better in all three);
  - vs the final Dense +0.159 and vs the trivial model +0.185 (Transformer
    better in all three).
- Per class (mean over seeds; final CNN, final RNN in brackets): NORM 0.931
  (0.946, 0.943), MI 0.900 (0.926, 0.918), STTC 0.918 (0.937, 0.928), CD
  0.879 (0.920, 0.913), HYP 0.886 (0.909, 0.888). The largest gap to the
  CNN is in CD (−0.041) and MI (−0.026).

## What each ablation shows

1. **The Transformer is the weakest of the three architectures with shared
   weights, but far above the Dense network.** On test it is 0.025 below
   the CNN and 0.015 below the RNN, in every seed, with a similar number of
   parameters. The gap to the Dense network (+0.16) is close to that of the
   CNN and the RNN.
2. **Self-attention step by step: the largest gains come from multiple
   heads and from the residual path with LayerNorm.** One layer with one
   head reaches 0.822. Four heads with W_O add 0.045; Add & Norm adds
   0.028; the FFN 0.008 (better in two seeds); a second block 0.003 (one
   seed). Each piece helps less than the one before. A full block beats a
   bare attention layer by 0.080 in all seeds.
3. **One attention layer is already an attention pooling over patches.**
   With 65,285 parameters it gets 0.822 on validation, well above the
   Dense network (test 0.744, `\denseFinalTestMacroAurocMean{}`), although
   it is linear in each patch: the same projection applied to every patch
   is the weight sharing the Dense network lacks. It is still 0.08 below
   the RNN with attention pooling (0.906), whose states have already
   processed the whole record before being pooled.
4. **The models do not use the order of the patches.** Putting the 100
   patches of every record in a random order changes the validation AUROC
   by 0.002 or less in all 15 configurations (seed 42), including those
   with sinusoidal or learned positions. Without positional encoding the
   change is exactly 0, as it must be (tested). In practice every model
   here classifies the *set* of 100 ms patches, not their sequence.
5. **Hence removing the positional encoding helps (+0.006, better in all
   seeds) and learned positions hurt (−0.006, worse in two).** Our
   explanation, not tested: if the order is not used, the encoding is only
   an extra signal added to every token that the network must learn to
   ignore; learned positions add 12,800 parameters for that, and stop
   earlier (best epoch 8.3 vs 11.7). Also not tested: why order does not
   help. The five superclasses are largely defined by beat morphology
   (shape of QRS, ST, T), which a set of patches keeps, while rhythm
   (RR intervals) needs the order. An RR-dependent label set would be the
   test.
6. **Heads, depth and readout barely matter beyond one full block.** 1, 4
   or 8 heads: within 0.001; 2 or 4 blocks: +0.001 (4 blocks: 7 s per
   epoch instead of 4); CLS token or mean: −0.001. None is better or worse
   in more than one seed.
7. **Shorter patches are better; long patches hurt.** 50 samples (20
   tokens) −0.013 in all seeds; 25 samples −0.002; 5 samples (200 tokens)
   +0.004, not significant in any seed but the second-best configuration,
   at twice the time per epoch. Our explanation, not tested: a patch is
   flattened into one vector by a single Linear layer, so a QRS complex is
   seen by different weights depending on where it falls inside the patch
   (the same argument as RNN point 7 in `06_rnn.md`). Shorter patches
   reduce this; a convolutional stem would remove it (rejected in D18 to
   keep the block about attention).
8. **Post-LN vs Pre-LN.** The result is the same (−0.001, worse in one
   seed), but the training loss at epoch 5 is higher with Post-LN in all
   three seeds (+0.010; `transformer_curves`, panel c). This weakly
   supports the course claim that Pre-LN optimises more easily (Part II
   slides 35-36); with only 2 blocks the effect on the final result is
   negligible. With 4 blocks Post-LN was not tried.
9. **The baseline overfits early** (`transformer_curves`, panel a): the
   validation loss is lowest around epoch 9-11 and then rises, while the
   training loss keeps falling. Mean best epochs are 6-20 across
   configurations; the three final runs stopped after 13, 22 and 17
   epochs *(log)*.
10. **Sensitivity over the 10 s is uniform** (`transformer_time_gradient`):
    the last 2 s get 0.18-0.21 of the sensitivity in every configuration
    except Post-LN (0.24), against 0.46 for the last state of the LSTM.
    This follows from the structure: self-attention and the mean readout
    treat all positions alike, with no recurrence that favours recent
    inputs.
11. **Attention statistics** (`transformer_attention.tex`, seed 42,
    validation):
    - The mean distance between a query and the patches it attends to is
      3.0-3.6 s in every configuration, the same as with uniform attention
      (3.33 s). Attention is not local.
    - Each query still concentrates its weight on a few keys: normalised
      entropy per query 0.07 (one head, one layer) to 0.52 (CLS), far from
      1 (uniform). The weight on the query's own patch is small (0.3-7 %).
    - So each query looks at a few patches chosen by content, anywhere in
      the record, not at its neighbours.
12. **Attention maps** (`transformer_attention_maps`,
    `transformer_attention`, seed 42):
    - The maps are made of vertical stripes: most queries attend to the
      same few keys. In the NORM example these keys fall at the QRS
      complexes of lead II (one stripe per beat for some heads; the single
      head of `sa_single` concentrates on one or two beats).
    - Different heads of the last layer pick different subsets of beats.
    - This alignment with the QRS is read from the figures only; no number
      in the summary measures it. Do not state it as a measured result.
    - Three of the five example records are missed by the final model
      (NORM 0.18 ≤ 0.44, MI 0.12 ≤ 0.39, HYP 0.15 ≤ 0.28) and two by
      `sa_single` (NORM, MI). The probabilities are in the column titles;
      read the weights with them in mind.

## Selection of the final model: caveats to state in the reports

- The final Transformer has **no positional encoding**: it is a
  permutation-invariant set model over 100 ms patches. This was allowed by
  the D18 rule (best of all 15), fixed before the runs. Say so explicitly
  when it is called "the Transformer".
- `no_pos` and `patch5` differ by 0.002 on validation; `patch5` was not
  compared directly with `no_pos`, and it keeps the sinusoidal encoding.
- The comparison with the RNN is direct (same 100 patches of 10 samples for
  the RNN baseline), but the final RNN is the convolutional hybrid `conv`.

## PyTorch details met in this block

- **Shapes.** Patches: `reshape(B, 12, T, P).permute(0, 2, 1, 3)
  .reshape(B, T, 12·P)`; attention scores (B, H, T, T); heads split with
  `view(B, T, H, d_k).permute(0, 2, 1, 3)` and joined back with
  `permute(0, 2, 1, 3).reshape(B, T, d_model)`; `view()` would fail there,
  because `permute()` leaves the tensor non-contiguous.
- **Scaling.** Scores are divided by √d_k (per head), not √d_model.
- **Masking.** The causal mask sets future scores to −inf before the
  softmax (`masked_fill`), so their weights are exactly 0 (tested).
- **Comparing with `nn.MultiheadAttention`.** It stores W_Q, W_K, W_V
  stacked in `in_proj_weight` (3·d, d); the test copies our three matrices
  into it and turns biases off. It averages the weights over heads unless
  `average_attn_weights=False`.
- **Positional encoding as a buffer.** The sinusoidal table is registered
  with `register_buffer(..., persistent=False)`: it moves to the GPU with
  the model but is neither trained nor saved in the checkpoint (it is
  rebuilt when the model is created).
- **CLS token.** An `nn.Parameter` of shape (1, 1, d) expanded to the batch
  and concatenated before the patches (`n_prefix = 1`); the positional
  encoding is added to all 101 tokens, so the CLS token takes position 0.
- **Attention maps** are returned by every block (`(x, weights)`) and
  collected in eval mode on the CPU, so dropout does not change them.

## Problems met

- None during the runs; no correction by OF or MM in this session.
- Before the runs, in the container: `fig.tight_layout()` clashed with the
  constrained layout of `eda.set_style` and was removed; one wrong test
  assertion was removed.

## For the next chats

- **Chat 08:** best Transformer = `no_pos` (checkpoints
  `results/checkpoints/transformer_no_pos_s4{2,3,4}_20261121-17{0222,0345,0545}.pt`,
  not committed). For the single-lead ablation:
  `TransformerClassifier(n_leads=1, **args)`. The CNN is still the
  strongest model.
- **Chat 13 (ECG-GPT):** reuse `src/models/attention.py`
  (`MultiHeadSelfAttention` and `TransformerBlock` with `causal=True`,
  `causal_mask(T)`; tested against `nn.MultiheadAttention` with a causal
  mask). Here a 2-block model over 100 patches trains in about 1 min (4 s
  per epoch); 200 tokens fit the GPU. Note that for generation the order
  matters, so the positional encoding that did not help here is needed
  there.
- **Chats 10 and 11:**
  - Inputs: `\input{tables/transformer_ablations.tex}`,
    `transformer_attention.tex` and `transformer_final.tex`. They need
    `booktabs` and `numbers.tex`; the final table also uses `\cnn...`,
    `\rnn...`, `\dense...` and `\trivial...` macros.
  - Figures: `transformer_curves`, `transformer_ablations`,
    `transformer_time_gradient`, `transformer_attention`,
    `transformer_attention_maps`.
  - State the caveat that the final model has no positional encoding and
    that no configuration used the order of the patches (point 4).
  - The explanations in points 5, 7 and 12 are hypotheses, not tested.
- **Chat 09, [verify]:** Vaswani et al. 2017 (Transformer, scaled
  dot-product and multi-head attention, Post-LN, sinusoidal encoding,
  dropout 0.1); Xiong et al. 2020 (Pre-LN trains more easily; course Part
  II slides 35-36); Devlin et al. 2019 (CLS token); Dosovitskiy et al.
  2021 (patches as tokens, learned position embeddings); Hendrycks and
  Gimpel 2016 (GELU).
