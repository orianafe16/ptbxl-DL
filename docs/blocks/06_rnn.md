# Block 06 — Recurrent networks: LSTM, GRU and RNN + attention (T11)

Chat DL 06, 2026-11-21. T11 done by MM, reviewed by OF. Everything
executed by MM on Oriana's laptop (RTX 3050 Ti): all runs at commit
`cae71d5` (logs with schema 2 and `git_dirty: false`). Decisions:
`docs/DECISIONS.md` D17 (this block), D3, D4 and D14.

Numbers below are copied from `report/numbers.tex` (3 decimals) only to
make this note readable. **Reports must use the macros** (`\rnn...{}`,
`\cnn...{}`, `\dense...{}`, `\trivial...{}`), never these figures. Values
marked *(log)* come from the per-run logs and have no macro.

## What the block delivers

| Item | File |
|---|---|
| Model | `src/models/rnn.py` (`RNNClassifier`, `AttentionPooling`; registered as `rnn`) |
| Ablations, two final models, summary, tables, figures, time estimate | `src/rnn_experiments.py` (generic helpers imported from `src/dense_experiments.py` and `src/cnn_experiments.py`) |
| Tests (framing, LSTM/GRU states, bidirectional `h_n`, causality, attention and its bound, sensitivity, grid, selection, whole pipeline on synthetic data) | `tests/test_rnn.py` (49 tests) |
| 36 ablation runs + 6 final runs (two configurations x 3 seeds) | `results/logs/rnn_*.json`, `results/predictions/rnn_*.npz` |
| Summary of the block (alias `rnn` in `report/runs.json`) | `results/logs/rnn_summary.json` |
| Tables | `report/tables/rnn_ablations.tex`, `rnn_attention.tex`, `rnn_final.tex` |
| Figures | `report/figures/rnn_curves`, `rnn_ablations`, `rnn_time_gradient`, `rnn_attention` (PDF, PNG) |

```
python -m src.rnn_experiments estimate     # time per epoch on this GPU
python -m src.rnn_experiments all --fast   # smoke test
python -m src.rnn_experiments all          # run (36) + final + report
python report/make_numbers.py
```

Run time on the laptop: about 1 h 35 min for runs and finals. Time per
epoch (`\rnnEpochTimeMean...{}`): 1.6-2.4 s with 100 frames (4 s
bidirectional), 7.5 s with the convolutional front end, 13-14 s with 1000
raw steps; a raw run lasted 8.5 min on average.

## Architecture

The record (batch, 12, 1000) is turned into a sequence (batch, T,
features) and read by the recurrent network; `batch_first=True` puts time
in the second axis, the opposite of `Conv1d`.

- **Baseline:** 100 non-overlapping frames of 10 samples (100 ms). Each
  step is a vector of 12 leads x 10 samples = 120 numbers, and nothing is
  discarded. A 2-layer unidirectional LSTM with 128 units reads them, and
  its last state `h_n[-1]` feeds dropout 0.5 and a Linear layer with 5
  logits. 260,741 parameters (`\rnnParamsBase{}`), about half of the CNN.
- **Readouts:** last state (baseline); mean of `out` over time; attention
  pooling. Attention gives each step a score, a softmax over time turns the
  scores into weights α_t, and Σ α_t h_t is the vector the classifier
  sees. The additive score is vᵀ tanh(W h_t + b) with 64 hidden units; the
  notebook score is tanh(wᵀ h_t + b).
- **Inputs:** frames (baseline); raw samples (1000 steps of 12 numbers);
  convolutional front end (the first three blocks of the CNN, 125 steps of
  128 features).
- Gradient clipping at total norm 1.0 in every configuration (RNN slide
  49); `CUBLAS_WORKSPACE_CONFIG=:4096:2` (PyTorch `nn.LSTM` documentation).
  The six final runs reproduced their ablation runs exactly (same
  validation AUROC and best epoch), so with this setting the GPU runs were
  deterministic *(log)*.

## Results

Validation macro AUROC, mean ± SD over seeds 42-44 (`rnn_ablations.tex`).
Δ = mean paired difference with the reference of the same seed; better /
worse = seeds whose paired 95 % interval lies wholly above / below 0. Last
2 s = share of the input sensitivity in the last 2 s of the record (0.2 if
uniform), seed 42.

| Configuration | Reference | Steps | Val. macro AUROC | Δ | better / worse | Last 2 s |
|---|---|---|---|---|---|---|
| Baseline (LSTM, last state) | — | 100 | 0.894 ± 0.000 | — | — | 0.458 |
| GRU | base | 100 | 0.911 ± 0.003 | +0.017 | 3 / 0 | 0.305 |
| 1 recurrent layer | base | 100 | 0.888 ± 0.003 | −0.006 | 0 / 2 | 0.405 |
| Bidirectional | base | 100 | 0.895 ± 0.001 | +0.001 | 0 / 0 | 0.256 |
| Mean over time | base | 100 | 0.906 ± 0.002 | +0.011 | 3 / 0 | 0.135 |
| Additive attention | base | 100 | 0.906 ± 0.003 | +0.012 | 3 / 0 | 0.138 |
| Notebook score | attn | 100 | 0.906 ± 0.001 | −0.000 | 0 / 0 | 0.135 |
| GRU + attention | gru | 100 | 0.915 ± 0.001 | +0.003 | 1 / 0 | 0.132 |
| Bidirectional + attention | bi | 100 | 0.909 ± 0.001 | +0.013 | 3 / 0 | 0.176 |
| Convolutional front end | base | 125 | 0.925 ± 0.002 | +0.031 | 3 / 0 | 0.326 |
| Raw samples | base | 1000 | 0.912 ± 0.003 | +0.017 | 3 / 0 | 0.430 |
| Raw samples + attention | raw | 1000 | 0.923 ± 0.002 | +0.012 | 3 / 0 | 0.184 |

The baseline's SD of 0.000 is real: seeds 42, 43, 44 give 0.8946, 0.8942,
0.8947 *(log)*.

**Final models** (chosen on validation by the D17 rules before the test
fold was touched; `rnn_final.tex`):
- Final RNN = `conv` (best of all twelve). Test macro AUROC
  0.918 ± 0.002; seed 42: 0.921 [0.913, 0.928].
- Final RNN + attention = `raw_attn` (best of the five attention
  configurations). Test 0.917 ± 0.001.
- Paired differences on test, per seed: `conv` vs the final CNN −0.008,
  −0.009, −0.012, the CNN better in all three seeds (mean
  `\rnnDiffCnnMean{}` = −0.009); vs the final Dense +0.174 and vs the
  trivial model +0.200, better in all seeds. `raw_attn` vs `conv`: −0.003,
  −0.000, +0.000, no difference in any seed; `raw_attn` vs the CNN: −0.011
  on average, worse in all seeds.
- Per class (mean over seeds, final CNN in brackets): NORM 0.943 (0.946),
  MI 0.918 (0.926), STTC 0.928 (0.937), CD 0.913 (0.920), HYP 0.888
  (0.909).

## What each ablation shows

1. **The RNN is close to the CNN but below it.** The best recurrent model
   is 0.009-0.012 below the final CNN on test, in every seed, with a
   similar number of parameters (502,373 vs 468,325). The gap to the
   Dense network (+0.17) is of the same order as the CNN's: what both
   share, and the Dense network lacks, is weights shared over time.
2. **Using every state matters more than the cell type or the
   direction.** Replacing the last state by the mean of the states adds
   0.011, by additive attention 0.012 (better in all seeds). Bidirectional
   with the last state adds nothing (+0.001). A second layer helps a
   little: one layer is worse in two seeds (−0.006).
3. **Attention here works like a learned mean.** Its weights are spread
   out: normalised entropy 0.92 (1 = uniform), about 71 of the 100 steps
   effectively attended, largest weight 0.035 against a uniform 0.01
   (`rnn_attention.tex`). Its gain over the last state equals that of
   mean pooling. The two are not compared directly; the difference of
   their means is 0.001. The bottleneck that attention removes (Attention
   slide 12: a single final vector) is real, but in this task averaging
   removes it just as well.
4. **The notebook's tanh score changes nothing in the result** (−0.000,
   0 / 0). Its weights are even flatter (entropy 0.98, 92 effective steps,
   largest weight 0.018, far below its bound of 0.069), so it behaves
   almost exactly like mean pooling.
5. **Attention helps the same with 100 or 1000 steps.** +0.012 at 1000
   steps, +0.012 at 100, +0.013 bidirectional, +0.003 with a GRU. The
   D17 hypothesis was that attention would help more where the last state
   struggles (longer sequences, unidirectional); it is not supported. The
   GRU result fits point 6: the GRU's last state already reaches 0.911,
   leaving less to gain.
6. **GRU beats LSTM by 0.017** in all seeds, with 25 % fewer parameters
   and in fewer epochs (best epoch 8.3 vs 11.7). The training loss at
   epoch 5 is lower too (−0.045). Our explanation, not tested: the GRU
   has fewer gates to learn, which matters with about 17,000 training
   records. The slides only say that the variants are "all about the
   same" (RNN slide 64, Greff et al. 2015, **[verify]**).
7. **1000 raw steps are better than 100 frames (+0.017), contrary to what
   we expected.** They are much harder to optimise: training loss +0.177
   higher at epoch 5, best epoch 33 vs 12, 8.5 min per run
   (`rnn_curves`, panel c). That is the BPTT difficulty of slides 40-48.
   But the validation result is better. Our explanation, not tested:
   frames flatten 10 consecutive samples into one vector, so a QRS
   complex is seen by different weights depending on where it falls
   inside the frame. With one sample per step the same recurrent weights
   process every sample, which is the weight sharing over time of the
   CNN.
8. **The convolutional front end is the best configuration (+0.031).**
   Local filters shared over time extract beat-level features; the LSTM
   only has to combine 125 of them. This is the CNN → LSTM pipeline of RNN
   slide 80. Points 7 and 8 together suggest that what helps is sharing
   weights over time, not recurrence itself; the CNN alone (D16) is still
   better.
9. **Which part of the 10 s the output depends on** (figure
   `rnn_time_gradient`, seed 42):
   - Last state of a unidirectional LSTM: 46 % of the sensitivity falls in
     the last 2 s, against 20 % if uniform, and half of it in the last
     2.3 s, while the first 2 s get 9 %.
     This is the short memory that vanishing gradients predict (RNN slide
     48).
   - 1000 raw steps: the same pattern, plus a spike in the last 0.1 s.
   - Bidirectional: both ends weigh more (33 % in the first 2 s, 26 % in
     the last).
   - Mean and attention: almost uniform, except for a drop in the last
     second. A structural reason, derivable from the model: with a
     unidirectional network, the input at time t can only change the
     states from t onwards, so the last samples affect few of the states
     being averaged.
   - The spread matters less for the result than this suggests. The
     bidirectional network spreads its sensitivity but gains nothing, so
     looking at more of the record is not, alone, what improves the AUROC.
10. **Attention weights over the ECG** (figure `rnn_attention`, final
    `raw_attn`, seed 42):
    - The weights rise and fall with the heartbeat: in all five records
      they form one bump per beat, around the QRS complexes.
    - The Spearman correlation between the weights and our "activity"
      proxy (mean absolute sample-to-sample change) is still only 0.03
      for this model and 0.02-0.27 across configurations. Possible
      reasons, not tested: the bump may sit just after the QRS (a state
      can only "know" a QRS after reading it), or the ranks of the many
      near-zero weights between beats may be noise. The figure shows the
      alignment; the number does not capture it. Do not cite the
      correlation as evidence that attention follows the QRS.
    - Row 2 (`attn_tanh`, 100 steps) is periodic too but much flatter. Its
      uniform weight is 0.01 instead of 0.001, so the two rows have
      different scales.
    - One of the five example records is missed by the final attention
      model (NORM, 0.13 ≤ 0.48; the CNN missed it too). In row 2 the
      NORM, MI and STTC records are missed. The probabilities are in the
      column titles; read the weights with them in mind.

## Selection of the final models: caveats to state in the reports

- The final RNN is a hybrid (convolutional front end + LSTM). This was
  allowed by the D17 rule (best of all twelve), fixed before the runs.
  Its comparison with the CNN is therefore "CNN" vs "first half of the
  CNN + LSTM": the recurrent part replaces the last CNN block and the
  global pooling, and does slightly worse.
- `conv` and `raw_attn` cannot be told apart on test (paired interval
  contains 0 in all seeds), and differ by 0.002 on validation.

## PyTorch details met in this block

- **Axis order.** `nn.LSTM(batch_first=True)` expects (batch, time,
  features); the signal is (batch, leads, time). Frames: `reshape(B, 12,
  100, 10).permute(0, 2, 1, 3).reshape(B, 100, 120)`; raw: `permute(0, 2,
  1)` (tested element by element).
- **What `nn.LSTM` returns.** `out` (all states of the top layer) and
  `(h_n, c_n)`; `nn.GRU` returns `out` and `h_n` only, so the code checks
  for a tuple. `h_n[-1]` equals `out[:, -1, :]` for a unidirectional
  network of any depth. With `bidirectional=True` the last state is the
  concatenation of `h_n[-2]` (forward) and `h_n[-1]` (backward). The
  backward final state is at `out[:, 0, H:]`; `out[:, -1, :]` would pair
  the forward state with a backward state that has read one step (tested).
- **Causality.** In a unidirectional network, changing the second half of
  the record leaves the states of the first half unchanged (tested).
- **No padding.** All records have 1000 samples, so no
  `pack_padded_sequence`.
- **`dropout` of `nn.LSTM`** only acts between stacked layers; it is left
  at 0 and dropout is applied to the vector before the classifier.
- **Gradient clipping** goes between `loss.backward()` and
  `optimizer.step()` (`clip_grad_norm_`, already in the shared loop,
  D14).
- **Gradients with respect to the input** need `requires_grad_(True)` on
  the input tensor and `torch.autograd.grad`. They are computed on the CPU
  in eval mode, because cuDNN only runs the RNN backward pass in training
  mode. The records of a batch do not interact in eval mode, so the
  gradient of the summed logit gives each record's gradient.
- **`torch.bmm`** computes the context vector: (B, 1, T) × (B, T, d) →
  (B, 1, d).

## Problems met

- None during the runs; no correction by OF or MM in this session.
- The attention-activity correlation turned out to be a poor summary of
  what the attention figure shows (point 10). It is kept in the table
  because it was fixed in D17 before the runs, with the caveat above.

## For the next chats

- **Chat 07 (Transformer):** the RNN reads 100 frames of 10 samples; a
  patch length of 10 in the Transformer would make the comparison direct
  (same steps, attention instead of recurrence). Points 3-5 are the RNN +
  attention results to compare with self-attention. `src/models/__init__.py`
  now also registers `rnn`: re-read it before adding `transformer`.
- **Chat 08:** best RNN = `conv` (checkpoints
  `results/checkpoints/rnn_conv_s4{2,3,4}_20261121-13{5334,5613,5903}.pt`,
  not committed). For the single-lead ablation:
  `RNNClassifier(n_leads=1, **args)` (tested). RNN + attention =
  `raw_attn` (`rnn_raw_attn_s4{2,3,4}_20261121-14{0307,0819,1318}.pt`).
  The CNN is still the strongest model.
- **Chat 13 (ECG-GPT):** an LSTM over 1000 raw samples trains (best epoch
  about 33, 13 s per epoch with batch 64) but slowly; clipping at 1.0 was
  enough to avoid instabilities.
- **Chats 10 and 11:**
  - Inputs: `\input{tables/rnn_ablations.tex}`, `rnn_attention.tex` and
    `rnn_final.tex`. They need `booktabs` and `numbers.tex`; the final
    table also uses `\cnn...`, `\dense...` and `\trivial...` macros.
  - Figures: `rnn_curves`, `rnn_ablations`, `rnn_time_gradient`,
    `rnn_attention`.
  - State the caveats of the final models (hybrid; `conv` ≈ `raw_attn`)
    and that attention behaves like a learned mean.
  - The explanations in points 6, 7 and 10 are hypotheses, not tested.
- **Chat 09, [verify]:** Greff et al. 2015 (LSTM variants "about the
  same", RNN slide 64); Bahdanau et al. 2014 (additive attention,
  Attention slides 3, 12-13); Mikolov 2010 (gradient clipping, RNN slide
  49); Hochreiter and Schmidhuber 1997 (LSTM).
