# Block 13 — Generative models: ECG-GPT and LSTM generators of lead II (T23)

Chat DL 13, 2026-11-21/22. T23 done by OF, reviewed by MM. Everything executed on Oriana's laptop (RTX 3050 Ti, 4 GB): the 24
runs at commit `f010618` (logs with schema 2 and `git_dirty: false`), the
summary at commit `6326057`. Decisions: `docs/DECISIONS.md` D19 (this block,
with its dated note of 2026-11-22), D13 (scope), D7 (reuse in chat DL 08),
D3, D4 and D14 (protocol).

Numbers below are copied from `report/numbers.tex` (3 decimals) only to make
this note readable. **Reports must use the macros** (`\generative...{}`),
never these figures. Values marked *(log)* come from the per-run logs and
have no macro.

## What the block delivers

| Item | File |
|---|---|
| Tokenizer on the microvolt grid, data and windows, next-token task with bits per sample, counting baselines, sampling (greedy, temperature, top-k), generation with a sliding window, statistics of generated signals | `src/generative.py` |
| Models: `ECGGPT` (decoder-only Transformer, causal blocks of `src/models/attention.py`) and `ECGLSTM`; `record_features()` for chat DL 08 | `src/models/generative.py` (not in `src/models/MODELS`: that registry feeds the classifiers' `src.train`) |
| Ablations, selection, generation, summary, tables, figures, time estimate | `src/generative_experiments.py` |
| Tests (tokenizer, windows, causality with and without the mask, parameter parity, task metrics against a direct computation, paired bootstrap, n-grams, sampling, generation, heart rate, grid, selection, whole pipeline on synthetic ECG-like data) | `tests/test_generative.py` (53 tests) |
| 24 runs (8 configurations x 3 seeds, validation only) | `results/logs/generator_*.json`, `results/predictions/generator_*.npz` |
| Summary of the block (alias `generative` in `report/runs.json`) | `results/logs/generative_summary.json` |
| Tables | `report/tables/generative_ablations.tex`, `generative_sampling.tex` |
| Figures | `report/figures/generative_curves`, `generative_ablations`, `generative_context`, `generative_samples`, `generative_sampling`, `generative_attention` (PDF, PNG) |

```
python -m src.generative_experiments estimate     # time and GPU memory
python -m src.generative_experiments all --fast   # smoke test
python -m src.generative_experiments run          # 24 runs, validation only
python -m src.generative_experiments report       # generation, summary,
                                                  # tables, figures
python report/make_numbers.py
```

Run time *(log)*: `run` about 6 h of training (median time per epoch,
`\generativeEpochTimeMedian...{}`: 6.0 s with L = 100, 14.3 s for the
baseline, 19.1 s with V = 2560, 34.0 s for the LSTM, 38.8 s with L = 500);
`ctx500` seed 44 took 12 h of wall time because the laptop slept during
three epochs (D19 note). `report` about 34 min, mostly generation (the LSTM
generates about 5 times slower than the Transformer with the same window).
Peak GPU memory in training (from `estimate`, not in the logs): 0.2-1.6 GB.
The first `report` ran out of GPU memory generating with L = 500; fixed
(D19 note) and run again.

## Data and tokens

- Lead II, 100 Hz, mV. Training: all 17,418 records of folds 1-8 (including
  those without a superclass, D12 rule c); validation: the 2,146 records of
  fold 9 with a superclass (`\generativeNRecordsTrain{}`,
  `\generativeNRecordsVal{}`). The test fold is never read.
- Every training value is an integer number of microvolts
  (`\generativeFracOffUvGridTrain{}` = 0.000): the tokenisation is exact.
- Range from −1.138 to 1.421 mV (`\generativeRangeLoUv{}`,
  `\generativeRangeHiUv{}`, in µV), 2.56 mV wide; clipped: 0.06 % of the
  training samples at each end, 0.04 % / 0.06 % in validation
  (`\generativeClipFrac...{}`).
- Quantisation error on validation: 2.864 µV inside the range with
  V = 256 (`\generativeQuantRmseInrangeUvVTwoFiveSix{}`; theory for 10
  integer levels: √((10² − 1)/12) = 2.87 µV), 11.545 µV with V = 64, 0 with
  V = 2560. Including the clipped samples the RMSE is about 11-16 µV
  (`\generativeQuantRmseUvV...{}`): the 0.1 % of clipped samples dominate it.
  For scale: validation standard deviation 162.688 µV
  (`\generativeValStdUv{}`), persistence error ("next = this") 94.147 µV
  (`\generativePersistenceRmseUv{}`).

## Architecture

```
ECG-GPT:  tokens (B, T ≤ L) -> Embedding(V, 128) + sinusoidal PE (position in the window)
          -> 2 x TransformerBlock(Pre-LN, 4 heads, FFN 512, dropout 0.1, causal=True)
          -> LayerNorm -> Linear(128, V) -> logits (B, T, V)       461,568 parameters (V = 256)
LSTM:     tokens -> Embedding(V, 128) -> LSTM(128 -> 160, 2 layers) -> Dropout(0.1)
          -> Linear(160, V) -> logits (B, T, V)                     465,664 parameters
```

Training: Adam 1e-3, batch 64 windows, at most 50 epochs, patience 5 on the
validation cross-entropy, gradient clipping 1.0. One random window of L + 1
samples per training record per epoch; fixed validation windows covering
every target.

## Results

Validation, mean ± SD over seeds 42-44; Δ = paired patient-bootstrap
difference in bits with the baseline, same seed (negative = fewer bits =
better); "3/3 better" = the whole interval below 0 in the three seeds.
Bits per sample are comparable across vocabularies; cross-entropy (CE,
nats per token) is not.

| Configuration | CE (nats) | Bits per sample | Δ bits vs base | Seeds | Last-position CE |
|---|---|---|---|---|---|
| ECG-GPT baseline (L = 250, V = 256) | 1.872 ± 0.003 | 6.022 ± 0.004 | — | — | 2.062 |
| LSTM (same size) | 1.809 ± 0.003 | 5.932 ± 0.004 | −0.090 | 3/3 better | 1.956 |
| Context 1 s (L = 100) | 1.981 ± 0.003 | 6.179 ± 0.005 | +0.157 | 3/3 worse | 2.004 |
| Context 5 s (L = 500) | 1.803 ± 0.004 | 5.923 ± 0.005 | −0.099 | 3/3 better | 2.319 |
| V = 64 (40 µV) | 0.899 ± 0.003 | 6.619 ± 0.005 | +0.597 | 3/3 worse | 1.045 |
| V = 2560 (1 µV, lossless) | 3.978 ± 0.010 | **5.738 ± 0.015** | −0.284 | 3/3 better | 4.196 |
| No positional encoding | 2.139 ± 0.003 | 6.407 ± 0.004 | +0.385 | 3/3 worse | 2.385 |
| No causal mask | 0.010 ± 0.000 | 3.336 ± 0.000 | not compared | — | 2.422 |

Baselines without training (bits per sample, same validation records):
uniform 11.322; unigram 8.904 and bigram 6.896 with V = 256 (8.917 / 7.220
with V = 64; 8.891 / 6.865 with V = 2560). Every selectable configuration
beats the bigram counts of its own vocabulary; the baseline by about 0.87
bits per sample (`\generativeBigramBitsVTwoFiveSix{}` vs
`\generativeBitsMeanBase{}`). No revision of D19 was needed.

**Pre-trained model for chat DL 08** (D19 rule: lowest mean validation bits
per sample, `no_mask` excluded): **`vocab2560`**, the ECG-GPT with 2560
levels of 1 µV (1,053,696 parameters). Checkpoints, tokenizer and model
arguments in `generative_summary.json`, key `pretrained_for_chat_08`.

### Generation (seed-42 models, 256 validation records, 2 s prompt, 8 s generated)

Reference: real heart rate found in 99.2 % of the real continuations, mean
76.4 bpm; Wasserstein distance between two different sets of real records
0.004 mV (`\generativeRealWOneOtherMv{}`).

| Model, sampling | Std ratio | Flat | W1 (mV) | HR found | HR error (bpm) |
|---|---|---|---|---|---|
| V = 2560: greedy | 0.008 | 1.000 | 0.041 | 0.000 | — |
| V = 2560: T = 0.5 | 0.214 | 0.879 | 0.053 | 0.160 | 23.5 |
| V = 2560: T = 0.8 | 0.785 | 0.039 | 0.028 | 0.883 | 21.8 |
| V = 2560: T = 1.0 | 1.136 | 0.000 | 0.021 | 0.988 | 19.9 |
| V = 2560: T = 1.5 | 2.763 | 0.000 | 0.197 | 0.996 | 49.8 |
| V = 2560: top-k 10 | 0.214 | 0.242 | 0.070 | 0.082 | 33.4 |
| LSTM: greedy | 0.026 | 1.000 | 0.054 | 0.000 | — |
| LSTM: T = 0.8 | 0.872 | 0.004 | 0.025 | 0.965 | 14.2 |
| LSTM: T = 1.0 | 1.018 | 0.000 | 0.010 | 0.988 | 13.1 |
| LSTM: top-k 10 | 0.857 | 0.000 | 0.021 | 0.895 | 14.7 |

At T = 1.0 for the other configurations (HR error, W1): baseline 15.1 bpm,
0.008 mV; L = 100 17.6, 0.012; L = 500 11.8, 0.013; V = 64 13.0, 0.016;
no positions 21.2, 0.014; no mask 30.3, 0.054.

## What each ablation shows

1. **The models learn far more than local smoothness.** The baseline beats
   the bigram counting model (the 2-gram language model of RNN slides
   93-94) by about 0.87 bits per sample and predicts the next sample with an
   RMSE of 42 µV against 94 µV for persistence.
2. **Context and the previous heartbeat** (`generative_context`). The
   prediction cost falls with the context available and drops a second time
   at about 0.9-1 s of context, about one RR interval at the mean real heart
   rate (76 bpm ≈ 0.79 s): with one previous beat in the window the model can
   anticipate the next QRS. *Hypothesis, consistent with the curve, not
   tested further.* L = 100 cannot see a whole previous beat most of the
   time and is worse in every seed; L = 500 is better than L = 250 in every
   seed, by less (−0.099 bits). Caveat (D19 note): at equal epochs L = 500
   sees twice as many samples as L = 250.
3. **Positional encoding matters for generation** (+0.385 bits without it,
   the largest harm after V = 64), unlike the classifier of chat DL 07, where
   removing it helped slightly (D18). A causal model without positions can
   only infer how far back something was from the content; the curve of
   `no_pos` only decreases slowly and has no drop at one beat, i.e. it does
   not seem to use the previous beat. *The explanation is a hypothesis.*
4. **LSTM vs causal Transformer.** With the same size, data and budget the
   LSTM codes the signal with 0.090 fewer bits (3/3 seeds) and its curve is
   below the Transformer's from about 0.03 s of context on. Both were
   still improving at epoch 50 (D19 note), so this is a comparison at equal
   budget, not between converged models. *Why the LSTM is better here is not
   established* (possible reasons: the recurrence suits a smooth, locally
   predictable signal; the Transformer has 2 blocks only).
5. **Vocabulary.** Finer tokens cost fewer bits per sample: 1 µV levels
   (−0.284) < 10 µV < 40 µV (+0.597). The bits metric assumes that all
   values inside a token are equally likely; with 40 µV tokens that
   assumption wastes bits, because the next sample is often within a few µV
   of the current one on flat segments. The cross-entropy alone would have
   ranked V = 64 first (0.899 nats) and V = 2560 last (3.978): this is why
   configurations are compared in bits, not in CE or perplexity.
6. **Without the causal mask the model reads the answer.** Its
   teacher-forced CE falls to about 0.01 nats within 3-4 epochs
   (`generative_curves` c), but at the last position of the window, the only
   one whose target is not in the input, it is worse than the baseline
   (2.422 vs 2.062 nats), and its generated signals are the worst of the
   grid at T = 1.0 (HR error 30.3 bpm, W1 0.054 mV). The mask is what makes
   teacher-forced training honest while still being parallel (Part II slides
   13-14).
7. **Sampling** (`generative_sampling`, `generative_samples`). Greedy search
   collapses to a flat line in every record (flat share 1.000 for both
   models): the most probable next value on a baseline is "the same value",
   the ECG version of the notebook's "I don't know. I don't know..." loops.
   Low temperature (0.5) is still mostly flat; T = 0.8-1.0 gives amplitude
   distributions close to the real ones (std ratio 0.8-1.1, W1 0.010-0.028
   mV, against 0.004 mV between two sets of real records) with beats at roughly the right rate; T = 1.5 gives noise
   (std ratio 2.8, HR error about 49 bpm).
8. **Top-k depends on the vocabulary.** With V = 2560, k = 10 keeps only
   values within about 10 µV of the most probable one: generation becomes
   almost greedy (flat share 0.242, HR found in 8 % of records). With the
   LSTM's V = 256, k = 10 spans about 100 µV and works (HR error 14.7 bpm).
   A top-k value is a statement about the token size.
9. **Generated beats are plausible but do not follow the prompt's rhythm
   exactly**: at T = 1.0 the heart-rate error is 12-20 bpm for most
   configurations, best with L = 500 (11.8 bpm), worst without positions
   (21.2) and without the mask (30.3).
10. **Attention** (`generative_attention`, V = 2560 model, one NORM window):
    layer 1 attends almost only to the last few samples (a local filter);
    in layer 2 some heads show bands parallel to the diagonal at a distance
    of about one beat, i.e. the position predicting a sample looks at the
    same phase of the previous beat. *Visual observation on one record,
    without a number that measures it.*

## Caveats to state in the reports

- **Budget-limited training**: almost every run stopped at the 50-epoch
  limit with its best epoch at or near 50; the remaining slope (about
  0.005-0.02 bits over the last 10 epochs) is much smaller than the
  differences reported, but no model converged (D19 note).
- **The last-position cross-entropy mixes in a record-end effect.** The
  cost per position rises sharply at the last position for every model.
  In the validation windows the last position is the last sample of the
  record in 1 of 4 windows (L = 250), 1 of 2 (L = 500) and 1 of 10
  (L = 100), and training windows almost never end there. Solving
  0.25 X + 0.75 × 5.9 = 6.29 (L = 250) and 0.5 X + 0.5 × 5.85 = 6.67
  (L = 500) gives the same X ≈ 7.5 bits for the record-end target, and the
  last sample-to-sample jumps of 20 raw records are larger than the
  record's median (27-30 µV vs 12 µV) *(assistant's check, not in the
  logs)*. *Hypothesis, not tested directly.* Consequence: the last-position
  CE is valid for comparing `no_mask` with the baseline (same windows), not
  across context lengths. Its effect on the overall bits is negligible (one
  target in 999).
- **Selection**: the pre-trained model is `vocab2560`, chosen by the D19
  rule on validation. It is also the worst generator with top-k 10 and low
  temperature (point 8): the rule selects the best *density model*, which
  is what the transfer of chat DL 08 needs, not the best sampler.
- The heart-rate detector is a simple threshold rule from the real prompt,
  not a validated QRS detector; it finds a rate in 99.2 % of the real
  continuations.
- The 2 s prompt plus 8 s generated use the first 10 s of each record only;
  nothing is generated without a prompt.

## PyTorch details met in this block

- **`CrossEntropyLoss` with sequences**: logits (B, T, V) must be flattened
  to (B·T, V) and targets to (B·T,), otherwise PyTorch reads T as the class
  axis (`NextTokenTask.loss`, written in chat DL 03).
- **`log_softmax` + `gather`** gives the cross-entropy of every position
  separately (needed for the cost-vs-context curve), which
  `F.cross_entropy(reduction="mean")` hides.
- **Causal mask by `masked_fill(~mask, -inf)` before the softmax**: the
  weights above the diagonal are exactly 0 (test
  `test_causal_attention_maps_are_lower_triangular`).
- **Buffers vs parameters**: the sinusoidal encoding is a non-persistent
  buffer (moves to the GPU, not trained, not in the checkpoint).
- **Generation with `torch.no_grad()`, `model.eval()` and a seeded
  `torch.Generator` on the GPU** (`torch.multinomial(..., generator=...)`);
  only the last position goes through the output layer (`next_logits`).
- **GPU memory of attention grows with L²**: generating 64 windows of 500
  tokens failed on the 4 GB GPU; batches now shrink as (250 / L)².
- **Random crops in a `Dataset`**: `__getitem__` draws the window start from
  a NumPy generator seeded with the run's seed; with `num_workers=0` the
  draws are reproducible.

## Problems met

- `report` out of GPU memory with L = 500 (fixed, D19 note of 2026-11-22).
- `ctx500` seed 44: the laptop slept for hours during three epochs; results
  unaffected, epoch time reported as a median.
- In `--fast` mode the example records of D16-D18 were not in the 256-record
  subset, so two figures were skipped; fixed by drawing the examples from
  the records in use with the same rule.
- Copying files to the laptop: one copy wrote the previous version of three
  files; detected by the assistant by comparing file sizes, copied again
  (logged in `AI_USAGE_LOG.md`).

## Open points

- **O5**: the weekly summary of 20/11 and its "simple autoregressive
  LLM-style model" notebook are still not in `contexto_DL/`; when they are,
  compare with this block.
- **[verify] for chat DL 09**: GPT-style decoder-only language models
  (Radford et al. 2018/2019); top-k sampling (Fan et al. 2018, arXiv
  1805.04833, cited in the class notebook); nucleus sampling (Holtzman et
  al. 2020); mu-law companding (WaveNet, van den Oord et al. 2016);
  causal Transformers without positional encoding (Haviv et al. 2022);
  autoregressive models of ECG or other physiological signals.
- Not done (possible extension): training until convergence; generation
  without a prompt; a validated QRS detector.

## For the next chats

- **Chat DL 08 (transfer, T13)**: pre-trained model = `vocab2560`
  (`generative_summary.json` → `pretrained_for_chat_08`: model arguments,
  checkpoints of seeds 42, 43, 44, tokenizer `lo_uv` −1138, `delta_uv` 1,
  `vocab` 2560). Load with
  `build_generator(args)` + `load_state_dict(torch.load(ckpt))`; tokenize
  lead II of every split with `make_tokenizer(lo_uv, vocab)` (fitted on the
  training folds only, so using it on fold 10 is not leakage);
  `model.record_features(tokens)` returns one 128-vector per record (mean
  over the 4 windows of 250 samples) and lets gradients through, for both
  the linear probe (freeze the model) and fine-tuning. The other
  checkpoints (e.g. the LSTM, second-best family) are available if chat 08
  wants a comparison.
- **Chats 10 and 11 (reports)**: area 3 of the professor's message. Tables
  `generative_ablations.tex` (wide: `\resizebox{\textwidth}{!}{...}`) and
  `generative_sampling.tex`; figures listed above. Say that training was
  budget-limited, explain bits vs cross-entropy, and mark as hypotheses
  points 2, 3, 4 and 10 and the record-end effect.
