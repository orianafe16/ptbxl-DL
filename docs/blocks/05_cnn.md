# Block 05 — 1D convolutional network (T10)

Chat DL 05, 2026-11-20. T10 done by OF, reviewed by MM. Everything
executed by OF on Oriana's laptop (RTX 3050 Ti): runs at commit `3f9439f`,
summary, tables and figures at commit `7eaf6ec` (all logs with schema 2 and
`git_dirty: false`). Decisions: `docs/DECISIONS.md` D16 (this block, two
entries), D3, D4 and D14.

Numbers below are copied from `report/numbers.tex` (3 decimals) only to
make this note readable. **Reports must use the macros** (`\cnn...{}`,
`\dense...{}`, `\trivial...{}`), never these figures. Values marked
*(seed-42 log)* come from the per-epoch history of one run and have no
macro.

## What the block delivers

| Item | File |
|---|---|
| Model | `src/models/cnn.py` (`CNN1D`, `ConvBlock`; registered as `cnn`) |
| Ablations, final model, summary, tables, figures | `src/cnn_experiments.py` (generic helpers imported from `src/dense_experiments.py`) |
| Tests (model, receptive field, CAM, hooks, grid, selection, paired comparisons, whole pipeline on synthetic data) | `tests/test_cnn.py` (44 tests) |
| 30 ablation runs + 3 final runs | `results/logs/cnn_*.json`, `results/predictions/cnn_*.npz` |
| Summary of the block (alias `cnn` in `report/runs.json`, 337 macros) | `results/logs/cnn_summary.json` |
| Ablation table, final table | `report/tables/cnn_ablations.tex`, `report/tables/cnn_final.tex` |
| Figures | `report/figures/cnn_curves`, `cnn_ablations`, `cnn_first_layer`, `cnn_feature_maps` (PDF, PNG) |

```
python -m src.cnn_experiments all --fast   # smoke test
python -m src.cnn_experiments all          # run (30) + final (3) + report
python report/make_numbers.py
```

Run time on the laptop: 47 min of training for the 30 ablations, about
5 min for the 3 final runs; 3.8 to 8.9 s per epoch depending on the
configuration (`\cnnEpochTimeMean...{}`), against 1.4 s for the Dense
baseline.

## Architecture

Input (batch, 12, 1000): the 12 leads are the input channels, time is the
axis the filters slide along. Four blocks, each

    Conv1d(k) -> BN -> ReLU -> Conv1d(k) -> BN  (+ shortcut)  -> ReLU -> MaxPool1d(2)

with 32, 64, 128, 128 channels and kernel k = 7 (70 ms at 100 Hz); the
shortcut is the identity in the last block and a 1×1 convolution + BN
where the number of channels changes (CNN slides 55-56). Then global
average pooling over time, dropout 0.5 and a Linear layer with 5 logits.
`\cnnParamsBase{}` = 468,325 parameters, about 1/7 of the Dense baseline.

Why it beats the Dense network (the point of the block):
- **Weight sharing.** A first-layer filter has 12 × 7 weights and is
  applied at every position, so a QRS complex is detected wherever it
  appears. The Dense network had one weight per (lead, sample) and had to
  learn the same shape at every position (block note 04).
- **Global average pooling.** Each of the 128 feature maps is averaged
  over time, so the classifier sees how much of each pattern the record
  contains, not where it is.
- **Receptive field.** One position of the last feature map depends on
  `\cnnRfSamplesBase{}` = 196 input samples (about 2 s, two or three
  beats); `CNN1D.receptive_field()` computes it and a test measures it
  from the gradient of a linearised copy of the network.

## Results

Validation macro AUROC, mean ± SD over seeds 42-44 (`cnn_ablations.tex`).
Δ = mean paired difference with the reference of the same seed (the
baseline, or 6 blocks for "6 blocks, no residual"); better / worse = seeds
whose paired 95 % interval lies wholly above / below 0. Training loss
gap = training loss at epoch 5 minus the reference's, mean over seeds
(seeds where it is higher). RF = receptive field in samples.

| Configuration | Parameters | RF | Val. macro AUROC | Δ | better / worse | Train. loss gap, epoch 5 |
|---|---|---|---|---|---|---|
| Baseline | 468,325 | 196 | 0.928 ± 0.003 | — | — | — |
| Kernel 3 | 208,741 | 76 | 0.927 ± 0.004 | −0.001 | 0 / 0 | +0.004 (3 of 3) |
| Kernel 15 | 987,493 | 436 | 0.930 ± 0.001 | +0.001 | 1 / 0 | +0.003 (3) |
| 2 blocks | 56,485 | 40 | 0.928 ± 0.002 | −0.000 | 1 / 0 | +0.040 (3) |
| 6 blocks | 928,613 | 820 | 0.926 ± 0.004 | −0.002 | 0 / 1 | +0.003 (3) |
| No batch norm | 466,469 | 196 | 0.927 ± 0.001 | −0.002 | 0 / 2 | +0.008 (3) |
| No residual connections | 457,029 | 196 | 0.931 ± 0.001 | +0.002 | 1 / 0 | +0.003 (3) |
| 6 blocks, no residual (vs. 6 blocks) | 917,317 | 820 | 0.927 ± 0.002 | +0.001 | 1 / 0 | +0.009 (3) |
| Flatten instead of GAP | 507,365 | 196 | 0.925 ± 0.001 | −0.003 | 0 / 2 | −0.004 (0) |
| No dropout | 468,325 | 196 | 0.932 ± 0.002 | +0.003 | 1 / 0 | −0.019 (0) |

Final model (selected on validation: `no_dropout`), test fold
(`cnn_final.tex`): macro AUROC 0.928 ± 0.002 over the three seeds; seed
42: 0.928 [0.921, 0.935]; macro F1 0.748 ± 0.006. Per class (mean over
seeds; Dense final in brackets): NORM 0.946 (0.801), MI 0.926 (0.691),
STTC 0.937 (0.747), CD 0.920 (0.691), HYP 0.909 (0.790). Paired
difference on test with the final Dense model of the same seed: +0.182
[0.170, 0.196] for seed 42, interval excluding 0 for all three seeds
(mean `\cnnDiffDenseMean{}` = 0.184); with the trivial model: mean
`\cnnDiffTrivialMean{}` = 0.209, excluding 0 for all seeds.

## What each ablation shows

1. **Convolutions are the change that matters.** The CNN gains about
   0.18 of macro AUROC over the Dense network on test, in every seed and
   in every class, with 1/7 of its parameters. In the Dense block the
   largest ablation effect was about 0.06 (one hidden layer, −0.063); here a change of
   architecture family moves it three times as much. This is the expected
   effect of weight sharing and pooling (Architecture, above).
2. **Inside the CNN, no factor matters much.** All ten configurations lie
   between 0.925 and 0.932 on validation, and no mean difference with
   the reference exceeds 0.003. The network is robust to these choices; the
   report should say so rather than over-interpret differences of 0.001-
   0.003. The only factors worse in two of three seeds are flatten
   (−0.003) and no batch norm (−0.002).
3. **Global average pooling vs flatten.** Flatten gives the classifier one
   weight per (channel, position), so it loses the translation invariance
   of the head; it is slightly worse and fits the training set faster
   (training loss gap −0.004) — the same "whole-record pattern" problem as
   the Dense network, on a much smaller scale because the convolutions
   before it are still shared.
4. **Batch norm matters much less than in the Dense network** (−0.002
   here, −0.053 there). Without it the CNN learns more slowly (training
   loss +0.008 at epoch 5, best epoch 11.7 vs 8.7 on average) but reaches
   almost the same AUROC. Our explanation, not tested: the CNN has fewer
   parameters, its convolutions see small local windows of an already
   z-scored input, and Adam adapts the step per parameter.
5. **Skip connections help optimisation, not the result.** Without
   shortcuts the training loss at epoch 5 is higher in all three seeds,
   and the gap is larger with 6 blocks (+0.009) than with 4 (+0.003),
   which is the argument of the CNN slides (49-51: deeper plain networks
   are harder to optimise). But the validation AUROC does not improve
   with them (+0.002 without them at 4 blocks, +0.001 at 6 blocks).
   With 8 or 12 convolutional layers and early stopping around epoch 10,
   the networks are too shallow for the optimisation difference to show
   in the result; the degradation in the slides appears with much deeper
   networks (**[verify]**: depths in He et al. 2016).
6. **Kernel size and depth: wide range, same result.** Receptive fields
   from 40 to 820 samples (0.4 to 8.2 s) all reach about 0.93. Even 2
   blocks (56,485 parameters, receptive field 0.4 s, less than one beat)
   matches the baseline, although it needs twice as many epochs and fits
   the training set more slowly (training loss +0.040 at epoch 5). For
   these five superclasses, local beat-level patterns averaged over the
   record seem to be enough; patterns spanning several beats add little.
   This is an interpretation, not a tested claim.
7. **Dropout before the classifier does not help.** No dropout is the
   best mean (+0.003, one seed of three significant). With global average
   pooling the classifier has only 645 parameters, so there is little to
   regularise, unlike the Dense network where dropout was the main
   defence (block note 04, item 2).
8. **Validation loss and AUROC disagree again, less than in the Dense
   network** (figure `cnn_curves`, panel a): baseline seed 42, lowest
   validation loss at epoch 8 and best AUROC at epoch 17, both with
   validation loss 0.272 *(seed-42 log)*.
9. **First-layer filters** (figure `cnn_first_layer`): the weights do not
   look like a heartbeat template, but their frequency responses form a
   bank of mostly low-frequency filters: median peak
   `\cnnFirstLayerPeakHzMedian{}` = 8.6 Hz, 11 of 32 peaking below 5 Hz
   and only 3 above 20 Hz. That most ECG energy lies at low frequencies
   is **[verify]** with chat 09; the figure only shows what the network
   kept.
10. **Feature maps and class activation maps** (figure
    `cnn_feature_maps`): in all five records the channels of block 1
    fire at every QRS complex, at the same moment as the beat: the same
    filter applied at every position. The last block still shows bands
    aligned with the beats. The class activation map (CAM) of the
    record's class must be read with the prediction, now printed in each
    column title (`\cnnExampleProb...{}`, `\cnnExampleThreshold...{}`):
    the MI (0.65 > 0.33), CD (0.99 > 0.66) and HYP (0.42 > 0.15) examples
    are classified correctly; the CD map is positive (orange) almost
    throughout, the MI and HYP maps alternate positive and negative
    stretches. The NORM (0.32 ≤ 0.62) and STTC (0.12 ≤ 0.33) examples are
    missed; the STTC map is negative (blue) almost throughout and the NORM
    map is mixed. The CAM shows where
    the evidence for or against a class lies, also when the model is
    wrong. The CAM idea is from Zhou et al. 2016 (**[verify]**).

## Selection of the final model: a caveat to state in the reports

As in the Dense block, the rule fixed in D16 before the runs (highest mean
validation macro AUROC) selected `no_dropout` (0.932 vs 0.928 for the
baseline), but the paired interval with the baseline excludes 0 in only
one seed of three. We keep the rule's choice (changing it after seeing the
results would be a post-hoc choice) and say so: the final CNN is the
baseline without dropout, same size and training.

## PyTorch details met in this block

- **Conv1d expects (batch, channels, time).** PTB-XL is stored as
  (records, 12, 1000), which is already this layout, so no `permute` is
  needed; a (batch, time, channels) tensor would silently treat the 1000
  samples as channels.
- **"Same" padding needs an odd kernel.** `padding=k // 2` keeps the
  length only for odd k; the model refuses even kernels, because the main
  path and the shortcut must have the same length to be added.
- **The skip connection is a plain `+`.** `out = out + self.shortcut(x)`
  adds no parameter with the identity shortcut; autograd sends the
  gradient through both branches.
- **Forward hooks** (`register_forward_hook`) read the output of a block
  without changing `forward`; they stay attached until removed, so
  `capture_maps` removes them in a `finally` block (tested).
- **Class activation map = logit decomposed over time.** With global
  average pooling and one Linear layer, `logit_c = b_c + mean_t
  CAM_c(t)` in eval mode (tested to 1e-5); in training mode dropout would
  break the identity.
- **`model.eval()` for the figures**: batch norm with running statistics,
  dropout off; the probabilities in the figure titles are equal to the
  saved validation predictions of the run.

## Problems met

- None during the runs. While writing the code, the first version of the
  interpretation figures had overlapping titles and called `tight_layout`
  on top of the project's constrained layout; both were fixed and checked
  on synthetic data before delivery (not an error found by OF or MM).
- After the full runs the assistant noticed that two of the five example
  records are misclassified, which makes their CAM negative; the
  probabilities and thresholds were added to the figure and the summary
  (D16, second entry). No run was repeated; the 290 earlier numbers of
  the summary came out identical.
- Coincidence checked: `flatten` and `no_dropout` have the same best
  epochs (12, 6, 10) in the three seeds; their logs, configurations and
  curves are different.

## For the next chats

- **Chat 08:** best CNN = `no_dropout` (model args in the final logs:
  channels 32-64-128-128, kernel 7, batch norm, residual, GAP, dropout 0;
  checkpoints `results/checkpoints/cnn_no_dropout_s4{2,3,4}_20261120-222{330,553,737}.pt`,
  not committed). For the single-lead ablation build it with
  `CNN1D(n_leads=1, **args)` (tested). It is the strongest model so far
  and the reference for the RNN and Transformer blocks.
- **Chat 06:** the "convolutional front-end" ablation can reuse
  `ConvBlock` from `src/models/cnn.py`.
- **Chats 10 and 11:** `\input{tables/cnn_ablations.tex}` and
  `\input{tables/cnn_final.tex}` (need `booktabs` and `numbers.tex`; the
  final table also uses `\dense...` and `\trivial...` macros); figures
  `cnn_curves`, `cnn_ablations`, `cnn_first_layer`, `cnn_feature_maps`.
  State the selection caveat, and that the differences between CNN
  configurations are small (item 2). Values with *(seed-42 log)* have no
  macro.
- **Chat 09, [verify]:** QRS width of about 80-100 ms (choice of kernel
  7); the basic block of He et al. 2016 and the depths at which plain
  networks degrade; class activation maps (Zhou et al. 2016); where the
  ECG energy lies in frequency (item 9).
