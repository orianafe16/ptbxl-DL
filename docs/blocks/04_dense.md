# Block 04 — Dense network (T9)

Chat DL 04, 2026-11-20. T9 done by OF, reviewed by MM. Everything
executed by OF on Oriana's laptop (RTX 3050 Ti), code at commit `fcc3c3b`.
Decisions: `docs/DECISIONS.md` D15 (this block), D3, D4 and D14.

Numbers below are copied from `report/numbers.tex` (3 decimals) only to
make this note readable. **Reports must use the macros** (`\dense...{}`,
`\trivial...{}`), never these figures. Values marked *(seed-42 log)* come
from the per-epoch history of one run and have no macro yet.

## What the block delivers

| Item | File |
|---|---|
| Model | `src/models/dense.py` (`DenseNet`, registered as `dense`) |
| Ablations, final model, summary, tables, figures | `src/dense_experiments.py` |
| Tests (model, grid, selection rule, whole pipeline on synthetic data) | `tests/test_dense.py` |
| 33 ablation runs + 3 final runs | `results/logs/dense_*.json`, `results/predictions/dense_*.npz` |
| Summary of the block (alias `dense` in `report/runs.json`) | `results/logs/dense_summary.json` |
| Ablation table, final table | `report/tables/dense_ablations.tex`, `report/tables/dense_final.tex` |
| Figures | `report/figures/dense_curves`, `dense_ablations`, `dense_first_layer` (PDF, PNG) |

```
python -m src.dense_experiments all --fast   # smoke test
python -m src.dense_experiments run          # 11 configurations x 3 seeds
python -m src.dense_experiments final        # selected configuration, test
python -m src.dense_experiments report       # summary, tables, figures
python report/make_numbers.py
```

## Architecture

The 12 x 1000 signal is flattened into 12,000 numbers (lead after lead) and
goes through hidden blocks Linear → BatchNorm1d → ReLU → Dropout, then a
Linear layer with 5 logits (no sigmoid: `BCEWithLogitsLoss` contains it).
Baseline: 2 hidden layers of 256 units, batch norm, dropout 0.5, per-lead
z-score of the input; `\denseParamsBase{}` = 3,140,357 parameters, 98 % of
them in the first Linear layer (12,000 x 256 weights).

Why the first layer matters: each hidden unit has one weight per (lead,
time sample), i.e. it compares the whole 10-second record with a single
12 x 1000 pattern. A QRS complex that appears 0.3 s later in another record
meets different weights, so the network must learn the same shape at every
position separately. This is the limitation of feed-forward networks that
convolutions remove (weekly summary 30/10).

## Results

Validation macro AUROC, mean ± SD over seeds 42-44
(`dense_ablations.tex`; Δ = mean paired difference with the baseline of the
same seed; "better / worse" = seeds whose paired 95 % interval lies wholly
above / below 0):

| Configuration | Val. macro AUROC | Δ | better / worse |
|---|---|---|---|
| Baseline | 0.757 ± 0.002 | — | — |
| No input z-score (raw mV) | 0.760 ± 0.002 | +0.003 | 0 / 0 |
| No batch norm | 0.704 ± 0.003 | −0.053 | 0 / 3 |
| No z-score and no batch norm | 0.719 ± 0.005 | −0.038 | 0 / 3 |
| No dropout | 0.725 ± 0.007 | −0.032 | 0 / 3 |
| Dropout 0.2 | 0.738 ± 0.001 | −0.019 | 0 / 3 |
| 1 hidden layer | 0.694 ± 0.003 | −0.063 | 0 / 3 |
| 3 hidden layers | 0.755 ± 0.001 | −0.002 | 0 / 0 |
| 4 hidden layers | 0.748 ± 0.005 | −0.009 | 0 / 1 |
| Width 64 | 0.743 ± 0.002 | −0.014 | 0 / 3 |
| Width 1024 (13.3 M parameters) | 0.753 ± 0.005 | −0.004 | 0 / 0 |

Final model (selected on validation: `no_norm`), test fold
(`dense_final.tex`): macro AUROC 0.744 ± 0.002 over the three seeds; seed
42: 0.746 [0.732, 0.759]; macro F1 0.511 ± 0.003. Per class (mean over
seeds): NORM 0.801, MI 0.691, STTC 0.747, CD 0.691, HYP 0.790. Paired
difference with the trivial model on test: +0.028 [0.013, 0.043] for seed
42, and the interval excludes 0 for all three seeds
(`\denseDiffTrivialSFourTwo{}` etc.).

## What each ablation shows

1. **The Dense network is only a little better than the trivial model.**
   The trivial model (logistic regression on the mean and standard
   deviation of each lead) reaches validation 0.741
   (`\denseTrivialValMacroAuroc{}`); the Dense baseline 0.757. On test the
   gain is +0.028 and significant, but the point estimates per class are
   lower than the trivial model's for MI (0.691 vs 0.727) and CD (0.691 vs
   0.712); we did not compute paired intervals per class, so the report
   should present this as an observation, not as a tested difference. This
   is the expected weakness of the architecture (whole-record patterns, no
   translation invariance) and the motivation of the CNN block.
2. **Dropout is the main defence against overfitting.** Without dropout
   the best epoch is 3 (mean over seeds `\denseBestEpochMeanNoDropout{}`):
   the training loss keeps falling while the validation loss rises from
   0.50 to 0.93 in 8 epochs *(seed-42 log)*. Dropout 0.2 sits in between
   (−0.019) and 0.5 is best. With 3.1 M parameters and 17,084 training
   records, the network can memorise the training set; dropout trains a
   different thinned network at every batch (FeedForward slides 36-41).
3. **Batch norm is essential here** (−0.053 without it in all three
   seeds), and it **absorbs the input scaling**: without batch norm the
   input scale matters, with it raw mV and z-score are indistinguishable
   (+0.003, no seed significant). Batch norm after the first Linear layer
   re-standardises its outputs whatever the scale of the input. A finding
   we did not expect: without batch norm, raw mV did better than z-score
   (−0.038 vs −0.053 from the baseline). Our explanation, not tested: the
   ECG in mV is already on a small, similar scale (training standard
   deviation per lead between 0.138 and 0.334 mV, logged in `data.lead_std`),
   unlike 0-255 image pixels, so the z-score multiplies the input by about
   3 to 7 and makes the first-layer pre-activations larger.
4. **Depth: one hidden layer is clearly not enough** (−0.063); two and
   three layers are equivalent; four layers are slightly worse (one seed
   of three significant). More layers add little because 98 % of the
   parameters are in the first layer anyway.
5. **Width: 64 units are too few** (−0.014, all seeds); 1024 units (4x the
   parameters) give nothing more and take about twice as long per epoch
   (`\denseEpochTimeMeanWidthOneZeroTwoFour{}` s vs
   `\denseEpochTimeMeanBase{}` s).
6. **Validation loss and AUROC disagree** (figure `dense_curves`, panel a):
   in the baseline the validation loss is lowest at epoch 7 (0.486) and
   then rises, while the validation AUROC keeps improving until epoch 19
   (loss 0.577 there) *(seed-42 log)*. The network keeps ranking the
   records better while its probabilities become over-confident. Early
   stopping follows the AUROC (D14), so it stops much later than a
   loss-based rule would.
7. **First-layer weights** (figure `dense_first_layer`, final model, the
   three units with the largest weight norm): they look like noise along
   time, with no recognisable heartbeat template, but they show vertical
   bands in which several limb leads take related or opposite signs at the
   same instant. A plausible reading, **[verify]** with chat 09: the limb
   leads are linear combinations of one another, so the network weighs
   them together. What the figure clearly shows is the point of item 1:
   the unit has one weight per instant, with nothing that repeats from one
   beat to the next.

## Selection of the final model: a caveat to state in the reports

D15 fixed the rule before the runs: the configuration with the highest
mean validation macro AUROC. It selected `no_norm` (0.760 vs 0.757 for the
baseline), but the paired interval with the baseline contains 0 for every
seed, so the two are statistically indistinguishable. We keep the rule's
choice (changing it after seeing the results would be a post-hoc choice)
and say so explicitly. In practice the final model is the baseline without
the input z-score: same size, same training.

## PyTorch details met in this block

- **Training loss is logged in training mode.** `train_one_epoch` averages
  the loss with dropout active, so it is higher than the loss of the same
  weights in `model.eval()`; this is why, in the first epochs, the
  training curve can lie above the validation curve (FeedForward slide 39).
- **`model.eval()` changes batch norm too**: in training it normalises with
  the statistics of the current batch; in evaluation with running averages
  collected during training. Forgetting `model.eval()` would make
  predictions depend on which records share the batch.
- **`BatchNorm1d` needs more than one record per batch in training.** With
  17,084 training records and batches of 64 the last batch has 60 records;
  a split whose size left a last batch of 1 would stop training with
  "Expected more than 1 value per channel". Not a problem here; worth
  knowing for the 10 % label fraction of chat 08.
- **Flatten order**: `nn.Flatten` on (batch, 12, 1000) puts the 1000
  samples of lead I first, then lead II, etc.; `DenseNet.first_layer`'s
  weight reshaped to (units, 12, 1000) gives back the per-lead patterns.

## Problems met

- **`report --fast` stopped with "differ in test_ecg_id"** (assistant's
  error, found by the assistant when the `_fast` summary was missing): the
  `--fast` final runs see 256 test records, the trivial run the whole
  fold, so they cannot be paired. Fixed on 2026-11-20 (no trivial
  comparison with `--fast`) with a regression test; the full runs were not
  affected.
- **`git_dirty: true` in every Dense log, although the code was
  committed.** The files of commit `fcc3c3b` are identical to the ones that
  ran (git blob hashes of the index compared with the files by the
  assistant), so the logs describe committed code. Cause, found from
  `git status --short` (OF, 20:00): `run_experiment` saves the run's own
  `results/predictions/<run_id>.npz` before calling
  `runlog.environment()`, and `git status --porcelain` lists untracked
  files, so every full run sees at least its own new file and is marked
  "dirty". The same holds for the trivial run of chat 03. Fixed afterwards
  in chat 03 (2026-11-20, DECISIONS D14, AI Error Bounty B5): the flag now
  looks only at the code under `src/` (`git status --porcelain
  --untracked-files=all -- src`), and logs carry `"schema": 2`. The Dense
  logs keep schema 1, where `git_dirty` is always true and says nothing.

## For the next chats

- **Chat 05 (CNN):** the Dense block is the reference to beat; keep batch
  norm and dropout, which matter most here. `src/dense_experiments.py` can
  be copied as the template for the CNN ablation script (grid, three seeds,
  paired differences, macro-only tables).
- **Chat 08:** best Dense model = `no_norm` (checkpoints
  `results/checkpoints/dense_no_norm_s4{2,3,4}_20261120-193*.pt`, not
  committed); it is the Dense entry of the 12-lead vs single-lead ablation.
- **Chats 10 and 11:** `\input{tables/dense_ablations.tex}` and
  `\input{tables/dense_final.tex}` (both need `booktabs` and
  `numbers.tex`); figures `dense_curves`, `dense_ablations`,
  `dense_first_layer`. State the selection caveat above.
- **Chat 09, [verify]:** dropout 0.5 as the usual rate (Srivastava et al.
  2014); batch norm before the activation in Ioffe and Szegedy 2015; limb
  leads as linear combinations of each other (Einthoven, Goldberger).
