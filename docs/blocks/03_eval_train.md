# Block 03 — Evaluation module and shared training loop (T7, T8)

Chat DL 03, 2026-11-20. T7 done by MM, reviewed by OF. T8 done by OF and
MM. Everything executed on Oriana's laptop.
Decisions: `docs/DECISIONS.md` D4 (entry of 2026-11-20) and D14.

## What the block delivers

| Item | File | Command |
|---|---|---|
| Metrics: AUROC, F1 thresholds, patient bootstrap, paired difference, perplexity | `src/metrics.py` | — |
| Tasks: 5-superclass classification, next-token prediction | `src/tasks.py` | — |
| Training loop, early stopping, seeds, PTB-XL command line | `src/train.py` | `python -m src.train --model <name> [--fast] [--final]` |
| One JSON log per run (with the `git_dirty` flag on `src/`) | `src/runlog.py` | — |
| Paired comparison of two runs | `src/compare.py` | `python -m src.compare RUN_A RUN_B [--split test]` |
| Model registry and trivial model | `src/models/` | — |
| LaTeX numbers from the logs | `report/make_numbers.py`, `report/runs.json` | `python report/make_numbers.py` |
| Tests | `tests/test_metrics.py`, `tests/test_train.py`, `tests/test_numbers.py` | `python -m pytest -q tests` |
| Output paths | `src/config.py` (`LOGS_DIR`, `PREDICTIONS_DIR`, `CHECKPOINT_DIR`, `FIGURES_DIR`) | — |

## How it works, step by step

1. **The loop does not know the problem.** `train_one_epoch` runs the
   five steps of every iteration (zero_grad, forward, loss, backward,
   step) with `model.train()`; `evaluate` runs with `model.eval()` and
   `torch.no_grad()`. The loss, the metrics and the metric that decides
   early stopping come from a *task* object. The same code therefore
   trains the classifiers and the autoregressive generator of D13.
2. **Exact means.** Each batch loss is multiplied by its number of terms
   (labels or tokens) and divided by the total at the end, so the epoch
   loss is the mean over the whole split even when the last batch is
   smaller. For the generator this gives the correct perplexity:
   exp(mean cross-entropy per token).
3. **Early stopping.** After every epoch the model is evaluated on
   validation. When the selection metric improves (macro AUROC for the
   classifiers, cross-entropy for the generator) a CPU copy of the weights
   is kept; after 5 epochs without improvement training stops and the
   best weights are loaded back.
4. **Seeds.** `random`, NumPy and PyTorch are seeded before the model is
   built, so the initial weights depend only on the seed; the same seed
   orders the training batches. cuDNN runs in deterministic mode.
5. **Final evaluation.** With the best weights: probabilities =
   sigmoid(logits), computed outside the model; per-class AUROC and its
   mean; F1 thresholds chosen on validation (grid 0.01-0.99) and applied
   unchanged to test; bootstrap intervals with the **patient** as the
   resampling unit, because records of the same patient are correlated.
   The test fold is evaluated only with `--final`, and a second final run
   with the same model, tag and seed is refused (D3).
6. **Outputs of a run** `<model>_<tag>_s<seed>_<date-time>`:
   `results/logs/<run_id>.json` (scalars, configuration, data description,
   curves, results, software versions, git commit and whether there were
   uncommitted changes in `src/`), `results/predictions/<run_id>.npz`
   (probabilities, labels, patient and ECG ids per split, used by
   `src/compare.py`) and `results/checkpoints/<run_id>.pt` (not committed).
   `--fast` runs end in `_fast` and are ignored by git and by
   `make_numbers.py`.
7. **Numbers for the reports.** `report/runs.json` lists the logs that may
   appear in a report, each under a short alias. `make_numbers.py` turns
   every key of their `scalars` into a macro: `n_with_1_labels` of alias
   `eda` becomes `\edaNWithOneLabels`; the key is cut at underscores and
   letter/digit boundaries and digits are spelled out, because LaTeX macro
   names accept letters only. Use them with braces: `\edaNRecordsAll{}`.

## How other chats use it

A new model (chats 04-07): add the class to `src/models/`, register it in
`src/models/__init__.py` (`MODELS["dense"] = DenseNet`) and run
`python -m src.train --model dense --tag base`. The model returns logits
(no sigmoid). For ablations with other constructor arguments, a short
script can call the loop directly:

```python
from src.train import TrainConfig, ptbxl_setup, run_experiment
from src.models.dense import DenseNet

cfg = TrainConfig(model="dense", tag="no_dropout", model_args={"dropout": 0.0})
loaders, info, task = ptbxl_setup(cfg)
run_experiment(lambda: DenseNet(**cfg.model_args), loaders, task, cfg, info)
```

Generator (chat 13): build loaders that yield `(tokens_in, tokens_target)`
as integer tensors of shape (batch, time), with the target shifted by one
step; the model returns (batch, time, vocab) logits without softmax. Then
`run_experiment(build_model, loaders, NextTokenTask(vocab_size), cfg, info)`
with `info` describing the tokenisation. Do not pass `--final`/`final=True`:
D13 keeps fold 10 out of the generation experiments.

Transfer learning (chat 08): set `requires_grad = False` on the frozen
layers before calling `run_experiment`; the optimiser only receives
trainable parameters and the log records both counts. Record the label
fraction and the pre-trained checkpoint in `info` or in `cfg.notes`.

Comparisons (all model chats): `python -m src.compare RUN_A RUN_B` gives
the interval of the difference in macro AUROC on the same patient
resamples; if it contains 0, neither model is better (D4). Its log can be
listed in `runs.json` like any other run.

## Checks run on Oriana's laptop (2026-11-20)

- `python -m pytest -q tests`: 44 passed (the 8 data tests, including the
  4 on the real cache, and 36 new ones). The new tests cover toy cases
  with known answers (perfect and reversed ranking, a known AUROC of 0.75,
  a known F1 of 0.5), patients kept together in the bootstrap, identical
  resamples for the same seed, overfitting of 32 records, early stopping
  after `patience` epochs, restored best weights, frozen layers unchanged,
  same history for the same seed, exact mean loss with a smaller last
  batch, a next-token model reaching perplexity close to 1 (and exactly V
  when untrained), the JSON written, the test fold untouched without
  `--final`, a second final run refused, record order of the `--fast`
  subset, macro names and number formats.
- `python -m src.train --model trivial --fast`: runs on the GPU; log
  ending in `_fast`.
- `python -m src.train --model trivial --final --tag e2e`: run
  `trivial_e2e_s42_20261120-183809`, early stopping after 33 epochs, best
  epoch 28; split sizes 17,084 / 2,146 / 2,158, equal to the labelled
  records of the EDA. Listed in `report/runs.json` as `trivial`.
- `python report/make_numbers.py`: 295 macros from `eda` and `trivial`.
- Checked by the assistant on copies of these files: the metrics
  recomputed from the saved predictions equal those of the JSON, and
  `numbers.tex` compiles with pdfLaTeX.

The trivial model (logistic regression on the mean and standard deviation
of each lead) is the **floor** of the final comparison: every real model
should beat it clearly. Its numbers are in the macros `\trivial...`, never
typed by hand.

Note on `git_dirty: true` in this log (commit `f090231`): the code of this
block was indeed not yet committed when the run was made, but that is not
why the flag is true. With the first version of `src/runlog.py` every full
run was "dirty", because the run writes its own predictions file, still
untracked, before `git status --porcelain` is called (AI Error Bounty B5,
found in chat DL 04; the explanation first given here was wrong). Fixed on
2026-11-20: the flag now looks only at the code under `src/` and logs carry
`schema` 2 (D14). In logs with `schema` 1 the flag means nothing. The files
committed after this run are byte-identical to those that produced it
(checked with SHA-256 before the run).

## Problems met and lessons

- **Stale copy of a file sent to the laptop** (same failure as AI Error
  Bounty B2, logged as B3): the first copy of `DECISIONS.md` with D14
  reached the folder without the D14 section. Detected by re-reading the
  folder before the next write. Since then every file sent is compared
  with SHA-256 against the tested version.
- **The loop had to be redesigned before any code** when the professor's
  message of 20/11 added the generative block: metrics accumulated batch by
  batch instead of keeping all outputs (about 560 million logits on the
  validation fold for a 256-token vocabulary).
- **Smoke-test runs in the repository:** without care, every `--fast` run
  of every model chat would end up on GitHub; hence the `_fast` suffix and
  the `.gitignore` rule.

## For the next chats

- Chats 04-07: register the model, use `--fast` first, then the base run
  and the ablations on validation only; `--final` once per final model
  (seeds 42, 43, 44).
- Chat 13: `NextTokenTask` and `metrics.perplexity` are ready and tested
  on a toy sequence; tokenisation and models are yours.
- Chats 10 and 11: numbers only through `numbers.tex`; add each reported
  run to `report/runs.json` with a letters-only alias.
