# ptbxl-dl

Deep Learning project of the courses PRDL and MLLB (MUTSC, UPM, 2026-2027).
Authors: Oriana Ferraz De Jesus and María Martínez García-Taheño.

Multi-label classification of the five diagnostic superclasses of PTB-XL
(12-lead ECG, 100 Hz) with dense, convolutional, recurrent and Transformer
models, compared under the same data split and evaluation protocol.

Status: data pipeline, exploratory analysis, evaluation module and shared
training loop in place; model blocks in progress.

## Structure

```
src/        Python modules (data, models, training, evaluation)
notebooks/  exploratory analysis and experiments
tests/      automated tests (pytest)
results/    logs/ with one JSON file per run
report/     figures/ and the script that builds the numbers of the reports
docs/       decisions, AI usage log and one note per block
data/       raw data and caches (not committed)
```

## Data

PTB-XL 1.0.3 (CC BY 4.0): https://physionet.org/content/ptb-xl/1.0.3/

The data are not part of this repository. Download the dataset and place
it so that this path exists:

```
data/raw/ptbxl/records100/
data/raw/ptbxl/ptbxl_database.csv
data/raw/ptbxl/scp_statements.csv
```

Only the 100 Hz records are used; `records500/` is not needed.

Check the files and build the cache (about 6 minutes, 1 GB):

```
python -m src.prepare_data
```

`src/dataset.py` builds the train (folds 1-8), validation (fold 9) and test
(fold 10) DataLoaders from the cache.

## Training, evaluation and report numbers

```
python -m src.train --model trivial --fast     # smoke test (not reported)
python -m src.train --model trivial            # full run, validation only
python -m src.train --model trivial --final    # also evaluates the test fold
python -m src.compare RUN_A RUN_B              # paired bootstrap difference
python report/make_numbers.py                  # report/numbers.tex
```

Each run writes `results/logs/<run_id>.json` (configuration, curves,
metrics with bootstrap intervals, software versions) and
`results/predictions/<run_id>.npz`; the best weights go to
`results/checkpoints/` (not committed). `report/make_numbers.py` turns the
logs listed in `report/runs.json` into LaTeX macros. Decisions: D4 and D14
in `docs/DECISIONS.md`.

## Model blocks

Dense network (decision D15): baseline, 11 ablation configurations with 3
seeds on validation, the final model on the test fold, and the summary log,
tables (`report/tables/`) and figures of the block.

```
python -m src.dense_experiments all --fast     # smoke test (not reported)
python -m src.dense_experiments run            # ablations, validation only
python -m src.dense_experiments final          # selected model, test fold
python -m src.dense_experiments report         # summary, tables, figures
```

1D CNN (decision D16): residual baseline, 10 ablation configurations
(kernel size, number of blocks, batch norm, skip connections, global pooling
vs flatten, dropout) with 3 seeds on validation, the final model on the test
fold compared with the final Dense and trivial models, and interpretation
figures (first-layer filters, feature maps, class activation maps).

```
python -m src.cnn_experiments all --fast       # smoke test (not reported)
python -m src.cnn_experiments run              # ablations, validation only
python -m src.cnn_experiments final            # selected model, test fold
python -m src.cnn_experiments report           # summary, tables, figures
```

Recurrent networks (decision D17): LSTM and GRU reading the record as 100
frames of 100 ms, an RNN + attention model, 12 ablation configurations
(cell, depth, direction, last state / mean / attention, attention score,
convolutional front end, 1000 raw steps) with 3 seeds on validation, two
final models on the test fold compared with the final CNN, Dense and
trivial models, and interpretation figures (input sensitivity over time,
attention weights).

```
python -m src.rnn_experiments estimate         # expected run time on this GPU
python -m src.rnn_experiments all --fast       # smoke test (not reported)
python -m src.rnn_experiments run              # ablations, validation only
python -m src.rnn_experiments final            # selected models, test fold
python -m src.rnn_experiments report           # summary, tables, figures
```

Self-attention and Transformer (decision D18): a Pre-LN Transformer encoder
over 100 patches of 100 ms, with multi-head self-attention written by hand
(`src/models/attention.py`, also usable with a causal mask), a step-by-step
study of self-attention (one head alone, several heads, Add & Norm, FFN, a
second block), 15 configurations (depth, heads, positional encoding, patch
length, CLS token, LayerNorm placement) with 3 seeds on validation, the
final model on the test fold compared with the final CNN, RNN, RNN +
attention, Dense and trivial models, and interpretation (patch order, input
sensitivity over time, attention statistics and maps).

```
python -m src.transformer_experiments estimate   # run time and GPU memory
python -m src.transformer_experiments all --fast # smoke test (not reported)
python -m src.transformer_experiments run        # ablations, validation only
python -m src.transformer_experiments final      # selected model, test fold
python -m src.transformer_experiments report     # summary, tables, figures
```

Generative models (decision D19): lead II quantised on the microvolt grid
into 256 tokens, an ECG-GPT (decoder-only Transformer with the causal mask
of `src/models/attention.py`) and an LSTM generator of the same size,
trained to predict the next sample; 8 configurations (architecture, context
length, vocabulary, positional encoding, causal mask) with 3 seeds on
validation, compared in bits per sample with counting baselines; generation
from a 2 s prompt with greedy, temperature and top-k sampling, compared
with the real signal (amplitude, heart rate). The test fold is not used.
The selected model is the self-supervised pre-trained model of the
transfer-learning block (D7).

```
python -m src.generative_experiments estimate   # run time and GPU memory
python -m src.generative_experiments all --fast # smoke test (not reported)
python -m src.generative_experiments run        # 8 configurations x 3 seeds
python -m src.generative_experiments report     # generation, summary,
                                                # tables, figures
```

## Setup (Windows, NVIDIA GPU)

```
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
python -m src.check_env
python -m pytest -q tests
```

The `cu126` build needs an NVIDIA driver that supports CUDA 12.6 or later
(`nvidia-smi` shows the supported version). Without a GPU, replace the
`torch` line by `pip install torch`.
