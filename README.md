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
