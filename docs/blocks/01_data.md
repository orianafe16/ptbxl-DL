# Block 01 — Data pipeline (T4, T5)

Chat DL 01, 2026-11-20. T4 done by OF, reviewed by MM. T5 done by MM,
reviewed by OF. Everything executed on Oriana's laptop.
Decisions: `docs/DECISIONS.md` D1, D2, D3, D5 and O3 (entries of 2026-11-20).

## What the block delivers

| Item | File | Command |
|---|---|---|
| Settings (paths, classes, leads, folds, seed) | `src/config.py` | — |
| Integrity check against `SHA256SUMS.txt` | `src/data.py::verify_raw` | `python -m src.prepare_data` |
| Superclass labels | `src/data.py::superclass_labels` | (same) |
| Single-file cache `data/cache/ptbxl_100hz.npz` | `src/data.py::build_cache` | (same) |
| Split, normalisation, Dataset, DataLoaders | `src/dataset.py` | used by the training code |
| Tests | `tests/test_data.py` | `python -m pytest -v tests` |
| Environment check | `src/check_env.py` | `python -m src.check_env` |

How other chats use it:

```python
from src.data import load_cache
from src.dataset import make_loaders

cache = load_cache()
loaders, info = make_loaders(cache, batch_size=64)   # normalize=True by default
for x, y in loaders["train"]:   # x: (B, 12, 1000) float32, y: (B, 5) float32
    ...
```

`info` holds the split sizes and the per-lead mean and standard deviation;
the training code (chat 03) must write it into each run's JSON log.

## Pipeline, step by step

1. **Raw data.** PTB-XL 1.0.3 ZIP downloaded by hand from PhysioNet,
   `records500/` deleted, files in `data/raw/ptbxl/` (ignored by git).
2. **Integrity.** Each file we use (`ptbxl_database.csv`,
   `scp_statements.csv`, every file under `records100/`) is hashed with
   SHA-256 and compared with the checksum list published with the dataset.
   A single changed byte changes the hash.
3. **Labels.** A record gets superclass *c* if any of its SCP codes is a
   diagnostic statement of class *c* in `scp_statements.csv`, whatever its
   likelihood. This is the rule of `example_physionet.py`, shipped with the
   dataset. Output: multi-hot float matrix (N, 5) in the order
   NORM, MI, STTC, CD, HYP.
4. **Cache.** Each record is read once with `wfdb`, checked (100 Hz, the
   12 expected leads, 1000 samples) and stored as (12, 1000): wfdb returns
   (time, lead), while PyTorch's `Conv1d` expects (channel, time). Stored
   with labels, `ecg_id`, `patient_id`, fold and `has_superclass`. Written
   to a temporary file and renamed, so an interrupted run never leaves a
   half-written cache.
5. **Split.** Official stratified folds: train 1-8, validation 9, test 10.
   All records of a patient are in the same fold.
6. **Records without superclass (provisional, O3).** Kept in the cache,
   left out of the three splits by default (`include_no_superclass=False`).
   A vector of five zeros would teach "none of the classes", but these
   records simply have no diagnostic statement. Chat 02 decides.
7. **Normalisation.** Per-lead z-score: for each lead, mean and standard
   deviation over all samples of the training records only; validation and
   test use the same values. Using validation or test records here would
   leak information about them (data leakage). Rejected alternatives: one
   global mean/std (leads have different ranges) and per-record
   standardisation (removes absolute amplitude, which is diagnostic, e.g.
   high voltage in hypertrophy). `normalize=False` for the Dense ablation.
8. **Dataset and DataLoader** (course slides `pytorch_dataset_dataloader-v2`,
   2-5). `__len__` gives the number of records of the split, `__getitem__`
   returns one `(x, y)`; the DataLoader groups items into batches. The
   Dataset keeps a reference to the full cache and a list of row indices,
   so the three splits do not copy the 1 GB array. Labels are float because
   `BCEWithLogitsLoss` needs float targets. Only the training loader
   shuffles, with a seeded `torch.Generator` so the order is reproducible.
   `num_workers=0`: on Windows each extra worker process would copy the
   in-memory cache.

## Checks run on Oriana's laptop (2026-11-20)

- `python -m src.check_env`: Python 3.12.10, PyTorch 2.14.1+cu126,
  RTX 3050 Ti Laptop GPU (4 GB), GPU test OK.
- `python -m src.prepare_data`: integrity check OK; cache with
  X (21799, 12, 1000), 18,869 patients; class counts equal to the table
  of the PhysioNet page; 411 records without superclass.
- `python -m pytest -v tests`: 8 passed (commit 9993fd0). Four synthetic
  tests (split by fold, exclusion of records without superclass,
  statistics computed on training records only, normalised batch with
  mean 0 and standard deviation 1) and four on the real cache (shapes,
  class counts against PhysioNet, no patient in two splits, batch shapes
  (32, 12, 1000) and (32, 5)).

The numbers above come from the printed output of these commands. For the
reports they must come from a JSON log (chat 02 writes `eda.json`), not
from this note.

## Problems met and lessons

- **Wrong likelihood rule proposed by the assistant** (AI Error Bounty B1):
  it first proposed to keep only codes with likelihood > 0, claiming it was
  the benchmark rule. Checked against `example_physionet.py` and the
  PhysioNet table: that rule loses 127 STTC records. Lesson: label rules are
  checked against the dataset's own files and published counts.
- **First commit failed:** git identity not configured; without a commit the
  branch `main` did not exist, hence `src refspec main does not match any`
  on push.
- **CUDA build:** driver 566.07 supports CUDA up to 12.7, so the cu126 build
  of PyTorch was used (cu128 would need a driver update).
- **Location:** the project was moved out of OneDrive (`C:\Users\ori\DL`),
  because tens of thousands of small files and a `.git` folder do not sync
  well.

## For the next chats

- Chat 02 (EDA): close O3; check how the 411 records without superclass
  are spread over folds and classes of other statements.
- Chat 03 (evaluation and training): use `make_loaders`; store `info` in
  each run's JSON; `BCEWithLogitsLoss` on 5 logits (no sigmoid in models).
