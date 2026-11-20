# Block 02 — Exploratory data analysis (T6)

Chat DL 02, 2026-11-20. T6 done by OF, reviewed by MM. Executed by OF
on Oriana's laptop.
Decisions: `docs/DECISIONS.md` D12 (scope, outputs, codings) and O3
(closed in D12).

## What the block delivers

| Item | File | Command |
|---|---|---|
| Analysis script (all numbers and figures) | `src/eda.py` | `python -m src.eda` |
| Notebook that runs the script and shows the figures | `notebooks/02_eda.ipynb` | Run All, `.venv` kernel |
| Every EDA number | `results/logs/eda.json` | written by the script |
| Figures, PDF (reports) and PNG (notebook) | `report/figures/eda_*` | written by the script |

`eda.json` has three parts: `scalars` (flat `name -> number`, for
`make_numbers.py`), `tables` (small tables behind the figures) and `meta`
(date, git commit, package versions). The script rewrites everything on
each run, so numbers and figures always come from one complete run.

**No number in the reports is taken from this note**: they come from
`eda.json` through `make_numbers.py`. The key names are given so that the
report chats can find them.

## Figures

| File | Shows |
|---|---|
| `eda_class_counts` | (a) labelled records per superclass; (b) prevalence per split |
| `eda_label_cooccurrence` | (a) P(column class given row class); (b) most frequent label combinations |
| `eda_fold_distribution` | prevalence of each class in each of the 10 folds |
| `eda_age_sex` | (a) age histogram per sex; (b) age per superclass |
| `eda_signal_quality` | (a) quality notes per split; (b) per class |
| `eda_lead_stats` | (a) std per lead; (b) peak-to-peak amplitude per lead (training) |
| `eda_example_traces` | lead II of one training record per superclass (seed 42) |
| `eda_example_12lead` | the 12 leads of the NORM example: one model input (12, 1000) |

## Findings

1. **Records without superclass (O3, closed).** Almost all carry a rhythm
   statement and most carry only rhythm statements (pacemaker, atrial
   fibrillation, atrial flutter first). Rhythm statements are not
   diagnostic in `scp_statements.csv`, so they give no superclass. A target
   of five zeros would be a false "no pathology" label, so they stay out of
   the three splits. They are spread evenly over the folds. Keys:
   `n_nosuper_*`, `nosuper_per_fold`, `nosuper_codes`.
2. **Folds are well stratified.** Class prevalence is almost identical in
   train, validation and test, and in every fold. Keys: `prev_<split>_<class>`,
   `max_fold_prev_gap_pp`.
3. **Class imbalance.** NORM is the largest class and HYP the smallest.
   Key: `imbalance_ratio_train`. Supports a threshold-free main metric
   (macro AUROC, D4); class weights in the loss are a choice for chat 03 or
   the model chats.
4. **Genuinely multi-label.** About a quarter of the records have more than
   one superclass, and some pairs are more frequent than single-class HYP
   (e.g. MI+CD). Supports 5 sigmoid outputs with `BCEWithLogitsLoss` rather
   than a softmax (D2). Keys: `frac_multilabel_all`, `n_with_<k>_labels`,
   `label_combinations`, `cooccurrence_*`.
5. **NORM with another class.** A few hundred records are NORM plus another
   superclass, almost all NORM + CD through mild conduction findings
   (IRBBB, IVCD, 1AVB). Not a pipeline error: these codes are in the
   original annotation. Whether this is clinically usual: **[verify]**.
   Keys: `n_norm_with_other`, `norm_with_other_codes`.
6. **Age.** NORM patients are younger than those of the four pathological
   classes (median ages). A model could partly learn age-related signal
   features as a shortcut; worth a sentence in the discussion. Keys:
   `age_median_<class>`, `age_*`.
7. **Sex.** Both codes are about equally frequent overall, but the share
   differs between classes. The meaning of 0/1 is not in the dataset files:
   **[verify]** (D12). Keys: `n_sex*`, `frac_sex1_*`.
8. **Patients with several records.** A minority of patients have several
   records (up to 10): if records were split at random, the same patient
   could be in train and test. This is why the patient-wise official folds
   matter (D3). Keys: `n_patients_several_records`, `max_records_per_patient`.
9. **Label quality.** All validation and test reports were validated by a
   person, but not all training reports. Why folds 9 and 10 are built this
   way: **[verify]** in the dataset paper. Keys: `frac_validated_by_human_*`.
10. **Signal quality.** About a third of the records have some quality note
    (noise, drift, extra beats...). Nothing is excluded (D12). Baseline
    drift is noted more often in validation and test than in training.
    Keys: `n_<column>_<split>`, `frac_any_quality_flag_*`,
    `quality_frac_per_class`.
11. **Leads differ in scale.** Precordial leads (V1-V6) have about twice
    the spread of the limb leads, which is the data behind per-lead
    normalisation (D3). The mean and std in `lead_stats_train` are the very
    values used by the DataLoader (same functions, same records). Keys:
    `lead_std_ratio_max_min`, `lead_mean_*`, `lead_std_*`.
12. **Integrity.** No missing values. One flat (constant) lead in the
    training set: V5 of record 12722, whose metadata already notes an
    electrode problem on V4 and V5. Kept (D12). Keys: `n_nan_values_train`,
    `flat_leads_train`.

## Techniques worth explaining in the defence

- **Co-occurrence with one product.** With the label matrix Y (records x 5,
  zeros and ones), `Y.T @ Y` counts in cell (i, j) the records with both
  class i and class j; the diagonal holds the class totals. Dividing each
  row by its diagonal gives P(j | i). The result is not symmetric.
- **Same labels and statistics as the models.** `load_table` builds labels
  with `src.data.superclass_labels`, and `check_against_cache` stops if they
  differ from the cache. Lead statistics use `split_indices` and
  `lead_stats` from `src.dataset`. The EDA therefore describes exactly what
  the models see.
- **Lazy reading of `.npz`.** Asking for two keys loads only those arrays,
  not the 1 GB of signals.
- **Box plots.** Box = quartiles 1 to 3, line = median. In `eda_age_sex`
  whiskers reach 1.5 interquartile ranges; in `eda_lead_stats` they are the
  5th and 95th percentiles, computed beforehand (`Axes.bxp`). No outlier
  points are drawn.
- **Example traces drawn at random (seed 42)** among training records with
  a single superclass, so they are typical, not hand-picked.
- **Colours.** Each split, class and sex keeps one colour in every figure;
  the categorical palette was checked with a colour-blindness validator.
  Lines also carry a marker and a direct label, so identity never depends
  on colour alone.
- **Codings.** Age 300 means older than 89 (`ptbxl_v102_changelog.txt`,
  line 2): counted apart, left out of histograms and medians. Signals are
  in mV (gain 1000 per mV in the record headers).

## Checks run on Oriana's laptop (2026-11-20)

- `python -m src.eda`, run four times as the parts were added (1-2, then
  3, 4 and 5); every printed number equal to the assistant's run on the
  metadata, except the signal statistics, which only the laptop has (full
  cache).
- `notebooks/02_eda.ipynb`, Run All: same values as the script
  (e.g. std ratio, flat lead), all figures shown.

## Problems met and lessons

- **Wrong claim by the assistant:** it reported `DECISIONS.md` D12 as
  written when the copy that reached the laptop was the old version (the
  file was sent right after editing). Found by the assistant when it
  re-read the file; rewritten and checked by file size. Logged in
  `AI_USAGE_LOG.md`. Lesson: after writing a shared file, re-read it from
  the folder.
- **Units checked on one header only** (`records100/00000/00001_lr.hea`);
  `src/data.py::load_signal` checks sampling rate, leads and shape but not
  units. Low risk (one dataset, one converter), noted as a limitation.
- **`src/eda.py` imports `src/dataset.py`, hence PyTorch.** Accepted, so
  that the EDA reuses the DataLoader's exact functions.

## Open points

- **[verify] for chat 09:** meaning of `sex` 0/1 (Wagner et al. 2020 or the
  PhysioNet page); why folds 9-10 contain only human-validated reports;
  NORM + mild conduction findings; how the benchmark of Strodthoff et al.
  treats records without superclass.

## For the next chats

- **Chat 03 (evaluation, `make_numbers.py`):** read `scalars` of
  `eda.json`. Some names contain digits or underscores that LaTeX macro
  names cannot (`n_with_1_labels`, `fold_1`, `lead_std_V1`, `prev_train_MI`):
  `make_numbers.py` needs a name mapping. Class imbalance
  (`imbalance_ratio_train`) is relevant if class weights are considered.
- **Chat 04 (Dense):** the `normalize=False` ablation feeds raw mV, whose
  scale differs between leads (`lead_stats_train`).
- **Chat 08 (transfer):** the records without superclass of folds 1-8 may
  be used without labels for pre-training (D12, decided with O1).
- **Chats 10 and 11 (reports):** figures and keys listed above; findings 5,
  7 and 9 need their [verify] resolved before they are written as facts.
