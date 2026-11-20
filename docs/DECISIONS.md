# DECISIONS.md — Deep Learning project (PRDL-DL / MLLB-DL), MUTSC-UPM 2026-2027

Team: Oriana Ferraz De Jesus, María Martínez García-Taheño.
Rule: decisions are written here **before** any code. A decision changes only by adding a dated entry, never by silently editing an old one.
Tags: **[verify]** = not yet checked against the primary source. **[open]** = not decided.

Last update: 2026-11-20.

## D0. Course requirements this plan must satisfy

Checked on 2026-11-14 in `contexto_DL/`.

| Requirement | Source |
|---|---|
| Minimum architectures: Dense, CNN, LSTM and, if possible, Transformer | Presentation III ("PRDL & MLLB Activities", Deep Learning project) |
| Ablation studies are mandatory | Presentation III; weekly summary 30/10 |
| One use case solved with CNN, RNN (LSTM, GRU, +attention), Transformer, transfer / self-supervised learning | Weekly summary 13/11 |
| Rubric: problem and data 0.5; DL models (CNN, RNN, Transformer) 3 points | Presentation III (grading slide) |
| Developed in working teams, but each member presents an individual report describing her specific activities | Presentation III |
| Goal is skills and understanding, not performance; simple, own experiments | Presentation III; Notebooks_GuideLines slides 2-5 |
| Reviewed first assignment (learning strategies) goes into the PRDL report | Notebooks_GuideLines slides 29-36 |
| AI use documented; AI Error Bounty | Presentation III |
| PyTorch details are an explicit topic of the MLLB report | Weekly summary 13/11 |

## D1. Dataset — DECIDED (2026-11-14)

**PTB-XL**, version 1.0.3, PhysioNet: https://physionet.org/content/ptb-xl/1.0.3/

- 21,799 clinical 12-lead ECG records of 10 s from 18,869 patients (PhysioNet page).
- Waveforms at 500 Hz and a downsampled 100 Hz version. **We use 100 Hz** (1,000 samples x 12 leads per record).
- Licence: CC BY 4.0. Checked on the PhysioNet page by OF and MM on 2026-11-14.
- Raw data are **not committed** to the repository. A download script fetches them; `data/` is in `.gitignore`.
- **2026-11-20 (OF + MM):** the full ZIP was downloaded by hand from PhysioNet and `records500/` deleted. Raw files live in `ptbxl-dl/data/raw/ptbxl/` (ignored by git). Instead of downloading, `python -m src.prepare_data` checks them against the dataset's `SHA256SUMS.txt` and builds a single-file cache `data/cache/ptbxl_100hz.npz`; the README explains where to get the data.
- Required citations: Wagner et al. 2020 (Scientific Data, doi 10.1038/s41597-020-0495-6), the PhysioNet dataset DOI (10.13026/kfzx-aw45) and the standard PhysioNet citation.

Reasons: biomedical signal, already curated, permissive licence, official folds, a published benchmark to compare against, small enough to train locally. Alternative kept as plan B: Drugs.com Reviews (UCI id 462), text, if a public BERT-type model turns out to be compulsory.

## D2. Task — DECIDED

Multi-label classification of the **5 diagnostic superclasses**: NORM, MI, STTC, CD, HYP.

- Labels are built from `scp_codes` through `scp_statements.csv` (`diagnostic_class`).
- A record can carry several superclasses, so the output is 5 logits with `BCEWithLogitsLoss` (no sigmoid in the model; float labels).
- How to treat records with no diagnostic superclass and statements with likelihood 0: **[open]**, decided after the EDA.
- **2026-11-20 (OF + MM), likelihood:** a record gets a superclass if any of its SCP codes is a diagnostic statement of that class, **whatever its likelihood** (including 0). Source: `example_physionet.py`, shipped with PTB-XL 1.0.3. Check: with this rule the counts are NORM 9514, MI 5469, STTC 5235, CD 4898, HYP 2649, equal to the table of the PhysioNet page; requiring likelihood > 0 gives STTC 5108. Counts computed by the assistant; they count only once reproduced by the T5 test on Oriana's laptop.
- **2026-11-20 (OF + MM), records without superclass (O3, provisional):** 411 records (same check). They stay in the cache, flagged `has_superclass = False`, and the DataLoader leaves them out of train, validation and test by default, with an option to include them. Reason: a vector of five zeros would teach "none of the classes", but these records simply have no diagnostic statement. Chat 02 closes O3 after the EDA (including their distribution across folds).

## D3. Data split — DECIDED

Use the stratified folds shipped with the dataset (`strat_fold`), which keep all records of a patient in the same fold.

- Train: folds 1-8. Validation: fold 9. Test: fold 10. Recommended split checked on the PhysioNet page by OF and MM on 2026-11-14.
- The test fold is touched once per final model. Model selection and all ablations use validation only.
- Normalisation statistics are computed on the training folds only.
- **2026-11-20 (OF + MM), normalisation:** per-lead z-score. For each of the 12 leads, mean and standard deviation over all samples of the training records (folds 1-8, same records as training, i.e. without the records excluded by O3); validation and test use those same values. Not chosen: one global mean/std (leads have different amplitude ranges) and per-record standardisation (it removes absolute amplitude, which is diagnostic, e.g. high voltages in HYP). The statistics are stored in each run's log, and normalisation can be switched off for the Dense ablation (D6). Implemented in `src/dataset.py`; `num_workers=0` because on Windows each worker process would copy the in-memory cache.

## D4. Evaluation protocol — DECIDED

- Main metric: **macro-averaged AUROC** over the 5 classes (the benchmark's metric, threshold-free).
- Secondary: per-class AUROC, macro F1 at thresholds chosen on validation.
- Uncertainty: bootstrap 95% confidence intervals on the test fold.
- Seeds: 3 seeds for the final model of each architecture; ablations with 1 seed unless a difference is within the CI, then 3. **[open]** to raise if time allows.
- Every run writes `results/logs/<run_id>.json` (config, seed, per-epoch curves, final metrics, software versions). No number is typed by hand in the reports: `report/make_numbers.py` generates them from these files.
- Results are only reported after one of us has executed the run on Oriana's laptop.

## D5. Framework and environment — DECIDED

- PyTorch. All training on Oriana's laptop (CUDA GPU). Colab/Kaggle only as fallback.
- A `--fast` mode (subset, few epochs) for smoke tests; full mode for reported results.
- Public GitHub repository with `src/`, `notebooks/`, `tests/`, `results/`, `report/`, `docs/`. Repository name and owner: **[open]**.
- **2026-11-20 (OF + MM):** repository https://github.com/orianafe16/ptbxl-DL, owned by Oriana. All work runs on Oriana's laptop; commits are made with her account only. Python 3.12 virtual environment `.venv` inside `ptbxl-dl/`; PyTorch with the CUDA 12.6 build (driver 566.07 supports up to CUDA 12.7). Environment checked with `python -m src.check_env` on 2026-11-20.
- Code, comments, figures and reports in English.

## D6. Models and ablation plan — DECIDED (scope), details may be refined after the EDA

Every model shares the same data pipeline, split, loss, metric and logging, so differences come from the architecture.

| Block | Baseline | Ablations (one factor at a time) | Owner |
|---|---|---|---|
| Dense (MLP) | Flattened signal, 2-3 hidden layers, ReLU | input standardisation on/off; dropout on/off; batch norm on/off; depth and width | Oriana |
| CNN 1D | Conv-BN-ReLU-pool blocks, global average pooling | kernel size; number of blocks; batch norm on/off; residual skip connections on/off; global pooling vs flatten | Oriana |
| RNN | LSTM over the (downsampled) sequence, dense layer on the last state | LSTM vs GRU; unidirectional vs bidirectional; last state vs mean pooling vs attention pooling; raw sequence vs convolutional front-end | María |
| Transformer | Patch embedding of the signal + small encoder | positional encoding on/off; patch length; number of layers and heads; CLS token vs mean pooling | María |
| Transfer learning | See D7 | from scratch vs frozen features vs fine-tuning; fraction of labelled data | Both |
| Cross-cutting | 12 leads | 12 leads vs single lead (lead II) on the best model of each family | Both |

Hyperparameter budgets are kept small and equal across families; the search space and budget are logged per run.

## D7. Transfer learning / pre-trained model — DECIDED (2026-11-14)

**We pre-train a model ourselves and transfer it.** Team decision by OF and MM; it has not been confirmed with the professor, so it stays a point to raise in the December class discussion.

- Pre-train the CNN on PTB-XL with a different objective from the target task (self-supervised, or other statement groups). Exact pretext task: **[open]**, decided after the CNN block.
- Reuse it on the 5-superclass task in three ways and compare: training from scratch, frozen feature extractor, fine-tuning.
- Repeat the comparison with 10% and 100% of the training labels.
- This matches the course definition of transfer learning (Notebooks_GuideLines, slide 32).
- The pre-training never sees validation or test folds.

Optional extension if time allows: transfer the same pre-trained model to a second small ECG dataset (candidate: PhysioNet/CinC Challenge 2017, single lead).

Not chosen: a public pre-trained ECG model. Caveats found on 2026-11-14: the open models seen (HuBERT-ECG, ECG-FM) included PTB-XL in their pre-training data, HuBERT-ECG is CC BY-NC 4.0 and its base model has about 93 M parameters.

## D8. Literature — DECIDED (same flow as the ML project)

Perplexity -> NotebookLM (inventory + literal quotes with file and page) -> text written only from those quotes. Anything without a quote stays `[verify]`. arXiv identifiers are checked as well as DOIs.

Reference points to verify through that flow before citing any number:
- Wagner et al. 2020, PTB-XL dataset paper.
- Strodthoff et al. 2021, "Deep Learning for ECG Analysis: Benchmarks and Insights from PTB-XL" (arXiv 2004.13701) and its repository `helme/ecg_ptbxl_benchmarking`.

## D9. Reports and authorship — DECIDED (2026-11-14)

- Four documents: PRDL-DL and MLLB-DL for Oriana, PRDL-DL and MLLB-DL for María, as the course rule requires.
- Shared: repository, data pipeline, result logs, figures and tables.
- Individual in each report: description of the author's own activities, the detailed analysis of the architectures she owns (D6), her discussion and conclusions, and her AI appendix.
- The contribution table in the reports is generated from the task log in `PLAN_DL.md`, which records who actually did and who reviewed each task.
- LaTeX on Overleaf with the template of `ML_ref_*_main.tex.txt`; MLLB keeps Section 0 on the AI environment.

## D10. AI use — DECIDED

- `docs/AI_USAGE_LOG.md`: prompts, what the assistant produced, what was checked and by whom.
- Every assistant error corrected by either of us is logged as an AI Error Bounty candidate (prompt, wrong answer, evidence, correction), attributed to whoever found it.

## D11. Working organisation — DECIDED (2026-11-14)

- The work is split into 13 chats of the Claude project (`CHATS_DL.md`): coordination, data, EDA, evaluation and training loop, one chat per architecture, transfer learning and final comparison, literature, one report chat per author, and review.
- Chats share nothing but the local folder. Coordination files live in the `DL/` root: `DECISIONS.md`, `PLAN_DL.md`, `HANDOFF.md`, `AI_USAGE_LOG.md`, `CHATS_DL.md`.
- Code and results live in `DL/ptbxl-dl/`, the future public repository. `contexto_DL/` (course material) never goes into it.
- Each model chat ends with a block note in `ptbxl-dl/docs/blocks/`, which is what the report chats read.
- **2026-11-20 (DL 00, decided for OF and MM at their request):** the project folder is `C:\Users\ori\DL`, outside OneDrive. `DECISIONS.md`, `PLAN_DL.md` and `AI_USAGE_LOG.md` move to `ptbxl-dl/docs/` as their only copy, so they are versioned and backed up on GitHub with the code; this makes them public. `CHATS_DL.md`, `HANDOFF.md` and `contexto_DL/` stay internal in the folder root and are copied to OneDrive (`MASTER\DL_backup`) with `backup_dl.bat`, which never deletes anything in the copy. Raw data and the cache are not backed up: they are re-downloaded, checked with `SHA256SUMS.txt` and rebuilt. Every chat closes with commit and push of `ptbxl-dl/` and a run of `backup_dl.bat`.

## Open items

| # | Item | Blocks |
|---|---|---|
| O1 | Pretext task for the pre-training (D7) | Transfer-learning block only |
| O3 | Handling of records without diagnostic superclass (D2); provisional rule since 2026-11-20 | Final rule after EDA (chat 02) |

Closed: O2 (licence and folds verified on PhysioNet, 2026-11-14). O4 (repository under Oriana's account, 2026-11-20).

## Change log

- 2026-11-14: first version; dataset decided.
- 2026-11-14: licence and recommended folds verified on PhysioNet by OF and MM (D1, D3). Transfer learning decided: own pre-trained model (D7), not confirmed with the professor.
- 2026-11-14: working organisation in chats added (D11).
- 2026-11-20: data location and integrity check (D1); likelihood rule and provisional O3 rule (D2); repository, owner and environment (D5); O4 closed. Decided by OF and MM in chat DL 01.
- 2026-11-20: normalisation rule (D3). Decided by OF and MM in chat DL 01.
- 2026-11-20: folder, location of the coordination files and backup rule (D11). Decided in chat DL 00.
