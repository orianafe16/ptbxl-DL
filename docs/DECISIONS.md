# DECISIONS.md — Deep Learning project (PRDL-DL / MLLB-DL), MUTSC-UPM 2026-2027

Team: Oriana Ferraz De Jesus, María Martínez García-Taheño.
Rule: decisions are written here **before** any code. A decision changes only by adding a dated entry, never by silently editing an old one.
Tags: **[verify]** = not yet checked against the primary source. **[open]** = not decided.

Last update: 2026-11-21.

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
| Each DL report starts with a structured summary of at most 3 pages (scope, core methodology, main findings, key conclusions) | Professor's message 20/11 |
| No overly complex tasks or large state-of-the-art models; focus on methodological clarity, not computational scale | Professor's message 20/11 |
| The report must explicitly cover three areas: (1) core architectures: CNN, RNN (LSTM and GRU), RNN + attention, self-attention, Transformer; (2) learning strategies: a foundational model trained in a self-supervised way, connected to transfer learning, fine-tuning and zero-shot learning; (3) basic generative AI: at least an autoregressive sequence generator (RNN or Transformer, LLM-like) | Professor's message 20/11 |
| Class notebooks may be used as examples or built upon | Professor's message 20/11 |
| Initial draft of the DL reports due on 1 December 2026 | Professor's message 20/11 |

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
- **2026-11-20 (OF + MM), O3 closed:** records without superclass stay out of train, validation and test (see D12).

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
- **2026-11-20 (OF + MM, chat DL 03), confidence intervals for ablations:** D3 keeps every ablation on the validation fold, so the "difference within the CI" rule above is applied with bootstrap intervals computed on **validation**. Bootstrap intervals on the **test** fold are computed only for final models (`--final`). Implementation details in D14.

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

**2026-11-20 (OF + MM, chat DL 00), after the professor's message of 20/11 (D0):**
- *RNN + attention* is a model of its own, not only a pooling ablation: GRU (or LSTM) with an attention layer over its hidden states, as in the class notebook `TODO_MLLB_Embeddings_GRU_Attention_SentimentAnalysis.ipynb`; the attention weights over time are plotted. Ablation: with vs without attention.
- *Self-attention* is studied step by step before the full Transformer: a single self-attention layer (no feed-forward block, no stacking) on the ECG patches, then the full encoder block (multi-head attention + FFN + skip connections), following the class notebooks 01-03 of `Transformer_Architecture_Intro`. Ablations add: self-attention layer alone vs full block; attention maps.
- Dense stays as the baseline (minimum set of Presentation III), with a reduced ablation set if time is short.
- Ablations remain mandatory; a reduced set is acceptable in the 1 December draft and is completed afterwards.

## D7. Transfer learning / pre-trained model — DECIDED (2026-11-14)

**We pre-train a model ourselves and transfer it.** Team decision by OF and MM; it has not been confirmed with the professor, so it stays a point to raise in the December class discussion.

- Pre-train the CNN on PTB-XL with a different objective from the target task (self-supervised, or other statement groups). Exact pretext task: **[open]**, decided after the CNN block.
- Reuse it on the 5-superclass task in three ways and compare: training from scratch, frozen feature extractor, fine-tuning.
- Repeat the comparison with 10% and 100% of the training labels.
- This matches the course definition of transfer learning (Notebooks_GuideLines, slide 32).
- The pre-training never sees validation or test folds.

Optional extension if time allows: transfer the same pre-trained model to a second small ECG dataset (candidate: PhysioNet/CinC Challenge 2017, single lead).

**2026-11-20 (OF + MM, chat DL 00): pretext task decided, O1 closed.** The pre-trained model is the autoregressive ECG model of D13: it is trained without labels to predict the next sample of the signal (self-supervised, GPT-like), which is the "foundational model trained in a self-supervised manner" asked for in the professor's message of 20/11. It is then reused on the 5-superclass task with the same lead as its input (lead II): from scratch vs frozen features (linear probe) vs fine-tuning, with 10% and 100% of the training labels. Pre-training may also use the training-fold records without superclass (D12, rule c). Zero-shot learning does not fit an ECG classifier without a shared text-signal space, so it is covered by theory in the PRDL report and by a short experiment on the class notebook `Notebooks/CLIP/Interacting_with_CLIP.ipynb` (task T24).

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
- **2026-11-20 (OF + MM, chat DL 00), after the professor's message of 20/11 (D0):** every DL report (all four) opens with a structured summary of at most 3 pages: scope, core methodology, main findings, key conclusions. The body is organised around the three areas of D0 (core architectures; learning strategies; generative AI) and kept short: depth goes into the author's own blocks, the rest is summarised and attributed. Each author submits her initial draft by 1 December 2026; the draft contains the results already executed and, for each area, at least the design and what is still missing.

## D10. AI use — DECIDED

- `docs/AI_USAGE_LOG.md`: prompts, what the assistant produced, what was checked and by whom.
- Every assistant error corrected by either of us is logged as an AI Error Bounty candidate (prompt, wrong answer, evidence, correction), attributed to whoever found it.

## D11. Working organisation — DECIDED (2026-11-14)

- The work is split into chats of the Claude project (`CHATS_DL.md`; 13 chats on 2026-11-14, plus DL 13 Generative on 2026-11-20): coordination, data, EDA, evaluation and training loop, one chat per architecture, transfer learning and final comparison, literature, one report chat per author, and review.
- Chats share nothing but the local folder. Coordination files live in the `DL/` root: `DECISIONS.md`, `PLAN_DL.md`, `HANDOFF.md`, `AI_USAGE_LOG.md`, `CHATS_DL.md`.
- Code and results live in `DL/ptbxl-dl/`, the future public repository. `contexto_DL/` (course material) never goes into it.
- Each model chat ends with a block note in `ptbxl-dl/docs/blocks/`, which is what the report chats read.
- **2026-11-20 (DL 00, decided for OF and MM at their request):** the project folder is `C:\Users\ori\DL`, outside OneDrive. `DECISIONS.md`, `PLAN_DL.md` and `AI_USAGE_LOG.md` move to `ptbxl-dl/docs/` as their only copy, so they are versioned and backed up on GitHub with the code; this makes them public. `CHATS_DL.md`, `HANDOFF.md` and `contexto_DL/` stay internal in the folder root and are copied to OneDrive (`MASTER\DL_backup`) with `backup_dl.bat`, which never deletes anything in the copy. Raw data and the cache are not backed up: they are re-downloaded, checked with `SHA256SUMS.txt` and rebuilt. Every chat closes with commit and push of `ptbxl-dl/` and a run of `backup_dl.bat`.

## D12. Exploratory data analysis — DECIDED (2026-11-20)

Decided by OF and MM in chat DL 02.

- Code: `src/eda.py` writes `results/logs/eda.json` and the figures `report/figures/eda_*` in one complete run; a short notebook `notebooks/02_eda.ipynb` calls its functions and shows the figures. Every EDA number in the reports comes from `eda.json`.
- Scope: labels, co-occurrence, demographics, folds and signal quality are described over all records, broken down by split (train / validation / test); describing the test distribution does not influence training. Anything that could guide preprocessing (per-lead signal statistics, amplitudes) is computed on the training folds only. Example traces come from training records with a single superclass, drawn with seed 42.
- `eda.json` layout: `scalars` (flat name -> number, read by `make_numbers.py`), `tables` (small tables behind the figures), `meta` (date, git commit, package versions).
- Signal quality: the free-text quality columns (`baseline_drift`, `static_noise`, `burst_noise`, `electrodes_problems`, `extra_beats`, `pacemaker`) are turned into present/absent flags and described; no record is excluded, so the official folds stay intact.
- Codings: `age` = 300 means older than 89 (`ptbxl_v102_changelog.txt`, line 2); these records form a separate ">89" group and are left out of means and histograms. The meaning of `sex` 0/1 is not stated in the dataset files nor in `contexto_DL/`: reported as "sex 0 / sex 1", **[verify]** until chat 09 finds a literal quote. Height and weight: only the number of missing values is reported.
- O3, proposed rule: (a) records without superclass stay out of the three splits; (c) chat 08 may use the training-fold ones without labels for pre-training (decided with O1). Closed once the EDA has been run on Oriana's laptop.
- **2026-11-20 (OF + MM), O3 closed with rule (a) + (c).** Evidence from `python -m src.eda`, run by OF on Oriana's laptop (`results/logs/eda.json`, keys `n_nosuper_*`, `nosuper_per_fold`, `nosuper_codes`): almost all of these records carry a rhythm statement and most carry only rhythm statements (pacemaker, atrial fibrillation, atrial flutter are the most frequent codes); in PTB-XL rhythm statements are not diagnostic, so they give no superclass. A target of five zeros would therefore be a false label ("no pathology"), not a real negative. They are spread evenly over the ten folds, so leaving them out does not unbalance any split. `include_no_superclass=False` (the default of `make_loaders`) stays as it is.

## D13. Generative block and self-supervised pre-training (ECG-GPT) — DECIDED (2026-11-20)

Decided by OF and MM in chat DL 00 after the professor's message of 20/11 (D0, areas 2 and 3). One model serves both areas: it is the autoregressive generator (area 3) and the self-supervised pre-trained model reused in D7 (area 2), so the whole report stays on one use case.

- Input: lead II of PTB-XL at 100 Hz, training folds only (folds 1-8, including the records without superclass, D12 rule c). Validation fold 9 for model selection; fold 10 untouched by generation experiments.
- Tokens: each sample is quantised into a small vocabulary (256 levels as a starting point), so that the signal becomes a sequence of discrete symbols, like the characters of the class text-generation notebooks. Quantisation scheme (uniform on a clipped range vs mu-law) and clipping range taken from training statistics (`eda.json`, `lead_stats_train`): **[open]**, decided in chat DL 13.
- Models: a small decoder-only (causal) Transformer and an LSTM, both trained with next-token cross-entropy (teacher forcing). Same data, tokens and budget, so the comparison RNN vs Transformer is fair.
- Evaluation of the generator: validation cross-entropy and perplexity; generated signals inspected visually and compared with real ones with simple statistics (amplitude distribution; heart rate from R peaks if feasible, **[open]**).
- Generation: sampling with temperature (and top-k), as in `Notebooks/RNN/Sampling_in_Text_Generation.ipynb`; greedy vs sampling.
- Ablations: LSTM vs causal Transformer; temperature; context length; vocabulary size; causal mask removed (to show why it is needed).
- Reuse for D7: the trained network without its next-token head gives the features for the 5-superclass task.
- Owner: Oriana, reviewer: María (proposed by the assistant in chat DL 00 to balance the blocks; **to be confirmed by OF and MM**).
- Fallback if the ECG generator does not train within a reasonable time: autoregressive character-level text generation built on the class notebooks (`Sampling_in_Text_Generation.ipynb`, `TO_DO_MLLB_Keras_RNN_TextGeneration.ipynb`, and the "simple autoregressive LLM-style model" of the weekly summary of 20/11), with the pre-training of D7 done separately on ECG.

## D14. Evaluation module, training loop and run logs — DECIDED (2026-11-20)

Decided by OF and MM in chat DL 03 (tasks T7 and T8), after the professor's message of 20/11: the loop must also serve the generative model of D13.

**Evaluation (`src/metrics.py`)**
- Probabilities are `sigmoid(logits)`, applied outside the model. AUROC per class with scikit-learn; macro AUROC is the plain mean of the five.
- F1 thresholds: for each class, the threshold that maximises its F1 on validation, searched on the grid 0.01, 0.02, ..., 0.99 (ties: the lowest threshold). The same thresholds are applied to the test fold. Reported: F1 per class and macro F1 (mean of the five). Not chosen: a fixed 0.5 for every class, which penalises the rare classes (HYP).
- Bootstrap: 1000 resamples, percentile interval (2.5 %, 97.5 %), seed 42. The **patient** is the resampling unit (all records of a drawn patient enter together, as many times as the patient is drawn), because records of the same patient are correlated; resampling records would give intervals that are too narrow. Only the predictions are resampled; the model is not retrained. A resample in which a class has no positive or no negative record is discarded and counted in the log.
- Paired comparison: the resamples depend only on the patients and the seed, so every model evaluated on the same split sees the same resamples. The interval of the difference in macro AUROC between two runs is computed on these resamples (`python -m src.compare`, which writes its own log in `results/logs/`); if it contains 0, neither model is declared better. To make this possible, each run stores its validation (and, for final runs, test) probabilities in `results/predictions/<run_id>.npz`.
- Perplexity for the generative model (D13): exp of the mean cross-entropy **per token**, accumulated over all tokens of the split (not averaged per batch).

**Training loop (`src/train.py`, `src/tasks.py`)**
- The loop is task-agnostic. It performs the five steps (zero_grad, forward, loss, backward, step), `model.train()` / `model.eval()`, `torch.no_grad()` in evaluation, early stopping, saving of the best weights and the run log. Everything specific to a task comes from a task object: the loss, the metrics (accumulated batch by batch, so that a generative model never keeps all its logits in memory), and the selection metric with its direction. Two tasks: multi-label classification of the 5 superclasses (selection: validation macro AUROC, higher is better) and next-token prediction (selection: validation cross-entropy, lower is better).
- The loop receives ready-made DataLoaders and the data description to log; it does not build data. The optimiser only receives the trainable parameters (`requires_grad=True`), so frozen layers (D7) need no special code; the log records total and trainable parameters.
- Defaults: Adam, learning rate 1e-3, no weight decay, batch size 64, at most 50 epochs, early stopping with patience 5 on the selection metric, no gradient clipping (optional parameter, off by default). The weights of the best epoch are restored at the end and saved to `results/checkpoints/<run_id>.pt` (`state_dict`, not committed). Each model chat may change the defaults; the values used are logged.
- Seeds: Python `random`, NumPy and PyTorch (CPU and GPU) are seeded at the start of each run, before the model is built; the same seed orders the training batches (`make_loaders`). cuDNN deterministic, without benchmark mode. PyTorch's fully deterministic mode is not enabled (some GPU operations raise errors with it), so two GPU runs may differ slightly; results are compared with a tolerance. Seeds of the final models: 42, 43, 44.
- Class weights in the loss (`pos_weight`) off by default, available as an option (computed on the training records as negatives / positives per class).
- The test fold is evaluated only with `--final`; the loop refuses a second final run with the same model, tag and seed (D3).
- `--fast`: 512 training, 256 validation and 256 test records drawn with the seed, 2 epochs, 50 bootstrap resamples; the log is marked `fast`, its run name ends in `_fast` (so git ignores its log and predictions) and `make_numbers.py` rejects it.
- Device: GPU if available; no mixed precision.
- Trivial model for the end-to-end check: logistic regression on the mean and standard deviation of each lead (24 numbers per record).

**Run logs and numbers**
- Run name: `<model>_<tag>_s<seed>_<YYYYMMDD-HHMMSS>`. One JSON per run in `results/logs/`, with `scalars` (flat name -> number, read by `make_numbers.py`), configuration, model size, the data description returned by `make_loaders`, per-epoch curves, best epoch, results per split, paths of checkpoint and predictions, and `meta` (date, git commit and whether there were uncommitted changes, software versions, device).
- `report/runs.json` maps a short alias (letters only) to a log file. `make_numbers.py` reads only the files listed there, so an old or test run cannot slip into a report.
- Macro names: alias + key, where the key is cut at underscores and at letter/digit boundaries, each letter part starts with a capital letter and the rest keeps its case, and each digit is spelled out (`n_with_1_labels` -> `\edaNWithOneLabels`, `prev_train_MI` -> `\edaPrevTrainMI`, `lead_std_V1` -> `\edaLeadStdVOne`). Two keys giving the same name stop the script with an error. A comment with the original key precedes each macro.
- Format: integers with a thousands separator (`21{,}799`); other numbers with 3 decimals; keys starting with `frac_` or `prev_` also give a percentage macro with suffix `Pct` and 1 decimal. A value that should print as an integer must be stored as an integer in the JSON.
- Paths of logs, checkpoints, predictions and figures live in `src/config.py`; `src/eda.py` uses them too.
- **2026-11-20 (OF + MM, chat DL 03, after AI Error Bounty B5 found in chat DL 04), `git_dirty`:** the flag now looks only at the code under `src/` (`git status --porcelain --untracked-files=all -- src`): it is true when a tracked source file was changed or deleted, or a new source file was not yet added. Before, `git status --porcelain` looked at the whole repository, and a run writes its own predictions file before the check, so every full run was "dirty". Not chosen: `--untracked-files=no` over the whole repository, which would miss a new model file used before `git add`. Logs get `schema` 2; in logs with `schema` 1 (the trivial run of chat 03 and the Dense runs of chat 04) `git_dirty` is always true and carries no information. Old logs are not rewritten. Code: `src/runlog.py::git_dirty`; tests: `tests/test_runlog.py`.

## D15. Dense block (T9) — DECIDED (2026-11-20)

Proposed by the assistant in chat DL 04 and accepted by OF and MM, who asked for the most suitable option and for the complete (final-version) set of experiments at once. Any value below changes only with a dated entry. Code: `src/models/dense.py`, `src/dense_experiments.py`, `tests/test_dense.py`.

**Architecture**
- The input (batch, 12, 1000) is flattened into 12,000 numbers, lead after lead. Each first-layer unit compares the whole record with one weight pattern, so a beat that appears at another time meets other weights (no translation invariance). This is the limitation of feed-forward networks that motivates the CNN (weekly summary 30/10).
- Hidden block: Linear → BatchNorm1d → ReLU → Dropout. Batch norm sits before the activation so that the pre-activations stay "close to the zero region where the activation functions usually work" (weekly summary 30/10); that this is the placement of the original paper (Ioffe and Szegedy 2015) is **[verify]**. The Linear layers keep their bias also with batch norm (redundant but harmless; the logged parameter counts include it). PyTorch's default initialisation.
- Output: Linear layer with 5 logits, no sigmoid (D2).
- Baseline: 2 hidden layers of 256 units, batch norm on, dropout 0.5 after each hidden layer, per-lead z-score of the input on (D3). About 3.1 M parameters, 98 % of them in the first layer (12,000 × 256). Dropout 0.5 as the usual rate for hidden layers: **[verify]** (Srivastava et al. 2014, cited on FeedForward slide 42).
- Training: the D14 defaults, the same for every configuration (Adam, learning rate 1e-3, batch 64, at most 50 epochs, patience 5 on validation macro AUROC, no weight decay, no class weights), so that only the ablated factor changes.

**Ablations.** The baseline has every component switched on; each configuration changes one factor and keeps everything else equal.

| Tag | Factor | Change from the baseline |
|---|---|---|
| `base` | — | — |
| `no_norm` | input scaling | no z-score: raw mV, each lead on its own scale (D3 option) |
| `no_bn` | batch norm | no batch norm |
| `no_norm_no_bn` | input scaling × batch norm | both off: the only two-factor configuration, to see whether batch norm after the first layer makes up for the missing input scaling (weekly summary 30/10 links the two) |
| `no_dropout` | dropout | rate 0 |
| `dropout02` | dropout | rate 0.2 |
| `depth1`, `depth3`, `depth4` | depth | 1, 3 or 4 hidden layers of 256 units |
| `width64`, `width1024` | width | 2 hidden layers of 64 or 1024 units |

**Protocol**
- Every configuration is run with the three seeds 42, 43 and 44 on validation (33 runs). For this block this replaces the D4 rule "1 seed, 3 if the difference is within the CI": the runs are cheap and three seeds show how much a result moves from seed to seed.
- Reported per configuration: mean and standard deviation over the seeds of the validation macro AUROC; the patient-bootstrap interval of seed 42; and, for each seed, the paired bootstrap interval of the difference with the baseline trained with the same seed (D14). A factor is said to change the result only when that interval excludes 0; the table shows in how many of the three seeds it is better and in how many worse.
- Final Dense model: the configuration with the highest mean validation macro AUROC over the three seeds, chosen before the test fold is touched (D3). It is trained again with `--final` for seeds 42, 43 and 44, so the test fold is evaluated once per seed. Reported on test: mean ± SD over the seeds (macro and per class), the intervals of the first seed, and the paired difference with the trivial model (D14) for each seed.
- Interpretation: the first-layer weights of the final model (seed 42) drawn as 12 × 1000 patterns, for the units with the largest weight norm (FeedForward slides 29-30).

**Outputs**
- `python -m src.dense_experiments run | final | report` (`all` runs the three steps; `run` and `final` skip what has already been run, so they can be resumed).
- `results/logs/dense_summary.json`: every number of the block in `scalars`; the script adds the alias `dense` to `report/runs.json`, so the macros are `\dense...`.
- `report/tables/dense_ablations.tex` and `report/tables/dense_final.tex`: made of macros only, so no number is typed by hand (new folder `report/tables/`).
- `report/figures/dense_curves`, `dense_ablations`, `dense_first_layer` (PDF and PNG).
- With `--fast` the same outputs end in `_fast` and git ignores them.

## D16. CNN block (T10) — DECIDED (2026-11-20)

Proposed by the assistant in chat DL 05 and accepted by OF and MM, who asked for the most suitable option and for the complete (final-version) set of experiments at once. Same protocol as D15 so the two blocks can be compared. Any value below changes only with a dated entry. Code: `src/models/cnn.py`, `src/cnn_experiments.py`, `tests/test_cnn.py`.

**Architecture**
- Input (batch, 12, 1000): the 12 leads are the input channels and time is the axis the filters slide along (1D convolution, as in the 1D model of `MLLB_Pytorch_HAR_CNN_TODO.ipynb`). Per-lead z-score on (D3); not ablated, because the Dense block showed that batch norm makes the input scaling irrelevant (D15, `docs/blocks/04_dense.md`).
- Block (ResNet basic block, `PRDL_MLLB_CNN.pptx` slides 46-62): Conv1d(k) → BatchNorm1d → ReLU → Conv1d(k) → BatchNorm1d, plus a shortcut, then ReLU and MaxPool1d(2). The shortcut is the identity when the number of channels does not change (no parameters, slide 55) and a 1×1 convolution + batch norm when it does (slide 56). "Same" padding (k // 2, odd k) keeps the length so that the two paths can be added. That this is exactly the basic block of He et al. 2016 is **[verify]**. Convolutions keep their bias also with batch norm, as in D15.
- D6 described the CNN baseline as Conv-BN-ReLU-pool blocks; D16 makes each block two convolutions with a skip connection, so that "residual on/off" changes only the shortcut (the same rule as D15: the baseline has every component switched on).
- Head: global average pooling over time → dropout 0.5 → Linear with 5 logits, no sigmoid (D2). Dropout 0.5 before the classifier as in the class notebook's 1D model.
- Baseline: 4 blocks with 32, 64, 128, 128 channels, kernel 7, batch norm, residual connections, global average pooling, dropout 0.5. 468,325 parameters (counted by the assistant; the log is the reference), about 1/7 of the Dense baseline. Length 1000 → 62 after the four poolings.
- Kernel 7 = 70 ms at 100 Hz, in the order of a QRS complex (about 80-100 ms: **[verify]**). The receptive field of one position of the last feature map is 196 samples (about 2 s, two to three beats); it is computed by `CNN1D.receptive_field()` and logged for every configuration.
- Training: the D14 defaults, identical to D15 (Adam, learning rate 1e-3, batch 64, at most 50 epochs, patience 5 on validation macro AUROC, no weight decay, no class weights).

**Ablations.** One factor at a time; the column "reference" is the configuration each one is compared with (paired bootstrap, D14).

| Tag | Factor | Change | Reference |
|---|---|---|---|
| `base` | — | — | — |
| `kernel3`, `kernel15` | kernel size | kernel 3 or 15 (receptive field 76 or 436 samples) | `base` |
| `depth2`, `depth6` | number of blocks | 2 blocks (32, 64) or 6 blocks (32, 64, 128, 128, 128, 128); receptive field 40 or 820 samples | `base` |
| `no_bn` | batch norm | no batch norm (also on the shortcut) | `base` |
| `no_residual` | residual connections | no shortcut | `base` |
| `depth6_no_residual` | residual connections × depth | 6 blocks without shortcut: tests the slides' claim that skip connections matter more as the network gets deeper | `depth6` |
| `flatten` | global pooling vs flatten | flatten the last feature map (128 × 62) into the classifier: one weight per position, no translation invariance in the head | `base` |
| `no_dropout` | dropout | rate 0 | `base` |

**Protocol** (as D15)
- Every configuration with seeds 42, 43, 44 on validation (30 runs). Reported per configuration: mean ± SD of the validation macro AUROC, the patient-bootstrap interval of seed 42, and for each seed the paired interval of the difference with its reference trained with the same seed; a factor changes the result only when that interval excludes 0 (table: seeds better / worse).
- Final CNN: the configuration with the highest mean validation macro AUROC, chosen before the test fold is touched (D3), trained again with `--final` for seeds 42, 43, 44. Reported on test: mean ± SD (macro and per class), intervals of the first seed, and for each seed the paired difference with the trivial model and with the final Dense model trained with the same seed (both on the whole test fold; skipped with `--fast`).

**Interpretation** (final model, seed 42)
- First-layer filters (`cnn_first_layer`): (a) the 8 filters with the largest weight norm, drawn as 12 leads × k taps; (b) the magnitude frequency response of every first-layer filter (RMS over leads, 0-50 Hz), sorted by peak frequency: the first layer read as a learnt filter bank (`PRDL_MLLB_CNN.pptx` slides 99-102).
- Feature maps (`cnn_feature_maps`): one validation record per superclass, among the records with a single superclass, drawn with seed 42. Rows: lead II of the (normalised) input; output of block 1 and of the last block (channels × time, aligned with the input), captured with forward hooks as in `Gemini_MNIST_Pytorch_featuremaps.ipynb`; and the class activation map of the record's class, only if the final model uses global average pooling (Zhou et al. 2016: **[verify]**). The CAM needs no extra training: with average pooling and one Linear layer, the logit equals the time mean of the CAM plus the bias (checked in `tests/test_cnn.py`).

**Outputs**
- `python -m src.cnn_experiments run | final | report` (`all` runs the three steps; `run` and `final` skip what has already been run).
- `results/logs/cnn_summary.json` (alias `cnn` in `report/runs.json`, macros `\cnn...`); `report/tables/cnn_ablations.tex` and `cnn_final.tex` (macros only); figures `cnn_curves`, `cnn_ablations`, `cnn_first_layer`, `cnn_feature_maps` (PDF and PNG). With `--fast` the outputs end in `_fast` and git ignores them.
- Generic helpers (paired bootstrap of two runs, macro names, figure style) are imported from `src/dense_experiments.py`, not copied.

**2026-11-20 (OF + MM, chat DL 05), after the full runs (commit 3f9439f): numbers added to the summary.** No run is repeated and no existing number changes (the summary was rebuilt and its 290 previous numbers came out identical); only `report` is run again.
- Training loss at epoch 5 for every configuration (`train_loss_ep5_mean_<tag>`) and its mean difference with the reference trained with the same seed (`train_loss_ep5_diff_mean_<tag>`, `n_seeds_train_loss_higher_<tag>`). With patience 5 every run lasts at least 6 epochs, so epoch 5 exists in all runs. Reason: the slides' claim about skip connections is about optimisation (training error), which the validation AUROC does not show. The loss is the training-mode loss logged by the loop (dropout active), so configurations with a different dropout are not comparable on it.
- First-layer filters of the final model (seed 42): median peak frequency and number of filters peaking below 5 Hz and above 20 Hz (`first_layer_*`).
- Example records of `cnn_feature_maps`: the probability the final model (seed 42) gives to each record's class and the class threshold (`example_prob_<class>`, `example_threshold_<class>`, `example_ecg_id_<class>`), also printed in the figure's column titles. Reason: some examples are misclassified, and their class activation map must be read with that in mind.

## D17. RNN block (T11) — DECIDED (2026-11-21)

Proposed by the assistant in chat DL 06 and accepted by OF and MM, who asked for the most suitable option and for the complete (final-version) set of experiments at once. T11 is done by MM and reviewed by OF. Same protocol as D15 and D16, so the three blocks can be compared. Any value below changes only with a dated entry. Code: `src/models/rnn.py`, `src/rnn_experiments.py`, `tests/test_rnn.py`.

**Input: the signal as a sequence**
- An RNN reads one vector per time step: `nn.LSTM(batch_first=True)` expects (batch, time, features), the opposite axis order of `Conv1d` (batch, channels, time), so the record is permuted (weekly summary 13/11).
- Baseline: non-overlapping **frames of 10 samples** (100 ms). At each step the RNN receives the 12 leads × 10 samples of one frame as a vector of 120 numbers, and the 10 s record becomes 100 steps. D6 wrote "(downsampled) sequence"; frames shorten the sequence in the same way without discarding any sample. Downsampling by 4 (25 Hz) would remove everything above 12.5 Hz, where the first CNN layer still kept filters (D16). Backpropagation through time then runs over 100 steps instead of 1000 (RNN slides 40-48: the unfolded network is a very deep feed-forward network with shared weights, and gradients vanish as they go back in time).
- Ablation `raw`: one sample per step (12 numbers, 1000 steps).
- Ablation `conv`: a **convolutional front end**, the first three blocks of the CNN baseline (`ConvBlock`, D16: 32, 64, 128 channels, kernel 7, batch norm, residual), turns the record into 125 steps of 128 features (one step every 8 samples, each seeing 92 input samples). The RNN reads that sequence: a CNN feature extractor followed by an LSTM (RNN slide 80).
- Per-lead z-score on (D3), not ablated.
- No padding: every record has 1000 samples, so all sequences have the same length and `pack_padded_sequence` is not needed (weekly summary 13/11 warns that zero padding is processed as real input).

**Recurrent network**
- Baseline: `nn.LSTM`, 2 stacked layers of 128 units, unidirectional, `batch_first=True`. The `dropout` argument of `nn.LSTM` only acts between stacked layers (PyTorch documentation; class notebook `MLLB_Pytorch_Embeddings_RNN_SentimentAnalysis.ipynb`) and is left at 0. PyTorch's default initialisation.
- Readout: how the sequence of hidden states becomes one vector for the classifier.
  - `last` (baseline, D6): final hidden state of the top layer, `h_n[-1]`, equal to `out[:, -1, :]` for a unidirectional network: the non-autoregressive output of weekly summary 6/11. Bidirectional: concatenation of the top layer's forward final state (`h_n[-2]`, which has read the record left to right) and backward final state (`h_n[-1]`, which has read it right to left; it sits at `out[:, 0, hidden:]`, not at `out[:, -1, :]`). Per the PyTorch documentation, forward and backward are directions 0 and 1. Checked in `tests/test_rnn.py`.
  - `mean`: mean of `out` over time (the pooling suggested at the end of the PyTorch RNN notebook).
  - `attention`: attention pooling over `out` (below).
- Head as in the CNN (D16): dropout 0.5, then a Linear layer with 5 logits and no sigmoid (D2).
- Baseline: 260,741 parameters (formula checked by a test; the log is the reference), about half of the CNN baseline.

**RNN + attention (the model of its own required by D6, entry of 20/11)**
- As in the class notebook `TODO_MLLB_Embeddings_GRU_Attention_SentimentAnalysis.ipynb`: one score per time step, a softmax over time gives weights α_t ≥ 0 that sum to 1, and the context vector Σ α_t h_t replaces the last state as the input of the classifier. In translation (Attention slides 12-15) the score compares the decoder state (query) with each encoder state (key). A classifier has no decoder, so the query is a learned vector, the same for every record, as in the notebook.
- Score of the attention configurations: additive (Bahdanau) attention, e_t = vᵀ tanh(W h_t + b), with a hidden layer of 64 units (Attention slide 22: "a feed-forward network with a single hidden layer").
- Ablation `attn_tanh`: the notebook's score, e_t = tanh(wᵀ h_t + b) (`Dense(1, activation='tanh')`). Its scores lie in [-1, 1], so no weight can be more than e² ≈ 7.4 times another. With 100 steps the largest possible weight is 1 / (1 + 99 e⁻²) ≈ 0.069, against a uniform 0.01. This configuration tests whether that bound pushes the attention towards mean pooling.

**Training**
- D14 defaults, as D15 and D16: Adam with learning rate 1e-3, batch 64, at most 50 epochs, patience 5 on validation macro AUROC, no weight decay, no class weights.
- Added for every configuration: **gradient clipping** at a total norm of 1.0 (RNN slide 49: exploding gradients, "clip or normalize values of the gradients"). The value 1.0 is our choice and is not tuned. Clipping is the same in every configuration, so it does not change the comparisons, and it is not ablated.
- GPU determinism: the PyTorch documentation of `nn.LSTM` warns of non-determinism of RNN functions on some cuDNN/CUDA versions and suggests `CUBLAS_WORKSPACE_CONFIG=:4096:2`. `src/rnn_experiments.py` sets it before CUDA starts (unless it is already set) and writes it to each run's log. The D14 tolerance still applies.

**Ablations.** One factor at a time; "reference" is the configuration each one is compared with (paired bootstrap, D14). The 1000-step configurations run last because they are the slowest.

| Tag | Factor | Change | Reference |
|---|---|---|---|
| `base` | — | LSTM 2 × 128, unidirectional, 100 frames of 100 ms, last state | — |
| `gru` | cell | GRU instead of LSTM | `base` |
| `layers1` | depth | 1 recurrent layer | `base` |
| `bi` | direction | bidirectional (128 units per direction) | `base` |
| `mean` | readout | mean of the states over time | `base` |
| `attn` | readout | additive attention | `base` |
| `attn_tanh` | attention score | the notebook's tanh score | `attn` |
| `gru_attn` | attention × cell | GRU + additive attention (the architecture of the class notebook) | `gru` |
| `bi_attn` | attention × direction | bidirectional + additive attention | `bi` |
| `conv` | input | convolutional front end (125 steps of 128 features) | `base` |
| `raw` | sequence length | one sample per step (1000 steps) | `base` |
| `raw_attn` | attention × sequence length | 1000 steps + additive attention | `raw` |

Why the three "attention ×" configurations: attention was introduced because a single final vector is a bottleneck for long sequences (Attention slide 12). If that is what happens here, attention should help more where the last state struggles: with 1000 steps more than with 100, and less in a bidirectional network, whose summary already contains a state that has read the beginning of the record.

**Protocol** (as D15 and D16)
- Every configuration with seeds 42, 43, 44 on validation (36 runs). Reported per configuration: mean ± SD of the validation macro AUROC, the patient-bootstrap interval of seed 42, and for each seed the paired interval of the difference with its reference trained with the same seed. A factor changes the result only when that interval excludes 0; the table shows in how many seeds it is better and in how many worse. The training loss at epoch 5 and its difference with the reference are logged too, as in D16.
- **Two final models**, both chosen by rules fixed now, before the test fold is touched (D3):
  1. the final RNN: the configuration with the highest mean validation macro AUROC among all twelve (same rule as D15 and D16);
  2. the final RNN + attention: the highest mean among the five attention configurations (`attn`, `attn_tanh`, `gru_attn`, `bi_attn`, `raw_attn`).

  When the two coincide there is only one final model. Each is trained again with `--final` for seeds 42, 43, 44.
- Reported on test: mean ± SD over the seeds (macro and per class), the intervals of the first seed, and for each seed the paired difference of the final RNN with the final CNN (D16), the final Dense (D15) and the trivial model. If the attention model is different, also its paired differences with the final RNN and with the final CNN.
- Before the long run, `python -m src.rnn_experiments estimate` measures the time per batch of every configuration on the laptop's GPU and prints the expected duration.

**Interpretation**
- *Input sensitivity over time* (`rnn_time_gradient`): for the seed-42 model of every configuration we take |∂ logit_c / ∂ x(lead, t)|, sum it over the leads, average it over the 5 classes and over 512 validation records drawn with seed 42, and normalise it to sum 1 over the 10 s. It shows which part of the record the output depends on. With the last state of a unidirectional RNN, vanishing gradients (slide 48) should concentrate it at the end of the record; bidirectional, mean and attention readouts should spread it. Summary numbers: share in the first and in the last 2 s (0.2 each if uniform), and the shortest final window that holds half of it (5 s if uniform). It is computed on the CPU in eval mode (dropout off), because cuDNN only runs the backward pass of an RNN in training mode.
- *Attention weights* (`rnn_attention`), seed-42 run of the final RNN + attention: lead II of the same five validation records as the CNN feature maps (one per superclass, D16), with α_t of each step drawn as bars. A second row shows the same records with the other score function (`attn_tanh` or `attn`, seed-42 ablation run).
- Attention statistics (`rnn_attention.tex`), for every attention configuration, seed 42, over the whole validation fold:
  - normalised entropy H(α) / log T (1 = uniform, i.e. the same as mean pooling);
  - effective number of attended steps, exp(H);
  - mean of the largest weight;
  - Spearman correlation between α_t and the signal's activity in that step (mean absolute sample-to-sample change over the 12 z-scored leads), averaged over the records.

  The correlation is our proxy for "attention goes where the signal changes fast", such as the QRS complexes; it is not a QRS detector.
- Learning curves (`rnn_curves`) and the ablation figure (`rnn_ablations`) as in D16.

**Outputs**
- `python -m src.rnn_experiments estimate | run | final | report | all` (`all` = run + final + report; `run` and `final` skip what has already been run, so they can be resumed).
- `results/logs/rnn_summary.json` (alias `rnn` in `report/runs.json`, macros `\rnn...`).
- Tables (macros only): `report/tables/rnn_ablations.tex`, `rnn_attention.tex`, `rnn_final.tex`. The final table also uses `\cnn...`, `\dense...` and `\trivial...` macros.
- Figures `rnn_curves`, `rnn_ablations`, `rnn_time_gradient`, `rnn_attention` (PDF and PNG). With `--fast` the outputs end in `_fast` and git ignores them.
- Generic helpers (paired bootstrap, run search, comparison of final models, example records, figure style) are imported from `src/dense_experiments.py` and `src/cnn_experiments.py`, not copied.

## D18. Self-attention and Transformer block (T12) — DECIDED (2026-11-21)

Proposed by the assistant in chat DL 07 and accepted by OF and MM, who asked for the most suitable option and for the complete (final-version) set of experiments at once. T12 is done by MM and reviewed by OF. Same protocol as D15-D17, so the four blocks can be compared. Any value below changes only with a dated entry. Code: `src/models/attention.py` (shared with chat DL 13), `src/models/transformer.py`, `src/transformer_experiments.py`, `tests/test_transformer.py`.

**Input: the signal as a sequence of patches**
- The record is cut into non-overlapping **patches of 10 samples** (100 ms); each patch (12 leads × 10 samples, flattened lead by lead, exactly as the RNN frames of D17) becomes one token. A Linear layer (120 → d_model) is the patch embedding: the "input embedding" of the architecture slides, with a real-valued patch instead of a word index. It is the same as a `Conv1d` with kernel = stride = 10 (checked by a test). 1000 samples → T = 100 tokens, the same steps as the RNN baseline, so the comparison "attention instead of recurrence" is direct.
- Patch length is an ablation (5, 25, 50 samples → 200, 40, 20 tokens). One sample per token (1000 tokens) is left out: the attention matrix grows with T² (Part I slides 102-103), and with batch 64, 4 heads and 1000 tokens each attention matrix alone takes about 1 GB per layer, more than the laptop's 4 GB GPU can hold with the copies kept for backpropagation. `estimate` prints the peak GPU memory of every configuration.
- Per-lead z-score on (D3), not ablated.

**Encoder (the classes of the notebooks `Transformer_Architecture_Intro` 01-05, rewritten for this task)**
- Scaled dot-product attention written by hand, as in notebooks 01-02: Q = XW_Q, K = XW_K, V = XW_V (Linear layers without bias, as in the notebooks), weights softmax(QKᵀ/√d_k) over the keys, output = weights · V. Multi-head: d_model is split into h heads of d_model/h dimensions, each head attends on its own, the heads are concatenated and mixed by W_O (notebook 02; Part I slides 51-58). Not `nn.MultiheadAttention`, so that every line can be explained; a test checks that our module gives the same output as `nn.MultiheadAttention` with the same weights (the "optional, advanced" cell of notebook 02).
- Optional causal mask (scores of future keys set to −∞ before the softmax, Part II slides 11-14), unused here and tested, so that chat DL 13 can stack the same blocks as a decoder-only model.
- Feed-forward network: Linear(d_model, 4·d_model) → GELU → Linear(4·d_model, d_model), applied to every token separately (notebooks 03 and 05; Part I slides 25-27 and 70: attention mixes information between tokens, the MLP within each token, and expands the dimension by 4).
- Block: **Pre-LN**, x = x + Dropout(MHSA(LN(x))), then x = x + Dropout(FFN(LN(x))), and a final LayerNorm after the last block (notebook 05; Part II slides 35-36: Pre-LN "makes the optimization much easier"; Xiong et al. 2020, arXiv 2002.04745, cited there: **[verify]**). LayerNorm rather than batch norm: it normalises each token over its features, independently of the batch (Part II slides 27-34).
- **Positional encoding**: sinusoidal, PE(pos, 2i) = sin(pos / 10000^(2i/d)), PE(pos, 2i+1) = cos(·), added to the patch embeddings (Part II slides 18-23). Without it, self-attention does not know the order of the tokens (Part II slide 11): with mean pooling the whole network would give the same output for any shuffling of the patches (checked by a test). Not learned in the baseline because the slides report that learned and sinusoidal encodings give "nearly identical results"; both alternatives are ablations.
- **Readout**: mean of the token outputs over the sequence (as the class notebook `MLLB_Text_Classification_Transformer.ipynb`, `GlobalAveragePooling1D`; mean pooling was also as good as attention in the RNN, D17). CLS token is an ablation.
- Head as in D16 and D17: dropout 0.5, then a Linear layer with 5 logits, no sigmoid (D2).
- **Baseline**: d_model 128, 4 heads (32 dimensions each), 2 blocks, FFN 512, dropout 0.1 on the output of each attention and FFN sub-layer before it is added to the residual stream (rate of the class notebook's Transformer block; that 0.1 is the rate of Vaswani et al. 2017 is **[verify]**), no dropout on the attention weights. About 0.41 M parameters (the log is the reference), between the RNN (0.26 M) and the CNN (0.47 M) baselines. PyTorch's default initialisation; learned position embeddings use `nn.Embedding` as in the class notebook; the CLS token starts at zero.

**Self-attention step by step (area 1 of the professor's message; D6 entry of 20/11)**

The study rebuilds the block one component at a time, in the order of the class notebooks, each step trained like every other configuration:
1. `sa_single`: a single self-attention layer with **one head**, no W_O, no residual connection, no LayerNorm, no FFN (notebook 01's `SelfAttention`), then mean pooling and the head. With mean pooling this model is exactly an attention pooling: its output is Σ_j ā_j v_j, where ā_j is the attention that patch j receives, averaged over the queries. So ā is the pooling weight of patch j, like α_t in the RNN + attention model (D17), but computed by comparing patches with each other instead of with a learned query (checked by a test).
2. `mhsa`: 4 heads and W_O (notebook 02).
3. `no_ffn`: plus "Add & Norm" (residual connection and Pre-LN).
4. `layers1`: plus the FFN = one complete Transformer block (notebooks 03, 05).
5. `base`: two blocks.

Part I slides 73 and 85 state that attention alone is a "linear(ish) mixing" of the inputs and that the FFN supplies the non-linear computation; steps 1-4 test this on the ECG.

**Training**
- D14 defaults, as D15-D17 (Adam, learning rate 1e-3, batch 64, at most 50 epochs, patience 5 on validation macro AUROC, no weight decay, no class weights), plus gradient clipping at total norm 1.0 as in D17, the same in every configuration and not ablated. No learning-rate warm-up: Pre-LN is the variant meant to train without it (Part II slide 36), and the `post_ln` ablation shows what happens to the original placement under the same schedule. If the full baseline does not train (validation AUROC stuck near 0.5), this is revised with a dated entry before any other run.
- `CUBLAS_WORKSPACE_CONFIG=:4096:2` as in D17 (attention is made of matrix products, which run on cuBLAS), written to each log.

**Ablations.** One factor at a time; "reference" is the configuration each one is compared with (paired bootstrap, D14).

| Tag | Factor | Change | Reference |
|---|---|---|---|
| `base` | — | Pre-LN, 2 blocks, d_model 128, 4 heads, patches of 10 (100 tokens), sinusoidal encoding, mean | — |
| `sa_single` | step by step: heads and W_O | 1 layer, 1 head, no W_O, no Add & Norm, no FFN | `mhsa` |
| `mhsa` | step by step: Add & Norm | 1 layer, 4 heads + W_O, no Add & Norm, no FFN | `no_ffn` |
| `no_ffn` | step by step: FFN | 1 layer with Add & Norm, no FFN | `layers1` |
| `layers1` | depth | 1 complete block | `base` |
| `layers4` | depth | 4 blocks | `base` |
| `heads1`, `heads8` | heads | 1 head (128 dimensions) or 8 heads (16 dimensions); same number of parameters | `base` |
| `no_pos` | positional encoding | none: the network sees a bag of patches | `base` |
| `learned_pos` | positional encoding | learned position embeddings (class notebook) | `base` |
| `patch5`, `patch25`, `patch50` | patch length | 200, 40 or 20 tokens | `base` |
| `cls` | readout | CLS token (a learned token prepended to the patches; its output feeds the head) | `base` |
| `post_ln` | LayerNorm placement | Post-LN, x = LN(x + sub-layer(x)), the original Transformer and the class text notebook; no final LayerNorm | `base` |

In the step-by-step rows the Δ is the effect of *removing* the component (the configuration minus the next step). "Add & Norm" (residual + LayerNorm) counts as one factor, as one box of the architecture diagram (Part I slide 10). One extra paired comparison is reported: `sa_single` vs `layers1` (one self-attention layer vs one complete block), the comparison D6 names.

**Protocol** (as D15-D17)
- Every configuration with seeds 42, 43, 44 on validation (45 runs). Reported per configuration: mean ± SD of the validation macro AUROC, the patient-bootstrap interval of seed 42, and for each seed the paired interval of the difference with its reference trained with the same seed; a factor changes the result only when that interval excludes 0 (table: seeds better / worse). Training loss at epoch 5 and its difference with the reference, as D16 and D17.
- **One final model**: the configuration with the highest mean validation macro AUROC among the fifteen, fixed now, before the test fold is touched (D3), trained again with `--final` for seeds 42, 43, 44. The step-by-step study is a validation study: its models do not go to the test fold unless one of them is selected.
- Reported on test: mean ± SD over the seeds (macro and per class), the intervals of the first seed, and for each seed the paired difference with the final CNN (D16), the final RNN and the final RNN + attention (D17), the final Dense (D15) and the trivial model (all on the whole test fold; skipped with `--fast`).
- Before the long run, `python -m src.transformer_experiments estimate` measures time per batch and peak GPU memory of every configuration.

**Interpretation** (seed-42 model of each configuration unless stated)
- *Patch order* (new): validation macro AUROC with the patches of every record shuffled (the same random permutation of the T patches for all leads of a record, drawn with seed 42), minus the normal validation AUROC. It shows how much a model uses the order of the patches. It is 0 by construction for `no_pos` (bag of patches), which the summary also checks.
- *Input sensitivity over time*: the same analysis as D17 (|∂ logit / ∂ input| over the 10 s, 512 validation records drawn with seed 42), so it can be set beside the RNN's: a recurrent network that reads its last state depends mostly on the end of the record; self-attention has no such built-in recency.
- *Attention statistics* (`transformer_attention.tex`), over the whole validation fold, for the first and the last layer, heads averaged, patch tokens only (with `cls` the CLS row and column are left out and the rows renormalised):
  - mean attention distance: Σ_j α_ij |i − j| in seconds, averaged over queries, heads and records, beside its value for uniform weights; small = local attention, like a convolution;
  - normalised entropy of each query's weights, H(α_i·) / log T (1 = uniform);
  - share of each query's weight on its own patch (the diagonal);
  - normalised entropy of the attention received, ā_j (the pooling weight of `sa_single`, see above).
- *Attention over the ECG* (`transformer_attention`): lead II of the same five validation records as D16 and D17 (one per superclass), with ā_j of each patch drawn as bars, for `sa_single` (row 1) and the final model's last layer (row 2). Probabilities and thresholds in the column titles, as D16 and D17.
- *Attention maps* (`transformer_attention_maps`): the full T × T maps (query time × key time, in seconds) of one of those records: `sa_single`; the final model's first and last layer (heads averaged); and each head of its last layer (at most four).
- Learning curves (`transformer_curves`: baseline losses; validation AUROC of the step-by-step configurations; training loss with Pre-LN vs Post-LN) and the ablation figure (`transformer_ablations`) as in D16 and D17.

**Outputs**
- `python -m src.transformer_experiments estimate | run | final | report | all` (`all` = run + final + report; `run` and `final` skip what has already been run).
- `results/logs/transformer_summary.json` (alias `transformer` in `report/runs.json`, macros `\transformer...`).
- Tables (macros only): `report/tables/transformer_ablations.tex`, `transformer_attention.tex`, `transformer_final.tex`. The final table also uses `\cnn...`, `\rnn...`, `\dense...` and `\trivial...` macros.
- Figures `transformer_curves`, `transformer_ablations`, `transformer_time_gradient`, `transformer_attention`, `transformer_attention_maps` (PDF and PNG). With `--fast` the outputs end in `_fast` and git ignores them.
- Generic helpers are imported from `src/dense_experiments.py`, `src/cnn_experiments.py` and `src/rnn_experiments.py`, not copied.

**Not chosen**
- A convolutional stem before the encoder: the best RNN was such a hybrid (D17), and the final Transformer would then compare "CNN + attention" with the CNN. The block studies attention itself.
- `nn.TransformerEncoderLayer`: it hides the attention weights and the order of the operations that the defence asks about.
- Learning-rate warm-up, weight decay, AdamW: they would make this block's training differ from the other three.

## D19. Generative block: ECG-GPT and LSTM generators (T23) — DECIDED (2026-11-21)

Proposed by the assistant in chat DL 13 and accepted by OF and MM, who asked for the most suitable option and for the complete (final-version) set of experiments at once. O6 closed: T23 is done by OF and reviewed by MM. It completes the open points of D13 ([open] quantisation, clipping range, heart rate). Same protocol as D15-D18 where it applies. Any value below changes only with a dated entry. Code: `src/generative.py` (data, task, baselines, generation, statistics), `src/models/generative.py` (models), `src/generative_experiments.py` (runs, summary, tables, figures), `tests/test_generative.py`.

**Data (D13, D12 rule c)**
- Lead II at 100 Hz, in mV as stored in the cache, without filtering or per-record normalisation (as every other block, D3; amplitude is diagnostic).
- Training: every record of folds 1-8, also those without a diagnostic superclass (labels are not used). Validation: fold 9, records with a superclass, i.e. the same validation records as every classifier and as chat DL 08. **The test fold is never read** by this block (no `final` step); chat DL 08 uses it for the 5-superclass task only.

**Tokens: uniform quantisation on the microvolt grid**
- PTB-XL stores every sample as an integer number of microvolts (gain 1000 per mV, `docs/blocks/02_eda.md`); the code checks it on the training data (`frac_off_uv_grid_train`). A value x becomes u = round(1000 x) µV.
- Range: 2.56 mV (2560 µV), the same for every vocabulary, centred on the midpoint of the 0.1 % and 99.9 % quantiles of the training values of lead II. Values outside it go to the first or last token (clipping); the clipped fractions of training and validation are logged. Why 2.56 mV: the 0.5-99.5 % range of lead II is about 1.3 mV (`eda.json`, `lead_stats_train`), but R peaks are only a few samples per beat and the 95th percentile of the peak-to-peak amplitude is about 2 mV, so a range set by the 0.5-99.5 % quantiles would clip tall R waves (e.g. HYP records).
- V levels of Δ = 2560 / V µV: token k holds the Δ integers lo + kΔ ... lo + (k+1)Δ − 1 and is decoded to their centre. Baseline V = 256, Δ = 10 µV (largest error 4.5 µV, against a lead II standard deviation of about 160 µV). Ablations: V = 64 (Δ = 40 µV) and V = 2560 (Δ = 1 µV: every microvolt value of the range is its own token, so nothing is lost inside the range).
- Not chosen: mu-law companding (finer levels near the baseline). With 10 µV levels the uniform error is already negligible, and uniform integer levels keep the vocabulary-independent metric below exact.

**Metric comparable across vocabularies: bits per sample**
- Cross-entropy per token (and perplexity, D14) depends on V: coarser tokens are easier to guess. The block also reports **bits per sample** = (CE + ln Δ) / ln 2: the cost in bits of coding the microvolt value of the next sample, if every value inside a token is equally likely. Uniform guessing gives log2(2560) = 11.3 bits whatever V is; with Δ = 1 it is the cross-entropy in bits. Configurations are compared with it; the patient bootstrap of D4/D14 is applied to it (each run stores the summed cross-entropy of every validation record).
- Also logged per run (validation): perplexity; the cross-entropy at every position of the window (position t predicts with t + 1 samples of context); the cross-entropy at the last position (the only position whose target is outside the window, so an honest prediction even without the causal mask); RMSE in µV of the expected value Σ p_k c_k against the centre of the target token.
- Baselines without training, on the same validation records: uniform; unigram and bigram language models made by counting the training tokens (add-one smoothing; RNN slides 93-94: 2-gram model), for each V; the persistence error "next sample = this sample" (µV).

**Windows and context length**
- Each training item is a window of L + 1 samples of one record starting at a random sample (drawn with the run's seed): input = the first L tokens, target = the same tokens shifted by one (teacher forcing). An epoch shows every training record once, through one random window. Baseline L = 250 (2.5 s, two or three beats); ablations L = 100 (1 s, about one beat) and L = 500 (5 s). The whole record (1000 tokens) is not used as context: the attention matrices of 1000 tokens do not fit the laptop's 4 GB GPU (D18).
- Validation: fixed windows of L that together cover every target 1..999 of every record (ceil(999 / L) evenly spaced windows; for L = 250 they start at 0, 250, 499, 749), the same at every epoch.

**Models** (same data, tokens, windows and training; about the same number of parameters; D13)
- **ECG-GPT** (baseline): token embedding (V × 128, one learned vector per level, as one vector per character in the class notebooks) + sinusoidal positional encoding of the position in the window (Part II slides 18-23; for generation order matters, unlike the classifier of D18) + 2 Pre-LN blocks of chat DL 07 (`TransformerBlock(causal=True)`: d_model 128, 4 heads, FFN 512, dropout 0.1) + final LayerNorm + Linear(128, V). 461,568 parameters with V = 256. The causal mask sets the scores of future positions to −∞, so position t only sees tokens 0..t and the whole window is trained in one parallel pass (Part II slides 11-14).
- **LSTM**: the same token embedding (128) + 2-layer LSTM with 160 units + dropout 0.1 + Linear(160, V). 465,664 parameters: the hidden size is chosen for parameter parity with the ECG-GPT (+0.9 %). Causal by construction.
- No softmax in the models (CrossEntropyLoss contains it, `NextTokenTask`). Not chosen: weight tying between embedding and head, learned positions, `nn.TransformerDecoder` (it hides the attention weights).

**Training**: D14 defaults as D15-D18 (Adam, learning rate 1e-3, batch 64, at most 50 epochs, early stopping with patience 5 on the validation cross-entropy), gradient clipping at total norm 1.0 as D17-D18, the same in every configuration; `CUBLAS_WORKSPACE_CONFIG=:4096:2` as D17-D18.

**Ablations.** One factor at a time, compared with the ECG-GPT baseline trained with the same seed.

| Tag | Factor | Change |
|---|---|---|
| `base` | — | ECG-GPT, L = 250, V = 256 |
| `lstm` | architecture | LSTM generator of the same size |
| `ctx100`, `ctx500` | context length | L = 100 or 500 |
| `vocab64`, `vocab2560` | vocabulary | V = 64 (40 µV) or 2560 (1 µV) |
| `no_pos` | positional encoding | none (a causal model can still infer positions from the mask, [verify] in the literature) |
| `no_mask` | causal mask | removed: the network sees the token it must predict |

`no_mask` is a demonstration of why the mask is needed: its teacher-forced loss is low because it copies the answer, so it is not compared by paired bootstrap and cannot be selected; the table reports its cross-entropy at the last position and its generated signals. Temperature and top-k are generation settings, not trained ablations (below).

**Protocol**
- Every configuration with seeds 42, 43, 44 (24 runs). Reported per configuration: mean ± SD of validation cross-entropy, perplexity and bits per sample; the patient-bootstrap interval of seed 42; for each seed the paired interval of the difference in bits with the baseline (better = whole interval below 0); the cross-entropy at the last position; the cross-entropy with 50-99 samples of context (available to every context length).
- **Pre-trained model for chat DL 08**, fixed now: the configuration with the lowest mean validation bits per sample among all except `no_mask`. Its three checkpoints (seeds 42-44), its tokenizer and `record_features()` (mean of the outputs before the head over the windows that cover the record) are handed over in the summary log (`pretrained_for_chat_08`).
- `python -m src.generative_experiments estimate` measures time per batch and peak GPU memory of every configuration before the long run.

**Generation** (seed-42 models)
- Prompt: the first 2 s (200 samples) of a validation record; 800 tokens (8 s) are generated and decoded to mV. 256 validation records drawn once with seed 42; another 256 drawn with it are a real-vs-real reference.
- Both models generate with the same sliding window: the next token is predicted from the last L tokens, appended, and the window moves by one, as the class notebook `TO_DO_MLLB_Keras_RNN_TextGeneration.ipynb` does. Not chosen: carrying the LSTM state (RNN slide 90), which would let the LSTM see more than the L samples it was trained on.
- Settings (`Sampling_in_Text_Generation.ipynb`): greedy search; temperature 0.5, 0.8, 1.0, 1.5; top-k with k = 10 at temperature 1. All six for the selected model and for the comparison model (the LSTM if a Transformer is selected, the baseline otherwise); temperature 1 for every configuration. A torch generator seeded with 42 makes sampling reproducible.
- Statistics of the generated 8 s against the real 8 s of the same records: median ratio of standard deviations; share of "flat" records (last 4 s with less than 10 % of the real standard deviation: the greedy loops of the notebook); Wasserstein-1 distance between amplitude distributions (mV), beside the same distance between two sets of real records; **heart rate from R peaks** (D13 [open], closed here): local maxima above median + ½ (maximum − median) of the real prompt, at least 0.3 s apart, heart rate = 60 / median RR. Reported: share of records with a heart rate, mean absolute difference from the real continuation. This is a simple threshold rule, not a validated QRS detector (records with negative QRS in lead II are not handled specially).

**Interpretation figures**: learning curves (including the training/validation/last-position loss without the mask), bits per configuration with the bigram baseline, bits per sample against the context available, real vs generated lead II for one validation record per superclass (the records of D16-D18), amplitude histograms and statistics per sampling setting, causal attention maps of one validation window.

**Outputs**
- `python -m src.generative_experiments estimate | run | report | all` (`all` = run + report; `run` skips what has already been run).
- `results/logs/generative_summary.json` (alias `generative` in `report/runs.json`, macros `\generative...`).
- Tables (macros only): `report/tables/generative_ablations.tex`, `generative_sampling.tex`. Figures `generative_curves`, `generative_ablations`, `generative_context`, `generative_samples`, `generative_sampling`, `generative_attention` (PDF and PNG). With `--fast` the outputs end in `_fast` and git ignores them.
- If the baseline does not beat the bigram baseline in bits per sample, the block is revised with a dated entry before anything else (and, if needed, the text fallback of D13).

**Not chosen**: the full record as context; mu-law; per-record normalisation or filtering; nucleus (top-p) sampling and beam search (named in the notebook, not needed for the comparison asked in D13); a KV cache for generation (speed only).

**2026-11-22 (OF + MM, chat DL 13), after `run` (24 runs at commit f010618, all with `git_dirty` false):**
- Almost every run stopped at the 50-epoch limit of D14 with its best epoch at or near 50 (`no_mask` stopped early): the generators were still improving slowly. Over the last 10 epochs validation bits per sample fell by about 0.005-0.02, much less than the differences between configurations, so the grid is kept as run (same budget for all, as D15-D18) and the report states that training was limited by the epoch budget, not by early stopping. Because an epoch is one window per record, longer contexts see more samples per epoch (L = 500: twice as many as L = 250); the context comparison is at equal epochs, not at equal samples.
- `ctx500` seed 44 has three epochs of several hours (the laptop slept); its results are unaffected. The summary reports the **median** time per epoch instead of the mean.
- `report` ran out of GPU memory generating with L = 500 (batches of 64 windows of 500 tokens). Generation now uses batches of 64 × (250 / L)² windows for L > 250 (16 for L = 500) and frees the GPU between models. Sampling stays seeded; the batch size of each context is fixed, so generation remains reproducible.

## Open items

| # | Item | Blocks |
|---|---|---|
| O5 | Weekly summary of 20/11 and the "simple autoregressive LLM-style model" notebook it mentions are not yet in `contexto_DL/`. D19 was built on the class notebooks that are there (`Sampling_in_Text_Generation.ipynb`, `TO_DO_MLLB_Keras_RNN_TextGeneration.ipynb`); add the summary when available and compare | Chat DL 13 (reference implementation) |

Closed: O2 (licence and folds verified on PhysioNet, 2026-11-14). O4 (repository under Oriana's account, 2026-11-20). O3 (records without superclass left out; D12, 2026-11-20). O1 (pretext task: autoregressive next-sample prediction, D7 and D13, 2026-11-20). O6 (T23 done by OF, reviewed by MM; D19, 2026-11-21).

## Change log

- 2026-11-14: first version; dataset decided.
- 2026-11-14: licence and recommended folds verified on PhysioNet by OF and MM (D1, D3). Transfer learning decided: own pre-trained model (D7), not confirmed with the professor.
- 2026-11-14: working organisation in chats added (D11).
- 2026-11-20: data location and integrity check (D1); likelihood rule and provisional O3 rule (D2); repository, owner and environment (D5); O4 closed. Decided by OF and MM in chat DL 01.
- 2026-11-20: normalisation rule (D3). Decided by OF and MM in chat DL 01.
- 2026-11-20: folder, location of the coordination files and backup rule (D11). Decided in chat DL 00.
- 2026-11-20: EDA scope, outputs and data codings (D12). Decided by OF and MM in chat DL 02.
- 2026-11-20: O3 closed (D12), after running the EDA part 1 on Oriana's laptop. Decided by OF and MM in chat DL 02.
- 2026-11-20: professor's message of 20/11 added to D0; D6 (RNN + attention model, self-attention study), D7 (pretext task, zero-shot), D9 (3-page summary, three areas, draft by 1 December) amended; D13 (ECG-GPT) added; O1 closed; O5 and O6 opened. Decided in chat DL 00.
- 2026-11-20: confidence intervals of the ablations on validation (D4); evaluation module, task-agnostic training loop, run logs and `make_numbers.py` (D14). Decided by OF and MM in chat DL 03.
- 2026-11-20: Dense block: architecture, ablation grid, three seeds for every ablation, selection of the final model, outputs (D15). Accepted by OF and MM in chat DL 04.
- 2026-11-20: `git_dirty` restricted to the code under `src/`; log schema 2 (D14). Decided by OF and MM in chat DL 03, after AI Error Bounty B5.
- 2026-11-20: CNN block: architecture, ablation grid with references, three seeds, selection of the final model, comparison with the final Dense model, interpretation figures, outputs (D16). Accepted by OF and MM in chat DL 05.
- 2026-11-21: RNN block: input as 100 frames, LSTM/GRU, readouts (last state, mean, attention), RNN + attention as a second final model, gradient clipping, ablation grid with references, interpretation (input sensitivity over time, attention weights), outputs (D17). Accepted by OF and MM in chat DL 06.
- 2026-11-21: Self-attention and Transformer block: patch tokens, hand-written multi-head attention shared with chat DL 13 (optional causal mask), Pre-LN encoder, self-attention study step by step, ablation grid with references, one final model, comparison with the final CNN, RNN, RNN + attention and Dense, interpretation (patch order, input sensitivity, attention statistics and maps), outputs (D18). Accepted by OF and MM in chat DL 07.
- 2026-11-21: Generative block: lead II tokens on the microvolt grid (range 2.56 mV, V = 256 / 64 / 2560), bits per sample as the vocabulary-independent metric, random training windows of L = 250 (100, 500), ECG-GPT (causal Transformer) and LSTM of equal size, ablation grid, counting baselines, selection rule of the pre-trained model for chat DL 08, generation settings and statistics (heart rate from R peaks), outputs (D19). O6 closed. Accepted by OF and MM in chat DL 13.
- 2026-11-22: D19 note after the runs: 50-epoch budget reached, median epoch time, smaller generation batches for long contexts. Decided by OF and MM in chat DL 13.
