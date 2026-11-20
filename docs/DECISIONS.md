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

## Open items

| # | Item | Blocks |
|---|---|---|
| O5 | Weekly summary of 20/11 and the "simple autoregressive LLM-style model" notebook it mentions are not yet in `contexto_DL/` | Chat DL 13 (reference implementation) |
| O6 | Owner of D13 (generative block): Oriana proposed, to be confirmed | PLAN_DL T23 |

Closed: O2 (licence and folds verified on PhysioNet, 2026-11-14). O4 (repository under Oriana's account, 2026-11-20). O3 (records without superclass left out; D12, 2026-11-20). O1 (pretext task: autoregressive next-sample prediction, D7 and D13, 2026-11-20).

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
