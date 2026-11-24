# PLAN_DL.md — Deep Learning project (PRDL-DL / MLLB-DL), MUTSC-UPM 2026-2027

Team: Oriana Ferraz De Jesus (OF), María Martínez García-Taheño (MM).
Dataset: PTB-XL 1.0.3, 5 diagnostic superclasses (see `DECISIONS.md`).
Working mode: both work together on Oriana's laptop. The assistant writes code, figures and report drafts and explains each technique; OF/MM execute, check and decide. No result counts until one of us has run it.

Last update: 2026-11-24.

## Roles

| | Oriana (OF) | María (MM) |
|---|---|---|
| Owns | EDA; Dense; CNN; generative autoregressive model (ECG-GPT); repository and decisions log | Data loading and split; evaluation protocol; RNN (LSTM/GRU) and RNN + attention; self-attention and Transformer |
| Reviews | MM's blocks | OF's blocks |
| Shared | Transfer learning and fine-tuning of the pre-trained ECG-GPT; zero-shot experiment; final comparison; cross-cutting ablation (12 leads vs 1 lead) | |
| Reports | Own PRDL-DL and MLLB-DL | Own PRDL-DL and MLLB-DL |

"Owns" means: proposes the ablations, runs the experiments, checks the code line by line, reads the related literature and writes that analysis in her report. "Reviews" means: re-reads code and results and signs off, or logs the problem found.

## Task log

Status: `todo` / `doing` / `done`. "Done by" and "Reviewed by" are filled in only when it has actually happened; the contribution table of the reports is generated from these columns.

| ID | Task | Owner | Reviewer | Status | Done by | Reviewed by | Date |
|---|---|---|---|---|---|---|---|
| T0 | Dataset shortlist (8 candidates) and choice | OF + MM | — | done | OF + MM | — | 2026-11-14 |
| T1 | `DECISIONS.md` and `PLAN_DL.md` first version | OF | MM | done | OF + MM (drafted with the assistant) | MM | 2026-11-20 |
| T2 | Decide the transfer-learning approach: own pre-trained model (not confirmed with the professor; raise it in the class discussion) | OF + MM | — | done | OF + MM | — | 2026-11-14 |
| T3 | Verify licence and recommended folds on PhysioNet | OF + MM | — | done | OF + MM | — | 2026-11-14 |
| T4 | Local project structure, `.gitignore`, environment file; GitHub repository later | OF | MM | done | OF (drafted with the assistant) | MM | 2026-11-20 |
| T5 | Download script, label construction, fold split, `Dataset`/`DataLoader` | MM | OF | done | MM (drafted with the assistant) | OF | 2026-11-20 |
| T6 | EDA: class balance, co-occurrence, demographics, signal quality, example traces | OF | MM | done | OF (drafted with the assistant) | MM | 2026-11-20 |
| T7 | Evaluation module: macro AUROC, bootstrap CI, JSON logging, `make_numbers.py` | MM | OF | done | MM (drafted with the assistant) | OF | 2026-11-20 |
| T8 | Shared training loop (zero_grad, forward, loss, backward, step; train/eval modes) | OF + MM | — | done | OF + MM (drafted with the assistant) | — | 2026-11-20 |
| T9 | Dense baseline + ablations | OF | MM | done | OF (drafted with the assistant) | MM | 2026-11-20 |
| T10 | CNN 1D + ablations | OF | MM | done | OF (drafted with the assistant) | MM | 2026-11-20 |
| T11 | LSTM/GRU (+bidirectional) and RNN + attention model + ablations | MM | OF | done | MM (drafted with the assistant) | OF | 2026-11-21 |
| T12 | Self-attention step by step (single layer, attention maps) and Transformer + ablations | MM | OF | done | MM (drafted with the assistant) | OF | 2026-11-21 |
| T13 | Transfer learning: ECG-GPT (T23) reused on the 5 superclasses, scratch vs frozen vs fine-tuning, 10% and 100% labels | OF + MM | — | done | OF + MM (drafted with the assistant) | — | 2026-11-23 |
| T14 | Cross-cutting ablation: 12 leads vs single lead | OF + MM | — | done | OF + MM (drafted with the assistant) | — | 2026-11-23 |
| T15 | Final comparison table and figures; comparison with published results | OF + MM | — | done | OF + MM (drafted with the assistant) | — | 2026-11-24 |
| T16 | Literature: dataset and CNN-related papers | OF | MM | done | OF (drafted with the assistant) | MM | 2026-11-24 |
| T17 | Literature: RNN/Transformer-related papers | MM | OF | done | MM (drafted with the assistant) | OF | 2026-11-24 |
| T18 | Reviewed first assignment (learning strategies) for the PRDL reports | each her own | cross | done | OF and MM, each her own (drafted with the assistant, without the first assignments: see change log) | cross (OF, MM) | 2026-11-24 |
| T19 | PRDL-DL report (theory + critical analysis), opening with a summary of at most 3 pages | each her own | cross | done | OF (drafted with the assistant); MM: her own report in chat DL 11 | MM (OF's report) | 2026-11-24 |
| T20 | MLLB-DL report (Section 0 AI environment, code, training, evaluation), opening with a summary of at most 3 pages | each her own | cross | done | OF (drafted with the assistant); MM: her own report in chat DL 11 | MM (OF's report) | 2026-11-24 |
| T21 | `AI_USAGE_LOG.md` and AI Error Bounty sheets | each her own entries | cross | done | OF (drafted with the assistant); MM: her own entries in chat DL 11 | MM (OF's entries) | 2026-11-24 |
| T22 | Adversarial review before delivery; preparation of the class discussion and oral defence | OF + MM | — | todo | | | |
| T23 | Generative autoregressive model (ECG-GPT): tokenisation of lead II, causal Transformer and LSTM, sampling with temperature, ablations (D13) | OF | MM | done | OF (drafted with the assistant) | MM | 2026-11-22 |
| T24 | Zero-shot learning: theory and a short experiment on the class CLIP notebook | OF + MM | — | done | OF + MM (drafted with the assistant) | — | 2026-11-23 |
| T25 | Initial draft of the DL reports submitted (due 1 December 2026) | each her own | — | todo | | | |

## Chats

Each task is worked in one chat of the Claude project; names, scope and start prompts are in `CHATS_DL.md` (internal working file, not in this repository).

| Chat | Tasks |
|---|---|
| DL 00 Coordination | T1, upkeep of this file and `DECISIONS.md` |
| DL 01 Data | T4, T5 |
| DL 02 EDA | T6 |
| DL 03 Evaluation and training | T7, T8 |
| DL 04 Dense | T9 |
| DL 05 CNN | T10 |
| DL 06 RNN | T11 |
| DL 07 Self-attention and Transformer | T12 |
| DL 08 Learning strategies and comparison | T13, T14, T15, T24 |
| DL 09 Literature | T16, T17, T18 |
| DL 10 Reports Oriana | T19, T20, T21 (OF) |
| DL 11 Reports María | T19, T20, T21 (MM) |
| DL 12 Review and defence | T22 |
| DL 13 Generative (ECG-GPT) | T23 |

## Order of work

No fixed calendar; the team moves on as blocks close. Dependencies:

1. Data (T4, T5) → EDA (T6) and evaluation/training loop (T7, T8).
2. Model blocks T9-T12 need T5, T7 and T8; they can run in parallel, one per author at a time.
3. T23 (ECG-GPT) needs T7 and T8; it reuses the causal attention of T12 if that is ready, but does not depend on it.
4. T13, T14, T15 need T23 and the model blocks; T24 can be done at any time.
5. T16-T18 (literature) run in parallel from the start.
6. T25 (draft, 1 December): drafts T19-T20 with the results executed by then, each of the three areas at least designed.
7. Afterwards: full ablations, final comparison, final reports (January), T22 before each delivery.

## Change log

- 2026-11-14: plan created; dataset decided (PTB-XL); individual reports per member confirmed.
- 2026-11-14: T2 and T3 closed; GitHub repository postponed, work starts in the local folder.
- 2026-11-14: project organised into 13 chats (`CHATS_DL.md`).
- 2026-11-20: T4 done (repository orianafe16/ptbxl-DL, environment checked); T5 started. Rule confirmed by OF and MM: each task is executed by its owner and reviewed by the other, all on Oriana's laptop.
- 2026-11-20: T5 done (8 tests pass on Oriana's laptop, commit 9993fd0); reviewed by OF.
- 2026-11-20: this file moved to `ptbxl-dl/docs/` (DECISIONS D11).
- 2026-11-20: T1 closed (reviewed by MM, confirmed by OF and MM in chat DL 00).
- 2026-11-20: T6 done (EDA script, notebook, figures and `results/logs/eda.json`, executed by OF on Oriana's laptop; O3 closed in DECISIONS D12); reviewed by MM.
- 2026-11-20: plan revised after the professor's message of 20/11: T11, T12, T13, T19, T20 reworded; T23 (ECG-GPT), T24 (zero-shot) and T25 (draft due 1 December) added; chat DL 13 added; no fixed calendar.
- 2026-11-20: T7 and T8 done (evaluation module, task-agnostic training loop, run logs, `make_numbers.py`; DECISIONS D14). 44 tests pass and the end-to-end run of the trivial model produced its JSON log and `numbers.tex` on Oriana's laptop; T7 reviewed by OF.
- 2026-11-20: T9 done (Dense model, 11 configurations x 3 seeds on validation, final model on test with 3 seeds, summary, tables and figures; DECISIONS D15). Executed by OF on Oriana's laptop; reviewed by MM.
- 2026-11-20: T7/T8 fix after AI Error Bounty B5 (found in chat DL 04): `git_dirty` now looks only at the code under `src/`, log schema 2 (DECISIONS D14). 77 tests pass on Oriana's laptop; a `--fast` run gives `git_dirty: true` before the commit and `false` after it (commit e3e49ff).
- 2026-11-20: T10 done (1D CNN, 10 configurations x 3 seeds on validation, final model on test with 3 seeds compared with the final Dense and trivial models, summary, tables and interpretation figures; DECISIONS D16). Executed by OF on Oriana's laptop (runs at commit 3f9439f, report at 7eaf6ec); reviewed by MM.
- 2026-11-21: T11 started in chat DL 06 (DECISIONS D17: LSTM/GRU over 100 frames, RNN + attention as a second final model, 12 configurations x 3 seeds). Done by MM, reviewed by OF; runs on Oriana's laptop.
- 2026-11-21: T11 done (LSTM/GRU and RNN + attention, 12 configurations x 3 seeds on validation, two final models on test with 3 seeds compared with the final CNN, Dense and trivial models, summary, tables and interpretation figures; DECISIONS D17). Executed by MM on Oriana's laptop (runs at commit cae71d5); reviewed by OF.
- 2026-11-22: T23 done (ECG-GPT and LSTM generators of lead II, 8 configurations x 3 seeds on validation, counting baselines, generation with greedy, temperature and top-k sampling, summary, tables and figures; pre-trained model for T13 selected: `vocab2560`; DECISIONS D19). Executed by OF on Oriana's laptop (runs at commit f010618, summary at 6326057); reviewed by MM. Owner of T23 confirmed (O6 closed).
- 2026-11-23: T13, T14 and T24 done in chat DL 08 (DECISIONS D20 and its note of 2026-11-23): 66 transfer runs, 15 one-lead runs and the zero-shot run, executed on Oriana's laptop (runs at f69c72f with the new code committed as 1b7b07f; zero-shot at 5f8d0ac). Done by OF and MM (shared tasks, each run by whoever it belonged to); no reviewer assigned in this plan. T15: final comparison table, per-class table, cross-block comparisons and figures done; the comparison with published results waits for literal quotes from chat DL 09 (O7), so T15 stays `doing`.
- 2026-11-24: T15, T16, T17 and T18 done in chat DL 09. O7 closed: `report/published.json` with the 8 models of the super-diag. column of Table I of Strodthoff et al. (journal version), each with file, page and quote; `python -m src.comparison report` and `python report/make_numbers.py` run by OF on Oriana's laptop (the final table now has the published rows). Literature: NotebookLM answers checked quote by quote against the PDFs (`docs/literature/quotes.md`), `report/references.bib`, and every `[verify]` of the project closed (`quotes.md`, section G). T18: the first assignments (class presentations) are not in the folder; by decision of OF and MM, T18 is the cited material for the learning-strategies annex asked for in the weekly summary of 9/10 (`docs/literature/learning_strategies.md`). T16 done by OF, reviewed by MM; T17 done by MM, reviewed by OF; T18 each her own, cross review.
- 2026-11-24: T19, T20, T21 and T25 in progress for OF (chat DL 10, not yet closed): full draft of both reports compiled locally (PRDL 39 pages, MLLB 28, 2-page summary each) and AI appendices with four Bounty sheets each; not yet compiled on Overleaf nor reviewed by MM, so not done. Shared LaTeX parts in `report/common/` for MM's reports.
- 2026-11-24: chat DL 08 reopened for a presentation fix found in chat DL 10: the legend of `comparison_overview` hid the zero-shot CLIP point; moved outside the axes (commit 032cea4); `comparison report` and `make_numbers.py` run again by OF (340 numbers, 4,484 macros, no number changed). No task status changes.
- 2026-11-24: chat DL 10 closed. T19, T20 and T21 done for OF: both reports and their AI appendices compile without errors (locally: PRDL 40 pages, MLLB 28, 2-page summary each, logo on the cover, no undefined reference or citation, BibTeX clean with the `references.bib` fixed by chat DL 11); reviewed by MM. Joint review of the four reports and the final version go to chat DL 12. T25 taken out of chat DL 10 by OF and MM's decision: back to `todo` (the submission of the 1 December draft is not tracked here).
