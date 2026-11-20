# PLAN_DL.md — Deep Learning project (PRDL-DL / MLLB-DL), MUTSC-UPM 2026-2027

Team: Oriana Ferraz De Jesus (OF), María Martínez García-Taheño (MM).
Dataset: PTB-XL 1.0.3, 5 diagnostic superclasses (see `DECISIONS.md`).
Working mode: both work together on Oriana's laptop. The assistant writes code, figures and report drafts and explains each technique; OF/MM execute, check and decide. No result counts until one of us has run it.

Last update: 2026-11-20.

## Roles

| | Oriana (OF) | María (MM) |
|---|---|---|
| Owns | EDA; Dense; CNN; repository and decisions log | Data loading and split; evaluation protocol; RNN (LSTM/GRU, attention); Transformer |
| Reviews | MM's blocks | OF's blocks |
| Shared | Transfer learning; final comparison; cross-cutting ablation (12 leads vs 1 lead) | |
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
| T7 | Evaluation module: macro AUROC, bootstrap CI, JSON logging, `make_numbers.py` | MM | OF | todo | | | |
| T8 | Shared training loop (zero_grad, forward, loss, backward, step; train/eval modes) | OF + MM | — | todo | | | |
| T9 | Dense baseline + ablations | OF | MM | todo | | | |
| T10 | CNN 1D + ablations | OF | MM | todo | | | |
| T11 | LSTM/GRU (+bidirectional, +attention) + ablations | MM | OF | todo | | | |
| T12 | Transformer + ablations | MM | OF | todo | | | |
| T13 | Transfer learning experiment: own pre-training, then scratch vs frozen vs fine-tuning | OF + MM | — | todo | | | |
| T14 | Cross-cutting ablation: 12 leads vs single lead | OF + MM | — | todo | | | |
| T15 | Final comparison table and figures; comparison with published results | OF + MM | — | todo | | | |
| T16 | Literature: dataset and CNN-related papers | OF | MM | todo | | | |
| T17 | Literature: RNN/Transformer-related papers | MM | OF | todo | | | |
| T18 | Reviewed first assignment (learning strategies) for the PRDL reports | each her own | cross | todo | | | |
| T19 | PRDL-DL report (theory + critical analysis) | each her own | cross | todo | | | |
| T20 | MLLB-DL report (Section 0 AI environment, code, training, evaluation) | each her own | cross | todo | | | |
| T21 | `AI_USAGE_LOG.md` and AI Error Bounty sheets | each her own entries | cross | todo | | | |
| T22 | Adversarial review before delivery; preparation of the class discussion and oral defence | OF + MM | — | todo | | | |

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
| DL 07 Transformer | T12 |
| DL 08 Transfer learning and comparison | T13, T14, T15 |
| DL 09 Literature | T16, T17, T18 |
| DL 10 Reports Oriana | T19, T20, T21 (OF) |
| DL 11 Reports María | T19, T20, T21 (MM) |
| DL 12 Review and defence | T22 |

## Order of work

1. T4 (local structure; T2 and T3 are closed).
2. T5, T6, T7, T8 (data, EDA, evaluation, training loop).
3. T9 -> T10 -> T11 -> T12, each with its ablations before moving on.
4. T13, T14, T15.
5. T16-T21 run in parallel from step 2; drafts ready for the class discussions in the last two course weeks of December; final reports in January.

## Change log

- 2026-11-14: plan created; dataset decided (PTB-XL); individual reports per member confirmed.
- 2026-11-14: T2 and T3 closed; GitHub repository postponed, work starts in the local folder.
- 2026-11-14: project organised into 13 chats (`CHATS_DL.md`).
- 2026-11-20: T4 done (repository orianafe16/ptbxl-DL, environment checked); T5 started. Rule confirmed by OF and MM: each task is executed by its owner and reviewed by the other, all on Oriana's laptop.
- 2026-11-20: T5 done (8 tests pass on Oriana's laptop, commit 9993fd0); reviewed by OF.
- 2026-11-20: this file moved to `ptbxl-dl/docs/` (DECISIONS D11).
- 2026-11-20: T1 closed (reviewed by MM, confirmed by OF and MM in chat DL 00).
- 2026-11-20: T6 done (EDA script, notebook, figures and `results/logs/eda.json`, executed by OF on Oriana's laptop; O3 closed in DECISIONS D12); reviewed by MM.
