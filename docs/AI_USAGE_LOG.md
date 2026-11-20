# AI_USAGE_LOG.md — Deep Learning project (PRDL-DL / MLLB-DL)

Team: Oriana Ferraz De Jesus (OF), María Martínez García-Taheño (MM).
Source for the AI appendix of the four reports. Append only. Lives in `ptbxl-dl/docs/` since 2026-11-20 (DECISIONS D11). Each entry says who was working and who checked what.

## 1. Session log

| Date | Chat | Who | Task given to the assistant | What the assistant produced | What we checked, and how |
|---|---|---|---|---|---|
| 2026-11-14 | DL 00 Coordination | OF + MM | Propose 5-8 real public datasets against the course requirements, flag anything not verified on the dataset page, compare them and recommend two; no code | Web search; shortlist of 8 datasets with links, licence, size and papers; list of unverified items; recommendation of PTB-XL (plan B: Drugs.com Reviews) | Course requirements re-read in Presentation III and Notebooks_GuideLines once the folder was shared; PTB-XL licence and recommended folds checked on the PhysioNet page by OF and MM |
| 2026-11-14 | DL 00 Coordination | OF + MM | Write the decisions log and the task plan; organise the project into chats | `DECISIONS.md`, `PLAN_DL.md`, `CHATS_DL.md`, `HANDOFF.md`, this file | Read by OF and MM: [to confirm] |
| 2026-11-20 | DL 01 Data | OF + MM | Step-by-step setup of the GitHub repository and the Python environment (T4) | Project skeleton (`src/`, `.gitignore`, `README.md`, `requirements.txt`, `src/check_env.py`) and git/venv commands; fixes for the failed first commit (git identity not set) and push (`src refspec main`) | Executed by OF on her laptop: push to orianafe16/ptbxl-DL, `git show --stat` (no file under `data/`), `check_env` (RTX 3050 Ti, CUDA 12.6, GPU test OK); reviewed by MM |
| 2026-11-20 | DL 01 Data | OF + MM | Recommend the likelihood rule and the provisional handling of records without superclass; write config, integrity check, labels and cache (T5, part 1) | Recommendation checked against `example_physionet.py` and the PhysioNet class table; `src/config.py`, `src/data.py`, `src/prepare_data.py`; dated entries in `DECISIONS.md` | Executed by MM on Oriana's laptop: SHA-256 check OK, cache 21,799 x 12 x 1,000, class counts equal to the PhysioNet table, 411 records without superclass; reviewed by OF |
| 2026-11-20 | DL 01 Data | OF + MM | Choose the normalisation; write Dataset, DataLoaders and tests (T5, part 2); write the block note | Comparison of per-lead, global and per-record normalisation with a recommendation; `src/dataset.py`, `tests/test_data.py`, `pytest.ini`, README update, dated entry in `DECISIONS.md` (D3), `docs/blocks/01_data.md` | Executed by MM on Oriana's laptop: `python -m pytest -v tests`, 8 passed (commit 9993fd0); reviewed by OF |
| 2026-11-20 | DL 00 Coordination | OF + MM | Fix the three items left for chat 00 (paths, backup, project instructions); analyse the options and take the best decision | Decision D11 (coordination files into `docs/`, backup script for internal files), updated `CHATS_DL.md`, new project instructions, `backup_dl.bat` | [to confirm by OF and MM] |

## 2. AI Error Bounty candidates

One entry per assistant error that one of us detected and corrected.

Template:

```
### B<n> — short title
- Date, chat, detected by: 
- Prompt: 
- Wrong answer: 
- Evidence that it was wrong: 
- Correction: 
- Reproduction (how to check it again): 
```

### B1 — Likelihood rule for the superclass labels stated without checking
- Date, chat, detected by: 2026-11-20, DL 01 Data; detected by the assistant itself when OF and MM asked it to analyse the best option; logged at their request.
- Prompt: OF and MM asked which decisions were needed before writing the label code ("¿Qué es lo mejor?").
- Wrong answer: the assistant proposed to assign a superclass only when the SCP code has likelihood > 0, claiming that "this is what the reference benchmark does" (stated from memory, without a source).
- Evidence that it was wrong: `example_physionet.py`, shipped with PTB-XL 1.0.3, assigns the superclass for every diagnostic code regardless of its likelihood. With that rule the counts are NORM 9514, MI 5469, STTC 5235, CD 4898, HYP 2649, equal to the table of the PhysioNet page; with likelihood > 0, STTC drops to 5108 (127 records lost) and the class-count test would fail.
- Correction: likelihood ignored (DECISIONS D2, entry of 2026-11-20).
- Reproduction (how to check it again): `python -m src.prepare_data` prints the counts with the adopted rule (run by MM on 2026-11-20: equal to the PhysioNet table). For the wrong rule, change `for code in codes` in `src/data.py::superclass_labels` to skip codes whose value in `scp_codes` is 0 and compare STTC.

## 3. Notes on unverified assistant statements

Statements the assistant flagged itself as unverified, and their outcome.

| Date | Statement | Outcome |
|---|---|---|
| 2026-11-14 | Train folds 1-8, validation fold 9, test fold 10 (written from memory) | Confirmed on PhysioNet by OF and MM, 2026-11-14 |
| 2026-11-14 | PTB-XL licence CC BY 4.0 (seen on the AWS registry entry, not on the dataset page) | Confirmed on PhysioNet by OF and MM, 2026-11-14 |
