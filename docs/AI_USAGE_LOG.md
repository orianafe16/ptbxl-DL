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
| 2026-11-20 | DL 02 EDA | OF + MM | Propose the EDA decisions (format, data scope, `eda.json` layout, O3 options, quality columns, codings); write the first part of the EDA (splits, records without superclass) | Decision proposals E1-E6, accepted as D12; codings checked in the dataset files (age 300 in `ptbxl_v102_changelog.txt`); preliminary look at the 411 records; `src/eda.py` parts 1-2 | Executed by OF on Oriana's laptop: numbers equal to the assistant's preliminary run; O3 closed from that output (D12); reviewed by MM |
| 2026-11-20 | DL 02 EDA | OF + MM | Complete the EDA: classes and co-occurrence, demographics and quality, per-lead statistics and example traces; notebook; block note | `src/eda.py` parts 3-5 (8 figures, `results/logs/eda.json`), palette checked with a colour-blindness validator, figures inspected by the assistant before delivery, `notebooks/02_eda.ipynb`, `docs/blocks/02_eda.md` | Executed by OF on Oriana's laptop after each part, printed numbers compared with the assistant's run; notebook Run All, figures opened; reviewed by MM |

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

### B2 — File reported as written when the old version had been copied
- Date, chat, detected by: 2026-11-20, DL 02 EDA; detected by the assistant itself when it re-read the file before the next change.
- Prompt: OF and MM accepted the EDA decisions ("sí a todo"); the assistant had to add them to `DECISIONS.md` as D12.
- Wrong answer: the assistant told OF and MM that D12 had been added to `ptbxl-dl/docs/DECISIONS.md`.
- Evidence that it was wrong: the file in the folder still had its old size (13,462 bytes) and no D12 section; the assistant had sent the file to the laptop immediately after editing it, and the copy sent was the pre-edit version. `src/eda.py`, sent in the same call, was correct.
- Correction: D12 written again together with the closing of O3, and the result checked in the folder by file size (16,251 bytes) before reporting it.
- Reproduction (how to check it again): the wrong copy was never committed, so git does not show it; the evidence is the two file sizes above. To check a write in future: compare the size or the content of the file in the folder with what the assistant says it wrote.

## 3. Notes on unverified assistant statements

Statements the assistant flagged itself as unverified, and their outcome.

| Date | Statement | Outcome |
|---|---|---|
| 2026-11-14 | Train folds 1-8, validation fold 9, test fold 10 (written from memory) | Confirmed on PhysioNet by OF and MM, 2026-11-14 |
| 2026-11-14 | PTB-XL licence CC BY 4.0 (seen on the AWS registry entry, not on the dataset page) | Confirmed on PhysioNet by OF and MM, 2026-11-14 |
| 2026-11-20 | Age 300 means older than 89 | Confirmed by the assistant in `ptbxl_v102_changelog.txt`, line 2 (D12) |
| 2026-11-20 | Signals are in mV | Checked by the assistant on one header (`records100/00000/00001_lr.hea`, gain 1000/mV); not checked for every record |
| 2026-11-20 | Meaning of `sex` 0/1 | Not in the dataset files: [verify], chat 09 |
| 2026-11-20 | Why folds 9-10 contain only human-validated reports | Observed in the data (`eda.json`); the reason is [verify], chat 09 |
| 2026-11-20 | NORM together with mild conduction findings (IRBBB, IVCD, 1AVB) is clinically usual | Not stated as a fact: [verify], chat 09 |
| 2026-11-20 | How the Strodthoff et al. benchmark treats records without superclass | [verify], chat 09 |
