# Block 08 — Learning strategies and final comparison (T13, T14, T15, T24)

Chat DL 08, 2026-11-22/23. Shared tasks of OF and MM (who did and who reviewed each one: see `docs/PLAN_DL.md`). Everything was executed on Oriana's laptop. Decisions: `docs/DECISIONS.md` D20 (this block, with its dated note of 2026-11-23), D7 and D13 (transfer design), D19 (pre-trained model), D15-D18 (final models of each family), D3, D4 and D14 (protocol).

The numbers below are copied from the summary logs (3 decimals) only to make this note readable. **The reports must use the macros** (`\transfer...{}`, `\leads...{}`, `\zeroshot...{}`, `\comparison...{}`), never these figures. Mean ± SD over seeds 42-44 unless stated.

## What the block delivers

| Item | File |
|---|---|
| Data of the labelled records (lead II tokens, 10 % label subset drawn by patient), `TransferClassifier`, `FeatureHead`, feature extraction | `src/transfer.py` |
| T13 runs, estimate, summary, tables, figures | `src/transfer_experiments.py` |
| T14 runs (final model of each family on lead II only), summary, table, figure | `src/leads_experiments.py` |
| T24 zero-shot CLIP (CIFAR-100 and ECG images), log, table, figures | `src/zero_shot.py`, `notebooks/08_zero_shot_clip.ipynb` |
| T15 final comparison (no training) | `src/comparison.py` |
| Tests (label subset, data order, frozen backbone, probe = frozen model, grid, renderer, zero-shot scores, whole pipeline on synthetic data with fake generators and a fake CLIP) | `tests/test_strategies.py` (23 tests) |
| Runs | 66 `transfer_*` (18 main with test, 12 learning-rate ablations, 36 probe-study heads), 15 `onelead_*`, `zeroshot` |
| Summaries (aliases in `report/runs.json`) | `transfer_summary`, `leads_summary`, `zeroshot`, `comparison_summary` |
| Tables (`report/tables/`) | `transfer_main`, `transfer_comparisons`, `transfer_probe_study`, `leads`, `zeroshot`, `comparison_final`, `comparison_per_class`, `comparison_cross` |
| Figures (`report/figures/`, PDF and PNG) | `transfer_curves`, `transfer_strategies`, `transfer_probe_study`, `leads_comparison`, `zeroshot_examples`, `zeroshot_ptbxl`, `comparison_overview`, `comparison_label_efficiency` |

Commits: T13 and T14 runs at `f69c72f` with the new files of D20 not yet committed (`git_dirty: true`; the files were checked unchanged and committed as `1b7b07f`, D20 note of 2026-11-23). Zero-shot at `5f8d0ac` (`git_dirty: false`), after the fix of AI Error Bounty B10.

Run time *(log)*: fine-tuning and scratch with 100 % of the labels about 46 s per epoch (best epoch about 6, runs of 11 epochs); with 10 % about 6 s per epoch; probe heads under 1 s per epoch. Zero-shot about 3 min, including the CIFAR-100 download.

## T13. Transfer from the self-supervised ECG-GPT (area 2)

Lead II, test fold, macro AUROC (`transfer_main.tex`):

| Labels | From scratch | Linear probe (frozen) | Fine-tuning |
|---|---|---|---|
| 100 % (17,084 records, 14,823 patients) | 0.757 ± 0.004 | 0.722 ± 0.003 | **0.826 ± 0.001** |
| 10 % (1,711 records, 1,503 patients) | 0.669 ± 0.002 | 0.711 ± 0.003 | **0.801 ± 0.006** |

Trainable parameters: 724,101 for scratch and fine-tuning, 645 for the probe (frozen backbone of 723,456).

Paired differences on test (`transfer_comparisons.tex`; seeds whose 95 % interval excludes 0):

| Comparison | Δ | Seeds |
|---|---|---|
| Fine-tuning − scratch, 100 % | +0.069 | 3/3 better |
| Fine-tuning − scratch, 10 % | +0.131 | 3/3 better |
| Fine-tuning 10 % − scratch 100 % | +0.043 | 3/3 better |
| Fine-tuning − probe, 100 % / 10 % | +0.104 / +0.090 | 3/3 better |
| Probe − scratch, 100 % | −0.035 | 3/3 worse |
| Probe − scratch, 10 % | +0.042 | 3/3 better |
| 100 % − 10 %: scratch / probe / fine-tuning | +0.088 / +0.011 / +0.026 | 3/3 better |

Ablations on validation (2 × 2 initialisation × learning rate):

| Comparison (validation) | Δ | Seeds |
|---|---|---|
| Fine-tuning at 1e-3 − at 1e-4, 100 % / 10 % | −0.006 / −0.002 | 2/3 worse / no seed differs |
| Scratch at 1e-4 − at 1e-3, 100 % / 10 % | −0.009 / −0.013 | 1/3 worse / 2/3 worse |
| Fine-tuning − scratch, both at 1e-4, 100 % / 10 % | +0.081 / +0.143 | 3/3 better |

Probe study (validation macro AUROC of the same linear probe on each generator of D19; `transfer_probe_study.tex`, `transfer_probe_study` figure):

| Generator | Bits per sample (D19) | Probe 100 % | Probe 10 % |
|---|---|---|---|
| vocab2560 (selected) | 5.738 | 0.745 | 0.733 |
| ctx500 | 5.923 | 0.775 | 0.759 |
| lstm | 5.932 | **0.800** | 0.777 |
| base | 6.022 | 0.775 | 0.763 |
| ctx100 | 6.179 | 0.757 | 0.745 |
| no_pos | 6.407 | 0.790 | **0.783** |
| vocab64 | 6.619 | 0.792 | 0.782 |
| Spearman (bits, AUROC) | | 0.500 | 0.714 |

Every other generator gives a better probe than the selected one (3/3 seeds, both fractions).

### What T13 shows

1. **Self-supervised pre-training pays off when the whole network is adapted.** Fine-tuning beats training the same network from scratch by about 0.07 with every label and by about 0.13 with 10 % of them, in every seed. It still wins when scratch uses the same small learning rate (+0.081 / +0.143 on validation), so the gain comes from the pre-trained weights, not from the rate.
2. **Label efficiency.** Fine-tuning with 10 % of the labels beats training from scratch with 100 % (+0.043, 3/3). Fine-tuning loses only 0.026 when going from 100 % to 10 % of the labels; scratch loses 0.088.
3. **The frozen features are weakly diagnostic.** The linear probe is worse than scratch with all the labels (−0.035) and better only with 10 % (+0.042). The representation learnt to predict the next sample is useful as a starting point, but it is not linearly separable by superclass. *Hypothesis:* the next-sample objective rewards local waveform detail more than the global features that separate the classes.
4. **The smaller learning rate for fine-tuning was not essential here.** At 1e-3, fine-tuning reaches the same validation AUROC within 0.006 (best epoch about 2 instead of about 6), worse in 2 of 3 seeds at 100 % and in no seed at 10 %. The choice of 1e-4 (D20) did no harm but made little difference.
5. **A better generator did not give better features (probe study).** The correlation between bits per sample and probe AUROC is positive (0.50 and 0.71 over 7 generators). The generator selected for being the best at generation (`vocab2560`, 1 µV levels) gives the worst probe; coarser tokens (`vocab64`), no positional encoding (`no_pos`) and the LSTM give the best. *Hypotheses, not tested:* (a) with 2560 levels the embedding must learn fine amplitude distinctions and spends its capacity on noise-level detail; (b) the LSTM's state summarises the window more than a mean of causal Transformer outputs. Selecting the pre-trained model by generation quality (D19) was a reasonable rule fixed in advance, but in hindsight it was not the best rule for transfer. This is a finding worth stating, not a reason to change D19 after the fact: the main grid was run with the model fixed in advance.
6. **Training details to mention.** Scratch and fine-tuning reach their best validation epoch quickly (about 6 with 100 %) and then the validation AUROC falls while the training loss keeps falling (overfitting, `transfer_curves` panel c). The probe with 10 % of the labels had its best epoch at about 47 of the 50 allowed: it was limited by the epoch budget (its validation AUROC was still rising very slowly; D20 note).

## T14. 12 leads vs lead II (cross-cutting ablation)

Test macro AUROC (`leads.tex`, `leads_comparison`):

| Family (final config.) | 12 leads | Lead II | Δ | Seeds |
|---|---|---|---|---|
| Dense (`no_norm`) | 0.744 ± 0.002 | 0.670 ± 0.004 | −0.074 | 3/3 worse |
| CNN (`no_dropout`) | 0.928 ± 0.002 | **0.857 ± 0.002** | −0.071 | 3/3 worse |
| RNN (`conv`) | 0.918 ± 0.002 | 0.846 ± 0.002 | −0.072 | 3/3 worse |
| RNN + attention (`raw_attn`) | 0.917 ± 0.001 | 0.852 ± 0.004 | −0.065 | 3/3 worse |
| Transformer (`no_pos`) | 0.903 ± 0.004 | 0.819 ± 0.002 | −0.084 | 3/3 worse |

Every family loses 0.065-0.084 with a single lead, and the CNN stays first and the Transformer last of the four non-dense families (RNN and RNN + attention swap places, within 0.006). The information of the other 11 leads is not redundant with lead II for these five superclasses. *Hypothesis:* MI and HYP are diagnosed from leads that "look at" specific regions of the heart (e.g. precordial leads), which lead II alone does not see; per-class numbers are in `comparison_per_class.tex`.

## T24. Zero-shot CLIP

`zeroshot.tex`, `zeroshot_examples`, `zeroshot_ptbxl`; CLIP ViT-B/32 (`ViT-B-32-quickgelu`, OpenAI weights) through open_clip 4.2.2.

| Measure | Value [95 % CI] |
|---|---|
| CIFAR-100, top-1, notebook prompt "This is a photo of a {label}" | 0.582 [0.572, 0.592] (chance 0.010) |
| CIFAR-100, top-1, bare label | 0.552 [0.543, 0.562] |
| CIFAR-100, top-5, notebook prompt / bare label | 0.850 / 0.814 |
| PTB-XL test images whose closest sentence is "an electrocardiogram" | 1.000 |
| PTB-XL test macro AUROC | 0.527 [0.516, 0.539] |
| Per class (test): NORM / MI / STTC / CD / HYP | 0.704 / 0.334 / 0.725 / 0.419 / 0.452 |
| Share of test records where each sentence wins: STTC / CD / others | 0.976 / 0.024 / 0.000 |
| Zero-shot − trivial model (paired, test) | −0.191, worse |

What it shows:
1. **Zero-shot works where the shared image-text space covers the task.** On CIFAR-100, without any training, CLIP picks the right class among 100 for 58 % of the images, and the notebook's sentence helps (+0.03 over the bare label).
2. **CLIP recognises the modality but not the diagnosis.** Every ECG image is closest to "an electrocardiogram" among the six modality sentences, but the macro AUROC on the five superclasses is barely above chance and far below the trivial model. For 98 % of the records the "ST-T changes" sentence wins whatever the record. Some classes are ranked *against* the truth (MI 0.33).
3. Why: zero-shot needs the class names and the inputs to meet in a space learnt from paired data (weekly summary 9/10, multimodal correspondence). Our ECG-GPT has no text side, so it cannot do zero-shot; CLIP has one, but its training pairs (web images and captions) apparently do not tie ECG traces to diagnoses (**[verify]**: what CLIP's training data contain is not in the course material). *Hypothesis:* the AUROC above 0.5 for NORM and STTC and below 0.5 for MI comes from simple visual properties of the drawn trace (amplitude, heart rate, flatness) that correlate with the classes, not from understanding the diagnosis; it was not tested.

## T15. Final comparison

`comparison_final.tex` (all models, test), `comparison_per_class.tex`, `comparison_cross.tex`; figures `comparison_overview` and `comparison_label_efficiency`. 18 rows: trivial; 5 families with 12 leads; the same 5 on lead II; 6 ECG-GPT configurations; zero-shot CLIP.

Cross-block paired differences on test (lead II models − the CNN trained on lead II with 100 % of the labels):

| Comparison | Δ | Seeds |
|---|---|---|
| ECG-GPT fine-tuning, 100 % − CNN lead II | −0.031 | 3/3 worse |
| ECG-GPT fine-tuning, 10 % − CNN lead II | −0.056 | 3/3 worse |
| ECG-GPT scratch / probe, 100 % − CNN lead II | −0.100 / −0.135 | 3/3 worse |

Reading: on the same input (lead II), the CNN trained from scratch with every label is still better than the fine-tuned ECG-GPT; pre-training closes most of the gap between the ECG-GPT from scratch (0.757) and the CNN (0.857), but not all of it. The architecture (a convolutional inductive bias) matters more than self-supervised pre-training on this dataset size. The ECG-GPT was pre-trained only on the training folds of PTB-XL itself (about 17,000 records), a small "foundation" by any standard: this is the main limit of the area-2 experiment.

No published results are in the table yet (O7): `report/published.json` must be filled by chat DL 09 with literal quotes.

## Hypotheses and limitations to state in the reports

- T13 point 3 (why the frozen features are weak), point 5 (why fewer bits do not give better features) and T14 (which leads carry MI and HYP) are hypotheses.
- The 10 % probe was limited by the 50-epoch budget.
- Early stopping and thresholds of the 10 % runs used the whole validation fold (as many labelled records as the 10 % training subset).
- The pre-training used the signals of every training record: 10 % of the labels, not of the data.
- CLIP's failure explanation is a hypothesis; what its training data contain is **[verify]**.
- The selection rule of D19 (generation quality) was not the best for transfer: say so as a finding.

## For the report chats (10 and 11)

- Area 2 of the professor's message: `transfer_main.tex`, `transfer_comparisons.tex` (wide: `\resizebox{\textwidth}{!}{...}`), `transfer_strategies` or `comparison_label_efficiency`, `transfer_probe_study`; zero-shot: `zeroshot.tex`, `zeroshot_examples`.
- Cross-cutting ablation: `leads.tex`, `leads_comparison`.
- Final comparison: `comparison_final.tex` (wide), `comparison_overview`.
- Key macros: `\transferTestAurocMeanFinetuneFOneZeroZero{}`, `\transferTestDiffMeanFinetuneFOneZeroVsScratchFOneZeroZero{}`, `\transferProbeSpearmanFOneZeroZero{}`, `\leadsTestDiffMeanCnn{}`, `\zeroshotTestMacroAuroc{}`, `\zeroshotCifarTopOneNotebook{}`, `\comparisonTestDiffMeanFinetuneFOneZeroZeroVsOneleadCnn{}`.
- Theory to connect (PRDL): self-supervised pre-training (pretext task: next sample) vs transfer learning (weekly summary 9/10); frozen feature extractor vs fine-tuning (class notebook `pytorch_transfer_learning_colab.ipynb`); zero-shot and multimodal correspondence (CLIP, weekly summary 9/10).
