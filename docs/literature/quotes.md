# Literal quotes (chat DL 09)

Flow of DECISIONS D8: Perplexity -> PDFs downloaded by OF and MM into `contexto_DL/literature/` (not in this repository) -> NotebookLM prompts N1-N4 of `prompts.md` -> every quote checked by the assistant against the text of the PDF (pdftotext, whitespace and ligatures normalised) on 2026-11-24. Reports are written only from the rows marked **checked** or **partial** (and only for the part stated). Anything else stays `[verify]`.

Page: printed page number, then the page of the PDF file when it differs. "pp. a-b": the sentence runs across a page break. Formulas are copied as printed (`x = Wu + b`).

Status: **checked** = verbatim in the PDF on that page and supports the claim as written; **partial** = verbatim, but supports only the part stated in "Use"; **[verify]** = no checked quote (PDF missing or claim not found).

## A. PTB-XL and the benchmark

| ID | Claim | Key | File | Page | Quote | Status / use |
|---|---|---|---|---|---|---|
| A1 | Published macro AUROC, superclass task (O7) | strodthoff2021deep | `Deep_Learning_for_ECG_Analysis_Benchmarks_and_Insights_from_PTB-XL.pdf` | 1523 (PDF 5), Table I | Table caption: "Overall discriminative performance of ECG classification algorithms on PTB-XL in terms of macro AUC." Values: column "super-diag." of Table I, transcribed in `report/published.json` | **checked** against the rendered table (the table is an image in this PDF). Journal version, not arXiv v1: the parenthesised intervals differ between versions and v1 has an extra `naive` row |
| A2 | Notation of the intervals | strodthoff2021deep | same | 1522 (PDF 4) | "we report the point estimate from evaluating on the whole test set and estimate lower and upper confidence intervals using the bootstrap examples." / "0.743(09) is supposed to be understood as 0.743 ± 0.009." | **checked**. Bootstrap with 10,000 iterations in the journal version (1,000 in arXiv v1) |
| A3 | Metric: macro AUC | strodthoff2021deep | same | 1522 (PDF 4) | "we report the macro-averaged area under the receiver operating characteristic curve (henceforth referred to as AUC), which is obtained by averaging class-wise AUCs over all classes." | **checked** |
| A4 | Splits and sampling rate | strodthoff2021deep | same | 1520 (PDF 2) | "we use the recommended train-test splits provided by PTB-XL [15], which consider patient assignments and use input data at a sampling frequency of 100 Hz." | **checked** |
| A5 | Folds 1-8 / 9 / 10 | strodthoff2021deep | same | 1521 (PDF 3) | "we train a classifier from scratch by training on the first eight folds using the ninth and tenth fold as validation and test sets, respectively." | **checked** |
| A6 | Records without a superclass are left out (closes the 2026-11-20 `[verify]`; same rule as our D12) | strodthoff2021deep | same | 1523 (PDF 5) | "For each experiment, we select only samples with at least one label in the given label selection." | **checked** |
| A7 | Their training differs from ours: windows and aggregation | strodthoff2021deep | same | 1521 (PDF 3) | "If not mentioned otherwise, we use a fixed window size of 2.5 seconds." / "These predictions are then aggregated using the element-wise maximum" | **checked**. Setting difference to describe in the text |
| A8 | Their optimiser and schedule | strodthoff2021deep | same | 1521 (PDF 3) | "We use 1-cycle learning rate scheduling during training [36] and the AdamW optimizer [37]." | **checked** |
| A9 | Only in the superclass task the ensemble is significantly better | strodthoff2021deep | same | 1523 (PDF 5) | "only in the super-diagnostic case the ensemble outperforms the best-performing single model in a statistically significant manner." | **checked** |
| A10 | Pre-training helps most with few labels (ECG, PTB-XL -> ICBEB2018) | strodthoff2021deep | same | 1524 (PDF 6) | "the performance gap between both approaches widens for smaller training sets and becomes significant (even at a confidence level below 0.0005) for a single training fold or fractions of it." | **checked** |
| A11 | Fine-tuning with gradual unfreezing and discriminative learning rates against catastrophic forgetting | strodthoff2021deep | same | 1521 (PDF 3) | "we use gradual unfreezing and discriminative learning rates [32], [35] to avoid catastrophic forgetting i.e. overwriting information captured during the initial training phase on PTB-XL." | **checked** |
| A12 | Dataset size | wagner2020ptbxl | `s41597-020-0495-6.pdf` | 1 | "The dataset comprises 21837 clinical 12-lead ECG records of 10 seconds length from 18885 patients." | **checked** |
| A13 | Sex coding (closes the 2026-11-20 `[verify]`) | wagner2020ptbxl | same | 6, Table 2 | "sex (male 0, female 1)" | **checked** |
| A14 | Why folds 9 and 10 contain only human-validated reports (closes the 2026-11-20 `[verify]`) | wagner2020ptbxl | same | 12 | "When assigning ECGs from a patient that does not carry this flag, we exclude the ninth and tenth fold from the set of folds the samples can be assigned to." (the flag: "A patient is considered clean if for all corresponding ECGs validated_by_human is set to true.") | **checked** |
| A15 | Folds 9 and 10 intended as validation and test | wagner2020ptbxl | same | 4, Fig. 6 caption | "The ninth and tenth fold are folds with a particularly high label quality that are supposed to be used as validation and test sets." | **checked** |
| A16 | Licence of the dataset | — | — | — | NotebookLM quoted the licence of the *article* (p. 15), not of the dataset | Not usable. Dataset licence CC BY 4.0 already confirmed on PhysioNet by OF and MM (2026-11-14); cite the PhysioNet page |

## B. Architectures and training

| ID | Claim | Key | File | Page | Quote | Status / use |
|---|---|---|---|---|---|---|
| B1 | Adam is less sensitive to the learning rate than SGD (D14) | kingma2015adam | `1412.6980v9.pdf` | 1 | "The hyper-parameters have intuitive interpretations and typically require little tuning." | **partial**: supports "Adam's hyper-parameters typically require little tuning", not a comparison with SGD. Rewrite the sentence or keep the comparison `[verify]` |
| B2 | Dropout 0.5 for hidden layers (D15) | srivastava2014dropout | `srivastava14a.pdf` | 1930 (PDF 2) | "p can be chosen using a validation set or can simply be set at 0.5, which seems to be close to optimal for a wide range of networks and tasks." | **checked**. p is the *retention* probability; for inputs the paper says it is closer to 1 |
| B3 | Batch norm before the activation (D15) | ioffe2015batch | `1502.03167v3.pdf` | 5 | "We add the BN transform immediately before the nonlinearity, by normalizing x = Wu + b." | **checked** |
| B4 | Degradation of plain deep networks (D16) | he2016deep | `1512.03385v1.pdf` | 1; 4 | "with the network depth increasing, accuracy gets saturated (which might be unsurprising) and then degrades rapidly." / "the deeper 34-layer plain net has higher validation error than the shallower 18-layer plain net." | **checked**. Write: degradation is shown from 18 to 34 layers; our CNNs have at most 12 convolutional layers |
| B5 | Class activation maps (D16) | zhou2016learning | `1512.04150v1.pdf` | pp. 2-3 | "Here we ignore the bias term: we explicitly set the input bias of the softmax to 0 as it has little to no impact on the classification performance." | **partial**: CAM is from Zhou et al.; they drop the bias and use a sum. Our identity (logit = bias + mean of the CAM) is our variant with the bias kept and global average pooling; say so |
| B6 | Gradient clipping for RNNs (D17) | pascanu2013difficulty | `1211.5063v2.pdf` | 6 | "Clipping has been shown to do well in practice and it forms the backbone of our approach." | **checked**. "First published in Mikolov 2010": no PDF, stays `[verify]` (from the course slides only) |
| B7 | LSTM variants perform about the same (D17) | greff2017lstm | `1503.04069v2.pdf` | 1 | "Our results show that none of the variants can improve upon the standard LSTM architecture significantly," | **checked** |
| B8 | Pre-LN optimises more easily than Post-LN (D18) | — | — | — | NotebookLM: NOT FOUND (Xiong et al. 2020 not in the folder) | `[verify]`; course slides (Transformer Part II, 35-36) remain the only source |
| B9 | QRS lasts about 80-100 ms (D16) | — | — | — | NotebookLM returned a frequency band, not a duration | **closed by rewording**: state only "kernel 7 = 70 ms at 100 Hz"; no clinical duration in the reports |
| B10 | Most ECG energy at low frequencies (block note 05) | adam2025comparative | `s41598-025-26476-1.pdf` | 12 | "while preserving the morphology of key ECG features: the P wave (0.67–5 Hz), T wave (1–7 Hz), and QRS complex (10–50 Hz)." | **partial**: the main waves lie below 50 Hz; "most energy" is not stated |
| B11 | MI visible in the chest leads (block note 08, T14) | wagner2009aha | `wagner-et-al-2009-aha-...pdf` | e265 (PDF 4) | "This will be expressed as ST elevation in some or all of leads V1 through V6." (anterior wall ischemia/infarction) | **partial**: supports that anterior MI shows in V1-V6, not lead II. HYP: no source, `[verify]` |
| B12 | Limb leads are linear combinations of each other (block note 04) | wagner2009aha | `wagner-et-al-2009-aha-...pdf` | e263 (PDF 2) | "The negative electrode for the 3 aV (or augmented) limb leads (aVR, aVL, and aVF) is provided by the averaged inputs from the 2 limb electrodes that do not serve as the positive electrode for a specified lead" | **partial**: the augmented leads are built from the same limb electrodes, so they are combinations of the same potentials. The reading of the first-layer bands stays a hypothesis |
| B13 | Transformer: Post-LN | vaswani2017attention | `Vaswani-2017.pdf` | 3 | "That is, the output of each sub-layer is LayerNorm(x + Sublayer(x)), where Sublayer(x) is the function implemented by the sub-layer itself." | **checked** |
| B14 | Transformer: scaled dot-product attention | vaswani2017attention | same | 4 | "We call our particular attention "Scaled Dot-Product Attention"" | **checked** |
| B15 | Transformer: multi-head attention | vaswani2017attention | same | 5 | "Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions." | **checked** |
| B16 | Transformer: sinusoidal positional encoding | vaswani2017attention | same | 6 | "In this work, we use sine and cosine functions of different frequencies:" | **checked** |
| B17 | Transformer: dropout 0.1 on each sub-layer output (D18) | vaswani2017attention | same | 7 | "We apply dropout [33] to the output of each sub-layer, before it is added to the sub-layer input and normalized." / "For the base model, we use a rate of Pdrop = 0.1." | **checked** |
| B18 | Additive attention over RNN states (D17) | bahdanau2015neural | `1409.0473v7.pdf` | 3 | "is an alignment model which scores how well the inputs around position j and the output at position i match." | **checked** |
| B19 | Residual building block; 1×1 projection shortcut when dimensions change (D16) | he2016deep | `1512.03385v1.pdf` | 2; 4 | "Residual learning: a building block." / "(B) The projection shortcut in Eqn.(2) is used to match dimensions (done by 1×1 convolutions)." | **checked**. Our block with batch norm and "same" padding is our 1D adaptation: say so |
| B20 | GRU | cho2014learning | `D14-1179.pdf` | 1725 (PDF 3) | "motivated by the LSTM unit but is much simpler to compute and implement." | **checked** |

## C. Learning strategies

| ID | Claim | Key | File | Page | Quote | Status / use |
|---|---|---|---|---|---|---|
| C1 | What CLIP's training pairs contain | radford2021learning | `2103.00020v1.pdf` | 3 | "we constructed a new dataset of 400 million (image, text) pairs collected form a variety of publicly available sources on the Internet." | **checked** (typo "form" is in the paper). Whether ECG traces with diagnoses are among them is **not** stated: the explanation of the CLIP failure stays a hypothesis |
| C2 | CLIP zero-shot rule | radford2021learning | same | 6 | "The cosine similarity of these embeddings is then calculated, scaled by a temperature parameter τ , and normalized into a probability distribution via a softmax." | **checked** |
| C3 | CLIP prompt template | radford2021learning | same | 7 | "we found that using the prompt template "A photo of a {label}." to be a good default" | **checked** |
| C4 | Linear probe vs fine-tuning | radford2021learning | same | 11 | "Fitting a linear classifier on a representation extracted from the model and measuring its performance on various datasets is a common approach. An alternative is measuring the performance of end-to-end fine-tuning of the model." | **checked** |
| C5 | Smaller learning rate for fine-tuning (D20) | soltanieh2023indistribution | `2304.06427v2.pdf` | 6 | "For fine-tuning, we reduce the learning rate to 5 × 10−3 to prevent catastrophic forgetting and minimize the chance of overwriting the network weights completely." | **checked** (ECG, self-supervised). See also A11 |
| C6 | Self-supervised pre-training on 12-lead ECG helps with fewer labels | mehari2022self | `2103.12676v2.pdf` | pp. 7-8 | "The performance level of the supervised models at 4 folds is approximately reached by the pretrained model trained only on two folds i.e. 50% of the training data." | **checked** |
| C7 | Definition: transfer learning | monka2022survey | `2201.11794v1.pdf` | 2 | "Transfer learning is the area of machine learning that groups approaches dealing with such an unpredictable distribution shift" | **checked** (survey on visual transfer learning) |
| C8 | Definition: zero-shot learning | monka2022survey | same | 4 | "Zero-Shot Learning is a visual transfer learning task with labeled source domain data and unlabeled target domain data." | **partial**: a visual-domain definition; the CLIP and GPT-3 senses are C2 and C9 |
| C9 | Definition: self-supervised learning (ECG) | soltanieh2022analysis | `2206.07656v1.pdf` | 1 | "Self-supervised learning is a feature learning technique for overcoming the challenges related to the scarcity of 'labelled' data" | **checked** |
| C10 | GPT-3 zero-shot and few-shot | brown2020language | — | — | NotebookLM quotes from arXiv 2005.14165, p. 3 | `[verify]`: the PDF is not in `contexto_DL/literature/`. The paper is Brown et al. 2020 (Mann is a co-author), not "Mann et al." |

## D. Generative autoregressive models

| ID | Claim | Key | File | Page | Quote | Status / use |
|---|---|---|---|---|---|---|
| D1 | Decoder-only GPT with masked self-attention | radford2018improving | `radford2018improving.pdf` | 5 | "We trained a 12-layer decoder-only transformer with masked self-attention heads (768 dimensional states and 12 attention heads)." | **checked** |
| D2 | Next-token objective | radford2018improving | same | 3 | "we use a standard language modeling objective to maximize the following likelihood:" (Eq. 1, L1(U) = Σ_i log P(u_i \| u_{i−k}, ..., u_{i−1}; Θ)) | **checked** |
| D3 | Top-k sampling (original source) | fan2018hierarchical | `P18-1082.pdf` | 893 (PDF 5) | "We randomly sample from the k = 10 most likely candidates from this distribution." | **checked**. NotebookLM had quoted Holtzman's description of top-k; Fan is the primary source |
| D4 | Nucleus (top-p) sampling | holtzman2020curious | `1904.09751v2.pdf` | 5 | "In practice this means selecting the highest probability tokens whose cumulative probability mass exceeds the pre-chosen threshold p." | **checked** |
| D5 | mu-law quantisation (WaveNet) | oord2016wavenet | — | — | NotebookLM quotes from arXiv 1609.03499, p. 3 | `[verify]`: PDF not in the folder |
| D6 | Causal Transformers without positional encoding still learn positions | haviv2022transformer | `2203.16634v2.pdf` | 1 | "we show that LMs without any explicit positional encoding are still competitive with standard models, and that this phenomenon is robust across different datasets, model sizes, and sequence lengths." | **checked**. Our D19 result (positions help when generating) is a different setting: say so |
| D7 | Token-based autoregressive models of ECG | han2025ecgbyte | `2412.14373v3.pdf` | 1 | "we propose ECG-Byte, an adapted byte pair encoding (BPE) tokenizer pipeline for autoregressive language modeling of ECGs." | **checked** |
| D8 | Quantised ECG as discrete symbols | han2025ecgbyte | same | 2 | "we use quantization to represent amplitude ranges as discrete symbols and obtain string representations of ECGs." | **checked** (closest to our tokenisation, D19) |
| D9 | Autoregressive ECG-token pre-training | yang2025heartllm | `2508.15338v2.pdf` | 1 | "we pretrain the model on an autoregressive ECG token forecasting task, allowing the LLM to capture temporal dynamics through its inherent language modeling capability." | **checked** |
| D10 | Text-to-ECG (Auto-TTE) | — | — | — | NotebookLM quotes arXiv 2303.09395 | `[verify]`: PDF not in the folder |
| D11 | Perplexity related to source entropy | chelba2017ngram | `46183.pdf` | 2 | "A commonly used quality measure for a given model M is related to the entropy of the underlying source and was introduced under the name of perplexity (PPL)" | **checked** |
| D12 | Cross-entropy as the training loss of an LM | chelba2017ngram | same | 3 | "The loss function optimized in training is the cross-entropy between the model pmf P (w\|h; θ) in some n-gram context h and the relative frequency" | **checked** |
| D13 | Entropy reported in bits per symbol | pascanu2013difficulty | `1211.5063v2.pdf` | 8 | "Results on the next character prediction task in entropy (bits/character)" | **partial**: a precedent for bits per symbol; "vocabulary-independent" is our own argument (bits = cross-entropy / ln 2 per sample) |

## F. Course materials (`contexto_DL/`, not in this repository)

Course materials are cited as course materials (slide or line), never as the primary paper they mention.

| ID | Claim | Source | Place | Quote | Status / use |
|---|---|---|---|---|---|
| F1 | Pre-LN optimises more easily than Post-LN (D18) | `Presentations/Attention and Transformers/PRDL_MLLB_Transformer_Architecture_Part_II.pptx` | slide 36 (slide 35 cites arXiv 2002.04745) | "With enough training, Pre-LN and Post-LN converge to similar functions. However, Pre-LN simply makes the optimization much easier." | **checked**; cite the course slides, not Xiong et al. (no PDF) |
| F2 | LSTM variants "all about the same" (D17) | `Presentations/RNN/PRDL_MLLB_RecurrentNetworks.pptx` | slide 64 | "Greff, et al. (2015) do a nice comparison of popular variants, finding that they're all about the same." | **checked**; primary source B7 |
| F3 | Transfer learning vs self-supervised learning | `PRDL_MLLB_summary_9_10_2026.txt` | lines 129-135 | "Both re-use knowledge, so they both can be "transfer learning", but usually transfer learning is trained using supervised learning (with manually labelled data), while self-supervised learning builds a more general model without manual labels, to address a complex task, called pretext task" | **checked** |
| F4 | Zero-shot with other modalities needs a shared embedding space | same | lines 189-195 | "In the case of multimodality the zero-shot is generally possible thanks to a Multimodal Correspondence that is to develop embedded representations of different modalities in the same vector space." | **checked** (the text file has lost characters elsewhere, not in this sentence) |
| F5 | Annex of takeaways on learning strategies required in the final PRDL report | same | lines 117-122 | "include an annex with your main takeaways in your final DL PRDL (and optionally MLLB) Report." | **checked**. Material for the annex: `learning_strategies.md` |
| F6 | Zero-shot and few-shot definitions come from the GPT-3 paper | same | lines 138-145 | "From an NLP perspective, several definitions were proposed in the research paper where GPT3 was published:" | **checked** as a course statement. The course lists "Mann, B., Ryder, N., ..." first; the paper's first author is Brown. The definitions themselves: no PDF, not quoted |

## G. Closure of every `[verify]` of the project (2026-11-24)

Rule for the reports: a statement is written as a fact only with a row above marked checked (or the checked part of a partial row). Everything else is reworded as below, stated as a hypothesis, or left out.

| Origin | `[verify]` | Outcome |
|---|---|---|
| D12, block note 02 | meaning of `sex` 0/1 | A13 (male 0, female 1) |
| D12, block note 02 | why folds 9-10 are human-validated | A14, A15 |
| block note 02 | NORM with mild conduction findings is clinically usual | No source: report the co-occurrence as a property of the annotation only, no clinical judgement |
| AI log 2026-11-20 | how the benchmark treats records without superclass | A6 (left out, as in our D12) |
| D14 | Adam less sensitive to the learning rate than SGD | Reword with B1: "its hyper-parameters typically require little tuning"; no comparison with SGD |
| D15 | dropout 0.5 | B2 |
| D15 | batch norm before the activation | B3 |
| block note 04 | limb-lead bands = linear combinations | B12 (partial): construction of the augmented leads is a fact; the reading of the weights stays a hypothesis |
| D16 | QRS 80-100 ms | Reworded (B9): only "kernel 7 = 70 ms at 100 Hz" |
| D16, block note 05 | basic block of He et al.; depth of degradation | B19 (our 1D adaptation), B4 (18 vs 34 layers; ours at most 12) |
| D16, block note 05 | CAM (Zhou et al.) | B5 (our variant keeps the bias, uses the mean) |
| block note 05 | most ECG energy at low frequencies | B10 (partial): "the main ECG waves lie below 50 Hz"; the filter bank result is ours |
| D17, block note 06 | Greff et al. | B7, F2 |
| D17, block note 06 | Bahdanau et al. | B18 |
| D17, block note 06 | Mikolov 2010 (first use of clipping) | No PDF: cite Pascanu et al. (B6) for clipping; drop "first published in" |
| block note 06 | Hochreiter and Schmidhuber 1997 | No PDF: name the LSTM without attributing it, or add the PDF |
| D18, block note 07 | Vaswani et al. (attention, multi-head, Post-LN, sinusoidal, dropout 0.1) | B13-B17 |
| D18, block note 07 | Xiong et al. 2020, Pre-LN | F1 (course slides) |
| block note 07 | Devlin (CLS), Dosovitskiy (patches), Hendrycks (GELU) | No PDF: describe the components without attribution |
| D19, block note 13 | GPT decoder-only (Radford et al.) | D1, D2 |
| D19, block note 13 | top-k (Fan et al.) | D3 |
| block note 13 | nucleus (Holtzman et al.) | D4 |
| D19, block note 13 | mu-law (WaveNet) | No PDF: write "non-uniform (mu-law) quantisation was not tried", without attribution |
| D19, block note 13 | causal Transformers without positional encoding | D6 (contrast with our D19 result) |
| block note 13 | autoregressive models of ECG | D7-D9 |
| block note 13 | bits per sample | D11-D13 (vocabulary independence is our argument) |
| D20, block note 08 | what CLIP's training pairs contain | C1; that ECG images with diagnoses are rare in them is **not** in the source: the CLIP failure stays a hypothesis |
| block note 08 | leads showing MI and HYP | B11 for MI (chest leads); HYP: no source, hypothesis only |
| HANDOFF chat 08 | transfer vs self-supervised vs zero-shot | F3, F4, F6, C1-C9; GPT-3 cited through the course summary only |

## E. Errors found in the NotebookLM answers (2026-11-24)

Logged as AI Error Bounty candidates (tool: NotebookLM; detected by the assistant in the checking stage).

1. N1, claim 1: values from the journal Table I, but attributed to the arXiv v1 file, its "TABLE II" caption and page 4. In v1 several intervals differ (e.g. resnet1d_wang .930(06), Wavelet+NN .874(08)) and there is a `naive` row that the journal table does not have. Also ".930(05)" was rendered as "95% CI 0.930 ± 0.005", while the paper defines it as the maximal absolute deviation (the interval need not be symmetric).
2. Wrong page numbers in most answers: Wagner 2020 2→1, 8→12/13, 4→6, 12→15; Pascanu 1→6 and 3→8; CLIP 4→6 and 22→11; Radford 2018 3→5; Holtzman 4→5; Chelba 1-2→2-3; Monka 3→2 and 3→4; Mehari 6→7-8; Soltanieh 2023 14→6; Srivastava 1941→1955; the filtering paper 8→12.
3. N1, claim 7: the licence quoted is the article's, not the dataset's.
4. N2, claim 8: a frequency band given as support for a QRS duration.
5. N4, claim 2: top-k attributed to Holtzman's description instead of the primary source Fan et al. (which was in the notebook).
6. N5: the "statements" listed are the source definitions themselves, and the three "NOT SUPPORTED" statements cannot be traced to any uploaded first assignment. N5 is not used until it is rerun with each author's first assignment.
