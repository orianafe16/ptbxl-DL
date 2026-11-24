# Literature prompts (chat DL 09)

Flow (DECISIONS D8): Perplexity finds candidates -> we download the PDFs by hand into `contexto_DL/literature/` -> NotebookLM extracts literal quotes with file and page -> the assistant checks every quote against the PDF and writes only from `quotes.md`. Anything without a checked quote stays `[verify]`.

Rules for every prompt: no invented papers, no quotes in stage 1, and "NOT FOUND" is an acceptable answer.

## Stage 1. Perplexity (candidate lists)

Paste each block as one prompt. Output wanted: a table, one row per paper.

### P1. PTB-XL, benchmark and ECG leads

```
I am writing a university deep learning report on PTB-XL (12-lead ECG, 5 diagnostic superclasses NORM, MI, STTC, CD, HYP, recommended folds 1-8 train / 9 validation / 10 test, 100 Hz).
Find the primary sources (peer-reviewed or arXiv) for:
1. The PTB-XL dataset paper (Wagner et al. 2020) and the PhysioNet dataset page.
2. The PTB-XL benchmark paper (Strodthoff et al., "Deep Learning for ECG Analysis: Benchmarks and Insights from PTB-XL"), both the arXiv version (2004.13701) and the journal version, and its code repository.
3. What the benchmark does with records that have no diagnostic superclass.
4. A clinical source on which ECG leads show myocardial infarction and which show ventricular hypertrophy.
5. The meaning of the sex coding (0/1) in PTB-XL, and why folds 9 and 10 contain only human-validated reports.
6. The usual duration of a QRS complex in healthy adults.
Return a table with: title, authors, year, venue, DOI, arXiv id (if any), direct open-access PDF link, and one line on which of the six points it supports. If you cannot find a source for a point, write "no source found". Do not give numbers or quotes. Do not invent identifiers.
```

### P2. Architectures and training

```
I need primary sources for these deep learning claims in a report on 1D ECG classification:
1. Adam optimiser and its lower sensitivity to the learning rate than plain SGD.
2. Dropout and the usual rate of 0.5 for hidden layers (Srivastava et al. 2014).
3. Batch normalisation and its placement before the activation (Ioffe and Szegedy 2015).
4. Residual networks and skip connections (He et al. 2016), and plain networks degrading at large depth.
5. Class activation maps (Zhou et al. 2016).
6. Gradient clipping for recurrent networks (Pascanu et al. 2013; Mikolov 2010), LSTM variants (Greff et al. 2015), GRU (Cho et al. 2014).
7. Additive attention over RNN states (Bahdanau et al. 2015), the Transformer (Vaswani et al. 2017), and Pre-LN versus Post-LN (Xiong et al. 2020).
8. Most of the ECG signal energy lying at low frequencies.
Return a table: title, authors, year, venue, DOI, arXiv id, direct open-access PDF link, claim number it supports. Mark "no source found" where that is the case. Do not give quotes or numbers. Do not invent identifiers.
```

### P3. Learning strategies

```
I need primary sources for a section on learning strategies (transfer learning, fine-tuning, self-supervised pre-training, zero-shot) applied to ECG:
1. CLIP (Radford et al. 2021): what its training pairs contain.
2. GPT-3 (Language Models are Few-Shot Learners, Brown et al. 2020; Mann is a co-author): zero-shot and few-shot.
3. A reference that contrasts transfer learning, self-supervised learning and zero-shot learning.
4. Self-supervised or contrastive pre-training on ECG signals, and its use for fine-tuning or linear probing with few labels.
5. Linear probing versus fine-tuning, and why a smaller learning rate is used for fine-tuning pre-trained layers.
6. Evidence on whether ECG images with diagnoses appear in general image-text training data.
Return a table: title, authors, year, venue, DOI, arXiv id, open-access PDF link, point(s) supported. Say "no source found" where needed. No quotes, no numbers, no invented identifiers.
```

### P4. Generative autoregressive models

```
I need primary sources for a section on autoregressive generation of ECG signals:
1. Decoder-only GPT models (Radford et al. 2018 and 2019).
2. Top-k sampling (Fan et al. 2018) and nucleus sampling (Holtzman et al. 2020).
3. mu-law quantisation of audio samples in WaveNet (van den Oord et al. 2016).
4. Causal Transformers without positional encoding (Haviv et al. 2022).
5. Autoregressive or token-based generative models of ECG or other biosignals (list only papers you can identify with certainty).
6. Cross-entropy and perplexity as evaluation of language models, and bits per symbol as a vocabulary-independent measure.
Return a table: title, authors, year, venue, DOI, arXiv id, open-access PDF link, point(s) supported. Say "no source found" where needed. No quotes, no numbers, no invented identifiers.
```

After each answer: open every DOI and arXiv id, download the PDF to `contexto_DL/literature/` as `<FirstAuthor><Year>_<ShortTitle>.pdf`, and discard any row that does not resolve.

## Stage 3. NotebookLM (literal quotes)

Upload only the verified PDFs of the theme. One prompt per theme. Replace the numbered list with the claims of the theme.

Common preamble (start every prompt with it):

```
Answer only from the uploaded sources. For each numbered claim below, return: (a) a literal quote copied exactly, at most 40 words; (b) the source file name; (c) the page number printed on the PDF page (not the viewer page). If a source does not contain the claim, write NOT FOUND. Do not paraphrase, summarise, combine sources or add anything that is not in the text. If a claim is only partly supported, quote the part that is supported and say what is missing.
```

### N1. PTB-XL and the benchmark (priority: O7)

```
[preamble]
Claims:
1. For the superclass diagnostic task, the macro-AUROC of every model in the benchmark table (give the model name, the exact value, the confidence interval if shown, and the table or figure it comes from).
2. The setting of those results: leads used, sampling rate, folds used for training, validation and test, and the metric definition.
3. How records without a diagnostic superclass are handled.
4. The number of records and the class counts of the superclasses.
5. Why folds 9 and 10 contain only human-validated reports.
6. The coding of the sex variable.
7. The licence of the dataset.
```

### N2. Architectures and training

```
[preamble]
Claims:
1. Adam is less sensitive to the learning rate than SGD.
2. Dropout rate 0.5 is the usual choice for hidden layers.
3. Batch normalisation is applied before the activation.
4. Residual blocks; plain networks degrade at large depth.
5. Class activation maps: the logit equals the bias plus the weighted mean of the activation maps.
6. Gradient clipping for recurrent networks; LSTM variants perform about the same.
7. Pre-LN optimises more easily than Post-LN.
8. QRS duration; low-frequency content of the ECG.
```

### N3. Learning strategies

```
[preamble]
Claims:
1. What the training pairs of CLIP contain and how they were collected.
2. CLIP zero-shot classification procedure (prompt, cosine similarity, softmax).
3. GPT-3 zero-shot and few-shot definitions.
4. Transfer learning versus self-supervised learning versus zero-shot learning (definitions).
5. Linear probing versus fine-tuning; smaller learning rate for fine-tuning.
6. Self-supervised pre-training on ECG and its gain with few labels.
```

### N4. Generative autoregressive models

```
[preamble]
Claims:
1. Decoder-only GPT: causal mask and next-token objective.
2. Top-k sampling.
3. Nucleus (top-p) sampling.
4. mu-law quantisation in WaveNet.
5. Causal Transformers can learn position information without positional encoding.
6. Autoregressive models of ECG or other biosignals and what they report.
7. Cross-entropy, perplexity and bits per symbol.
```

### N5. First assignment (learning strategies), one notebook per author

Upload that author's first assignment together with the sources of N3.

```
[preamble]
For each statement in the uploaded assignment that makes a factual claim about transfer learning, fine-tuning, self-supervised learning or zero-shot learning, list the statement, then the literal quote from the uploaded papers that supports it (file and page), or NOT SUPPORTED. List statements without support separately at the end.
```

## Stage 4. Checks (assistant, in this chat)

Paste NotebookLM's answers here. For each quote: confirm it appears verbatim in the PDF and on that page; fix the page if it is the viewer page; send back to `[verify]` anything that fails. Then `docs/literature/quotes.md` (key, file, page, quote, claim, status), `references.bib` (DOI and arXiv checked) and `report/published.json` (O7) are written from the checked quotes only.
