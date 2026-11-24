# Learning strategies: cited material for the PRDL annex (T18)

Chat DL 09, 2026-11-24. The weekly summary of 9/10 asks for "an annex with your main takeaways" on learning strategies in the final DL PRDL report (`quotes.md`, F5). The first assignments (class presentations) are not in the folder; by decision of OF and MM this material is built without them, from the checked quotes of `quotes.md` (IDs in brackets) and linked to what area 2 of our project actually did (D7, D20, block note 08). Each author writes her own annex text in her report chat (DL 10, DL 11); numbers come only from `numbers.tex`.

## 1. Transfer learning

- What it is: re-using knowledge learnt on one task or domain for another [C7]. In the course view, it is usually trained with manually labelled data [F3].
- Two ways of re-using a pre-trained network: a linear classifier on frozen features (linear probe) or end-to-end fine-tuning [C4].
- Fine-tuning uses a smaller learning rate so that the pre-trained weights are not overwritten (catastrophic forgetting) [C5, A11].
- On ECG, pre-training on PTB-XL and fine-tuning on a smaller dataset helps most when the target training set is small [A10].
- **In our project (T13):** `scratch` vs `probe` vs `finetune` of the ECG-GPT on the 5 superclasses, with 100 % and 10 % of the labels; fine-tuning at a learning rate ten times smaller than the head-only probe (D20), and the 2 × 2 ablation of initialisation × learning rate.

## 2. Self-supervised learning

- What it is: learning features without manual labels, to overcome label scarcity [C9]; the model solves a pretext task and is then re-used for downstream tasks [F3].
- Course point: both transfer and self-supervised learning re-use knowledge; the difference is the labels used in pre-training [F3].
- On 12-lead ECG, self-supervised pre-training reaches the supervised performance with less labelled data [C6].
- A generative pretext task is the GPT recipe: a decoder-only Transformer trained with the next-token objective, then fine-tuned [D1, D2]. Token-based autoregressive pre-training has also been applied to ECG [D7-D9].
- **In our project (T23 → T13):** the ECG-GPT is pre-trained on lead II of the training folds with next-sample prediction (pretext task, no labels), then transferred. "10 %" means 10 % of the labels, not of the signals (D20). The probe study found that the generator chosen for generation quality was not the one with the best frozen features: a self-supervised objective is not automatically the best for every downstream task (finding, block note 08).

## 3. Zero-shot learning

- What it is (visual transfer view): classes of the target task are not seen with labels during training [C8]. In NLP the zero-shot / few-shot terms come from the GPT-3 paper (course summary) [F6].
- With other modalities, zero-shot needs a shared embedding space for inputs and class descriptions ("multimodal correspondence") [F4].
- CLIP is that kind of model: trained on 400 million image-text pairs from the Internet [C1]; it classifies by cosine similarity between the image and each class prompt, scaled and passed through a softmax [C2], with prompts such as "A photo of a {label}." [C3].
- **In our project (T24):** CLIP on CIFAR-100 (where zero-shot should work) and on ECG images of PTB-XL with one sentence per superclass. Our ECG-GPT has no text side, so it can be transferred but not used zero-shot (D20). Why CLIP fails on ECG diagnosis is a **hypothesis**: the source says what kind of data CLIP was trained on [C1], not whether ECG traces with diagnoses are among it.

## 4. Connections (for the "main takeaways")

- Transfer, self-supervised and zero-shot are not exclusive: our ECG-GPT is self-supervised pre-training followed by transfer (probe or fine-tuning); CLIP is pre-training on paired data followed by zero-shot use [F3, F4].
- What decides the strategy is what is available: labels for the target task (transfer), unlabelled signals (self-supervised), or a shared space with text (zero-shot).
- The amount of target labels matters: the benefit of pre-training grows as the labelled set shrinks [A10, C6], which is why T13 was run at 100 % and 10 %.

## Not usable without a source (do not write as fact)

- Definitions quoted directly from the GPT-3 paper (no PDF in the folder; F6 only says that the course takes them from there).
- Any statement about how many ECG images CLIP's training data contain.
