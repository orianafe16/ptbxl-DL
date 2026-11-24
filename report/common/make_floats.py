"""Write one LaTeX float file per shared figure and table (report/common/floats/).

Chat DL 10, 2026-11-24. Figures, tables and numbers are shared by the four DL
reports (DECISIONS D9): each float is written here once, with its caption and
label, so that both authors \\input the same file. Captions describe what is
drawn and contain no measured number (numbers live in the tables and in
numbers.tex). Run from report/:  python common/make_floats.py

Labels: fig:<name> and tab:<name>, <name> = the file name in figures/ or
tables/ (underscores kept).
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "floats"

# (name, caption). Figure files: figures/<name>.pdf
FIGURES = [
    # ---- Data and EDA (T6, OF) ----
    ("eda_class_counts",
     "PTB-XL superclasses. (a) Labelled records per superclass (a record may "
     "carry several). (b) Prevalence of each class in the training (folds 1--8), "
     "validation (fold 9) and test (fold 10) splits."),
    ("eda_label_cooccurrence",
     "Multi-label structure. (a) Co-occurrence: percentage of the records of the "
     "row class that also carry the column class (not symmetric). (b) Most "
     "frequent label combinations; blue: one superclass, orange: several."),
    ("eda_fold_distribution",
     "Prevalence of each superclass in each of the ten official folds; the "
     "shaded folds are validation (9) and test (10)."),
    ("eda_age_sex",
     "Demographics. (a) Age histogram by sex (sex 0 = male, sex 1 = female "
     "\\cite{wagner2020ptbxl}); ages above 89 are coded 300 in the dataset and not "
     "drawn. (b) Age per superclass (box: quartiles, line: median, whiskers: 1.5 "
     "interquartile ranges)."),
    ("eda_signal_quality",
     "Signal-quality notes of the metadata. (a) Share of records with each note "
     "per split. (b) Share of the records of each class with each note. No record "
     "is excluded."),
    ("eda_lead_stats",
     "Per-lead statistics of the training records. (a) Standard deviation of the "
     "samples (the values used by the per-lead z-score). (b) Peak-to-peak "
     "amplitude per record (box: quartiles; whiskers: 5th and 95th percentiles)."),
    ("eda_example_traces",
     "Lead II of one training record per superclass (records with a single "
     "superclass, drawn at random with seed 42)."),
    ("eda_example_12lead",
     "One model input: the 12 leads $\\times$ 1000 samples (10\\,s at 100\\,Hz) of "
     "the NORM example."),
    # ---- Dense (T9, OF) ----
    ("dense_curves",
     "Dense network, seed 42. (a) Baseline: training loss (measured in training "
     "mode, with dropout) and validation loss; dashed line: best epoch by "
     "validation macro AUROC. (b) The same without dropout. (c) Validation macro "
     "AUROC with and without input z-score and batch norm."),
    ("dense_ablations",
     "Dense ablations: validation macro AUROC of every configuration (dots: seeds "
     "42, 43, 44; bar: mean; dashed line: baseline mean)."),
    ("dense_first_layer",
     "First-layer weights of the final Dense model (\\texttt{no\\_norm}, seed 42) for "
     "the three hidden units with the largest weight norm, reshaped to 12 leads "
     "$\\times$ 1000 samples."),
    # ---- CNN (T10, OF) ----
    ("cnn_curves",
     "1D CNN, seed 42. (a) Baseline training and validation loss; dashed line: "
     "best epoch. (b) Validation macro AUROC without batch norm, without residual "
     "connections and with flatten instead of global average pooling. (c) "
     "Training loss with and without residual connections, at 4 and 6 blocks."),
    ("cnn_ablations",
     "CNN ablations: validation macro AUROC of every configuration (dots: seeds "
     "42, 43, 44; bar: mean; dashed line: baseline mean). Note the narrow "
     "horizontal range."),
    ("cnn_first_layer",
     "First convolutional layer of the final CNN (\\texttt{no\\_dropout}, seed 42). "
     "(a) The eight filters with the largest norm (12 leads $\\times$ 7 taps). (b) "
     "Frequency response of every filter (RMS over leads, normalised), sorted by "
     "peak frequency."),
    ("cnn_feature_maps",
     "Final CNN on one validation record per superclass (seed 42). Rows: lead II "
     "(z-score), feature maps of block 1 and of the last block, and the class "
     "activation map of the record's class (orange: evidence for the class, blue: "
     "against). Column titles: predicted probability and the class threshold "
     "chosen on validation; NORM and STTC are missed."),
    # ---- RNN (T11, MM) ----
    ("rnn_curves",
     "Recurrent networks, seed 42. (a) LSTM baseline losses. (b) Validation macro "
     "AUROC of the baseline, GRU, one recurrent layer and bidirectional LSTM. (c) "
     "Training loss with 100 frames, the convolutional front end, 1000 raw steps "
     "and 1000 steps with attention."),
    ("rnn_ablations",
     "RNN ablations: validation macro AUROC (dots: seeds 42, 43, 44; bar: mean; "
     "dashed line: baseline mean). Rows marked ``vs.'' are compared with their own "
     "reference."),
    ("rnn_time_gradient",
     "Which part of the 10\\,s the output depends on (share of the input gradient "
     "per 0.1\\,s, seed 42, 512 validation records; dashed: uniform). (a) Readout "
     "and direction. (b) Input sequence."),
    ("rnn_attention",
     "Attention weights over time (bars) on lead II for one validation record per "
     "superclass: final RNN + attention (\\texttt{raw\\_attn}, top) and the notebook "
     "tanh score over 100 frames (bottom); dashed: uniform weight $1/T$. Titles: "
     "predicted probability and threshold."),
    # ---- Transformer (T12, MM) ----
    ("transformer_curves",
     "Transformer encoder, seed 42. (a) Baseline losses. (b) Self-attention step "
     "by step: validation macro AUROC from one single-head attention layer to two "
     "full blocks. (c) Training loss with Pre-LN and Post-LN."),
    ("transformer_ablations",
     "Transformer ablations: validation macro AUROC (dots: seeds 42, 43, 44; bar: "
     "mean; dashed line: baseline mean)."),
    ("transformer_time_gradient",
     "Which part of the 10\\,s the Transformer output depends on (share of the "
     "input gradient per 0.1\\,s, seed 42, 512 validation records). (a) Position, "
     "readout, one layer. (b) Patch length."),
    ("transformer_attention",
     "Attention received by each patch (bars) on lead II, one validation record "
     "per superclass: single attention layer (\\texttt{sa\\_single}, top) and last "
     "layer of the final Transformer (\\texttt{no\\_pos}, bottom). Titles: predicted "
     "probability and threshold."),
    ("transformer_attention_maps",
     "Self-attention maps of one validation NORM record (row: query patch, column: "
     "key patch): the single attention layer, both layers of the final Transformer "
     "(heads averaged) and the four heads of its last layer."),
    # ---- Generative (T23, OF) ----
    ("generative_curves",
     "Generators of lead II, seed 42. (a) Cross-entropy per token of the ECG-GPT "
     "baseline and the LSTM. (b) Validation bits per sample of every "
     "configuration. (c) The ECG-GPT without the causal mask: training and "
     "validation cross-entropy over all positions and at the last position, with "
     "the baseline's last position for reference."),
    ("generative_ablations",
     "Generator ablations: validation bits per sample (lower is better; dots: "
     "seeds 42, 43, 44; bar: mean; dashed line: baseline mean; dotted: bigram "
     "counting model with $V = 256$)."),
    ("generative_context",
     "Prediction cost against the context available (validation, seed 42): bits "
     "per sample at each position of the window, i.e.\\ predicting with $t+1$ past "
     "samples. Logarithmic horizontal axis."),
    ("generative_samples",
     "Lead II of one validation record per superclass: real 2\\,s prompt (black), "
     "real continuation (grey) and generated continuation (colour) of the selected "
     "ECG-GPT ($V = 2560$, $T = 1$), the LSTM ($T = 1$) and the ECG-GPT with greedy "
     "search."),
    ("generative_sampling",
     "Sampling settings (256 validation records, 2\\,s prompt, 8\\,s generated). (a) "
     "Amplitude distributions of the generated signals against the real ones "
     "($V = 2560$). (b) Median ratio of generated to real standard deviation. (c) "
     "Mean absolute heart-rate error."),
    ("generative_attention",
     "Causal self-attention of the selected ECG-GPT ($V = 2560$, seed 42) on one "
     "validation window of 2.5\\,s (NORM record, right): both layers with the heads "
     "averaged and the four heads of layer 2. Row: predicting position; column: "
     "attended position; everything above the diagonal is zero."),
    # ---- Learning strategies and comparison (T13, T14, T15, T24; OF + MM) ----
    ("transfer_curves",
     "Transfer from the ECG-GPT, lead II, seed 42: validation macro AUROC with "
     "(a) 100\\,\\% and (b) 10\\,\\% of the training labels, and (c) training loss of "
     "the three strategies."),
    ("transfer_strategies",
     "ECG-GPT reused for the five superclasses (lead II): test macro AUROC of "
     "training from scratch, linear probe on the frozen network and fine-tuning, "
     "with 10\\,\\% and 100\\,\\% of the training labels (dots: seeds; line: mean)."),
    ("transfer_probe_study",
     "Probe study: validation bits per sample of each generator of the "
     "generative block against the validation macro AUROC of a linear probe on its "
     "frozen features (mean over seeds), with 100\\,\\% and 10\\,\\% of the labels."),
    ("leads_comparison",
     "Final model of each family with 12 leads and with lead II only: test macro "
     "AUROC (one line per seed)."),
    ("zeroshot_examples",
     "Zero-shot CLIP on lead II drawn as an image (one validation record per "
     "superclass): the image given to CLIP and the probability of each class "
     "sentence; orange: the true class."),
    ("zeroshot_ptbxl",
     "Zero-shot CLIP on the test fold: AUROC per superclass and macro AUROC with "
     "95\\,\\% patient-bootstrap intervals; dashed line: chance."),
    ("comparison_overview",
     "Every final model on the test fold: macro AUROC (dots: seeds; bar: mean; "
     "dashed: chance). Blue: 12 leads; orange: lead II; green: ECG-GPT transfer "
     "(lead II); yellow: zero-shot CLIP (lead II images)."),
    ("comparison_label_efficiency",
     "Test macro AUROC against the share of training labels used (0\\,\\% = "
     "zero-shot) for the lead-II models, with the CNN on lead II and on 12 leads "
     "(100\\,\\% of the labels) as horizontal references."),
]

# (name, caption). Table files: tables/<name>.tex (generated by the blocks)
TABLES = [
    ("dense_ablations",
     "Dense ablations on the validation fold: mean $\\pm$ SD over seeds 42--44; "
     "$\\Delta$ = mean paired difference with the baseline of the same seed; seeds "
     "better / worse = seeds whose paired 95\\,\\% patient-bootstrap interval lies "
     "wholly above / below 0."),
    ("dense_final",
     "Final Dense model (\\texttt{no\\_norm}) on the test fold, with the trivial "
     "model (logistic regression on the mean and standard deviation of each lead)."),
    ("cnn_ablations",
     "CNN ablations on the validation fold (same conventions as "
     "Table~\\ref{tab:dense_ablations}). RF: receptive field of one position of "
     "the last feature map, in input samples."),
    ("cnn_final",
     "Final CNN (\\texttt{no\\_dropout}) on the test fold, with the final Dense and "
     "trivial models, and paired differences per seed (95\\,\\% patient bootstrap)."),
    ("rnn_ablations",
     "RNN ablations on the validation fold (conventions of "
     "Table~\\ref{tab:dense_ablations}; each row is compared with its own "
     "reference). Last 2\\,s: share of the input sensitivity in the last 2\\,s of "
     "the record (uniform: 0.2)."),
    ("rnn_attention",
     "Attention statistics of the RNN + attention models (seed 42, validation)."),
    ("rnn_final",
     "Final RNN (\\texttt{conv}) and RNN + attention (\\texttt{raw\\_attn}) on the "
     "test fold, with the final CNN, Dense and trivial models."),
    ("transformer_ablations",
     "Transformer ablations on the validation fold (conventions of "
     "Table~\\ref{tab:dense_ablations}). The step-by-step rows are compared with "
     "the next step. Shuffled: change in validation macro AUROC when the patches "
     "of every record are put in random order."),
    ("transformer_attention",
     "Attention statistics of the Transformer configurations (seed 42, "
     "validation)."),
    ("transformer_final",
     "Final Transformer (\\texttt{no\\_pos}) on the test fold, with the final CNN, "
     "RNN, RNN + attention, Dense and trivial models."),
    ("generative_ablations",
     "Generators of lead II on the validation fold (mean $\\pm$ SD over seeds "
     "42--44). CE: cross-entropy per token (nats), not comparable across "
     "vocabularies; bits per sample: comparable; $\\Delta$ bits: mean paired "
     "difference with the baseline (negative = better). The model without the "
     "causal mask is not compared (it sees its target)."),
    ("generative_sampling",
     "Generation statistics (seed-42 models, 256 validation records, 2\\,s prompt, "
     "8\\,s generated): median ratio of generated to real standard deviation; share "
     "of flat records; Wasserstein-1 distance between amplitude distributions; "
     "share of records with a detected heart rate and mean absolute heart-rate "
     "error."),
    ("transfer_main",
     "ECG-GPT reused for the five superclasses (lead II): the three strategies "
     "with 100\\,\\% and 10\\,\\% of the training labels (mean $\\pm$ SD over seeds "
     "42--44)."),
    ("transfer_comparisons",
     "Paired differences in macro AUROC between transfer configurations (mean over "
     "seeds; seeds whose 95\\,\\% patient-bootstrap interval lies wholly above / "
     "below 0)."),
    ("transfer_probe_study",
     "Probe study: the same linear probe on the frozen features of every "
     "generator of the generative block (validation, mean over seeds)."),
    ("leads",
     "12 leads against lead II only, final model of each family (test fold, mean "
     "over seeds; $\\Delta$ = paired difference, lead II minus 12 leads)."),
    ("zeroshot",
     "Zero-shot CLIP ViT-B/32: CIFAR-100 (where zero-shot should work) and PTB-XL "
     "lead II images (95\\,\\% bootstrap intervals; patients as the unit on "
     "PTB-XL)."),
    ("comparison_final",
     "Final comparison on the test fold. Mean $\\pm$ SD over seeds; interval: 95\\,\\% "
     "patient bootstrap of the first seed. Published rows: macro AUC of the "
     "superclass task in Table~I of \\cite{strodthoff2021deep}, obtained in a "
     "different setting (described in the text)."),
    ("comparison_per_class",
     "Test AUROC per superclass of every final model (mean over seeds)."),
    ("comparison_cross",
     "Cross-block paired differences on the test fold (mean over seeds)."),
]


# Short titles for the list of figures / tables
SHORT_FIG = {'eda_class_counts': 'Records and prevalence per superclass and split', 'eda_label_cooccurrence': 'Co-occurrence and combinations of superclasses', 'eda_fold_distribution': 'Class prevalence per fold', 'eda_age_sex': 'Age and sex', 'eda_signal_quality': 'Signal-quality notes', 'eda_lead_stats': 'Per-lead spread and amplitude', 'eda_example_traces': 'Lead II of one record per superclass', 'eda_example_12lead': 'One model input: 12 leads', 'dense_curves': 'Dense network: learning curves', 'dense_ablations': 'Dense network: ablations', 'dense_first_layer': 'Dense network: first-layer weights', 'cnn_curves': '1D CNN: learning curves', 'cnn_ablations': '1D CNN: ablations', 'cnn_first_layer': '1D CNN: first-layer filters', 'cnn_feature_maps': '1D CNN: feature maps and class activation maps', 'rnn_curves': 'Recurrent networks: learning curves', 'rnn_ablations': 'Recurrent networks: ablations', 'rnn_time_gradient': 'Recurrent networks: input sensitivity over time', 'rnn_attention': 'RNN + attention: attention weights over time', 'transformer_curves': 'Transformer: learning curves and self-attention step by step', 'transformer_ablations': 'Transformer: ablations', 'transformer_time_gradient': 'Transformer: input sensitivity over time', 'transformer_attention': 'Transformer: attention received by each patch', 'transformer_attention_maps': 'Transformer: self-attention maps', 'generative_curves': 'Generators: learning curves', 'generative_ablations': 'Generators: ablations in bits per sample', 'generative_context': 'Generators: prediction cost against context', 'generative_samples': 'Generators: real and generated lead II', 'generative_sampling': 'Generators: sampling settings', 'generative_attention': 'ECG-GPT: causal self-attention maps', 'transfer_curves': 'Transfer from the ECG-GPT: learning curves', 'transfer_strategies': 'Transfer from the ECG-GPT: strategies and label fractions', 'transfer_probe_study': 'Probe study: generator quality against feature quality', 'leads_comparison': '12 leads against lead II', 'zeroshot_examples': 'Zero-shot CLIP: examples', 'zeroshot_ptbxl': 'Zero-shot CLIP on PTB-XL', 'comparison_overview': 'All final models on the test fold', 'comparison_label_efficiency': 'Test AUROC against the share of labels'}
SHORT_TAB = {'dense_ablations': 'Dense network: ablations (validation)', 'dense_final': 'Dense network: final model (test)', 'cnn_ablations': '1D CNN: ablations (validation)', 'cnn_final': '1D CNN: final model (test)', 'rnn_ablations': 'Recurrent networks: ablations (validation)', 'rnn_attention': 'RNN + attention: attention statistics', 'rnn_final': 'Recurrent networks: final models (test)', 'transformer_ablations': 'Transformer: ablations (validation)', 'transformer_attention': 'Transformer: attention statistics', 'transformer_final': 'Transformer: final model (test)', 'generative_ablations': 'Generators: ablations (validation)', 'generative_sampling': 'Generators: generation statistics', 'transfer_main': 'Transfer from the ECG-GPT: main results', 'transfer_comparisons': 'Transfer from the ECG-GPT: paired differences', 'transfer_probe_study': 'Probe study', 'leads': '12 leads against lead II', 'zeroshot': 'Zero-shot CLIP', 'comparison_final': 'Final comparison (test)', 'comparison_per_class': 'Test AUROC per superclass', 'comparison_cross': 'Cross-block paired differences'}


def figure(name: str, caption: str) -> str:
    return (f"% Shared float (common/make_floats.py). Figure: figures/{name}.pdf\n"
            "\\begin{figure}[htbp]\n\\centering\n"
            f"\\includegraphics[width=\\textwidth,height=0.42\\textheight,keepaspectratio]{{figures/{name}.pdf}}\n"
            f"\\caption[{SHORT_FIG[name]}]{{{caption}}}\n\\label{{fig:{name}}}\n\\end{{figure}}\n")


def table(name: str, caption: str) -> str:
    return (f"% Shared float (common/make_floats.py). Table: tables/{name}.tex\n"
            "\\begin{table}[htbp]\n\\centering\n"
            f"\\caption[{SHORT_TAB[name]}]{{{caption}}}\n\\label{{tab:{name}}}\n\\small\n"
            "\\begin{adjustbox}{max width=\\textwidth}\n"
            f"\\input{{tables/{name}}}\n"
            "\\end{adjustbox}\n\\end{table}\n")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for name, cap in FIGURES:
        (OUT / f"fig_{name}.tex").write_text(figure(name, cap), encoding="utf-8")
    for name, cap in TABLES:
        (OUT / f"tab_{name}.tex").write_text(table(name, cap), encoding="utf-8")
    print(f"{len(FIGURES)} figure floats and {len(TABLES)} table floats in {OUT}")


if __name__ == "__main__":
    main()
