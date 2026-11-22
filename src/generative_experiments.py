"""Generative block (T23): ECG-GPT and LSTM generators of lead II, ablations,
generation, statistics, and the pre-trained model for chat DL 08.

Decisions: docs/DECISIONS.md D19 (this block), D13 (scope), D7 (reuse),
D3, D4 and D14 (protocol). Same structure as src/transformer_experiments.py.

    python -m src.generative_experiments estimate    # time per epoch and GPU
                                                     # memory, before the runs
    python -m src.generative_experiments all --fast  # smoke test of everything
    python -m src.generative_experiments run         # 8 configurations x 3
                                                     # seeds, validation only
    python -m src.generative_experiments report      # baselines, selection,
                                                     # generation, summary,
                                                     # tables, figures

`run` can be stopped and started again: a configuration and seed that
already has a (non --fast) log is skipped. There is no `final` step: the
generative experiments never read the test fold (D13); chat DL 08 uses it
for the 5-superclass task only.

How an ablation works: the baseline is the ECG-GPT, a 2-block causal
Transformer over windows of 250 tokens (2.5 s) of lead II quantised into
256 levels of 10 uV. Each other configuration changes ONE factor - the
architecture (LSTM), the context length, the vocabulary size, the
positional encoding or the causal mask - and keeps everything else equal
(data, windows, loss, optimiser, learning rate, batch size, early stopping,
gradient clipping, seeds). Configurations are compared in bits per sample,
which does not depend on the vocabulary (src/generative.py).

The pre-trained model handed to chat DL 08 is the configuration with the
lowest mean validation bits per sample, chosen by the rule of D19 before
any transfer experiment.

Outputs of `report`:
    results/logs/generative_summary.json   numbers for make_numbers.py
                                           (alias "generative" in runs.json)
    report/tables/generative_ablations.tex  configurations and baselines
    report/tables/generative_sampling.tex   statistics of generated signals
    report/figures/generative_curves.*      learning curves (seed 42)
    report/figures/generative_ablations.*   bits per sample per configuration
    report/figures/generative_context.*     cross-entropy vs position in the
                                            window (context available)
    report/figures/generative_samples.*     real vs generated lead II
    report/figures/generative_sampling.*    greedy, temperature and top-k
    report/figures/generative_attention.*   causal attention maps
With --fast every output name ends in _fast and git ignores it.
"""
import os

# Matrix products on the GPU (attention, LSTM) are made reproducible by
# cuBLAS with this variable, as in D17 and D18. It must be set before CUDA
# starts.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:2")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from datetime import datetime  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src import cnn_experiments as cx  # noqa: E402   generic helpers
from src import config, runlog  # noqa: E402
from src import dense_experiments as dx  # noqa: E402
from src import generative as gx  # noqa: E402
from src import rnn_experiments as rx  # noqa: E402
from src.dense_experiments import Paths  # noqa: E402
from src.models.generative import build_generator  # noqa: E402
from src.train import TrainConfig, rel, run_experiment, set_seed  # noqa: E402

MODEL = "generator"                   # run names: generator_<tag>_s<seed>_...
SEEDS = (42, 43, 44)                  # D14
ALIAS = "generative"                  # alias of the summary in runs.json
SUMMARY_NAME = "generative_summary"
GRAD_CLIP = 1.0                       # max total gradient norm (D19, as D17)
N_GEN = 256                           # validation records for generation
FAST_N_GEN = 16

# Baselines (D19). Every ablation starts from one of them.
BASE_ARGS = {"arch": "gpt", "vocab": 256, "context": 250, "d_model": 128,
             "num_heads": 4, "num_layers": 2, "d_ff": 512, "dropout": 0.1,
             "pos_encoding": "sinusoidal", "causal": True}
LSTM_ARGS = {"arch": "lstm", "vocab": 256, "context": 250, "d_embed": 128,
             "hidden_size": 160, "num_layers": 2, "dropout": 0.1}


@dataclass(frozen=True)
class Ablation:
    tag: str                  # run name part; letters, digits, underscores
    factor: str               # family, used to group the table and figure
    label: str                # short English description for the report
    changes: dict = field(default_factory=dict)
    reference: str = "base"   # configuration it is compared with
    base: str = "gpt"         # "gpt" -> BASE_ARGS, "lstm" -> LSTM_ARGS
    selectable: bool = True   # may become the pre-trained model of chat 08

    @property
    def model_args(self) -> dict:
        start = BASE_ARGS if self.base == "gpt" else LSTM_ARGS
        return {**start, **self.changes}

    @property
    def vocab(self) -> int:
        return int(self.model_args["vocab"])

    @property
    def context(self) -> int:
        return int(self.model_args["context"])


ABLATIONS = [
    Ablation("base", "baseline",
             "ECG-GPT: causal Transformer, 2 blocks, L = 250, V = 256"),
    Ablation("lstm", "architecture", "LSTM, 2 layers of 160 (same size)",
             base="lstm"),
    Ablation("ctx100", "context length", "Context of 100 samples (1 s)",
             {"context": 100}),
    Ablation("vocab64", "vocabulary", "64 levels of 40 uV", {"vocab": 64}),
    Ablation("no_pos", "positional encoding", "No positional encoding",
             {"pos_encoding": "none"}),
    Ablation("no_mask", "causal mask", "No causal mask (sees the future)",
             {"causal": False}, selectable=False),
    # Larger output layer and longer windows go last, so a long `run` has
    # done everything else first.
    Ablation("vocab2560", "vocabulary", "2560 levels of 1 uV (lossless)",
             {"vocab": 2560}),
    Ablation("ctx500", "context length", "Context of 500 samples (5 s)",
             {"context": 500}),
]
BY_TAG = {a.tag: a for a in ABLATIONS}
VOCABS = sorted({a.vocab for a in ABLATIONS})


# ---------------------------------------------------------------------------
# 1. Configurations and runs
# ---------------------------------------------------------------------------
def make_config(ab: Ablation, seed: int, fast: bool = False) -> TrainConfig:
    """TrainConfig of one configuration: the D14 defaults (Adam, learning
    rate 1e-3, batch 64, at most 50 epochs, patience 5 on the validation
    cross-entropy) plus gradient clipping at GRAD_CLIP, as D17 and D18.
    `normalize` does not apply: the input is made of tokens (D19)."""
    cfg = TrainConfig(model=MODEL, tag=ab.tag, seed=seed,
                      model_args=ab.model_args, normalize=False,
                      grad_clip=GRAD_CLIP, fast=fast, final=False,
                      notes=ab.label)
    if fast:                          # D14: 2 epochs, 50 resamples
        cfg.max_epochs, cfg.n_boot = 2, 50
    return cfg


def find_runs(paths: Paths, fast: bool) -> dict:
    """(tag, seed) -> log of the latest matching generator run."""
    return cx.find_runs(paths, fast, False, model=MODEL)


def setup(ab: Ablation, cfg: TrainConfig, data: gx.GenData):
    """Loaders, data description and task of one configuration."""
    loaders, info = gx.make_loaders(data, ab.vocab, ab.context,
                                    cfg.batch_size, cfg.seed)
    info["cublas_workspace_config"] = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    task = gx.GenerativeTask(data.tokenizer(ab.vocab), ab.context,
                             info["val_windows_per_record"],
                             {"val": data.patient_id["val"]},
                             {"val": data.ecg_id["val"]},
                             n_boot=cfg.n_boot, seed=config.SEED)
    return loaders, info, task


def _train(ab: Ablation, seed: int, fast: bool, data: gx.GenData,
           paths: Paths) -> dict:
    cfg = make_config(ab, seed, fast)
    loaders, info, task = setup(ab, cfg, data)
    _, record = run_experiment(lambda: build_generator(cfg.model_args),
                               loaders, task, cfg, info, logs_dir=paths.logs,
                               checkpoint_dir=paths.checkpoints,
                               predictions_dir=paths.predictions)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()      # free the GPU before the next model
    return record


def load_data(cache: dict | None, fast: bool) -> gx.GenData:
    cache = dx._load_cache(cache)
    return gx.GenData(cache, fast=fast)


def run_ablations(seeds=SEEDS, only=None, fast: bool = False,
                  cache: dict | None = None, paths: Paths = Paths(),
                  data: gx.GenData | None = None) -> None:
    """Train every configuration with every seed, on validation only."""
    tags = only or [a.tag for a in ABLATIONS]
    unknown = set(tags) - set(BY_TAG)
    if unknown:
        raise ValueError(f"unknown configurations: {sorted(unknown)}")
    data = data or load_data(cache, fast)
    done = find_runs(paths, fast=False)
    todo = [(t, s) for t in tags for s in seeds]
    for i, (tag, seed) in enumerate(todo, 1):
        if not fast and (tag, seed) in done:
            print(f"[{i}/{len(todo)}] {tag} seed {seed}: already run "
                  f"({done[(tag, seed)]['run_id']}), skipped")
            continue
        print(f"\n[{i}/{len(todo)}] {tag} seed {seed}: {BY_TAG[tag].label}")
        _train(BY_TAG[tag], seed, fast, data, paths)


# ---------------------------------------------------------------------------
# 2. Selection of the pre-trained model (validation only)
# ---------------------------------------------------------------------------
def val_bits(runs: dict, seeds=SEEDS) -> dict:
    """tag -> validation bits per sample, one per seed (seed order). Only
    configurations run with every seed are returned."""
    out = {}
    for ab in ABLATIONS:
        values = [runs[(ab.tag, s)]["scalars"]["val_bits_per_sample"]
                  for s in seeds if (ab.tag, s) in runs]
        if len(values) == len(seeds):
            out[ab.tag] = values
    return out


def select_pretrained(runs: dict, seeds=SEEDS) -> str:
    """The selectable configuration with the LOWEST mean validation bits
    per sample over the seeds (D19). `no_mask` is not selectable: its
    teacher-forced loss is low because it reads the answer."""
    scores = val_bits(runs, seeds)
    missing = [a.tag for a in ABLATIONS if a.tag not in scores]
    if missing:
        raise RuntimeError(f"configurations not run with every seed "
                           f"{tuple(seeds)}: {missing}. Selecting now would "
                           "ignore them.")
    pool = [a.tag for a in ABLATIONS if a.selectable]
    return min(pool, key=lambda t: float(np.mean(scores[t])))


def comparison_model(selected: str) -> str:
    """The second model whose generation is studied in detail: the LSTM if
    a Transformer was selected, the Transformer baseline otherwise."""
    return "lstm" if BY_TAG[selected].base == "gpt" else "base"


# ---------------------------------------------------------------------------
# 3. Time and memory estimate on this machine (before the long runs)
# ---------------------------------------------------------------------------
def estimate(cache: dict | None = None, n_steps: int = 10,
             epochs=(15, 30, 50), seeds=SEEDS, fast: bool = False,
             data: gx.GenData | None = None) -> dict:
    """Seconds per epoch and peak GPU memory of every configuration,
    measured on one real batch of its own windows (as
    transformer_experiments.estimate). The attention matrix grows with the
    square of the context, the output layer with the vocabulary. Nothing
    is written."""
    data = data or load_data(cache, fast)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device}; {len(data.idx['train'])} training records, "
          f"{len(data.idx['val'])} validation records")
    print(f"\n{'configuration':<14}{'L':>5}{'V':>6}{'params':>11}"
          f"{'train ms':>10}{'eval ms':>9}{'GPU MB':>8}{'s/epoch':>9}"
          f"  minutes per run at {', '.join(map(str, epochs))} epochs")
    out = {}
    for ab in ABLATIONS:
        cfg = make_config(ab, seeds[0], fast)
        loaders, _, task = setup(ab, cfg, data)
        x, y = next(iter(loaders["train"]))
        x, y = x.to(device), y.to(device)
        set_seed(seeds[0])
        model = build_generator(ab.model_args).to(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        t_tr = rx._time_steps(model, x, y, task, True, n_steps, device)
        t_ev = rx._time_steps(model, x, y, task, False, n_steps, device)
        mem = (torch.cuda.max_memory_allocated(device) / 2 ** 20
               if device.type == "cuda" else float("nan"))
        per_epoch = (len(loaders["train"]) * t_tr
                     + len(loaders["val"]) * t_ev)
        out[ab.tag] = per_epoch
        n_par = sum(p.numel() for p in model.parameters())
        mins = "  ".join(f"{per_epoch * e / 60:6.1f}" for e in epochs)
        print(f"{ab.tag:<14}{ab.context:>5}{ab.vocab:>6}{n_par:>11,}"
              f"{1000 * t_tr:>10.1f}{1000 * t_ev:>9.1f}{mem:>8.0f}"
              f"{per_epoch:>9.1f}  {mins}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    total = sum(out.values()) * len(seeds)
    print(f"\nwhole `run` ({len(ABLATIONS)} configurations x {len(seeds)} "
          "seeds, without evaluation overheads): " +
          ", ".join(f"{total * e / 3600:.1f} h at {e} epochs"
                    for e in epochs))
    return out


# ---------------------------------------------------------------------------
# 4. Models, generation and statistics
# ---------------------------------------------------------------------------
def load_model(rec: dict):
    """The generator of a run with its best weights, in eval mode, on CPU."""
    model = build_generator(rec["model"]["args"])
    state = torch.load(config.ROOT / rec["checkpoint"], map_location="cpu",
                       weights_only=True)
    model.load_state_dict(state)
    return model.eval()


def generation_rows(data: gx.GenData, n: int,
                    seed: int = config.SEED) -> tuple[np.ndarray, np.ndarray]:
    """Positions (in data.mv["val"]) of the n validation records whose
    prompts are continued, and of n OTHER validation records used as a
    real-vs-real reference; drawn once with the seed (D19)."""
    n_val = len(data.idx["val"])
    k = min(n, n_val // 2)
    perm = np.random.default_rng(seed).permutation(n_val)
    return np.sort(perm[:k]), np.sort(perm[k:2 * k])


_GENERATED: dict = {}              # (run id, setting, rows) -> signals


def generate_for(model, data: gx.GenData, rows: np.ndarray, setting: str,
                 device, seed: int = config.SEED,
                 run_id: str | None = None) -> np.ndarray:
    """Lead II (mV) generated after the 2 s prompt of each record:
    (len(rows), n_samples - PROMPT_SAMPLES). With a run id the result is
    kept, so the figures reuse the signals of the summary."""
    key = None if run_id is None else (run_id, setting, seed,
                                       tuple(np.asarray(rows).tolist()))
    if key is not None and key in _GENERATED:
        return _GENERATED[key]
    out = _generate(model, data, rows, setting, device, seed)
    if key is not None:
        _GENERATED[key] = out
    return out


def _generate(model, data: gx.GenData, rows: np.ndarray, setting: str,
              device, seed: int) -> np.ndarray:
    _, temperature, top_k = next(s for s in gx.SETTINGS if s[0] == setting)
    tok = data.tokenizer(model.vocab)
    prompt = tok.encode(data.mv["val"][rows, :gx.PROMPT_SAMPLES])
    n_new = data.n_samples - gx.PROMPT_SAMPLES
    seq = gx.generate(model, prompt, n_new, temperature, top_k, seed, device)
    return tok.decode(seq[:, gx.PROMPT_SAMPLES:])


def _free_gpu() -> None:
    """Give the GPU memory of the previous model back before the next."""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def generation_numbers(model, tag: str, settings, data: gx.GenData,
                       rows: np.ndarray, device,
                       run_id: str | None = None) -> dict:
    """Statistics of the signals generated by one model, per setting.
    Undefined values (NaN, e.g. no heart rate found) are left out."""
    prompt = data.mv["val"][rows, :gx.PROMPT_SAMPLES]
    real = data.mv["val"][rows, gx.PROMPT_SAMPLES:]
    out = {}
    for setting in settings:
        t0 = time.perf_counter()
        gen = generate_for(model, data, rows, setting, device, run_id=run_id)
        st = gx.generation_stats(prompt, real, gen)
        out.update({f"gen_{tag}_{setting}_{k}": v for k, v in st.items()
                    if not (isinstance(v, float) and math.isnan(v))})
        print(f"  {tag} {setting}: std ratio {st['std_ratio']:.3f}, flat "
              f"{st['frac_flat']:.3f}, W1 {st['w1_mv']:.4f} mV, HR MAE "
              f"{st['hr_mae']:.1f} bpm ({time.perf_counter() - t0:.0f} s)")
    return out


def baseline_numbers(data: gx.GenData) -> dict:
    """Counting baselines and facts about the tokenisation (D19)."""
    s = {}
    desc = data.description()
    for key in ("range_lo_uv", "range_hi_uv", "span_uv",
                "clip_frac_low_train", "clip_frac_high_train",
                "clip_frac_low_val", "clip_frac_high_val",
                "frac_off_uv_grid_train"):
        s[key] = desc[key]
    s["n_records_train"] = desc["n_records"]["train"]
    s["n_records_val"] = desc["n_records"]["val"]
    s["persistence_rmse_uv"] = gx.persistence_rmse_uv(data.mv["val"])
    s["val_std_uv"] = float(data.mv["val"].std() * 1000.0)
    for v in VOCABS:
        tok = data.tokenizer(v)
        nb = gx.ngram_bits(data.tokens(v, "train"), data.tokens(v, "val"), tok)
        s[f"delta_uv_v{v}"] = int(tok.delta_uv)
        s[f"quant_rmse_uv_v{v}"] = gx.quantisation_rmse_uv(data.mv["val"], tok)
        s[f"quant_rmse_inrange_uv_v{v}"] = gx.quantisation_rmse_uv(
            data.mv["val"], tok, in_range_only=True)
        for k in ("unigram_bits", "bigram_bits", "unigram_ce", "bigram_ce"):
            s[f"{k}_v{v}"] = nb[k]
    s["uniform_bits"] = math.log2(gx.SPAN_UV)
    return s


# ---------------------------------------------------------------------------
# 5. Summary log for make_numbers.py
# ---------------------------------------------------------------------------
def _pred(rec: dict) -> dict:
    p = dx.load_predictions(rec, "val")
    return {"nll": p["record_nll"], "ntok": p["record_ntok"],
            "delta": rec["data"]["tokenizer"]["delta_uv"],
            "ecg_id": p["ecg_id"], "patient_id": p["patient_id"],
            "pos_ce": p["pos_ce"]}


def paired(rec_a: dict, rec_b: dict, n_boot: int) -> dict:
    a, b = _pred(rec_a), _pred(rec_b)
    return gx.paired_bits_diff(a, b, a["patient_id"], n_boot, config.SEED)


def build_summary(paths: Paths = Paths(), fast: bool = False, seeds=SEEDS,
                  data: gx.GenData | None = None,
                  device: torch.device | None = None) -> dict:
    """Every number of the block in one log with a flat `scalars` block
    (alias "generative"). With the data, the baselines and the generation
    statistics are added too."""
    runs = find_runs(paths, fast=fast)
    n_boot = 50 if fast else 1000
    s: dict = {"grad_clip": GRAD_CLIP, "n_seeds": len(seeds)}
    sources = []
    for ab in ABLATIONS:
        recs = [runs[(ab.tag, seed)] for seed in seeds if (ab.tag, seed) in runs]
        if not recs:
            continue
        sources += [r["run_id"] for r in recs]
        t = ab.tag
        sc = [r["scalars"] for r in recs]
        s[f"params_{t}"] = int(sc[0]["n_params"])
        s[f"context_{t}"] = ab.context
        s[f"vocab_{t}"] = ab.vocab
        s[f"delta_uv_{t}"] = int(recs[0]["data"]["tokenizer"]["delta_uv"])
        for key, name in (("val_loss", "ce"), ("val_perplexity", "ppl"),
                          ("val_bits_per_sample", "bits"),
                          ("val_ce_last", "ce_last"), ("val_rmse_uv", "rmse_uv")):
            vals = [x[key] for x in sc]
            s[f"{name}_mean_{t}"] = float(np.mean(vals))
            s[f"{name}_sd_{t}"] = dx._sd(vals)
        s[f"best_epoch_mean_{t}"] = float(np.mean([x["best_epoch"] for x in sc]))
        s[f"epochs_run_mean_{t}"] = float(np.mean([x["epochs_run"] for x in sc]))
        # Median, not mean: an epoch during which the laptop slept would
        # dominate a mean (ctx500 seed 44, D19 entry of 2026-11-22).
        s[f"epoch_time_median_{t}"] = float(np.median(
            [np.median(r["history"]["time_s"]) for r in recs]))
        first = runs.get((t, seeds[0]))
        if first is not None:
            f = first["scalars"]
            s[f"bits_first_{t}"] = f["val_bits_per_sample"]
            s[f"bits_lo_first_{t}"] = f["val_bits_per_sample_lo"]
            s[f"bits_hi_first_{t}"] = f["val_bits_per_sample_hi"]
            pos = _pred(first)["pos_ce"]
            # context of 50 to 99 samples: available to every configuration
            s[f"ce_ctx50_99_first_{t}"] = float(np.mean(pos[49:99]))
        if t == "base" or not ab.selectable:
            continue                  # no_mask: no meaningful paired test
        diffs, better, worse = [], 0, 0
        for seed in seeds:
            ref = runs.get((ab.reference, seed))
            if (t, seed) not in runs or ref is None:
                continue
            print(f"paired difference {t} - {ab.reference}, seed {seed} ...")
            d = paired(runs[(t, seed)], ref, n_boot)
            diffs.append(d["diff"])
            s[f"diff_s{seed}_{t}"] = d["diff"]
            s[f"diff_lo_s{seed}_{t}"] = d["lower"]
            s[f"diff_hi_s{seed}_{t}"] = d["upper"]
            better += d["upper"] < 0          # fewer bits: whole interval < 0
            worse += d["lower"] > 0
        if diffs:
            s[f"diff_mean_{t}"] = float(np.mean(diffs))
            s[f"n_seeds_better_{t}"] = int(better)
            s[f"n_seeds_worse_{t}"] = int(worse)
            s[f"n_seeds_compared_{t}"] = len(diffs)

    selected = other = None
    if all((a.tag, seed) in runs for a in ABLATIONS for seed in seeds):
        selected = select_pretrained(runs, seeds)
        other = comparison_model(selected)

    if data is not None:
        s.update(baseline_numbers(data))
    if data is not None and selected is not None:
        device = device or torch.device("cuda" if torch.cuda.is_available()
                                        else "cpu")
        rows, other_rows = generation_rows(data, FAST_N_GEN if fast else N_GEN)
        s["gen_n_records"] = int(len(rows))
        s["gen_prompt_s"] = gx.PROMPT_SAMPLES / config.SAMPLING_RATE
        s["gen_new_s"] = (data.n_samples - gx.PROMPT_SAMPLES) / config.SAMPLING_RATE
        s.update({k: v for k, v in gx.real_reference_stats(
            data.mv["val"][rows, :gx.PROMPT_SAMPLES],
            data.mv["val"][rows, gx.PROMPT_SAMPLES:],
            data.mv["val"][other_rows, gx.PROMPT_SAMPLES:]).items()
            if not (isinstance(v, float) and math.isnan(v))})
        all_settings = [name for name, _, _ in gx.SETTINGS]
        for ab in ABLATIONS:
            rec = runs[(ab.tag, seeds[0])]
            try:
                model = load_model(rec)
            except FileNotFoundError as e:
                print(f"generation with {ab.tag} skipped: {e}")
                continue
            print(f"generation with {ab.tag} (seed {seeds[0]}) ...")
            settings = all_settings if ab.tag in (selected, other) else ["t10"]
            s.update(generation_numbers(model, ab.tag, settings, data, rows,
                                        device, run_id=rec["run_id"]))
            del model
            _free_gpu()

    handover = None
    if selected is not None:
        handover = {
            "selected_config": selected,
            "model_args": BY_TAG[selected].model_args,
            "checkpoints": {str(seed): runs[(selected, seed)]["checkpoint"]
                            for seed in seeds},
            "run_ids": {str(seed): runs[(selected, seed)]["run_id"]
                        for seed in seeds},
            "tokenizer": runs[(selected, seeds[0])]["data"]["tokenizer"],
            "lead": gx.LEAD,
            "features": "model.record_features(tokens) -> (batch, d): mean "
                        "of the outputs before the head over the windows of "
                        "window_starts(n_samples, context)",
        }
    return {
        "run_id": SUMMARY_NAME + ("_fast" if fast else ""),
        "schema": runlog.SCHEMA_VERSION,
        "task": "generative_block_summary",
        "fast": fast,
        "scalars": s,
        "seeds": list(seeds),
        "selected_config": selected,
        "comparison_config": other,
        "selection_rule": "lowest mean validation bits per sample over the "
                          "seeds, among the selectable configurations "
                          "(all but no_mask)",
        "baseline_args": BASE_ARGS,
        "lstm_args": LSTM_ARGS,
        "grad_clip": GRAD_CLIP,
        "ablations": [{"tag": a.tag, "factor": a.factor, "label": a.label,
                       "model_args": a.model_args, "reference": a.reference,
                       "selectable": a.selectable} for a in ABLATIONS],
        "settings": [{"name": n, "temperature": t, "top_k": k}
                     for n, t, k in gx.SETTINGS],
        "pretrained_for_chat_08": handover,
        "source_runs": sources,
        "meta": runlog.environment(),
    }


# ---------------------------------------------------------------------------
# 6. LaTeX tables (made of macros only: no number is written here)
# ---------------------------------------------------------------------------
def _m(key: str, alias: str = ALIAS) -> str:
    """LaTeX macro of a summary key, e.g. bits_mean_base ->
    \\generativeBitsMeanBase{}."""
    return "\\" + dx.make_numbers_module().macro_name(alias, key) + "{}"


def _tex(text: str) -> str:
    return (text.replace("&", "\\&").replace("uV", "$\\mu$V")
            .replace(" x ", " $\\times$ ").replace("L = ", "$L$ = ")
            .replace("V = ", "$V$ = "))


def ablation_table(summary: dict) -> str:
    s = summary["scalars"]
    lines = [
        "% Generated by src/generative_experiments.py from results/logs/"
        f"{summary['run_id']}.json.",
        "% Needs report/numbers.tex (alias \"generative\" in report/runs.json)"
        " and \\usepackage{booktabs}. Wide: use \\resizebox{\\textwidth}{!}{...}.",
        "% CE: validation cross-entropy per token (nats); it depends on the"
        " vocabulary. Bits: bits per sample, comparable across vocabularies.",
        "% Delta: paired difference in bits with the baseline, same seed;"
        " negative = fewer bits. Last: CE at the last position of the window.",
        "\\begin{tabular}{llrrrccccr}",
        "\\toprule",
        "Factor & Configuration & Parameters & $L$ & $V$ & CE (nats) & "
        "Bits per sample & $\\Delta$ bits & Seeds better / worse & "
        "Last-position CE \\\\",
        "\\midrule",
    ]
    previous = None
    for ab in ABLATIONS:
        t = ab.tag
        if f"bits_mean_{t}" not in s:
            continue
        if previous is not None and ab.factor != previous:
            lines.append("\\addlinespace")
        previous = ab.factor
        ce = f"{_m('ce_mean_' + t)} $\\pm$ {_m('ce_sd_' + t)}"
        bits = f"{_m('bits_mean_' + t)} $\\pm$ {_m('bits_sd_' + t)}"
        if f"diff_mean_{t}" in s:
            diff = _m(f"diff_mean_{t}")
            seeds = (f"{_m('n_seeds_better_' + t)} / {_m('n_seeds_worse_' + t)}"
                     f" of {_m('n_seeds_compared_' + t)}")
        else:
            diff, seeds = "---", "---"
        lines.append(f"{_tex(ab.factor)} & {_tex(ab.label)} & "
                     f"{_m('params_' + t)} & {_m('context_' + t)} & "
                     f"{_m('vocab_' + t)} & {ce} & {bits} & {diff} & {seeds}"
                     f" & {_m('ce_last_mean_' + t)} \\\\")
    if "uniform_bits" in s:
        lines += ["\\midrule",
                  "\\multicolumn{10}{l}{\\emph{Baselines without training "
                  "(bits per sample on the same validation records)}} \\\\",
                  f"\\multicolumn{{6}}{{l}}{{Uniform over the range}} & "
                  f"{_m('uniform_bits')} & & & \\\\"]
        for v in VOCABS:
            if f"bigram_bits_v{v}" not in s:
                continue
            lines.append(
                f"\\multicolumn{{6}}{{l}}{{Unigram / bigram counts, $V$ = {v}}}"
                f" & {_m(f'unigram_bits_v{v}')} / {_m(f'bigram_bits_v{v}')}"
                " & & & \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(lines)


def sampling_table(summary: dict) -> str | None:
    s = summary["scalars"]
    tags = [t for t in (summary.get("selected_config"),
                        summary.get("comparison_config"))
            if t and f"gen_{t}_greedy_std_ratio" in s]
    if not tags:
        return None
    lines = [
        "% Generated by src/generative_experiments.py from results/logs/"
        f"{summary['run_id']}.json.",
        "% 2 s of real lead II as prompt, 8 s generated, compared with the real"
        " 8 s of the same validation records (seed-42 models).",
        "% Std ratio: median std(generated) / std(real). Flat: share of"
        " records whose last 4 s are almost constant.",
        "% W1: Wasserstein distance between amplitude distributions (mV). HR:"
        " heart rate from R peaks (simple threshold rule).",
        "\\begin{tabular}{llccccc}",
        "\\toprule",
        "Model & Sampling & Std ratio & Flat & W1 (mV) & HR found & "
        "HR error (bpm) \\\\",
        "\\midrule",
    ]
    for i, t in enumerate(tags):
        if i:
            lines.append("\\addlinespace")
        model_name = _tex(_short(t))
        for name, _, _ in gx.SETTINGS:
            k = f"gen_{t}_{name}_"
            if k + "std_ratio" not in s:
                continue
            hr_err = _m(k + "hr_mae") if k + "hr_mae" in s else "---"
            lines.append(
                f"{model_name} & {SETTING_TEX[name]} & "
                f"{_m(k + 'std_ratio')} & {_m(k + 'frac_flat')} & "
                f"{_m(k + 'w1_mv')} & {_m(k + 'hr_defined_frac')} & "
                f"{hr_err} \\\\")
    lines += ["\\midrule",
              f"Real signals & --- & --- & --- & {_m('real_w1_other_mv')} & "
              f"{_m('real_hr_defined_frac')} & --- \\\\",
              "\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(lines)


SETTING_TEX = {"greedy": "greedy", "t05": "$T = 0.5$", "t08": "$T = 0.8$",
               "t10": "$T = 1.0$", "t15": "$T = 1.5$",
               "topk10": "top-$k$ 10, $T = 1.0$"}


# ---------------------------------------------------------------------------
# 7. Figures
# ---------------------------------------------------------------------------
def _short(tag: str) -> str:
    return {"base": "ECG-GPT (baseline)", "lstm": "LSTM",
            "ctx100": "context 1 s", "ctx500": "context 5 s",
            "vocab64": "V = 64", "vocab2560": "V = 2560",
            "no_pos": "no positional encoding",
            "no_mask": "no causal mask"}.get(tag, tag)


def figure_curves(runs: dict, paths: Paths, suffix: str,
                  seed: int = SEEDS[0]) -> bool:
    """(a) training and validation cross-entropy of the baseline and the
    LSTM; (b) validation bits per sample per epoch of every configuration
    except no_mask; (c) no_mask: training CE, validation CE over all
    positions, and validation CE at the last position (the only honest
    one)."""
    if any((t, seed) not in runs for t in BY_TAG):
        print("generative_curves: some seed-42 runs are missing; skipped")
        return False
    import matplotlib.pyplot as plt
    eda = dx._style()
    col = eda.SERIES_COLOURS
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))
    for i, tag in enumerate(["base", "lstm"]):
        h = runs[(tag, seed)]["history"]
        axes[0].plot(h["epoch"], h["train_loss"], color=col[i], linewidth=1.5,
                     linestyle="--", label=f"{_short(tag)}: training")
        axes[0].plot(h["epoch"], h["val_loss"], color=col[i], linewidth=2,
                     label=f"{_short(tag)}: validation")
    axes[0].set(title="(a) Cross-entropy per token", xlabel="epoch",
                ylabel="CE (nats)")
    axes[0].legend(fontsize=7)
    others = [a.tag for a in ABLATIONS if a.selectable]
    for i, tag in enumerate(others):
        h = runs[(tag, seed)]["history"]
        axes[1].plot(h["epoch"], h["val_bits_per_sample"],
                     color=col[i % len(col)], linewidth=1.6,
                     linestyle="-" if i < len(col) else ":",
                     label=_short(tag))
    axes[1].set(title="(b) Validation bits per sample", xlabel="epoch",
                ylabel="bits per sample")
    axes[1].legend(fontsize=6)
    h = runs[("no_mask", seed)]["history"]
    axes[2].plot(h["epoch"], h["train_loss"], color=col[0], linewidth=1.5,
                 linestyle="--", label="training (all positions)")
    axes[2].plot(h["epoch"], h["val_loss"], color=col[0], linewidth=2,
                 label="validation (all positions)")
    axes[2].plot(h["epoch"], h["val_ce_last"], color=col[1], linewidth=2,
                 label="validation, last position")
    hb = runs[("base", seed)]["history"]
    axes[2].plot(hb["epoch"], hb["val_ce_last"], color=eda.INK_MUTED,
                 linewidth=1.2, linestyle=":",
                 label="baseline, last position")
    axes[2].set(title="(c) Without the causal mask", xlabel="epoch",
                ylabel="CE (nats)")
    axes[2].legend(fontsize=7)
    fig.suptitle(f"Generators of lead II, seed {seed}", color=eda.INK,
                 fontsize=10)
    dx._save(fig, paths, "generative_curves" + suffix)
    return True


def figure_ablations(runs: dict, summary: dict, paths: Paths, suffix: str,
                     seeds=SEEDS) -> bool:
    """Validation bits per sample of every selectable configuration (dots:
    seeds; bar: mean), with the counting baselines as reference lines."""
    scores = {a.tag: [runs[(a.tag, s)]["scalars"]["val_bits_per_sample"]
                      for s in seeds if (a.tag, s) in runs]
              for a in ABLATIONS if a.selectable}
    scores = {t: v for t, v in scores.items() if v}
    if "base" not in scores:
        print("generative_ablations: no baseline run; figure skipped")
        return False
    import matplotlib.pyplot as plt
    eda = dx._style()
    groups = {"baseline": eda.INK, "architecture": eda.SERIES_COLOURS[0],
              "context length": eda.SERIES_COLOURS[1],
              "vocabulary": eda.SERIES_COLOURS[2],
              "positional encoding": eda.SERIES_COLOURS[3]}
    order = [a for a in ABLATIONS if a.tag in scores]
    fig, ax = plt.subplots(figsize=(7.5, 0.36 * len(order) + 1.4))
    y = np.arange(len(order))[::-1]
    for yi, ab in zip(y, order):
        v = scores[ab.tag]
        c = groups.get(ab.factor, eda.INK)
        ax.plot([np.mean(v)] * 2, [yi - 0.28, yi + 0.28], color=c,
                linewidth=2.5, solid_capstyle="round")
        ax.scatter(v, [yi] * len(v), s=22, color=c, alpha=0.8,
                   edgecolors="white", linewidths=0.8, zorder=3)
    s = summary["scalars"]
    base_mean = float(np.mean(scores["base"]))
    ax.axvline(base_mean, color=eda.INK_MUTED, linewidth=1, linestyle="--")
    ax.text(base_mean, y[0] + 0.7, "baseline mean", color=eda.INK_MUTED,
            fontsize=8, ha="center")
    if "bigram_bits_v256" in s:
        ax.axvline(s["bigram_bits_v256"], color=eda.SERIES_COLOURS[4],
                   linewidth=1, linestyle=":")
        ax.text(s["bigram_bits_v256"], y[-1] - 0.9, "bigram counts (V = 256)",
                color=eda.SERIES_COLOURS[4], fontsize=8, ha="center")
    ax.set_yticks(y, [_short(a.tag) for a in order], fontsize=8)
    ax.set_ylim(y[-1] - 1.3, y[0] + 1.1)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("validation bits per sample (lower is better; dots: seeds "
                  f"{', '.join(map(str, seeds))}; bar: mean)")
    dx._save(fig, paths, "generative_ablations" + suffix)
    return True


def figure_context(runs: dict, summary: dict, paths: Paths, suffix: str,
                   seed: int = SEEDS[0]) -> bool:
    """Bits per sample of each position of the validation windows: position
    t predicts with t + 1 samples of context. Shows how much of the past
    each model uses (one heartbeat is about 0.6-1 s)."""
    tags = [t for t in ("base", "lstm", "ctx100", "ctx500", "no_pos")
            if (t, seed) in runs]
    if "base" not in tags:
        print("generative_context: baseline missing; skipped")
        return False
    import matplotlib.pyplot as plt
    eda = dx._style()
    col = eda.SERIES_COLOURS
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    for i, tag in enumerate(tags):
        p = _pred(runs[(tag, seed)])
        bits = (p["pos_ce"] + math.log(p["delta"])) / math.log(2)
        ctx_s = (np.arange(len(bits)) + 1) / config.SAMPLING_RATE
        ax.plot(ctx_s, bits, color=col[i], linewidth=1.5, label=_short(tag))
    ax.set_xscale("log")
    ax.set(xlabel="context available (s, log scale)",
           ylabel="bits per sample",
           title=f"Prediction cost vs context (validation, seed {seed})")
    ax.legend(fontsize=8)
    dx._save(fig, paths, "generative_context" + suffix)
    return True


def _example_rows(data: gx.GenData, cache: dict | None,
                  seed: int = config.SEED) -> dict:
    """class -> position in data.mv["val"] of one validation record with
    that superclass only, drawn with the seed. Same rule and draws as
    cnn_experiments.example_records, so with the whole validation fold
    these are the example records of D16-D18; with --fast they are drawn
    from the subset."""
    if cache is None:
        return {}
    val = data.idx["val"]
    y = cache["y"][val]
    single = y.sum(axis=1) == 1
    rng = np.random.default_rng(seed)
    out = {}
    for j, c in enumerate(config.CLASSES):
        rows = np.flatnonzero(single & (y[:, j] == 1))
        if len(rows):
            out[c] = int(rng.choice(rows))
    return out


def figure_samples(runs: dict, summary: dict, data: gx.GenData,
                   cache: dict | None, paths: Paths, suffix: str,
                   device) -> bool:
    """One validation record per superclass (rows): the real 2 s prompt
    (black), the real continuation (grey) and the generated continuation
    (colour), for the selected model at T = 1.0, the comparison model at
    T = 1.0 and the selected model with greedy search (columns)."""
    sel, other = summary.get("selected_config"), summary.get("comparison_config")
    examples = _example_rows(data, cache)
    if sel is None or not examples:
        print("generative_samples: selection or examples missing; skipped")
        return False
    import matplotlib.pyplot as plt
    eda = dx._style()
    columns = [(sel, "t10"), (other, "t10"), (sel, "greedy")]
    classes = list(examples)
    rows = np.array([examples[c] for c in classes])
    fig, axes = plt.subplots(len(classes), len(columns),
                             figsize=(11, 1.6 * len(classes) + 0.8),
                             sharex=True, squeeze=False)
    t = np.arange(data.n_samples) / config.SAMPLING_RATE
    p = gx.PROMPT_SAMPLES
    for j, (tag, setting) in enumerate(columns):
        rec = runs[(tag, SEEDS[0])]
        gen = generate_for(load_model(rec), data, rows, setting, device,
                           run_id=rec["run_id"])
        _free_gpu()
        for i, c in enumerate(classes):
            ax = axes[i, j]
            x = data.mv["val"][rows[i]]
            ax.plot(t[:p], x[:p], color=eda.INK, linewidth=0.7)
            ax.plot(t[p:], x[p:], color=eda.INK_MUTED, linewidth=0.6,
                    alpha=0.5)
            ax.plot(t[p:], gen[i], color=eda.SERIES_COLOURS[j], linewidth=0.7)
            ax.axvline(t[p], color=eda.INK_MUTED, linewidth=0.6, linestyle=":")
            ax.set_yticks([])
            ax.grid(False)
            if i == 0:
                ax.set_title(f"{_short(tag)}, {gx.SETTING_LABELS[setting]}",
                             fontsize=8)
            if j == 0:
                ax.set_ylabel(f"{c}\n(ecg_id "
                              f"{int(data.ecg_id['val'][rows[i]])})",
                              fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("time (s)")
    fig.suptitle("Lead II: real prompt (black, 0-2 s), real continuation "
                 "(grey) and generated continuation (colour)",
                 color=eda.INK, fontsize=10)
    dx._save(fig, paths, "generative_samples" + suffix)
    return True


def figure_sampling(runs: dict, summary: dict, data: gx.GenData,
                    paths: Paths, suffix: str, device) -> bool:
    """(a) amplitude distribution of the generated 8 s for each sampling
    setting of the selected model, against the real continuations;
    (b) std ratio and (c) heart-rate error per setting for the selected
    and the comparison model (numbers from the summary)."""
    sel, other = summary.get("selected_config"), summary.get("comparison_config")
    s = summary["scalars"]
    if sel is None or f"gen_{sel}_greedy_std_ratio" not in s:
        print("generative_sampling: generation numbers missing; skipped")
        return False
    import matplotlib.pyplot as plt
    eda = dx._style()
    col = eda.SERIES_COLOURS
    rows, _ = generation_rows(data, int(s["gen_n_records"]))
    real = data.mv["val"][rows, gx.PROMPT_SAMPLES:]
    rec = runs[(sel, SEEDS[0])]
    model = load_model(rec)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    # bin edges on the token grid, so every bin holds the same number of
    # quantisation levels (otherwise coarse vocabularies look like combs)
    tok = data.tokenizer(model.vocab)
    step = tok.delta_uv * max(1, round(
        (np.quantile(real, 0.999) - np.quantile(real, 0.001)) * 1000
        / (80 * tok.delta_uv)))
    k0 = math.floor((np.quantile(real, 0.001) * 1000 - tok.lo_uv) / step)
    k1 = math.ceil((np.quantile(real, 0.999) * 1000 - tok.lo_uv) / step)
    bins = (tok.lo_uv + np.arange(k0, k1 + 1) * step - 0.5) / 1000.0
    axes[0].hist(real.ravel(), bins=bins, density=True, histtype="stepfilled",
                 color=eda.GRID, label="real")
    for i, setting in enumerate(["greedy", "t05", "t10", "t15"]):
        gen = generate_for(model, data, rows, setting, device,
                           run_id=rec["run_id"])
        axes[0].hist(gen.ravel(), bins=bins, density=True, histtype="step",
                     linewidth=1.3, color=col[i],
                     label=gx.SETTING_LABELS[setting])
    axes[0].set_yscale("log")
    axes[0].set(title=f"(a) Amplitudes, {_short(sel)}", xlabel="lead II (mV)",
                ylabel="density (log)")
    axes[0].legend(fontsize=7)
    names = [n for n, _, _ in gx.SETTINGS]
    x = np.arange(len(names))
    for k, (key, title, ylabel) in enumerate(
            [("std_ratio", "(b) Std generated / real", "median ratio"),
             ("hr_mae", "(c) Heart-rate error", "mean |HR gen - HR real| (bpm)")]):
        ax = axes[k + 1]
        for i, tag in enumerate([t for t in (sel, other) if t]):
            vals = [s.get(f"gen_{tag}_{n}_{key}", math.nan) for n in names]
            ax.plot(x, vals, marker="o", color=col[i], linewidth=1.5,
                    label=_short(tag))
        if key == "std_ratio":
            ax.axhline(1.0, color=eda.INK_MUTED, linewidth=1, linestyle="--")
        ax.set_xticks(x, [gx.SETTING_LABELS[n].replace(", T = 1.0", "")
                          for n in names], rotation=30, fontsize=7)
        ax.set(title=title, ylabel=ylabel)
        ax.legend(fontsize=7)
    fig.suptitle(f"Sampling settings ({int(s['gen_n_records'])} validation "
                 "records, 2 s prompt, 8 s generated)", color=eda.INK,
                 fontsize=10)
    dx._save(fig, paths, "generative_sampling" + suffix)
    return True


def figure_attention(runs: dict, summary: dict, data: gx.GenData,
                     cache: dict | None, paths: Paths, suffix: str) -> bool:
    """Causal attention maps (row: predicting position, column: attended
    position, seconds) of one validation window of the NORM example record:
    first and last layer of the Transformer (the selected one, or the
    baseline if the LSTM was selected), heads averaged, each head of the
    last layer, and the lead II of the window."""
    sel = summary.get("selected_config")
    tag = sel if sel and BY_TAG[sel].base == "gpt" else "base"
    examples = _example_rows(data, cache)
    if (tag, SEEDS[0]) not in runs or not examples:
        print("generative_attention: run or examples missing; skipped")
        return False
    c = "NORM" if "NORM" in examples else next(iter(examples))
    row = examples[c]
    model = load_model(runs[(tag, SEEDS[0])])
    L = model.context
    tok = data.tokenizer(model.vocab)
    x = torch.as_tensor(tok.encode(data.mv["val"][row:row + 1, :L]),
                        dtype=torch.long)
    maps = model.attention_maps(x)[0].numpy()        # (layers, heads, L, L)
    import matplotlib.pyplot as plt
    eda = dx._style()
    fig, axes = plt.subplots(2, 4, figsize=(12, 6.4))
    sec = L / config.SAMPLING_RATE
    ext = [0, sec, sec, 0]

    def show(ax, A, title):
        vmax = float(np.quantile(A[np.tril_indices(L)], 0.995))
        im = ax.imshow(A, cmap=eda.SEQUENTIAL, vmin=0, vmax=vmax, extent=ext,
                       interpolation="nearest")
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("attended position (s)", fontsize=7)
        ax.set_ylabel("predicting position (s)", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=6)

    show(axes[0, 0], maps[0].mean(axis=0), "layer 1, heads averaged")
    show(axes[0, 1], maps[-1].mean(axis=0),
         f"layer {maps.shape[0]}, heads averaged")
    axes[0, 2].axis("off")
    t = np.arange(L) / config.SAMPLING_RATE
    axes[0, 3].plot(t, data.mv["val"][row, :L], color=eda.INK, linewidth=0.7)
    axes[0, 3].set(xlim=(0, sec), xlabel="time (s)", ylabel="mV")
    axes[0, 3].set_title(f"lead II, {c} record (ecg_id "
                         f"{int(data.ecg_id['val'][row])})", fontsize=8)
    for k in range(4):
        if k < maps.shape[1]:
            show(axes[1, k], maps[-1, k], f"layer {maps.shape[0]}, head {k + 1}")
        else:
            axes[1, k].axis("off")
    fig.suptitle(f"Causal self-attention of {_short(tag)} (seed {SEEDS[0]}) "
                 "on one validation window", color=eda.INK, fontsize=10)
    dx._save(fig, paths, "generative_attention" + suffix)
    return True


# ---------------------------------------------------------------------------
# 8. report: summary log + tables + figures
# ---------------------------------------------------------------------------
def write_report(fast: bool = False, paths: Paths = Paths(), seeds=SEEDS,
                 cache: dict | None = None,
                 data: gx.GenData | None = None) -> dict:
    suffix = "_fast" if fast else ""
    cache = dx._load_cache(cache)
    data = data or gx.GenData(cache, fast=fast)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summary = build_summary(paths, fast, seeds, data, device)
    path = runlog.write_run(summary, paths.logs)
    print(f"summary: {rel(path)} ({len(summary['scalars'])} numbers)")

    paths.tables.mkdir(parents=True, exist_ok=True)
    (paths.tables / f"generative_ablations{suffix}.tex").write_text(
        ablation_table(summary), encoding="utf-8")
    text = sampling_table(summary)
    if text is not None:
        (paths.tables / f"generative_sampling{suffix}.tex").write_text(
            text, encoding="utf-8")
    print(f"tables: {rel(paths.tables)}/generative_*{suffix}.tex")

    runs = find_runs(paths, fast=fast)
    figure_curves(runs, paths, suffix, seeds[0])
    figure_ablations(runs, summary, paths, suffix, seeds)
    figure_context(runs, summary, paths, suffix, seeds[0])
    figure_samples(runs, summary, data, cache, paths, suffix, device)
    figure_sampling(runs, summary, data, paths, suffix, device)
    figure_attention(runs, summary, data, cache, paths, suffix)
    print(f"figures: {rel(paths.figures)}/generative_*{suffix}.pdf|png")

    if not fast:
        _register_alias(paths)
    h = summary.get("pretrained_for_chat_08")
    if h:
        print(f"\npre-trained model for chat DL 08: {h['selected_config']} "
              f"(checkpoints: {', '.join(h['checkpoints'].values())})")
    return summary


def _register_alias(paths: Paths) -> None:
    """Add "generative": "generative_summary" to report/runs.json."""
    with open(paths.runs_file, encoding="utf-8") as f:
        runs = json.load(f)
    if runs.get(ALIAS) != SUMMARY_NAME:
        runs[ALIAS] = SUMMARY_NAME
        with open(paths.runs_file, "w", encoding="utf-8") as f:
            json.dump(runs, f, indent=2)
            f.write("\n")
        print(f'added "{ALIAS}": "{SUMMARY_NAME}" to {rel(paths.runs_file)}')
    print("next: python report/make_numbers.py")


# ---------------------------------------------------------------------------
def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("step", choices=["estimate", "run", "report", "all"])
    p.add_argument("--fast", action="store_true",
                   help="smoke test: small subset, 2 epochs, outputs *_fast")
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    p.add_argument("--only", nargs="+", default=None,
                   help="run only these configurations (step run)")
    a = p.parse_args(argv)
    t0 = datetime.now()
    cache = dx._load_cache(None)
    data = gx.GenData(cache, fast=a.fast)
    if a.step == "estimate":
        estimate(seeds=a.seeds, fast=a.fast, data=data)
    if a.step in ("run", "all"):
        run_ablations(a.seeds, a.only, a.fast, paths=Paths(), data=data)
    if a.step in ("report", "all"):
        write_report(a.fast, seeds=a.seeds, cache=cache, data=data)
    print(f"\ndone in {datetime.now() - t0}")


if __name__ == "__main__":
    main()
