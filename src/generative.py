"""Generative block (T23, DECISIONS D13 and D19): data, metrics, generation.

    1. Tokenizer       lead II in microvolts -> integers in [0, V)
    2. Data            lead II of the training folds (all records, D12 rule
                       c) and of validation fold 9; random training windows,
                       fixed validation windows. The test fold is never read.
    3. GenerativeTask  next-token cross-entropy (src/tasks.py NextTokenTask)
                       plus bits per sample, CE per position, RMSE, per-record
                       sums for the paired patient bootstrap (D4, D14)
    4. Baselines       uniform, unigram and bigram (counting, no training),
                       persistence
    5. Generation      greedy, temperature and top-k sampling with a sliding
                       window (class notebooks Sampling_in_Text_Generation and
                       TO_DO_MLLB_Keras_RNN_TextGeneration)
    6. Statistics      of generated vs real signals: amplitude, flatness,
                       heart rate from R peaks

Everything here is used by src/generative_experiments.py and tested in
tests/test_generative.py.
"""
import math
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src import config, metrics
from src.models.generative import window_starts
from src.tasks import NextTokenTask

LEAD = "II"
SPAN_UV = 2560                 # tokenised range: 2.56 mV (D19)
CLIP_QUANTILES = (0.001, 0.999)
FAST_SIZES = {"train": 512, "val": 256}   # --fast, as D14


# ---------------------------------------------------------------------------
# 1. Tokenizer
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Tokenizer:
    """Uniform quantiser on the microvolt grid (D19).

    PTB-XL stores every sample as an integer number of microvolts (gain
    1000 per mV), so a value x (mV) is first turned into the integer
    u = round(1000 x). Token k covers the Delta integers
    lo + k Delta, ..., lo + (k + 1) Delta - 1 (microvolts); values below
    lo or above lo + V Delta - 1 go to the first or last token (clipping).
    With Delta = 1 every microvolt value of the range is its own token, so
    the tokenisation loses nothing inside the range.
    """
    lo_uv: int
    delta_uv: int
    vocab: int

    @property
    def span_uv(self) -> int:
        return self.vocab * self.delta_uv

    @property
    def centers_uv(self) -> np.ndarray:
        """Value given back for each token: the middle of its Delta integer
        microvolts, (V,)."""
        return (self.lo_uv + np.arange(self.vocab) * self.delta_uv
                + (self.delta_uv - 1) / 2.0)

    def encode(self, x_mv: np.ndarray) -> np.ndarray:
        """mV -> tokens (int64, same shape)."""
        u = np.rint(np.asarray(x_mv, dtype=np.float64) * 1000.0).astype(np.int64)
        # floor_divide rounds towards -infinity, also for negative values
        k = np.floor_divide(u - self.lo_uv, self.delta_uv)
        return np.clip(k, 0, self.vocab - 1)

    def decode(self, tokens) -> np.ndarray:
        """tokens -> mV (float32): the centre of each token's interval."""
        t = np.asarray(tokens, dtype=np.int64)
        return (self.centers_uv[t] / 1000.0).astype(np.float32)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["span_uv"] = self.span_uv
        return d


def make_tokenizer(lo_uv: int, vocab: int, span_uv: int = SPAN_UV) -> Tokenizer:
    """Tokenizer of `vocab` levels over the fixed span starting at lo_uv.
    The span must be a whole number of microvolts per level."""
    if vocab < 2 or span_uv % vocab:
        raise ValueError(f"span {span_uv} uV is not divisible by vocab {vocab}")
    return Tokenizer(int(lo_uv), span_uv // vocab, int(vocab))


def range_start(train_mv: np.ndarray, span_uv: int = SPAN_UV,
                quantiles=CLIP_QUANTILES) -> int:
    """lo (integer microvolts) of the tokenised range, from TRAINING values
    only: the range of span_uv is centred on the midpoint of their 0.1 % and
    99.9 % quantiles, so both tails are clipped about equally (D19). The
    same lo is used for every vocabulary size."""
    q_lo, q_hi = np.quantile(np.asarray(train_mv, dtype=np.float64).ravel(),
                             quantiles) * 1000.0
    return int(round((q_lo + q_hi) / 2.0 - span_uv / 2.0))


def tokenizer_from_log(record: dict) -> Tokenizer:
    """The tokenizer a run used (stored in its log), e.g. for chat DL 08."""
    t = record["data"]["tokenizer"]
    return Tokenizer(int(t["lo_uv"]), int(t["delta_uv"]), int(t["vocab"]))


# ---------------------------------------------------------------------------
# 2. Data
# ---------------------------------------------------------------------------
def lead_signals(cache: dict, lead: str = LEAD) -> np.ndarray:
    """Lead II of every record of the cache, (N, n_samples) float32 in mV."""
    return np.ascontiguousarray(cache["X"][:, config.LEADS.index(lead), :],
                                dtype=np.float32)


def generative_splits(cache: dict, fast: bool = False,
                      seed: int = config.SEED) -> dict:
    """Cache rows of the two splits of this block (D13, D19):
    train = folds 1-8, every record (also those without a diagnostic
            superclass: labels are not used, D12 rule c);
    val   = fold 9, records with a superclass (the same validation records
            as every classifier and as chat DL 08).
    Fold 10 (test) is never returned. --fast: 512 / 256 rows drawn with the
    seed."""
    fold = cache["fold"]
    idx = {"train": np.flatnonzero(np.isin(fold, config.TRAIN_FOLDS)),
           "val": np.flatnonzero((fold == config.VAL_FOLD)
                                 & cache["has_superclass"])}
    if fast:
        rng = np.random.default_rng(seed)
        idx = {s: np.sort(rng.choice(r, size=min(FAST_SIZES[s], len(r)),
                                     replace=False)) for s, r in idx.items()}
    assert not np.isin(fold[np.concatenate(list(idx.values()))],
                       config.TEST_FOLD).any()
    return idx


class TrainWindows(Dataset):
    """One item per training record: a window of `context` input tokens
    and the same window shifted by one sample as target (teacher forcing).
    The window starts at a random sample of the record, drawn from a
    generator seeded with the run's seed, so every epoch sees every record
    once, each time through a different window (random crops)."""

    def __init__(self, tokens: np.ndarray, context: int, seed: int):
        self.tokens, self.context = tokens, int(context)
        self.n_samples = tokens.shape[1]
        if self.context > self.n_samples - 1:
            raise ValueError("context longer than the record")
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, i: int):
        s = int(self.rng.integers(0, self.n_samples - self.context))
        w = torch.from_numpy(self.tokens[i, s:s + self.context + 1]
                             .astype(np.int64))
        return w[:-1], w[1:]                     # input, next-token target


class ValWindows(Dataset):
    """Fixed windows covering every target of every validation record
    (window_starts), record by record: the same targets are scored at
    every epoch and by every model with the same context."""

    def __init__(self, tokens: np.ndarray, context: int):
        self.tokens, self.context = tokens, int(context)
        self.starts = window_starts(tokens.shape[1], context)

    @property
    def n_windows(self) -> int:
        return len(self.starts)

    def __len__(self) -> int:
        return len(self.tokens) * self.n_windows

    def __getitem__(self, j: int):
        r, k = divmod(j, self.n_windows)
        s = self.starts[k]
        w = torch.from_numpy(self.tokens[r, s:s + self.context + 1]
                             .astype(np.int64))
        return w[:-1], w[1:]


class GenData:
    """Lead II and the two splits, built once per command; tokens are made
    per vocabulary size from the same range (D19)."""

    def __init__(self, cache: dict, fast: bool = False,
                 seed: int = config.SEED):
        self.fast = fast
        self.idx = generative_splits(cache, fast, seed)
        signals = lead_signals(cache)
        self.mv = {s: signals[r] for s, r in self.idx.items()}
        self.patient_id = {s: cache["patient_id"][r] for s, r in self.idx.items()}
        self.ecg_id = {s: cache["ecg_id"][r] for s, r in self.idx.items()}
        self.n_samples = signals.shape[1]
        self.lo_uv = range_start(self.mv["train"])
        self._tokens: dict = {}
        self._description: dict | None = None

    def tokenizer(self, vocab: int) -> Tokenizer:
        return make_tokenizer(self.lo_uv, vocab)

    def tokens(self, vocab: int, split: str) -> np.ndarray:
        """Tokens of one split, int16 (V <= 2560 fits) to keep memory small."""
        key = (vocab, split)
        if key not in self._tokens:
            self._tokens[key] = self.tokenizer(vocab).encode(
                self.mv[split]).astype(np.int16)
        return self._tokens[key]

    def description(self) -> dict:
        """Facts about the data and the range, for the logs (D19)."""
        if self._description is None:
            self._description = self._describe()
        return dict(self._description)

    def _describe(self) -> dict:
        tr, va = self.mv["train"], self.mv["val"]
        hi = self.lo_uv + SPAN_UV - 1
        u_tr = tr.astype(np.float64) * 1000.0
        return {
            "lead": LEAD,
            "n_records": {s: int(len(r)) for s, r in self.idx.items()},
            "n_samples": int(self.n_samples),
            "train_folds": config.TRAIN_FOLDS, "val_fold": config.VAL_FOLD,
            "train_includes_no_superclass": True,
            "range_lo_uv": int(self.lo_uv), "range_hi_uv": int(hi),
            "span_uv": SPAN_UV, "clip_quantiles": list(CLIP_QUANTILES),
            "clip_frac_low_train": float(np.mean(np.rint(u_tr) < self.lo_uv)),
            "clip_frac_high_train": float(np.mean(np.rint(u_tr) > hi)),
            "clip_frac_low_val": float(np.mean(np.rint(va * 1000.0) < self.lo_uv)),
            "clip_frac_high_val": float(np.mean(np.rint(va * 1000.0) > hi)),
            "frac_off_uv_grid_train": float(np.mean(
                np.abs(u_tr - np.rint(u_tr)) > 1e-3)),
            "fast_subset": FAST_SIZES if self.fast else None,
        }


def make_loaders(data: GenData, vocab: int, context: int, batch_size: int,
                 seed: int) -> tuple[dict, dict]:
    """Training and validation DataLoaders of one configuration, and the
    data description for its log. No "test" loader exists."""
    generator = torch.Generator().manual_seed(seed)
    train = TrainWindows(data.tokens(vocab, "train"), context, seed)
    val = ValWindows(data.tokens(vocab, "val"), context)
    loaders = {
        "train": DataLoader(train, batch_size=batch_size, shuffle=True,
                            generator=generator, num_workers=0,
                            pin_memory=torch.cuda.is_available()),
        "val": DataLoader(val, batch_size=batch_size, shuffle=False,
                          num_workers=0, pin_memory=torch.cuda.is_available()),
    }
    info = data.description()
    info.update({"tokenizer": data.tokenizer(vocab).to_dict(),
                 "context": int(context),
                 "train_windows_per_record_per_epoch": 1,
                 "val_windows_per_record": val.n_windows,
                 "val_window_starts": val.starts})
    return loaders, info


# ---------------------------------------------------------------------------
# 3. Task: next-token cross-entropy and the block's metrics
# ---------------------------------------------------------------------------
def bits_per_sample(mean_ce_nats: float, delta_uv: int) -> float:
    """Vocabulary-independent cost of the model (D19): bits per sample to
    code the microvolt value of the next sample, assuming each value of a
    token's interval equally likely: (CE + ln Delta) / ln 2. Uniform
    guessing over the whole range gives log2(span in uV) whatever V is, so
    models with different vocabularies can be compared."""
    return (mean_ce_nats + math.log(delta_uv)) / math.log(2.0)


def record_bits(record_nll: np.ndarray, record_ntok: np.ndarray,
                delta_uv: int) -> float:
    return bits_per_sample(float(record_nll.sum() / record_ntok.sum()),
                           delta_uv)


class GenerativeTask(NextTokenTask):
    """NextTokenTask (D14: cross-entropy per token, perplexity, early
    stopping on the validation cross-entropy) plus, for each split:

    bits_per_sample  vocabulary-independent cost (bits_per_sample())
    ce_last          cross-entropy at the LAST position of the windows, the
                     one position whose target is outside the window: an
                     honest prediction even without the causal mask
    rmse_uv          RMSE (uV) of the expected value sum_k p_k c_k against
                     the centre of the target token
    and, in the predictions file, the cross-entropy at every position of
    the window (how much the context helps) and the summed cross-entropy of
    every record (for the paired patient bootstrap, D4).
    """

    def __init__(self, tokenizer: Tokenizer, context: int, n_windows: int,
                 groups: dict, record_ids: dict, n_boot: int = 1000,
                 seed: int = config.SEED):
        super().__init__(tokenizer.vocab)
        self.tokenizer, self.context = tokenizer, int(context)
        self.n_windows = int(n_windows)
        self.groups, self.record_ids = groups, record_ids
        self.n_boot, self.seed = n_boot, seed
        self._centers = torch.as_tensor(tokenizer.centers_uv,
                                        dtype=torch.float32)

    def start(self):
        L = self.context
        return {"pos_nll": torch.zeros(L, dtype=torch.float64),
                "pos_n": torch.zeros(L, dtype=torch.float64),
                "win_nll": [], "win_n": [], "se": 0.0, "n": 0}

    def collect(self, state, output, target):
        logp = F.log_softmax(output.float(), dim=-1)           # (B, T, V)
        nll = -logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)  # (B, T)
        B, T = target.shape
        state["pos_nll"][:T] += nll.sum(dim=0).double().cpu()
        state["pos_n"][:T] += B
        state["win_nll"].append(nll.sum(dim=1).double().cpu())
        state["win_n"].append(torch.full((B,), T, dtype=torch.float64))
        c = self._centers.to(output.device)
        expected = logp.exp() @ c                                # (B, T)
        state["se"] += float(((expected - c[target]) ** 2).sum())
        state["n"] += target.numel()
        return state

    def epoch_metrics(self, mean_loss, state) -> dict:
        last = self.context - 1
        ce_last = (float(state["pos_nll"][last] / state["pos_n"][last])
                   if state["pos_n"][last] > 0 else math.nan)
        return {"loss": mean_loss,
                "perplexity": metrics.perplexity(mean_loss),
                "bits_per_sample": bits_per_sample(mean_loss,
                                                   self.tokenizer.delta_uv),
                "ce_last": ce_last,
                "rmse_uv": math.sqrt(state["se"] / max(state["n"], 1))}

    def record_sums(self, state) -> tuple[np.ndarray, np.ndarray]:
        """Summed cross-entropy (nats) and number of scored targets of each
        record, in loader order (windows are record by record)."""
        nll = torch.cat(state["win_nll"]).numpy()
        n = torch.cat(state["win_n"]).numpy()
        return (nll.reshape(-1, self.n_windows).sum(axis=1),
                n.reshape(-1, self.n_windows).sum(axis=1))

    def report(self, results: dict):
        details, scalars, arrays = {}, {}, {}
        delta = self.tokenizer.delta_uv
        for split, (mean_loss, state) in results.items():
            m = self.epoch_metrics(mean_loss, state)
            rec_nll, rec_n = self.record_sums(state)
            lo, hi = bits_bootstrap(rec_nll, rec_n, self.groups[split],
                                    delta, self.n_boot, self.seed)
            pos_ce = (state["pos_nll"] / state["pos_n"].clamp_min(1)).numpy()
            details[split] = {**m, "n_records": int(len(rec_nll)),
                              "n_targets": int(rec_n.sum()),
                              "bits_per_sample_ci": [lo, hi],
                              "bootstrap": {"n_boot": self.n_boot,
                                            "seed": self.seed, "alpha": 0.05,
                                            "unit": "patient"}}
            scalars.update({f"{split}_{k}": v for k, v in m.items()})
            scalars[f"{split}_bits_per_sample_lo"] = lo
            scalars[f"{split}_bits_per_sample_hi"] = hi
            arrays[f"{split}_pos_ce"] = pos_ce.astype(np.float64)
            arrays[f"{split}_record_nll"] = rec_nll
            arrays[f"{split}_record_ntok"] = rec_n
            arrays[f"{split}_patient_id"] = np.asarray(self.groups[split])
            arrays[f"{split}_ecg_id"] = np.asarray(self.record_ids[split])
        return details, scalars, arrays


def bits_bootstrap(record_nll: np.ndarray, record_ntok: np.ndarray,
                   groups: np.ndarray, delta_uv: int, n_boot: int = 1000,
                   seed: int = config.SEED, alpha: float = 0.05):
    """Percentile interval of bits per sample, resampling PATIENTS (all
    records of a drawn patient enter together, D14). Returns (lo, hi)."""
    values = [record_bits(record_nll[i], record_ntok[i], delta_uv)
              for i in metrics.patient_bootstrap(groups, n_boot, seed)]
    lo, hi = np.percentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def paired_bits_diff(a: dict, b: dict, groups: np.ndarray,
                     n_boot: int = 1000, seed: int = config.SEED,
                     alpha: float = 0.05) -> dict:
    """bits(A) - bits(B) on the same patient resamples (paired, D4).
    a, b: {"nll", "ntok", "delta", "ecg_id"} of two runs on the same
    validation records. Negative = A codes the signal with fewer bits."""
    if not np.array_equal(a["ecg_id"], b["ecg_id"]):
        raise ValueError("the two runs did not score the same records")
    diffs = [record_bits(a["nll"][i], a["ntok"][i], a["delta"])
             - record_bits(b["nll"][i], b["ntok"][i], b["delta"])
             for i in metrics.patient_bootstrap(groups, n_boot, seed)]
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"diff": record_bits(a["nll"], a["ntok"], a["delta"])
            - record_bits(b["nll"], b["ntok"], b["delta"]),
            "lower": float(lo), "upper": float(hi), "n_boot": n_boot,
            "seed": seed}


# ---------------------------------------------------------------------------
# 4. Baselines without training
# ---------------------------------------------------------------------------
def ngram_bits(train_tokens: np.ndarray, val_tokens: np.ndarray,
               tokenizer: Tokenizer, alpha: float = 1.0) -> dict:
    """Unigram and bigram models made by counting on the training tokens,
    with add-alpha (Laplace) smoothing, scored on every validation target
    1..n-1 (RNN slides 93-94: a 2-gram language model). Bits per sample as
    for the networks; uniform over the range = log2(span)."""
    V = tokenizer.vocab
    tr = train_tokens.astype(np.int64)
    va = val_tokens.astype(np.int64)
    uni = np.bincount(tr.ravel(), minlength=V).astype(np.float64) + alpha
    log_uni = np.log(uni / uni.sum())
    ce_uni = float(-log_uni[va[:, 1:]].mean())
    pairs = np.bincount((tr[:, :-1] * V + tr[:, 1:]).ravel(),
                        minlength=V * V).astype(np.float64).reshape(V, V)
    pairs += alpha
    log_bi = np.log(pairs / pairs.sum(axis=1, keepdims=True))
    ce_bi = float(-log_bi[va[:, :-1], va[:, 1:]].mean())
    d = tokenizer.delta_uv
    return {"uniform_bits": math.log2(tokenizer.span_uv),
            "unigram_ce": ce_uni, "unigram_bits": bits_per_sample(ce_uni, d),
            "bigram_ce": ce_bi, "bigram_bits": bits_per_sample(ce_bi, d)}


def persistence_rmse_uv(val_mv: np.ndarray) -> float:
    """RMSE (uV) of the naive prediction "next sample = this sample"."""
    u = val_mv.astype(np.float64) * 1000.0
    return float(np.sqrt(np.mean((u[:, 1:] - u[:, :-1]) ** 2)))


def quantisation_rmse_uv(val_mv: np.ndarray, tokenizer: Tokenizer,
                         in_range_only: bool = False) -> float:
    """RMSE (uV) between the signal and decode(encode(signal)). By default
    clipped samples are included (their error can be large); with
    in_range_only only the samples inside the range count, whose error is
    at most (Delta - 1) / 2 uV."""
    x = np.asarray(val_mv, dtype=np.float64)
    err = tokenizer.decode(tokenizer.encode(x)).astype(np.float64) - x
    if in_range_only:
        u = np.rint(x * 1000.0)
        err = err[(u >= tokenizer.lo_uv)
                  & (u <= tokenizer.lo_uv + tokenizer.span_uv - 1)]
    return float(np.sqrt(np.mean(err ** 2)) * 1000.0)


# ---------------------------------------------------------------------------
# 5. Generation
# ---------------------------------------------------------------------------
# (name, temperature, top_k); temperature 0 = greedy (D19)
SETTINGS = [("greedy", 0.0, None), ("t05", 0.5, None), ("t08", 0.8, None),
            ("t10", 1.0, None), ("t15", 1.5, None), ("topk10", 1.0, 10)]
SETTING_LABELS = {"greedy": "greedy", "t05": "T = 0.5", "t08": "T = 0.8",
                  "t10": "T = 1.0", "t15": "T = 1.5",
                  "topk10": "top-k 10, T = 1.0"}
PROMPT_SAMPLES = 200           # 2 s of real signal
GEN_BATCH = 64


def sample_next(logits: torch.Tensor, temperature: float,
                top_k: int | None = None,
                generator: torch.Generator | None = None) -> torch.Tensor:
    """Choose the next token of each sequence from its logits (batch, V)
    (Sampling_in_Text_Generation.ipynb, parts 2-4):
      temperature 0: greedy search, the most probable token;
      otherwise the logits are divided by the temperature before the
      softmax (low T: sharper, more repetitive; high T: flatter, more
      random) and with top_k only the k most probable tokens are kept and
      their probabilities renormalised; then one token is drawn."""
    if temperature <= 0:
        return logits.argmax(dim=-1)
    z = logits.float() / temperature
    if top_k is not None:
        kth = torch.topk(z, min(top_k, z.shape[-1]), dim=-1).values[:, -1:]
        z = z.masked_fill(z < kth, float("-inf"))
    p = torch.softmax(z, dim=-1)
    return torch.multinomial(p, 1, generator=generator).squeeze(-1)


@torch.no_grad()
def generate(model, prompt: np.ndarray, n_new: int, temperature: float,
             top_k: int | None = None, seed: int = config.SEED,
             device: torch.device | None = None,
             batch: int = GEN_BATCH) -> np.ndarray:
    """Autoregressive generation (TO_DO_MLLB_Keras_RNN_TextGeneration,
    "make predictions"): the next token is predicted from the last
    `context` tokens, appended, and the window slides by one, n_new times.
    Both generators use the same sliding window (D19). prompt: (n, P)
    tokens; returns (n, P + n_new) int64."""
    device = device or torch.device("cpu")
    model = model.to(device).eval()
    # The attention matrices grow with the square of the window: longer
    # contexts generate fewer records at a time (same memory as L = 250).
    batch = max(1, int(batch * min(1.0, (250 / model.context) ** 2)))
    gen = torch.Generator(device=device).manual_seed(seed)
    out = []
    for i in range(0, len(prompt), batch):
        seq = torch.as_tensor(prompt[i:i + batch], dtype=torch.long,
                              device=device)
        for _ in range(n_new):
            logits = model.next_logits(seq[:, -model.context:])
            nxt = sample_next(logits, temperature, top_k, gen)
            seq = torch.cat([seq, nxt.unsqueeze(1)], dim=1)
        out.append(seq.cpu().numpy())
    return np.concatenate(out)


# ---------------------------------------------------------------------------
# 6. Statistics of generated signals
# ---------------------------------------------------------------------------
MIN_RR_S = 0.3                 # R peaks at least 0.3 s apart (<= 200 bpm)


def peak_threshold(prompt_mv: np.ndarray) -> np.ndarray:
    """Per-record R-peak threshold, from the REAL prompt only (so it is not
    tuned to the generated signal): median + half the distance from the
    median to the maximum, (n,)."""
    med = np.median(prompt_mv, axis=1)
    return med + 0.5 * (prompt_mv.max(axis=1) - med)


def heart_rate(x_mv: np.ndarray, threshold: float,
               fs: int = config.SAMPLING_RATE) -> float:
    """Heart rate (bpm) = 60 / median RR interval, with R peaks = local
    maxima above `threshold` at least MIN_RR_S apart (scipy find_peaks).
    NaN with fewer than 2 peaks. A simple rule, not a validated QRS
    detector (D19)."""
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(x_mv, height=threshold,
                          distance=max(1, int(round(MIN_RR_S * fs))))
    if len(peaks) < 2:
        return math.nan
    return float(60.0 * fs / np.median(np.diff(peaks)))


def generation_stats(prompt_mv: np.ndarray, real_mv: np.ndarray,
                     gen_mv: np.ndarray) -> dict:
    """Generated continuation vs the real continuation of the same records
    (all in mV, (n, M)):
      std_ratio     median over records of std(generated) / std(real)
      frac_flat     share of records whose generated last 4 s have less
                    than 10 % of the real continuation's std ("stuck")
      w1_mv         Wasserstein-1 distance between the pooled amplitude
                    distributions (mV)
      hr_*          heart rate from R peaks (threshold from the real prompt)
    """
    from scipy.stats import wasserstein_distance
    std_real = real_mv.std(axis=1)
    std_gen = gen_mv.std(axis=1)
    ok = std_real > 0
    tail = gen_mv[:, -min(400, gen_mv.shape[1]):].std(axis=1)
    thr = peak_threshold(prompt_mv)
    hr_real = np.array([heart_rate(r, t) for r, t in zip(real_mv, thr)])
    hr_gen = np.array([heart_rate(g, t) for g, t in zip(gen_mv, thr)])
    both = ~np.isnan(hr_real) & ~np.isnan(hr_gen)
    return {
        "std_ratio": float(np.median(std_gen[ok] / std_real[ok])),
        "frac_flat": float(np.mean(tail[ok] < 0.1 * std_real[ok])),
        "w1_mv": float(wasserstein_distance(gen_mv.ravel(), real_mv.ravel())),
        "hr_defined_frac": float(np.mean(~np.isnan(hr_gen))),
        "hr_mean": float(np.nanmean(hr_gen)) if (~np.isnan(hr_gen)).any()
        else math.nan,
        "hr_mae": float(np.mean(np.abs(hr_gen[both] - hr_real[both])))
        if both.any() else math.nan,
        "hr_n_compared": int(both.sum()),
    }


def real_reference_stats(prompt_mv: np.ndarray, real_mv: np.ndarray,
                         other_real_mv: np.ndarray) -> dict:
    """Reference values from real signals only: heart rate of the real
    continuations, their std, and the Wasserstein distance between the
    real continuations and those of OTHER validation records (how far two
    sets of real ECGs are from each other)."""
    from scipy.stats import wasserstein_distance
    thr = peak_threshold(prompt_mv)
    hr = np.array([heart_rate(r, t) for r, t in zip(real_mv, thr)])
    return {"real_hr_defined_frac": float(np.mean(~np.isnan(hr))),
            "real_hr_mean": float(np.nanmean(hr)),
            "real_std_mv": float(np.median(real_mv.std(axis=1))),
            "real_w1_other_mv": float(wasserstein_distance(
                other_real_mv.ravel(), real_mv.ravel()))}
