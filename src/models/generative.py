"""Autoregressive generators of lead II (T23, DECISIONS D13 and D19).

Two next-token models with the same input and output, so that the
comparison "recurrence vs causal self-attention" is fair (D13):

    ECGGPT   decoder-only (causal) Transformer: the ECG-GPT
    ECGLSTM  2-layer LSTM language model

Both read a window of T <= L tokens (L = `context`) and return, at every
position t, logits over the V tokens for the token at t + 1 (next-token
prediction with teacher forcing, RNN slides 88-90; Part II slides 11-14):

    tokens (batch, T)                  integers in [0, V): quantised lead II
      -> token embedding (batch, T, d)   nn.Embedding(V, d): one learned
                                         vector per quantisation level, as one
                                         vector per character in the class
                                         text-generation notebooks
      GPT:  + sinusoidal positional encoding (position in the window)
            -> num_layers x TransformerBlock(causal=True)
                                         src/models/attention.py (chat DL 07)
            -> final LayerNorm (Pre-LN)
      LSTM: -> nn.LSTM (2 layers) -> dropout
      -> Linear(d, V) -> logits (batch, T, V)

No softmax in the models: CrossEntropyLoss contains it (src/tasks.py,
NextTokenTask), and sampling applies its own softmax with a temperature
(src/generative.py).

These classes are not in src/models/MODELS: that registry feeds
`python -m src.train`, which trains the 5-superclass classifiers on
(batch, 12, 1000) signals. The generators are built by
src/generative_experiments.py with build_generator().
"""
import torch
from torch import nn

from src.models.attention import TransformerBlock, sinusoidal_encoding

ARCHS = ("gpt", "lstm")
POS_ENCODINGS = ("sinusoidal", "none")


def window_starts(n_samples: int, context: int) -> list[int]:
    """Start of each window of `context` inputs (+ 1 target) that together
    cover every target 1 .. n_samples - 1 of a record: ceil((n - 1) / L)
    windows, evenly spaced, the first at 0 and the last ending at the last
    sample (D19). With n = 1000: L = 250 -> 0, 250, 499, 749."""
    if not 1 <= context <= n_samples - 1:
        raise ValueError(f"context must be in [1, {n_samples - 1}]")
    n_win = -(-(n_samples - 1) // context)               # ceiling division
    last = n_samples - 1 - context
    if n_win == 1:
        return [0]
    return [round(i * last / (n_win - 1)) for i in range(n_win)]


class _Generator(nn.Module):
    """What both generators share: the next-token logits used for sampling
    and the record-level features used by chat DL 08 (transfer, D7)."""
    arch = "?"
    context: int
    vocab: int

    def features(self, tokens: torch.Tensor) -> torch.Tensor:
        """(batch, T) tokens -> (batch, T, d) representations before the
        next-token head."""
        raise NotImplementedError

    def _head_input(self, h: torch.Tensor) -> torch.Tensor:
        return h

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """(batch, T) -> (batch, T, V) logits; position t predicts t + 1."""
        return self.head(self._head_input(self.features(tokens)))

    def next_logits(self, window: torch.Tensor) -> torch.Tensor:
        """Logits of the token that follows the window, (batch, V). Only the
        last position goes through the head: during generation the other
        positions are not needed."""
        return self.head(self._head_input(self.features(window)[:, -1]))

    def record_features(self, tokens: torch.Tensor) -> torch.Tensor:
        """One vector per record for the 5-superclass task (chat DL 08):
        the record (batch, n_samples) is cut into the windows of
        window_starts(), each window goes through the network without its
        head, and the outputs are averaged over positions and windows
        (mean readout, as D17 and D18). Gradients flow, so the same call
        serves the frozen model (linear probe) and fine-tuning."""
        n = tokens.shape[1]
        starts = window_starts(n, self.context) if n > self.context else [0]
        T = min(n, self.context)
        windows = torch.stack([tokens[:, s:s + T] for s in starts], dim=1)
        b, w = windows.shape[:2]
        h = self.features(windows.reshape(b * w, T))           # (b*w, T, d)
        return h.mean(dim=1).reshape(b, w, -1).mean(dim=1)     # (b, d)


class ECGGPT(_Generator):
    """Decoder-only Transformer over lead II tokens (the ECG-GPT, D19).

    The blocks are chat DL 07's TransformerBlock with causal=True: the
    attention scores of every future position are set to -inf before the
    softmax, so position t only sees tokens 0..t and the whole window is
    trained in one parallel pass (Part II slides 13-14). causal=False is
    the `no_mask` ablation: the network can then read the token it must
    predict.

    Positions: sinusoidal encoding of the position IN THE WINDOW (0..L-1),
    added to the token embeddings (Part II slides 18-23). Training windows
    start anywhere in the record and generation slides the window, so the
    position never means absolute time. pos_encoding="none" is the
    `no_pos` ablation.
    """
    arch = "gpt"

    def __init__(self, vocab: int = 256, context: int = 250,
                 d_model: int = 128, num_heads: int = 4, num_layers: int = 2,
                 d_ff: int | None = None, dropout: float = 0.1,
                 pos_encoding: str = "sinusoidal", causal: bool = True):
        super().__init__()
        if vocab < 2 or context < 1 or num_layers < 1:
            raise ValueError("vocab >= 2, context >= 1, num_layers >= 1")
        if pos_encoding not in POS_ENCODINGS:
            raise ValueError(f"pos_encoding must be one of {POS_ENCODINGS}")
        self.vocab, self.context = int(vocab), int(context)
        self.d_model, self.causal = d_model, bool(causal)
        self.pos_encoding = pos_encoding
        d_ff = 4 * d_model if d_ff is None else d_ff
        self.tok_emb = nn.Embedding(vocab, d_model)
        if pos_encoding == "sinusoidal":
            # Buffer: follows the model to the GPU, is not trained and is
            # left out of the checkpoint (persistent=False); rebuilt here.
            self.register_buffer("pe", sinusoidal_encoding(context, d_model),
                                 persistent=False)
        else:
            self.pe = None
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout, norm="pre",
                             causal=self.causal)
            for _ in range(num_layers)])
        self.ln_final = nn.LayerNorm(d_model)   # Pre-LN: one more LayerNorm
        self.head = nn.Linear(d_model, vocab)   # logits over the vocabulary

    def _embed(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.dim() != 2:
            raise ValueError("tokens must be (batch, T)")
        T = tokens.shape[1]
        if T > self.context:
            raise ValueError(f"window of {T} tokens > context {self.context}")
        h = self.tok_emb(tokens)                         # (batch, T, d)
        if self.pe is not None:
            h = h + self.pe[:T]                          # broadcast over batch
        return h

    def features(self, tokens: torch.Tensor) -> torch.Tensor:
        h = self._embed(tokens)
        for block in self.blocks:
            h, _ = block(h)
        return self.ln_final(h)

    @torch.no_grad()
    def attention_maps(self, tokens: torch.Tensor) -> torch.Tensor:
        """Attention weights of every block, (batch, layers, heads, T, T);
        row = query (predicting) position, column = key position. With the
        causal mask everything above the diagonal is exactly 0."""
        h = self._embed(tokens)
        maps = []
        for block in self.blocks:
            h, w = block(h)
            maps.append(w)
        return torch.stack(maps, dim=1)


class ECGLSTM(_Generator):
    """LSTM language model over lead II tokens (D19).

    nn.LSTM(batch_first=True) reads (batch, T, features); at step t its
    state has seen tokens 0..t only, so it is causal by construction: no
    mask is needed. Hidden size 160 gives about the same number of
    parameters as the ECG-GPT baseline (D19). Dropout on the outputs of the
    top layer, before the head; PyTorch's `dropout` argument of nn.LSTM
    (between layers) stays 0, as in D17.

    `context` is the window used in training and generation; the LSTM
    itself could read longer sequences.
    """
    arch = "lstm"

    def __init__(self, vocab: int = 256, context: int = 250,
                 d_embed: int = 128, hidden_size: int = 160,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        if vocab < 2 or context < 1 or num_layers < 1 or hidden_size < 1:
            raise ValueError("invalid LSTM generator arguments")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.vocab, self.context = int(vocab), int(context)
        self.hidden_size = int(hidden_size)
        self.tok_emb = nn.Embedding(vocab, d_embed)
        self.rnn = nn.LSTM(d_embed, hidden_size, num_layers=num_layers,
                           batch_first=True)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Linear(hidden_size, vocab)

    def features(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.dim() != 2:
            raise ValueError("tokens must be (batch, T)")
        out, _ = self.rnn(self.tok_emb(tokens))           # (batch, T, hidden)
        return out

    def _head_input(self, h: torch.Tensor) -> torch.Tensor:
        return self.drop(h)


def build_generator(model_args: dict) -> _Generator:
    """{"arch": "gpt" | "lstm", ...constructor arguments} -> model."""
    args = dict(model_args)
    arch = args.pop("arch", "gpt")
    if arch not in ARCHS:
        raise ValueError(f"arch must be one of {ARCHS}")
    return (ECGGPT if arch == "gpt" else ECGLSTM)(**args)
