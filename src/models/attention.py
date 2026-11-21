"""Self-attention building blocks, written by hand (T12, DECISIONS D18).

Shared by the Transformer classifier of chat DL 07 (src/models/transformer.py)
and the decoder-only generator of chat DL 13 (ECG-GPT, D13), which uses the
same blocks with causal=True.

They follow the class notebooks Transformer_Architecture_Intro 01-05:

    01  scaled dot-product attention, one head           -> attention()
    02  several heads, output projection W_O              -> MultiHeadSelfAttention
    03  feed-forward network, skip connections            -> FeedForward, TransformerBlock
    05  LayerNorm before each sub-layer (Pre-LN)          -> TransformerBlock(norm="pre")

Shapes (batch_first, as in the notebooks): a sequence is (batch, T, d_model);
inside the attention each head is (batch, heads, T, head_dim).

nn.MultiheadAttention is not used, so that every step can be read and
explained; tests/test_transformer.py checks that MultiHeadSelfAttention gives
the same output as nn.MultiheadAttention with the same weights.
"""
import math

import torch
from torch import nn

NORMS = ("pre", "post", "none")


def causal_mask(T: int, device=None) -> torch.Tensor:
    """(T, T) boolean mask, True where query i may look at key j, i.e. j <= i.

    A decoder that predicts the next token must not see the future (Part II
    slides 11-14). With the mask, the whole sequence is processed in one
    pass and every position still only uses the positions before it, which
    is what makes training parallel (unlike an RNN, slide 13)."""
    return torch.ones(T, T, dtype=torch.bool, device=device).tril()


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
              mask: torch.Tensor | None = None):
    """Scaled dot-product attention (notebook 01, cell 5; Part I slides
    36-41).

    q, k: (..., T, d_k); v: (..., T, d_v); mask: broadcastable to (..., T, T),
    True = allowed.

    1. scores = q k^T: one dot product per (query, key) pair, (..., T, T)
    2. / sqrt(d_k): without it the dot products grow with d_k and push the
       softmax into regions with tiny gradients (Part I slide 49)
    3. masked scores -> -inf, so their weight after the softmax is exactly 0
    4. softmax over the KEYS (last axis): each query's weights are >= 0 and
       sum to 1
    5. output = weights v: each query receives a weighted mean of the values

    Returns (output (..., T, d_v), weights (..., T, T)).
    """
    d_k = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return weights @ v, weights


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention (notebook 02; Part I slides 51-58).

    The same sequence x gives the queries, keys and values (self-attention:
    Q = x W_Q, K = x W_K, V = x W_V). d_model is split into `num_heads`
    heads of head_dim = d_model / num_heads; each head attends on its own
    ("representation subspaces", slide 51); the heads are concatenated back
    to d_model and mixed by W_O.

    out_proj=False leaves W_O out, as notebook 01's single-head layer. The
    projections have no bias, as in the notebooks.
    causal=True applies causal_mask() in every forward pass (chat DL 13).
    """

    def __init__(self, d_model: int, num_heads: int, out_proj: bool = True,
                 causal: bool = False):
        super().__init__()
        if num_heads < 1 or d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model, self.num_heads = d_model, num_heads
        self.head_dim = d_model // num_heads
        self.causal = causal
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False) if out_proj \
            else nn.Identity()

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, T, d_model) -> (batch, heads, T, head_dim): view() cuts
        the last axis into heads x head_dim, permute() moves the heads
        before time (slides 57-58)."""
        b, t, _ = x.shape
        return x.view(b, t, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

    def _combine(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, heads, T, head_dim) -> (batch, T, d_model). permute()
        makes the tensor non-contiguous, so reshape() (not view()) is
        needed."""
        b, h, t, d = x.shape
        return x.permute(0, 2, 1, 3).reshape(b, t, h * d)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        """x: (batch, T, d_model). mask: optional (T, T) or broadcastable,
        True = allowed; combined with the causal mask if causal=True.
        Returns (output (batch, T, d_model), weights (batch, heads, T, T))."""
        if self.causal:
            cm = causal_mask(x.size(1), x.device)
            mask = cm if mask is None else (mask & cm)
        q, k, v = (self._split(W(x)) for W in (self.W_Q, self.W_K, self.W_V))
        out, weights = attention(q, k, v, mask)
        return self.W_O(self._combine(out)), weights


class FeedForward(nn.Module):
    """Position-wise feed-forward network (notebooks 03 and 05): the same
    2-layer MLP applied to every token separately. Attention mixes
    information BETWEEN tokens; the FFN transforms each token, expanding it
    to d_ff (4 x d_model) and back (Part I slides 27, 70-78)."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.W1 = nn.Linear(d_model, d_ff)
        self.W2 = nn.Linear(d_ff, d_model)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.W2(self.act(self.W1(x)))


class TransformerBlock(nn.Module):
    """One encoder block: self-attention and FFN, each with a skip
    connection ("Add") and LayerNorm ("Norm").

    norm="pre" (notebook 05; Part II slides 35-36):
        x = x + Dropout(MHSA(LN1(x)))
        x = x + Dropout(FFN(LN2(x)))
    norm="post" (original Transformer, class text notebook):
        x = LN1(x + Dropout(MHSA(x)))
        x = LN2(x + Dropout(FFN(x)))
    norm="none": no LayerNorm.

    residual=False drops the skip connections (x = Dropout(MHSA(x)), ...)
    and ffn=False drops the FFN sub-layer: together with out_proj and
    num_heads they rebuild the step-by-step study of D18.

    The residual stream keeps d_model dimensions throughout, so the block
    can be stacked; the attention weights are returned for the figures.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int,
                 dropout: float = 0.1, norm: str = "pre", ffn: bool = True,
                 residual: bool = True, out_proj: bool = True,
                 causal: bool = False):
        super().__init__()
        if norm not in NORMS:
            raise ValueError(f"norm must be one of {NORMS}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.norm, self.residual = norm, residual
        self.attn = MultiHeadSelfAttention(d_model, num_heads, out_proj,
                                           causal)
        self.ffn = FeedForward(d_model, d_ff) if ffn else None
        use_ln = norm != "none"
        self.ln1 = nn.LayerNorm(d_model) if use_ln else nn.Identity()
        self.ln2 = nn.LayerNorm(d_model) if (use_ln and ffn) else nn.Identity()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def _sublayer(self, x, f, ln):
        """Apply one sub-layer f with its skip connection and LayerNorm."""
        if self.norm == "pre":
            out = f(ln(x))
            return (x + self.drop(out[0]) if self.residual
                    else self.drop(out[0])), out[1]
        out = f(x)
        y = x + self.drop(out[0]) if self.residual else self.drop(out[0])
        return ln(y), out[1]          # post: LN after the addition

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        """x: (batch, T, d_model) -> (same shape, weights (batch, heads, T, T))."""
        x, weights = self._sublayer(x, lambda z: self.attn(z, mask), self.ln1)
        if self.ffn is not None:
            x, _ = self._sublayer(x, lambda z: (self.ffn(z), None), self.ln2)
        return x, weights


def sinusoidal_encoding(T: int, d_model: int) -> torch.Tensor:
    """(T, d_model) sinusoidal positional encoding (Part II slides 19-23):

        PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

    Each pair of dimensions is a sine and cosine of one frequency; low
    dimensions change fast with the position, high ones slowly."""
    if d_model % 2:
        raise ValueError("d_model must be even for the sinusoidal encoding")
    pos = torch.arange(T, dtype=torch.float32).unsqueeze(1)          # (T, 1)
    freq = torch.exp(-math.log(10000.0)
                     * torch.arange(0, d_model, 2, dtype=torch.float32)
                     / d_model)                                      # (d/2,)
    pe = torch.zeros(T, d_model)
    pe[:, 0::2] = torch.sin(pos * freq)
    pe[:, 1::2] = torch.cos(pos * freq)
    return pe
