"""Transformer encoder for the 5 superclasses (T12, DECISIONS D18).

    signal (batch, 12, 1000)
      -> patches (batch, T, 12 * patch)      T = 1000 / patch tokens; each
                                             token is 12 leads x `patch`
                                             samples, flattened lead by lead
                                             (as the RNN frames, D17)
      -> patch embedding (batch, T, d)       Linear(12 * patch, d_model)
      -> [CLS token] + positional encoding   order of the patches
      -> L x TransformerBlock                self-attention + FFN
                                             (src/models/attention.py)
      -> final LayerNorm (Pre-LN only)
      -> one vector per record               mean over the tokens, or the
                                             output of the CLS token
      -> dropout -> Linear -> 5 logits       same head as the CNN and RNN

Every record has 1000 samples, so all sequences have the same length and no
padding mask is needed. The output is made of logits: no sigmoid (D2).
"""
import torch
from torch import nn

from src.models.attention import TransformerBlock, sinusoidal_encoding

POS_ENCODINGS = ("sinusoidal", "learned", "none")
READOUTS = ("mean", "cls")


class TransformerClassifier(nn.Module):
    def __init__(self, n_leads: int = 12, n_samples: int = 1000,
                 n_classes: int = 5, patch: int = 10, d_model: int = 128,
                 num_heads: int = 4, num_layers: int = 2,
                 d_ff: int | None = None, ffn: bool = True,
                 residual: bool = True, norm: str = "pre",
                 out_proj: bool = True, pos_encoding: str = "sinusoidal",
                 readout: str = "mean", dropout: float = 0.1,
                 head_dropout: float = 0.5):
        super().__init__()
        if patch < 1 or n_samples % patch:
            raise ValueError(f"patch must divide n_samples={n_samples}")
        if pos_encoding not in POS_ENCODINGS:
            raise ValueError(f"pos_encoding must be one of {POS_ENCODINGS}")
        if readout not in READOUTS:
            raise ValueError(f"readout must be one of {READOUTS}")
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        if not 0.0 <= head_dropout < 1.0:
            raise ValueError("head_dropout must be in [0, 1)")
        self.patch = int(patch)
        self.n_steps = n_samples // patch           # number of patch tokens
        self.step_samples = self.patch              # samples per token
        self.n_prefix = 1 if readout == "cls" else 0
        self.readout, self.norm = readout, norm
        self.pos_encoding = pos_encoding
        self.d_model = d_model
        n_tokens = self.n_steps + self.n_prefix
        d_ff = 4 * d_model if d_ff is None else d_ff

        # Patch embedding: one Linear layer applied to every patch, i.e. a
        # Conv1d with kernel = stride = patch (tested).
        self.embed = nn.Linear(n_leads * patch, d_model)
        # CLS token: one learned vector, the same for every record, put in
        # front of the patches; after the encoder its output summarises the
        # record (BERT, Part I slides 11-12). Starts at zero (D18).
        self.cls_token = (nn.Parameter(torch.zeros(1, 1, d_model))
                          if readout == "cls" else None)
        if pos_encoding == "sinusoidal":
            # A buffer: moved to the GPU with the model, but not a parameter,
            # so the optimiser does not change it. persistent=False keeps it
            # out of the state_dict (checkpoint); it is rebuilt here.
            self.register_buffer("pe", sinusoidal_encoding(n_tokens, d_model),
                                 persistent=False)
            self.pos_embedding = None
        elif pos_encoding == "learned":
            # One learned vector per position, as the class text notebook's
            # TokenAndPositionEmbedding (nn.Embedding, default init N(0, 1)).
            self.pe = None
            self.pos_embedding = nn.Embedding(n_tokens, d_model)
        else:
            self.pe = None
            self.pos_embedding = None

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout, norm, ffn,
                             residual, out_proj)
            for _ in range(num_layers)])
        # Pre-LN leaves the residual stream un-normalised after the last
        # block, so one more LayerNorm is applied (notebook 05, ln_final).
        self.ln_final = nn.LayerNorm(d_model) if norm == "pre" \
            else nn.Identity()
        self.dropout = nn.Dropout(head_dropout) if head_dropout > 0 \
            else nn.Identity()
        self.classifier = nn.Linear(d_model, n_classes)          # logits

    # ------------------------------------------------------------------
    def tokens(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, leads, samples) -> (batch, T, leads * patch): token t
        holds samples [t * patch, (t + 1) * patch) of lead 0, then of lead
        1, ... (the same layout as the RNN frames)."""
        b, c, n = x.shape
        t = n // self.patch
        x = x.reshape(b, c, t, self.patch).permute(0, 2, 1, 3)  # (B, T, C, P)
        return x.reshape(b, t, c * self.patch)

    def embed_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Patch embeddings, the CLS token and the positional encoding:
        (batch, n_prefix + T, d_model), the input of the first block."""
        h = self.embed(self.tokens(x))
        if self.cls_token is not None:
            h = torch.cat([self.cls_token.expand(h.size(0), -1, -1), h], dim=1)
        if self.pe is not None:
            h = h + self.pe                     # (T, d) broadcast over batch
        elif self.pos_embedding is not None:
            pos = torch.arange(h.size(1), device=h.device)
            h = h + self.pos_embedding(pos)
        return h

    def encode(self, x: torch.Tensor):
        """Token outputs (batch, n_prefix + T, d_model) after the final
        LayerNorm, and the attention weights of every block, a list of
        (batch, heads, n_prefix + T, n_prefix + T)."""
        h = self.embed_tokens(x)
        weights = []
        for block in self.blocks:
            h, w = block(h)
            weights.append(w)
        return self.ln_final(h), weights

    def summarize(self, x: torch.Tensor) -> torch.Tensor:
        """One vector per record: mean of the patch tokens, or the CLS
        output."""
        h, _ = self.encode(x)
        if self.readout == "cls":
            return h[:, 0]
        return h.mean(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.dropout(self.summarize(x)))

    @torch.no_grad()
    def attention_maps(self, x: torch.Tensor) -> torch.Tensor:
        """Attention weights of every block, (batch, layers, heads, T', T')
        with T' = n_prefix + T; row = query, column = key. Call in eval
        mode (no dropout is applied to the weights anyway)."""
        _, weights = self.encode(x)
        return torch.stack(weights, dim=1)
