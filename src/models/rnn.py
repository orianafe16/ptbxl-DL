"""Recurrent networks (LSTM, GRU) for the 5 superclasses (T11, DECISIONS D17).

A recurrent network reads a sequence one step at a time. At step t its
hidden state is computed from the current input x(t) AND the previous state
h(t-1) (weekly summary 6/11), so the same weights are applied at every step
and the state carries what the network has seen so far.

    signal (batch, 12, 1000)
      -> sequence (batch, T, features)      front end: frames or convolutions
      -> LSTM / GRU                         out: (batch, T, hidden), h_n
      -> one vector per record              readout: last state, mean or
                                            attention over the T states
      -> dropout -> Linear -> 5 logits      same head as the CNN (D16)

Axis order: nn.LSTM(batch_first=True) expects (batch, time, features), the
opposite of Conv1d's (batch, channels, time), so the signal is permuted
(weekly summary 13/11). Every record has exactly 1000 samples, so there is
no padding and pack_padded_sequence is not needed.

The output is made of logits: no sigmoid in the model (D2).
"""
import math

import torch
from torch import nn

from src.models.cnn import ConvBlock

CELLS = {"lstm": nn.LSTM, "gru": nn.GRU}
FRONT_ENDS = ("frames", "conv")
READOUTS = ("last", "mean", "attention")
ATTENTIONS = ("additive", "notebook")


class AttentionPooling(nn.Module):
    """Attention over the hidden states of the RNN (D17).

    For every time step a score e_t is computed from h_t; a softmax over
    time turns the scores into weights alpha_t >= 0 that sum to 1; the
    context vector sum_t alpha_t h_t replaces the last state as the input
    of the classifier (class notebook
    TODO_MLLB_Embeddings_GRU_Attention_SentimentAnalysis.ipynb, steps 1-3).

    In translation (Attention slides 12-15) the score compares the decoder
    state (the query) with each encoder state (the keys). A classifier has
    no decoder, so the query is a learned vector, the same for every record.

    kind="additive": e_t = v . tanh(W h_t + b), a feed-forward network with
        one hidden layer of `dim` units (Bahdanau, Attention slide 22).
    kind="notebook": e_t = tanh(w . h_t + b), the notebook's Dense(1,
        activation='tanh'). Its scores lie in [-1, 1], so no weight can be
        more than e^2 (about 7.4) times another (see max_weight_bound).
    """

    def __init__(self, d_in: int, dim: int = 64, kind: str = "additive"):
        super().__init__()
        if kind not in ATTENTIONS:
            raise ValueError(f"attention must be one of {ATTENTIONS}")
        self.kind = kind
        if kind == "additive":
            self.proj = nn.Linear(d_in, dim)
            self.v = nn.Linear(dim, 1, bias=False)   # the learned "query"
        else:
            self.score_layer = nn.Linear(d_in, 1)

    def scores(self, h: torch.Tensor) -> torch.Tensor:
        """h: (batch, T, d) -> e: (batch, T)."""
        if self.kind == "additive":
            e = self.v(torch.tanh(self.proj(h)))     # (batch, T, 1)
        else:
            e = torch.tanh(self.score_layer(h))      # (batch, T, 1)
        # nn.Linear acts on the last axis, at every time step with the same
        # weights: it is already "time distributed" (weekly summary 13/11).
        return e.squeeze(-1)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        alpha = torch.softmax(self.scores(h), dim=1)            # over time
        context = torch.bmm(alpha.unsqueeze(1), h).squeeze(1)   # (batch, d)
        return context, alpha

    @staticmethod
    def max_weight_bound(n_steps: int) -> float:
        """Largest weight the notebook score allows: one score at +1, all
        the others at -1, so alpha_max = e / (e + (T - 1) / e)."""
        return 1.0 / (1.0 + (n_steps - 1) * math.exp(-2.0))


class RNNClassifier(nn.Module):
    def __init__(self, n_leads: int = 12, n_samples: int = 1000,
                 n_classes: int = 5, cell: str = "lstm",
                 hidden_size: int = 128, num_layers: int = 2,
                 bidirectional: bool = False, front_end: str = "frames",
                 frame: int = 10, conv_channels: tuple | list = (32, 64, 128),
                 readout: str = "last", attention: str = "additive",
                 attention_dim: int = 64, dropout: float = 0.5):
        super().__init__()
        if cell not in CELLS:
            raise ValueError(f"cell must be one of {sorted(CELLS)}")
        if front_end not in FRONT_ENDS:
            raise ValueError(f"front_end must be one of {FRONT_ENDS}")
        if readout not in READOUTS:
            raise ValueError(f"readout must be one of {READOUTS}")
        if attention not in ATTENTIONS:
            raise ValueError(f"attention must be one of {ATTENTIONS}")
        if hidden_size < 1 or num_layers < 1:
            raise ValueError("hidden_size and num_layers must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.cell, self.front_end, self.readout = cell, front_end, readout
        self.n_samples = n_samples
        self.bidirectional = bool(bidirectional)
        self.hidden_size = int(hidden_size)

        if front_end == "frames":
            # `frame` consecutive samples of the 12 leads form one input
            # vector: 1000 samples -> T = 1000 / frame steps. frame=1 is the
            # raw signal, one sample per step.
            if frame < 1 or n_samples % frame:
                raise ValueError(f"frame must divide n_samples={n_samples}")
            self.frame = int(frame)
            self.conv = None
            self.n_steps = n_samples // frame
            self.step_samples = self.frame
            d_in = n_leads * frame
        else:
            # The first blocks of the CNN (D16) as a feature extractor; each
            # block halves the length, so 3 blocks give 1000 / 8 = 125 steps
            # of conv_channels[-1] features (CNN -> LSTM, RNN slide 80).
            if len(conv_channels) == 0:
                raise ValueError("the conv front end needs at least one block")
            self.frame = None
            blocks, c_in = [], n_leads
            for c_out in conv_channels:
                blocks.append(ConvBlock(c_in, c_out, 7))
                c_in = c_out
            self.conv = nn.Sequential(*blocks)
            self.step_samples = 2 ** len(conv_channels)
            self.n_steps = n_samples // self.step_samples
            d_in = c_in

        # PyTorch's `dropout` argument of nn.LSTM only acts between stacked
        # layers; it is left at 0 (D17).
        self.rnn = CELLS[cell](d_in, hidden_size, num_layers=num_layers,
                               batch_first=True, bidirectional=bidirectional)
        d_out = hidden_size * (2 if bidirectional else 1)
        self.attention = (AttentionPooling(d_out, attention_dim, attention)
                          if readout == "attention" else None)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(d_out, n_classes)            # logits

    # ------------------------------------------------------------------
    def sequence(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, leads, samples) -> (batch, T, features), what the RNN
        reads. Frames: the leads x frame samples of each step are flattened
        lead by lead into one vector."""
        if self.conv is not None:
            return self.conv(x).permute(0, 2, 1)       # (B, C, T) -> (B, T, C)
        b, c, n = x.shape
        t = n // self.frame
        x = x.reshape(b, c, t, self.frame).permute(0, 2, 1, 3)  # (B, T, C, F)
        return x.reshape(b, t, c * self.frame)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Hidden states of the top layer at every step, out (batch, T, d),
        and the final state used by readout="last", (batch, d).

        nn.LSTM returns out and (h_n, c_n); nn.GRU returns out and h_n (no
        cell state). h_n has one row per layer and direction, layer by
        layer: for the top layer, h_n[-1] is the forward direction of a
        unidirectional network (equal to out[:, -1, :]); with two
        directions h_n[-2] is the forward state, which has read the record
        left to right, and h_n[-1] the backward one, which has read it right
        to left and sits at out[:, 0, hidden:], not at out[:, -1, :]
        (weekly summary 13/11; PyTorch docs: directions 0 and 1).
        """
        out, h = self.rnn(self.sequence(x))
        h_n = h[0] if isinstance(h, tuple) else h     # LSTM: (h_n, c_n)
        last = torch.cat([h_n[-2], h_n[-1]], dim=1) if self.bidirectional \
            else h_n[-1]
        return out, last

    def summarize(self, x: torch.Tensor):
        """One vector per record and, with attention, the weights over time
        (batch, T); None for the other readouts."""
        out, last = self.encode(x)
        if self.readout == "last":
            return last, None
        if self.readout == "mean":
            return out.mean(dim=1), None
        return self.attention(out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z, _ = self.summarize(x)
        return self.classifier(self.dropout(z))

    @torch.no_grad()
    def attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """alpha_t of every record, (batch, T). Step t covers the input
        samples [t * step_samples, (t + 1) * step_samples)."""
        if self.attention is None:
            raise ValueError("this model has no attention readout")
        return self.summarize(x)[1]
