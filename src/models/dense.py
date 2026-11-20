"""Dense (fully connected) network for the 5 superclasses (T9, DECISIONS D15).

The 12 x 1000 signal is flattened into one vector of 12,000 numbers and
passed through hidden blocks of

    Linear -> BatchNorm1d (optional) -> ReLU -> Dropout (optional)

followed by a Linear layer with one output per class. The output is made of
logits: no sigmoid in the model, because BCEWithLogitsLoss applies it inside
(D2) and the metrics apply it outside (D14).

What the flattening means: every neuron of the first hidden layer has one
weight per (lead, time sample) and compares the WHOLE record with that
weight pattern. A heartbeat that appears 0.3 s later in another record hits
different weights, so the network has to learn the same shape at every
position separately. This is the limitation of feed-forward networks on
signals that motivates convolutions (weekly summary 30/10).
"""
import torch
from torch import nn


class DenseNet(nn.Module):
    def __init__(self, n_leads: int = 12, n_samples: int = 1000,
                 n_classes: int = 5, hidden_sizes: tuple | list = (256, 256),
                 batch_norm: bool = True, dropout: float = 0.5):
        super().__init__()
        if len(hidden_sizes) == 0:
            raise ValueError("at least one hidden layer is needed")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.hidden_sizes = tuple(int(h) for h in hidden_sizes)

        layers: list[nn.Module] = [nn.Flatten()]     # (B, 12, 1000) -> (B, 12000)
        width_in = n_leads * n_samples
        for width in self.hidden_sizes:
            layers.append(nn.Linear(width_in, width))
            if batch_norm:
                # Normalises each unit's pre-activation over the batch (zero
                # mean, unit variance, then a learned scale and shift), so it
                # stays in the region where ReLU is useful.
                layers.append(nn.BatchNorm1d(width))
            layers.append(nn.ReLU())
            if dropout > 0:
                # Training: each unit is switched off with probability p and
                # the survivors are scaled by 1 / (1 - p). Evaluation
                # (model.eval()): nothing is dropped and nothing is scaled.
                layers.append(nn.Dropout(dropout))
            width_in = width
        layers.append(nn.Linear(width_in, n_classes))  # logits, no sigmoid
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @property
    def first_layer(self) -> nn.Linear:
        """The Linear layer that sees the flattened signal; its weight has
        shape (hidden units, 12 * 1000) and can be read as 12 x 1000
        patterns (FeedForward slides 29-30)."""
        return self.net[1]
