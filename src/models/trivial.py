"""Trivial model for the end-to-end check of the training loop (D14).

Logistic regression on 24 summary numbers per record: the mean and the
standard deviation over time of each of the 12 leads. It ignores the shape
of the waveform, so any real model should beat it; it gives a floor for
the comparison and trains in seconds.
"""
import torch
from torch import nn


class LeadStatsLinear(nn.Module):
    def __init__(self, n_leads: int = 12, n_classes: int = 5):
        super().__init__()
        self.linear = nn.Linear(2 * n_leads, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, leads, time) -> features (batch, 2 * leads)
        features = torch.cat([x.mean(dim=2), x.std(dim=2)], dim=1)
        return self.linear(features)    # logits: no sigmoid (D2)
