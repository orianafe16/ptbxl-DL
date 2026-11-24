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
