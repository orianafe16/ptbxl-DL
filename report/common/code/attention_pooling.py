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

