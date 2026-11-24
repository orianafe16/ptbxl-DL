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
