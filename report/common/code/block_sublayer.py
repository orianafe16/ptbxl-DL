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
