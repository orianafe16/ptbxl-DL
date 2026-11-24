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
