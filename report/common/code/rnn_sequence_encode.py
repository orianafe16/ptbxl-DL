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
