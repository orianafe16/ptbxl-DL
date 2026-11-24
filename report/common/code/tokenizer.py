    def encode(self, x_mv: np.ndarray) -> np.ndarray:
        """mV -> tokens (int64, same shape)."""
        u = np.rint(np.asarray(x_mv, dtype=np.float64) * 1000.0).astype(np.int64)
        # floor_divide rounds towards -infinity, also for negative values
        k = np.floor_divide(u - self.lo_uv, self.delta_uv)
        return np.clip(k, 0, self.vocab - 1)
