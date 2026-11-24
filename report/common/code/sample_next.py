def sample_next(logits: torch.Tensor, temperature: float,
                top_k: int | None = None,
                generator: torch.Generator | None = None) -> torch.Tensor:
    """Choose the next token of each sequence from its logits (batch, V)
    (Sampling_in_Text_Generation.ipynb, parts 2-4):
      temperature 0: greedy search, the most probable token;
      otherwise the logits are divided by the temperature before the
      softmax (low T: sharper, more repetitive; high T: flatter, more
      random) and with top_k only the k most probable tokens are kept and
      their probabilities renormalised; then one token is drawn."""
    if temperature <= 0:
        return logits.argmax(dim=-1)
    z = logits.float() / temperature
    if top_k is not None:
        kth = torch.topk(z, min(top_k, z.shape[-1]), dim=-1).values[:, -1:]
        z = z.masked_fill(z < kth, float("-inf"))
    p = torch.softmax(z, dim=-1)
    return torch.multinomial(p, 1, generator=generator).squeeze(-1)
