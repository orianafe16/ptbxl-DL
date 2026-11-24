def causal_mask(T: int, device=None) -> torch.Tensor:
    """(T, T) boolean mask, True where query i may look at key j, i.e. j <= i.

    A decoder that predicts the next token must not see the future (Part II
    slides 11-14). With the mask, the whole sequence is processed in one
    pass and every position still only uses the positions before it, which
    is what makes training parallel (unlike an RNN, slide 13)."""
    return torch.ones(T, T, dtype=torch.bool, device=device).tril()


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
              mask: torch.Tensor | None = None):
    """Scaled dot-product attention (notebook 01, cell 5; Part I slides
    36-41).

    q, k: (..., T, d_k); v: (..., T, d_v); mask: broadcastable to (..., T, T),
    True = allowed.

    1. scores = q k^T: one dot product per (query, key) pair, (..., T, T)
    2. / sqrt(d_k): without it the dot products grow with d_k and push the
       softmax into regions with tiny gradients (Part I slide 49)
    3. masked scores -> -inf, so their weight after the softmax is exactly 0
    4. softmax over the KEYS (last axis): each query's weights are >= 0 and
       sum to 1
    5. output = weights v: each query receives a weighted mean of the values

    Returns (output (..., T, d_v), weights (..., T, T)).
    """
    d_k = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return weights @ v, weights
