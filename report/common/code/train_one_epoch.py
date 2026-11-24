def train_one_epoch(model: nn.Module, loader, task: Task,
                    optimizer: torch.optim.Optimizer, device: torch.device,
                    grad_clip: float | None = None) -> float:
    """One pass over the training set; returns the mean training loss."""
    model.train()                     # dropout and batch norm in training mode
    total, n = 0.0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad()                  # 1. clear old gradients
        output = model(x)                      # 2. forward
        loss = task.loss(output, y)            # 3. loss (mean over the batch)
        loss.backward()                        # 4. backpropagation
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()                       # 5. update the weights
        k = task.n_terms(y)
        total += loss.item() * k               # .item(): a Python number,
        n += k                                 # so no graph is kept alive
    return total / n


@torch.no_grad()                      # no gradients: faster, less memory
def evaluate(model: nn.Module, loader, task: Task,
             device: torch.device) -> tuple[float, object]:
    """Mean loss over the whole split and whatever the task collects."""
    model.eval()                      # dropout off, batch norm uses its
    total, n = 0.0, 0                 # running statistics
    state = task.start()
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        output = model(x)
        # Sum, then divide by the number of terms at the end: the exact mean
        # over the split, even if the last batch is smaller.
        total += task.loss(output, y, reduction="sum").item()
        n += task.n_terms(y)
        state = task.collect(state, output, y)
    return total / n, state
