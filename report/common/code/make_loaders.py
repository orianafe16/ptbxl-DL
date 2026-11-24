    X, y = cache["X"], cache["y"]
    idx = split_indices(cache, include_no_superclass)
    mean, std = lead_stats(X, idx["train"]) if normalize else (None, None)

    # A seeded generator makes the shuffling order reproducible.
    generator = torch.Generator().manual_seed(seed)
    loaders = {}
    for split, rows in idx.items():
        loaders[split] = DataLoader(
            PTBXLDataset(X, y, rows, mean, std),
            batch_size=batch_size,
            shuffle=(split == "train"),        # only the training set
            generator=generator if split == "train" else None,
            num_workers=0,   # Windows: extra workers would copy the cache
            pin_memory=torch.cuda.is_available(),
        )
