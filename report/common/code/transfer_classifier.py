class TransferClassifier(nn.Module):
    """Generator backbone + record vector + new classification head.

    freeze=True is the linear probe: the backbone's parameters do not
    require gradients (the optimiser of src/train.py only receives
    trainable parameters, D14) and the backbone stays in eval mode even
    when the loop calls model.train(): its dropout is off, so a record
    always gives the same vector, as a fixed feature extractor should."""

    def __init__(self, backbone: nn.Module, n_classes: int = len(config.CLASSES),
                 head_dropout: float = 0.5, freeze: bool = False):
        super().__init__()
        self.backbone = backbone
        self.freeze = bool(freeze)
        if self.freeze:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
        self.drop = nn.Dropout(head_dropout)
        self.head = nn.Linear(feature_dim(backbone), n_classes)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            self.backbone.eval()
        return self

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.head(self.drop(self.backbone.record_features(tokens)))
