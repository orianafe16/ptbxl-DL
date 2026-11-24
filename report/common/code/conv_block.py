class ConvBlock(nn.Module):
    """Two convolutions with an optional skip connection, then pooling.

    Padding k // 2 ("same" padding, odd k) keeps the length, so the main
    path and the shortcut can be added sample by sample; MaxPool1d(2) then
    halves the length.
    """

    def __init__(self, c_in: int, c_out: int, kernel_size: int,
                 batch_norm: bool = True, residual: bool = True):
        super().__init__()
        pad = kernel_size // 2

        def norm():
            return nn.BatchNorm1d(c_out) if batch_norm else nn.Identity()

        self.conv1 = nn.Conv1d(c_in, c_out, kernel_size, padding=pad)
        self.bn1 = norm()
        self.conv2 = nn.Conv1d(c_out, c_out, kernel_size, padding=pad)
        self.bn2 = norm()
        self.relu = nn.ReLU()
        if not residual:
            self.shortcut = None
        elif c_in == c_out:
            self.shortcut = nn.Identity()       # adds no parameter (slide 55)
        else:
            # Projection shortcut: a 1x1 convolution only changes the number
            # of channels, so that x can be added to F(x) (slide 56).
            self.shortcut = nn.Sequential(nn.Conv1d(c_in, c_out, 1), norm())
        self.pool = nn.MaxPool1d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.shortcut is not None:
            # The block learns the residual F(x) = H(x) - x; if the identity
            # is the best it can do, it only has to push F(x) to zero
            # (slide 54). The gradient also flows straight through the "+".
            out = out + self.shortcut(x)
        return self.pool(self.relu(out))
