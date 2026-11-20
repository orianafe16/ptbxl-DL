"""1D convolutional network for the 5 superclasses (T10, DECISIONS D16).

Input (batch, 12, 1000): the 12 leads are the input channels and the 1000
samples (10 s at 100 Hz) the time axis, as in the 1D model of the class
notebook MLLB_Pytorch_HAR_CNN_TODO.ipynb (9 sensor channels there).

    block 1 ... block N  ->  global average pooling  ->  dropout  ->  Linear

Each block (ResNet-style basic block, PRDL_MLLB_CNN.pptx slides 46-62):

    main path:  Conv1d(k) -> BN -> ReLU -> Conv1d(k) -> BN
    shortcut:   identity, or Conv1d(1x1) -> BN when the number of channels
                changes (slide 56: x and F(x) must have the same size)
    output:     MaxPool1d(2)( ReLU(main + shortcut) )

What a convolution changes with respect to the Dense network (D15): a
filter of k samples slides along time with the SAME weights at every
position, so a QRS complex is detected wherever it appears (weight sharing,
translation equivariance), with k x channels weights instead of one weight
per time sample. Global average pooling then averages each feature map over
time, so the classifier sees "how much of each pattern there is", not
"where it is" (translation invariance).

The output is made of logits: no sigmoid in the model, because
BCEWithLogitsLoss applies it inside (D2) and the metrics outside (D14).
"""
import torch
from torch import nn


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


class CNN1D(nn.Module):
    def __init__(self, n_leads: int = 12, n_samples: int = 1000,
                 n_classes: int = 5,
                 channels: tuple | list = (32, 64, 128, 128),
                 kernel_size: int = 7, batch_norm: bool = True,
                 residual: bool = True, pooling: str = "gap",
                 dropout: float = 0.5):
        super().__init__()
        if len(channels) == 0:
            raise ValueError("at least one convolutional block is needed")
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd ('same' padding)")
        if pooling not in ("gap", "flatten"):
            raise ValueError("pooling must be 'gap' or 'flatten'")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.channels = tuple(int(c) for c in channels)
        self.kernel_size = int(kernel_size)
        self.pooling = pooling

        blocks, c_in = [], n_leads
        for c_out in self.channels:
            blocks.append(ConvBlock(c_in, c_out, kernel_size, batch_norm,
                                    residual))
            c_in = c_out
        self.blocks = nn.ModuleList(blocks)

        # Length of the last feature map: every block halves it (floor).
        length = n_samples
        for _ in self.channels:
            length //= 2
        if length < 1:
            raise ValueError(f"{len(self.channels)} blocks need more than "
                             f"{n_samples} samples")
        self.out_length = length
        n_features = c_in if pooling == "gap" else c_in * length
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(n_features, n_classes)   # logits

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Output of the last block: (batch, channels, out_length)."""
        for block in self.blocks:
            x = block(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.features(x)
        if self.pooling == "gap":
            z = a.mean(dim=2)             # global average pooling over time
        else:
            z = a.flatten(1)              # one weight per (channel, position)
        return self.classifier(self.dropout(z))

    @property
    def first_conv(self) -> nn.Conv1d:
        """Convolution that sees the raw leads; weight (filters, 12, k)."""
        return self.blocks[0].conv1

    def receptive_field(self) -> int:
        """Number of input samples that one position of the last feature
        map depends on (computed layer by layer: a layer with kernel k adds
        (k - 1) x jump samples, where jump is the product of the strides
        before it; pooling with stride 2 doubles the jump)."""
        r, jump = 1, 1
        for _ in self.channels:
            r += 2 * (self.kernel_size - 1) * jump    # two convolutions
            r += (2 - 1) * jump                       # MaxPool1d(2)
            jump *= 2
        return r

    def class_activation_map(self, x: torch.Tensor) -> torch.Tensor:
        """Class activation map, (batch, n_classes, out_length).

        With global average pooling followed by one Linear layer,
            logit_c = b_c + sum_k w_ck * mean_t A_k(t)
                    = b_c + mean_t [ sum_k w_ck A_k(t) ],
        so CAM_c(t) = sum_k w_ck A_k(t) says how much each time position of
        the last feature map pushes the logit of class c up or down; its
        mean over time plus the bias is exactly the logit (in eval mode,
        where dropout does nothing). Only defined for pooling="gap".
        """
        if self.pooling != "gap":
            raise ValueError("the class activation map needs pooling='gap'")
        a = self.features(x)                               # (B, K, L)
        return torch.einsum("ck,bkl->bcl", self.classifier.weight, a)
