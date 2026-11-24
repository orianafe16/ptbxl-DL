"""Copy short excerpts of the repository code into report/common/code/ for the
MLLB listings (chat DL 10, 2026-11-24). The excerpts are copied verbatim, so the
report never shows code that differs from the repository; the first line number
is kept so that a listing can be numbered as in the source file:

    \\lstinputlisting[firstnumber=<start>]{common/code/<name>.py}

Run from report/ after any change to the files below:  python common/make_code_excerpts.py
It stops if an excerpt no longer starts with the expected text.
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent.parent / "src"
OUT = HERE / "code"

# name: (file under src/, first line, last line, expected start of the first line)
EXCERPTS = {
    "train_one_epoch": ("train.py", 101, 139, "def train_one_epoch"),
    "multilabel_loss": ("tasks.py", 91, 95, "    def loss"),
    "patient_bootstrap": ("metrics.py", 101, 119, "def patient_bootstrap"),
    "dense_net": ("models/dense.py", 34, 51, "        layers: list"),
    "conv_block": ("models/cnn.py", 31, 70, "class ConvBlock"),
    "causal_attention": ("models/attention.py", 29, 62, "def causal_mask"),
    "tokenizer": ("generative.py", 69, 74, "    def encode"),
    "sample_next": ("generative.py", 468, 485, "def sample_next"),
    "transfer_classifier": ("transfer.py", 185, 212, "class TransferClassifier"),
}


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for name, (rel, a, b, start) in EXCERPTS.items():
        lines = (SRC / rel).read_text(encoding="utf-8").splitlines()
        if not lines[a - 1].startswith(start):
            raise SystemExit(f"{rel}:{a} no longer starts with {start!r}")
        (OUT / f"{name}.py").write_text("\n".join(lines[a - 1:b]) + "\n", encoding="utf-8")
    print(f"{len(EXCERPTS)} excerpts in {OUT}")


if __name__ == "__main__":
    main()
