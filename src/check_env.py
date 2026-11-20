"""Print the software versions and check that PyTorch can use the GPU.

Run from the repository root:
    python -m src.check_env
"""
import platform
import sys


def main() -> None:
    print(f"Python      : {sys.version.split()[0]} ({platform.system()})")

    # Import each library separately so that a missing one is reported
    # by name instead of stopping the whole check.
    for name in ["numpy", "pandas", "sklearn", "matplotlib", "wfdb", "torch"]:
        try:
            module = __import__(name)
            print(f"{name:<12}: {module.__version__}")
        except ImportError:
            print(f"{name:<12}: NOT INSTALLED")

    try:
        import torch
    except ImportError:
        return

    # torch.cuda.is_available() is True only if the installed PyTorch build
    # includes CUDA support AND the NVIDIA driver is recent enough for it.
    if not torch.cuda.is_available():
        print("GPU         : not available, PyTorch would run on the CPU")
        return

    print(f"GPU         : {torch.cuda.get_device_name(0)}")
    print(f"CUDA (torch): {torch.version.cuda}")

    # Small real computation on the GPU: creating the tensor is not enough
    # to prove that the device works.
    x = torch.rand(1000, 1000, device="cuda")
    y = x @ x
    print(f"GPU test    : OK, result on {y.device}")


if __name__ == "__main__":
    main()
