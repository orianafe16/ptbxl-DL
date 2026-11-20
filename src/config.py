"""Project-wide settings: paths, classes, folds and the random seed.

Every other module imports its settings from here, so a path or a fold
is changed in one place only.
"""
from pathlib import Path

# Repository root: the folder that contains src/.
ROOT = Path(__file__).resolve().parents[1]

# Raw PTB-XL 1.0.3 files, placed by hand (data/ is not committed).
RAW_DIR = ROOT / "data" / "raw" / "ptbxl"

# Single-file cache with every 100 Hz signal and its labels (built by
# `python -m src.prepare_data`).
CACHE_PATH = ROOT / "data" / "cache" / "ptbxl_100hz.npz"

SAMPLING_RATE = 100        # Hz; we use the 100 Hz version only (DECISIONS D1)
N_SAMPLES = 1000           # 10 s x 100 Hz
LEADS = ["I", "II", "III", "AVR", "AVL", "AVF",
         "V1", "V2", "V3", "V4", "V5", "V6"]

# The five diagnostic superclasses (DECISIONS D2). The order fixes the
# column order of every label matrix and every output layer.
CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]

# Official stratified folds (DECISIONS D3).
TRAIN_FOLDS = [1, 2, 3, 4, 5, 6, 7, 8]
VAL_FOLD = 9
TEST_FOLD = 10

SEED = 42
