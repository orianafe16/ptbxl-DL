"""One JSON log per run in results/logs/ (DECISIONS D4, D14).

Every number of the reports comes from these files through
report/make_numbers.py, so a log must say exactly what was run: the
configuration, the data, the curves, the results and the software.
"""
import json
import platform
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import sklearn
import torch

from src import config

# 2: git_dirty looks only at the code under src/ (D14, second entry of
# 2026-11-20). In logs with schema 1 it is always true and means nothing.
SCHEMA_VERSION = 2

# Code whose state decides whether a run is "dirty".
CODE_PATHS = ("src",)


def make_run_id(model_name: str, tag: str, seed: int,
                when: datetime | None = None) -> str:
    """<model>_<tag>_s<seed>_<YYYYMMDD-HHMMSS>, e.g.
    dense_base_s42_20261121-101500."""
    when = when or datetime.now()
    return f"{model_name}_{tag}_s{seed}_{when:%Y%m%d-%H%M%S}"


def _git(*args, cwd: Path = config.ROOT) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=cwd,
                              capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_dirty(root: Path = config.ROOT) -> bool | None:
    """True when the code under src/ differs from the last commit.

    Counted: tracked files changed (staged or not), deleted, or new files
    not yet added (git still skips what .gitignore excludes, such as
    __pycache__). Not counted: results/, report/ and docs/. A run writes
    its own predictions file before this check, so looking at the whole
    repository made every full run "dirty" (AI Error Bounty B5). None when
    git is not available.
    """
    status = _git("status", "--porcelain", "--untracked-files=all", "--",
                  *CODE_PATHS, cwd=root)
    return None if status is None else bool(status)


def environment(device: torch.device | None = None) -> dict:
    """Date, git commit and software versions, for reproducibility.

    `git_dirty` is True when the code under src/ had uncommitted changes:
    the commit alone then does not describe the code that produced the run.
    """
    return {
        "created": datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git("rev-parse", "--short", "HEAD"),
        "git_dirty": git_dirty(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "device": (torch.cuda.get_device_name(device)
                   if device is not None and device.type == "cuda"
                   else "cpu"),
        "platform": platform.platform(),
    }


def _to_json(o):
    """numpy numbers and arrays are not JSON types: convert them."""
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, Path):
        return o.as_posix()
    raise TypeError(f"not JSON serialisable: {type(o)}")


def write_run(record: dict, logs_dir: Path = config.LOGS_DIR) -> Path:
    """Write <logs_dir>/<run_id>.json and return its path."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    path = logs_dir / f"{record['run_id']}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False, default=_to_json)
    return path


def load_run(name: str, logs_dir: Path = config.LOGS_DIR) -> dict:
    """Load a log by run id (with or without .json)."""
    path = logs_dir / (name if name.endswith(".json") else f"{name}.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def previous_final_runs(model_name: str, tag: str, seed: int,
                        logs_dir: Path = config.LOGS_DIR) -> list[str]:
    """Run ids of earlier non-fast --final runs with the same model, tag
    and seed. The test fold is used once per final model (D3)."""
    found = []
    for path in sorted(logs_dir.glob(f"{model_name}_{tag}_s{seed}_*.json")):
        with open(path, encoding="utf-8") as f:
            rec = json.load(f)
        if rec.get("final") and not rec.get("fast"):
            found.append(rec["run_id"])
    return found
