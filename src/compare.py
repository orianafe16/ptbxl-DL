"""Paired bootstrap comparison of two runs (DECISIONS D4, D14).

    python -m src.compare RUN_A RUN_B            # on validation
    python -m src.compare RUN_A RUN_B --split test

Both runs must have predictions on the same records (same split, same data
options). The interval of macro AUROC(A) - macro AUROC(B) is computed on
the same patient resamples; if it contains 0, neither model is declared
better. The result is written as a log, results/logs/compare_*.json, so
make_numbers.py can read it like any other run.
"""
import argparse
from datetime import datetime

import numpy as np

from src import config, metrics, runlog


def load_predictions(run_id: str, split: str) -> dict:
    rec = runlog.load_run(run_id)
    if rec["predictions"] is None:
        raise ValueError(f"{run_id} has no predictions file")
    with np.load(config.ROOT / rec["predictions"]) as data:
        if f"{split}_prob" not in data.files:
            raise ValueError(f"{run_id} has no {split} predictions "
                             f"(test needs a --final run)")
        return {k[len(split) + 1:]: data[k] for k in data.files
                if k.startswith(f"{split}_")}


def compare(run_a: str, run_b: str, split: str = "val",
            n_boot: int = 1000, seed: int = config.SEED) -> dict:
    a, b = load_predictions(run_a, split), load_predictions(run_b, split)
    # Same records in the same order, otherwise the pairing is meaningless.
    for key in ("ecg_id", "patient_id", "true"):
        if key in a and key in b and not np.array_equal(a[key], b[key]):
            raise ValueError(f"the two runs differ in {split}_{key}: "
                             "they were not evaluated on the same records")
    d = metrics.paired_bootstrap_diff(a["true"], a["prob"], b["prob"],
                                      a["patient_id"], n_boot=n_boot,
                                      seed=seed)
    fast = run_a.endswith("_fast") or run_b.endswith("_fast")
    return {
        "run_id": f"compare_{run_a}__{run_b}__{split}" + ("_fast" if fast else ""),
        "schema": runlog.SCHEMA_VERSION,
        "task": "paired_comparison",
        "fast": fast,
        "runs": {"a": run_a, "b": run_b, "split": split},
        "scalars": {"diff_macro_auroc": d["diff"],
                    "diff_macro_auroc_lo": d["lower"],
                    "diff_macro_auroc_hi": d["upper"],
                    "n_boot": d["n_boot"], "n_skipped": d["n_skipped"]},
        "meta": {**runlog.environment(),
                 "created": datetime.now().isoformat(timespec="seconds")},
    }


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("run_a")
    p.add_argument("run_b")
    p.add_argument("--split", default="val", choices=["val", "test"])
    a = p.parse_args(argv)
    rec = compare(a.run_a, a.run_b, a.split)
    path = runlog.write_run(rec)
    s = rec["scalars"]
    verdict = ("interval contains 0: no difference claimed"
               if s["diff_macro_auroc_lo"] <= 0 <= s["diff_macro_auroc_hi"]
               else "interval excludes 0")
    print(f"macro AUROC(A) - macro AUROC(B) on {a.split}: "
          f"{s['diff_macro_auroc']:.4f} "
          f"[{s['diff_macro_auroc_lo']:.4f}, {s['diff_macro_auroc_hi']:.4f}]"
          f" ({verdict}); skipped resamples: {s['n_skipped']}")
    print(f"written to {path.relative_to(config.ROOT).as_posix()}")


if __name__ == "__main__":
    main()
