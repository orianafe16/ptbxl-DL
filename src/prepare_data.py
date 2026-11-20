"""Check the raw PTB-XL files and build the 100 Hz cache.

Run from the repository root:
    python -m src.prepare_data               # full check and cache
    python -m src.prepare_data --limit 50    # quick smoke test
"""
import argparse
import time

from src import config
from src.data import build_cache, verify_raw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="only the first N records (smoke test)")
    parser.add_argument("--skip-verify", action="store_true",
                        help="skip the SHA-256 check of the raw files")
    args = parser.parse_args()

    if not args.skip_verify:
        print("Checking SHA-256 of the raw files (about a minute)...")
        start = time.time()
        problems = verify_raw()
        if problems:
            print(f"{len(problems)} problems found, first ones:")
            for p in problems[:10]:
                print("  ", p)
            raise SystemExit(1)
        print(f"Raw files OK ({time.time() - start:.0f} s)")

    print("Building the cache...")
    start = time.time()
    data = build_cache(limit=args.limit)
    size_mb = config.CACHE_PATH.stat().st_size / 1e6
    print(f"Cache written: {config.CACHE_PATH} ({size_mb:.0f} MB, "
          f"{time.time() - start:.0f} s)")

    # Summary to paste back into the chat.
    y = data["y"]
    print(f"X shape            : {data['X'].shape}")
    print(f"y shape            : {y.shape}")
    print(f"records / patients : {len(y)} / {len(set(data['patient_id']))}")
    for j, name in enumerate(config.CLASSES):
        print(f"  {name:<5}: {int(y[:, j].sum())}")
    print(f"without superclass : {int((~data['has_superclass']).sum())}")


if __name__ == "__main__":
    main()
