"""Generate a subset ``index.json`` for overfitting / debugging experiments.

Usage examples::

    # Random sample of 4 clips
    python make_subset_index.py --index data/index.json -n 4 --output data/index_overfit.json

    # First 2 clips (deterministic)
    python make_subset_index.py --index data/index.json -n 2 --strategy first --output data/index_2.json

    # Pick by clip indices (0-based)
    python make_subset_index.py --index data/index.json --pick 0 3 7 --output data/index_pick.json

    # Filter by source dataset
    python make_subset_index.py --index data/index.json --source 4dnex --output data/index_4dnex.json

    # Filter by source, then random sample 200 clips
    python make_subset_index.py --index data/index.json --source omniworld_hoi4d -n 200 --seed 42 --strategy random --output data/index_overfit_hoi4d_200.json

Then train with::

    accelerate launch ... finetune.py --data_root data --index_file data/index_overfit.json ...
"""

import argparse
import json
import random
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a subset index.json")
    parser.add_argument("--index", type=str, required=True, help="Path to the source index.json")
    parser.add_argument("--output", "-o", type=str, required=True, help="Path to write the subset index.json")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-n", type=int, help="Number of clips to sample")
    group.add_argument("--pick", type=int, nargs="+", help="Specific clip indices (0-based) to select")

    parser.add_argument("--source", type=str, default=None, help="Filter clips by source before selection")

    parser.add_argument(
        "--strategy", choices=["random", "first"], default="random",
        help="Sampling strategy when using -n (default: random)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    args = parser.parse_args()

    with open(args.index, "r", encoding="utf-8") as f:
        index = json.load(f)

    clips = index["clips"]
    total = len(clips)

    # Optional pre-filter by source_dataset.
    if args.source is not None:
        clips = [c for c in clips if c.get("source") == args.source]
        if not clips:
            print(f"Error: no clips found with source='{args.source}'", file=sys.stderr)
            sys.exit(1)

    filtered_total = len(clips)

    if args.pick is not None:
        bad = [i for i in args.pick if i < 0 or i >= filtered_total]
        if bad:
            print(f"Error: indices out of range (total {filtered_total}): {bad}", file=sys.stderr)
            sys.exit(1)
        subset = [clips[i] for i in args.pick]
    else:
        n = min(args.n, filtered_total)
        if args.strategy == "first":
            subset = clips[:n]
        else:
            random.seed(args.seed)
            subset = random.sample(clips, n)

    # Preserve original config, update clip list
    out_index = {**index, "clips": subset}

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_index, f, indent=2, ensure_ascii=False)

    if args.source is None:
        print(f"Wrote {len(subset)}/{total} clips to {out_path}")
    else:
        print(
            f"Wrote {len(subset)}/{filtered_total} filtered clips "
            f"(source='{args.source}', original total={total}) to {out_path}"
        )


if __name__ == "__main__":
    main()
