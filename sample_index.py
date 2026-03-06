"""Sample N clips from an index.json and write a new index.json.

Usage:
    python sample_index.py --index ./data/index.json --n 100
    python sample_index.py --index ./data/index.json --n 50 --seed 42 --out index_sample.json
"""

import argparse
import json
import random
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description="Sample N clips from index.json")
    p.add_argument("--index", type=str, required=True, help="Input index.json")
    p.add_argument("--n", type=int, required=True, help="Number of clips to sample")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p.add_argument("--out", type=str, default=None,
                   help="Output path (default: index_sample_{n}.json next to input)")
    args = p.parse_args()

    index_path = Path(args.index)
    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)

    clips = index["clips"]
    n = min(args.n, len(clips))

    random.seed(args.seed)
    sampled = random.sample(clips, n)

    # Recompute statistics.
    sources = {}
    vids_by_src = {}
    for e in sampled:
        s = str(e["source"])
        vids_by_src.setdefault(s, set()).add(str(e["video_id"]))
        sources.setdefault(s, {"videos": 0, "clips": 0})["clips"] += 1
    for s in sources:
        sources[s]["videos"] = len(vids_by_src[s])

    index["statistics"] = {
        "num_videos": len({str(e["video_id"]) for e in sampled}),
        "num_clips": n,
        "sources": sources,
    }
    index["clips"] = sampled

    out_path = Path(args.out) if args.out else index_path.parent / f"index_sample_{n}.json"
    out_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Sampled {n}/{len(clips)} clips -> {out_path}")


if __name__ == "__main__":
    main()
