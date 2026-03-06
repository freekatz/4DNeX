"""Generate index.json from existing videos/ directory or collect cache.

Scans videos/{source}/{video_id}/{clip_id}/ for complete clips (or reads
a collect cache .pkl file) and produces an index.json compatible with
process_dataset.py.

Usage:
    # Scan from disk
    python gen_index.py --data_root /path/to/dataset

    # Use collect cache (much faster on FUSE mounts)
    python gen_index.py --data_root /path/to/dataset \
        --collect_cache .collect_cache_omniworld_hoi4d_xxxx.pkl

    # Custom output path
    python gen_index.py --data_root /path/to/dataset --out index_partial.json
"""

import argparse
import hashlib
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List

_EXPECTED_FILES = ["video.mp4", "xyz.mp4", "first_frame.png", "caption.txt", "meta.json"]


def _entries_from_cache(data_root: Path, cache_path: Path,
                        existing_by_path: Dict[str, Dict]) -> tuple:
    """Build entries from a collect cache .pkl without scanning disk."""
    print(f"[gen_index] Loading collect cache: {cache_path.name} ...")
    with open(cache_path, "rb") as f:
        descriptors = pickle.load(f)
    print(f"[gen_index] {len(descriptors)} descriptors loaded")

    videos_root = data_root / "videos"
    entries: List[Dict] = []
    vids_by_src: Dict[str, set] = {}
    clip_counter: Dict[str, int] = {}
    complete, incomplete, reused = 0, 0, 0

    for desc in descriptors:
        idx = clip_counter.get(desc.video_id, 0)
        clip_counter[desc.video_id] = idx + 1
        clip_id = f"clip_{idx}"
        path_key = f"{desc.source_dataset}/{desc.video_id}/{clip_id}"

        # Reuse existing entry if already in index (skip disk reads).
        if path_key in existing_by_path:
            entries.append(existing_by_path[path_key])
            vids_by_src.setdefault(desc.source_dataset, set()).add(desc.video_id)
            reused += 1
            complete += 1
            continue

        clip_dir = videos_root / desc.source_dataset / desc.video_id / clip_id
        if not all((clip_dir / f).exists() for f in _EXPECTED_FILES):
            incomplete += 1
            continue

        complete += 1
        meta = json.loads((clip_dir / "meta.json").read_text(encoding="utf-8"))
        caption_hash = hashlib.sha256(desc.caption.strip().encode()).hexdigest()[:16]

        entries.append({
            "path": path_key,
            "video_id": desc.video_id,
            "clip_index": idx,
            "source": desc.source_dataset,
            "original_frames": meta.get("original_frames", 81),
            "is_padded": meta.get("is_padded", False),
            "split": "train",
            "text_latent_path": f"latents_cache/{caption_hash}.pt",
        })
        vids_by_src.setdefault(desc.source_dataset, set()).add(desc.video_id)

        if (complete - reused) % 500 == 0 and (complete - reused) > 0:
            print(f"  {complete} complete ({reused} reused, {complete - reused} new) ...", flush=True)

    print(f"[gen_index] {complete} complete ({reused} reused), {incomplete} incomplete")
    return entries, vids_by_src


def _entries_from_scan(data_root: Path, existing_by_path: Dict[str, Dict]) -> tuple:
    """Build entries by scanning videos/ directory."""
    videos_root = data_root / "videos"
    entries: List[Dict] = []
    vids_by_src: Dict[str, set] = {}
    scanned, incomplete, reused = 0, 0, 0

    for source_dir in sorted(videos_root.iterdir()):
        if not source_dir.is_dir():
            continue
        source = source_dir.name
        print(f"[gen_index] Scanning {source} ...", flush=True)
        for video_dir in sorted(source_dir.iterdir()):
            if not video_dir.is_dir():
                continue
            video_id = video_dir.name
            for clip_dir in sorted(video_dir.iterdir()):
                if not clip_dir.is_dir():
                    continue
                scanned += 1
                clip_id = clip_dir.name
                path_key = f"{source}/{video_id}/{clip_id}"

                # Reuse existing entry if already in index.
                if path_key in existing_by_path:
                    entries.append(existing_by_path[path_key])
                    vids_by_src.setdefault(source, set()).add(video_id)
                    reused += 1
                    continue

                if not all((clip_dir / f).exists() for f in _EXPECTED_FILES):
                    incomplete += 1
                    continue

                meta = json.loads((clip_dir / "meta.json").read_text(encoding="utf-8"))
                caption = (clip_dir / "caption.txt").read_text(encoding="utf-8").strip()
                caption_hash = hashlib.sha256(caption.encode()).hexdigest()[:16]

                entries.append({
                    "path": path_key,
                    "video_id": video_id,
                    "clip_index": meta.get("clip_index", 0),
                    "source": source,
                    "original_frames": meta.get("original_frames", 81),
                    "is_padded": meta.get("is_padded", False),
                    "split": "train",
                    "text_latent_path": f"latents_cache/{caption_hash}.pt",
                })
                vids_by_src.setdefault(source, set()).add(video_id)

                if len(entries) % 500 == 0:
                    print(f"  {len(entries)} complete clips ...", flush=True)

    print(f"[gen_index] Scanned {scanned} dirs: {len(entries)} complete "
          f"({reused} reused, {len(entries) - reused} new), {incomplete} incomplete")
    return entries, vids_by_src


def main() -> None:
    p = argparse.ArgumentParser(description="Generate index.json from existing data")
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--out", type=str, default=None,
                   help="Output path (default: {data_root}/index.json)")
    p.add_argument("--collect_cache", type=str, default=None,
                   help="Path to .collect_cache_*.pkl (faster than disk scan)")
    args = p.parse_args()

    data_root = Path(args.data_root)
    out_path = Path(args.out) if args.out else data_root / "index.json"

    # Load existing index to reuse entries (skip disk reads for known clips).
    existing_by_path: Dict[str, Dict] = {}
    if out_path.exists():
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            for c in existing.get("clips", []):
                existing_by_path[str(c["path"])] = c
            print(f"[gen_index] Existing index: {len(existing_by_path)} clips")
        except (json.JSONDecodeError, KeyError):
            pass

    if args.collect_cache:
        cache_path = Path(args.collect_cache)
        if not cache_path.is_absolute():
            cache_path = data_root / cache_path
        entries, vids_by_src = _entries_from_cache(data_root, cache_path, existing_by_path)
    else:
        if not (data_root / "videos").exists():
            print(f"videos/ not found at {data_root / 'videos'}")
            return
        entries, vids_by_src = _entries_from_scan(data_root, existing_by_path)

    if not entries:
        print("[gen_index] No complete clips found.")
        return

    # Merge: preserve clips from existing index that weren't in this scan.
    existing_clips = list(existing_by_path.values())
    new_paths = {str(e["path"]) for e in entries}
    merged = [e for e in existing_clips if str(e["path"]) not in new_paths] + entries

    # Recompute statistics from merged list.
    sources_stats: Dict[str, Dict[str, int]] = {}
    all_vids_by_src: Dict[str, set] = {}
    for e in merged:
        s = str(e["source"])
        all_vids_by_src.setdefault(s, set()).add(str(e["video_id"]))
        sources_stats.setdefault(s, {"videos": 0, "clips": 0})["clips"] += 1
    for s in sources_stats:
        sources_stats[s]["videos"] = len(all_vids_by_src.get(s, set()))

    # Read config from first clip's meta.json
    first_meta_path = data_root / "videos" / merged[0]["path"] / "meta.json"
    m = json.loads(first_meta_path.read_text(encoding="utf-8"))

    index = {
        "version": "2.0",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_name": "Dataset",
        "config": {
            "num_frames": m.get("num_frames", 81),
            "resolution": m.get("resolution", [480, 720]),
            "fps": m.get("fps", 24),
            "pad_mode": "reverse",
            "latents_dir": "latents",
            "xyz_normalize": "percentile_2_98",
        },
        "statistics": {
            "num_videos": len({str(e["video_id"]) for e in merged}),
            "num_clips": len(merged),
            "sources": sources_stats,
        },
        "clips": merged,
    }

    # Atomic write.
    tmp_path = out_path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.rename(out_path)
    print(f"[gen_index] Written {out_path}: {len(merged)} clips "
          f"({len(entries)} new/updated, {len(merged) - len(entries)} preserved)")


if __name__ == "__main__":
    main()
