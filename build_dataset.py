"""Build 4D dataset from a single raw source.

Produces the ``videos/`` tree and ``index.json`` described in
docs/dataset-structure.md.  Latent encoding (``latents/``) is handled
by a separate downstream script.

Run once per source::

    python build_dataset.py --source 4dnex
    python build_dataset.py --source omniworld_game
    python build_dataset.py --source omniworld_hoi4d
"""

import argparse
import hashlib
import json
import pickle
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import imageio.v2 as imageio
import numpy as np

from core.datasets.adapters import (
    VideoClip,
    VideoClipDescriptor,
    collect_4dnex,
    collect_omniworld_game,
    collect_omniworld_hoi4d,
)

# ---------------------------------------------------------------------------
# Geometry / array helpers
# ---------------------------------------------------------------------------


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def resize_and_center_crop(
    frames: np.ndarray,
    out_h: int,
    out_w: int,
    interpolation: int,
) -> np.ndarray:
    """Resize shortest side then center-crop to (out_h, out_w)."""
    t, h, w = frames.shape[:3]
    if h / w < out_h / out_w:
        new_h = out_h
        new_w = int(round(w * out_h / h))
    else:
        new_w = out_w
        new_h = int(round(h * out_w / w))
    resized = np.empty((t, new_h, new_w, frames.shape[3]), dtype=frames.dtype)
    for i in range(t):
        resized[i] = cv2.resize(frames[i], (new_w, new_h), interpolation=interpolation)
    y0 = (new_h - out_h) // 2
    x0 = (new_w - out_w) // 2
    return resized[:, y0:y0 + out_h, x0:x0 + out_w].copy()


def reverse_pad(arr: np.ndarray, target_len: int) -> np.ndarray:
    """Pad by ping-pong (forward, reverse, forward, ...) until *target_len*."""
    n = arr.shape[0]
    if n >= target_len:
        return arr
    if n == 1:
        reps = [target_len] + [1] * (arr.ndim - 1)
        return np.tile(arr, reps)
    cycle = list(range(n)) + list(range(n - 2, 0, -1))
    indices = [cycle[i % len(cycle)] for i in range(target_len)]
    return arr[indices]


def depth_to_world_xyz(
    depth: np.ndarray,
    k: np.ndarray,
    c2w: np.ndarray,
) -> np.ndarray:
    """Back-project a depth map to world coordinates using intrinsics + c2w."""
    h, w = depth.shape[:2]
    u, v = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32),
    )
    z = depth.astype(np.float32)
    x = (u - k[0, 2]) * z / max(float(k[0, 0]), 1e-6)
    y = (v - k[1, 2]) * z / max(float(k[1, 1]), 1e-6)
    cam = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    world = cam @ c2w[:3, :3].T + c2w[:3, 3]
    return world.reshape(h, w, 3)


def build_xyz_sequence(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
) -> np.ndarray:
    """depth [T,H,W] + K [T,3,3] + c2w [T,4,4] → xyz [T,H,W,3]."""
    t, h, w = depth.shape[:3]
    out = np.empty((t, h, w, 3), dtype=np.float32)
    for i in range(t):
        out[i] = depth_to_world_xyz(depth[i], intrinsics[i], extrinsics[i])
    return out


def compute_xyz_norm(
    xyz: np.ndarray,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """Percentile-based normalisation parameters."""
    flat = xyz.reshape(-1, 3)
    p2 = np.percentile(flat, 2, axis=0).astype(np.float32)
    p98 = np.percentile(flat, 98, axis=0).astype(np.float32)
    center = (p2 + p98) / 2.0
    scale = float(np.max((p98 - p2) / 2.0))
    scale = max(scale, 1e-6)
    return center, scale, np.array([2, 98], dtype=np.int32)


def adjust_intrinsics_for_resize_crop(
    K: np.ndarray,
    src_h: int,
    src_w: int,
    out_h: int,
    out_w: int,
) -> np.ndarray:
    """Adjust [T,3,3] intrinsics for the same resize-and-center-crop transform."""
    if src_h / src_w < out_h / out_w:
        new_h = out_h
        new_w = int(round(src_w * out_h / src_h))
    else:
        new_w = out_w
        new_h = int(round(src_h * out_w / src_w))
    sx = new_w / src_w
    sy = new_h / src_h
    x0 = (new_w - out_w) // 2
    y0 = (new_h - out_h) // 2
    K_adj = K.copy()
    K_adj[:, 0, 0] *= sx
    K_adj[:, 1, 1] *= sy
    K_adj[:, 0, 2] = K_adj[:, 0, 2] * sx - x0
    K_adj[:, 1, 2] = K_adj[:, 1, 2] * sy - y0
    return K_adj


# ---------------------------------------------------------------------------
# Clip processing & writing
# ---------------------------------------------------------------------------


def process_clip(
    clip: VideoClip,
    out_h: int,
    out_w: int,
    num_frames: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict, int, bool]:
    """Process a single clip into target-resolution videos + metadata."""
    rgb = clip.rgb_frames
    depth = clip.depth_frames
    intr = clip.intrinsics
    ext = clip.extrinsics_c2w

    t_valid = min(rgb.shape[0], depth.shape[0], intr.shape[0], ext.shape[0])
    rgb, depth, intr, ext = rgb[:t_valid], depth[:t_valid], intr[:t_valid], ext[:t_valid]
    original_frames = int(t_valid)
    is_padded = t_valid < num_frames

    rgb = reverse_pad(rgb, num_frames)
    depth = reverse_pad(depth, num_frames)
    intr = reverse_pad(intr, num_frames)
    ext = reverse_pad(ext, num_frames)

    depth_h, depth_w = depth.shape[1], depth.shape[2]
    xyz = build_xyz_sequence(depth, intr, ext)
    del depth

    w2f = np.linalg.inv(ext[0]).astype(np.float32)
    xyz = np.einsum("thwc,dc->thwd", xyz, w2f[:3, :3], optimize=True)
    xyz += w2f[:3, 3].reshape(1, 1, 1, 3)
    ext = np.einsum("ij,tjk->tik", w2f, ext, optimize=True)

    intr_adj = adjust_intrinsics_for_resize_crop(intr, depth_h, depth_w, out_h, out_w)

    rgb = resize_and_center_crop(rgb, out_h, out_w, cv2.INTER_LINEAR)
    xyz = resize_and_center_crop(xyz, out_h, out_w, cv2.INTER_NEAREST)

    center, scale, percentile = compute_xyz_norm(xyz)
    np.subtract(xyz, center.reshape(1, 1, 1, 3), out=xyz)
    xyz /= np.float32(scale)
    np.clip(xyz, -1.0, 1.0, out=xyz)
    xyz_vis = np.clip((xyz + np.float32(1.0)) * np.float32(127.5), 0, 255).astype(np.uint8)
    del xyz

    meta_extra = {
        "camera": {
            "intrinsics": intr_adj.tolist(),
            "extrinsics_c2w": ext.tolist(),
        },
        "xyz_norm": {
            "center": center.tolist(),
            "scale": float(scale),
            "percentile": percentile.tolist(),
        },
    }
    return rgb, xyz_vis, intr_adj, meta_extra, original_frames, is_padded


_CLIP_EXPECTED_FILES = ["video.mp4", "xyz.mp4", "first_frame.png", "caption.txt", "meta.json"]


def _copy_with_retry(src: Path, dst: Path, attempts: int = 5) -> None:
    for attempt in range(attempts):
        try:
            shutil.copy2(src, dst)
            return
        except OSError:
            if attempt == attempts - 1:
                raise
            time.sleep(2 ** attempt)


def write_clip_files(
    out_root: Path,
    clip: VideoClipDescriptor,
    clip_id: str,
    clip_index: int,
    split: str,
    out_h: int,
    out_w: int,
    num_frames: int,
    skip_existing: bool = True,
    staging_root: Path | None = None,
) -> Tuple[Dict[str, object], bool]:
    """Write one clip to disk and return ``(index_entry, skipped)``."""
    rel_path = Path("videos") / clip.source_dataset / clip.video_id / clip_id
    final_dir = out_root / rel_path

    entry_base = {
        "path": f"{clip.source_dataset}/{clip.video_id}/{clip_id}",
        "video_id": clip.video_id,
        "clip_index": clip_index,
        "source": clip.source_dataset,
        "split": split,
    }

    # Fast path: already complete.
    if skip_existing and all((final_dir / f).exists() for f in _CLIP_EXPECTED_FILES):
        meta = json.loads((final_dir / "meta.json").read_text(encoding="utf-8"))
        return {
            **entry_base,
            "original_frames": meta.get("original_frames", num_frames),
            "is_padded": meta.get("is_padded", False),
        }, True

    # Write to local staging dir (if set) to avoid FUSE seek issues.
    write_dir = (staging_root / rel_path) if staging_root else final_dir
    _ensure_dir(write_dir)
    if staging_root:
        _ensure_dir(final_dir)

    loaded_clip = clip.load()
    rgb, xyz_vis, _intr_adj, extra, original_frames, is_padded = process_clip(
        loaded_clip, out_h, out_w, num_frames,
    )
    del loaded_clip

    rgb_u8 = rgb if rgb.dtype == np.uint8 else rgb.astype(np.uint8)
    imageio.mimwrite(write_dir / "video.mp4", list(rgb_u8), fps=clip.fps)
    imageio.mimwrite(write_dir / "xyz.mp4", list(xyz_vis), fps=clip.fps)
    imageio.imwrite(write_dir / "first_frame.png", rgb_u8[0])
    del rgb, rgb_u8, xyz_vis

    (write_dir / "caption.txt").write_text(clip.caption.strip(), encoding="utf-8")

    last_idx = min(len(clip.frame_ids) - 1, num_frames - 1)
    meta = {
        "video_id": clip.video_id,
        "clip_id": clip_id,
        "clip_index": clip_index,
        "source_dataset": clip.source_dataset,
        "source_entry_id": clip.source_entry_id,
        "num_frames": num_frames,
        "original_frames": original_frames,
        "fps": int(clip.fps),
        "resolution": [out_h, out_w],
        "frame_range_in_source": [int(clip.frame_ids[0]), int(clip.frame_ids[last_idx])],
        "is_padded": is_padded,
        **extra,
    }
    (write_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False), encoding="utf-8",
    )

    if staging_root:
        for fname in _CLIP_EXPECTED_FILES:
            _copy_with_retry(write_dir / fname, final_dir / fname)
        shutil.rmtree(write_dir, ignore_errors=True)

    return {
        **entry_base,
        "original_frames": original_frames,
        "is_padded": is_padded,
    }, False


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------


def write_index(
    out_root: Path,
    args: argparse.Namespace,
    entries: List[Dict[str, object]],
) -> None:
    sources: Dict[str, Dict[str, int]] = {}
    vids_by_src: Dict[str, set] = {}
    for e in entries:
        s = str(e["source"])
        vids_by_src.setdefault(s, set()).add(str(e["video_id"]))
        sources.setdefault(s, {"videos": 0, "clips": 0})["clips"] += 1
    for s in sources:
        sources[s]["videos"] = len(vids_by_src.get(s, set()))

    # Merge with existing index.json if present (preserves clips from other sources).
    index_path = out_root / "index.json"
    existing_clips: List[Dict[str, object]] = []
    if index_path.exists():
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            existing_clips = existing.get("clips", [])
        except (json.JSONDecodeError, KeyError):
            pass

    # Deduplicate: new entries override existing ones with same path.
    new_paths = {str(e["path"]) for e in entries}
    merged = [e for e in existing_clips if str(e["path"]) not in new_paths] + entries

    # Recompute statistics from merged list.
    sources = {}
    vids_by_src = {}
    for e in merged:
        s = str(e["source"])
        vids_by_src.setdefault(s, set()).add(str(e["video_id"]))
        sources.setdefault(s, {"videos": 0, "clips": 0})["clips"] += 1
    for s in sources:
        sources[s]["videos"] = len(vids_by_src.get(s, set()))

    index = {
        "version": "2.0",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_name": "Dataset",
        "config": {
            "num_frames": args.num_frames,
            "resolution": [args.resolution_h, args.resolution_w],
            "fps": args.fps,
            "pad_mode": "reverse",
            "latents_dir": "latents",
            "xyz_normalize": "percentile_2_98",
        },
        "statistics": {
            "num_videos": len({str(e["video_id"]) for e in merged}),
            "num_clips": len(merged),
            "sources": sources,
        },
        "clips": merged,
    }

    # Atomic write: write to tmp file then rename, so a crash never corrupts index.json.
    tmp_path = out_root / ".index.json.tmp"
    tmp_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.rename(index_path)


# ---------------------------------------------------------------------------
# Source registry
# ---------------------------------------------------------------------------

_COLLECT_FNS = {
    "4dnex": lambda raw_root, args: collect_4dnex(
        raw_root, fps=args.fps, num_frames=args.num_frames, stride=args.stride,
    ),
    "omniworld_game": lambda raw_root, args: collect_omniworld_game(
        raw_root, fps=args.fps,
    ),
    "omniworld_hoi4d": lambda raw_root, args: collect_omniworld_hoi4d(
        raw_root, fps=args.fps,
    ),
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build 4D dataset from a single raw source")
    p.add_argument("--raw_root", type=str, default="./raw_data")
    p.add_argument("--out", type=str, default="./data")
    p.add_argument("--num_frames", type=int, default=81)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--resolution_h", type=int, default=480)
    p.add_argument("--resolution_w", type=int, default=720)
    p.add_argument("--stride", type=int, default=40)
    p.add_argument(
        "--source", required=True,
        choices=list(_COLLECT_FNS.keys()),
        help="Which data source to process (run once per source)",
    )
    p.add_argument("--split", type=str, default="train",
                   help="Split label written into index.json entries (default: train)")
    p.add_argument("--no_skip_existing", action="store_true",
                   help="Re-process clips even when output files already exist")
    p.add_argument("--recollect", action="store_true",
                   help="Force re-scan raw data instead of using cached collect results")
    p.add_argument("--tmp_dir", type=str, default=None,
                   help="Local staging directory for ffmpeg writes; files are copied "
                        "to --out afterwards (required when --out is a FUSE mount)")
    return p.parse_args()


def _fmt_elapsed(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


def _fmt_eta(elapsed: float, done: int, total: int) -> str:
    if done == 0:
        return "?"
    return _fmt_elapsed(elapsed / done * (total - done))


def main() -> None:
    args = parse_args()
    raw_root = Path(args.raw_root)
    out_root = Path(args.out)
    _ensure_dir(out_root / "videos")

    skip_existing = not args.no_skip_existing
    staging_root = Path(args.tmp_dir) if args.tmp_dir else None

    # ---- config summary ----
    print(f"[build_dataset] source={args.source}  raw_root={raw_root}  out={out_root}")
    if staging_root:
        print(f"  tmp_dir={staging_root}  (staging -> copy to out)")
    print(f"  resolution={args.resolution_h}x{args.resolution_w}  "
          f"num_frames={args.num_frames}  fps={args.fps}  stride={args.stride}")
    print(f"  split={args.split}  skip_existing={skip_existing}")

    # ---- collect (with cache) ----
    t0 = time.time()
    cache_key = json.dumps({
        "source": args.source,
        "raw_root": str(raw_root.resolve()),
        "fps": args.fps,
        "num_frames": args.num_frames,
        "stride": args.stride,
    }, sort_keys=True)
    cache_hash = hashlib.sha256(cache_key.encode()).hexdigest()[:16]
    cache_path = out_root / f".collect_cache_{args.source}_{cache_hash}.pkl"

    descriptors: List[VideoClipDescriptor] = []
    if not args.recollect and cache_path.exists():
        print(f"[collect] Loading cached descriptors from {cache_path.name} ...")
        with open(cache_path, "rb") as f:
            descriptors = pickle.load(f)
        n_videos = len({d.video_id for d in descriptors})
        print(f"[collect] {len(descriptors)} clips from {n_videos} videos (cached)")
    else:
        print(f"[collect] Scanning {args.source} ...")
        descriptors = _COLLECT_FNS[args.source](raw_root, args)
        if descriptors:
            _ensure_dir(out_root)
            with open(cache_path, "wb") as f:
                pickle.dump(descriptors, f, protocol=pickle.HIGHEST_PROTOCOL)
        n_videos = len({d.video_id for d in descriptors})
        print(f"[collect] {len(descriptors)} clips from {n_videos} videos "
              f"({_fmt_elapsed(time.time() - t0)}, saved to {cache_path.name})")

    if not descriptors:
        print("[collect] No valid clips found. Check --raw_root and --source.")
        return

    # ---- assign clip indices ----
    clip_counter: Dict[str, int] = {}
    work_items: list = []
    for desc in descriptors:
        idx = clip_counter.get(desc.video_id, 0)
        clip_counter[desc.video_id] = idx + 1
        work_items.append((desc, f"clip_{idx}", idx))
    del descriptors

    # ---- build (serial) ----
    entries: List[Dict[str, object]] = []
    processed, skipped = 0, 0
    total = len(work_items)
    t_build = time.time()

    errors = 0
    print(f"[build] Processing {total} clips ...")
    for i, (desc, clip_id, clip_idx) in enumerate(work_items):
        done = i + 1
        clip_path = f"{desc.source_dataset}/{desc.video_id}/{clip_id}"

        # Retry on transient I/O errors (FUSE mounts can be flaky).
        last_err = None
        for attempt in range(3):
            try:
                entry, was_skipped = write_clip_files(
                    out_root, desc,
                    clip_id=clip_id,
                    clip_index=clip_idx,
                    split=args.split,
                    out_h=args.resolution_h,
                    out_w=args.resolution_w,
                    num_frames=args.num_frames,
                    skip_existing=skip_existing,
                    staging_root=staging_root,
                )
                last_err = None
                break
            except Exception as e:
                last_err = e
                if attempt < 2:
                    print(f"  [{done}/{total}] RETRY {clip_path} (attempt {attempt+1}): {e}")
                    time.sleep(2 ** attempt)

        elapsed = time.time() - t_build
        eta = _fmt_eta(elapsed, done, total)
        if last_err is not None:
            errors += 1
            print(f"  [{done}/{total}] ERROR {clip_path}: {last_err}  "
                  f"({_fmt_elapsed(elapsed)}, ETA {eta})")
            continue

        entries.append(entry)
        if was_skipped:
            skipped += 1
            print(f"  [{done}/{total}] skip  {entry['path']}  "
                  f"({_fmt_elapsed(elapsed)}, ETA {eta})")
        else:
            processed += 1
            print(f"  [{done}/{total}] built {entry['path']}  "
                  f"({_fmt_elapsed(elapsed)}, ETA {eta})")

    write_index(out_root, args, entries)
    total_time = _fmt_elapsed(time.time() - t0)
    print(f"[build] Done in {total_time}. "
          f"{len(entries)} clips (built: {processed}, skipped: {skipped}, errors: {errors}) "
          f"-> {out_root}/index.json")


if __name__ == "__main__":
    main()
