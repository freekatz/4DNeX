"""OmniWorld-HOI4D raw data adapter.

Expected layout::

    raw_data/omniworld/annotations/OmniWorld-HOI4D/
        omniworld_hoi4d_{start}_{end}/          (batch groups)
            {scene_dir_name}/
                ├── camera/
                │   ├── split_info.json          (contains scene_name for video path)
                │   └── recon/split_0/info.json  (droidclib-format intrinsics + extrinsics)
                ├── prior_depth/XXXXX.png        (5-digit zero-padded, uint16)
                └── text/S_E.txt                 (clip-level captions)

    raw_data/hoi4d/{scene_name}/align_rgb/image.mp4
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import imageio.v2 as imageio
import numpy as np

from core.datasets.adapters.base import (
    VideoClipDescriptor,
    read_video_frames,
    register_loader,
)


def _load_omniworld_hoi4d(params: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Module-level loader for OmniWorld-HOI4D clips (picklable)."""
    rgb_frames = read_video_frames(Path(params["video_path"]), params["frame_ids"])
    depth_frames = np.stack(
        [
            imageio.imread(
                str(Path(params["depth_dir"]) / f"{fid:05d}.png")
            ).astype(np.float32)
            for fid in params["frame_ids"]
        ],
        axis=0,
    )
    return rgb_frames, depth_frames, params["intrinsics"], params["extrinsics"]


register_loader("omniworld_hoi4d", _load_omniworld_hoi4d)


def collect(
    raw_root: Path,
    fps: int = 24,
) -> List[VideoClipDescriptor]:
    """Scan OmniWorld-HOI4D and return :class:`VideoClipDescriptor` instances."""
    ann_root = raw_root / "omniworld" / "annotations" / "OmniWorld-HOI4D"
    hoi4d_root = raw_root / "hoi4d"
    clips: List[VideoClipDescriptor] = []

    if not ann_root.exists():
        return clips

    # Collect scene directories — handle both flat layout and batch-grouped layout.
    scene_dirs: List[Path] = []
    for entry in sorted(ann_root.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name.startswith("omniworld_hoi4d_"):
            scene_dirs.extend(sorted(d for d in entry.iterdir() if d.is_dir()))
        else:
            scene_dirs.append(entry)

    for scene_dir in scene_dirs:
        if not scene_dir.is_dir():
            continue
        scene_dir_name = scene_dir.name
        video_id = scene_dir_name

        split_info_path = scene_dir / "camera" / "split_info.json"
        recon_info_path = scene_dir / "camera" / "recon" / "split_0" / "info.json"
        text_dir = scene_dir / "text"
        depth_dir = scene_dir / "prior_depth"

        if not all(
            p.exists()
            for p in [split_info_path, recon_info_path, text_dir, depth_dir]
        ):
            continue

        with open(split_info_path, "r", encoding="utf-8") as f:
            split_info = json.load(f)
        with open(recon_info_path, "r", encoding="utf-8") as f:
            recon_info = json.load(f)

        # Derive RGB video path from scene_name hierarchy
        scene_name = split_info["scene_name"]
        rgb_video_path = hoi4d_root / scene_name / "align_rgb" / "image.mp4"
        if not rgb_video_path.exists():
            continue

        # Build per-frame look-up from recon info
        split_ids = recon_info["split"]
        exts = np.array(recon_info["extrinsics"], dtype=np.float32)
        ck = recon_info.get("crop_intrinsic", recon_info.get("orig_intrinsic", {}))
        k = np.array(
            [
                [float(ck.get("fx", 1)), 0, float(ck.get("cx", 0))],
                [0, float(ck.get("fy", 1)), float(ck.get("cy", 0))],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )

        intr_map: Dict[int, np.ndarray] = {}
        ext_map: Dict[int, np.ndarray] = {}
        for i, fid in enumerate(split_ids):
            if i < len(exts):
                intr_map[int(fid)] = k
                ext_map[int(fid)] = exts[i]

        # Each TXT file defines a clip by its frame range
        for txt_file in sorted(text_dir.glob("*.txt")):
            stem = txt_file.stem
            if "_" not in stem:
                continue
            s_str, e_str = stem.split("_")
            start, end = int(s_str), int(e_str)
            frame_ids = list(range(start, end + 1))

            with open(txt_file, "r", encoding="utf-8") as f:
                caption = f.read().strip()

            # Validate every frame has camera params + depth
            if not all(
                fid in intr_map
                and fid in ext_map
                and (depth_dir / f"{fid:05d}.png").exists()
                for fid in frame_ids
            ):
                continue

            intrinsics = np.stack([intr_map[fid] for fid in frame_ids], axis=0)
            extrinsics = np.stack([ext_map[fid] for fid in frame_ids], axis=0)

            clips.append(
                VideoClipDescriptor(
                    source_dataset="omniworld_hoi4d",
                    source_entry_id=f"omniworld_hoi4d_{scene_dir_name}_{start}_{end}",
                    video_id=video_id,
                    frame_ids=frame_ids,
                    caption=caption,
                    fps=fps,
                    _load_params={
                        "video_path": str(rgb_video_path),
                        "frame_ids": frame_ids,
                        "depth_dir": str(depth_dir),
                        "intrinsics": intrinsics.copy(),
                        "extrinsics": extrinsics.copy(),
                    },
                )
            )
    return clips
