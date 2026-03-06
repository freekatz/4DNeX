"""OmniWorld-Game raw data adapter.

Expected layout::

    raw_data/omniworld/
    ├── annotations/OmniWorld-Game/{scene_id}/{scene_id}_others/
    │       ├── droidclib/split_N.json
    │       ├── text/SSSSSS_EEEEEE.json
    │       └── ...
    ├── annotations/OmniWorld-Game/{scene_id}/{scene_id}_depth_0000/depth/
    └── videos/OmniWorld-Game/{scene_id}/{scene_id}_rgb_0000/color/
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import imageio.v2 as imageio
import numpy as np

from core.datasets.adapters.base import VideoClipDescriptor, register_loader


def _load_omniworld_game(params: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Module-level loader for OmniWorld-Game clips (picklable)."""
    rgb_frames = np.stack(
        [imageio.imread(p).astype(np.uint8) for p in params["rgb_paths"]], axis=0,
    )
    depth_frames = np.stack(
        [imageio.imread(p).astype(np.float32) for p in params["depth_paths"]], axis=0,
    )
    return rgb_frames, depth_frames, params["intrinsics"], params["extrinsics"]


register_loader("omniworld_game", _load_omniworld_game)


def _choose_depth_path(
    depth_dir: Path,
    frame_id: int,
) -> Optional[Path]:
    """Find the depth file for *frame_id*, falling back to nearest available."""
    candidate = depth_dir / f"{frame_id:06d}.png"
    if candidate.exists():
        return candidate
    if not depth_dir.exists():
        return None
    available = sorted(
        int(p.stem) for p in depth_dir.glob("*.png") if p.stem.isdigit()
    )
    if not available:
        return None
    nearest_id = min(available, key=lambda x: abs(x - frame_id))
    return depth_dir / f"{nearest_id:06d}.png"


def collect(
    raw_root: Path,
    fps: int = 24,
) -> List[VideoClipDescriptor]:
    """Scan OmniWorld-Game and return :class:`VideoClipDescriptor` instances."""
    ann_root = raw_root / "omniworld" / "annotations" / "OmniWorld-Game"
    rgb_root = raw_root / "omniworld" / "videos" / "OmniWorld-Game"
    clips: List[VideoClipDescriptor] = []

    if not ann_root.exists():
        return clips

    for scene_dir in sorted(ann_root.glob("*")):
        if not scene_dir.is_dir():
            continue
        scene_id = scene_dir.name
        video_id = scene_id
        others = scene_dir / f"{scene_id}_others"
        depth_dir = scene_dir / f"{scene_id}_depth_0000" / "depth"
        color_dir = rgb_root / scene_id / f"{scene_id}_rgb_0000" / "color"
        droidclib_dir = others / "droidclib"
        text_dir = others / "text"
        if not droidclib_dir.exists() or not text_dir.exists() or not color_dir.exists():
            continue

        # Build per-frame intrinsics / extrinsics look-up from all splits
        intr_map: Dict[int, np.ndarray] = {}
        ext_map: Dict[int, np.ndarray] = {}

        for split_file in sorted(droidclib_dir.glob("split_*.json")):
            with open(split_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            split_ids = data["split"]
            exts = np.array(data["extrinsics"], dtype=np.float32)
            ck = data.get("crop_intrinsic", data.get("orig_intrinsic", {}))
            k = np.array(
                [
                    [float(ck.get("fx", 1)), 0, float(ck.get("cx", 0))],
                    [0, float(ck.get("fy", 1)), float(ck.get("cy", 0))],
                    [0, 0, 1],
                ],
                dtype=np.float32,
            )
            for i, fid in enumerate(split_ids):
                if i < len(exts):
                    intr_map[int(fid)] = k
                    ext_map[int(fid)] = exts[i]

        # Each caption JSON defines a clip by its frame range
        for cap_file in sorted(text_dir.glob("*.json")):
            stem = cap_file.stem
            if "_" not in stem:
                continue
            s_str, e_str = stem.split("_")
            start, end = int(s_str), int(e_str)
            frame_ids = list(range(start, end + 1))

            with open(cap_file, "r", encoding="utf-8") as f:
                cap_data = json.load(f)
            captions = cap_data.get("captions", {})
            caption = captions.get(
                "Video_Caption", captions.get("Short_Caption", ""),
            ).strip()

            # Validate every frame has camera params + depth (lightweight check)
            rgb_paths = []
            depth_paths = []
            intrs = []
            exts_list = []
            valid = True
            for fid in frame_ids:
                rgb_path = color_dir / f"{fid:06d}.png"
                d_path = _choose_depth_path(depth_dir, fid)
                if (
                    not rgb_path.exists()
                    or d_path is None
                    or fid not in intr_map
                    or fid not in ext_map
                ):
                    valid = False
                    break
                rgb_paths.append(rgb_path)
                depth_paths.append(d_path)
                intrs.append(intr_map[fid])
                exts_list.append(ext_map[fid])

            if not valid:
                continue

            clips.append(
                VideoClipDescriptor(
                    source_dataset="omniworld_game",
                    source_entry_id=f"omniworld_game_{scene_id}_{start:06d}_{end:06d}",
                    video_id=video_id,
                    frame_ids=frame_ids,
                    caption=caption,
                    fps=fps,
                    _load_params={
                        "rgb_paths": [str(p) for p in rgb_paths],
                        "depth_paths": [str(p) for p in depth_paths],
                        "intrinsics": np.stack(intrs, axis=0),
                        "extrinsics": np.stack(exts_list, axis=0),
                    },
                )
            )

    return clips
