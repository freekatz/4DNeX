"""4DNeX-10M raw data adapter.

Expected layout under ``raw_root / "4dnex"``::

    4dnex/
    ├── caption/
    │   ├── dynamic1.csv
    │   ├── dynamic2.csv
    │   └── ...
    ├── dynamic1/{video_id}/clip_{start}-{end}/
    │   ├── pred_traj.txt
    │   ├── pred_intrinsics.txt
    │   └── frame_XXXX.npy
    ├── dynamic2/{video_id}/clip_{start}-{end}/
    │   └── ...
    └── raw/
        ├── dynamic1/{video_id}.mp4
        ├── dynamic2/{video_id}.mp4
        └── ...
"""

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from core.datasets.adapters.base import (
    VideoClipDescriptor,
    make_windows,
    quat_wxyz_to_c2w,
    read_video_frames,
    register_loader,
)


def _load_4dnex(params: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Module-level loader for 4DNeX clips (picklable)."""
    rgbs = read_video_frames(Path(params["video_path"]), params["frame_ids"])
    depths = np.stack(
        [np.load(p).astype(np.float32) for p in params["npy_paths"]],
        axis=0,
    )
    return rgbs, depths, params["intrinsics"], params["extrinsics"]


register_loader("4dnex", _load_4dnex)


def _parse_caption_csv(csv_path: Path) -> Dict[str, str]:
    """Parse a caption CSV → ``{zero-padded 8-digit id: caption}``."""
    mapping: Dict[str, str] = {}
    if not csv_path.exists():
        return mapping
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            number = str(row.get("number", "")).strip()
            caption = str(row.get("caption", "")).strip()
            if number:
                mapping[number.zfill(8)] = caption
    return mapping


def collect(
    raw_root: Path,
    fps: int = 24,
    num_frames: int = 81,
    stride: int = 40,
) -> List[VideoClipDescriptor]:
    """Scan ``raw_data/4dnex/`` and return :class:`VideoClipDescriptor` instances.

    Frame data is NOT loaded here — only paths and metadata are recorded.
    """
    base = raw_root / "4dnex"
    clips: List[VideoClipDescriptor] = []

    # Find all dynamicN subdirectories
    dynamic_dirs = sorted(
        d for d in base.glob("dynamic*")
        if d.is_dir() and d.name != "dynamic"  # skip if bare "dynamic" exists alongside dynamicN
    )
    # Fallback: support legacy single "dynamic" directory
    if not dynamic_dirs:
        legacy = base / "dynamic"
        if legacy.is_dir():
            dynamic_dirs = [legacy]

    for dynamic_dir in dynamic_dirs:
        subdir_name = dynamic_dir.name  # e.g. "dynamic1"

        # Load caption CSV for this subdirectory
        cap_map = _parse_caption_csv(base / "caption" / f"{subdir_name}.csv")

        for video_dir in sorted(dynamic_dir.glob("*")):
            if not video_dir.is_dir():
                continue
            src_video_id = video_dir.name  # e.g. "00000028"
            video_path = base / "raw" / subdir_name / f"{src_video_id}.mp4"
            if not video_path.exists():
                continue
            caption = cap_map.get(src_video_id, "")

            for clip_dir in sorted(video_dir.glob("clip_*")):
                if not clip_dir.is_dir():
                    continue
                frame_npys = sorted(clip_dir.glob("frame_*.npy"))
                if not frame_npys:
                    continue

                # Parse clip range from directory name, e.g. "clip_000000-000053"
                parts = clip_dir.name.replace("clip_", "").split("-")
                if len(parts) != 2:
                    continue
                start_src = int(parts[0])
                end_src = int(parts[1])

                traj_path = clip_dir / "pred_traj.txt"
                intr_path = clip_dir / "pred_intrinsics.txt"
                if not traj_path.exists() or not intr_path.exists():
                    continue

                # Only load lightweight trajectory/intrinsic metadata to
                # determine clip length and window splits — NOT the heavy
                # depth/rgb frames.
                traj = np.loadtxt(traj_path, dtype=np.float32)
                intr = np.loadtxt(intr_path, dtype=np.float32).reshape(-1, 3, 3)

                clip_len = end_src - start_src + 1
                c2w_list = []
                for i in range(min(clip_len, traj.shape[0])):
                    _, tx, ty, tz, qw, qx, qy, qz = traj[i].tolist()
                    c2w_list.append(quat_wxyz_to_c2w(tx, ty, tz, qw, qx, qy, qz))
                c2w = np.stack(c2w_list, axis=0)

                t_valid = min(len(frame_npys), intr.shape[0], c2w.shape[0])
                intr = intr[:t_valid]
                c2w = c2w[:t_valid]
                all_frame_ids = list(range(start_src, start_src + t_valid))
                valid_npys = frame_npys[:t_valid]

                for ws, we in make_windows(t_valid, num_frames, stride):
                    win_frame_ids = all_frame_ids[ws:we]

                    clips.append(
                        VideoClipDescriptor(
                            source_dataset="4dnex",
                            source_entry_id=(
                                f"4dnex_{src_video_id}"
                                f"_{start_src + ws:06d}_{start_src + we - 1:06d}"
                            ),
                            video_id=src_video_id,
                            frame_ids=win_frame_ids,
                            caption=caption,
                            fps=fps,
                            _load_params={
                                "video_path": str(video_path),
                                "frame_ids": win_frame_ids,
                                "npy_paths": [str(p) for p in valid_npys[ws:we]],
                                "intrinsics": intr[ws:we].copy(),
                                "extrinsics": c2w[ws:we].copy(),
                            },
                        )
                    )
    return clips
