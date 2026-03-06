"""Clip dataclass and shared adapter utilities."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import imageio.v2 as imageio
import numpy as np
from scipy.spatial.transform import Rotation

# ---------------------------------------------------------------------------
# Loader registry — allows picklable dispatch for multiprocessing
# ---------------------------------------------------------------------------

_LOADERS: Dict[str, Callable[[dict], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]] = {}


def register_loader(source_name: str, fn: Callable) -> None:
    """Register a module-level loader function for *source_name*."""
    _LOADERS[source_name] = fn


def _dispatch_load(
    source_name: str, params: dict,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return _LOADERS[source_name](params)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class VideoClip:
    """Intermediate representation produced by every adapter.

    Fields
    ------
    source_dataset   e.g. "4dnex", "omniworld_game", "omniworld_hoi4d"
    source_entry_id  unique id traceable back to raw data
    video_id         target video_id in the output tree
    frame_ids        original frame indices in the source
    rgb_frames       [T, H_rgb, W_rgb, 3]  uint8
    depth_frames     [T, H_depth, W_depth]  float32
    intrinsics       [T, 3, 3]  float32, calibrated to depth resolution
    extrinsics_c2w   [T, 4, 4]  float32, camera-to-world
    caption          text description
    fps              frames per second
    """

    source_dataset: str
    source_entry_id: str
    video_id: str
    frame_ids: List[int]
    rgb_frames: np.ndarray
    depth_frames: np.ndarray
    intrinsics: np.ndarray
    extrinsics_c2w: np.ndarray
    caption: str
    fps: int


@dataclass
class VideoClipDescriptor:
    """Lightweight descriptor that defers heavy data loading.

    Stores only metadata and picklable load parameters.  Call :meth:`load`
    to materialise the full :class:`VideoClip` (one at a time to avoid OOM).
    """

    source_dataset: str
    source_entry_id: str
    video_id: str
    frame_ids: List[int]
    caption: str
    fps: int
    _load_params: dict = field(default_factory=dict)

    def load(self) -> "VideoClip":
        """Load frame data and return a materialised :class:`VideoClip`."""
        rgb, depth, intr, ext = _dispatch_load(self.source_dataset, self._load_params)
        return VideoClip(
            source_dataset=self.source_dataset,
            source_entry_id=self.source_entry_id,
            video_id=self.video_id,
            frame_ids=self.frame_ids,
            rgb_frames=rgb,
            depth_frames=depth,
            intrinsics=intr,
            extrinsics_c2w=ext,
            caption=self.caption,
            fps=self.fps,
        )


# ---------------------------------------------------------------------------
# Shared utilities used by multiple adapters
# ---------------------------------------------------------------------------


def read_video_frames(video_path: Path, frame_ids: Sequence[int]) -> np.ndarray:
    """Read specific frames from a video file by index.

    Returns [T, H, W, 3] uint8.
    """
    reader = imageio.get_reader(str(video_path))
    frames = [reader.get_data(int(fid)).astype(np.uint8) for fid in frame_ids]
    reader.close()
    return np.stack(frames, axis=0)


def quat_wxyz_to_c2w(
    tx: float,
    ty: float,
    tz: float,
    qw: float,
    qx: float,
    qy: float,
    qz: float,
) -> np.ndarray:
    """Convert translation + wxyz quaternion to a 4x4 camera-to-world matrix."""
    # scipy expects (x, y, z, w) ordering
    r = Rotation.from_quat([qx, qy, qz, qw]).as_matrix().astype(np.float32)
    m = np.eye(4, dtype=np.float32)
    m[:3, :3] = r
    m[:3, 3] = [tx, ty, tz]
    return m


def make_windows(
    length: int,
    num_frames: int,
    stride: int,
) -> List[Tuple[int, int]]:
    """Return (start, end) windows for slicing a sequence into fixed-length clips.

    Short sequences produce a single window ``(0, length)`` which the
    builder will reverse-pad later.
    """
    if length < num_frames:
        return [(0, length)]
    windows: List[Tuple[int, int]] = []
    start = 0
    while start + num_frames <= length:
        windows.append((start, start + num_frames))
        start += stride
    # ensure the tail is covered
    if windows[-1][1] < length:
        windows.append((length - num_frames, length))
    return windows
