"""Dataset for Wan I2V training on the dataset structure.

Reads pre-computed latents produced by ``process_dataset.py`` and the
``videos/`` tree produced by ``build_dataset.py``.  All heavy encoding
(VAE / CLIP / UMT5) is expected to have been done offline; this loader
simply loads ``.pt`` files and applies training-time normalisation.

Directory layout expected under ``data_root``::

    data/
    ├── index.json
    ├── videos/{source}/{video_id}/{clip_id}/
    │   ├── video.mp4, xyz.mp4, first_frame.png, caption.txt, meta.json
    ├── latents/{source}/{video_id}/{clip_id}/
    │   ├── rgb_latent.pt, xyz_latent.pt, visual_embeds.pt
    └── latents_cache/
        └── {caption_hash}.pt          # {"text_embeds": ..., "text_ids": ...}
"""

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from typing_extensions import override

from core.constants import LOG_LEVEL, LOG_NAME
from .utils import generate_uniform_pointmap, preprocess_image_with_resize

if TYPE_CHECKING:
    from core.trainer import Trainer

logger = logging.getLogger(LOG_NAME)
logger.setLevel(LOG_LEVEL)


# ---------------------------------------------------------------------------
# Latent normalisation constants (XYZ pointmap channel)
# ---------------------------------------------------------------------------
ENCODED_PM_MEAN = -0.13
ENCODED_PM_STD = 1.70

# ---------------------------------------------------------------------------
# !!! TEMP HACK (MEMORY DEBUG) !!!
# Keep only first 13 latent frames to match refer behavior
# (roughly corresponds to 49 real frames before VAE temporal compression).
# ---------------------------------------------------------------------------
HACK_FORCE_LATENT_FRAMES = 13


class BaseDataset(Dataset):
    """Base dataset that reads from the ``index.json`` structure.

    Each sample corresponds to a *clip* entry in ``index.json``.  All
    latent tensors are expected to already exist on disk (written by
    ``process_dataset.py``).

    Args:
        data_root:   Root of the dataset (contains ``index.json``).
        device:      Torch device (unused — data is loaded to CPU).
        trainer:     Reference to the Trainer (used for resolution info).
        index_file:  Optional path to a custom index JSON (e.g. a subset
                     for overfitting).  Defaults to ``data_root/index.json``.
    """

    # Per-clip files that must exist under ``latents/{clip_path}/``
    REQUIRED_LATENT_FILES = [
        "rgb_latent.pt",
        "xyz_latent.pt",
        "visual_embeds.pt",
    ]

    def __init__(
        self,
        data_root: str | Path,
        device: torch.device,
        trainer: "Trainer" = None,
        index_file: str | Path | None = None,
        **kwargs,
    ) -> None:
        super().__init__()

        data_root = Path(data_root)
        self.data_root = data_root
        self.trainer = trainer
        self.device = device

        # ---- load clip list from index.json ----
        index_path = Path(index_file) if index_file is not None else data_root / "index.json"
        if not index_path.exists():
            raise FileNotFoundError(
                f"index.json not found at {index_path}. "
                "Run build_dataset.py + process_dataset.py first."
            )
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)

        self.clips: List[dict] = index["clips"]
        config = index.get("config", {})
        self.latents_dir_name: str = config.get("latents_dir", "latents")

        self.videos_root = data_root / "videos"
        self.latents_root = data_root / self.latents_dir_name

        # Uniform pointmap for conditioning image padding
        train_res = self.trainer.args.train_resolution
        uniform_pm = torch.from_numpy(
            generate_uniform_pointmap(train_res[1], train_res[2])
        ).permute(2, 0, 1)  # [3, H, W]
        self.uniform_pointmap = uniform_pm * 2 - 1  # → [-1, 1]

        # Validate that all required latent files exist
        missing = []
        for clip in self.clips:
            lat_dir = self.latents_root / clip["path"]
            for fname in self.REQUIRED_LATENT_FILES:
                if not (lat_dir / fname).exists():
                    missing.append(str(lat_dir / fname))
            # Text latent is stored in latents_cache/ (shared by caption hash)
            text_path = clip.get("text_latent_path")
            if text_path and not (data_root / text_path).exists():
                missing.append(str(data_root / text_path))
        if missing:
            n = len(missing)
            preview = missing[:5]
            raise FileNotFoundError(
                f"{n} latent file(s) missing. Run process_dataset.py first. "
                f"Examples: {preview}"
            )

        logger.info(f"Loaded {len(self.clips)} clips from {index_path}")

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        for _ in range(len(self.clips)):
            try:
                return self.getitem(index)
            except Exception:
                index = (index + 1) % len(self.clips)
        raise RuntimeError("All clips failed to load")

    def getitem(self, index: int) -> Dict[str, Any]:
        clip = self.clips[index]
        clip_path = clip["path"]
        lat_dir = self.latents_root / clip_path
        vid_dir = self.videos_root / clip_path

        # ---- Load pre-computed latents ----
        encoded_video = torch.load(lat_dir / "rgb_latent.pt", map_location="cpu", weights_only=True)
        encoded_pm = torch.load(lat_dir / "xyz_latent.pt", map_location="cpu", weights_only=True)
        image_embedding = torch.load(lat_dir / "visual_embeds.pt", map_location="cpu", weights_only=True)

        # Text latent from shared cache (latents_cache/{hash}.pt)
        text_data = torch.load(
            self.data_root / clip["text_latent_path"], map_location="cpu", weights_only=True,
        )
        prompt_embedding = text_data["text_embeds"]

        # Ensure RGB and XYZ latents have matching temporal dimensions
        num_frames = min(encoded_video.shape[1], encoded_pm.shape[1])
        encoded_video = encoded_video[:, :num_frames, :, :]
        encoded_pm = encoded_pm[:, :num_frames, :, :]

        # !!! TEMP HACK (MEMORY DEBUG): force 13 latent frames like refer !!!
        # This intentionally changes training behavior to reduce memory usage.
        if num_frames < HACK_FORCE_LATENT_FRAMES:
            raise ValueError(
                f"Insufficient latent frames: got {num_frames}, "
                f"need >= {HACK_FORCE_LATENT_FRAMES} for temporary memory-debug hack"
            )
        if num_frames != HACK_FORCE_LATENT_FRAMES:
            logger.warning(
                "[TEMP HACK ENABLED] Forcing latent frames from %s to %s to match refer memory profile.",
                num_frames,
                HACK_FORCE_LATENT_FRAMES,
            )
        encoded_video = encoded_video[:, :HACK_FORCE_LATENT_FRAMES, :, :]
        encoded_pm = encoded_pm[:, :HACK_FORCE_LATENT_FRAMES, :, :]

        # ---- XYZ training normalisation ----
        encoded_pm = (encoded_pm - ENCODED_PM_MEAN) / ENCODED_PM_STD

        # ---- Concatenate RGB + XYZ latents along width (last dim) ----
        encoded_video = torch.cat([encoded_video, encoded_pm], dim=-1)

        # ---- Conditioning image (first frame + uniform pointmap) ----
        first_frame_path = vid_dir / "first_frame.png"
        _, image = self.preprocess(None, first_frame_path)
        image = self.image_transform(image)
        image = torch.cat([image, self.uniform_pointmap], dim=-1)

        return {
            "image": image,
            "prompt_embedding": prompt_embedding,
            "encoded_video": encoded_video,
            "image_embedding": image_embedding,
            "video_metadata": {
                "num_frames": encoded_video.shape[1],
                "height": encoded_video.shape[2],
                "width": encoded_video.shape[3],
            },
        }

    # ------------------------------------------------------------------
    # Abstract methods – implemented by subclasses
    # ------------------------------------------------------------------

    def preprocess(
        self, video_path: Path | None, image_path: Path | None
    ) -> Tuple[torch.Tensor | None, torch.Tensor | None]:
        raise NotImplementedError("Subclass must implement this method")

    def video_transform(self, frames: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclass must implement this method")

    def image_transform(self, image: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclass must implement this method")


class DatasetWithResize(BaseDataset):
    """Concrete dataset that resizes conditioning images to fixed dims.

    Args:
        max_num_frames: Maximum frames (unused — latents are pre-computed).
        height:  Target height for conditioning image.
        width:   Target width for conditioning image.
    """

    def __init__(
        self, max_num_frames: int, height: int, width: int, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.max_num_frames = max_num_frames
        self.height = height
        self.width = width

        self.__frame_transforms = transforms.Compose(
            [transforms.Lambda(lambda x: x / 255.0 * 2.0 - 1.0)]
        )
        self.__image_transforms = self.__frame_transforms

    @override
    def preprocess(
        self, video_path: Path | None, image_path: Path | None
    ) -> Tuple[torch.Tensor | None, torch.Tensor | None]:
        video = None  # videos are pre-encoded; no need to load raw frames
        if image_path is not None:
            image = preprocess_image_with_resize(image_path, self.height, self.width)
        else:
            image = None
        return video, image

    @override
    def video_transform(self, frames: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.__frame_transforms(f) for f in frames], dim=0)

    @override
    def image_transform(self, image: torch.Tensor) -> torch.Tensor:
        return self.__image_transforms(image)
