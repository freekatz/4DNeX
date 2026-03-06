"""Visualize 4D pointmaps using Rerun."""

import glob
import json
import pickle
from pathlib import Path

import imageio
import numpy as np
import rerun as rr
import tyro
from tqdm.auto import tqdm


def visualize_pointmap(pm, name, downsample_factor=1):
    """Log a Pointmap object to Rerun across all frames.

    Args:
        pm: Pointmap instance with xyz [F, H, W, 3] and rgb [F, H, W, 3].
        name: Scene name for Rerun entity paths.
        downsample_factor: Spatial downsampling for point clouds.
    """
    has_camera = pm.cams2world is not None
    has_intrinsics = pm.K is not None

    # Log camera trajectory as a static line strip (all positions at once).
    if has_camera:
        positions = pm.cams2world[:, :3, 3]  # [F, 3]
        rr.log(
            f"{name}/camera_trajectory",
            rr.LineStrips3D([positions], colors=[[0, 255, 255]]),
            static=True,
        )

    for i in tqdm(range(pm.num_frames), desc=name):
        xyz = pm.xyz[i]     # [H, W, 3]
        rgb = pm.rgb[i]     # [H, W, 3]

        # Downsample spatially
        xyz_ds = xyz[::downsample_factor, ::downsample_factor].reshape(-1, 3)
        rgb_ds = rgb[::downsample_factor, ::downsample_factor].reshape(-1, 3)

        rr.set_time("frame", sequence=i)

        rr.log(
            f"{name}/point_cloud",
            rr.Points3D(
                positions=xyz_ds,
                colors=(rgb_ds * 255).astype(np.uint8),
            ),
        )

        if has_camera:
            rr.log(
                f"{name}/camera",
                rr.Transform3D(
                    translation=pm.cams2world[i, :3, 3],
                    mat3x3=pm.cams2world[i, :3, :3],
                ),
            )

            if has_intrinsics:
                K = pm.K[i]
                rr.log(
                    f"{name}/camera/image",
                    rr.Pinhole(
                        image_from_camera=K,
                        width=pm.width,
                        height=pm.height,
                    ),
                )

            rr.log(f"{name}/camera/image/rgb", rr.Image(rgb))
        else:
            rr.log(f"{name}/rgb_image", rr.Image(rgb))


def load_from_clip_dir(clip_dir, max_frames):
    """Load from a clip directory containing video.mp4, xyz.mp4, and meta.json."""
    from core.datasets.dataclass import Pointmap

    clip_dir = Path(clip_dir)
    meta_path = clip_dir / "meta.json"
    xyz_mp4 = clip_dir / "xyz.mp4"
    rgb_mp4 = clip_dir / "video.mp4"

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    xyz_norm = meta["xyz_norm"]
    center = np.array(xyz_norm["center"], dtype=np.float32)
    scale = np.float32(xyz_norm["scale"])

    camera = meta.get("camera", {})
    extrinsics = np.array(camera["extrinsics_c2w"], dtype=np.float32) if "extrinsics_c2w" in camera else None
    intrinsics = np.array(camera["intrinsics"], dtype=np.float32) if "intrinsics" in camera else None

    xyz_reader = imageio.get_reader(str(xyz_mp4), "ffmpeg")
    rgb_reader = imageio.get_reader(str(rgb_mp4), "ffmpeg")
    num_frames = min(len(xyz_reader), len(rgb_reader), max_frames)
    if extrinsics is not None:
        num_frames = min(num_frames, extrinsics.shape[0])

    xyz_frames = []
    rgb_frames = []
    for i in range(num_frames):
        xyz_frames.append(xyz_reader.get_data(i).astype(np.float32))
        rgb_frames.append(rgb_reader.get_data(i).astype(np.float32) / 255.0)

    xyz_frames = np.stack(xyz_frames)  # [F, H, W, 3], uint8 range [0, 255]
    rgb_frames = np.stack(rgb_frames)  # [F, H, W, 3], [0, 1]

    # Denormalize: reverse of build_dataset.py encoding
    # encode: xyz_vis = clip((xyz - center) / scale, -1, 1) * 127.5 + 127.5
    # decode: xyz = (pixel / 127.5 - 1) * scale + center
    xyz_frames = (xyz_frames / 127.5 - 1.0) * scale + center.reshape(1, 1, 1, 3)

    pm = Pointmap(
        xyz=xyz_frames,
        rgb=rgb_frames,
        cams2world=extrinsics[:num_frames] if extrinsics is not None else None,
        K=intrinsics[:num_frames] if intrinsics is not None else None,
    )
    yield pm, clip_dir.name


def load_from_pkl(pkl_dir, max_frames):
    """Load Pointmap objects from pkl files."""
    pkl_list = sorted(glob.glob(f'{pkl_dir}/*.pkl'))
    for pkl_path in pkl_list:
        pm = pickle.load(open(pkl_path, 'rb'))
        yield pm, pkl_path


def load_from_npy_mp4(pointmap_npy, rgb_mp4, max_frames):
    """Load from a pointmap npy file and an RGB mp4 file."""
    from core.datasets.dataclass import Pointmap

    pointmap = np.load(pointmap_npy)  # [F, H, W, 3]
    rgb_reader = imageio.get_reader(rgb_mp4, "ffmpeg")
    num_frames = min(pointmap.shape[0], len(rgb_reader), max_frames)

    rgb_frames = []
    for i in range(num_frames):
        rgb_frames.append(rgb_reader.get_data(i).astype(np.float32) / 255.0)
    rgb_frames = np.stack(rgb_frames)  # [F, H, W, 3]

    pm = Pointmap(
        xyz=pointmap[:num_frames],
        rgb=rgb_frames,
    )
    yield pm, pointmap_npy


def load_from_mp4_mp4(pointmap_mp4, rgb_mp4, max_frames):
    """Load from pointmap mp4 and RGB mp4 files."""
    from core.datasets.dataclass import Pointmap

    pointmap_reader = imageio.get_reader(pointmap_mp4, "ffmpeg")
    rgb_reader = imageio.get_reader(rgb_mp4, "ffmpeg")
    num_frames = min(len(pointmap_reader), len(rgb_reader), max_frames)

    xyz_frames = []
    rgb_frames = []
    for i in range(num_frames):
        xyz_frames.append(pointmap_reader.get_data(i).astype(np.float32) / 255.0)
        rgb_frames.append(rgb_reader.get_data(i).astype(np.float32) / 255.0)
    xyz_frames = np.stack(xyz_frames)  # [F, H, W, 3]
    rgb_frames = np.stack(rgb_frames)  # [F, H, W, 3]

    pm = Pointmap(
        xyz=xyz_frames,
        rgb=rgb_frames,
    )
    yield pm, pointmap_mp4


def main(
    downsample_factor: int = 1,
    max_frames: int = 100,
    clip_dir: str = None,
    pkl_dir: str = None,
    pointmap_mp4: str = None,
    rgb_mp4: str = None,
    pointmap_npy: str = None,
    rr_recording: str = "pointmap_log.rrd",
) -> None:
    """Visualize 4D pointmaps using Rerun.

    Data sources (pick one):
        --clip_dir: Clip directory with video.mp4, xyz.mp4, meta.json.
        --pkl_dir: Directory containing Pointmap .pkl files.
        --pointmap_npy + --rgb_mp4: Pointmap npy file and RGB mp4 file.
        --pointmap_mp4 + --rgb_mp4: Pointmap mp4 file and RGB mp4 file.
    """
    rr.init(rr_recording)

    if clip_dir is not None:
        loader = load_from_clip_dir(clip_dir, max_frames)
    elif pointmap_npy is not None and rgb_mp4 is not None:
        loader = load_from_npy_mp4(pointmap_npy, rgb_mp4, max_frames)
    elif pointmap_mp4 is not None and rgb_mp4 is not None:
        loader = load_from_mp4_mp4(pointmap_mp4, rgb_mp4, max_frames)
    elif pkl_dir is not None:
        loader = load_from_pkl(pkl_dir, max_frames)
    else:
        raise ValueError(
            "Provide --clip_dir, --pkl_dir, --pointmap_npy + --rgb_mp4, "
            "or --pointmap_mp4 + --rgb_mp4"
        )

    for pm, name in loader:
        visualize_pointmap(pm, name, downsample_factor)

    rr.save(rr_recording)
    print(f"Saved Rerun recording to {rr_recording}")


if __name__ == "__main__":
    tyro.cli(main)
