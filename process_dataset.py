"""Encode pre-built dataset videos into VAE / CLIP / text latents.

Reads the ``videos/`` tree produced by ``build_dataset.py`` and writes
latent tensors into ``latents/`` following ``docs/dataset-structure.md``.

Each clip directory under ``latents/{source}/{video_id}/{clip_id}/``
will contain:

  - ``rgb_latent.pt``    — VAE-encoded RGB video
  - ``xyz_latent.pt``    — VAE-encoded XYZ point-map video
  - ``visual_embeds.pt`` — CLIP image embedding (first frame)
  - ``text_embeds.pt``   — UMT5 text embedding
  - ``text_ids.pt``      — UMT5 text token IDs

Usage
-----
.. code-block:: bash

    # Single GPU
    python process_dataset.py \\
        --dataset_root ./data \\
        --model_path ./pretrained/Wan2.1-I2V-14B-480P-Diffusers

    # Multi-GPU (clips are auto-sharded across GPUs)
    torchrun --nproc_per_node=4 process_dataset.py \\
        --dataset_root ./data \\
        --model_path ./pretrained/Wan2.1-I2V-14B-480P-Diffusers

    # Force re-encode everything
    python process_dataset.py \\
        --dataset_root ./data \\
        --model_path ./pretrained/Wan2.1-I2V-14B-480P-Diffusers \\
        --no_skip_existing

    # Write latents into a versioned directory
    python process_dataset.py \\
        --dataset_root ./data \\
        --model_path ./pretrained/Wan2.1-I2V-14B-480P-Diffusers \\
        --latents_dir latents_v2
"""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import PIL.Image
import torch

# decord is preferred for efficient video reading but may not be available
# on all platforms.  Fall back to imageio when missing.
try:
    import decord  # isort:skip  # Must import after torch (see decord GH issues)

    _HAS_DECORD = True
except ImportError:
    _HAS_DECORD = False


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------


@torch.no_grad()
def encode_video(frames: torch.Tensor, vae) -> torch.Tensor:
    """Encode a video tensor through the Wan VAE.

    Parameters
    ----------
    frames : torch.Tensor
        Shape ``[B, F, H, W, C]``, values in ``[-1, 1]``.
    vae : AutoencoderKLWan

    Returns
    -------
    torch.Tensor
        Normalised latent, shape ``[B, z_dim, T', H', W']``.
    """
    assert frames.dim() == 5, f"Expected [B,F,H,W,C], got {frames.shape}"
    video = frames.to(device=vae.device, dtype=vae.dtype)
    video = video.permute(0, 4, 1, 2, 3)  # → [B, C, F, H, W]

    latent = vae.encode(video).latent_dist.sample()

    # Apply per-channel latent normalisation from VAE config
    z_dim = vae.config.z_dim
    latents_mean = (
        torch.tensor(vae.config.latents_mean)
        .view(1, z_dim, 1, 1, 1)
        .to(latent.device, latent.dtype)
    )
    latents_std = (
        1.0
        / torch.tensor(vae.config.latents_std)
        .view(1, z_dim, 1, 1, 1)
        .to(latent.device, latent.dtype)
    )
    latent = (latent - latents_mean) * latents_std
    return latent


@torch.no_grad()
def encode_text(
    prompt: str,
    tokenizer,
    text_encoder,
    max_seq_length: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode a text prompt with UMT5.

    Returns
    -------
    prompt_embeds : torch.Tensor
        ``[max_seq_length, hidden_size]`` on CPU.
    text_ids : torch.Tensor
        ``[max_seq_length]`` long on CPU.
    """
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_seq_length,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    input_ids = text_inputs.input_ids        # [1, seq_len]
    mask = text_inputs.attention_mask         # [1, seq_len]
    seq_len = mask.gt(0).sum(dim=1).long()   # [1]

    prompt_embeds = text_encoder(
        input_ids.to(device), mask.to(device),
    ).last_hidden_state  # [1, max_seq_length, hidden_size]

    # Keep only real tokens, zero-pad back to max_seq_length
    embeds = prompt_embeds[0][: seq_len[0]]  # [actual_len, hidden_size]
    pad_len = max_seq_length - embeds.size(0)
    if pad_len > 0:
        embeds = torch.cat(
            [embeds, embeds.new_zeros(pad_len, embeds.size(1))], dim=0,
        )

    return embeds.cpu(), input_ids[0].cpu()


@torch.no_grad()
def encode_image(
    image: PIL.Image.Image,
    image_processor,
    image_encoder,
    device: torch.device,
) -> torch.Tensor:
    """Encode a PIL image with CLIP vision encoder.

    Returns
    -------
    torch.Tensor
        ``[num_tokens, hidden_size]`` (penultimate hidden state), on CPU.
    """
    inputs = image_processor(images=image, return_tensors="pt").to(device)
    outputs = image_encoder(**inputs, output_hidden_states=True)
    return outputs.hidden_states[-2][0].cpu()


# ---------------------------------------------------------------------------
# Video I/O
# ---------------------------------------------------------------------------


def read_video_frames(path: Path) -> np.ndarray:
    """Read all frames from a video file.

    Uses *decord* when available (faster, lower memory); otherwise falls
    back to *imageio*.

    Returns ``[T, H, W, 3]`` uint8 ndarray.
    """
    if _HAS_DECORD:
        vr = decord.VideoReader(str(path))
        # .get_batch() returns NDArray regardless of bridge setting.
        return vr.get_batch(range(len(vr))).asnumpy()
    else:
        import imageio.v2 as imageio

        reader = imageio.get_reader(str(path))
        frames = [f for f in reader]
        reader.close()
        return np.stack(frames, axis=0)


# ---------------------------------------------------------------------------
# Single-clip processing
# ---------------------------------------------------------------------------

EXPECTED_LATENT_FILES = [
    "rgb_latent.pt",
    "xyz_latent.pt",
    "visual_embeds.pt",
]


def _caption_cache_path(clip_info: dict, videos_root: Path, latents_cache_dir: Path) -> Path:
    """Return the cached text latent path for a clip caption."""
    vid_dir = videos_root / clip_info["path"]
    caption = (vid_dir / "caption.txt").read_text(encoding="utf-8").strip()
    caption_hash = hashlib.sha256(caption.encode()).hexdigest()[:16]
    return latents_cache_dir / f"{caption_hash}.pt"


def is_clip_fully_processed(
    clip_info: dict,
    videos_root: Path,
    latents_root: Path,
    latents_cache_dir: Path,
) -> bool:
    """Check whether a clip already has all required per-clip and text cache outputs."""
    lat_dir = latents_root / clip_info["path"]
    clip_done = all((lat_dir / f).exists() for f in EXPECTED_LATENT_FILES)
    if not clip_done:
        return False
    return _caption_cache_path(clip_info, videos_root, latents_cache_dir).exists()


def process_clip(
    clip_info: dict,
    videos_root: Path,
    latents_root: Path,
    vae,
    tokenizer,
    text_encoder,
    image_processor,
    image_encoder,
    max_text_seq_length: int,
    device: torch.device,
    latents_cache_dir: Path,
    skip_existing: bool = True,
) -> bool:
    """Process one clip: read videos → encode → write latents.

    Returns ``True`` if any encoding work was done, ``False`` if fully
    skipped.
    """
    clip_path = clip_info["path"]  # e.g. "4dnex/00000028/clip_0"
    vid_dir = videos_root / clip_path
    lat_dir = latents_root / clip_path
    did_work = False

    # Check if per-clip latents are all done
    clip_done = skip_existing and all(
        (lat_dir / f).exists() for f in EXPECTED_LATENT_FILES
    )

    if not clip_done:
        lat_dir.mkdir(parents=True, exist_ok=True)

        # ---- 1. RGB latent ----
        rgb_latent_path = lat_dir / "rgb_latent.pt"
        if not (skip_existing and rgb_latent_path.exists()):
            rgb_frames = read_video_frames(vid_dir / "video.mp4")
            rgb_tensor = torch.from_numpy(rgb_frames).float()
            rgb_tensor = rgb_tensor / 127.5 - 1.0  # uint8 [0,255] → [-1, 1]
            rgb_tensor = rgb_tensor.unsqueeze(0)    # [1, T, H, W, 3]
            rgb_latent = encode_video(rgb_tensor, vae)
            torch.save(rgb_latent[0].cpu(), rgb_latent_path)
            del rgb_frames, rgb_tensor, rgb_latent
            did_work = True

        # ---- 2. XYZ latent ----
        xyz_latent_path = lat_dir / "xyz_latent.pt"
        if not (skip_existing and xyz_latent_path.exists()):
            xyz_frames = read_video_frames(vid_dir / "xyz.mp4")
            xyz_tensor = torch.from_numpy(xyz_frames).float()
            xyz_tensor = xyz_tensor / 127.5 - 1.0  # uint8 [0,255] → [-1, 1]
            xyz_tensor = xyz_tensor.unsqueeze(0)    # [1, T, H, W, 3]
            xyz_latent = encode_video(xyz_tensor, vae)
            torch.save(xyz_latent[0].cpu(), xyz_latent_path)
            del xyz_frames, xyz_tensor, xyz_latent
            did_work = True

        # ---- 3. Visual embeds (CLIP, first frame) ----
        visual_embeds_path = lat_dir / "visual_embeds.pt"
        if not (skip_existing and visual_embeds_path.exists()):
            first_frame = PIL.Image.open(vid_dir / "first_frame.png").convert("RGB")
            visual_embeds = encode_image(first_frame, image_processor, image_encoder, device)
            torch.save(visual_embeds, visual_embeds_path)
            del first_frame, visual_embeds
            did_work = True

    # ---- 4. Text embeds + token IDs (cached by caption hash) ----
    caption = (vid_dir / "caption.txt").read_text(encoding="utf-8").strip()
    caption_hash = hashlib.sha256(caption.encode()).hexdigest()[:16]
    cache_path = latents_cache_dir / f"{caption_hash}.pt"
    if not (skip_existing and cache_path.exists()):
        text_embeds, text_ids = encode_text(
            caption, tokenizer, text_encoder, max_text_seq_length, device,
        )
        latents_cache_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"text_embeds": text_embeds, "text_ids": text_ids}, cache_path)
        del text_embeds, text_ids
        did_work = True

    return did_work


# ---------------------------------------------------------------------------
# Sharding helper
# ---------------------------------------------------------------------------


def split_list(lst: list, num_segments: int) -> list:
    """Split *lst* into *num_segments* roughly equal contiguous parts."""
    n = len(lst)
    seg_size = n // num_segments
    remainder = n % num_segments
    segments: List[list] = []
    start = 0
    for i in range(num_segments):
        extra = 1 if i < remainder else 0
        end = start + seg_size + extra
        segments.append(lst[start:end])
        start = end
    return segments


def _get_distributed_info() -> Tuple[int, int]:
    """Return ``(rank, world_size)`` from torchrun env vars.

    Falls back to ``(0, 1)`` when not launched via ``torchrun``.
    """
    rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", 0)))
    world = int(os.environ.get("LOCAL_WORLD_SIZE", os.environ.get("WORLD_SIZE", 1)))
    return rank, world


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Encode dataset videos into VAE / CLIP / text latents",
    )
    p.add_argument(
        "--dataset_root",
        type=str,
        default="./data",
        help="Root of the dataset (must contain index.json + videos/)",
    )
    p.add_argument(
        "--model_path",
        type=str,
        default="./pretrained/Wan2.1-I2V-14B-480P-Diffusers",
        help="Path to pretrained Wan 2.1 model (Diffusers format)",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device (default: cuda if available, else cpu)",
    )
    p.add_argument(
        "--max_text_seq_length",
        type=int,
        default=512,
        help="Maximum token length for UMT5 text encoder",
    )
    p.add_argument(
        "--no_skip_existing",
        action="store_true",
        help="Re-encode clips even when latent files already exist",
    )
    p.add_argument(
        "--latents_dir",
        type=str,
        default=None,
        help="Override latents directory name (default: from index.json config.latents_dir)",
    )
    p.add_argument(
        "--index",
        type=str,
        default=None,
        help="Path to index.json (default: {dataset_root}/index.json). "
             "Use with gen_index.py to process a partially-built dataset.",
    )
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
    dataset_root = Path(args.dataset_root)
    index_path = Path(args.index) if args.index else dataset_root / "index.json"

    if not index_path.exists():
        raise FileNotFoundError(
            f"index.json not found at {index_path}. "
            "Run build_dataset.py or gen_index.py first."
        )

    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)

    # ---- resolve directories ----
    videos_root = dataset_root / "videos"
    latents_dir_name = (
        args.latents_dir
        or index.get("config", {}).get("latents_dir", "latents")
    )
    latents_root = dataset_root / latents_dir_name

    # ---- distributed / multi-GPU ----
    rank, world_size = _get_distributed_info()
    is_distributed = world_size > 1

    if is_distributed:
        torch.distributed.init_process_group(backend="nccl")

    # ---- detect done/pending clips before sharding ----
    all_clips: List[dict] = index["clips"]
    latents_cache_dir = dataset_root / "latents_cache"
    skip_existing = not args.no_skip_existing

    if skip_existing:
        t_scan = time.time()
        pending_clips: List[dict] = []
        for clip_info in all_clips:
            if not is_clip_fully_processed(clip_info, videos_root, latents_root, latents_cache_dir):
                pending_clips.append(clip_info)
        scan_elapsed = _fmt_elapsed(time.time() - t_scan)
        done_count = len(all_clips) - len(pending_clips)
        if rank == 0:
            print(
                f"[process_dataset] Pre-scan done ({scan_elapsed}): "
                f"done={done_count}, pending={len(pending_clips)}, total={len(all_clips)}"
            )
    else:
        pending_clips = all_clips

    # ---- shard only pending clips (better distributed balance) ----
    if world_size > 1:
        clips = split_list(pending_clips, world_size)[rank]
    else:
        clips = pending_clips

    # ---- device ----
    if is_distributed:
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
    else:
        device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device(device_str)

    # ---- config summary ----
    print(f"[process_dataset] rank={rank}/{world_size}  device={device}")
    print(f"  dataset_root: {dataset_root}")
    print(f"  videos_root:  {videos_root}")
    print(f"  latents_root: {latents_root}")
    print(f"  latents_cache: {latents_cache_dir}")
    print(f"  model_path:   {args.model_path}")
    print(f"  clips_to_process: {len(clips)}/{len(pending_clips)} pending  "
          f"skip_existing={skip_existing}  "
          f"max_text_seq_length={args.max_text_seq_length}")

    if len(clips) == 0:
        print(f"[process_dataset] rank={rank} No pending clips assigned, skipping model load.")
        if is_distributed:
            torch.distributed.barrier()
    else:
        # ---- load models ----
        t_load = time.time()
        from diffusers import AutoencoderKLWan
        from transformers import (
            AutoTokenizer,
            CLIPImageProcessor,
            CLIPVisionModel,
            UMT5EncoderModel,
        )

        print("[process_dataset] Loading VAE ...")
        vae = AutoencoderKLWan.from_pretrained(args.model_path, subfolder="vae")
        vae.to(device)
        vae.eval()
        vae.enable_slicing()
        vae.enable_tiling()

        print("[process_dataset] Loading text encoder ...")
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path, subfolder="tokenizer",
        )
        text_encoder = UMT5EncoderModel.from_pretrained(
            args.model_path, subfolder="text_encoder",
        )
        text_encoder.to(device)
        text_encoder.eval()

        print("[process_dataset] Loading image encoder ...")
        image_processor = CLIPImageProcessor.from_pretrained(
            args.model_path, subfolder="image_processor",
        )
        image_encoder = CLIPVisionModel.from_pretrained(
            args.model_path, subfolder="image_encoder",
        )
        image_encoder.to(device)
        image_encoder.eval()

        print(f"[process_dataset] Models loaded ({_fmt_elapsed(time.time() - t_load)})")

        # ---- process clips ----
        processed, skipped, errors = 0, 0, 0
        total = len(clips)
        t_encode = time.time()

        for i, clip_info in enumerate(clips):
            done = i + 1
            elapsed = time.time() - t_encode
            eta = _fmt_eta(elapsed, done - 1, total) if i > 0 else "?"
            try:
                did_work = process_clip(
                    clip_info=clip_info,
                    videos_root=videos_root,
                    latents_root=latents_root,
                    vae=vae,
                    tokenizer=tokenizer,
                    text_encoder=text_encoder,
                    image_processor=image_processor,
                    image_encoder=image_encoder,
                    max_text_seq_length=args.max_text_seq_length,
                    device=device,
                    latents_cache_dir=latents_cache_dir,
                    skip_existing=skip_existing,
                )
                elapsed = time.time() - t_encode
                eta = _fmt_eta(elapsed, done, total)
                if did_work:
                    processed += 1
                    print(f"  [{done}/{total}] encode {clip_info['path']}  "
                          f"({_fmt_elapsed(elapsed)}, ETA {eta})")
                else:
                    skipped += 1
                    print(f"  [{done}/{total}] skip   {clip_info['path']}  "
                          f"({_fmt_elapsed(elapsed)}, ETA {eta})")
            except Exception as e:
                errors += 1
                elapsed = time.time() - t_encode
                eta = _fmt_eta(elapsed, done, total)
                print(f"  [{done}/{total}] ERROR  {clip_info['path']}: {e}  "
                      f"({_fmt_elapsed(elapsed)}, ETA {eta})")

            # Periodically free GPU cache
            if torch.cuda.is_available() and done % 10 == 0:
                torch.cuda.empty_cache()

        total_time = _fmt_elapsed(time.time() - t_encode)
        print(f"[process_dataset] rank={rank} Done in {total_time}. "
              f"encoded: {processed}, skipped: {skipped}, errors: {errors}, "
              f"total: {total}")

    if is_distributed:
        torch.distributed.barrier()

    # ---- Update index.json with text_latent_path (rank 0 only) ----
    if rank == 0:
        print("[process_dataset] Updating index.json with text_latent_path ...")
        for clip_info in all_clips:
            cache_path = _caption_cache_path(clip_info, videos_root, latents_cache_dir)
            clip_info["text_latent_path"] = f"latents_cache/{cache_path.name}"
        with open(index_path, "r", encoding="utf-8") as f:
            idx = json.load(f)
        idx["clips"] = all_clips
        # Atomic write: tmp file then rename to avoid corruption on crash.
        tmp_path = index_path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(idx, f, ensure_ascii=False, indent=2)
        tmp_path.rename(index_path)
        print(f"[process_dataset] index.json updated ({len(all_clips)} clips)")

    if is_distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()
