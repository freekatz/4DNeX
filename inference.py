"""inference script: generate RGB video + XYZ pointmap from a single image."""

import argparse
import hashlib
import os
import pickle

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import imageio
import numpy as np
import torch

from core.inference.pipeline import generate_video
from core.datasets.dataclass import Pointmap
from core.datasets.dataset import ENCODED_PM_MEAN, ENCODED_PM_STD
from core.inference.tokenizer import WanTokenizer


def decode_and_save(latents, tokenizer, save_path, mode='rgb', fps=24):
    """Decode latents and save as video.

    Args:
        latents: [16, F, H, W] latent tensor (no batch dim).
        tokenizer: WanTokenizer instance for VAE decoding.
        save_path: Path to save the mp4 file.
        mode: 'rgb' for standard decode, 'xyz' for XYZ with denormalization.

    Returns:
        frames: [F, H, W, 3] numpy array. RGB in [0, 1]; XYZ in original scale.
    """
    latents = latents[None]  # Add batch dim → [1, 16, F, H, W]

    if mode == 'xyz':
        # Apply dataset-specific denormalization for XYZ before VAE decode
        latents = latents * ENCODED_PM_STD + ENCODED_PM_MEAN

    frames = tokenizer.decode(latents)  # [F, H, W, 3]
    mp4_path = save_path if save_path.endswith('.mp4') else save_path.rsplit('.', 1)[0] + '.mp4'
    imageio.mimwrite(mp4_path, (frames * 255).clip(0, 255).astype(np.uint8), fps=fps)

    if mode == 'rgb':
        frames = frames.clip(0, 1)
    return frames


def configure_decode_mode(tokenizer, fast=True):
    """Configure VAE decode mode.

    fast=True: disable slicing/tiling for maximum decode throughput (higher memory).
    fast=False: enable slicing/tiling for lower memory (slower).
    """
    model = tokenizer.model
    if fast:
        if hasattr(model, "disable_slicing"):
            model.disable_slicing()
        if hasattr(model, "disable_tiling"):
            model.disable_tiling()
    else:
        if hasattr(model, "enable_slicing"):
            model.enable_slicing()
        if hasattr(model, "enable_tiling"):
            model.enable_tiling()


def save_pointmap(xyz_frames, rgb_frames, save_path):
    """Save combined RGB + XYZ as a Pointmap pickle.

    Args:
        xyz_frames: [F, H, W, 3] numpy array of XYZ coordinates.
        rgb_frames: [F, H, W, 3] numpy array of RGB values in [0, 1].
        save_path: Path to save the pickle file.
    """
    pm = Pointmap(
        xyz=xyz_frames,
        rgb=rgb_frames.clip(min=0, max=1),
    )
    with open(save_path, 'wb') as f:
        pickle.dump(pm, f)


def get_latent_cache_path(cache_dir, index, prompt, image_path, args):
    cache_key_raw = (
        f"idx={index}|prompt={prompt}|image={image_path}|"
        f"seed={args.seed}|frames={args.num_frames}|steps={args.num_inference_steps}|"
        f"cfg={args.guidance_scale}|h={args.height}|w={args.width}|"
        f"rank={args.rank}|lora={args.lora_path}"
    )
    cache_key = hashlib.sha1(cache_key_raw.encode("utf-8")).hexdigest()[:16]
    return os.path.join(cache_dir, f"{index:05d}_{cache_key}.pt")


def save_latent_cache(cache_path, latents_rgb, latents_xyz, meta):
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    payload = {
        "latents_rgb": latents_rgb.detach().cpu(),
        "latents_xyz": latents_xyz.detach().cpu(),
        "meta": meta,
    }
    torch.save(payload, cache_path)


def load_latent_cache(cache_path):
    payload = torch.load(cache_path, map_location="cpu")
    if isinstance(payload, dict) and "latents_rgb" in payload and "latents_xyz" in payload:
        return payload["latents_rgb"], payload["latents_xyz"], payload.get("meta", {})
    raise ValueError(f"Invalid latent cache format: {cache_path}")


def load_list(path_or_value):
    """Load a list from a file (one item per line) or treat as a single value."""
    if os.path.isfile(path_or_value):
        with open(path_or_value, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    return [path_or_value]


def load_from_clip_dir(clip_dir):
    """Load prompt and image lists from a clip directory (or multiple clips).

    Supports:
        - Single clip dir: videos/{source}/{video_id}/{clip_id}/
        - Parent dir containing multiple clip_* subdirs
    Returns:
        (prompt_list, image_list)
    """
    from pathlib import Path
    clip_dir = Path(clip_dir)

    # Collect clip directories
    if (clip_dir / "meta.json").exists():
        clip_dirs = [clip_dir]
    else:
        clip_dirs = sorted(d for d in clip_dir.iterdir() if d.is_dir() and (d / "meta.json").exists())
        if not clip_dirs:
            raise ValueError(f"No clip directories (with meta.json) found in {clip_dir}")

    prompt_list, image_list = [], []
    for d in clip_dirs:
        caption_file = d / "caption.txt"
        image_file = d / "first_frame.png"
        if not caption_file.exists():
            raise FileNotFoundError(f"Missing caption.txt in {d}")
        if not image_file.exists():
            raise FileNotFoundError(f"Missing first_frame.png in {d}")
        prompt_list.append(caption_file.read_text(encoding="utf-8").strip())
        image_list.append(str(image_file))

    return prompt_list, image_list


def main(args):
    if args.clip_dir is not None:
        prompt_list, image_list = load_from_clip_dir(args.clip_dir)
    else:
        if args.prompt is None or args.image is None:
            raise ValueError("Provide --clip_dir, or both --prompt and --image")
        prompt_list = load_list(args.prompt)
        image_list = load_list(args.image)

    assert len(prompt_list) == len(image_list), \
        f"Prompt count ({len(prompt_list)}) != image count ({len(image_list)})"

    vae_path = os.path.join(args.model_path, 'vae')
    if args.decode_device == 'auto':
        decode_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        decode_device = args.decode_device

    # Lazy-load tokenizer after denoising to avoid competing VRAM with inference pipeline.
    tokenizer = None

    # For fastest decode on CUDA, avoid moving latents to CPU and back.
    effective_latents_on_cpu = args.latents_on_cpu
    if decode_device == 'cuda' and args.decode_fast and args.latents_on_cpu:
        print("[Info] decode_device=cuda 且 decode_fast=true，自动设置 latents_on_cpu=false 以避免额外拷贝并加速解码")
        effective_latents_on_cpu = False

    latent_cache_dir = args.latent_cache_dir or os.path.join(args.out, "latent_cache")

    os.makedirs(args.out, exist_ok=True)

    # Determine which indices this process handles
    all_indices = list(range(len(prompt_list)))
    if args.idx != -1:
        all_indices = [args.idx]
    elif args.num_shards > 1:
        all_indices = [i for i in all_indices if i % args.num_shards == args.shard_id]
        print(f"[Shard {args.shard_id}/{args.num_shards}] Processing {len(all_indices)}/{len(prompt_list)} samples")

    for i in all_indices:

        prompt, image_path = prompt_list[i], image_list[i]
        suffix = 'POINTMAP_STYLE.'
        prompt = prompt + ' ' + suffix

        print(f"[{i}/{len(prompt_list)}] Generating: {prompt[:60]}...")

        cache_path = get_latent_cache_path(latent_cache_dir, i, prompt, image_path, args)
        loaded_from_cache = False
        if args.use_latent_cache and os.path.exists(cache_path):
            print(f"  Loading denoised latents from cache: {cache_path}")
            latents_rgb, latents_xyz, _ = load_latent_cache(cache_path)
            loaded_from_cache = True
        else:
            latents_rgb, latents_xyz = generate_video(
                prompt=prompt,
                image_or_video_path=image_path,
                model_path=args.model_path,
                lora_path=args.lora_path,
                num_frames=args.num_frames,
                width=args.width,
                height=args.height,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                seed=args.seed,
                lora_rank=args.rank,
                dtype=torch.bfloat16,
                offload_mode=args.offload_mode,
                output_on_cpu=effective_latents_on_cpu,
            )
            if args.use_latent_cache:
                meta = {
                    "seed": args.seed,
                    "num_frames": args.num_frames,
                    "num_inference_steps": args.num_inference_steps,
                    "guidance_scale": args.guidance_scale,
                    "height": args.height,
                    "width": args.width,
                    "lora_path": args.lora_path,
                }
                save_latent_cache(cache_path, latents_rgb, latents_xyz, meta)
                print(f"  Saved denoised latents to cache: {cache_path}")

        if decode_device == "cuda" and not loaded_from_cache:
            # Ensure decode runs on GPU without extra host-device transfers.
            latents_rgb = latents_rgb.to("cuda")
            latents_xyz = latents_xyz.to("cuda")

        # Load decoder after denoising; inference pipeline has been released in generate_video.
        if tokenizer is None:
            print(f"  Loading decode tokenizer on {decode_device}...")
            tokenizer = WanTokenizer(model_path=vae_path, device=decode_device)
            configure_decode_mode(tokenizer, fast=args.decode_fast)
            if args.decode_fast:
                print("  Decode mode: fast (disable slicing/tiling)")
            else:
                print("  Decode mode: memory-safe (enable slicing/tiling)")

        # Decode and save RGB video
        print("  Decoding RGB latent...")
        rgb_path = os.path.join(args.out, f'{i:05d}_rgb.mp4')
        rgb_frames = decode_and_save(latents_rgb, tokenizer, rgb_path, mode='rgb', fps=args.fps)

        # Decode and save XYZ pointmap video
        print("  Decoding XYZ latent...")
        xyz_path = os.path.join(args.out, f'{i:05d}_xyz.mp4')
        xyz_frames = decode_and_save(latents_xyz, tokenizer, xyz_path, mode='xyz', fps=args.fps)

        # Save combined pointmap
        print("  Saving pointmap pickle...")
        pkl_path = os.path.join(args.out, f'{i:05d}.pkl')
        save_pointmap(xyz_frames, rgb_frames, pkl_path)

        print(f"  Saved: {rgb_path}, {xyz_path}, {pkl_path}")

        # Optional: post-processing with camera parameter optimization
        if args.optimize:
            from pm_registration import optimise_xyz_batch, depth_to_3d_points
            print("  Running post-optimization...")
            if not torch.cuda.is_available():
                raise RuntimeError("--optimize requires CUDA for pm_registration optimization")
            F, H, W, _ = xyz_frames.shape
            xyz_tensor = torch.from_numpy(xyz_frames).cuda()
            # Center XY coordinates
            xyz_tensor[..., :2] = xyz_tensor[..., :2] - 0.5
            depth_map, K, R, t = optimise_xyz_batch(xyz_tensor, n_iters=args.opt_iters)

            # Reconstruct optimized pointmap
            updated_xyz = []
            for fi in range(F):
                pts = depth_to_3d_points(depth_map[fi], K, R[fi], t[fi])
                updated_xyz.append(pts)
            updated_xyz = np.stack(updated_xyz).reshape(F, H, W, 3)

            cams2world = np.eye(4)[None].repeat(F, axis=0)
            for fi in range(F):
                cams2world[fi, :3, :3] = R[fi].cpu().numpy()
                cams2world[fi, :3, 3] = t[fi].cpu().numpy()

            pm = Pointmap(
                xyz=updated_xyz,
                rgb=rgb_frames.clip(min=0, max=1),
                depth=depth_map.cpu().numpy(),
                cams2world=cams2world,
                K=K.cpu().numpy()[None].repeat(F, axis=0),
            )

            opt_pkl_path = os.path.join(args.out, f'{i:05d}_optimized.pkl')
            with open(opt_pkl_path, 'wb') as f:
                pickle.dump(pm, f)
            print(f"  Saved optimized: {opt_pkl_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate RGB + XYZ from a single image")
    parser.add_argument("--prompt", type=str, default=None, help="Prompt string or path to prompt list file")
    parser.add_argument("--image", type=str, default=None, help="Image path or path to image list file")
    parser.add_argument("--clip_dir", type=str, default=None,
                        help="Clip directory (reads caption.txt + first_frame.png). "
                             "Can be a single clip or parent dir with multiple clip_* subdirs")
    parser.add_argument("--idx", type=int, default=-1, help="Process only this index (-1 for all)")
    parser.add_argument("--shard_id", type=int, default=0, help="Shard index for multi-GPU inference (0-based)")
    parser.add_argument("--num_shards", type=int, default=1, help="Total number of shards (= number of GPUs)")
    parser.add_argument("--model_path", type=str, default="pretrained/Wan2.1-I2V-14B-480P-Diffusers",
                        help="Path to pretrained Wan model")
    parser.add_argument("--lora_path", type=str, required=True,
                        help="Path to LoRA weights directory (containing pytorch_lora_weights.safetensors + learnable_domain_embeddings.pt)")
    parser.add_argument("--out", type=str, default="results", help="Output directory")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames to generate")
    parser.add_argument("--height", type=int, default=None, help="Output video height (must match model constraints)")
    parser.add_argument("--width", type=int, default=None, help="Output video width (must match model constraints)")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Diffusion steps")
    parser.add_argument("--guidance_scale", type=float, default=5.0, help="CFG scale")
    parser.add_argument("--fps", type=int, default=24, help="Output video FPS")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--rank", type=int, default=64, help="LoRA rank")
    parser.add_argument("--offload_mode", type=str, default="model",
                        choices=["model", "sequential", "none"],
                        help="CPU offload strategy: "
                             "'model' = whole-model offload (~28GB), "
                             "'sequential' = per-layer offload (~8GB, slower), "
                             "'none' = all on GPU (fastest, ~30GB+)")
    parser.add_argument("--latents_on_cpu", action=argparse.BooleanOptionalAction, default=True,
                        help="Move generated latents to CPU before returning (recommended to avoid decode-time OOM)")
    parser.add_argument("--decode_device", type=str, default="cuda", choices=["cpu", "cuda", "auto"],
                        help="Device for VAE decode. Use cpu to reduce GPU peak memory")
    parser.add_argument("--decode_fast", action=argparse.BooleanOptionalAction, default=True,
                        help="Use fastest decode mode (disable VAE slicing/tiling). May use more memory")
    parser.add_argument("--use_latent_cache", action=argparse.BooleanOptionalAction, default=True,
                        help="If cache exists, load denoised latents directly and skip denoising")
    parser.add_argument("--latent_cache_dir", type=str, default=None,
                        help="Directory to store/load denoised latent cache (default: <out>/latent_cache)")
    parser.add_argument("--optimize", action="store_true",
                        help="Run post-optimization on generated pointmaps")
    parser.add_argument("--opt_iters", type=int, default=1500,
                        help="Number of optimization iterations")
    args = parser.parse_args()
    main(args)
