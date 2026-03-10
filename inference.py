"""inference script: generate RGB video + XYZ pointmap from a single image.

Loads the 4DNeX pipeline (WanTransformer3DModelDembSameRope + LoRA) once,
then generates dual RGB + XYZ latents for each input sample.
"""

import argparse
import gc
import hashlib
import logging
import os
import pickle
import datetime

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import imageio
import numpy as np
import torch
from PIL import Image
from transformers import CLIPVisionModel
from diffusers.utils.loading_utils import load_image

from core.models.wan import WanTransformer3DModelDembSameRope
from core.models.wan_pipeline import WanSameRopeImageToVideoPipeline
from core.datasets.dataclass import Pointmap
from core.datasets.dataset import ENCODED_PM_MEAN, ENCODED_PM_STD
from core.models.wan_tokenizer import WanTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    latents = latents[None]  # Add batch dim -> [1, 16, F, H, W]

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


def get_latent_cache_path(cache_dir, index, prompt, image_path, args, weights_path):
    cache_key_raw = (
        f"idx={index}|prompt={prompt}|image={image_path}|"
        f"seed={args.seed}|frames={args.num_frames}|steps={args.num_inference_steps}|"
        f"cfg={args.guidance_scale}|h={args.height}|w={args.width}|"
        f"rank={args.rank}|weights={weights_path}"
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
    """Load a list from a .txt file (one item per line) or treat as a single value."""
    if os.path.isfile(path_or_value) and path_or_value.lower().endswith('.txt'):
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


def resolve_weights_path(args):
    """Resolve adapter/checkpoint directory for inference weights.

    Priority:
    1) --weights_path (recommended for training artifacts)
    2) --lora_path (backward compatible)
    """
    if args.weights_path:
        return args.weights_path
    if args.lora_path:
        return args.lora_path
    raise ValueError("Provide --weights_path (recommended) or --lora_path")


def resolve_output_dir(base_out: str) -> str:
    """Create a run-unique output directory by appending a timestamp suffix."""
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_out}-{ts}"


# ---------------------------------------------------------------------------
# Pipeline loading & inference
# ---------------------------------------------------------------------------

def load_pipeline(model_path, lora_path, lora_rank, offload_mode, dtype=torch.bfloat16):
    """Load the 4DNeX inference pipeline once.

    Returns a ready-to-call pipeline with LoRA weights fused and offload configured.
    """
    logger.info("Loading image encoder...")
    image_encoder = CLIPVisionModel.from_pretrained(
        model_path, subfolder="image_encoder", torch_dtype=torch.float32,
    )

    logger.info("Loading transformer (WanTransformer3DModelDembSameRope)...")
    transformer = WanTransformer3DModelDembSameRope.from_pretrained(
        model_path, subfolder="transformer", torch_dtype=dtype,
    )

    # Load learnable domain embeddings
    demb_file = os.path.join(lora_path, "learnable_domain_embeddings.pt")
    if os.path.exists(demb_file):
        learnable_domain_embeddings = torch.load(demb_file, map_location="cpu")
        transformer.learnable_domain_embeddings.data = learnable_domain_embeddings.to(
            transformer.device, transformer.dtype
        )
        logger.info(f"Loaded learnable_domain_embeddings from {demb_file}")
    else:
        logger.warning(f"learnable_domain_embeddings.pt not found at {lora_path}, using zeros")

    logger.info("Building pipeline...")
    pipe = WanSameRopeImageToVideoPipeline.from_pretrained(
        model_path,
        image_encoder=image_encoder,
        transformer=transformer,
        torch_dtype=dtype,
    )

    # Load and fuse LoRA weights
    lora_weights_file = os.path.join(lora_path, "pytorch_lora_weights.safetensors")
    if os.path.exists(lora_weights_file):
        logger.info(f"Loading LoRA weights from {lora_weights_file}")
        pipe.load_lora_weights(lora_path, weight_name="pytorch_lora_weights.safetensors")
        pipe.fuse_lora(components=["transformer"], lora_scale=0.5)
        logger.info("LoRA weights loaded and fused at scale 0.5")
    else:
        logger.warning(f"pytorch_lora_weights.safetensors not found at {lora_path}")

    # VAE memory optimization
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    # CPU offload strategy
    if offload_mode == "sequential":
        pipe.enable_sequential_cpu_offload()
        logger.info("Using sequential CPU offload (low memory, slower)")
    elif offload_mode == "model":
        pipe.enable_model_cpu_offload()
        logger.info("Using model CPU offload (moderate memory, faster)")
    elif offload_mode == "none":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe.to(device)
        logger.info(f"Using no offload, pipeline on {device}")

    return pipe


def run_pipeline(pipe, prompt, image_path, num_frames, width, height,
                 num_inference_steps, guidance_scale, seed, output_on_cpu):
    """Run a single inference pass and return split (latents_rgb, latents_xyz)."""
    image = load_image(image=image_path)

    max_area = 480 * 720
    aspect_ratio = image.height / image.width
    mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
    resolved_height = int(height or (round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value))
    resolved_width = int(width or (round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value))
    image = image.resize((resolved_width, resolved_height))

    with torch.inference_mode():
        video_generate = pipe(
            height=resolved_height,
            width=resolved_width,
            prompt=prompt,
            image=image,
            num_videos_per_prompt=1,
            num_inference_steps=num_inference_steps,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            generator=torch.Generator().manual_seed(seed),
            output_type="latent",
        ).frames[0]

    # Split double-width latent into RGB and XYZ halves
    # video_generate shape: [C, F, H, W*2]
    half_w = video_generate.shape[-1] // 2
    latents_rgb = video_generate[..., :half_w]
    latents_xyz = video_generate[..., half_w:]

    if output_on_cpu:
        latents_rgb = latents_rgb.cpu()
        latents_xyz = latents_xyz.cpu()

    return latents_rgb, latents_xyz


def release_pipeline(pipe):
    """Release pipeline VRAM so the decoder can use it."""
    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main(args):
    weights_path = resolve_weights_path(args)
    run_out = resolve_output_dir(args.out)
    print(f"[Info] Output directory: {run_out}")

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
        print("[Info] decode_device=cuda and decode_fast=true, setting latents_on_cpu=false to avoid extra copies")
        effective_latents_on_cpu = False

    latent_cache_dir = args.latent_cache_dir or os.path.join(run_out, "latent_cache")

    os.makedirs(run_out, exist_ok=True)

    # Determine which indices this process handles
    all_indices = list(range(len(prompt_list)))
    if args.idx != -1:
        all_indices = [args.idx]
    elif args.num_shards > 1:
        all_indices = [i for i in all_indices if i % args.num_shards == args.shard_id]
        print(f"[Shard {args.shard_id}/{args.num_shards}] Processing {len(all_indices)}/{len(prompt_list)} samples")

    # Check if any sample needs denoising (not all cached)
    needs_denoising = False
    for i in all_indices:
        prompt, image_path = prompt_list[i], image_list[i]
        prompt = prompt + ' POINTMAP_STYLE.'
        cache_path = get_latent_cache_path(latent_cache_dir, i, prompt, image_path, args, weights_path)
        if not (args.use_latent_cache and os.path.exists(cache_path)):
            needs_denoising = True
            break

    # Load pipeline once (only if needed)
    pipe = None
    if needs_denoising:
        pipe = load_pipeline(
            model_path=args.model_path,
            lora_path=weights_path,
            lora_rank=args.rank,
            offload_mode=args.offload_mode,
            dtype=torch.bfloat16,
        )

    # Collect all latent results before decoding, so we can release the pipeline first
    latent_results = {}

    for i in all_indices:
        prompt, image_path = prompt_list[i], image_list[i]
        suffix = 'POINTMAP_STYLE.'
        prompt = prompt + ' ' + suffix

        print(f"[{i}/{len(prompt_list)}] Generating: {prompt[:60]}...")

        cache_path = get_latent_cache_path(latent_cache_dir, i, prompt, image_path, args, weights_path)
        if args.use_latent_cache and os.path.exists(cache_path):
            print(f"  Loading denoised latents from cache: {cache_path}")
            latents_rgb, latents_xyz, _ = load_latent_cache(cache_path)
        else:
            latents_rgb, latents_xyz = run_pipeline(
                pipe=pipe,
                prompt=prompt,
                image_path=image_path,
                num_frames=args.num_frames,
                width=args.width,
                height=args.height,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                seed=args.seed,
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
                    "weights_path": weights_path,
                }
                save_latent_cache(cache_path, latents_rgb, latents_xyz, meta)
                print(f"  Saved denoised latents to cache: {cache_path}")

        latent_results[i] = (latents_rgb, latents_xyz)

    # Release pipeline VRAM before decoding
    if pipe is not None:
        release_pipeline(pipe)
        pipe = None

    # Decode all latents
    for i in all_indices:
        latents_rgb, latents_xyz = latent_results[i]

        if decode_device == "cuda":
            latents_rgb = latents_rgb.to("cuda")
            latents_xyz = latents_xyz.to("cuda")

        # Load decoder lazily
        if tokenizer is None:
            print(f"  Loading decode tokenizer on {decode_device}...")
            tokenizer = WanTokenizer(model_path=vae_path, device=decode_device)
            configure_decode_mode(tokenizer, fast=args.decode_fast)
            if args.decode_fast:
                print("  Decode mode: fast (disable slicing/tiling)")
            else:
                print("  Decode mode: memory-safe (enable slicing/tiling)")

        # Decode and save RGB video
        print(f"  [{i}] Decoding RGB latent...")
        rgb_path = os.path.join(run_out, f'{i:05d}_rgb.mp4')
        rgb_frames = decode_and_save(latents_rgb, tokenizer, rgb_path, mode='rgb', fps=args.fps)

        # Decode and save XYZ pointmap video
        print(f"  [{i}] Decoding XYZ latent...")
        xyz_path = os.path.join(run_out, f'{i:05d}_xyz.mp4')
        xyz_frames = decode_and_save(latents_xyz, tokenizer, xyz_path, mode='xyz', fps=args.fps)

        # Save combined pointmap
        print(f"  [{i}] Saving pointmap pickle...")
        pkl_path = os.path.join(run_out, f'{i:05d}.pkl')
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

            opt_pkl_path = os.path.join(run_out, f'{i:05d}_optimized.pkl')
            with open(opt_pkl_path, 'wb') as f:
                pickle.dump(pm, f)
            print(f"  Saved optimized: {opt_pkl_path}")

        # Free decoded frames
        del latents_rgb, latents_xyz
    del latent_results


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
    parser.add_argument("--weights_path", type=str, default=None,
                        help="Path to training checkpoint/adapter directory (recommended, e.g. training/checkpoints/step-000010)")
    parser.add_argument("--lora_path", type=str, default=None,
                        help="Backward-compatible alias of --weights_path")
    parser.add_argument("--out", type=str, default="results", help="Output directory prefix; timestamp suffix will be auto-appended")
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
