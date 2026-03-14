"""inference script: generate RGB video + XYZ pointmap from a single image.

Loads the One4D dual-branch pipeline (WanTransformer3DModelDualBranch + LoRA) once,
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
from transformers import AutoTokenizer, CLIPImageProcessor, UMT5EncoderModel
from diffusers.utils.loading_utils import load_image
from diffusers import AutoencoderKLWan, FlowMatchEulerDiscreteScheduler
from peft import LoraConfig, set_peft_model_state_dict

from core.models.wan import WanTransformer3DModelDualBranch
from core.models.wan_pipeline import WanDualBranchPipeline
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


def write_2x2_video(gt_rgb_path, gt_xyz_path, pred_rgb_frames, pred_xyz_frames, out_path, fps, target_size=None):
    """Write a single MP4 with 2x2 layout: top row GT RGB | GT XYZ, bottom row Pred RGB | Pred XYZ.

    All four quads are resized to the same (H, W) if needed. Processes frame-by-frame to limit memory.
    pred_*_frames: [F, H, W, 3] numpy, RGB in [0, 1] or [0, 255], XYZ in any scale.
    """
    from PIL import Image
    reader_rgb = imageio.get_reader(gt_rgb_path)
    reader_xyz = imageio.get_reader(gt_xyz_path)
    n_pred = len(pred_rgb_frames)
    writer = None
    for idx, (fr, fx) in enumerate(zip(reader_rgb, reader_xyz)):
        if idx >= n_pred:
            break
        fr = np.asarray(fr)
        fx = np.asarray(fx)
        pr = np.asarray(pred_rgb_frames[idx])
        px = np.asarray(pred_xyz_frames[idx])
        if pr.max() <= 1.0:
            pr = (pr * 255).clip(0, 255).astype(np.uint8)
        if px.max() <= 1.0:
            px = (px * 255).clip(0, 255).astype(np.uint8)
        h, w = pr.shape[0], pr.shape[1]
        if target_size is None:
            target_size = (w, h)
        tw, th = target_size
        if fr.shape[0] != th or fr.shape[1] != tw:
            fr = np.array(Image.fromarray(fr).resize((tw, th), Image.Resampling.LANCZOS))
        if fx.shape[0] != th or fx.shape[1] != tw:
            fx = np.array(Image.fromarray(fx).resize((tw, th), Image.Resampling.LANCZOS))
        if pr.shape[0] != th or pr.shape[1] != tw:
            pr = np.array(Image.fromarray(pr).resize((tw, th), Image.Resampling.LANCZOS))
        if px.shape[0] != th or px.shape[1] != tw:
            px = np.array(Image.fromarray(px).resize((tw, th), Image.Resampling.LANCZOS))
        top = np.concatenate([fr, fx], axis=1)
        bottom = np.concatenate([pr, px], axis=1)
        frame = np.concatenate([top, bottom], axis=0)
        if writer is None:
            writer = imageio.get_writer(str(out_path), fps=fps)
        writer.append_data(frame)
    if writer is not None:
        writer.close()
    reader_rgb.close()
    reader_xyz.close()


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
    payload = torch.load(cache_path, map_location="cpu", weights_only=True)
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
        (prompt_list, image_list, clip_dirs) where clip_dirs[i] is the Path for sample i
        (for merge_output: GT video.mp4 / xyz.mp4 live under clip_dirs[i]).
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

    return prompt_list, image_list, clip_dirs


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


def resolve_output_dir(base_out: str, result_name: str | None = None) -> str:
    """Resolve output directory.

    If result_name is given (e.g. 'step-800'), use results/result-step-800.
    Otherwise append a timestamp suffix for a unique run dir.
    """
    if result_name:
        return os.path.join(base_out, f"result-{result_name}")
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(base_out, f"result-{ts}")


# ---------------------------------------------------------------------------
# Pipeline loading & inference
# ---------------------------------------------------------------------------

def load_pipeline(model_path, lora_path, lora_rank, zcl_layers, offload_mode, dtype=torch.bfloat16):
    """Load the inference pipeline once.

    Returns a ready-to-call pipeline with LoRA weights fused and offload configured.
    """
    logger.info("Loading tokenizer/text encoder/image processor...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer")
    text_encoder = UMT5EncoderModel.from_pretrained(model_path, subfolder="text_encoder", torch_dtype=torch.float32)
    image_processor = CLIPImageProcessor.from_pretrained(model_path, subfolder="image_processor")

    logger.info("Loading image encoder...")
    image_encoder = CLIPVisionModel.from_pretrained(
        model_path, subfolder="image_encoder", torch_dtype=torch.float32,
    )

    logger.info("Loading VAE + scheduler...")
    vae = AutoencoderKLWan.from_pretrained(model_path, subfolder="vae", torch_dtype=dtype)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler")

    logger.info("Loading transformer (WanTransformer3DModelDualBranch)...")
    transformer = WanTransformer3DModelDualBranch.from_pretrained(
        model_path, subfolder="transformer", torch_dtype=dtype, zcl_layers=tuple(zcl_layers),
    )

    logger.info("Building pipeline...")
    pipe = WanDualBranchPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        image_encoder=image_encoder,
        image_processor=image_processor,
        transformer=transformer,
        vae=vae,
        scheduler=scheduler,
    )

    # Load dual modality-specific LoRA adapters (rgb/xyz)
    lora_adapters_file = os.path.join(lora_path, "lora_adapters.pt")
    if os.path.exists(lora_adapters_file):
        logger.info(f"Loading dual LoRA adapters from {lora_adapters_file}")
        lora_payload = torch.load(lora_adapters_file, map_location="cpu", weights_only=True)
        adapters_state = lora_payload.get("adapters", {})
        peft_config = lora_payload.get("peft_config")

        if peft_config is None:
            peft_config = {
                "r": lora_rank,
                "lora_alpha": max(1, lora_rank // 2),
                "target_modules": ["to_q", "to_k", "to_v", "to_out.0"],
                "init_lora_weights": True,
            }

        lora_cfg = LoraConfig(**peft_config)
        transformer.add_adapter(lora_cfg, adapter_name="rgb")
        transformer.add_adapter(lora_cfg, adapter_name="xyz")

        for adapter_name in ("rgb", "xyz"):
            if adapter_name not in adapters_state:
                raise KeyError(f"Adapter '{adapter_name}' not found in {lora_adapters_file}")
            set_peft_model_state_dict(transformer, adapters_state[adapter_name], adapter_name=adapter_name)

        if hasattr(transformer, "set_adapter"):
            transformer.set_adapter("rgb")

        logger.info("Dual LoRA adapters loaded (no fuse; branch adapter switched inside model forward)")
    else:
        logger.warning(f"lora_adapters.pt not found at {lora_path}; running without trained LoRA adapters")

    # Load ZCL control-link weights (trained bidirectional cross-modal links)
    zcl_file = os.path.join(lora_path, "zcl_links.pt")
    if os.path.exists(zcl_file):
        logger.info(f"Loading ZCL weights from {zcl_file}")
        zcl_state = torch.load(zcl_file, map_location="cpu", weights_only=True)
        if hasattr(transformer, "zcl_rgb_from_xyz") and "zcl_rgb_from_xyz" in zcl_state:
            transformer.zcl_rgb_from_xyz.load_state_dict(zcl_state["zcl_rgb_from_xyz"], strict=False)
        if hasattr(transformer, "zcl_xyz_from_rgb") and "zcl_xyz_from_rgb" in zcl_state:
            transformer.zcl_xyz_from_rgb.load_state_dict(zcl_state["zcl_xyz_from_rgb"], strict=False)
    else:
        logger.warning(f"zcl_links.pt not found at {lora_path}; ZCL links remain zero-initialized")

    # Load XYZ patch embedding if present (trained 16ch conv; from_pretrained inits from RGB patch_embedding)
    pe_xyz_file = os.path.join(lora_path, "patch_embedding_xyz.pt")
    if os.path.exists(pe_xyz_file) and hasattr(transformer, "patch_embedding_xyz"):
        logger.info(f"Loading XYZ patch embedding from {pe_xyz_file}")
        transformer.patch_embedding_xyz.load_state_dict(
            torch.load(pe_xyz_file, map_location="cpu", weights_only=True),
            strict=True,
        )
    elif hasattr(transformer, "patch_embedding_xyz"):
        logger.warning(
            f"patch_embedding_xyz.pt not found at {lora_path}; XYZ patch embedding remains from_pretrained init"
        )

    # ZCL and patch_embedding_xyz are loaded in fp32; cast to inference dtype so
    # forward (conv3d / linear) doesn't hit input (bf16) vs weight/bias (float) mismatch.
    if hasattr(transformer, "zcl_rgb_from_xyz"):
        transformer.zcl_rgb_from_xyz.to(dtype=dtype)
    if hasattr(transformer, "zcl_xyz_from_rgb"):
        transformer.zcl_xyz_from_rgb.to(dtype=dtype)
    if hasattr(transformer, "patch_embedding_xyz"):
        transformer.patch_embedding_xyz.to(dtype=dtype)

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
    """Run a single inference pass and return (latents_rgb, latents_xyz)."""
    image = load_image(image=image_path)

    max_area = 480 * 720
    aspect_ratio = image.height / image.width
    mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
    resolved_height = int(height or (round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value))
    resolved_width = int(width or (round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value))
    image = image.resize((resolved_width, resolved_height))

    with torch.inference_mode():
        latents_rgb, latents_xyz = pipe(
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
            return_dict=False,
        )

    # Each latent is [B, C, F, H, W] — squeeze batch dim
    latents_rgb = latents_rgb.squeeze(0)  # [C, F, H, W]
    latents_xyz = latents_xyz.squeeze(0)  # [C, F, H, W]

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
    run_out = resolve_output_dir(args.out, args.result_name)
    print(f"[Info] Output directory: {run_out}")

    if args.clip_dir is not None:
        prompt_list, image_list, clip_dirs = load_from_clip_dir(args.clip_dir)
    else:
        if args.prompt is None or args.image is None:
            raise ValueError("Provide --clip_dir, or both --prompt and --image")
        prompt_list = load_list(args.prompt)
        image_list = load_list(args.image)
        clip_dirs = None

    if args.merge_output and (clip_dirs is None or not clip_dirs):
        raise ValueError("--merge_output requires --clip_dir (GT video.mp4/xyz.mp4 per clip)")

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
            zcl_layers=args.zcl_layers,
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

        # Optional: 2x2 merged video (top: GT RGB | GT XYZ, bottom: Pred RGB | Pred XYZ)
        if args.merge_output and clip_dirs is not None:
            clip_d = clip_dirs[i]
            gt_rgb = clip_d / "video.mp4"
            gt_xyz = clip_d / "xyz.mp4"
            if gt_rgb.exists() and gt_xyz.exists():
                merged_path = os.path.join(run_out, f'{i:05d}_merged.mp4')
                print(f"  [{i}] Writing 2x2 merged video (GT top, pred bottom)...")
                write_2x2_video(
                    str(gt_rgb), str(gt_xyz),
                    rgb_frames, xyz_frames,
                    merged_path, args.fps,
                )
                print(f"  Saved merged: {merged_path}")
            else:
                print(f"  [{i}] Skip merged (missing {gt_rgb} or {gt_xyz})")

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
    parser.add_argument("--merge_output", action="store_true",
                        help="Write an extra 2x2 video per sample: top row GT RGB | GT XYZ, bottom row Pred RGB | Pred XYZ. Requires --clip_dir with video.mp4 and xyz.mp4 in each clip.")
    parser.add_argument("--idx", type=int, default=-1, help="Process only this index (-1 for all)")
    parser.add_argument("--shard_id", type=int, default=0, help="Shard index for multi-GPU inference (0-based)")
    parser.add_argument("--num_shards", type=int, default=1, help="Total number of shards (= number of GPUs)")
    parser.add_argument("--model_path", type=str, default="pretrained/Wan2.1-I2V-14B-480P-Diffusers",
                        help="Path to pretrained Wan model")
    parser.add_argument("--weights_path", type=str, default=None,
                        help="Path to training checkpoint/adapter directory (recommended, e.g. training/checkpoints/step-000010)")
    parser.add_argument("--lora_path", type=str, default=None,
                        help="Backward-compatible alias of --weights_path")
    parser.add_argument("--out", type=str, default="results", help="Output directory prefix")
    parser.add_argument("--result_name", type=str, default=None,
                        help="Explicit result folder name (e.g. step-800 -> results/result-step-800). If not set, a timestamp suffix is used")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames to generate")
    parser.add_argument("--height", type=int, default=None, help="Output video height (must match model constraints)")
    parser.add_argument("--width", type=int, default=None, help="Output video width (must match model constraints)")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Diffusion steps")
    parser.add_argument("--guidance_scale", type=float, default=5.0, help="CFG scale")
    parser.add_argument("--fps", type=int, default=24, help="Output video FPS")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--rank", type=int, default=64, help="LoRA rank")
    parser.add_argument(
        "--zcl_layers",
        type=str,
        default="3,11,19,27,35",
        help="Comma-separated DiT block indices for ZCL, e.g. 3,11,19,27,35",
    )
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
    args.zcl_layers = [int(x.strip()) for x in args.zcl_layers.split(",") if x.strip()]
    main(args)
