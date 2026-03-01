"""One4D inference script: generate RGB video + XYZ pointmap from a single image."""

import argparse
import os
import pickle

import imageio
import numpy as np
import torch

from core.inference.one4d import generate_video_one4d
from core.dataclass import Pointmap
from core.tokenizer.wan import WanTokenizer


def decode_and_save(latents, tokenizer, save_path, mode='rgb'):
    """Decode latents and save as video.

    Args:
        latents: [16, F, H, W] latent tensor (no batch dim).
        tokenizer: WanTokenizer instance for VAE decoding.
        save_path: Path to save the mp4 file.
        mode: 'rgb' for standard decode, 'xyz' for XYZ with denormalization.

    Returns:
        frames: [F, H, W, 3] numpy array in [0, 1].
    """
    latents = latents[None]  # Add batch dim → [1, 16, F, H, W]

    if mode == 'xyz':
        # Apply dataset-specific denormalization for XYZ before VAE decode
        encoded_pm_mean = -0.13
        encoded_pm_std = 1.70
        latents = latents * encoded_pm_std + encoded_pm_mean

    frames = tokenizer.decode(latents)  # [F, H, W, 3] in [0, 1]
    mp4_path = save_path if save_path.endswith('.mp4') else save_path.rsplit('.', 1)[0] + '.mp4'
    imageio.mimwrite(mp4_path, (frames * 255).clip(0, 255).astype(np.uint8), fps=24)
    return frames


def save_pointmap(xyz_frames, rgb_frames, save_path):
    """Save combined RGB + XYZ as a Pointmap pickle.

    Args:
        xyz_frames: [F, H, W, 3] numpy array of XYZ coordinates.
        rgb_frames: [F, H, W, 3] numpy array of RGB values in [0, 1].
        save_path: Path to save the pickle file.
    """
    pm = Pointmap()
    pm.init_dummy(xyz_frames.shape[0], xyz_frames.shape[1], xyz_frames.shape[2])
    pm.pcd = xyz_frames.reshape(*pm.pcd.shape)
    pm.rgb = rgb_frames.clip(min=0, max=1)
    pm.colors = pm.rgb.reshape(*pm.colors.shape)
    with open(save_path, 'wb') as f:
        pickle.dump(pm, f)


def main(args):
    prompt_list = []
    with open(args.prompt, 'r') as f:
        for line in f.readlines():
            prompt_list.append(line.strip())

    image_list = []
    with open(args.image, 'r') as f:
        for line in f.readlines():
            image_list.append(line.strip())

    assert len(prompt_list) == len(image_list), \
        f"Prompt count ({len(prompt_list)}) != image count ({len(image_list)})"

    # Load tokenizer for decoding
    vae_path = os.path.join(args.model_path, 'vae')
    tokenizer = WanTokenizer(model_path=vae_path)

    os.makedirs(args.out, exist_ok=True)

    for i in range(len(prompt_list)):
        if args.idx != -1 and i != args.idx:
            continue

        prompt, image_path = prompt_list[i], image_list[i]
        suffix = 'POINTMAP_STYLE.'
        prompt = prompt + ' ' + suffix

        print(f"[{i}/{len(prompt_list)}] Generating: {prompt[:60]}...")

        latents_rgb, latents_xyz = generate_video_one4d(
            prompt=prompt,
            image_or_video_path=image_path,
            model_path=args.model_path,
            one4d_weights_path=args.weights_path,
            num_frames=args.num_frames,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            rank=args.rank,
            lora_alpha=args.lora_alpha,
            dtype=torch.bfloat16,
            offload_mode=args.offload_mode,
        )

        # Decode and save RGB video
        rgb_path = os.path.join(args.out, f'{i:05d}_rgb.mp4')
        rgb_frames = decode_and_save(latents_rgb, tokenizer, rgb_path, mode='rgb')

        # Decode and save XYZ pointmap video
        xyz_path = os.path.join(args.out, f'{i:05d}_xyz.mp4')
        xyz_frames = decode_and_save(latents_xyz, tokenizer, xyz_path, mode='xyz')

        # Save combined pointmap
        pkl_path = os.path.join(args.out, f'{i:05d}.pkl')
        save_pointmap(xyz_frames, rgb_frames, pkl_path)

        print(f"  Saved: {rgb_path}, {xyz_path}, {pkl_path}")

        # Optional: post-processing with camera parameter optimization
        if args.optimize:
            from pm_registration import optimise_xyz_batch, depth_to_3d_points
            print("  Running post-optimization...")
            xyz_tensor = torch.from_numpy(xyz_frames).cuda()
            # Center XY coordinates
            xyz_tensor[..., :2] = xyz_tensor[..., :2] - 0.5
            depth_map, K, R, t = optimise_xyz_batch(xyz_tensor, n_iters=args.opt_iters)

            # Reconstruct optimized pointmap
            updated_pcd = []
            for fi in range(xyz_tensor.shape[0]):
                pts = depth_to_3d_points(depth_map[fi], K, R[fi], t[fi])
                updated_pcd.append(pts)
            updated_pcd = np.stack(updated_pcd)

            pm = pickle.load(open(pkl_path, 'rb'))
            pm.pcd = updated_pcd
            pm.depth = depth_map.cpu().numpy()
            pm.cams2world = np.eye(4)[None].repeat(xyz_tensor.shape[0], axis=0)
            for fi in range(xyz_tensor.shape[0]):
                pm.cams2world[fi, :3, :3] = R[fi].cpu().numpy()
                pm.cams2world[fi, :3, 3] = t[fi].cpu().numpy()
            pm.K = K.cpu().numpy()[None].repeat(xyz_tensor.shape[0], axis=0)

            opt_pkl_path = os.path.join(args.out, f'{i:05d}_optimized.pkl')
            with open(opt_pkl_path, 'wb') as f:
                pickle.dump(pm, f)
            print(f"  Saved optimized: {opt_pkl_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="One4D: Generate RGB + XYZ from a single image")
    parser.add_argument("--prompt", type=str, required=True, help="Path to prompt list file")
    parser.add_argument("--image", type=str, required=True, help="Path to image list file")
    parser.add_argument("--idx", type=int, default=-1, help="Process only this index (-1 for all)")
    parser.add_argument("--model_path", type=str, default="pretrained/Wan2.1-I2V-14B-480P-Diffusers",
                        help="Path to pretrained Wan model")
    parser.add_argument("--weights_path", type=str, required=True,
                        help="Path to One4D trained weights (dir or .safetensors file)")
    parser.add_argument("--out", type=str, default="results/one4d", help="Output directory")
    parser.add_argument("--num_frames", type=int, default=49, help="Number of frames to generate")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Diffusion steps")
    parser.add_argument("--guidance_scale", type=float, default=5.0, help="CFG scale")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--rank", type=int, default=64, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=float, default=32, help="LoRA alpha")
    parser.add_argument("--offload_mode", type=str, default="model",
                        choices=["model", "sequential", "none"],
                        help="CPU offload strategy: "
                             "'model' = whole-model offload (~28GB), "
                             "'sequential' = per-layer offload (~8GB, slower), "
                             "'none' = all on GPU (fastest, ~30GB+)")
    parser.add_argument("--optimize", action="store_true",
                        help="Run post-optimization on generated pointmaps")
    parser.add_argument("--opt_iters", type=int, default=1500,
                        help="Number of optimization iterations")
    args = parser.parse_args()
    main(args)
