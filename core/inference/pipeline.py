"""Inference: generate dual RGB + XYZ pointmap videos using PEFT LoRA.

Uses WanTransformer3DModelDembSameRope with standard PEFT LoRA loading
(pytorch_lora_weights.safetensors + learnable_domain_embeddings.pt).
"""

import gc
import logging
import os
from typing import Optional

import numpy as np
import torch
from PIL import Image
from transformers import CLIPVisionModel
from diffusers.utils.loading_utils import load_image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_video(
    prompt: str,
    model_path: str,
    lora_path: str,
    image_or_video_path: str,
    num_frames: int = 81,
    width: Optional[int] = None,
    height: Optional[int] = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 5.0,
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 42,
    lora_rank: int = 64,
    offload_mode: str = "model",
    output_on_cpu: bool = True,
):
    """Generate dual RGB + XYZ videos using reference-compatible LoRA weights.

    Loads:
    - WanTransformer3DModelDembSameRope from model_path/transformer
    - learnable_domain_embeddings.pt from lora_path
    - pytorch_lora_weights.safetensors from lora_path (fused at scale 0.5)

    Returns:
        (latents_rgb, latents_xyz): each [16, F, H, W_half] tensor
    """
    from core.models.trainer import (
        WanTransformer3DModelDembSameRope,
        WanSameRopeWBWImageToVideoPipeline,
    )

    with torch.inference_mode():
        # Load image encoder
        image_encoder = CLIPVisionModel.from_pretrained(
            model_path, subfolder="image_encoder", torch_dtype=torch.float32,
        )

        # Load custom transformer
        transformer = WanTransformer3DModelDembSameRope.from_pretrained(
            model_path, subfolder="transformer", torch_dtype=dtype,
        )

        # Load learnable_domain_embeddings from lora_path
        demb_file = os.path.join(lora_path, "learnable_domain_embeddings.pt")
        if os.path.exists(demb_file):
            learnable_domain_embeddings = torch.load(demb_file, map_location="cpu")
            transformer.learnable_domain_embeddings.data = learnable_domain_embeddings.to(
                transformer.device, transformer.dtype
            )
            logger.info(f"Loaded learnable_domain_embeddings from {demb_file}")
        else:
            logger.warning(f"learnable_domain_embeddings.pt not found at {lora_path}, using zeros")

        # Build pipeline
        pipe = WanSameRopeWBWImageToVideoPipeline.from_pretrained(
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

        # Load and resize image
        image = load_image(image=image_or_video_path)

        max_area = 480 * 720
        aspect_ratio = image.height / image.width
        mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
        resolved_height = int(height or (round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value))
        resolved_width = int(width or (round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value))
        image = image.resize((resolved_width, resolved_height))

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

        # Run inference — pipeline returns double-width latent
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

        del pipe, image_encoder, transformer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return latents_rgb, latents_xyz
