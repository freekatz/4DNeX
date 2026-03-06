"""inference: generate dual RGB + XYZ pointmap videos using PEFT LoRA."""

import copy
import gc
import logging
import os
from typing import Optional

import numpy as np
import torch
from peft import LoraConfig
from transformers import CLIPVisionModel
from diffusers.utils.loading_utils import load_image
from tqdm.auto import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_video(
    prompt: str,
    model_path: str,
    weights_path: str,
    image_or_video_path: str,
    num_frames: int = 49,
    width: Optional[int] = None,
    height: Optional[int] = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 5.0,
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 42,
    rank: int = 64,
    lora_alpha: int = 32,
    target_modules: Optional[list[str]] = None,
    offload_mode: str = "model",
    output_on_cpu: bool = True,
):
    """
    Generate dual RGB + XYZ videos with PEFT LoRA.

    Args:
        weights_path: Path to weights.safetensors or directory containing it.
        rank: LoRA rank (must match training).
        lora_alpha: LoRA alpha (must match training).
        target_modules: LoRA target modules (must match training).
        offload_mode: Memory optimization strategy:
            - "model": enable_model_cpu_offload (whole model on/off GPU, fast but ~28GB peak)
            - "sequential": enable_sequential_cpu_offload (per-submodule offload, slow but ~8GB peak)
            - "none": no offload, everything on GPU (fastest, requires ~30GB+)
    """
    from core.models.trainer import (
        WanDualTransformer3DModel,
        WanDualI2VPipeline,
    )
    from safetensors.torch import load_file

    if target_modules is None:
        target_modules = ["to_q", "to_k", "to_v", "to_out.0", "ffn.net.0.proj", "ffn.net.2"]

    with torch.inference_mode():
        # Load image encoder
        image_encoder = CLIPVisionModel.from_pretrained(
            model_path, subfolder="image_encoder", torch_dtype=torch.float32,
        )

        # Load dual-branch transformer (inherits WanTransformer3DModel)
        transformer = WanDualTransformer3DModel.from_pretrained(
            model_path, subfolder="transformer", torch_dtype=dtype,
        )

        # Add PEFT LoRA adapters (must match training config)
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=lora_alpha,
            init_lora_weights=True,
            target_modules=target_modules,
        )
        transformer.add_adapter(lora_config, adapter_name="rgb")
        transformer.add_adapter(lora_config, adapter_name="xyz")
        transformer.set_adapter(["rgb", "xyz"])

        # Load trained weights
        if os.path.isfile(weights_path):
            weights_file = weights_path
        else:
            weights_file = os.path.join(weights_path, "weights.safetensors")

        if not os.path.exists(weights_file):
            raise FileNotFoundError(
                f"Weights not found at: {weights_file}. "
                "Please provide --weights_path as a file or directory containing weights.safetensors"
            )

        from peft import set_peft_model_state_dict
        combined = load_file(weights_file)

        # Split and load RGB LoRA
        rgb_state = {k[4:]: v for k, v in combined.items() if k.startswith("rgb.")}
        if rgb_state:
            set_peft_model_state_dict(transformer, rgb_state, adapter_name="rgb")
            logger.info(f"Loaded {len(rgb_state)} RGB LoRA weights")

        # Split and load XYZ LoRA
        xyz_state = {k[4:]: v for k, v in combined.items() if k.startswith("xyz.")}
        if xyz_state:
            set_peft_model_state_dict(transformer, xyz_state, adapter_name="xyz")
            logger.info(f"Loaded {len(xyz_state)} XYZ LoRA weights")

        # Load DLC + patch_embedding_xyz
        extra_state = {
            k: v for k, v in combined.items()
            if not k.startswith("rgb.") and not k.startswith("xyz.")
        }
        if extra_state:
            transformer.load_state_dict(extra_state, strict=False)
            logger.info(f"Loaded {len(extra_state)} extra weights (DLC + patch_embedding_xyz)")

        del combined

        # Build pipeline
        pipe = WanDualI2VPipeline.from_pretrained(
            model_path,
            image_encoder=image_encoder,
            transformer=transformer,
            torch_dtype=dtype,
        )
        pipe.transformer.eval()
        if torch.cuda.is_available():
            logger.info("Compiling transformer with torch.compile for faster inference...")
            pipe.transformer = torch.compile(pipe.transformer)
        pipe.text_encoder.eval()
        pipe.image_encoder.eval()
        pipe.vae.eval()
        del image_encoder, transformer
        gc.collect()

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
        # offload_mode == "none": keep everything on GPU

        generator = torch.Generator().manual_seed(seed)

        # Run dual-branch denoising
        latents_rgb, latents_xyz = _run_inference(
            pipe, prompt, image, resolved_height, resolved_width, num_frames,
            num_inference_steps, guidance_scale, generator, dtype,
            offload_mode=offload_mode,
        )

        if output_on_cpu:
            latents_rgb = latents_rgb.cpu()
            latents_xyz = latents_xyz.cpu()

        del pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return latents_rgb, latents_xyz


def _run_inference(
    pipe, prompt, image, height, width, num_frames,
    num_inference_steps, guidance_scale, generator, dtype,
    offload_mode="model",
):
    """Run the dual-branch denoising loop.

    Uses sequential CFG (two separate forward passes for cond/uncond) instead of
    batched CFG to avoid doubling GPU memory. This matches the upstream
    WanImageToVideoPipeline behavior.
    """
    device = pipe._execution_device
    transformer_dtype = pipe.transformer.dtype
    do_cfg = guidance_scale > 1.0

    # Encode text
    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt=prompt,
        do_classifier_free_guidance=do_cfg,
    )

    # Encode image for cross-attention
    image_embeds = pipe.encode_image(image, device)

    # Cast embeddings to transformer dtype to avoid dtype mismatches
    prompt_embeds = prompt_embeds.to(transformer_dtype)
    if negative_prompt_embeds is not None:
        negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)
    image_embeds = image_embeds.to(transformer_dtype)

    # For "model" offload mode, manually manage component placement.
    # The hook chain expects: text_encoder→image_encoder→transformer→vae
    # but we call vae directly (skipping transformer), so image_encoder stays on GPU.
    # For "sequential" mode, accelerate handles per-submodule offload automatically.
    use_manual_offload = offload_mode == "model" and hasattr(pipe, '_all_hooks') and pipe._all_hooks
    if use_manual_offload:
        pipe.image_encoder.to("cpu")
        torch.cuda.empty_cache()

    # Prepare image latent for conditioning
    image_tensor = pipe.video_processor.preprocess(image, height=height, width=width).to(
        device=device, dtype=pipe.vae.dtype
    )

    # Prepare dual latents + condition (calls vae.encode internally)
    latents_rgb, latents_xyz, condition = pipe.prepare_latents(
        image_tensor, batch_size=1, num_channels_latents=16,
        height=height, width=width, num_frames=num_frames,
        dtype=dtype, device=device, generator=generator,
    )
    latents_rgb = latents_rgb.to(transformer_dtype)
    latents_xyz = latents_xyz.to(transformer_dtype)
    condition = condition.to(transformer_dtype)

    # Offload VAE after encoding, before transformer denoising loop
    if use_manual_offload:
        pipe.vae.to("cpu")
        torch.cuda.empty_cache()

    # Scheduler — use two separate instances so each branch gets the full
    # sigma schedule (FlowMatchEulerDiscreteScheduler.step() increments an
    # internal _step_index on every call).
    scheduler_rgb = pipe.scheduler
    scheduler_xyz = copy.deepcopy(pipe.scheduler)
    scheduler_rgb.set_timesteps(num_inference_steps, device=device)
    scheduler_xyz.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler_rgb.timesteps

    # Denoising loop with sequential CFG
    step_iterator = tqdm(
        enumerate(timesteps),
        total=len(timesteps),
        desc="Dual-branch denoising",
        leave=False,
    )

    for i, t in step_iterator:
        if i % 10 == 0:
            logger.info(f"  Denoising step {i}/{len(timesteps)}")

        # Build inputs (batch_size=1, no doubling)
        rgb_input = torch.cat([latents_rgb, condition], dim=1).to(transformer_dtype)  # [1, 36, F, H, W]
        xyz_input = latents_xyz.to(transformer_dtype)  # [1, 16, F, H, W]
        timestep = t.unsqueeze(0)

        # Conditioned forward pass
        pred_rgb, pred_xyz = pipe.transformer(
            hidden_states_rgb=rgb_input,
            hidden_states_xyz=xyz_input,
            encoder_hidden_states=prompt_embeds,
            encoder_hidden_states_image=image_embeds,
            timestep=timestep,
        )

        # Unconditional forward pass (sequential CFG)
        if do_cfg and negative_prompt_embeds is not None:
            pred_rgb_uncond, pred_xyz_uncond = pipe.transformer(
                hidden_states_rgb=rgb_input,
                hidden_states_xyz=xyz_input,
                encoder_hidden_states=negative_prompt_embeds,
                encoder_hidden_states_image=image_embeds,
                timestep=timestep,
            )
            pred_rgb = pred_rgb_uncond + guidance_scale * (pred_rgb - pred_rgb_uncond)
            pred_xyz = pred_xyz_uncond + guidance_scale * (pred_xyz - pred_xyz_uncond)

        # Scheduler step (separate schedulers to avoid double-stepping _step_index)
        latents_rgb = scheduler_rgb.step(pred_rgb, t, latents_rgb).prev_sample
        latents_xyz = scheduler_xyz.step(pred_xyz, t, latents_xyz).prev_sample
        latents_rgb = latents_rgb.to(transformer_dtype)
        latents_xyz = latents_xyz.to(transformer_dtype)

    return latents_rgb[0], latents_xyz[0]
