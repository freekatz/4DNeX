"""One4D inference pipeline: dual-branch (RGB + XYZ) denoising.

WanDualBranchPipeline extends WanImageToVideoPipeline with a custom __call__
that runs two independent latent streams through the dual-branch transformer,
connected only through ZCL links.

Key differences from the standard pipeline:
- Two sets of noise latents (RGB + XYZ), independently initialised
- Conditioning (first frame + mask) is fed ONLY to the RGB branch
- XYZ branch gets zero-padded condition channels
- Denoising returns two separate latent tensors
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import copy

import torch

from diffusers import (
    AutoencoderKLWan,
    FlowMatchEulerDiscreteScheduler,
    WanImageToVideoPipeline,
)
from diffusers.pipelines.wan.pipeline_wan import WanPipelineOutput
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.wan.pipeline_wan_i2v import retrieve_latents

from transformers import AutoTokenizer, CLIPImageProcessor, CLIPVisionModel, UMT5EncoderModel
from typing_extensions import override

from core.models.wan import WanTransformer3DModelDualBranch


class WanDualBranchPipeline(WanImageToVideoPipeline):
    """One4D dual-branch inference pipeline.

    Overrides __call__ to perform joint RGB+XYZ denoising through the
    dual-branch transformer. Each branch has its own latent stream and
    LoRA adapter; cross-modal information flows only through ZCL links.
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        text_encoder: "UMT5EncoderModel",
        image_encoder: CLIPVisionModel,
        image_processor: CLIPImageProcessor,
        transformer: WanTransformer3DModelDualBranch,
        vae: AutoencoderKLWan,
        scheduler: FlowMatchEulerDiscreteScheduler,
    ):
        super().__init__(tokenizer, text_encoder, image_encoder, image_processor, transformer, vae, scheduler)

    def prepare_dual_latents(
        self,
        image,
        batch_size: int,
        num_channels_latents: int = 16,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare dual-branch latents and conditions.

        Returns:
            latents_rgb: Random noise [B, C, F, H_lat, W_lat]
            latents_xyz: Independent random noise [B, C, F, H_lat, W_lat]
            condition_rgb: [mask + VAE-encoded first frame] for RGB branch
            condition_xyz: Zero-padded (no conditioning) for XYZ branch
        """
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        latent_height = height // self.vae_scale_factor_spatial
        latent_width = width // self.vae_scale_factor_spatial

        shape = (batch_size, num_channels_latents, num_latent_frames, latent_height, latent_width)

        # Independent noise for each branch
        latents_rgb = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        latents_xyz = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        # ---- RGB condition: first frame VAE-encoded + mask ----
        image = image.unsqueeze(2)  # [B, 3, 1, H, W]
        video_condition = torch.cat(
            [image, image.new_zeros(image.shape[0], image.shape[1], num_frames - 1, height, width)], dim=2
        )
        video_condition = video_condition.to(device=device, dtype=self.vae.dtype)

        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(device, dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            device, dtype
        )

        latent_condition = retrieve_latents(self.vae.encode(video_condition), sample_mode="argmax")
        latent_condition = latent_condition.repeat(batch_size, 1, 1, 1, 1)
        latent_condition = latent_condition.to(dtype)
        latent_condition = (latent_condition - latents_mean) * latents_std

        # Mask: 1.0 for first frame, 0.0 for the rest
        mask_lat_size = torch.ones(batch_size, 1, num_frames, latent_height, latent_width)
        mask_lat_size[:, :, list(range(1, num_frames))] = 0
        first_frame_mask = mask_lat_size[:, :, 0:1]
        first_frame_mask = torch.repeat_interleave(first_frame_mask, dim=2, repeats=self.vae_scale_factor_temporal)
        mask_lat_size = torch.concat([first_frame_mask, mask_lat_size[:, :, 1:, :]], dim=2)
        mask_lat_size = mask_lat_size.view(batch_size, -1, self.vae_scale_factor_temporal, latent_height, latent_width)
        mask_lat_size = mask_lat_size.transpose(1, 2)
        mask_lat_size = mask_lat_size.to(device=device, dtype=dtype)

        condition_rgb = torch.concat([mask_lat_size, latent_condition], dim=1)
        condition_xyz = torch.zeros_like(condition_rgb)

        return latents_rgb, latents_xyz, condition_rgb, condition_xyz

    @torch.no_grad()
    def __call__(
        self,
        image,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        image_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "np",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        max_sequence_length: int = 512,
    ):
        """Dual-branch denoising: jointly denoise RGB and XYZ latents.

        Returns:
            If output_type == "latent": WanPipelineOutput with frames = (latents_rgb, latents_xyz)
            Otherwise: WanPipelineOutput with frames = (video_rgb, video_xyz)
        """
        device = self._execution_device
        transformer_dtype = self.transformer.dtype

        # Round num_frames
        num_frames = 1 + (num_frames - 1) // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        # Determine batch_size
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # Encode prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

        # Encode image (CLIP)
        if image_embeds is None:
            image_embeds = self.encode_image(image, device)
        image_embeds = image_embeds.repeat(batch_size, 1, 1)
        image_embeds = image_embeds.to(transformer_dtype)

        # Prepare timesteps — each branch needs its own scheduler instance
        # because scheduler.step() advances an internal step_index.
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        scheduler_xyz = copy.deepcopy(self.scheduler)
        timesteps = self.scheduler.timesteps

        # Preprocess image for VAE
        num_channels_latents = self.vae.config.z_dim
        image_tensor = self.video_processor.preprocess(image, height=height, width=width).to(
            device, dtype=torch.float32
        )

        # Prepare dual latents
        latents_rgb, latents_xyz, condition_rgb, condition_xyz = self.prepare_dual_latents(
            image_tensor,
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            height, width, num_frames,
            torch.float32, device, generator,
        )

        # ---- DEBUG: verify conditioning is valid ----
        print(f"[DEBUG] latents_rgb: shape={latents_rgb.shape}, dtype={latents_rgb.dtype}, "
              f"min={latents_rgb.min():.4f}, max={latents_rgb.max():.4f}, "
              f"nan={latents_rgb.isnan().any()}, inf={latents_rgb.isinf().any()}")
        print(f"[DEBUG] condition_rgb: shape={condition_rgb.shape}, dtype={condition_rgb.dtype}, "
              f"min={condition_rgb.min():.4f}, max={condition_rgb.max():.4f}, "
              f"nan={condition_rgb.isnan().any()}, inf={condition_rgb.isinf().any()}")
        # Mask channels (first 4 of condition_rgb): should have 1.0 for first frame, 0.0 for rest
        mask_ch = condition_rgb[:, :4]
        cond_ch = condition_rgb[:, 4:]
        print(f"[DEBUG] mask channels: nonzero_ratio={mask_ch.abs().gt(0.01).float().mean():.4f}, "
              f"first_frame_mask_vals={mask_ch[0, :, 0, 0, 0].tolist()}")
        print(f"[DEBUG] cond latent channels: abs_mean={cond_ch.abs().mean():.4f}, "
              f"first_frame_abs_mean={cond_ch[:, :, 0].abs().mean():.4f}, "
              f"other_frames_abs_mean={cond_ch[:, :, 1:].abs().mean():.6f}")
        print(f"[DEBUG] condition_xyz: all_zero={condition_xyz.abs().max() == 0}")
        print(f"[DEBUG] prompt_embeds: shape={prompt_embeds.shape}, abs_mean={prompt_embeds.abs().mean():.4f}, "
              f"nan={prompt_embeds.isnan().any()}")
        print(f"[DEBUG] image_embeds: shape={image_embeds.shape}, abs_mean={image_embeds.abs().mean():.4f}, "
              f"nan={image_embeds.isnan().any()}")
        print(f"[DEBUG] transformer dtype={transformer_dtype}, device={device}")
        print(f"[DEBUG] CFG enabled={self.do_classifier_free_guidance}, guidance_scale={guidance_scale}")

        # Check LoRA adapters
        if hasattr(self.transformer, "peft_config"):
            for name, cfg in self.transformer.peft_config.items():
                print(f"[DEBUG] LoRA adapter '{name}': r={cfg.r}, alpha={cfg.lora_alpha}, "
                      f"target_modules={cfg.target_modules}")
        else:
            print("[DEBUG] WARNING: No peft_config found on transformer — LoRA may not be loaded!")

        # Check ZCL link norms
        if hasattr(self.transformer, "zcl_rgb_from_xyz"):
            for k, v in self.transformer.zcl_rgb_from_xyz.items():
                w_norm = v.weight.norm().item()
                print(f"[DEBUG] ZCL rgb_from_xyz[{k}] weight_norm={w_norm:.6f}")
                break  # just check first layer
        if hasattr(self.transformer, "zcl_xyz_from_rgb"):
            for k, v in self.transformer.zcl_xyz_from_rgb.items():
                w_norm = v.weight.norm().item()
                print(f"[DEBUG] ZCL xyz_from_rgb[{k}] weight_norm={w_norm:.6f}")
                break

        # ---- Denoising loop ----
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                timestep = t.expand(latents_rgb.shape[0])

                # Construct model inputs
                input_rgb = torch.cat([latents_rgb, condition_rgb], dim=1).to(transformer_dtype)
                input_xyz = torch.cat([latents_xyz, condition_xyz], dim=1).to(transformer_dtype)

                # Conditional forward (with prompt)
                pred_rgb, pred_xyz = self.transformer(
                    hidden_states=input_rgb,
                    hidden_states_xyz=input_xyz,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    encoder_hidden_states_image=image_embeds,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )

                # CFG: unconditional forward (with negative prompt)
                if self.do_classifier_free_guidance:
                    uncond_rgb, uncond_xyz = self.transformer(
                        hidden_states=input_rgb,
                        hidden_states_xyz=input_xyz,
                        timestep=timestep,
                        encoder_hidden_states=negative_prompt_embeds,
                        encoder_hidden_states_image=image_embeds,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )
                    pred_rgb = uncond_rgb + guidance_scale * (pred_rgb - uncond_rgb)
                    pred_xyz = uncond_xyz + guidance_scale * (pred_xyz - uncond_xyz)

                # Euler step for each branch independently
                latents_rgb = self.scheduler.step(pred_rgb, t, latents_rgb, return_dict=False)[0]
                latents_xyz = scheduler_xyz.step(pred_xyz, t, latents_xyz, return_dict=False)[0]

                # ---- DEBUG: monitor denoising at first, middle, and last step ----
                if i in (0, len(timesteps) // 2, len(timesteps) - 1):
                    print(f"[DEBUG step {i}/{len(timesteps)-1}, t={t.item():.1f}] "
                          f"pred_rgb: min={pred_rgb.min():.4f}, max={pred_rgb.max():.4f}, "
                          f"nan={pred_rgb.isnan().any()}, abs_mean={pred_rgb.abs().mean():.4f}")
                    print(f"  latents_rgb: min={latents_rgb.min():.4f}, max={latents_rgb.max():.4f}, "
                          f"nan={latents_rgb.isnan().any()}, abs_mean={latents_rgb.abs().mean():.4f}")
                    print(f"  pred_xyz: min={pred_xyz.min():.4f}, max={pred_xyz.max():.4f}, "
                          f"nan={pred_xyz.isnan().any()}, abs_mean={pred_xyz.abs().mean():.4f}")
                    print(f"  latents_xyz: min={latents_xyz.min():.4f}, max={latents_xyz.max():.4f}, "
                          f"nan={latents_xyz.isnan().any()}, abs_mean={latents_xyz.abs().mean():.4f}")

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        self._current_timestep = None

        if output_type == "latent":
            video = (latents_rgb, latents_xyz)
        else:
            # Decode each branch separately through VAE
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents_rgb.device, latents_rgb.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
                1, self.vae.config.z_dim, 1, 1, 1
            ).to(latents_rgb.device, latents_rgb.dtype)

            latents_rgb_dec = latents_rgb.to(self.vae.dtype)
            latents_rgb_dec = latents_rgb_dec / latents_std.to(self.vae.dtype) + latents_mean.to(self.vae.dtype)
            video_rgb = self.vae.decode(latents_rgb_dec, return_dict=False)[0]
            video_rgb = self.video_processor.postprocess_video(video_rgb, output_type=output_type)

            latents_xyz_dec = latents_xyz.to(self.vae.dtype)
            latents_xyz_dec = latents_xyz_dec / latents_std.to(self.vae.dtype) + latents_mean.to(self.vae.dtype)
            video_xyz = self.vae.decode(latents_xyz_dec, return_dict=False)[0]
            video_xyz = self.video_processor.postprocess_video(video_xyz, output_type=output_type)

            video = (video_rgb, video_xyz)

        self.maybe_free_model_hooks()

        if not return_dict:
            return video

        return WanPipelineOutput(frames=video)
