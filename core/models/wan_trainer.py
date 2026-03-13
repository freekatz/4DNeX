"""WanTrainer: Training loop for One4D dual-branch LoRA fine-tuning on Wan2.1.

Model definitions are in core.models.wan.
Pipeline definition is in core.models.wan_pipeline.
"""

from typing import Any, Dict, List, Tuple

import torch
import numpy as np
from PIL import Image

from diffusers import (
    AutoencoderKLWan,
    FlowMatchEulerDiscreteScheduler,
    WanImageToVideoPipeline,
)
from accelerate.logging import get_logger
from transformers import AutoTokenizer, CLIPImageProcessor, CLIPVisionModel, UMT5EncoderModel
from typing_extensions import override

from core.constants import LOG_LEVEL, LOG_NAME
from core.models.wan import WanTransformer3DModelDualBranch
from core.schemas import Components
from core.trainer import Trainer
from core.utils import unwrap_model


logger = get_logger(LOG_NAME, LOG_LEVEL)


class WanTrainer(Trainer):
    UNLOAD_LIST = ["text_encoder", "image_encoder", "image_processor"]

    @override
    def load_components(self) -> Dict[str, Any]:
        components = Components()
        model_path = str(self.args.model_path)

        components.pipeline_cls = WanImageToVideoPipeline
        components.tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer")
        components.text_encoder = UMT5EncoderModel.from_pretrained(model_path, subfolder="text_encoder")
        components.transformer = WanTransformer3DModelDualBranch.from_pretrained(
            model_path,
            subfolder="transformer",
            zcl_layers=tuple(self.args.zcl_layers),
        )
        components.vae = AutoencoderKLWan.from_pretrained(model_path, subfolder="vae")
        components.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler")
        components.image_encoder = CLIPVisionModel.from_pretrained(model_path, subfolder="image_encoder")
        components.image_processor = CLIPImageProcessor.from_pretrained(model_path, subfolder="image_processor")

        return components

    @override
    def prepare_models(self) -> None:
        self.state.transformer_config = self.components.transformer.config

    @override
    def initialize_pipeline(self) -> WanImageToVideoPipeline:
        pipe = WanImageToVideoPipeline(
            tokenizer=self.components.tokenizer,
            text_encoder=self.components.text_encoder,
            vae=self.components.vae,
            transformer=unwrap_model(self.accelerator, self.components.transformer),
            scheduler=self.components.scheduler,
            image_encoder=self.components.image_encoder,
            image_processor=self.components.image_processor,
        )
        return pipe

    @override
    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        vae = self.components.vae
        video = video.to(vae.device, dtype=vae.dtype)
        latent_dist = vae.encode(video).latent_dist
        latent = latent_dist.sample()
        latents_mean = (
            torch.tensor(vae.config.latents_mean)
            .view(1, vae.config.z_dim, 1, 1, 1)
            .to(latent.device, latent.dtype)
        )
        latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(
            1, vae.config.z_dim, 1, 1, 1
        ).to(latent.device, latent.dtype)
        latent = (latent - latents_mean) * latents_std
        return latent

    @override
    def encode_text(self, prompt: str) -> torch.Tensor:
        prompt_token_ids = self.components.tokenizer(
            prompt, padding="max_length", max_length=512,
            truncation=True, add_special_tokens=True, return_tensors="pt",
        )
        prompt_token_ids = prompt_token_ids.input_ids
        prompt_embedding = self.components.text_encoder(
            prompt_token_ids.to(self.accelerator.device)
        )[0]
        return prompt_embedding

    @override
    def collate_fn(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        ret = {
            "encoded_videos_rgb": [],
            "encoded_videos_xyz": [],
            "prompt_embedding": [],
            "images": [],
            "image_embedding": [],
        }
        for sample in samples:
            ret["encoded_videos_rgb"].append(sample["encoded_video_rgb"])
            ret["encoded_videos_xyz"].append(sample["encoded_video_xyz"])
            ret["prompt_embedding"].append(sample["prompt_embedding"])
            ret["images"].append(sample["image"])
            ret["image_embedding"].append(sample["image_embedding"])

        ret["encoded_videos_rgb"] = torch.stack(ret["encoded_videos_rgb"])
        ret["encoded_videos_xyz"] = torch.stack(ret["encoded_videos_xyz"])
        ret["prompt_embedding"] = torch.stack(ret["prompt_embedding"])
        ret["images"] = torch.stack(ret["images"])
        ret["image_embedding"] = torch.stack(ret["image_embedding"])
        return ret

    def get_sigmas(self, timesteps, n_dim=4, dtype=torch.float32):
        sigmas = self.components.scheduler.sigmas.to(device=self.accelerator.device, dtype=dtype)
        schedule_timesteps = self.components.scheduler.timesteps.to(self.accelerator.device)
        timesteps = timesteps.to(self.accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    @override
    def compute_loss(self, batch) -> torch.Tensor:
        """Dual-branch flow matching loss (One4D architecture).

        RGB and XYZ branches are processed independently with shared timestep
        but independent noise. Conditioning is fed ONLY to the RGB branch.
        """
        prompt_embedding = batch["prompt_embedding"].to(self.components.transformer.dtype)
        latent_rgb = batch["encoded_videos_rgb"].to(self.components.transformer.dtype)
        latent_xyz = batch["encoded_videos_xyz"].to(self.components.transformer.dtype)
        images = batch["images"]
        image_embedding = batch["image_embedding"].to(self.components.transformer.dtype)

        batch_size, num_channels, num_frames, height, width = latent_rgb.shape
        vae_scale_factor_temporal = 2 ** sum(self.components.vae.config.temperal_downsample)

        _, seq_len, _ = prompt_embedding.shape
        prompt_embedding = prompt_embedding.view(batch_size, seq_len, -1).to(dtype=latent_rgb.dtype)

        # ---- Build condition for RGB branch ONLY ----
        num_real_frames = (num_frames - 1) * vae_scale_factor_temporal + 1
        images = images.unsqueeze(2)  # [B, 3, 1, H, W]
        video_condition = torch.cat([
            images,
            images.new_zeros(images.shape[0], images.shape[1], num_real_frames - 1, images.shape[3], images.shape[4]),
        ], dim=2)
        with torch.no_grad():
            latent_condition_rgb = self.encode_video(video_condition)

        # RGB mask: 1.0 for known first frame, 0.0 for the rest
        mask_lat_size = torch.ones(
            batch_size, 1, num_real_frames,
            latent_condition_rgb.shape[3], latent_condition_rgb.shape[4],
        )
        mask_lat_size[:, :, list(range(1, num_real_frames))] = 0
        first_frame_mask = mask_lat_size[:, :, 0:1]
        first_frame_mask = torch.repeat_interleave(
            first_frame_mask, dim=2, repeats=vae_scale_factor_temporal,
        )
        mask_lat_size = torch.concat([first_frame_mask, mask_lat_size[:, :, 1:, :]], dim=2)
        mask_lat_size = mask_lat_size.view(
            batch_size, -1, vae_scale_factor_temporal,
            latent_condition_rgb.shape[-2], latent_condition_rgb.shape[-1],
        )
        mask_lat_size = mask_lat_size.transpose(1, 2)
        mask_lat_size = mask_lat_size.to(latent_condition_rgb)

        condition_rgb = torch.concat([mask_lat_size, latent_condition_rgb], dim=1)

        # XYZ branch gets NO condition — pad with zeros to match channel count
        condition_xyz = torch.zeros_like(condition_rgb)

        # ---- Sample shared timestep ----
        timesteps_idx = torch.randint(
            0, self.components.scheduler.config.num_train_timesteps, (batch_size,),
        )
        timesteps_idx = timesteps_idx.long()
        timesteps = self.components.scheduler.timesteps[timesteps_idx].to(device=latent_rgb.device)
        sigmas = self.get_sigmas(timesteps, n_dim=latent_rgb.ndim, dtype=latent_rgb.dtype)

        # ---- Independent noise per modality ----
        noise_rgb = torch.randn_like(latent_rgb)
        noise_xyz = torch.randn_like(latent_xyz)

        noisy_rgb = (1.0 - sigmas) * latent_rgb + sigmas * noise_rgb
        noisy_xyz = (1.0 - sigmas) * latent_xyz + sigmas * noise_xyz

        target_rgb = noise_rgb - latent_rgb
        target_xyz = noise_xyz - latent_xyz

        # ---- Construct model inputs (noisy_latent + condition along channel dim) ----
        input_rgb = torch.cat([noisy_rgb, condition_rgb], dim=1)
        input_xyz = torch.cat([noisy_xyz, condition_xyz], dim=1)

        # ---- Dual-branch forward pass ----
        pred_rgb, pred_xyz = self.components.transformer(
            hidden_states=input_rgb,
            hidden_states_xyz=input_xyz,
            encoder_hidden_states=prompt_embedding,
            encoder_hidden_states_image=image_embedding,
            timestep=timesteps,
            return_dict=False,
        )

        # ---- Separate losses, summed ----
        loss_rgb = torch.mean(
            ((pred_rgb.float() - target_rgb.float()) ** 2).reshape(batch_size, -1), dim=1,
        ).mean()
        loss_xyz = torch.mean(
            ((pred_xyz.float() - target_xyz.float()) ** 2).reshape(batch_size, -1), dim=1,
        ).mean()

        loss = loss_rgb + loss_xyz

        # Store per-branch losses for logging
        self.state.latest_loss_metrics = {
            "loss_rgb": loss_rgb.item(),
            "loss_xyz": loss_xyz.item(),
            "sampled_timestep": float(timesteps.float().mean().item()),
        }

        return loss

    @override
    def validation_step(
        self, eval_data: Dict[str, Any], pipe: WanImageToVideoPipeline,
    ) -> List[Tuple[str, Image.Image | List[Image.Image]]]:
        prompt, image, video = eval_data["prompt"], eval_data["image"], eval_data["video"]
        return []
