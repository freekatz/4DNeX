"""WanTrainer: Training loop for 4DNeX (LoRA fine-tuning on Wan2.1).

Model definitions are in core.models.wan.
Pipeline definition is in core.models.pipeline.
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
from diffusers.utils import logging

from transformers import AutoTokenizer, CLIPImageProcessor, CLIPVisionModel, UMT5EncoderModel
from typing_extensions import override

from core.models.wan import WanTransformer3DModelDembSameRope
from core.schemas import Components
from core.trainer import Trainer
from core.utils import unwrap_model


logger = logging.get_logger(__name__)


class WanTrainer(Trainer):
    UNLOAD_LIST = ["text_encoder", "image_encoder", "image_processor"]

    @override
    def load_components(self) -> Dict[str, Any]:
        components = Components()
        model_path = str(self.args.model_path)

        components.pipeline_cls = WanImageToVideoPipeline
        components.tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer")
        components.text_encoder = UMT5EncoderModel.from_pretrained(model_path, subfolder="text_encoder")
        components.transformer = WanTransformer3DModelDembSameRope.from_pretrained(
            model_path, subfolder="transformer",
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
        ret = {"encoded_videos": [], "prompt_embedding": [], "images": [], "image_embedding": []}
        for sample in samples:
            ret["encoded_videos"].append(sample["encoded_video"])
            ret["prompt_embedding"].append(sample["prompt_embedding"])
            ret["images"].append(sample["image"])
            ret["image_embedding"].append(sample["image_embedding"])

        ret["encoded_videos"] = torch.stack(ret["encoded_videos"])
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
        """Single-stream flow matching loss with dual-modality mask."""
        prompt_embedding = batch["prompt_embedding"].to(self.components.transformer.dtype)
        latent = batch["encoded_videos"].to(self.components.transformer.dtype)
        images = batch["images"]
        image_embedding = batch["image_embedding"].to(self.components.transformer.dtype)

        batch_size, num_channels, num_frames, height, width = latent.shape
        vae_scale_factor_temporal = 2 ** sum(self.components.vae.config.temperal_downsample)

        _, seq_len, _ = prompt_embedding.shape
        prompt_embedding = prompt_embedding.view(batch_size, seq_len, -1).to(dtype=latent.dtype)

        # Build condition from first frame image (already concatenated RGB + pointmap along width)
        num_real_frames = (num_frames - 1) * vae_scale_factor_temporal + 1
        images = images.unsqueeze(2)
        video_condition = torch.cat([
            images,
            images.new_zeros(images.shape[0], images.shape[1], num_real_frames - 1, images.shape[3], images.shape[4]),
        ], dim=2)
        with torch.no_grad():
            latent_condition = self.encode_video(video_condition)

        # Build mask with 0/0.5/1 values
        mask_lat_size = torch.ones(
            latent_condition.shape[0], 1, num_real_frames,
            latent_condition.shape[3], latent_condition.shape[4],
        )
        mask_lat_size[:, :, list(range(1, num_real_frames))] = 0
        first_frame_mask = mask_lat_size[:, :, 0:1]
        # Mark pointmap half of first frame with 0.5
        first_frame_mask[:, :, :, :, latent_condition.shape[4] // 2:] = 0.5
        first_frame_mask = torch.repeat_interleave(
            first_frame_mask, dim=2, repeats=vae_scale_factor_temporal,
        )
        mask_lat_size = torch.concat([first_frame_mask, mask_lat_size[:, :, 1:, :]], dim=2)
        mask_lat_size = mask_lat_size.view(
            batch_size, -1, vae_scale_factor_temporal,
            latent_condition.shape[-2], latent_condition.shape[-1],
        )
        mask_lat_size = mask_lat_size.transpose(1, 2)
        mask_lat_size = mask_lat_size.to(latent_condition)

        condition = torch.concat([mask_lat_size, latent_condition], dim=1)

        # Sample timestep
        timesteps_idx = torch.randint(
            0, self.components.scheduler.config.num_train_timesteps, (batch_size,),
        )
        timesteps_idx = timesteps_idx.long()
        timesteps = self.components.scheduler.timesteps[timesteps_idx].to(device=latent.device)
        sigmas = self.get_sigmas(timesteps, n_dim=latent.ndim, dtype=latent.dtype)

        # Add noise
        noise = torch.randn_like(latent)
        noisy_latents = (1.0 - sigmas) * latent + sigmas * noise
        target = noise - latent

        # Concatenate noisy latent with condition
        latent_model_input = torch.cat([noisy_latents, condition], dim=1)

        # Single forward pass through transformer
        predicted_noise = self.components.transformer(
            hidden_states=latent_model_input,
            encoder_hidden_states=prompt_embedding,
            encoder_hidden_states_image=image_embedding,
            timestep=timesteps,
            return_dict=False,
        )[0]

        loss = torch.mean(
            ((predicted_noise.float() - target.float()) ** 2).reshape(batch_size, -1), dim=1,
        )
        loss = loss.mean()

        return loss

    @override
    def validation_step(
        self, eval_data: Dict[str, Any], pipe: WanImageToVideoPipeline,
    ) -> List[Tuple[str, Image.Image | List[Image.Image]]]:
        prompt, image, video = eval_data["prompt"], eval_data["image"], eval_data["video"]
        return []
