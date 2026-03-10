"""4DNeX inference pipeline: dual-modality (RGB + Pointmap) latent preparation.

WanSameRopeImageToVideoPipeline extends WanImageToVideoPipeline with:
- Doubled latent width for dual modality
- Uniform pointmap as initial geometric condition
- Three-level mask (1.0 / 0.5 / 0.0)
"""

from typing import List, Optional, Tuple, Union

import torch

from diffusers import (
    AutoencoderKLWan,
    FlowMatchEulerDiscreteScheduler,
    WanImageToVideoPipeline,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.wan.pipeline_wan_i2v import retrieve_latents

from transformers import AutoTokenizer, CLIPImageProcessor, CLIPVisionModel, UMT5EncoderModel
from typing_extensions import override

from core.models.wan import WanTransformer3DModelDembSameRope
from core.datasets.utils import generate_uniform_pointmap


class WanSameRopeImageToVideoPipeline(WanImageToVideoPipeline):
    """4DNeX Modified: Pipeline with dual-modality latent preparation.

    Key differences from original diffusers WanImageToVideoPipeline:
    1. Width doubled: latent_width = width * 2 (for RGB + Pointmap concatenation)
    2. Uniform pointmap: generate_uniform_pointmap() as initial condition
    3. Three-level mask: 1.0 (known RGB), 0.5 (partial Pointmap), 0.0 (unknown)
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        text_encoder: UMT5EncoderModel,
        image_encoder: CLIPVisionModel,
        image_processor: CLIPImageProcessor,
        transformer: WanTransformer3DModelDembSameRope,
        vae: AutoencoderKLWan,
        scheduler: FlowMatchEulerDiscreteScheduler,
    ):
        super().__init__(tokenizer, text_encoder, image_encoder, image_processor, transformer, vae, scheduler)

    @override
    def prepare_latents(
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
        latents: Optional[torch.Tensor] = None,
        last_image: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare dual-modality latents (RGB + Pointmap).

        Modifications from base class:
        - latent_width *= 2 (double width for dual modality)
        - Append uniform_pointmap to image
        - Mask value 0.5 for pointmap half
        """
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        latent_height = height // self.vae_scale_factor_spatial
        # Double width for dual-modality
        latent_width = width * 2 // self.vae_scale_factor_spatial

        shape = (batch_size, num_channels_latents, num_latent_frames, latent_height, latent_width)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)

        # Concatenate uniform pointmap along width
        image = image.unsqueeze(2)
        pointmap = generate_uniform_pointmap(height, width)
        pointmap = torch.from_numpy(pointmap).to(device=device, dtype=dtype).permute(2, 0, 1)[None, :, None, :, :] * 2 - 1
        image = torch.concat([image, pointmap], dim=4)  # Now image is [B, 3, 1, H, W*2]

        if last_image is None:
            video_condition = torch.cat(
                [image, image.new_zeros(image.shape[0], image.shape[1], num_frames - 1, height, width * 2)], dim=2
            )
        else:
            last_image = last_image.unsqueeze(2)
            video_condition = torch.cat(
                [image, image.new_zeros(image.shape[0], image.shape[1], num_frames - 2, height, width * 2), last_image],
                dim=2,
            )
        video_condition = video_condition.to(device=device, dtype=self.vae.dtype)

        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )

        if isinstance(generator, list):
            latent_condition = [
                retrieve_latents(self.vae.encode(video_condition), sample_mode="argmax") for _ in generator
            ]
            latent_condition = torch.cat(latent_condition)
        else:
            latent_condition = retrieve_latents(self.vae.encode(video_condition), sample_mode="argmax")
            latent_condition = latent_condition.repeat(batch_size, 1, 1, 1, 1)

        latent_condition = latent_condition.to(dtype)
        latent_condition = (latent_condition - latents_mean) * latents_std

        # Three-level mask: 1.0 (known RGB), 0.5 (partial Pointmap), 0.0 (unknown)
        mask_lat_size = torch.ones(batch_size, 1, num_frames, latent_height, latent_width)

        if last_image is None:
            mask_lat_size[:, :, list(range(1, num_frames))] = 0
        else:
            mask_lat_size[:, :, list(range(1, num_frames - 1))] = 0
        first_frame_mask = mask_lat_size[:, :, 0:1]
        # Mark pointmap half (right half) as 0.5 instead of 1.0
        first_frame_mask[:, :, :, :, latent_condition.shape[4] // 2:] = 0.5
        first_frame_mask = torch.repeat_interleave(first_frame_mask, dim=2, repeats=self.vae_scale_factor_temporal)
        mask_lat_size = torch.concat([first_frame_mask, mask_lat_size[:, :, 1:, :]], dim=2)
        mask_lat_size = mask_lat_size.view(batch_size, -1, self.vae_scale_factor_temporal, latent_height, latent_width)
        mask_lat_size = mask_lat_size.transpose(1, 2)
        mask_lat_size = mask_lat_size.to(latent_condition.device)

        return latents, torch.concat([mask_lat_size, latent_condition], dim=1)
