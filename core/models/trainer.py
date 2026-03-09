"""Model architecture and trainer for WanTransformer3DModelDembSameRope.

Ported from the reference 4DNeX project (demb_samerope_trainer.py).
Architecture: single-stream dual-modality (RGB + pointmap concatenated along width)
with learnable domain embeddings, shared RoPE, and standard PEFT LoRA.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image

from diffusers import (
    AutoencoderKLWan,
    FlowMatchEulerDiscreteScheduler,
    WanImageToVideoPipeline,
    WanTransformer3DModel,
)
from diffusers.configuration_utils import register_to_config
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers
from diffusers.utils.torch_utils import randn_tensor
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import get_1d_rotary_pos_embed
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.normalization import FP32LayerNorm
from diffusers.models.modeling_utils import ModelMixin

from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from transformers import AutoTokenizer, CLIPImageProcessor, CLIPVisionModel, UMT5EncoderModel
from typing_extensions import override

from core.schemas import Components
from core.trainer import Trainer
from core.utils import unwrap_model, cast_training_params
from core.datasets.utils import generate_uniform_pointmap


logger = logging.get_logger(__name__)


def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


# =============================================================================
# Attention Processor
# =============================================================================

class WanAttnProcessor2_0:
    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("WanAttnProcessor2_0 requires PyTorch 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        if rotary_emb is not None:

            def apply_rotary_emb(hidden_states: torch.Tensor, freqs: torch.Tensor):
                x_rotated = torch.view_as_complex(hidden_states.to(torch.float64).unflatten(3, (-1, 2)))
                x_out = torch.view_as_real(x_rotated * freqs).flatten(3, 4)
                return x_out.type_as(hidden_states)

            query = apply_rotary_emb(query, rotary_emb)
            key = apply_rotary_emb(key, rotary_emb)

        # I2V task
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img = attn.add_k_proj(encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)
            value_img = attn.add_v_proj(encoder_hidden_states_img)

            key_img = key_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            value_img = value_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)

            hidden_states_img = F.scaled_dot_product_attention(
                query, key_img, value_img, attn_mask=None, dropout_p=0.0, is_causal=False
            )
            hidden_states_img = hidden_states_img.transpose(1, 2).flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


# =============================================================================
# RoPE (width-halving with duplication for dual-modality)
# =============================================================================

class WanRotaryPosEmb(nn.Module):
    def __init__(
        self, attention_head_dim: int, patch_size: Tuple[int, int, int], max_seq_len: int, theta: float = 10000.0
    ):
        super().__init__()

        self.attention_head_dim = attention_head_dim
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len

        h_dim = w_dim = 2 * (attention_head_dim // 6)
        t_dim = attention_head_dim - h_dim - w_dim

        freqs = []
        for dim in [t_dim, h_dim, w_dim]:
            freq = get_1d_rotary_pos_embed(
                dim, max_seq_len, theta, use_real=False, repeat_interleave_real=False, freqs_dtype=torch.float64
            )
            freqs.append(freq)
        self.freqs = torch.cat(freqs, dim=1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        # Modality concat along width: halve width for positional encoding
        width = width // 2
        p_t, p_h, p_w = self.patch_size
        ppf, pph, ppw = num_frames // p_t, height // p_h, width // p_w

        freqs = self.freqs.to(hidden_states.device)
        freqs = freqs.split_with_sizes(
            [
                self.attention_head_dim // 2 - 2 * (self.attention_head_dim // 6),
                self.attention_head_dim // 6,
                self.attention_head_dim // 6,
            ],
            dim=1,
        )

        freqs_f = freqs[0][:ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_h = freqs[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_w = freqs[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)
        # Duplicate freqs for both modality halves
        freqs_f = torch.cat([freqs_f, freqs_f], dim=2)
        freqs_h = torch.cat([freqs_h, freqs_h], dim=2)
        freqs_w = torch.cat([freqs_w, freqs_w], dim=2)
        freqs = torch.cat([freqs_f, freqs_h, freqs_w], dim=-1).reshape(1, 1, ppf * pph * ppw * 2, -1)
        return freqs


# =============================================================================
# Transformer Block
# =============================================================================

class WanTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: str = "rms_norm_across_heads",
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        added_kv_proj_dim: Optional[int] = None,
    ):
        super().__init__()

        # 1. Self-attention
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            processor=WanAttnProcessor2_0(),
        )

        # 2. Cross-attention
        self.attn2 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            added_kv_proj_dim=added_kv_proj_dim,
            added_proj_bias=True,
            processor=WanAttnProcessor2_0(),
        )
        self.norm2 = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()

        # 3. Feed-forward
        self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)

        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
            self.scale_shift_table + temb.float()
        ).chunk(6, dim=1)

        # 1. Self-attention
        norm_hidden_states = (self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(hidden_states)
        attn_output = self.attn1(hidden_states=norm_hidden_states, rotary_emb=rotary_emb)
        hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)

        # 2. Cross-attention
        norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
        attn_output = self.attn2(hidden_states=norm_hidden_states, encoder_hidden_states=encoder_hidden_states)
        hidden_states = hidden_states + attn_output

        # 3. Feed-forward
        norm_hidden_states = (self.norm3(hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa).type_as(
            hidden_states
        )
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states)

        return hidden_states


# =============================================================================
# DembSameRope Transformer Model
# =============================================================================

class WanTransformer3DModelDembSameRope(WanTransformer3DModel, ModelMixin):
    """Wan Transformer with learnable domain embeddings and shared RoPE.

    Single-stream architecture: RGB and pointmap are concatenated along the width
    dimension. Distinguished by learnable_domain_embeddings (one per modality half).
    Uses custom WanTransformerBlock with WanAttnProcessor2_0.
    """

    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = ["patch_embedding", "condition_embedder", "norm"]
    _no_split_modules = ["WanTransformerBlock"]
    _keep_in_fp32_modules = ["time_embedder", "scale_shift_table", "norm1", "norm2", "norm3"]
    _keys_to_ignore_on_load_unexpected = ["norm_added_q"]

    @register_to_config
    def __init__(
        self,
        patch_size: Tuple[int] = (1, 2, 2),
        num_attention_heads: int = 40,
        attention_head_dim: int = 128,
        in_channels: int = 16,
        out_channels: int = 16,
        text_dim: int = 4096,
        freq_dim: int = 256,
        ffn_dim: int = 13824,
        num_layers: int = 40,
        cross_attn_norm: bool = True,
        qk_norm: Optional[str] = "rms_norm_across_heads",
        eps: float = 1e-6,
        image_dim: Optional[int] = None,
        added_kv_proj_dim: Optional[int] = None,
        rope_max_seq_len: int = 1024,
        pos_embed_seq_len: Optional[int] = None,
    ) -> None:
        super().__init__(
            patch_size=patch_size,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            in_channels=in_channels,
            out_channels=out_channels,
            text_dim=text_dim,
            freq_dim=freq_dim,
            ffn_dim=ffn_dim,
            num_layers=num_layers,
            cross_attn_norm=cross_attn_norm,
            qk_norm=qk_norm,
            eps=eps,
            image_dim=image_dim,
            added_kv_proj_dim=added_kv_proj_dim,
            rope_max_seq_len=rope_max_seq_len,
            pos_embed_seq_len=pos_embed_seq_len,
        )

        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels

        # Override RoPE with width-halving version
        self.rope = WanRotaryPosEmb(attention_head_dim, patch_size, rope_max_seq_len)

        # Override blocks with custom WanTransformerBlock
        self.blocks = nn.ModuleList(
            [
                WanTransformerBlock(
                    inner_dim, ffn_dim, num_attention_heads, qk_norm, cross_attn_norm, eps, added_kv_proj_dim
                )
                for _ in range(num_layers)
            ]
        )

        # Learnable domain embeddings: [2, inner_dim] — one for each modality half
        self.learnable_domain_embeddings = nn.Parameter(torch.zeros(2, inner_dim))

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        rotary_emb = self.rope(hidden_states)

        hidden_states = self.patch_embedding(hidden_states)
        # Add domain embeddings to each modality half
        first_half_domain_emb, second_half_domain_emb = self.learnable_domain_embeddings.chunk(2, dim=0)
        hidden_states = torch.cat([
            hidden_states[:, :, :, :, :post_patch_width // 2] + first_half_domain_emb[..., None, None, None],
            hidden_states[:, :, :, :, post_patch_width // 2:] + second_half_domain_emb[..., None, None, None],
        ], dim=4)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image
        )
        timestep_proj = timestep_proj.unflatten(1, (6, -1))

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        # Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.blocks:
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb
                )
        else:
            for block in self.blocks:
                hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)

        # Output norm, projection & unpatchify
        shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)

        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)

        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], **kwargs):
        try:
            model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
            if model.learnable_domain_embeddings.is_meta:
                model.learnable_domain_embeddings = nn.Parameter(
                    torch.zeros(model.learnable_domain_embeddings.shape, dtype=model.dtype)
                ).to(model.device)
                logger.info("Convert Meta learnable domain embeddings to zeros.")
            logger.info("Loaded Custom Model checkpoint directly.")
            return model
        except Exception as e:
            logger.error(f"Failed to load as Custom Model: {e}")
            logger.info("Attempting to load as WanTransformer3DModel and convert...")
            base_model = WanTransformer3DModel.from_pretrained(pretrained_model_name_or_path, **kwargs)
            config = dict(base_model.config)
            model = cls(**config)
            model_dict = model.state_dict()
            filtered_dict = {
                k: v for k, v in base_model.state_dict().items()
                if k in model_dict and model_dict[k].shape == v.shape
            }
            for k in base_model.state_dict().keys():
                if k not in filtered_dict:
                    logger.info(f"Skipping key {k} due to size mismatch.")
            model_dict.update(filtered_dict)
            model.load_state_dict(model_dict)
        return model


# =============================================================================
# Inference Pipeline
# =============================================================================

class WanSameRopeWBWImageToVideoPipeline(WanImageToVideoPipeline):
    """Pipeline with dual-modality latent preparation (RGB + pointmap along width)."""

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
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        latent_height = height // self.vae_scale_factor_spatial
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

        image = image.unsqueeze(2)
        pointmap = generate_uniform_pointmap(height, width)
        pointmap = torch.from_numpy(pointmap).to(device=device, dtype=dtype).permute(2, 0, 1)[None, :, None, :, :] * 2 - 1
        image = torch.concat([image, pointmap], dim=4)
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

        mask_lat_size = torch.ones(batch_size, 1, num_frames, latent_height, latent_width)

        if last_image is None:
            mask_lat_size[:, :, list(range(1, num_frames))] = 0
        else:
            mask_lat_size[:, :, list(range(1, num_frames - 1))] = 0
        first_frame_mask = mask_lat_size[:, :, 0:1]
        first_frame_mask[:, :, :, :, latent_condition.shape[4] // 2:] = 0.5
        first_frame_mask = torch.repeat_interleave(first_frame_mask, dim=2, repeats=self.vae_scale_factor_temporal)
        mask_lat_size = torch.concat([first_frame_mask, mask_lat_size[:, :, 1:, :]], dim=2)
        mask_lat_size = mask_lat_size.view(batch_size, -1, self.vae_scale_factor_temporal, latent_height, latent_width)
        mask_lat_size = mask_lat_size.transpose(1, 2)
        mask_lat_size = mask_lat_size.to(latent_condition.device)

        return latents, torch.concat([mask_lat_size, latent_condition], dim=1)


# =============================================================================
# Trainer
# =============================================================================

class WanDualTrainer(Trainer):
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
    def prepare_trainable_parameters(self):
        """Use PEFT LoRA (single adapter) + learnable_domain_embeddings."""
        logger.info("Initializing trainable parameters (PEFT LoRA + learnable_domain_embeddings)")

        weight_dtype = self.state.weight_dtype

        # Freeze everything first
        for attr_name, component in vars(self.components).items():
            if hasattr(component, "requires_grad_"):
                component.requires_grad_(False)

        # Add single PEFT LoRA adapter
        lora_config = LoraConfig(
            r=self.args.rank,
            lora_alpha=self.args.lora_alpha,
            init_lora_weights=True,
            target_modules=self.args.target_modules,
        )
        self.components.transformer.add_adapter(lora_config)

        # Unfreeze learnable_domain_embeddings
        for name, param in self.components.transformer.named_parameters():
            if 'learnable_domain_embeddings' in name:
                param.requires_grad_(True)
                logger.info(f"Training {name} after adding LoRA")

        # Log trainable parameter count
        trainable_count = sum(
            p.numel() for p in self.components.transformer.parameters() if p.requires_grad
        )
        total_count = sum(p.numel() for p in self.components.transformer.parameters())
        logger.info(
            f"Trainable parameters: {trainable_count:,} / {total_count:,} "
            f"({trainable_count / total_count * 100:.2f}%)"
        )

        # Move non-transformer components to device
        ignore_list = ["transformer"] + self.UNLOAD_LIST
        ignore_set = set(ignore_list)
        components = self.components.model_dump()
        for name, component in components.items():
            if not isinstance(component, type) and hasattr(component, "to"):
                if name not in ignore_set:
                    setattr(self.components, name, component.to(self.accelerator.device, dtype=weight_dtype))

        if self.args.gradient_checkpointing:
            self.components.transformer.enable_gradient_checkpointing()

        # Register save/load hooks compatible with reference project format
        self._register_hooks(lora_config)

    def _register_hooks(self, lora_config):
        """Register save/load hooks that produce reference-compatible weight files.

        Saves:
        - pytorch_lora_weights.safetensors (standard PEFT format)
        - learnable_domain_embeddings.pt (separate file)
        """

        def save_model_hook(models, weights, output_dir):
            if self.accelerator.is_main_process:
                for model in models:
                    unwrapped = unwrap_model(self.accelerator, model)
                    if isinstance(unwrapped, WanTransformer3DModelDembSameRope):
                        # Save LoRA weights via standard PEFT pipeline API
                        transformer_lora_layers = get_peft_model_state_dict(unwrapped)
                        self.components.pipeline_cls.save_lora_weights(
                            output_dir,
                            transformer_lora_layers=transformer_lora_layers,
                        )

                        # Save learnable_domain_embeddings separately
                        demb_path = os.path.join(output_dir, "learnable_domain_embeddings.pt")
                        torch.save(unwrapped.learnable_domain_embeddings.data.cpu(), demb_path)
                        logger.info(f"Saved LoRA weights + learnable_domain_embeddings to {output_dir}")

                    if weights:
                        weights.pop()

        def load_model_hook(models, input_dir):
            while len(models) > 0:
                model = models.pop()
                unwrapped = unwrap_model(self.accelerator, model)
                if isinstance(unwrapped, WanTransformer3DModelDembSameRope):
                    # Load LoRA weights
                    lora_state_dict = self.components.pipeline_cls.lora_state_dict(input_dir)
                    transformer_state_dict = {
                        f'{k.replace("transformer.", "")}': v
                        for k, v in lora_state_dict.items()
                        if k.startswith("transformer.")
                    }
                    incompatible_keys = set_peft_model_state_dict(
                        unwrapped, transformer_state_dict, adapter_name="default"
                    )
                    if incompatible_keys is not None:
                        unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
                        if unexpected_keys:
                            logger.warning(
                                f"Loading adapter weights led to unexpected keys: {unexpected_keys}"
                            )

                    # Load learnable_domain_embeddings
                    demb_path = os.path.join(input_dir, "learnable_domain_embeddings.pt")
                    if os.path.exists(demb_path):
                        demb = torch.load(demb_path, map_location="cpu")
                        unwrapped.learnable_domain_embeddings.data = demb.to(
                            unwrapped.learnable_domain_embeddings.device,
                            unwrapped.learnable_domain_embeddings.dtype,
                        )
                        logger.info(f"Loaded learnable_domain_embeddings from {demb_path}")

        self.accelerator.register_save_state_pre_hook(save_model_hook)
        self.accelerator.register_load_state_pre_hook(load_model_hook)

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
