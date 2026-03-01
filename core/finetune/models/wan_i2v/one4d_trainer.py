"""
One4D: Unified Framework for Joint RGB Video Generation and 4D Geometric Reconstruction

This module implements the One4D architecture on top of the 4DNeX codebase using PEFT LoRA:
- Decoupled LoRA Control (DLC): Two PEFT LoRA adapters ("rgb" and "xyz") on the same base model
- Zero-initialized Control Links (ZCL) at output level for cross-modal consistency
- Unified Masked Conditioning (UMC): Condition injection only into RGB branch
- Two full forward passes per timestep (one per adapter) for gradient checkpointing compatibility

Reference: arXiv 2511.18922
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import os

from diffusers import (
    AutoencoderKLWan,
    FlowMatchEulerDiscreteScheduler,
    WanImageToVideoPipeline,
    WanTransformer3DModel,
)
from diffusers.configuration_utils import register_to_config
from diffusers.utils import logging
from diffusers.utils.torch_utils import randn_tensor
from diffusers.models.embeddings import get_1d_rotary_pos_embed
from diffusers.models.modeling_utils import ModelMixin
from PIL import Image
import numpy as np
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from transformers import AutoTokenizer, CLIPImageProcessor, CLIPVisionModel, UMT5EncoderModel
from typing_extensions import override

from core.finetune.schemas import Wan_Components as Components
from core.finetune.trainer import Trainer
from core.finetune.utils import unwrap_model, cast_training_params
from core.finetune.models.wan_i2v.sft_trainer import retrieve_latents

from ..utils import register


logger = logging.get_logger(__name__)


# =============================================================================
# Building Blocks
# =============================================================================

class ZeroInitControlLink(nn.Module):
    """Zero-initialized linear layer for cross-modal control.

    Applied at the output level to enable interaction between RGB and XYZ branches.
    Initialized to zero so training starts with independence; links gradually learn alignment.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)
        nn.init.zeros_(self.linear.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


# =============================================================================
# RoPE (standard single-modality, no width doubling)
# =============================================================================

class WanRotaryPosEmbOne4D(nn.Module):
    """Standard RoPE for single-modality input (no width//2 doubling)."""

    def __init__(
        self, attention_head_dim: int, patch_size: Tuple[int, int, int],
        max_seq_len: int, theta: float = 10000.0
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
                dim, max_seq_len, theta,
                use_real=False, repeat_interleave_real=False, freqs_dtype=torch.float64,
            )
            freqs.append(freq)
        self.freqs = torch.cat(freqs, dim=1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        ppf = num_frames // p_t
        pph = height // p_h
        ppw = width // p_w

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
        freqs = torch.cat([freqs_f, freqs_h, freqs_w], dim=-1).reshape(1, 1, ppf * pph * ppw, -1)
        return freqs


# =============================================================================
# One4D Transformer Model (inherits WanTransformer3DModel, uses PEFT LoRA)
# =============================================================================

class WanTransformer3DModelOne4D(WanTransformer3DModel, ModelMixin):
    """One4D Transformer: dual-branch processing using PEFT LoRA.

    Inherits the full WanTransformer3DModel (same blocks, same forward).
    Adds:
    - patch_embedding_xyz: separate patch embedding for XYZ branch (16 channels)
    - WanRotaryPosEmbOne4D: single-modality RoPE (no width doubling)
    - ZCL: zero-initialized control links at output level
    - Two PEFT LoRA adapters ("rgb" and "xyz") added by trainer

    Forward does two full passes through base model blocks (one per adapter).
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

        # XYZ branch patch embedding (16 channels, no condition)
        self.patch_embedding_xyz = nn.Conv3d(
            out_channels, inner_dim, kernel_size=patch_size, stride=patch_size,
        )

        # Override base model's RoPE with single-modality version
        self.rope = WanRotaryPosEmbOne4D(attention_head_dim, patch_size, rope_max_seq_len)

        # Output-level ZCL (2 modules, bidirectional)
        self.zcl_rgb_from_xyz = ZeroInitControlLink(inner_dim)
        self.zcl_xyz_from_rgb = ZeroInitControlLink(inner_dim)

    def _run_blocks(self, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb):
        """Run all transformer blocks on hidden_states."""
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.blocks:
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb,
                )
        else:
            for block in self.blocks:
                hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)
        return hidden_states

    def _output_proj(
        self, hidden_states, temb,
        batch_size, post_patch_num_frames, post_patch_height, post_patch_width,
    ):
        p_t, p_h, p_w = self.config.patch_size

        shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)
        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)

        hidden_states = (
            self.norm_out(hidden_states.float()) * (1 + scale) + shift
        ).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width,
            p_t, p_h, p_w, -1
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)
        return output

    def forward(
        self,
        hidden_states_rgb: torch.Tensor,
        hidden_states_xyz: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Dual-branch forward with two processing modes:

        Training (grad enabled): Two full forward passes (one per adapter).
            Gradient checkpointing requires consistent adapter state within each pass.

        Inference (no grad): Sequential per-block processing.
            Runs RGB block → XYZ block for each layer, sharing frozen base weights.
            Saves memory by not keeping two full sets of intermediate activations.

        Args:
            hidden_states_rgb: [B, 36, F, H, W] - noisy RGB latent + condition
            hidden_states_xyz: [B, 16, F, H, W] - noisy XYZ latent
        Returns:
            pred_rgb, pred_xyz: [B, 16, F, H, W] each
        """
        batch_size = hidden_states_xyz.shape[0]
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = hidden_states_xyz.shape[2] // p_t
        post_patch_height = hidden_states_xyz.shape[3] // p_h
        post_patch_width = hidden_states_xyz.shape[4] // p_w

        # 1. RoPE (single-modality, computed from XYZ shape)
        rotary_emb = self.rope(hidden_states_xyz)

        # 2. Patch embedding
        h_rgb = self.patch_embedding(hidden_states_rgb).flatten(2).transpose(1, 2)  # [B, N, D]
        h_xyz = self.patch_embedding_xyz(hidden_states_xyz).flatten(2).transpose(1, 2)  # [B, N, D]

        # 3. Condition embedding (shared, computed once)
        temb, timestep_proj, enc_hs, enc_hs_img = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image,
        )
        timestep_proj = timestep_proj.unflatten(1, (6, -1))
        if enc_hs_img is not None:
            enc_hs = torch.concat([enc_hs_img, enc_hs], dim=1)

        if torch.is_grad_enabled() and self.gradient_checkpointing:
            # Training mode: two full forward passes for gradient checkpointing safety
            self.set_adapter("rgb")
            h_rgb = self._run_blocks(h_rgb, enc_hs, timestep_proj, rotary_emb)

            self.set_adapter("xyz")
            h_xyz = self._run_blocks(h_xyz, enc_hs, timestep_proj, rotary_emb)
        else:
            # Inference mode: per-block interleaved processing (memory efficient)
            for block in self.blocks:
                self.set_adapter("rgb")
                h_rgb = block(h_rgb, enc_hs, timestep_proj, rotary_emb)
                self.set_adapter("xyz")
                h_xyz = block(h_xyz, enc_hs, timestep_proj, rotary_emb)

        # 6. Output-level ZCL (bidirectional, on hidden_states before unpatchify)
        h_rgb_linked = h_rgb + self.zcl_rgb_from_xyz(h_xyz)
        h_xyz_linked = h_xyz + self.zcl_xyz_from_rgb(h_rgb)

        # 7. Output projection & unpatchify
        out_rgb = self._output_proj(
            h_rgb_linked, temb, batch_size,
            post_patch_num_frames, post_patch_height, post_patch_width,
        )
        out_xyz = self._output_proj(
            h_xyz_linked, temb, batch_size,
            post_patch_num_frames, post_patch_height, post_patch_width,
        )

        return out_rgb, out_xyz

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """Load from WanTransformer3DModel base weights.

        super().from_pretrained() uses init_empty_weights() (meta device) and only
        materializes params that exist in the checkpoint. One4D-specific params
        (patch_embedding_xyz, zcl_*) are not in the base checkpoint, so they remain
        on meta device. We must materialize them explicitly afterward.
        """
        try:
            model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)

            # Materialize One4D-specific params that stayed on meta device
            # (not present in base WanTransformer3DModel checkpoint)
            meta_params_found = False
            for name, param in list(model.named_parameters()):
                if param.is_meta:
                    meta_params_found = True
                    if 'zcl_' in name:
                        # ZCL should be zero-initialized
                        value = torch.zeros(param.shape, dtype=model.dtype)
                    elif 'patch_embedding_xyz' in name and 'weight' in name:
                        # Initialize from base patch_embedding (first 16 channels)
                        base_pe = model.patch_embedding.weight
                        if not base_pe.is_meta:
                            value = base_pe.data[:, :param.shape[1]].clone().to(model.dtype)
                        else:
                            value = torch.empty(param.shape, dtype=model.dtype)
                            nn.init.kaiming_uniform_(value, a=math.sqrt(5))
                    elif 'patch_embedding_xyz' in name and 'bias' in name:
                        base_bias = model.patch_embedding.bias
                        if base_bias is not None and not base_bias.is_meta:
                            value = base_bias.data.clone().to(model.dtype)
                        else:
                            value = torch.zeros(param.shape, dtype=model.dtype)
                    else:
                        value = torch.empty(param.shape, dtype=model.dtype)

                    # Navigate to parent module and replace the parameter
                    parts = name.split('.')
                    parent = model
                    for part in parts[:-1]:
                        parent = getattr(parent, part)
                    setattr(parent, parts[-1], nn.Parameter(value))

            if meta_params_found:
                logger.info("Materialized One4D-specific meta tensors (patch_embedding_xyz, zcl_*)")

            logger.info("Loaded One4D checkpoint directly.")
            return model
        except Exception as e:
            logger.info(f"Direct load failed ({e}), loading from base WanTransformer3DModel...")

        base_model = WanTransformer3DModel.from_pretrained(
            pretrained_model_name_or_path, **kwargs,
        )
        config = dict(base_model.config)

        # Remove internal keys
        for k in ['_class_name', '_diffusers_version', '_name_or_path']:
            config.pop(k, None)

        model = cls(**config)
        model_dict = model.state_dict()

        base_state = base_model.state_dict()
        filtered_dict = {
            k: v for k, v in base_state.items()
            if k in model_dict and model_dict[k].shape == v.shape
        }
        for k in base_state.keys():
            if k not in filtered_dict:
                logger.info(f"Skipping key {k} due to size mismatch or not in One4D.")

        # Initialize patch_embedding_xyz from base patch_embedding (16ch subset)
        if "patch_embedding.weight" in base_state and "patch_embedding_xyz.weight" in model_dict:
            base_pe_w = base_state["patch_embedding.weight"]  # [D, 36, 1, 2, 2]
            xyz_pe_w = model_dict["patch_embedding_xyz.weight"]  # [D, 16, 1, 2, 2]
            if base_pe_w.shape[0] == xyz_pe_w.shape[0]:
                # Copy the first 16 input channels from base
                filtered_dict["patch_embedding_xyz.weight"] = base_pe_w[:, :xyz_pe_w.shape[1]]
                logger.info("Initialized patch_embedding_xyz.weight from base patch_embedding (first 16ch)")
        if "patch_embedding.bias" in base_state and "patch_embedding_xyz.bias" in model_dict:
            filtered_dict["patch_embedding_xyz.bias"] = base_state["patch_embedding.bias"]

        model_dict.update(filtered_dict)
        model.load_state_dict(model_dict)
        logger.info(f"Loaded {len(filtered_dict)} keys from base model into One4D")

        del base_model, base_state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return model


# =============================================================================
# One4D Inference Pipeline
# =============================================================================

class WanOne4DImageToVideoPipeline(WanImageToVideoPipeline):
    """One4D inference pipeline with dual-branch denoising."""

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        text_encoder: UMT5EncoderModel,
        image_encoder: CLIPVisionModel,
        image_processor: CLIPImageProcessor,
        transformer: WanTransformer3DModelOne4D,
        vae: AutoencoderKLWan,
        scheduler: FlowMatchEulerDiscreteScheduler,
    ):
        super().__init__(
            tokenizer, text_encoder, image_encoder, image_processor,
            transformer, vae, scheduler,
        )

    @override
    def prepare_latents(
        self,
        image,
        batch_size: int,
        num_channels_latents: int = 16,
        height: int = 480,
        width: int = 720,
        num_frames: int = 49,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        last_image: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare dual latents and RGB-only condition."""
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        latent_height = height // self.vae_scale_factor_spatial
        latent_width = width // self.vae_scale_factor_spatial

        shape = (batch_size, num_channels_latents, num_latent_frames, latent_height, latent_width)

        latents_rgb = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        latents_xyz = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        # Condition: only RGB first frame
        image = image.unsqueeze(2)
        video_condition = torch.cat(
            [image, image.new_zeros(image.shape[0], image.shape[1], num_frames - 1, height, width)],
            dim=2,
        )
        video_condition = video_condition.to(device=device, dtype=self.vae.dtype)

        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(device, dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
            1, self.vae.config.z_dim, 1, 1, 1
        ).to(device, dtype)

        latent_condition = retrieve_latents(self.vae.encode(video_condition), sample_mode="argmax")
        latent_condition = latent_condition.repeat(batch_size, 1, 1, 1, 1)
        latent_condition = latent_condition.to(dtype)
        latent_condition = (latent_condition - latents_mean) * latents_std

        # Mask: first frame = 1, rest = 0
        mask_lat_size = torch.ones(batch_size, 1, num_frames, latent_height, latent_width)
        mask_lat_size[:, :, list(range(1, num_frames))] = 0
        first_frame_mask = mask_lat_size[:, :, 0:1]
        first_frame_mask = torch.repeat_interleave(
            first_frame_mask, dim=2, repeats=self.vae_scale_factor_temporal,
        )
        mask_lat_size = torch.concat([first_frame_mask, mask_lat_size[:, :, 1:, :]], dim=2)
        mask_lat_size = mask_lat_size.view(
            batch_size, -1, self.vae_scale_factor_temporal, latent_height, latent_width,
        )
        mask_lat_size = mask_lat_size.transpose(1, 2)
        mask_lat_size = mask_lat_size.to(latent_condition.device)

        condition = torch.concat([mask_lat_size, latent_condition], dim=1)

        return latents_rgb, latents_xyz, condition


# =============================================================================
# One4D Trainer
# =============================================================================

class WanOne4DTrainer(Trainer):
    UNLOAD_LIST = ["text_encoder", "image_encoder", "image_processor"]

    @override
    def load_components(self) -> Dict[str, Any]:
        components = Components()
        model_path = str(self.args.model_path)

        components.pipeline_cls = WanImageToVideoPipeline
        components.tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer")
        components.text_encoder = UMT5EncoderModel.from_pretrained(model_path, subfolder="text_encoder")
        components.transformer = WanTransformer3DModelOne4D.from_pretrained(
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
        """Use PEFT LoRA with two named adapters (rgb, xyz) + ZCL + patch_embedding_xyz."""
        logger.info("Initializing One4D trainable parameters (PEFT dual LoRA + ZCL)")

        weight_dtype = self.state.weight_dtype

        # Freeze everything first
        for attr_name, component in vars(self.components).items():
            if hasattr(component, "requires_grad_"):
                component.requires_grad_(False)

        # Add two PEFT LoRA adapters
        lora_config = LoraConfig(
            r=self.args.rank,
            lora_alpha=self.args.lora_alpha,
            init_lora_weights=True,
            target_modules=self.args.target_modules,
        )
        self.components.transformer.add_adapter(lora_config, adapter_name="rgb")
        self.components.transformer.add_adapter(lora_config, adapter_name="xyz")
        # Enable both adapters for training (both get gradients)
        self.components.transformer.set_adapter(["rgb", "xyz"])
        logger.info(f"Added PEFT LoRA adapters 'rgb' and 'xyz' (rank={self.args.rank})")

        # Unfreeze ZCL + patch_embedding_xyz
        for name, param in self.components.transformer.named_parameters():
            if 'zcl_' in name or 'patch_embedding_xyz' in name:
                param.requires_grad_(True)

        # Log trainable parameter count
        trainable_count = sum(
            p.numel() for p in self.components.transformer.parameters() if p.requires_grad
        )
        total_count = sum(p.numel() for p in self.components.transformer.parameters())
        logger.info(
            f"One4D trainable parameters: {trainable_count:,} / {total_count:,} "
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

        # Register custom saving/loading hooks
        self._register_one4d_hooks(lora_config)

    def _register_one4d_hooks(self, lora_config):
        """Register custom save/load hooks for One4D (PEFT LoRA + ZCL + patch_embedding_xyz)."""

        def save_model_hook(models, weights, output_dir):
            if self.accelerator.is_main_process:
                for model in models:
                    unwrapped = unwrap_model(self.accelerator, model)
                    if isinstance(unwrapped, WanTransformer3DModelOne4D):
                        # Save RGB LoRA
                        rgb_state = get_peft_model_state_dict(unwrapped, adapter_name="rgb")
                        rgb_state_prefixed = {f"rgb.{k}": v for k, v in rgb_state.items()}

                        # Save XYZ LoRA
                        xyz_state = get_peft_model_state_dict(unwrapped, adapter_name="xyz")
                        xyz_state_prefixed = {f"xyz.{k}": v for k, v in xyz_state.items()}

                        # Save ZCL + patch_embedding_xyz
                        extra_state = {
                            k: v for k, v in unwrapped.state_dict().items()
                            if 'zcl_' in k or 'patch_embedding_xyz' in k
                        }

                        # Combine all into one file
                        combined = {**rgb_state_prefixed, **xyz_state_prefixed, **extra_state}
                        save_path = os.path.join(output_dir, "one4d_weights.safetensors")
                        from safetensors.torch import save_file
                        save_file(combined, save_path)
                        logger.info(
                            f"Saved One4D weights: {len(rgb_state)} rgb LoRA + "
                            f"{len(xyz_state)} xyz LoRA + {len(extra_state)} extra to {save_path}"
                        )

                    if weights:
                        weights.pop()

        def load_model_hook(models, input_dir):
            while len(models) > 0:
                model = models.pop()
                unwrapped = unwrap_model(self.accelerator, model)
                if isinstance(unwrapped, WanTransformer3DModelOne4D):
                    load_path = os.path.join(input_dir, "one4d_weights.safetensors")
                    if os.path.exists(load_path):
                        from safetensors.torch import load_file
                        combined = load_file(load_path)

                        # Split RGB LoRA
                        rgb_state = {k[4:]: v for k, v in combined.items() if k.startswith("rgb.")}
                        if rgb_state:
                            set_peft_model_state_dict(unwrapped, rgb_state, adapter_name="rgb")

                        # Split XYZ LoRA
                        xyz_state = {k[4:]: v for k, v in combined.items() if k.startswith("xyz.")}
                        if xyz_state:
                            set_peft_model_state_dict(unwrapped, xyz_state, adapter_name="xyz")

                        # Load ZCL + patch_embedding_xyz
                        extra_state = {
                            k: v for k, v in combined.items()
                            if not k.startswith("rgb.") and not k.startswith("xyz.")
                        }
                        if extra_state:
                            unwrapped.load_state_dict(extra_state, strict=False)

                        logger.info(f"Loaded One4D weights from {load_path}")

        self.accelerator.register_save_state_pre_hook(save_model_hook)
        self.accelerator.register_load_state_pre_hook(load_model_hook)

    @override
    def initialize_pipeline(self) -> WanImageToVideoPipeline:
        pipe = WanOne4DImageToVideoPipeline(
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
        prompt_embedding = batch["prompt_embedding"].to(self.components.transformer.dtype)
        latent = batch["encoded_videos"].to(self.components.transformer.dtype)
        images = batch["images"]
        image_embedding = batch["image_embedding"].to(self.components.transformer.dtype)

        batch_size, num_channels, num_frames, height, width = latent.shape
        vae_scale_factor_temporal = 2 ** sum(self.components.vae.config.temperal_downsample)

        # Split RGB and XYZ latents (concatenated along width in dataset)
        W = width // 2
        latent_rgb = latent[..., :W]
        latent_xyz = latent[..., W:]

        _, seq_len, _ = prompt_embedding.shape
        prompt_embedding = prompt_embedding.view(batch_size, seq_len, -1).to(dtype=latent_rgb.dtype)

        # Build RGB-only condition (UMC)
        image_W = images.shape[-1] // 2
        image_rgb = images[..., :image_W]

        num_real_frames = (num_frames - 1) * vae_scale_factor_temporal + 1
        image_rgb = image_rgb.unsqueeze(2)
        video_condition = torch.cat(
            [image_rgb, image_rgb.new_zeros(
                image_rgb.shape[0], image_rgb.shape[1],
                num_real_frames - 1, image_rgb.shape[3], image_rgb.shape[4],
            )],
            dim=2,
        )
        with torch.no_grad():
            latent_condition = self.encode_video(video_condition)

        # Mask: first frame = 1, rest = 0
        mask_lat_size = torch.ones(
            latent_condition.shape[0], 1, num_real_frames,
            latent_condition.shape[3], latent_condition.shape[4],
        )
        mask_lat_size[:, :, list(range(1, num_real_frames))] = 0
        first_frame_mask = mask_lat_size[:, :, 0:1]
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
        timesteps = self.components.scheduler.timesteps[timesteps_idx].to(device=latent_rgb.device)
        sigmas = self.get_sigmas(timesteps, n_dim=latent_rgb.ndim, dtype=latent_rgb.dtype)

        # Add noise
        noise_rgb = torch.randn_like(latent_rgb)
        noise_xyz = torch.randn_like(latent_xyz)
        noisy_rgb = (1.0 - sigmas) * latent_rgb + sigmas * noise_rgb
        noisy_xyz = (1.0 - sigmas) * latent_xyz + sigmas * noise_xyz

        # Build model inputs
        rgb_input = torch.cat([noisy_rgb, condition], dim=1)  # [B, 36, F, H, W]
        xyz_input = noisy_xyz  # [B, 16, F, H, W]

        # Forward (transformer handles adapter switching internally)
        pred_rgb, pred_xyz = self.components.transformer(
            hidden_states_rgb=rgb_input,
            hidden_states_xyz=xyz_input,
            encoder_hidden_states=prompt_embedding,
            encoder_hidden_states_image=image_embedding,
            timestep=timesteps,
        )

        # Flow matching target
        target_rgb = noise_rgb - latent_rgb
        target_xyz = noise_xyz - latent_xyz

        loss_rgb = torch.mean(
            ((pred_rgb.float() - target_rgb.float()) ** 2).reshape(batch_size, -1), dim=1,
        ).mean()
        loss_xyz = torch.mean(
            ((pred_xyz.float() - target_xyz.float()) ** 2).reshape(batch_size, -1), dim=1,
        ).mean()

        return loss_rgb + loss_xyz

    @override
    def validation_step(
        self, eval_data: Dict[str, Any], pipe: WanOne4DImageToVideoPipeline,
    ) -> List[Tuple[str, Image.Image | List[Image.Image]]]:
        prompt, image, video = eval_data["prompt"], eval_data["image"], eval_data["video"]
        return []


# Register with the model registry
register("wan-i2v-one4d", "lora", WanOne4DTrainer)
