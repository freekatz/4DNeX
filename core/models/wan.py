"""Wan model definitions for One4D dual-branch architecture.

WanTransformer3DModelDualBranch: Transformer with decoupled RGB/XYZ branches,
separate LoRA adapters, and Zero-initialized Cross-modal Links (ZCL).

Each modality (RGB, XYZ) is processed independently through the same frozen
DiT backbone with its own LoRA adapter. Cross-modal information flows only
through sparse ZCL links at designated layers.
"""

from typing import Any, Dict, Optional, Tuple, Union, cast

import torch
import torch.nn as nn

from diffusers.configuration_utils import register_to_config
from diffusers.utils import logging
from diffusers.utils.constants import USE_PEFT_BACKEND
from diffusers.utils.peft_utils import scale_lora_layers, unscale_lora_layers
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.transformers.transformer_wan import WanTransformer3DModel, WanTransformerBlock


logger = logging.get_logger(__name__)


# Keep the old name as an alias for backward compatibility with trainer imports.
WanTransformer3DModelDembSameRope = None  # Will be set after class definition


class WanTransformer3DModelDualBranch(WanTransformer3DModel, ModelMixin):
    """One4D dual-branch Wan Transformer.

    Key differences from the original diffusers WanTransformer3DModel:
    1. Accepts TWO hidden_states inputs (RGB and XYZ), each [B, in_ch, F, H, W]
    2. Processes each branch through the SAME frozen DiT blocks with separate LoRA adapters
    3. Returns TWO output tensors (predicted velocity for RGB and XYZ)
    4. ZCL links provide sparse bidirectional cross-modal interaction at selected layers
    5. Uses the parent's standard RoPE (no width-halving needed)
    """

    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = ["patch_embedding", "condition_embedder", "norm"]
    _no_split_modules = ["WanTransformerBlock"]
    _keep_in_fp32_modules = ["time_embedder", "scale_shift_table", "norm1", "norm2", "norm3"]
    _keys_to_ignore_on_load_unexpected = ["norm_added_q"]

    @register_to_config
    def __init__(
        self,
        patch_size: Tuple[int, ...] = (1, 2, 2),
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
        zcl_layers: Tuple[int, ...] = (3, 11, 19, 27, 35),
    ) -> None:
        super().__init__(
            patch_size=cast(Tuple[int], patch_size),
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

        # Use the parent's standard RoPE (self.rope from WanTransformer3DModel.__init__).
        # No width-halving needed since each branch has its own [H, W] spatial dims.

        # Re-create blocks to ensure they are exactly WanTransformerBlock from diffusers.
        block_qk_norm = qk_norm or "rms_norm_across_heads"
        self.blocks = nn.ModuleList(
            [
                WanTransformerBlock(
                    inner_dim, ffn_dim, num_attention_heads, block_qk_norm, cross_attn_norm, eps, added_kv_proj_dim
                )
                for _ in range(num_layers)
            ]
        )

        # ZCL: lightweight bidirectional control links at selected DiT layers.
        self.zcl_layers = tuple(sorted(set(int(i) for i in zcl_layers)))
        self.zcl_rgb_from_xyz = nn.ModuleDict()
        self.zcl_xyz_from_rgb = nn.ModuleDict()
        for layer_idx in self.zcl_layers:
            rgb_from_xyz = nn.Linear(inner_dim, inner_dim, bias=True)
            xyz_from_rgb = nn.Linear(inner_dim, inner_dim, bias=True)
            nn.init.zeros_(rgb_from_xyz.weight)
            nn.init.zeros_(rgb_from_xyz.bias)
            nn.init.zeros_(xyz_from_rgb.weight)
            nn.init.zeros_(xyz_from_rgb.bias)
            key = str(layer_idx)
            self.zcl_rgb_from_xyz[key] = rgb_from_xyz
            self.zcl_xyz_from_rgb[key] = xyz_from_rgb

    def _apply_zcl(
        self,
        hidden_states_rgb: torch.Tensor,
        hidden_states_xyz: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        key = str(layer_idx)
        if key not in self.zcl_rgb_from_xyz:
            return hidden_states_rgb, hidden_states_xyz

        z_rgb_src = hidden_states_rgb
        z_xyz_src = hidden_states_xyz

        hidden_states_rgb = z_rgb_src + self.zcl_rgb_from_xyz[key](z_xyz_src)
        hidden_states_xyz = z_xyz_src + self.zcl_xyz_from_rgb[key](z_rgb_src)

        return hidden_states_rgb, hidden_states_xyz

    def _run_decoupled_block(
        self,
        block: nn.Module,
        hidden_states_rgb: torch.Tensor,
        hidden_states_xyz: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep_proj: torch.Tensor,
        rotary_emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run a single DiT block on both branches with adapter switching."""
        if USE_PEFT_BACKEND and hasattr(self, "set_adapter"):
            self.set_adapter("rgb")
        hidden_states_rgb = block(hidden_states_rgb, encoder_hidden_states, timestep_proj, rotary_emb)

        if USE_PEFT_BACKEND and hasattr(self, "set_adapter"):
            self.set_adapter("xyz")
        hidden_states_xyz = block(hidden_states_xyz, encoder_hidden_states, timestep_proj, rotary_emb)

        if USE_PEFT_BACKEND and hasattr(self, "set_adapter"):
            self.set_adapter("rgb")

        return hidden_states_rgb, hidden_states_xyz

    def _unpatch(
        self, hidden_states: torch.Tensor,
        batch_size: int, post_patch_num_frames: int, post_patch_height: int, post_patch_width: int,
        p_t: int, p_h: int, p_w: int,
    ) -> torch.Tensor:
        """Reshape tokens back to video tensor [B, C_out, F, H, W]."""
        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        return hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        hidden_states_xyz: Optional[torch.Tensor] = None,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Transformer2DModelOutput]:
        """Forward pass with dual RGB/XYZ branches.

        Args:
            hidden_states: RGB branch input [B, in_ch, F, H, W]
                          (noisy_latent + condition + mask concatenated along channels).
            hidden_states_xyz: XYZ branch input [B, in_ch, F, H, W]
                              (noisy_latent + zero-padded condition/mask).
                              If None, falls back to single-branch (parent) behavior.
            timestep: Diffusion timestep.
            encoder_hidden_states: Text embeddings.
            encoder_hidden_states_image: CLIP image embeddings.

        Returns:
            If hidden_states_xyz is provided: (output_rgb, output_xyz) tuple,
            each [B, out_ch, F, H, W].
            If hidden_states_xyz is None: standard single-branch output.
        """
        if hidden_states_xyz is None:
            # Fallback: single-branch mode (standard WanTransformer3DModel behavior)
            return super().forward(
                hidden_states, timestep, encoder_hidden_states,
                encoder_hidden_states_image=encoder_hidden_states_image,
                return_dict=return_dict, attention_kwargs=attention_kwargs,
            )

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
        if isinstance(self.config, dict):
            patch_size = self.config["patch_size"]
        else:
            patch_size = self.config.patch_size
        p_t, p_h, p_w = patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        # Standard RoPE — shared by both branches (same spatial dims).
        rotary_emb = self.rope(hidden_states)

        # Patch-embed each branch independently through the SAME patch_embedding.
        tokens_rgb = self.patch_embedding(hidden_states)
        tokens_rgb = tokens_rgb.flatten(2).transpose(1, 2)  # [B, N, D]

        tokens_xyz = self.patch_embedding(hidden_states_xyz)
        tokens_xyz = tokens_xyz.flatten(2).transpose(1, 2)  # [B, N, D]

        # Condition embeddings (shared by both branches).
        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image
        )
        timestep_proj = timestep_proj.unflatten(1, (6, -1))

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        # Process through DiT blocks with adapter switching.
        if torch.is_grad_enabled() and self.gradient_checkpointing and self._gradient_checkpointing_func is not None:
            # Gradient checkpointing: run two separate forwards per block (RGB then XYZ).
            # This avoids packing/unpacking branches into a single tensor.
            for layer_idx, block in enumerate(self.blocks):
                # Keep adapter switch inside checkpointed call so backward recomputation
                # also uses the correct branch adapter.
                def _rgb_branch(h, ehs, tp, re):
                    if USE_PEFT_BACKEND and hasattr(self, "set_adapter"):
                        self.set_adapter("rgb")
                    return block(h, ehs, tp, re)

                tokens_rgb = self._gradient_checkpointing_func(
                    _rgb_branch,
                    tokens_rgb,
                    encoder_hidden_states,
                    timestep_proj,
                    rotary_emb,
                )

                def _xyz_branch(h, ehs, tp, re):
                    if USE_PEFT_BACKEND and hasattr(self, "set_adapter"):
                        self.set_adapter("xyz")
                    return block(h, ehs, tp, re)

                tokens_xyz = self._gradient_checkpointing_func(
                    _xyz_branch,
                    tokens_xyz,
                    encoder_hidden_states,
                    timestep_proj,
                    rotary_emb,
                )

                if USE_PEFT_BACKEND and hasattr(self, "set_adapter"):
                    self.set_adapter("rgb")

                tokens_rgb, tokens_xyz = self._apply_zcl(tokens_rgb, tokens_xyz, layer_idx)
        else:
            for layer_idx, block in enumerate(self.blocks):
                tokens_rgb, tokens_xyz = self._run_decoupled_block(
                    block, tokens_rgb, tokens_xyz,
                    encoder_hidden_states, timestep_proj, rotary_emb,
                )
                tokens_rgb, tokens_xyz = self._apply_zcl(tokens_rgb, tokens_xyz, layer_idx)

        # Final projection — apply independently to each branch.
        shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)
        shift = shift.to(tokens_rgb.device)
        scale = scale.to(tokens_rgb.device)

        tokens_rgb = (self.norm_out(tokens_rgb.float()) * (1 + scale) + shift).type_as(tokens_rgb)
        tokens_rgb = self.proj_out(tokens_rgb)
        output_rgb = self._unpatch(tokens_rgb, batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w)

        tokens_xyz = (self.norm_out(tokens_xyz.float()) * (1 + scale) + shift).type_as(tokens_xyz)
        tokens_xyz = self.proj_out(tokens_xyz)
        output_xyz = self._unpatch(tokens_xyz, batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w)

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output_rgb, output_xyz)

        return Transformer2DModelOutput(sample=output_rgb)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """Load model weights and materialize missing ZCL params if needed.

        With low_cpu_mem_usage/meta initialization, newly introduced parameters
        (zcl_*) that are absent from checkpoint can remain on meta tensors.
        DeepSpeed later fails when moving such tensors to real devices.
        """
        model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)

        # Pick a stable dtype from any non-meta parameter.
        param_dtype = None
        for p in model.parameters():
            if not getattr(p, "is_meta", False):
                param_dtype = p.dtype
                break
        if param_dtype is None:
            param_dtype = torch.float32

        def _materialize_linear_if_meta(linear: nn.Linear) -> None:
            if getattr(linear.weight, "is_meta", False):
                weight = torch.zeros(
                    (linear.out_features, linear.in_features),
                    dtype=param_dtype,
                    device="cpu",
                )
                linear.weight = nn.Parameter(weight, requires_grad=linear.weight.requires_grad)
            if linear.bias is not None and getattr(linear.bias, "is_meta", False):
                bias = torch.zeros((linear.out_features,), dtype=param_dtype, device="cpu")
                linear.bias = nn.Parameter(bias, requires_grad=linear.bias.requires_grad)

        for linear in model.zcl_rgb_from_xyz.values():
            _materialize_linear_if_meta(linear)
        for linear in model.zcl_xyz_from_rgb.values():
            _materialize_linear_if_meta(linear)

        return model


# Backward-compatibility alias
WanTransformer3DModelDembSameRope = WanTransformer3DModelDualBranch
