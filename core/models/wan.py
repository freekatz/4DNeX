"""Wan model definitions for 4DNeX.

WanRotaryPosEmb: Width-halving RoPE for dual-modality (RGB + Pointmap).
WanTransformer3DModelDembSameRope: Transformer with learnable domain embeddings and shared RoPE.
"""

from typing import Any, Dict, Optional, Tuple, Union
import os

import torch
import torch.nn as nn

from diffusers import WanTransformer3DModel
from diffusers.configuration_utils import register_to_config
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers
from diffusers.models.embeddings import get_1d_rotary_pos_embed
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.transformers.transformer_wan import WanTransformerBlock


logger = logging.get_logger(__name__)


# =============================================================================
# RoPE (width-halving with duplication for dual-modality)
# =============================================================================

class WanRotaryPosEmb(nn.Module):
    """4DNeX Modified: Width-halving RoPE for dual-modality (RGB + Pointmap).

    Difference from original diffusers WanRotaryPosEmb:
    - Original: Uses full width for RoPE computation
    - 4DNeX: Halves width first, computes RoPE, then duplicates frequencies

    Reason: RGB and Pointmap at same spatial position should get identical
    positional encoding, with modality distinction handled by domain embeddings.
    """
    def __init__(
        self, attention_head_dim: int, patch_size: Tuple[int], max_seq_len: int, theta: float = 10000.0
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
        # 4DNeX KEY DIFFERENCE 1: Halve width for dual-modality RoPE
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
        # 4DNeX KEY DIFFERENCE 2: Duplicate frequencies for both modalities
        freqs_f = torch.cat([freqs_f, freqs_f], dim=2)
        freqs_h = torch.cat([freqs_h, freqs_h], dim=2)
        freqs_w = torch.cat([freqs_w, freqs_w], dim=2)
        freqs = torch.cat([freqs_f, freqs_h, freqs_w], dim=-1).reshape(1, 1, ppf * pph * ppw * 2, -1)
        return freqs


# =============================================================================
# DembSameRope Transformer Model
# =============================================================================

class WanTransformer3DModelDembSameRope(WanTransformer3DModel, ModelMixin):
    """4DNeX Modified: Wan Transformer with learnable domain embeddings and shared RoPE.

    Key differences from original diffusers WanTransformer3DModel:
    1. NEW: learnable_domain_embeddings [2, inner_dim] - modality-specific embeddings
    2. REPLACED: self.rope = WanRotaryPosEmb (width-halving version)
    3. USES: Imported WanTransformerBlock from diffusers (exact same as original)
    4. MODIFIED: forward() injects domain embeddings after patch embedding

    Architecture: Single-stream dual-modality (RGB + pointmap concatenated along width).
    Domain distinction: learnable_domain_embeddings, not separate model branches.
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

        # Replace standard RoPE with width-halving version
        self.rope = WanRotaryPosEmb(attention_head_dim, patch_size, rope_max_seq_len)

        # Replace blocks with diffusers WanTransformerBlock
        self.blocks = nn.ModuleList(
            [
                WanTransformerBlock(
                    inner_dim, ffn_dim, num_attention_heads, qk_norm, cross_attn_norm, eps, added_kv_proj_dim
                )
                for _ in range(num_layers)
            ]
        )

        # Learnable domain embeddings: [2, inner_dim] to distinguish RGB vs Pointmap
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
        # Inject domain embeddings: emb[0] to RGB half, emb[1] to Pointmap half
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

        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.blocks:
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb
                )
        else:
            for block in self.blocks:
                hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)

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
        """Backward-compatible loading from original Wan weights.

        Try-except fallback mechanism:
          1. Try to load as 4DNeX model (with learnable_domain_embeddings)
          2. If fails, load original Wan weights and filter by shape compatibility
          3. New parameters (learnable_domain_embeddings) remain zero-initialized
        """
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
            # Filter: only load weights with matching shapes
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
