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
    _skip_layerwise_casting_patterns = ["patch_embedding", "patch_embedding_xyz", "condition_embedder", "norm"]
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

        # XYZ branch: independent patch embedding (16 ch only, no condition padding).
        # Initialized from pretrained in from_pretrained(); allows XYZ to learn its own input mapping.
        self.patch_embedding_xyz = nn.Conv3d(
            16,
            inner_dim,
            kernel_size=patch_size,
            stride=patch_size,
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

    def _get_peft_model(self) -> "WanTransformer3DModelDualBranch":
        """Return the innermost model that holds PEFT adapters (unwrap DDP/DeepSpeed).
        set_adapter must be called on this so the correct LoRA branch is active per block.
        """
        m: nn.Module = self
        while hasattr(m, "module"):
            m = m.module
        return cast(WanTransformer3DModelDualBranch, m)

    def _set_adapter_safe(self, adapter_name: str) -> None:
        """Switch active LoRA adapter on the actual PEFT model (unwrapped)."""
        if not USE_PEFT_BACKEND:
            return
        peft_model = self._get_peft_model()
        if hasattr(peft_model, "set_adapter"):
            peft_model.set_adapter(adapter_name)

    def _make_ckpt_branch_fn(
        self, block: nn.Module, adapter_name: str,
    ):
        """Build a closure that correctly captures *this* block for checkpoint recomputation.

        Defining the closure inside a separate method creates a new scope per call,
        so ``block`` is bound to the function parameter (captured by value at call
        time) rather than the loop variable in ``forward()`` (which would be
        late-bound to the last block).
        """
        def _branch(h, ehs, tp, re):
            self._set_adapter_safe(adapter_name)
            return block(h, ehs, tp, re)
        return _branch

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
        self._set_adapter_safe("rgb")
        hidden_states_rgb = block(hidden_states_rgb, encoder_hidden_states, timestep_proj, rotary_emb)

        self._set_adapter_safe("xyz")
        hidden_states_xyz = block(hidden_states_xyz, encoder_hidden_states, timestep_proj, rotary_emb)

        self._set_adapter_safe("rgb")

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

        # RGB: shared patch_embedding (36 ch = noisy + condition + mask).
        tokens_rgb = self.patch_embedding(hidden_states)
        tokens_rgb = tokens_rgb.flatten(2).transpose(1, 2)  # [B, N, D]

        # XYZ: independent patch_embedding (16 ch = noisy latent only; no zero-padded condition).
        tokens_xyz = self.patch_embedding_xyz(hidden_states_xyz[:, :16, :, :, :])
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
                # _make_ckpt_branch_fn creates a new scope per call, binding
                # *this* block to the closure so backward recomputation uses the
                # correct layer (not the last block from the loop).
                tokens_rgb = self._gradient_checkpointing_func(
                    self._make_ckpt_branch_fn(block, "rgb"),
                    tokens_rgb,
                    encoder_hidden_states,
                    timestep_proj,
                    rotary_emb,
                )

                tokens_xyz = self._gradient_checkpointing_func(
                    self._make_ckpt_branch_fn(block, "xyz"),
                    tokens_xyz,
                    encoder_hidden_states,
                    timestep_proj,
                    rotary_emb,
                )

                self._set_adapter_safe("rgb")

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
        """Load pretrained weights; materialize & initialize new parameters.

        Parameters absent from checkpoint (ZCL links, patch_embedding_xyz)
        may remain on meta device when loaded with low_cpu_mem_usage=True.
        This method materializes them on CPU and applies proper initialization.
        """
        model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        cls._materialize_new_params(model)
        return model

    @staticmethod
    def _materialize_new_params(model: "WanTransformer3DModelDualBranch") -> None:
        """Materialize & initialize all parameters not present in checkpoint."""
        param_dtype = next(
            (p.dtype for p in model.parameters() if not getattr(p, "is_meta", False)),
            torch.float32,
        )

        def _materialize_param(param: nn.Parameter, shape: tuple) -> nn.Parameter:
            """Replace a meta Parameter with a zero-filled CPU tensor."""
            if not getattr(param, "is_meta", False):
                return param
            return nn.Parameter(
                torch.zeros(shape, dtype=param_dtype, device="cpu"),
                requires_grad=param.requires_grad,
            )

        # --- ZCL links (zero-initialized) ---
        for module_dict in (model.zcl_rgb_from_xyz, model.zcl_xyz_from_rgb):
            for linear in module_dict.values():
                linear.weight = _materialize_param(linear.weight, linear.weight.shape)
                if linear.bias is not None:
                    linear.bias = _materialize_param(linear.bias, linear.bias.shape)

        # --- XYZ patch embedding (initialized from RGB patch_embedding[:, :16]) ---
        pe_rgb = model.patch_embedding
        pe_xyz = model.patch_embedding_xyz

        # Materialize: derive shape from the loaded RGB patch_embedding.
        rgb_w = pe_rgb.weight  # [out_ch, in_ch=36, kt, kh, kw]
        xyz_shape = (rgb_w.shape[0], 16, *rgb_w.shape[2:])
        pe_xyz.weight = _materialize_param(pe_xyz.weight, xyz_shape)
        if pe_xyz.bias is not None:
            pe_xyz.bias = _materialize_param(pe_xyz.bias, (rgb_w.shape[0],))

        # Initialize from RGB's first 16 input channels.
        if not getattr(rgb_w, "is_meta", False):
            with torch.no_grad():
                pe_xyz.weight.data.copy_(rgb_w[:, :16].to(param_dtype))
                if pe_rgb.bias is not None and not getattr(pe_rgb.bias, "is_meta", False):
                    pe_xyz.bias.data.copy_(pe_rgb.bias.to(param_dtype))


# Backward-compatibility alias
WanTransformer3DModelDembSameRope = WanTransformer3DModelDualBranch
