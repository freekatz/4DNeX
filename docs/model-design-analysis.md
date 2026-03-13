# 4DNeX 模型设计与训练问题分析

本文档总结当前 **WanTransformer3DModelDualBranch**（One4D 双分支）模型设计中的潜在缺陷，以及可能导致无法正常训练的原因。

---

## 一、模型设计缺陷

### 1. 输入通道与 condition 一致性依赖 checkpoint

- **现象**：`WanTransformer3DModel` 的 `in_channels` 来自预训练权重配置；训练时 `input_rgb = cat(noisy_rgb, condition_rgb)`，总通道数为 `16 + condition_rgb.shape[1]`。condition 由 mask（4 通道）+ 首帧 latent（16 通道）组成，即 **20 通道**，故实际输入为 **36 通道**。
- **风险**：若加载的 checkpoint 不是 I2V 版本（例如 `in_channels=16`），或与当前 condition 构建方式不一致，`patch_embedding` 会因通道数不匹配而报错或产生错误计算。
- **建议**：在 `compute_loss` 或 `prepare_trainable_parameters` 中增加断言：  
  `assert input_rgb.shape[1] == self.components.transformer.config.in_channels`，并在文档中明确要求使用 Wan2.1 **I2V** 预训练权重。

### 2. 双分支 forward 的 return_dict 行为不一致

- **位置**：`core/models/wan.py` 第 311–314 行。
- **现象**：当 `return_dict=True` 时，返回 `Transformer2DModelOutput(sample=output_rgb)`，**仅包含 RGB 分支**，XYZ 分支被丢弃。
- **影响**：任何通过 `return_dict=True` 并期望同时拿到两个分支的代码会只得到 RGB，容易埋下静默错误。
- **建议**：要么在 `Transformer2DModelOutput` 中增加 `sample_xyz` 字段（若 diffusers 允许），要么在文档和类型注解中明确：双分支调用必须使用 `return_dict=False` 并解包 `(pred_rgb, pred_xyz)`。当前 trainer 已使用 `return_dict=False`，需确保推理/验证路径一致。

### 3. ZCL 零初始化在 meta device 下的 materialize 时机

- **位置**：`wan.py` 的 `from_pretrained` 与 `trainer.py` 的 `prepare_for_training`。
- **现象**：使用 `low_cpu_mem_usage=True` 或 meta device 加载时，新增的 `zcl_rgb_from_xyz` / `zcl_xyz_from_rgb` 可能仍为 meta tensor；若未在 `from_pretrained` 中 materialize，DeepSpeed 等在后端移动参数到设备时会失败。
- **现状**：`from_pretrained` 已对 ZCL 的 Linear 做 `_materialize_linear_if_meta`，且 `prepare_for_training` 会检查 meta 并报错。若未关闭 `low_cpu_mem_usage` 或 checkpoint 中缺少 ZCL，仍可能触发。
- **建议**：在文档中写明：加载 One4D 双分支时若使用 DeepSpeed/大模型加载，建议先确认 ZCL 已 materialize，或使用 `low_cpu_mem_usage=False` 做一次本地加载测试。

### 4. 验证阶段未使用双分支 pipeline

- **位置**：`core/models/wan_trainer.py` 的 `initialize_pipeline` 与 `validation_step`。
- **现象**：`initialize_pipeline` 返回的是标准 `WanImageToVideoPipeline`，而非 `WanDualBranchPipeline`；且 `validation_step` 固定返回 `[]`，不生成任何验证样本。
- **影响**：即使开启 `do_validation`，也不会对双分支（RGB+XYZ）做可视化或指标评估，无法发现双分支推理时的 shape/数值问题。
- **建议**：若需要验证双分支行为，应使用 `WanDualBranchPipeline` 并实现 `validation_step` 返回至少 RGB（及可选 XYZ）的生成结果；若暂时不验证双分支，在配置或文档中说明“当前验证仅支持单分支或未启用”。

### 5. VAE 配置键名拼写

- **位置**：`core/models/wan_trainer.py` 第 148 行。
- **代码**：`vae_scale_factor_temporal = 2 ** sum(self.components.vae.config.temperal_downsample)`。
- **现象**：`temperal` 为拼写错误，正确应为 `temporal`。若 VAE 的 config 中键名为 `temporal_downsample`，此处会触发 `AttributeError` 导致训练启动即失败。
- **建议**：与 diffusers 中 `AutoencoderKLWan` 的 config 对齐：若官方为 `temperal_downsample` 则保留并加注释；否则改为 `temporal_downsample`，并做兼容（如 `getattr(config, 'temporal_downsample', getattr(config, 'temperal_downsample'))`）。

### 6. 梯度 checkpoint 与 adapter 切换的副作用

- **位置**：`wan.py` 中带 `_gradient_checkpointing_func` 的循环。
- **现象**：在 checkpoint 的 backward 重算时，会再次执行 `set_adapter("rgb")` / `set_adapter("xyz")`。当前实现是“每 block 先 RGB 再 XYZ，最后设回 rgb”，逻辑正确，但依赖 PEFT 的 `set_adapter` 为可重入的全局状态。
- **风险**：若 PEFT 或多进程下 adapter 状态不同步，可能出现某次重算用了错误分支的 LoRA。
- **建议**：在单元测试或小 batch 过一遍带 gradient checkpointing 的 forward+backward，确认 loss 与梯度与关闭 checkpoint 时一致（或误差在数值误差范围内）。

---

## 二、无法正常训练的可能原因

### 1. 预训练权重与输入 shape 不匹配（最常见）

- **表现**：`patch_embedding` 或第一层即报 shape 错误（如 `RuntimeError: shape mismatch`）。
- **原因**：使用的不是 Wan2.1 **I2V** 的 transformer，或 I2V 的 `in_channels` 与当前 condition 构建不一致（例如不同帧数/不同 mask 通道数）。
- **排查**：打印 `input_rgb.shape` 与 `transformer.config.in_channels`，确认 `input_rgb.shape[1] == in_channels`。

### 2. VAE config 键名错误

- **表现**：训练刚开始就在 `compute_loss` 中报错：`AttributeError: '...' object has no attribute 'temperal_downsample'`（或 `temporal_downsample` 缺失）。
- **原因**：`wan_trainer.py` 使用了 `temperal_downsample`，与本地/版本 diffusers 中 VAE config 键名不一致。
- **排查**：查看 `self.components.vae.config` 的实际键名，并统一为同一键名或做兼容。

### 3. 数据集中 latent 的 device/dtype 或缺失

- **表现**：`encoded_videos_rgb` / `encoded_videos_xyz` 在 collate 或第一次 forward 时报错（device、dtype、或维度错误）。
- **原因**：`process_dataset.py` 保存的是 `rgb_latent[0]`，形状为 `(z_dim, T, H, W)`；dataset 按 `(C, T, H, W)` 使用并与 `condition_rgb` 在时间维上一致。若某处把 T 和 C 搞反，或漏做归一化/device 放置，会出错。
- **排查**：在 `__getitem__` 和 `collate_fn` 后打印 `encoded_video_rgb.shape`、`encoded_video_xyz.shape`，以及 `train_resolution` 对应的 `num_frames`，确认与 condition 的帧数一致（如 13 latent 帧对应 HACK_FORCE_LATENT_FRAMES）。

### 4. LoRA/多 adapter 未正确挂载或激活

- **表现**：训练能跑但 loss 不下降、或梯度几乎为 0；或报错 `set_adapter` 不存在/未找到 adapter。
- **原因**：`add_adapter(..., "rgb")` / `add_adapter(..., "xyz")` 未在正确的 transformer 上调用；或 forward 中未在每 block 前后 `set_adapter`，导致始终用同一组 LoRA。
- **排查**：确认 `prepare_trainable_parameters` 中先 `add_adapter` 再 `requires_grad_(True)` 的只有 `zcl_` 和 `lora_`；并确认 forward 中每个 block 对 RGB 用 `set_adapter("rgb")`、对 XYZ 用 `set_adapter("xyz")`。

### 5. DeepSpeed 恢复时 transformer 被重新实例化

- **位置**：`trainer.py` 的 `load_model_hook`（DeepSpeed 分支）。
- **现象**：resume 时用 `from_pretrained` + `add_adapter` 新建了一个 transformer，但 optimizer/accelerator 里可能仍引用旧模型，导致恢复后训练的是未加载 checkpoint 的实例，或 state 不一致。
- **排查**：Resume 后打印一层参数范数或 LoRA 权重范数，确认与上次保存的 checkpoint 一致；并确认仅有一个 transformer 实例被 `accelerator.prepare`。

### 6. 混合精度与 ZCL/LoRA 的 dtype

- **现象**：训练中出现 NaN/Inf，或某几步后 loss 爆炸。
- **原因**：ZCL 或 LoRA 在 bf16/fp16 下数值不稳定；或 `cast_training_params` 只 cast 了部分参数，与 backward 时的 dtype 不一致。
- **排查**：先用 `mixed_precision="no"` 跑若干 step，若正常再开 bf16；并确认 ZCL 与 LoRA 在 optimizer 中为 fp32（若设计如此）。

### 7. 数据集 index 或 latent 文件缺失

- **表现**：`FileNotFoundError: ... rgb_latent.pt not found` 或 `index.json` 缺失。
- **原因**：未先运行 `build_dataset.py` 与 `process_dataset.py`，或 `data_root` / `index_file` 指向错误路径。
- **排查**：按 `docs/dataset-structure.md` 检查目录与必需文件是否齐全。

---

## 三、建议的修复与检查顺序

1. **先确认 VAE config 键名**：在 `wan_trainer.compute_loss` 入口打印 `getattr(self.components.vae.config, 'temperal_downsample', None)` 与 `getattr(..., 'temporal_downsample', None)`，并统一使用正确键名或兼容写法。
2. **再确认输入通道数**：在第一次 `compute_loss` 中打印 `input_rgb.shape[1]` 与 `self.components.transformer.config.in_channels`，不匹配则报错并提示使用 I2V 权重或调整 condition。
3. **关闭 gradient checkpointing 做一次短跑**：排除 checkpoint 重算与 adapter 切换的交互问题。
4. **实现最小验证**：用 `WanDualBranchPipeline` 在 1 个 step 后生成 1 条 RGB（及可选 XYZ）视频，确认推理 path 与训练 path 的 condition 一致。
5. **文档化**：在 README 或 `docs/` 中写明：必须使用 Wan2.1 I2V 预训练、推荐 `train_resolution` 与 latent 帧数、以及当前验证的适用范围（单分支/双分支/未启用）。

以上为当前代码库下可归纳的模型设计缺陷与训练失败原因；按顺序排查通常能定位大多数“无法正常训练”的问题。
