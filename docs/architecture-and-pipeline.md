# 4DNeX 模型架构与训练/推理流程

## 目录

- [4DNeX 模型架构与训练/推理流程](#4dnex-模型架构与训练推理流程)
  - [目录](#目录)
  - [1. 模型网络结构](#1-模型网络结构)
    - [1.1 总体架构](#11-总体架构)
    - [1.2 WanTransformer3DModelDembSameRope](#12-wantransformer3dmodeldembsamerope)
    - [1.3 WanTransformerBlock](#13-wantransformerblock)
    - [1.4 WanAttnProcessor2\_0](#14-wanattnprocessor2_0)
    - [1.5 WanRotaryPosEmb (Width-Halving RoPE)](#15-wanrotaryposemb-width-halving-rope)
    - [1.6 Learnable Domain Embeddings](#16-learnable-domain-embeddings)
    - [1.7 LoRA 适配器](#17-lora-适配器)
  - [2. 训练流程](#2-训练流程)
    - [2.1 入口与调用链](#21-入口与调用链)
    - [2.2 组件加载](#22-组件加载)
    - [2.3 可训练参数准备](#23-可训练参数准备)
    - [2.4 数据集加载](#24-数据集加载)
    - [2.5 训练循环](#25-训练循环)
    - [2.6 损失计算 (compute\_loss)](#26-损失计算-compute_loss)
    - [2.7 检查点保存与恢复](#27-检查点保存与恢复)
  - [3. 推理流程](#3-推理流程)
    - [3.1 入口与参数](#31-入口与参数)
    - [3.2 模型加载与 LoRA 融合](#32-模型加载与-lora-融合)
    - [3.3 Pipeline 前处理 (prepare\_latents)](#33-pipeline-前处理-prepare_latents)
    - [3.4 去噪与输出拆分](#34-去噪与输出拆分)
    - [3.5 VAE 解码与后处理](#35-vae-解码与后处理)
  - [附录: 关键文件索引](#附录-关键文件索引)

---

## 1. 模型网络结构

### 1.1 总体架构

4DNeX 采用**单流双模态 (single-stream dual-modality)** 架构，基于 Wan2.1 I2V 14B 视频生成模型进行 LoRA 微调。核心思想：

- 将 **RGB 视频** 和 **XYZ 点云图 (pointmap)** 沿宽度维度拼接，形成双倍宽度的输入
- 通过 **learnable domain embeddings** 区分两个模态
- 使用 **width-halving RoPE** 确保两个模态共享相同的位置编码
- **单个 PEFT LoRA 适配器** 微调所有注意力和前馈层

```
输入: [B, C, F, H, W*2]  (RGB + Pointmap 宽度拼接)
        │
        ▼
┌─────────────────────────┐
│  Patch Embedding (1×2×2)│
│  + Domain Embeddings    │
└──────────┬──────────────┘
           │  [B, seq_len, inner_dim]
           ▼
┌─────────────────────────┐
│  × 40 Transformer Blocks│
│  Self-Attn (RoPE)       │
│  Cross-Attn (text+img)  │
│  FFN (GELU-approx)      │
│  AdaLN modulation       │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  Output Norm + Proj     │
│  Unpatchify             │
└──────────┬──────────────┘
           │
           ▼
输出: [B, C, F, H, W*2]  (预测噪声)
        │
     split along W
       ┌─┴─┐
       ▼   ▼
     RGB   XYZ
```

### 1.2 WanTransformer3DModelDembSameRope

**文件**: `core/models/trainer.py:282-463`

继承 `WanTransformer3DModel` + `ModelMixin`，在基础模型上进行以下修改：

```python
class WanTransformer3DModelDembSameRope(WanTransformer3DModel, ModelMixin):
    # 默认配置 (14B 模型)
    patch_size = (1, 2, 2)           # 时间×高度×宽度 的 patch 分块
    num_attention_heads = 40
    attention_head_dim = 128         # inner_dim = 40 × 128 = 5120
    in_channels = 16                 # VAE latent 通道数
    out_channels = 16
    text_dim = 4096                  # UMT5 文本编码维度
    ffn_dim = 13824                  # FFN 中间层维度
    num_layers = 40                  # Transformer 层数
```

**关键初始化覆盖**:

```python
# 替换标准 RoPE 为 width-halving 版本
self.rope = WanRotaryPosEmb(attention_head_dim, patch_size, rope_max_seq_len)

# 替换标准 block 为自定义 WanTransformerBlock
self.blocks = nn.ModuleList([WanTransformerBlock(...) for _ in range(num_layers)])

# 新增：可学习域嵌入 [2, 5120]
self.learnable_domain_embeddings = nn.Parameter(torch.zeros(2, inner_dim))
```

**`forward()` 关键步骤** (`core/models/trainer.py:354-433`):

1. **计算 RoPE**: `rotary_emb = self.rope(hidden_states)` — width 减半 + 频率复制
2. **Patch 嵌入**: `hidden_states = self.patch_embedding(hidden_states)`
3. **添加域嵌入**:
   ```python
   first_half, second_half = self.learnable_domain_embeddings.chunk(2, dim=0)
   hidden_states = concat([
       hidden_states[:, :, :, :, :W//2] + first_half,   # RGB 半边
       hidden_states[:, :, :, :, W//2:] + second_half,  # Pointmap 半边
   ], dim=4)
   ```
4. **展平为序列**: `hidden_states.flatten(2).transpose(1, 2)` → `[B, seq_len, dim]`
5. **条件嵌入**: 时间步嵌入 + 文本编码 + 图像编码
6. **40 层 Transformer Block**: 自注意力(RoPE) + 交叉注意力(text+img) + FFN
7. **输出投影 + unpatchify**: 恢复 `[B, C, F, H, W*2]` 形状

**`from_pretrained()` 兼容加载** (`core/models/trainer.py:435-462`):

```python
@classmethod
def from_pretrained(cls, path, **kwargs):
    try:
        model = super().from_pretrained(path, **kwargs)  # 尝试直接加载
    except:
        base = WanTransformer3DModel.from_pretrained(path, **kwargs)  # 退化加载基础模型
        model = cls(**base.config)
        model.load_state_dict(filtered_base_dict)  # 按 shape 过滤
    return model
```

### 1.3 WanTransformerBlock

**文件**: `core/models/trainer.py:196-275`

每个 block 包含三个子层和 **AdaLN (Adaptive Layer Normalization)** 调制：

```
                ┌──── scale_shift_table ─────┐
                │  nn.Parameter([1, 6, dim]) │
                └──────────┬─────────────────┘
                           │ chunk(6)
            ┌──────────────┼──────────────────────┐
            ▼              ▼                      ▼
   shift_msa, scale_msa, gate_msa    c_shift, c_scale, c_gate
            │              │                      │
            ▼              ▼                      ▼
┌──────────────────┐ ┌──────────────┐  ┌──────────────────┐
│ 1. Self-Attention│ │ 2. Cross-Attn│  │ 3. Feed-Forward  │
│ norm1(x)*(1+s)+b │ │ norm2(x)     │  │ norm3(x)*(1+s)+b │
│ attn1 + RoPE     │ │ attn2 + I2V  │  │ FFN (GELU-approx)│
│ x + out * gate   │ │ x + out      │  │ x + out * gate   │
└──────────────────┘ └──────────────┘  └──────────────────┘
```

Self-Attention 和 FFN 使用 _gated residual_（乘以 gate 参数），Cross-Attention 使用直接残差相加。

### 1.4 WanAttnProcessor2_0

**文件**: `core/models/trainer.py:65-136`

自定义注意力处理器，支持：

- **Self-Attention + RoPE**: 将 `rotary_emb` 应用到 query 和 key
- **I2V Cross-Attention**: 当 `attn.add_k_proj is not None` 时，额外处理图像上下文（CLIP 视觉特征）

```python
# RoPE 应用 (complex number rotation)
x_rotated = torch.view_as_complex(hidden_states.to(float64).unflatten(3, (-1, 2)))
x_out = torch.view_as_real(x_rotated * freqs).flatten(3, 4)

# I2V 额外注意力
if encoder_hidden_states_img is not None:
    hidden_states_img = F.scaled_dot_product_attention(query, key_img, value_img)
    hidden_states = hidden_states + hidden_states_img  # 直接相加
```

文本上下文固定 512 token，图像上下文为 `encoder_hidden_states.shape[1] - 512` 个 token。

### 1.5 WanRotaryPosEmb (Width-Halving RoPE)

**文件**: `core/models/trainer.py:143-189`

核心创新：位置编码基于**半宽度**计算，然后复制给两个模态。

```python
def forward(self, hidden_states):
    B, C, F, H, W = hidden_states.shape
    width = W // 2  # 关键：宽度减半

    # 计算基于 (F, H, W//2) 的频率
    freqs_f = freqs[0][:ppf]  # 时间维度
    freqs_h = freqs[1][:pph]  # 高度维度
    freqs_w = freqs[2][:ppw]  # 宽度维度 (ppw = W//2 // patch_w)

    # 沿宽度维度复制一份 → 两个模态共享相同位置编码
    freqs_f = torch.cat([freqs_f, freqs_f], dim=2)
    freqs_h = torch.cat([freqs_h, freqs_h], dim=2)
    freqs_w = torch.cat([freqs_w, freqs_w], dim=2)
```

**频率维度分配**:

- `h_dim = w_dim = 2 * (attention_head_dim // 6)` ≈ 42
- `t_dim = attention_head_dim - h_dim - w_dim` ≈ 44
- 总计: 128 维 (= `attention_head_dim`)

### 1.6 Learnable Domain Embeddings

```python
self.learnable_domain_embeddings = nn.Parameter(torch.zeros(2, inner_dim))  # [2, 5120]
```

- `[0]`: RGB 模态嵌入 — 加到 patch embedding 的前半宽度
- `[1]`: Pointmap 模态嵌入 — 加到 patch embedding 的后半宽度
- 初始值为零向量
- 训练时 `requires_grad_(True)`
- 保存为独立文件 `learnable_domain_embeddings.pt`（原始 tensor）

### 1.7 LoRA 适配器

使用 PEFT 库的标准 LoRA：

```python
LoraConfig(
    r=64,              # LoRA rank
    lora_alpha=32,     # LoRA scaling factor
    target_modules=[   # 微调的目标模块
        "to_q", "to_k", "to_v", "to_out.0",
        "ffn.net.0.proj", "ffn.net.2"
    ],
)
```

- 单个 LoRA 适配器应用于整个 transformer（不区分 RGB/XYZ 分支）
- 推理时以 `lora_scale=0.5` 融合到基础权重

---

## 2. 训练流程

### 2.1 入口与调用链

**入口文件**: `finetune.py`

```python
# finetune.py
from core.models.trainer import WanTrainer
from core.schemas import Args

def main():
    args = Args.parse_args()
    trainer = WanTrainer(args)
    trainer.fit()
```

**调用链** (`Trainer.fit()` at `core/trainer.py:811-821`):

```
fit()
├── check_setting()             # 检查 UNLOAD_LIST 配置
├── prepare_models()            # 加载 transformer config
├── prepare_dataset()           # 构建 DatasetWithResize + DataLoader
├── prepare_trainable_parameters()  # 冻结 + 添加 LoRA + 注册 hooks
├── prepare_optimizer()         # AdamW + LR scheduler
├── prepare_for_training()      # accelerator.prepare()
├── prepare_for_validation()    # 加载验证数据 (可选)
├── prepare_trackers()          # TensorBoard / SwanLab
└── train()                     # 主训练循环
```

### 2.2 组件加载

**WanTrainer.load_components()** (`core/models/trainer.py:577-592`):

```python
components.pipeline_cls = WanImageToVideoPipeline
components.tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer")
components.text_encoder = UMT5EncoderModel.from_pretrained(model_path, subfolder="text_encoder")
components.transformer = WanTransformer3DModelDembSameRope.from_pretrained(
    model_path, subfolder="transformer")
components.vae = AutoencoderKLWan.from_pretrained(model_path, subfolder="vae")
components.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler")
components.image_encoder = CLIPVisionModel.from_pretrained(model_path, subfolder="image_encoder")
components.image_processor = CLIPImageProcessor.from_pretrained(model_path, subfolder="image_processor")
```

加载完成后，`UNLOAD_LIST = ["text_encoder", "image_encoder", "image_processor"]` 中的组件不会移到 GPU（训练时不需要这些组件，因为数据集已预计算编码）。

### 2.3 可训练参数准备

**WanTrainer.prepare_trainable_parameters()** (`core/models/trainer.py:599-648`):

```
1. 冻结所有参数                          requires_grad_(False)
2. 添加 PEFT LoRA 适配器                  transformer.add_adapter(lora_config)
3. 解冻 learnable_domain_embeddings       param.requires_grad_(True)
4. 移动非 transformer 组件到 GPU           component.to(device, dtype=weight_dtype)
5. 启用 gradient checkpointing (可选)     transformer.enable_gradient_checkpointing()
6. 注册 save/load hooks                   _register_hooks(lora_config)
```

可训练参数 ≈ LoRA 参数 (全部注意力层 + FFN) + `learnable_domain_embeddings` (2×5120 = 10240)

### 2.4 数据集加载

**文件**: `core/datasets/dataset.py`

**BaseDataset** 从 `index.json` 加载 clip 列表，每个 clip 的预计算文件：

| 文件                              | 说明                                    |
| --------------------------------- | --------------------------------------- |
| `latents/{path}/rgb_latent.pt`    | VAE 编码后的 RGB latent `[16, F, H, W]` |
| `latents/{path}/xyz_latent.pt`    | VAE 编码后的 XYZ latent `[16, F, H, W]` |
| `latents/{path}/visual_embeds.pt` | CLIP 视觉嵌入                           |
| `latents_cache/{hash}.pt`         | UMT5 文本嵌入 `{"text_embeds": ...}`    |

**`getitem()` 数据处理流程** (`core/datasets/dataset.py:141-185`):

```python
# 1. 加载预计算 latents
encoded_video = load("rgb_latent.pt")     # [16, F, H, W]
encoded_pm = load("xyz_latent.pt")        # [16, F, H, W]

# 2. XYZ 归一化 (mean=-0.13, std=1.70)
encoded_pm = (encoded_pm - ENCODED_PM_MEAN) / ENCODED_PM_STD

# 3. 沿宽度拼接 → [16, F, H, W*2]
encoded_video = torch.cat([encoded_video, encoded_pm], dim=-1)

# 4. 条件图像: 首帧 RGB + uniform pointmap → [3, H, W*2]
image = torch.cat([first_frame, uniform_pointmap], dim=-1)

# 返回: {image, prompt_embedding, encoded_video, image_embedding}
```

### 2.5 训练循环

**`Trainer.train()`** (`core/trainer.py:406-655`):

```
for epoch in range(train_epochs):
    for step, batch in enumerate(data_loader):
        with accelerator.accumulate(transformer):
            # 1. 计算损失
            loss = self.compute_loss(batch)

            # 2. 反向传播
            accelerator.backward(loss)

            # 3. 梯度同步时
            if accelerator.sync_gradients:
                grad_norm = clip_grad_norm_(max_grad_norm=1.0)
                branch_grad_norms = compute_one4d_branch_grad_norms()
                    # → { "optim/grad_norm_lora": ...,
                    #     "optim/grad_norm_domain_emb": ... }

            # 4. 优化器更新
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        # 5. 进度记录
        ema_loss = 0.95 * ema_loss + 0.05 * loss
        logs = { loss, ema, lr, grad_norm, ... }

        # 6. 检查点保存 (每 N 步)
        maybe_save_checkpoint(global_step)

        # 7. 验证 (可选, 每 M 步)
        if do_validation: validate(global_step)
```

**监控指标**:

| 指标                           | 描述                          |
| ------------------------------ | ----------------------------- |
| `train/loss`                   | 当前 step 的 MSE 损失         |
| `train/loss_ema`               | 指数移动平均损失 (β=0.95)     |
| `optim/grad_norm`              | 全局梯度范数                  |
| `optim/grad_norm_lora`         | LoRA 参数梯度范数             |
| `optim/grad_norm_domain_emb`   | 域嵌入梯度范数                |
| `optim/weight_norm_lora`       | LoRA 权重 L2 范数 (每 10 步)  |
| `optim/weight_norm_domain_emb` | 域嵌入权重 L2 范数 (每 10 步) |
| `perf/step_time_sec`           | 每步耗时                      |
| `system/gpu_mem_allocated_gb`  | GPU 显存占用                  |

### 2.6 损失计算 (compute_loss)

**WanTrainer.compute_loss()** (`core/models/trainer.py:780-856`):

```python
def compute_loss(self, batch) -> torch.Tensor:
    # 输入
    latent = batch["encoded_videos"]       # [B, 16, F, H, W*2] (RGB+XYZ 拼接)
    prompt_embedding = batch["prompt_embedding"]
    images = batch["images"]               # [B, 3, H, W*2] (首帧 RGB+pointmap)
    image_embedding = batch["image_embedding"]

    # --- 构建条件 ---
    # 首帧图像 → 扩展为视频长度 (后续帧全零)
    video_condition = cat([images, zeros(B, 3, F-1, H, W*2)], dim=2)
    latent_condition = vae.encode(video_condition)  # [B, 16, F', H', W'*2]

    # 构建 mask [B, 1, F, H', W'*2]:
    #   1.0  = 首帧 RGB 半边 (已知条件)
    #   0.5  = 首帧 pointmap 半边 (部分条件)
    #   0.0  = 其他帧 (待生成)

    condition = concat([mask, latent_condition], dim=1)  # [B, 17, F', H', W'*2]

    # --- Flow matching 噪声添加 ---
    timesteps = random_uniform(0, num_train_timesteps)
    sigmas = scheduler.sigmas[timesteps]
    noise = randn_like(latent)
    noisy_latents = (1 - sigma) * latent + sigma * noise
    target = noise - latent                     # 线性插值目标

    # --- 前向传播 ---
    input = concat([noisy_latents, condition], dim=1)  # [B, 33, F', H', W'*2]
    predicted = transformer(input, timestep, text_embed, img_embed)

    # --- MSE 损失 ---
    loss = mean((predicted - target) ** 2)
    return loss
```

**Flow Matching 公式**:

- 噪声采样: `noisy = (1-σ) · x + σ · ε`
- 目标: `target = ε - x`
- 损失: `L = ||f_θ(noisy, t) - target||²`

### 2.7 检查点保存与恢复

**保存 hook** (`core/models/trainer.py:658-676`):

每个检查点目录 (`checkpoints/step-NNNNNN/`) 包含：

```
step-NNNNNN/
├── pytorch_lora_weights.safetensors   # PEFT 标准格式 LoRA 权重
├── learnable_domain_embeddings.pt     # torch.save(tensor) 原始张量
├── optimizer.bin                      # 优化器状态
├── scheduler.bin                      # LR scheduler 状态
└── random_states_*.pkl                # RNG 状态
```

保存逻辑：

```python
# LoRA 权重 → 标准 PEFT pipeline API
transformer_lora_layers = get_peft_model_state_dict(unwrapped)
pipeline_cls.save_lora_weights(output_dir, transformer_lora_layers=transformer_lora_layers)

# 域嵌入 → 单独保存为原始 tensor
torch.save(unwrapped.learnable_domain_embeddings.data.cpu(), demb_path)
```

**加载 hook** (`core/models/trainer.py:678-708`):

```python
# LoRA 权重
lora_state_dict = pipeline_cls.lora_state_dict(input_dir)
set_peft_model_state_dict(unwrapped, transformer_state_dict, adapter_name="default")

# 域嵌入
demb = torch.load(demb_path, map_location="cpu")
unwrapped.learnable_domain_embeddings.data = demb.to(device, dtype)
```

---

## 3. 推理流程

### 3.1 入口与参数

**入口文件**: `inference.py`

```bash
python inference.py \
    --prompt "描述文字 POINTMAP_STYLE." \
    --image path/to/input.png \
    --model_path pretrained/Wan2.1-I2V-14B-480P-Diffusers \
    --lora_path pretrained/4dnex-lora \
    --out results/ \
    --rank 64 \
    --num_frames 81 \
    --seed 42
```

| 参数                    | 默认值                                     | 说明             |
| ----------------------- | ------------------------------------------ | ---------------- |
| `--model_path`          | `pretrained/Wan2.1-I2V-14B-480P-Diffusers` | 基础模型路径     |
| `--lora_path`           | (必须)                                     | LoRA 权重目录    |
| `--num_frames`          | 81                                         | 生成帧数         |
| `--num_inference_steps` | 50                                         | 去噪步数         |
| `--guidance_scale`      | 5.0                                        | CFG scale        |
| `--rank`                | 64                                         | LoRA rank        |
| `--offload_mode`        | `"model"`                                  | CPU offload 策略 |
| `--seed`                | 42                                         | 随机种子         |

**批量推理**: 支持 `--clip_dir` 自动读取目录下 `caption.txt` + `first_frame.png`。

### 3.2 模型加载与 LoRA 融合

**文件**: `core/inference/pipeline.py:22-148`

```python
def generate_video(prompt, model_path, lora_path, image_or_video_path, ...):
    # 1. 加载 CLIP 图像编码器
    image_encoder = CLIPVisionModel.from_pretrained(model_path, subfolder="image_encoder")

    # 2. 加载自定义 transformer (from_pretrained 自动兼容)
    transformer = WanTransformer3DModelDembSameRope.from_pretrained(
        model_path, subfolder="transformer", torch_dtype=bfloat16)

    # 3. 加载 learnable_domain_embeddings
    demb = torch.load(lora_path / "learnable_domain_embeddings.pt")
    transformer.learnable_domain_embeddings.data = demb.to(device, dtype)

    # 4. 构建 pipeline
    pipe = WanSameRopeWBWImageToVideoPipeline.from_pretrained(
        model_path, image_encoder=image_encoder, transformer=transformer)

    # 5. 加载 & 融合 LoRA (scale=0.5)
    pipe.load_lora_weights(lora_path, weight_name="pytorch_lora_weights.safetensors")
    pipe.fuse_lora(components=["transformer"], lora_scale=0.5)

    # 6. CPU offload 策略
    pipe.enable_model_cpu_offload()  # 或 sequential / none
```

### 3.3 Pipeline 前处理 (prepare_latents)

**WanSameRopeWBWImageToVideoPipeline.prepare_latents()** (`core/models/trainer.py:484-566`):

```python
def prepare_latents(self, image, ...):
    # 1. latent 空间尺寸: 宽度翻倍
    latent_width = width * 2 // vae_scale_factor_spatial

    # 2. 随机采样初始噪声 [B, 16, F', H', W'*2]
    latents = randn_tensor(shape)

    # 3. 构建首帧条件:
    #    - 左半: 输入图像
    #    - 右半: uniform pointmap (generate_uniform_pointmap → [-1,1])
    image = image.unsqueeze(2)
    pointmap = generate_uniform_pointmap(height, width) * 2 - 1
    image = concat([image, pointmap], dim=4)  # [B, 3, 1, H, W*2]

    # 4. 扩展为视频条件 (后续帧全零)
    video_condition = cat([image, zeros(B, 3, F-1, H, W*2)], dim=2)

    # 5. VAE 编码条件
    latent_condition = vae.encode(video_condition)
    latent_condition = (latent_condition - latents_mean) * latents_std

    # 6. 构建 mask [B, 1, F, H', W'*2]:
    #    首帧 RGB 半边 = 1.0
    #    首帧 pointmap 半边 = 0.5
    #    其他帧 = 0.0
    mask_lat_size[:, :, 0:1, :, W'//2:] = 0.5  # pointmap 半边标记

    return latents, concat([mask, latent_condition], dim=1)
```

### 3.4 去噪与输出拆分

```python
# Pipeline.__call__() 内部去噪循环
video_generate = pipe(
    prompt=prompt, image=image, ...,
    output_type="latent",     # 返回 latent 而非解码后的视频
).frames[0]                   # [16, F', H', W'*2]

# 拆分双倍宽度 latent
half_w = video_generate.shape[-1] // 2
latents_rgb = video_generate[..., :half_w]   # [16, F', H', W']
latents_xyz = video_generate[..., half_w:]   # [16, F', H', W']
```

### 3.5 VAE 解码与后处理

**文件**: `inference.py:20-44`

```python
# RGB 解码
latents_rgb = latents_rgb[None]  # [1, 16, F', H', W']
rgb_frames = tokenizer.decode(latents_rgb)  # → [F, H, W, 3] numpy, [0, 1]
imageio.mimwrite(rgb_path, (rgb_frames * 255).astype(uint8))

# XYZ 解码 (需要反归一化)
latents_xyz = latents_xyz * ENCODED_PM_STD + ENCODED_PM_MEAN  # 反转训练时的归一化
xyz_frames = tokenizer.decode(latents_xyz)  # → [F, H, W, 3] numpy

# 合并为 Pointmap pickle
pm = Pointmap(xyz=xyz_frames, rgb=rgb_frames.clip(0, 1))
pickle.dump(pm, open(pkl_path, 'wb'))
```

**输出文件**:

| 文件                    | 说明                             |
| ----------------------- | -------------------------------- |
| `{i:05d}_rgb.mp4`       | RGB 视频                         |
| `{i:05d}_xyz.mp4`       | XYZ 点云图视频                   |
| `{i:05d}.pkl`           | Pointmap pickle (包含 xyz + rgb) |
| `{i:05d}_optimized.pkl` | (可选) 经相机参数优化后的点云图  |

---

## 4. 与原版 Wan2.1 的全部差异

下表汇总 4DNeX 相对于 diffusers 库原版 `WanTransformer3DModel` / `WanImageToVideoPipeline` 的所有修改点。标注代码位置均在 `core/models/trainer.py` 中。

### 4.1 差异总览

| # | 修改点 | 原版 Wan2.1 | 4DNeX | 作用 |
|---|--------|------------|-------|------|
| 1 | RoPE 宽度计算 | `ppw = width // p_w` | `ppw = (width // 2) // p_w` | 让两个模态共享同一套位置编码 |
| 2 | RoPE 频率复制 | 无 | `torch.cat([freqs, freqs], dim=2)` | 复制后序列长度翻倍匹配双倍宽度 |
| 3 | learnable_domain_embeddings | 不存在 | `nn.Parameter(zeros(2, 5120))` | 让模型区分 RGB 与 Pointmap 模态 |
| 4 | forward() 中域嵌入注入 | patch 后直接 flatten | patch 后按宽度分两半各加域嵌入再 flatten | 将模态信息注入 token 级特征 |
| 5 | prepare_latents 宽度翻倍 | `latent_width = width // vae_sf` | `latent_width = width * 2 // vae_sf` | 为 RGB + Pointmap 双模态腾出空间 |
| 6 | prepare_latents pointmap 拼接 | 条件仅含输入图像 | `concat([image, uniform_pointmap], dim=W)` | 推理时用均匀 pointmap 作为初始条件 |
| 7 | Mask 0.5 标记 | mask 仅 0/1 | pointmap 半边首帧 mask = 0.5 | 区分"已知 RGB"与"部分已知 pointmap" |
| 8 | 输出拆分 | 直接输出 `[B,C,F,H,W]` | 沿 W 维度拆成 RGB + XYZ 两个 `[B,C,F,H,W/2]` | 分离两个模态的生成结果 |
| 9 | from_pretrained 兼容加载 | 标准 load | try 直接加载 → 失败则加载基础模型按 shape 过滤转换 | 兼容原版 Wan 权重 |
| 10 | LoRA 融合 scale | 不适用 | `lora_scale=0.5` | 参考项目的经验超参 |

### 4.2 逐项详解

#### 差异 1-2: Width-Halving RoPE

**位置**: `core/models/trainer.py:143-189` (class `WanRotaryPosEmb`)

```python
# ===== 原版 Wan2.1 (diffusers) =====
def forward(self, hidden_states):
    B, C, F, H, W = hidden_states.shape
    ppw = W // p_w                                    # ← 直接用完整宽度
    freqs_w = freqs[2][:ppw].view(1, 1, ppw, -1)
    # → 最终 reshape: ppf * pph * ppw

# ===== 4DNeX 修改 =====
def forward(self, hidden_states):
    B, C, F, H, W = hidden_states.shape
    width = W // 2                                    # ← [差异1] 宽度减半
    ppw = width // p_w
    freqs_w = freqs[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)
    freqs_f = torch.cat([freqs_f, freqs_f], dim=2)   # ← [差异2] 沿宽度维度复制
    freqs_h = torch.cat([freqs_h, freqs_h], dim=2)
    freqs_w = torch.cat([freqs_w, freqs_w], dim=2)
    # → 最终 reshape: ppf * pph * ppw * 2
```

**原理**: 输入宽度是 `W*2`（RGB + Pointmap 拼接），若直接对整个宽度计算 RoPE，左半的位置 0 和右半的位置 0 会获得不同编码，但它们实际上是同一个空间位置的两个模态。Width-halving 先对半宽计算位置频率，再复制一份，使得 `pos(RGB, x=0)` 与 `pos(Pointmap, x=0)` 获得完全相同的位置编码。模态的区分交给 domain embeddings 处理。

#### 差异 3-4: Learnable Domain Embeddings

**位置**: `core/models/trainer.py:352` (声明), `:386-391` (注入)

```python
# ===== 原版 Wan2.1 =====
# __init__: 无 learnable_domain_embeddings
# forward:
hidden_states = self.patch_embedding(hidden_states)
hidden_states = hidden_states.flatten(2).transpose(1, 2)   # ← 直接 flatten

# ===== 4DNeX 修改 =====
# __init__:
self.learnable_domain_embeddings = nn.Parameter(torch.zeros(2, inner_dim))  # ← [差异3]

# forward:
hidden_states = self.patch_embedding(hidden_states)
first_half, second_half = self.learnable_domain_embeddings.chunk(2, dim=0)
hidden_states = torch.cat([                                                 # ← [差异4]
    hidden_states[:, :, :, :, :W//2] + first_half[..., None, None, None],   #  RGB 半边 + emb[0]
    hidden_states[:, :, :, :, W//2:] + second_half[..., None, None, None],  #  XYZ 半边 + emb[1]
], dim=4)
hidden_states = hidden_states.flatten(2).transpose(1, 2)
```

**原理**: RoPE 被设计为对两个模态给出相同的位置编码，因此模型无法区分同一位置的 RGB token 和 Pointmap token。Domain embeddings 是两个可学习的 5120 维向量，分别加到 patch embedding 的左半（RGB）和右半（Pointmap）上，为模型提供模态判别信号。初始化为零，训练时与 LoRA 参数一起优化。

#### 差异 5-7: prepare_latents 双模态条件

**位置**: `core/models/trainer.py:484-566` (class `WanSameRopeImageToVideoPipeline`)

```python
# ===== 原版 Wan2.1 =====
latent_width = width // vae_scale_factor_spatial              # ← 标准宽度
shape = (B, 16, T_lat, H_lat, latent_width)

image = image.unsqueeze(2)
video_condition = cat([image, zeros(B, 3, F-1, H, W)], dim=2) # ← 仅输入图像
# ...
mask_lat_size[:, :, 0:1] = 1                                  # ← mask: 首帧=1, 其余=0
# (首帧 mask 直接 repeat_interleave)

# ===== 4DNeX 修改 =====
latent_width = width * 2 // vae_scale_factor_spatial           # ← [差异5] 宽度翻倍
shape = (B, 16, T_lat, H_lat, latent_width)

image = image.unsqueeze(2)
pointmap = generate_uniform_pointmap(H, W) * 2 - 1             # ← [差异6] 合成 uniform pm
image = concat([image, pointmap], dim=4)                        #    拼接为 [B, 3, 1, H, W*2]
video_condition = cat([image, zeros(B, 3, F-1, H, W*2)], dim=2)
# ...
first_frame_mask[:, :, :, :, W_lat//2:] = 0.5                 # ← [差异7] pointmap 半边=0.5
```

**原理**:
- **差异 5**: 噪声 latent 的宽度翻倍，给 RGB 和 Pointmap 各留一半空间
- **差异 6**: 推理时没有真实 pointmap 输入，用 `generate_uniform_pointmap()` 生成一个均匀分布的默认 pointmap（归一化到 [-1, 1]），作为 Pointmap 模态的初始条件
- **差异 7**: Mask 值用三级语义：`1.0` = 完全已知（首帧 RGB，直接来自输入图像），`0.5` = 部分已知（首帧 Pointmap，仅为均匀初值），`0.0` = 未知（其余帧的噪声区域）。模型可以据此调整不同区域的去噪策略

#### 差异 8: 输出拆分

**位置**: `core/inference/pipeline.py:133-137`

```python
# ===== 原版 Wan2.1 =====
video = pipe(...).frames[0]   # → [C, F, H, W] 直接就是最终视频 latent

# ===== 4DNeX 修改 =====
video = pipe(..., output_type="latent").frames[0]  # → [C, F, H, W*2]
half_w = video.shape[-1] // 2
latents_rgb = video[..., :half_w]                  # ← [差异8] 左半 = RGB
latents_xyz = video[..., half_w:]                  #           右半 = Pointmap
```

**原理**: 由于输入是双倍宽度的拼接 latent，输出自然也是双倍宽度。沿宽度维度切分后分别送入 VAE 解码，得到 RGB 视频和 XYZ pointmap 视频。

#### 差异 9: from_pretrained 兼容加载

**位置**: `core/models/trainer.py:435-462`

```python
# ===== 原版 Wan2.1 =====
model = WanTransformer3DModel.from_pretrained(path)  # 直接加载

# ===== 4DNeX 修改 =====
try:
    model = super().from_pretrained(path)             # 先尝试加载 4DNeX 格式
    if model.learnable_domain_embeddings.is_meta:
        model.learnable_domain_embeddings = zeros(...)  # meta → zeros
except:
    base = WanTransformer3DModel.from_pretrained(path)  # 回退: 加载原版 Wan 权重
    model = cls(**base.config)                          # 创建 4DNeX 实例
    filtered = {k: v for k, v in base.state_dict()      # 按 shape 匹配过滤
                if k in model_dict and shapes_match}
    model.load_state_dict(filtered, strict=False)       # 非严格加载
```

**原理**: 4DNeX 模型新增了 `learnable_domain_embeddings` 参数和自定义的 `WanTransformerBlock`（block 内部的子模块名称与原版一致，但 `WanRotaryPosEmb` 结构不同）。当加载原版 Wan 预训练权重时，新增参数不存在于权重文件中。fallback 机制先加载原版权重到基础模型，再按 shape 匹配迁移到 4DNeX 模型，新增参数（如 `learnable_domain_embeddings`）保持零初始化。

#### 差异 10: LoRA 融合 scale 0.5

**位置**: `core/inference/pipeline.py:88`

```python
# ===== 原版 Wan2.1 =====
# 不涉及 LoRA

# ===== 4DNeX =====
pipe.fuse_lora(components=["transformer"], lora_scale=0.5)  # ← scale 0.5 而非 1.0
```

**原理**: LoRA 的效果强度通过 `lora_scale` 控制：`W_fused = W_base + scale * (B × A)`。使用 0.5 而非默认的 1.0 可以减弱 LoRA 的修改幅度，让输出更接近基础模型的分布，避免过度偏移。这是参考项目通过实验确定的超参数。

---

## 附录: 关键文件索引

| 文件                         | 作用                                     |
| ---------------------------- | ---------------------------------------- |
| `finetune.py`                | 训练入口                                 |
| `inference.py`               | 推理入口                                 |
| `core/models/trainer.py`     | 模型架构 + 训练器 (WanTrainer)           |
| `core/inference/pipeline.py` | 推理 pipeline (模型加载 + LoRA 融合)     |
| `core/trainer.py`            | 基类 Trainer (训练循环 + 验证 + 日志)    |
| `core/datasets/dataset.py`   | 数据集 (预计算 latent 加载 + 双模态拼接) |
| `core/datasets/utils.py`     | 工具函数 (uniform pointmap 等)           |
| `core/schemas/args.py`       | 训练参数定义                             |
| `core/schemas/components.py` | Pipeline 组件容器                        |
| `core/schemas/state.py`      | 训练状态容器                             |
