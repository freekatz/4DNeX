# 4D Dataset Structure 设计文档

## 1. 设计定位

统一的训练数据组织格式，DataLoader 直接消费。

### 设计优先级

1. **可用性**: 每个 clip 自包含，拿到就能训练
2. **训练效率**: 全局 index 支持快速采样；预编码数据直接 `torch.load`
3. **可共享性**: `videos/` 与 `latents/` 分离，共享数据集时只发 `videos/` + `index.json`，接收方自行编码
4. **存储效率**: latent 用 `.pt` 格式；text 编码按 caption hash 去重，相同 caption 只存一份

---

## 2. 目录结构

```
dataset_root/
│
├── index.json                                  # 全局索引 (DataLoader 入口)
│
├── videos/                                     # 可共享部分 (原始数据 + 元信息)
│   └── {source}/{video_id}/{clip_id}/
│       ├── video.mp4                           # RGB 视频 (81帧)
│       ├── xyz.mp4                             # XYZ point map 视频 (81帧)
│       ├── first_frame.png                     # 首帧 RGB
│       ├── caption.txt                         # Caption
│       └── meta.json                           # Clip 元信息 (含逐帧相机参数)
│
├── latents/                                    # 私有部分 — per-clip 编码
│   └── {source}/{video_id}/{clip_id}/
│       ├── rgb_latent.pt                       # RGB VAE latent
│       ├── xyz_latent.pt                       # XYZ VAE latent
│       └── visual_embeds.pt                    # CLIP image embedding (首帧)
│
└── latents_cache/                              # 私有部分 — text 编码去重缓存
    └── {caption_hash}.pt                       # {"text_embeds": tensor, "text_ids": tensor}
```

### 路径层级说明

| 层级   | 说明                                 | 示例                                              |
| ------ | ------------------------------------ | ------------------------------------------------- |
| source | 数据来源                             | `4dnex`, `omniworld_game`, `omniworld_hoi4d`      |
| video_id | 源视频的唯一 ID                    | `00000028`, `0365cd4c75bc`, `ZY2021_H1_C1_N19`    |
| clip_id  | 从 0 开始的 clip 序号              | `clip_0`, `clip_1`, `clip_2`                      |

### Text 编码去重

同一视频的多个 clip 通常共享相同 caption。text 编码按 caption 内容的 SHA-256 前 16 位 hash 存到 `latents_cache/`，相同 caption 只编码和存储一次。

每个 clip 在 `index.json` 中通过 `text_latent_path` 字段指向对应的缓存文件。

### 共享场景

```bash
# 共享给他人: 只发 videos/ + index.json
scp -r dataset_root/videos/ dataset_root/index.json remote:/data/

# 接收方自行编码 latent
python process_dataset.py --dataset_root /data/ --model_path ...
```

### 版本切换

换编码器后，新版本写入 `latents_v2/`，修改 `index.json` 中 `config.latents_dir` 即可:

```
dataset_root/
├── videos/          # 不变
├── latents/         # v1
├── latents_v2/      # v2
└── latents_cache/   # text 缓存可跨版本复用
```

---

## 3. 文件格式详述

### 3.1 `video.mp4` — RGB 视频

| 属性   | 规格               |
| ------ | ------------------ |
| 帧数   | 81 (固定)          |
| 分辨率 | 480×720 (全局统一) |
| 编码   | H.264, yuv420p     |
| FPS    | 24                 |

不足 81 帧时使用 reverse padding（ping-pong 循环拼接）。meta.json 中记录 `original_frames`。

### 3.2 `xyz.mp4` — XYZ Point Map 视频

| 属性     | 规格                             |
| -------- | -------------------------------- |
| 帧数     | 81 (与 RGB 对齐)                 |
| 分辨率   | 480×720 (与 RGB 相同)            |
| 通道含义 | R=X, G=Y, B=Z (归一化到 [0,255]) |
| 编码     | H.264, yuv420p                   |

归一化参数存储在 meta.json 的 `xyz_norm` 中。

### 3.3 `first_frame.png` — 首帧 RGB

480×720 PNG。用于 Image-to-Video 条件输入、CLIP 图像编码。

### 3.4 `caption.txt` — 文本描述

纯文本文件，一条 caption。

| 来源            | 取值                |
| --------------- | ------------------- |
| OmniWorld-Game  | Video_Caption       |
| OmniWorld-HOI4D | txt 文件原文        |
| 4DNeX-10M       | video-level caption |

### 3.5 `meta.json` — Clip 元信息

```json
{
  "video_id": "0365cd4c75bc",
  "clip_id": "clip_3",
  "clip_index": 3,

  "source_dataset": "omniworld_game",
  "source_entry_id": "omniworld_game_0365cd4c75bc_000001_000082",

  "num_frames": 81,
  "original_frames": 81,
  "fps": 24,
  "resolution": [480, 720],
  "frame_range_in_source": [121, 202],
  "is_padded": false,

  "camera": {
    "intrinsics": [
      [[1014.19, 0, 360.0], [0, 1014.19, 240.0], [0, 0, 1]],
      "... (81 frames)"
    ],
    "extrinsics_c2w": [
      [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
      "... (81 frames, aligned to first frame)"
    ]
  },

  "xyz_norm": {
    "center": [0.12, -0.03, 0.85],
    "scale": 2.34,
    "percentile": [2, 98]
  }
}
```

**相机参数**:

- `intrinsics`: `[T, 3, 3]` — 逐帧内参矩阵，已调整到输出分辨率
- `extrinsics_c2w`: `[T, 4, 4]` — 逐帧 camera-to-world SE(3)，已对齐到首帧坐标系（首帧为单位矩阵）

### 3.6 预编码数据

#### Per-clip latents（`latents/{source}/{video_id}/{clip_id}/`）

| 文件               | 内容                        | Shape              | 大小    |
| ------------------ | --------------------------- | ------------------ | ------- |
| `rgb_latent.pt`    | RGB VAE latent              | `[16, 21, 60, 90]` | ~7.3 MB |
| `xyz_latent.pt`    | XYZ VAE latent              | `[16, 21, 60, 90]` | ~7.3 MB |
| `visual_embeds.pt` | CLIP image embedding (首帧) | `[257, 1280]`      | ~1.3 MB |

#### Text 缓存（`latents_cache/{caption_hash}.pt`）

单个 `.pt` 文件，`torch.save` 保存的 dict：

```python
{
    "text_embeds": tensor,  # [512, 4096] float32, ~8.4 MB
    "text_ids": tensor,     # [512] long, ~4 KB
}
```

相同 caption 的所有 clip 共享同一个缓存文件。

---

## 4. 全局索引 `index.json`

```json
{
  "version": "2.0",
  "created_at": "2026-03-04T12:00:00",
  "dataset_name": "Dataset",

  "config": {
    "num_frames": 81,
    "resolution": [480, 720],
    "fps": 24,
    "pad_mode": "reverse",
    "latents_dir": "latents",
    "xyz_normalize": "percentile_2_98"
  },

  "statistics": {
    "num_videos": 3,
    "num_clips": 50,
    "sources": {
      "4dnex": { "videos": 1, "clips": 1 },
      "omniworld_game": { "videos": 1, "clips": 43 },
      "omniworld_hoi4d": { "videos": 1, "clips": 6 }
    }
  },

  "clips": [
    {
      "path": "omniworld_game/0365cd4c75bc/clip_0",
      "video_id": "0365cd4c75bc",
      "clip_index": 0,
      "source": "omniworld_game",
      "original_frames": 81,
      "is_padded": false,
      "split": "train",
      "text_latent_path": "latents_cache/a1b2c3d4e5f67890.pt"
    }
  ]
}
```

**路径拼接规则**:

```python
clip = index["clips"][i]
vid_dir  = f"{root}/videos/{clip['path']}"
lat_dir  = f"{root}/{index['config']['latents_dir']}/{clip['path']}"
text_pt  = f"{root}/{clip['text_latent_path']}"
```

**DataLoader 用法**:

```python
class Unified4DDataset(Dataset):
    def __init__(self, root, split="train"):
        index = json.load(open(f"{root}/index.json"))
        self.root = root
        self.latents_base = index["config"]["latents_dir"]
        self.clips = [c for c in index["clips"] if c["split"] == split]

    def __getitem__(self, idx):
        info = self.clips[idx]
        lat_dir = os.path.join(self.root, self.latents_base, info["path"])
        text_data = torch.load(
            os.path.join(self.root, info["text_latent_path"]),
            weights_only=True,
        )
        return {
            "rgb_latent": torch.load(f"{lat_dir}/rgb_latent.pt", weights_only=True),
            "xyz_latent": torch.load(f"{lat_dir}/xyz_latent.pt", weights_only=True),
            "visual_embeds": torch.load(f"{lat_dir}/visual_embeds.pt", weights_only=True),
            "text_embeds": text_data["text_embeds"],
            "text_ids": text_data["text_ids"],
        }
```

---

## 5. 预处理流程

```
                        build_dataset.py                    process_dataset.py
                        (per source)                        (per dataset, multi-GPU)

raw data ──▶ collect ──▶ for each clip:                     for each clip:
                         ├─ load RGB + depth + camera        ├─ RGB mp4 → VAE → rgb_latent.pt
                         ├─ depth + K + c2w → XYZ            ├─ XYZ mp4 → VAE → xyz_latent.pt
                         ├─ align to first camera frame      ├─ first_frame → CLIP → visual_embeds.pt
                         ├─ resize + center crop              └─ caption → UMT5 → latents_cache/{hash}.pt
                         ├─ XYZ normalize (percentile)
                         ├─ write video.mp4, xyz.mp4       ──▶ update index.json (add text_latent_path)
                         ├─ write first_frame.png
                         ├─ write caption.txt, meta.json
                         └─ write index.json
```

- `build_dataset.py`：每次处理一个 source（`--source 4dnex`），串行处理
- `process_dataset.py`：支持多 GPU（`torchrun --nproc_per_node=N`），clip 按 rank 平分
- `--tmp_dir`：FUSE 挂载时 ffmpeg 无法 seek，先写本地再 copy
- collect 结果缓存到 `.collect_cache_{source}_{hash}.pkl`，`--recollect` 强制刷新
- 两个脚本均支持增量处理（`skip_existing`），中断后可继续

---

## 6. 存储开销估算

以 480×720、81 帧/clip 为例:

### 单 clip

| 数据项            | 大小       | 位置           |
| ----------------- | ---------- | -------------- |
| video.mp4         | ~0.5-2 MB  | videos/        |
| xyz.mp4           | ~0.5-2 MB  | videos/        |
| first_frame.png   | ~0.5-1 MB  | videos/        |
| caption.txt       | ~1 KB      | videos/        |
| meta.json         | ~5 KB      | videos/        |
| rgb_latent.pt     | ~7.3 MB    | latents/       |
| xyz_latent.pt     | ~7.3 MB    | latents/       |
| visual_embeds.pt  | ~1.3 MB    | latents/       |
| text cache (共享) | ~8.4 MB    | latents_cache/ |
| **videos/ 小计**  | **~2 MB**  | 共享部分       |
| **latents/ 小计** | **~16 MB** | 私有部分       |

text 缓存按唯一 caption 计，同视频多个 clip 共享一份（~8.4 MB/caption）。

### 规模估算

| 规模   | clips  | videos/ | latents/ | latents_cache/ | 总计    |
| ------ | ------ | ------- | -------- | -------------- | ------- |
| 小     | 50     | ~100 MB | ~800 MB  | ~100 MB        | ~1 GB   |
| 中等   | 1,000  | ~2 GB   | ~16 GB   | ~1 GB          | ~19 GB  |
| 大规模 | 10,000 | ~20 GB  | ~160 GB  | ~5 GB          | ~185 GB |
