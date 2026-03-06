# Dataset Builder — 设计文档

> 日期：2026-03-04 · 分支：baseline-v2

本文档记录 `build_dataset.py` 及 `core/datasets/adapters/` 的完整设计决策、实现细节、已知问题与结论。

---

## 目录

1. [目标与范围](#1-目标与范围)
2. [整体架构](#2-整体架构)
3. [目录结构](#3-目录结构)
4. [三个数据源详解](#4-三个数据源详解)
5. [Adapter 设计](#5-adapter-设计)
6. [Builder 处理流程](#6-builder-处理流程)
7. [XYZ 点图生成与归一化](#7-xyz-点图生成与归一化)
8. [深度物理尺度分析](#8-深度物理尺度分析)
9. [帧数处理鲁棒性](#9-帧数处理鲁棒性)
10. [Meta 信息字段说明](#10-meta-信息字段说明)
11. [与旧管线及论文的对比](#11-与旧管线及论文的对比)
12. [已修复的正确性问题](#12-已修复的正确性问题)
13. [运行方式](#13-运行方式)
14. [后续工作](#14-后续工作)

---

## 1. 目标与范围

`build_dataset.py` 将三个异构原始数据源统一构建为符合 `docs/dataset-structure.md` 的训练数据集。

**职责范围**：只生成 `videos/` 目录树 + `index.json`。Latent 编码（`latents/`）由独立的下游脚本处理（尚未实现）。

**输入**：`raw_data/` 下的三个原始数据集  
**输出**：`data/videos/{source}/{video_id}/{clip_id}/` + `index.json`

---

## 2. 整体架构

采用 **Adapter → VideoClip → Builder** 三层架构：

```
raw_data/4dnex/        ─── 4dnex adapter ──┐
raw_data/omniworld/    ─── game adapter ─────┤──▸ List[VideoClip] ──▸ Builder ──▸ videos/ + index.json
raw_data/hoi4d/        ─── hoi4d adapter ────┘
```

- **Adapter 层** (`core/datasets/adapters/`)：每个数据源一个模块，负责扫描原始数据、组装成 `VideoClip`
- **VideoClip**：统一的中间表示（dataclass），包含 RGB、深度、内参、外参、caption 等
- **Builder 层** (`build_dataset.py`)：接收 `VideoClip`，执行 resize/crop、XYZ 反投影、归一化、写入磁盘

**解耦原则**：adapter 不做任何几何变换（不 resize、不归一化），只负责「读取→组装」；所有几何处理集中在 builder。

---

## 3. 目录结构

### 3.1 输出目录

```
data/
├── index.json                                    # 全局索引
└── videos/
    ├── 4dnex/                                  # source_dataset 层级
    │   └── 00000028/                             # video_id（去掉源前缀）
    │       └── clip_0/
    │           ├── video.mp4                      # 81帧 480×720 H.264
    │           ├── xyz.mp4                        # 归一化 XYZ 点图视频
    │           ├── first_frame.png                # 首帧图像
    │           ├── caption.txt                    # 文本描述
    │           └── meta.json                      # 元信息
    ├── omniworld_game/
    │   └── 0365cd4c75bc/
    │       ├── clip_0/ ... clip_42/
    └── omniworld_hoi4d/
        └── ZY20210800001_H1_C1_N19_S100_s02_T1/
            ├── clip_0/ ... clip_6/
```

### 3.2 目录层级选择

选择 `videos/source/video_id/clip` 而非 `source/videos/video_id/clip`，理由：

- 一个 `videos/` 入口对 DataLoader 最友好，`glob("videos/*/*/clip_*/video.mp4")` 即可拿到所有数据
- 与未来的 `latents/` 平行排列，清晰
- 语义上这是一个「统一数据集」而非「松散联邦」

### 3.3 代码文件布局

```
build_dataset.py                    # Builder 主脚本
core/datasets/adapters/
    __init__.py                     # 统一导出
    base.py                         # VideoClip dataclass + 共享工具
    fdnex.py                      # 4DNeX-10M 适配器
    omniworld_game.py               # OmniWorld-Game 适配器
    omniworld_hoi4d.py              # OmniWorld-HOI4D 适配器
```

---

## 4. 三个数据源详解

### 4.1 4DNeX-10M (raw_data/4dnex/)

```
4dnex/
├── caption/dynamic.csv                     # CSV: number, caption
├── dynamic/{video_id}/clip_{start}-{end}/  # 4DNeX-10M 输出
│   ├── pred_traj.txt                       # TUM 格式: idx tx ty tz qw qx qy qz
│   ├── pred_intrinsics.txt                 # 9 floats per row → 3×3
│   └── frame_XXXX.npy                      # float16 深度图 (288×512)
└── raw/dynamic/{video_id}.mp4              # 源视频
```

| 属性         | 值                                              |
| ------------ | ----------------------------------------------- |
| 深度格式     | `.npy` float16                                  |
| 深度值域     | ~0.1 – ~1.2                                     |
| 深度类型     | **相对深度**（scale-ambiguous，4DNeX-10M 输出）   |
| RGB 分辨率   | 视频原始分辨率                                  |
| 深度分辨率   | 288×512                                         |
| 内参         | `pred_intrinsics.txt`，对应 288×512 深度分辨率  |
| 外参         | `pred_traj.txt`，TUM 格式，**qw qx qy qz 顺序** |
| 相机平移量级 | ~0.001                                          |
| Caption      | `dynamic.csv`，按 8 位零填充 number 匹配        |
| 帧数         | 变长（如 54 帧），可能不足 81 帧                |

**注意**：scipy `Rotation.from_quat()` 要求 `[qx, qy, qz, qw]` 顺序，而 TUM 文件是 `qw, qx, qy, qz`，adapter 中已做转换。

### 4.2 OmniWorld-Game (raw_data/omniworld/ + videos/)

```
omniworld/
├── annotations/OmniWorld-Game/{scene_id}/
│   ├── {scene_id}_others/
│   │   ├── droidclib/split_N.json          # 内参 + 4×4 外参
│   │   └── text/SSSSSS_EEEEEE.json         # Caption JSON
│   └── {scene_id}_depth_0000/depth/        # 深度图 (可能稀疏)
└── videos/OmniWorld-Game/{scene_id}/{scene_id}_rgb_0000/color/  # PNG 逐帧
```

| 属性         | 值                                                             |
| ------------ | -------------------------------------------------------------- |
| 深度格式     | `.png` uint16                                                  |
| 深度值域     | ~3000 – ~62000                                                 |
| 深度类型     | **uint16 未知单位**（游戏引擎渲染，scale factor 不明）         |
| RGB 分辨率   | 720×1280                                                       |
| 深度分辨率   | 720×1280                                                       |
| 内参         | `droidclib/split_N.json` 中 `crop_intrinsic: {fx, fy, cx, cy}` |
| 外参         | 同 JSON 中 `extrinsics: [N, 4, 4]`，直接是 c2w 矩阵            |
| 相机平移量级 | ~0.04 – 0.13                                                   |
| Caption      | `text/SSSSSS_EEEEEE.json` → `captions.Video_Caption`           |
| 帧数         | 由 caption JSON 文件名定义的范围，变长                         |
| 深度稀疏性   | 部分帧可能缺少深度，adapter 支持 `nearest` 或 `skip` 策略      |

**文件名格式**：RGB 和深度均为 **6 位** 零填充（`000001.png`）。

### 4.3 OmniWorld-HOI4D (raw_data/omniworld/ + raw_data/hoi4d/)

```
omniworld/annotations/OmniWorld-HOI4D/{scene_dir}/
├── camera/
│   ├── split_info.json              # scene_name → 导出 RGB 视频路径
│   └── recon/split_0/info.json      # 内参 + 4×4 外参
├── prior_depth/XXXXX.png            # 深度图
└── text/S_E.txt                     # 纯文本 caption

hoi4d/{scene_name}/align_rgb/image.mp4    # RGB 视频
```

| 属性         | 值                                                              |
| ------------ | --------------------------------------------------------------- |
| 深度格式     | `.png` uint16                                                   |
| 深度值域     | ~500 – ~1400                                                    |
| 深度类型     | **uint16 未知单位**（prior depth，scale factor 不明）           |
| RGB 分辨率   | 1080×1920（视频）                                               |
| 深度分辨率   | 1080×1920                                                       |
| 内参         | `recon/split_0/info.json` 中 `crop_intrinsic: {fx, fy, cx, cy}` |
| 外参         | 同 JSON 中 `extrinsics: [N, 4, 4]`，直接是 c2w 矩阵             |
| 相机平移量级 | ~0.02 – 8.3                                                     |
| Caption      | `text/S_E.txt`，纯文本                                          |
| 帧数         | 由 caption txt 文件名定义的范围，变长                           |

**文件名格式**：深度为 **5 位** 零填充（`00000.png`）——与 Game 的 6 位不同。

---

## 5. Adapter 设计

### 5.1 VideoClip 数据结构

```python
@dataclass
class VideoClip:
    source_dataset: str        # "4dnex" / "omniworld_game" / "omniworld_hoi4d"
    source_entry_id: str       # 可溯源的唯一 ID
    video_id: str              # 输出目录中的 video_id（无源前缀）
    frame_ids: List[int]       # 源中的原始帧索引
    rgb_frames: np.ndarray     # [T, H_rgb, W_rgb, 3]  uint8
    depth_frames: np.ndarray   # [T, H_depth, W_depth]  float32
    intrinsics: np.ndarray     # [T, 3, 3]  float32，校准到深度分辨率
    extrinsics_c2w: np.ndarray # [T, 4, 4]  float32，camera-to-world
    caption: str               # 文本描述
    fps: int                   # 帧率
```

**设计要点**：

- 内参校准到**深度分辨率**（不是 RGB 分辨率），因为反投影用的是深度图
- RGB 和深度分辨率可以不同（如 4DNeX-10M 深度 288×512 vs 视频原始分辨率）
- `video_id` 不含源前缀——因为目录结构已有 `source_dataset` 层级

### 5.2 共享工具函数（base.py）

| 函数                                       | 用途                     |
| ------------------------------------------ | ------------------------ |
| `read_video_frames(path, frame_ids)`       | 按帧号随机读取视频帧     |
| `quat_wxyz_to_c2w(tx,ty,tz,qw,qx,qy,qz)`   | 四元数转 4×4 c2w 矩阵    |
| `make_windows(length, num_frames, stride)` | 长序列切分为固定长度窗口 |

### 5.3 各 Adapter 的 collect() 接口

```python
# 4dnex
collect(raw_root, fps=24, num_frames=81, stride=40) → List[VideoClipDescriptor]

# omniworld_game
collect(raw_root, fps=24) → List[VideoClipDescriptor]

# omniworld_hoi4d
collect(raw_root, fps=24) → List[VideoClipDescriptor]
```

> `VideoClipDescriptor` 只存储元数据和加载参数，帧数据在 `write_clip_files` 中按需加载，
> 避免 collect 阶段一次性将所有帧载入内存导致 OOM。

**4DNeX-10M 特殊处理**：

- 长 clip 通过 `make_windows()` 滑窗切分为多个子 clip
- `source_entry_id` 使用实际源帧索引（如 `4dnex_00000028_000000_000053`）

**OmniWorld-Game 特殊处理**：

- 深度可能是稀疏的，`_choose_depth_path()` 自动查找最近可用帧作为回退
- 在 glob 前检查 `depth_dir.exists()` 避免对不存在的目录调用 glob
- clip 范围由 caption JSON 文件名定义，任一帧缺数据即跳过整个 clip

**OmniWorld-HOI4D 特殊处理**：

- RGB 来自 `hoi4d/` 下的视频文件，路径通过 `split_info.json` 的 `scene_name` 导出
- 深度文件名 5 位零填充（与 Game 的 6 位不同）
- clip 范围由 caption txt 文件名定义

---

## 6. Builder 处理流程

`process_clip()` 的完整流水线：

```
VideoClip
  │
  ├─ 1. 截断到最短公共长度（RGB/depth/intr/ext 取 min）
  │
  ├─ 2. reverse_pad 到 num_frames (81)
  │     不足时：将序列反转后拼接到末尾，如 [1,2,3] pad 到 5 → [1,2,3,3,2]
  │     内参和外参同步 pad（保证每帧都有对应参数）
  │
  ├─ 3. build_xyz_sequence: depth + K + c2w → 世界坐标 XYZ
  │     在深度原生分辨率下做反投影（非 resize 后的分辨率）
  │
  ├─ 4. align_first_camera: 所有 XYZ 和外参变换到第 0 帧相机坐标系
  │     w2f = inv(c2w[0])
  │     xyz = w2f @ xyz
  │     ext = w2f @ ext
  │
  ├─ 5. adjust_intrinsics: 原始内参 → 适配 resize+crop 后的 480×720 分辨率
  │     fx *= sx, fy *= sy, cx = cx*sx - x0, cy = cy*sy - y0
  │
  ├─ 6. resize_and_center_crop:
  │     RGB → INTER_LINEAR
  │     XYZ → INTER_NEAREST（避免插值引入无意义的 3D 坐标）
  │
  ├─ 7. compute_xyz_norm: percentile 2/98 归一化 → [-1, 1]
  │     center = (p2 + p98) / 2  (per-channel)
  │     scale = max((p98 - p2) / 2)  (三轴取最大，保持等比例)
  │     clip to [-1, 1]
  │
  └─ 8. 映射到 [0, 255] uint8 → 存为 xyz.mp4
```

---

## 7. XYZ 点图生成与归一化

### 7.1 反投影公式

```
cam_x = (u - cx) * z / fx
cam_y = (v - cy) * z / fy
cam_z = z
world = cam @ R^T + t          # R, t 来自 c2w 矩阵
```

### 7.2 三层归一化链

本项目中 XYZ 从原始深度到最终训练输入经过**三层归一化**：

| 层级            | 输入          | 操作                                     | 输出                              | 位置                              |
| --------------- | ------------- | ---------------------------------------- | --------------------------------- | --------------------------------- |
| **Pixel-space** | 世界坐标 XYZ  | percentile 2/98 + 线性缩放               | [-1, 1] float → [0,255] uint8 MP4 | `build_dataset.py`                |
| **VAE latent**  | [-1, 1] float | Wan VAE encode + `(latent - mean) * std` | latent space                      | `core/tokenizer/wan.py`           |
| **训练 latent** | VAE latent    | `(latent - (-0.13)) / 1.70`              | 分布对齐                          | `wan_dataset.py` / `inference.py` |

第一层的 `center` 和 `scale` 存储在 `meta.json` 的 `xyz_norm` 字段中，可用于反归一化。

### 7.3 Per-clip 归一化 vs Per-frame 归一化

- **新管线**（`build_dataset.py`）：**per-clip**，对整个 clip 的所有帧 flatten 后计算 percentile
- **旧管线**（`build_wan_dataset.py`）：**per-frame**，按 axis=1 独立计算

**Per-clip 更合理**：保证同一 clip 内时序一致，不会因单帧 outlier 导致帧间归一化跳变。

### 7.4 首帧对齐

所有 XYZ 点和外参都变换到第 0 帧相机坐标系：

- 推理时首帧用 `generate_uniform_pointmap` 生成 [-1,1] 倾斜平面初始化
- 训练数据的首帧 c2w 为单位阵，与推理初始化一致

---

## 8. 深度物理尺度分析

### 8.1 三个源的深度数据对比

| 属性                 | 4DNeX-10M                      | OmniWorld-Game      | OmniWorld-HOI4D     |
| -------------------- | ---------------------------- | ------------------- | ------------------- |
| dtype                | float16                      | uint16              | uint16              |
| 值域                 | 0.1 – 1.2                    | 3000 – 62000        | 500 – 1400          |
| 类型                 | **scale-ambiguous 相对深度** | **uint16 未知单位** | **uint16 未知单位** |
| 相机平移量级         | ~0.001                       | ~0.04 – 0.13        | ~0.02 – 8.3         |
| depth/translation 比 | ~450:1                       | ~250000:1           | 不确定              |

### 8.2 对训练的影响

**不需要在 adapter 层做 depth scale factor 转换**。原因：

1. **Per-clip percentile 归一化天然吞掉了 scale 差异**——无论原始深度是 0.1 还是 34000，归一化后都是 [-1, 1]
2. **模型学的不是绝对 3D 坐标**——DiT 学习的是「首帧+prompt → XYZ 运动模式」的条件分布
3. **反投影出的 XYZ 在每个 clip 内部是几何一致的**——深度、内参、外参来自同一个坐标系
4. **与旧管线和论文一致**——`build_wan_dataset.py` 和 `core/annotation.py` 同样不做 scale 转换

### 8.3 XYZ norm scale 实际数据

```
4dnex/00000028:              scale=0.3606
omniworld_game/0365cd4c75bc:   scale=15890.2871
omniworld_hoi4d/ZY..._T1:     scale=712.5691
```

这就是 per-clip 归一化的效果——不同源的 scale 差 4 个数量级，但归一化后训练模型看到的都是 [-1, 1] 范围。

### 8.4 若未来需要确定 OmniWorld depth scale

可通过**重投影误差实验**确定：

1. 取两帧，已知外参平移差 ||t_B - t_A||
2. 假设 `depth_metric = depth_uint16 / scale_factor`
3. 反投影帧 A 的深度到 3D → 用帧 B 外参投影回 2D → 计算重投影误差
4. 遍历不同 `scale_factor`，误差最小的即为正确值

但当前**没有文档记录 scale factor**（JSON 中无 `depth_scale` 字段），不应硬编码猜测值。

---

## 9. 帧数处理鲁棒性

| 场景                    | 处理方式                                                             | 状态 |
| ----------------------- | -------------------------------------------------------------------- | ---- |
| 帧数 < num_frames       | 4DNeX-10M: `make_windows` → `(0, length)` → builder `reverse_pad` 补齐 | ✅   |
| 帧数 = num_frames       | 直接使用                                                             | ✅   |
| 帧数 > num_frames       | 4DNeX-10M: `make_windows` 滑窗切多个 clip                              | ✅   |
| OmniWorld 帧缺失        | 任一帧缺数据即跳过整个 clip（保守但安全）                            | ✅   |
| `frame_range_in_source` | 取 `clip.frame_ids[0..last_idx]`，不含 pad 帧                        | ✅   |
| pad 后内参/外参         | 同步 reverse_pad，不越界                                             | ✅   |
| RGB 与 depth 分辨率不同 | 各自独立 resize+crop，反投影在深度原生分辨率下完成                   | ✅   |

**已知限制**：OmniWorld adapter 对数据完整性要求严格——如果 3000 帧中只有 1 帧缺深度，整个 clip 被丢弃。这是有意为之的安全策略。

---

## 10. Meta 信息字段说明

每个 clip 目录下的 `meta.json` 包含：

| 字段                    | 类型         | 说明                               |
| ----------------------- | ------------ | ---------------------------------- |
| `video_id`              | string       | 视频 ID（不含源前缀）              |
| `clip_id`               | string       | clip 名称（如 `clip_0`）           |
| `clip_index`            | int          | clip 在该 video 下的序号           |
| `source_dataset`        | string       | 数据源标识                         |
| `source_entry_id`       | string       | 可溯源的唯一 ID                    |
| `num_frames`            | int          | 输出帧数（固定 81）                |
| `original_frames`       | int          | 原始有效帧数（pad 前）             |
| `fps`                   | int          | 帧率                               |
| `resolution`            | [H, W]       | 输出分辨率                         |
| `frame_range_in_source` | [start, end] | 源中的帧范围                       |
| `is_padded`             | bool         | 是否做了 reverse pad               |
| `camera.intrinsics`     | [T, 3×3]     | 每帧内参（**已调整到输出分辨率**） |
| `camera.extrinsics_c2w` | [T, 4×4]     | 每帧外参（首帧对齐后）             |
| `xyz_norm.center`       | [x, y, z]    | XYZ 归一化中心                     |
| `xyz_norm.scale`        | float        | XYZ 归一化尺度                     |
| `xyz_norm.percentile`   | [2, 98]      | 使用的百分位数                     |

---

## 11. 与旧管线及论文的对比

### 11.1 与 build_wan_dataset.py 对比

| 维度            | 旧管线                           | 新管线                                       |
| --------------- | -------------------------------- | -------------------------------------------- |
| 架构            | 单文件，仅支持 4DNeX-10M           | Adapter + Builder，支持 3 个源               |
| 目录结构        | flat `videos/{video_id}/clip_N/` | 三级 `videos/{source}/{video_id}/{clip_id}/` |
| 归一化粒度      | Per-frame (axis=1)               | **Per-clip** (全帧 flatten)                  |
| 内参调整        | ❌ 不调整                        | ✅ `adjust_intrinsics_for_resize_crop()`     |
| 深度 scale 转换 | ❌ 无                            | ❌ 同样无（设计一致）                        |
| index.json      | 无缩进                           | indent=2                                     |

### 11.2 与 One-4D / 4DNeX 论文对比

| 论文描述                                            | 实现对应                                         |
| --------------------------------------------------- | ------------------------------------------------ |
| "6D video: RGB + XYZ pixel-aligned point maps"      | `video.mp4` + `xyz.mp4`                          |
| "$$X^{init}$$ 倾斜深度平面"                         | `generate_uniform_pointmap()`                    |
| "Latent-space $$\hat{x} = \frac{x - \mu}{\sigma}$$" | `encoded_pm_mean=-0.13, encoded_pm_std=1.70`     |
| "Width-wise fusion"                                 | `torch.concat([rgb_latent, xyz_latent], dim=-1)` |
| 论文未提及 pixel-space percentile norm              | 这是实现层面细节，旧管线和新管线均采用           |

---

## 12. 已修复的正确性问题

| #   | 问题                                                        | 修复                                                                |
| --- | ----------------------------------------------------------- | ------------------------------------------------------------------- |
| 1   | 内参未因 resize+crop 调整                                   | 新增 `adjust_intrinsics_for_resize_crop()`，meta 中存储调整后的内参 |
| 2   | `_choose_depth_path` 对不存在目录调用 glob                  | 添加 `depth_dir.exists()` 守卫                                      |
| 3   | `process_clip` 返回冗余数据（未使用的 intr/ext/xyz_normed） | 精简为 6 个返回值                                                   |
| 4   | 4DNeX-10M `source_entry_id` 使用窗口局部索引                  | 改为实际源帧索引 `start_src + ws`                                   |
| 5   | `index.json` 无缩进                                         | 添加 `indent=2`                                                     |
| 6   | `video_id` 前缀与 source_dataset 冗余                       | 去掉源前缀，依赖目录层级区分                                        |

---

## 13. `build_dataset.py` — 构建 videos + index

### 设计

- 每次处理**一个数据源**（`--source` 必选），多源分别运行
- **串行处理**：逐个 clip collect → load → process → write，避免多线程/多进程带来的 ffmpeg 冲突和内存问题
- **边 collect 边 build**：使用 `deque` 流式消费 descriptor，collect 完成后立即开始 build，无需等待全部完成
- **payload 分离架构**：`build_clip_payload()` 负责 load + compute（返回内存中的 payload），`write_clip_payload()` 负责 I/O（写文件 + 拷贝），方便未来扩展
- **增量处理**：`skip_existing` 检查输出文件是否完整，中断后可继续
- **collect 缓存**：首次扫描结果存为 `.collect_cache_{source}_{hash}.pkl`，参数变化时自动失效，`--recollect` 强制刷新
- **FUSE 兼容**：`--tmp_dir` 先写本地磁盘再拷到 TOS，避免 ffmpeg seek 失败；拷贝有 5 次指数退避重试
- **index.json 流式写入**：entries 先写 jsonl 临时文件，最后组装为 index.json，避免大量 entry 占内存

### 运行

```bash
# 逐个数据源处理
python build_dataset.py --source 4dnex
python build_dataset.py --source omniworld_game
python build_dataset.py --source omniworld_hoi4d

# 输出到 TOS（FUSE 挂载），用本地 tmp_dir 中转
python build_dataset.py --source omniworld_hoi4d \
  --out /mnt/tos/mydataset --tmp_dir /tmp/build_cache

# 完整参数
python build_dataset.py \
  --raw_root ./raw_data \
  --out ./data \
  --source 4dnex \
  --num_frames 81 \
  --fps 24 \
  --resolution_h 480 \
  --resolution_w 720 \
  --stride 40 \
  --split train

# 强制重新处理
python build_dataset.py --source 4dnex --no_skip_existing

# 强制重新扫描原始数据
python build_dataset.py --source 4dnex --recollect
```

| 参数                 | 默认值       | 说明                                    |
| -------------------- | ------------ | --------------------------------------- |
| `--source`           | （必选）     | 要处理的数据源                          |
| `--raw_root`         | `./raw_data` | 原始数据根目录                          |
| `--out`              | `./data`     | 输出目录                                |
| `--num_frames`       | 81           | 每个 clip 的帧数                        |
| `--fps`              | 24           | 输出帧率                                |
| `--resolution_h/w`   | 480 / 720    | 输出分辨率                              |
| `--stride`           | 40           | 4DNeX-10M 长序列滑窗步长               |
| `--split`            | train        | 写入 index.json 的 split 标签           |
| `--no_skip_existing` | off          | 强制重新处理已存在输出文件的 clip       |
| `--recollect`        | off          | 强制重新扫描原始数据（忽略 collect 缓存） |
| `--tmp_dir`          | 无           | 本地临时目录，FUSE 挂载时必需           |

---

## 14. `process_dataset.py` — 编码 latents

### 设计

- 读取 `build_dataset.py` 产出的 `videos/` + `index.json`，编码为 latent tensors
- **多 GPU 支持**：通过 `torchrun` 启动，clip 列表按 rank **连续切分**（不是 round-robin），每张卡处理自己的分片
- **分片逻辑**：`split_list(all_clips, world_size)[rank]`，例如 100 clips / 2 GPU → rank 0 取 clip 0-49，rank 1 取 clip 50-99
- **text 编码去重**：text_embeds + text_ids 按 caption SHA-256 hash 存到 `latents_cache/{hash}.pt`，相同 caption 只编码一次
- **增量处理**：per-clip latent（rgb/xyz/visual）和 text cache 分别检查是否已存在
- **index.json 更新**：处理完成后 rank 0 自动为所有 clip 补上 `text_latent_path` 字段

### 编码产物

| 编码项 | 模型 | 存储位置 | 去重 |
| ------ | ---- | -------- | ---- |
| rgb_latent.pt | Wan VAE | `latents/{path}/` | per-clip |
| xyz_latent.pt | Wan VAE | `latents/{path}/` | per-clip |
| visual_embeds.pt | CLIP ViT | `latents/{path}/` | per-clip |
| {hash}.pt (text) | UMT5 | `latents_cache/` | 按 caption hash 去重 |

### 运行

```bash
# 单 GPU
python process_dataset.py \
  --dataset_root ./data \
  --model_path ./pretrained/Wan2.1-I2V-14B-480P-Diffusers

# 多 GPU（clip 自动按 rank 平分）
torchrun --nproc_per_node=4 process_dataset.py \
  --dataset_root ./data \
  --model_path ./pretrained/Wan2.1-I2V-14B-480P-Diffusers

# 强制全部重新编码
python process_dataset.py \
  --dataset_root ./data \
  --model_path ./pretrained/Wan2.1-I2V-14B-480P-Diffusers \
  --no_skip_existing

# 使用不同的 latents 目录（版本切换）
python process_dataset.py \
  --dataset_root ./data \
  --model_path ./pretrained/Wan2.1-I2V-14B-480P-Diffusers \
  --latents_dir latents_v2
```

| 参数                   | 默认值  | 说明                                          |
| ---------------------- | ------- | --------------------------------------------- |
| `--dataset_root`       | `./data`| 数据集根目录（需含 index.json + videos/）     |
| `--model_path`         | （必选）| Wan 2.1 预训练模型路径（Diffusers 格式）      |
| `--device`             | auto    | 单 GPU 时的设备（分布式时自动按 rank 分配）   |
| `--max_text_seq_length` | 512    | UMT5 最大 token 长度                          |
| `--no_skip_existing`   | off     | 强制重新编码所有 clip                         |
| `--latents_dir`        | auto    | latents 目录名（默认从 index.json 读取）      |

### GPU 分片示例

```
torchrun --nproc_per_node=2  (100 clips)

rank 0: clips[0:50]   → GPU 0
rank 1: clips[50:100]  → GPU 1

每个 rank 独立处理自己的分片，互不重叠。
text cache 通过文件系统天然去重（两个 rank 可能同时
编码同一 caption，但结果相同，写入幂等）。
```

---

## 15. 完整流水线

```bash
# Step 1: 构建 videos + index（逐个 source，串行）
python build_dataset.py --source omniworld_hoi4d \
  --raw_root ./raw_data --out ./data --tmp_dir /tmp/build_cache
python build_dataset.py --source omniworld_game \
  --raw_root ./raw_data --out ./data --tmp_dir /tmp/build_cache

# Step 2: 编码 latents（多 GPU 并行）
torchrun --nproc_per_node=4 process_dataset.py \
  --dataset_root ./data \
  --model_path ./pretrained/Wan2.1-I2V-14B-480P-Diffusers

# 产出结构:
# ./data/
# ├── index.json          (含 text_latent_path)
# ├── videos/             (RGB/XYZ MP4 + meta)
# ├── latents/            (per-clip VAE/CLIP latents)
# └── latents_cache/      (text embeddings, 按 hash 去重)
```
