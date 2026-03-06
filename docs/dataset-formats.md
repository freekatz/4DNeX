# 数据集格式文档

本文档记录 4DNeX 项目目前使用的三类数据集的组织结构、文件格式和字段说明。

---

## 总览

| 属性             | 4DNeX-10M             | OmniWorld-Game        | OmniWorld-HOI4D                           |
| ---------------- | --------------------- | --------------------- | ----------------------------------------- |
| 数据根目录       | `data/`               | `raw_data/omniworld/` | `raw_data/omniworld/` + `raw_data/hoi4d/` |
| 场景类型         | 动态场景 / 静态场景   | 游戏引擎渲染          | 真实人手-物体交互                         |
| RGB 来源         | MP4 视频              | PNG 序列帧            | MP4 视频                                  |
| RGB 分辨率       | 视频原始分辨率        | 1280×720              | 1920×1080                                 |
| 深度格式         | float16 `.npy` (逐帧) | uint16 灰度 PNG       | uint16 灰度 PNG                           |
| 深度分辨率       | 288×512               | 1280×720              | 1920×1080                                 |
| 深度连续性       | 逐帧连续              | 稀疏 (非每帧都有)     | 逐帧连续                                  |
| 相机格式         | TXT (轨迹 + 内参)     | JSON (两种格式)       | JSON (droidclib 格式)                     |
| 标注文本         | CSV                   | JSON (6 种 caption)   | TXT 纯文本                                |
| 帧率             | 未指定                | 24 fps                | 未明确指定                                |
| 当前样本场景数   | 1                     | 1                     | 1                                         |
| 当前样本 clip 数 | 1                     | 43                    | 6                                         |

---

## 1. 4DNeX-10M 数据集

### 1.1 目录结构

```
data/
├── caption/
│   ├── dynamic1.csv                   # 动态子集 1 的视频级别 caption
│   ├── dynamic2.csv                   # 动态子集 2 的视频级别 caption
│   └── ...
├── dynamic1/
│   └── {video_id}/                    # 8 位数字 ID，如 00000028
│       └── clip_{start}-{end}/        # 6 位帧号范围，如 clip_000000-000053
│           ├── pred_traj.txt          # 相机轨迹 (位姿)
│           ├── pred_intrinsics.txt    # 相机内参
│           ├── frame_0000.npy         # 逐帧深度图
│           ├── frame_0001.npy
│           ├── ...
│           └── frame_XXXX.npy
├── dynamic2/
│   └── {video_id}/
│       └── clip_{start}-{end}/
│           ├── pred_traj.txt
│           ├── pred_intrinsics.txt
│           ├── frame_0000.npy
│           └── ...
├── static/
│   └── (结构类似，包含 .npz 文件)
└── raw/
    ├── dynamic1/
    │   └── {video_id}.mp4             # 原始 RGB 视频
    └── dynamic2/
        └── {video_id}.mp4
```

4DNeX-10M 数据有三种子类型:

| 子类型    | 标识                        | 数据形式                                      |
| --------- | --------------------------- | --------------------------------------------- |
| dynamic   | 目录中含 `pred_traj.txt`    | 逐帧 `.npy` 深度 + TXT 轨迹/内参              |
| dynamic_3 | `dynamic/` 下的 `.npz` 文件 | NPZ 打包 (images, depths, intrinsic, cam_c2w) |
| static    | `static/` 下的 `.npz` 文件  | NPZ 打包 (pts3d, poses)，无独立深度图         |

### 1.2 文件格式

#### pred_traj.txt — 相机轨迹

每行 8 个空格分隔的浮点数，每帧一行:

```
frame_id  tx  ty  tz  qw  qx  qy  qz
```

| 列  | 含义              | 示例值                            |
| --- | ----------------- | --------------------------------- |
| 0   | 帧序号 (0-based)  | 0.0, 1.0, ..., 53.0               |
| 1-3 | 平移 (tx, ty, tz) | -0.000339, 0.000668, -0.0000955   |
| 4   | 四元数 qw         | 0.999821 (接近 1.0，近似单位旋转) |
| 5-7 | 四元数 qx, qy, qz | -0.01645, -0.00863, -0.00359      |

**四元数约定**: 文件中顺序为 **qw, qx, qy, qz**。使用 scipy 时需重排为 (qx, qy, qz, qw)。

**坐标系**: camera-to-world SE(3) 变换。

#### pred_intrinsics.txt — 相机内参

每行 9 个浮点数，为 3×3 相机内参矩阵按行展平:

```
fx  0  cx  0  fy  cy  0  0  1
```

还原为矩阵:

```
┌ fx   0   cx ┐     ┌ 558.01   0     256 ┐
│  0  fy   cy │  =  │   0    558.01  144 │
└  0   0    1 ┘     └   0      0      1  ┘
```

本数据集中所有帧内参相同（固定焦距）。

#### frame_XXXX.npy — 深度图

- 格式: NumPy 二进制数组
- Shape: `(288, 512)` — 单帧
- Dtype: `float16`
- 值域: ~0.1 到 ~1.2
- 命名: 4 位补零，如 `frame_0000.npy` 到 `frame_0053.npy`

#### dynamic_3 子类型 NPZ

NPZ 内包含:

| Key         | Shape        | Dtype            | 说明                                 |
| ----------- | ------------ | ---------------- | ------------------------------------ |
| `images`    | [T, H, W, 3] | uint8 或 float32 | RGB 图像                             |
| `depths`    | [T, H, W]    | float32          | 深度图                               |
| `intrinsic` | [3, 3]       | float32          | 单个内参矩阵 (所有帧共享)            |
| `cam_c2w`   | [T, 4, 4]    | float32          | camera-to-world SE(3)，已是 4×4 矩阵 |

#### static 子类型 NPZ

NPZ 内以嵌套 dict 形式存储:

| Key     | Shape        | 说明                  |
| ------- | ------------ | --------------------- |
| `pts3d` | [T, H, W, 3] | 预计算的 3D 点云      |
| `poses` | [T, 4, 4]    | camera-to-world SE(3) |

静态子类型无独立深度图，直接提供世界坐标系点云。

### 1.3 Caption 格式

`caption/dynamic1.csv`, `caption/dynamic2.csv`, etc. (每个动态子集对应一个 CSV):

| 列名      | 类型 | 说明                                     |
| --------- | ---- | ---------------------------------------- |
| `number`  | int  | 视频 ID (无前导零)，如 28 对应 00000028  |
| `caption` | str  | 长文本描述，包含场景、主体、运动、氛围等 |

Caption 按视频级别组织（一个视频一条 caption），而非按 clip 级别。

---

## 2. OmniWorld-Game 数据集

### 2.1 目录结构

```
raw_data/omniworld/
├── annotations/OmniWorld-Game/
│   └── {scene_id}/                              # 12 位十六进制，如 0365cd4c75bc
│       ├── {scene_id}_others/
│       │   ├── fps.txt                          # 帧率信息
│       │   ├── split_info.json                  # 时间分段定义
│       │   ├── camera/
│       │   │   ├── split_0.json                 # 相机参数 (focals + quats + trans)
│       │   │   ├── split_1.json
│       │   │   └── ...
│       │   ├── droidclib/
│       │   │   ├── split_0.json                 # 相机参数 (4×4 extrinsics)
│       │   │   ├── split_1.json
│       │   │   └── ...
│       │   ├── text/
│       │   │   ├── 000001_000081.json           # clip 级别 caption
│       │   │   ├── 000041_000121.json
│       │   │   └── ...
│       │   ├── subject_masks/
│       │   │   └── split_N.json                 # COCO RLE 编码 mask
│       │   └── gdino_mask/
│       │       └── XXXXXX.png                   # Grounding DINO mask
│       └── {scene_id}_depth_0000/
│           └── depth/
│               └── XXXXXX.png                   # 深度图 (稀疏)
└── videos/OmniWorld-Game/
    └── {scene_id}/
        └── {scene_id}_rgb_0000/
            └── color/
                └── XXXXXX.png                   # RGB 帧 (连续)
```

### 2.2 文件格式

#### fps.txt

纯文本，记录帧率:

```
FPS: 24.0
Processing time: 1177.11 seconds
```

#### split_info.json — 时间分段

```json
{
  "scene_name": "0365cd4c75bc",
  "split_num": 6,
  "split": [
    [0, 1, 2, ..., 400],      // split_0: 401 帧
    [401, 402, ..., 801],      // split_1: 401 帧
    [802, 803, ..., 979],      // split_2: 178 帧
    [983, 984, ..., 1383],     // split_3: 401 帧 (注意: 980-982 缺失)
    [1384, 1385, ..., 1784],   // split_4: 401 帧
    [1785, 1786, ..., 2161]    // split_5: 377 帧
  ]
}
```

每个 split 是一段时间连续的帧序号数组。帧序号不一定严格连续（可能存在跳帧，如 split_2 到 split_3 之间的 980-982）。

#### camera/split_N.json — 相机参数 (focal + quat 格式)

```json
{
  "focals": [1002.456, 1002.458, ...],          // [N] 每帧焦距 (像素)
  "quats": [[w, x, y, z], ...],                 // [N, 4] 四元数 (wxyz 顺序)
  "trans": [[tx, ty, tz], ...],                  // [N, 3] 平移
  "cx": 640.0,                                   // 主点 x
  "cy": 360.0,                                   // 主点 y
  "reproj_error_before_refine": 40.78,
  "reproj_error_after_refine": 34.45,
  "input_size": 0.5,
  "invalid_frame": []
}
```

**注意**: 前若干帧的四元数可能未归一化（norm >> 1），属于无效帧。有效帧四元数 norm ≈ 1.0。
`invalid_frame` 字段可能为空，不可靠地标记无效帧，需自行通过 norm 检测。

**四元数约定**: `[w, x, y, z]` 顺序。使用 scipy 时需重排为 `[x, y, z, w]`。

#### droidclib/split_N.json — 相机参数 (4×4 extrinsics 格式)

```json
{
  "split": [0, 1, 2, ..., 400],                 // 该 split 包含的帧号
  "game_crop_bound": [0, 0, 0, 0],              // 裁剪边界 (未使用时为全零)
  "crop_intrinsic": {
    "fx": 1014.193, "fy": 1014.192,
    "cx": 640.0, "cy": 360.0
  },
  "orig_intrinsic": {
    "fx": 1014.193, "fy": 1014.192,
    "cx": 640.0, "cy": 360.0
  },
  "extrinsics": [                                // [N, 4, 4] camera-to-world
    [[1.0, -8.04e-05, ...], [...], [...], [0,0,0,1]],
    ...
  ]
}
```

`droidclib` 格式比 `camera` 格式更完整，已提供 4×4 extrinsics 矩阵，无需从四元数转换。**推荐优先使用 droidclib**。

两种格式的内参略有差异（droidclib 为单一固定内参，camera 提供逐帧焦距）。

#### text/SSSSSS_EEEEEE.json — Clip 级别 Caption

文件名编码帧范围: `000001_000081.json` 表示帧 1 到帧 81。

```json
{
  "captions": {
    "Short_Caption": "简短描述 (1-2 句)",
    "PC_Caption": "角色动作描述 (人物中心视角)",
    "Background_Caption": "背景环境描述",
    "Camera_Caption": "镜头运动描述",
    "Video_Caption": "综合完整描述 (最详细)",
    "Key_Tags": "逗号分隔的语义标签"
  }
}
```

6 种 caption 类型从不同维度描述同一段视频。`Video_Caption` 最完整，适合作为训练文本。

#### depth/XXXXXX.png — 深度图

- 格式: 16-bit 灰度 PNG
- 分辨率: 1280×720
- Dtype: uint16
- 值域: ~3150 到 ~34280
- 命名: 6 位补零，如 `000000.png`

**稀疏性**: 深度目录下的帧不连续。并非每个 RGB 帧都有对应深度图。加载时需检测缺失帧。

#### color/XXXXXX.png — RGB 帧

- 格式: 24-bit RGB PNG
- 分辨率: 1280×720
- 命名: 6 位补零，如 `000000.png` 到 `002160.png`
- 连续性: 每帧都有

#### subject_masks/split_N.json — 主体 Mask

以帧为 key 的字典，值为 COCO RLE 编码 mask:

```json
{
  "000000.png": {
    "frame_idx": 0,
    "mask_rle": {
      "size": [720, 1280],
      "counts": "mU\\:3\\f01N1O200O10O001N2O..."
    }
  },
  ...
}
```

解码需要 `pycocotools.mask.decode()` 或等效实现。

#### gdino_mask/XXXXXX.png — Grounding DINO Mask

- 格式: 灰度 PNG
- 分辨率: 1280×720
- 内容: Grounding DINO 模型生成的物体/语义分割 mask

---

## 3. OmniWorld-HOI4D 数据集

### 3.1 目录结构

```
raw_data/omniworld/
└── annotations/OmniWorld-HOI4D/
    └── {scene_dir_name}/                          # 如 ZY20210800001_H1_C1_N19_S100_s02_T1
        ├── camera/
        │   ├── split_info.json                    # 分段信息
        │   ├── image_list.json                    # 帧列表
        │   └── recon/
        │       └── split_0/
        │           └── info.json                  # 相机内外参 (同 droidclib 格式)
        ├── prior_depth/
        │   ├── 00000.png                          # 深度图 (连续)
        │   ├── 00001.png
        │   └── ...
        ├── text/
        │   ├── 0_80.txt                           # clip 级别文本描述
        │   ├── 40_120.txt
        │   └── ...
        └── flow/
            └── XXXXX/                             # 光流帧目录
                ├── flow_u_16.png
                ├── flow_v_16.png
                └── flow_vis.png

raw_data/hoi4d/
└── {scene_name 各级目录}/                         # 如 ZY20210800001/H1/C1/N19/S100/s02/T1
    └── align_rgb/
        └── image.mp4                              # 原始 RGB 视频
```

**注意**: RGB 视频位于 `raw_data/hoi4d/` 目录下（与 annotations 不在同一 root 下），路径由 `scene_name` 中的 `/` 分隔层级结构推导。

`scene_dir_name`（下划线连接）和 `scene_name`（斜线分隔）的对应关系:

- `scene_dir_name`: `ZY20210800001_H1_C1_N19_S100_s02_T1`
- `scene_name`: `ZY20210800001/H1/C1/N19/S100/s02/T1`

`scene_name` 存储在 `split_info.json` 中。

### 3.2 场景 ID 命名规则

`ZY20210800001_H1_C1_N19_S100_s02_T1` 各段含义推测:

| 段            | 示例          | 推测含义                   |
| ------------- | ------------- | -------------------------- |
| ZY{date}{seq} | ZY20210800001 | 日期标识 2021-08，序号 001 |
| H{n}          | H1            | 高度/视角编号              |
| C{n}          | C1            | 相机编号                   |
| N{n}          | N19           | 场景编号                   |
| S{n}          | S100          | 拍摄编号                   |
| s{n}          | s02           | 子序列编号                 |
| T{n}          | T1            | 轨迹/Take 编号             |

### 3.3 文件格式

#### split_info.json — 分段信息

```json
{
  "data_root": "hdd:s3://HOI4D/",
  "scene_name": "ZY20210800001/H1/C1/N19/S100/s02/T1",
  "idx": 1,
  "split_num": 1,
  "split": [[0, 1, 2, ..., 299]]
}
```

HOI4D 通常只有 1 个 split，包含全部 300 帧。

#### image_list.json — 帧列表

```json
["00000.jpg", "00001.jpg", ..., "00299.jpg"]
```

5 位补零的 JPEG 文件名列表。

#### camera/recon/split_N/info.json — 相机参数

格式与 OmniWorld-Game 的 droidclib 完全一致:

```json
{
  "split": [0, 1, 2, ..., 299],
  "game_crop_bound": [0, 0, 0, 0],
  "crop_intrinsic": {
    "fx": 1060.296, "fy": 1061.507,
    "cx": 971.521, "cy": 523.262
  },
  "orig_intrinsic": {
    "fx": 1060.296, "fy": 1061.507,
    "cx": 971.521, "cy": 523.262
  },
  "extrinsics": [
    [[1.0, ...], [...], [...], [0,0,0,1]],
    ...
  ]
}
```

extrinsics 为 [N, 4, 4] camera-to-world SE(3) 矩阵。内参为固定值（所有帧共享）。

#### prior_depth/XXXXX.png — 深度图

- 格式: 16-bit 灰度 PNG
- 分辨率: 1920×1080
- Dtype: uint16
- 值域: ~500 到 ~1400
- 命名: **5 位**补零，如 `00000.png` 到 `00299.png`
- 连续性: 每帧都有 (共 300 帧)

注意深度图命名为 5 位补零，与 OmniWorld-Game 的 6 位不同。

#### text/S_E.txt — 文本描述

文件名编码帧范围: `0_80.txt` 表示帧 0 到帧 80（包含两端）。

内容为纯文本段落描述，约 1000-1200 字符:

```
The video, shot from a firstperson perspective, captures a person
interacting with a small toy car placed on a white drawer...
```

多个 clip 之间帧范围有重叠（滑动窗口方式），如:
`0_80.txt`, `40_120.txt`, `80_160.txt`, `120_200.txt`, `160_240.txt`, `200_280.txt`

#### flow/XXXXX/ — 光流数据

每帧一个子目录，包含:

- `flow_u_16.png`: 水平光流分量 (16-bit PNG)
- `flow_v_16.png`: 垂直光流分量 (16-bit PNG)
- `flow_vis.png`: 光流可视化图

当前 pipeline 未使用光流数据。

#### align_rgb/image.mp4 — RGB 视频

- 格式: MPEG-4
- 分辨率: 推测 1920×1080 (与深度图一致)
- 帧数: 对应 image_list.json 中的帧数 (如 300 帧)
- 位置: `raw_data/hoi4d/{scene_name}/align_rgb/image.mp4`

---

## 格式差异对比

### 深度图格式

| 属性     | 4DNeX-10M (dynamic) | OmniWorld-Game  | OmniWorld-HOI4D |
| -------- | ------------------- | --------------- | --------------- |
| 文件格式 | `.npy` (NumPy)      | `.png` (16-bit) | `.png` (16-bit) |
| 分辨率   | 288×512             | 1280×720        | 1920×1080       |
| 数据类型 | float16             | uint16          | uint16          |
| 值域     | ~0.1 - ~1.2         | ~3150 - ~34280  | ~500 - ~1400    |
| 连续性   | 每帧都有            | 稀疏            | 每帧都有        |
| 帧号格式 | `frame_XXXX` (4位)  | `XXXXXX` (6位)  | `XXXXX` (5位)   |

### 相机参数格式

| 属性       | 4DNeX-10M       | OmniWorld-Game (camera) | OmniWorld-Game (droidclib) | OmniWorld-HOI4D  |
| ---------- | --------------- | ----------------------- | -------------------------- | ---------------- |
| 格式       | TXT             | JSON                    | JSON                       | JSON             |
| 外参表示   | 四元数+平移     | 四元数+平移             | 4×4 矩阵                   | 4×4 矩阵         |
| 四元数顺序 | qw,qx,qy,qz     | w,x,y,z                 | —                          | —                |
| 内参       | 逐帧 3×3        | 逐帧焦距+固定主点       | 固定 fx,fy,cx,cy           | 固定 fx,fy,cx,cy |
| 坐标系     | camera-to-world | camera-to-world         | camera-to-world            | camera-to-world  |

### Caption 格式

| 属性           | 4DNeX-10M      | OmniWorld-Game                            | OmniWorld-HOI4D |
| -------------- | -------------- | ----------------------------------------- | --------------- |
| 文件格式       | CSV            | JSON                                      | TXT             |
| 粒度           | 视频级别       | clip 级别                                 | clip 级别       |
| caption 类型数 | 1              | 6 (Short/PC/Background/Camera/Video/Tags) | 1               |
| 文件名规则     | `{subdir}.csv` | `SSSSSS_EEEEEE.json`                      | `S_E.txt`       |
| 帧范围包含端点 | —              | 包含两端                                  | 包含两端        |
| 窗口重叠       | —              | 有重叠                                    | 有重叠          |
