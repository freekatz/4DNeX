"""
scripts/vis_dynamic.py — 可视化 data/dynamic 下单个 clip 的所有数据

输出（保存到 --out_dir，默认 ./vis_out/<scene>/<clip>/）：
  rgb_frames/      原始视频帧（从 MP4 裁剪的对应 clip 帧）
  depth_frames/    深度图（伪彩色，viridis）
  conf_frames/     置信度图（conf_*.npy，伪彩色）
  mask_frames/     动态 mask（白=动态区域）
  overlay_frames/  RGB + mask 叠加（动态区域高亮红色）
  camera_traj.png  相机轨迹（3D scatter，XYZ 位移随时间着色）
  summary.png      4 路拼图：第 0/中间/最后帧的 RGB / 深度 / 置信度 / mask

运行示例：
  python scripts/vis_dynamic.py
  python scripts/vis_dynamic.py --scene 00000028 --clip clip_000000-000053 --out_dir ./vis_out
"""

import os
import sys
import argparse
import numpy as np
import cv2
import imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from pathlib import Path
from scipy.spatial.transform import Rotation


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def to_colormap(arr, cmap='viridis', vmin=None, vmax=None):
    """将单通道 float 数组映射为 RGB uint8 伪彩色图。"""
    arr = arr.astype(np.float32)
    vmin = arr.min() if vmin is None else vmin
    vmax = arr.max() if vmax is None else vmax
    norm = np.clip((arr - vmin) / (vmax - vmin + 1e-8), 0, 1)
    cm = plt.get_cmap(cmap)
    rgb = (cm(norm)[..., :3] * 255).astype(np.uint8)
    return rgb


def pose_to_xyz(traj):
    """
    从 TUM 格式轨迹提取相机中心 XYZ。
    每行格式：[timestamp, tx, ty, tz, qx, qy, qz, qw]
    """
    return traj[:, 1:4]  # [T, 3]


def traj_to_cam2world(traj):
    """
    TUM 轨迹 → cam2world 齐次矩阵 [T, 4, 4]。
    """
    T = traj.shape[0]
    # qx qy qz qw → rotation matrix
    quats = np.concatenate([traj[:, 5:], traj[:, 4:5]], axis=-1)  # xyzw → wxyz→xyzw already correct for scipy
    # scipy from_quat expects [x,y,z,w]
    R = Rotation.from_quat(quats).as_matrix()  # [T,3,3]
    t = traj[:, 1:4, None]                     # [T,3,1]
    mat = np.concatenate([R, t], axis=-1)       # [T,3,4]
    bottom = np.tile(np.array([[0, 0, 0, 1]], dtype=np.float32), (T, 1, 1))
    return np.concatenate([mat, bottom], axis=1).astype(np.float32)  # [T,4,4]


# ── 主可视化流程 ──────────────────────────────────────────────────────────────

def visualize_clip(scene_dir: Path, clip_dir: Path, raw_video: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. 读取元数据 ──────────────────────────────────────────────────────────
    traj  = np.loadtxt(clip_dir / 'pred_traj.txt')         # [T, 8]
    intr  = np.loadtxt(clip_dir / 'pred_intrinsics.txt')   # [T, 9]
    T     = traj.shape[0]

    # clip 起止帧（从目录名解析）
    clip_name = clip_dir.name                               # e.g. clip_000000-000053
    parts = clip_name.replace('clip_', '').split('-')
    clip_start = int(parts[0])
    clip_end   = int(parts[1])
    assert T == clip_end - clip_start + 1, (
        f"Traj length {T} != clip length {clip_end - clip_start + 1}"
    )

    print(f"  Scene : {scene_dir.name}")
    print(f"  Clip  : {clip_name}  ({T} frames, start={clip_start})")

    # ── 2. 读取 MP4 帧 ─────────────────────────────────────────────────────────
    reader = imageio.get_reader(str(raw_video))
    rgb_frames = []
    for t in range(T):
        frame = reader.get_data(clip_start + t)   # [H, W, 3] uint8
        rgb_frames.append(frame)
    reader.close()
    rgb_frames = np.stack(rgb_frames)             # [T, H, W, 3]
    H_raw, W_raw = rgb_frames.shape[1:3]
    print(f"  RGB   : {H_raw}x{W_raw} x {T} frames")

    # ── 3. 读取深度 / 置信度 / mask ────────────────────────────────────────────
    depths = []
    confs  = []
    masks  = []
    for t in range(T):
        depths.append(np.load(clip_dir / f'frame_{t:04d}.npy').astype(np.float32))
        confs.append(np.load(clip_dir  / f'conf_{t}.npy').astype(np.float32))
        masks.append(cv2.imread(str(clip_dir / f'dynamic_mask_{t}.png'), cv2.IMREAD_GRAYSCALE))

    depths = np.stack(depths)   # [T, H, W]
    confs  = np.stack(confs)    # [T, H, W]
    masks  = np.stack(masks)    # [T, H, W]
    H, W   = depths.shape[1:3]
    print(f"  Depth : {H}x{W}, range=[{depths.min():.3f}, {depths.max():.3f}]")
    print(f"  Conf  : range=[{confs.min():.4f}, {confs.max():.4f}]")
    print(f"  Mask  : {(masks > 0).mean()*100:.1f}% dynamic pixels (avg)")

    # 深度 / 置信度 全局归一化（统一色标，方便对比帧间变化）
    d_vmin, d_vmax = np.percentile(depths, 2), np.percentile(depths, 98)
    c_vmin, c_vmax = confs.min(), confs.max()

    # ── 4. 保存逐帧图像 ────────────────────────────────────────────────────────
    dirs = {
        'rgb':     out_dir / 'rgb_frames',
        'depth':   out_dir / 'depth_frames',
        'conf':    out_dir / 'conf_frames',
        'mask':    out_dir / 'mask_frames',
        'overlay': out_dir / 'overlay_frames',
    }
    for d in dirs.values():
        d.mkdir(exist_ok=True)

    for t in range(T):
        rgb   = rgb_frames[t]
        depth = depths[t]
        conf  = confs[t]
        mask  = masks[t]

        # resize rgb → depth 分辨率（可视化对齐用）
        rgb_small = cv2.resize(rgb, (W, H))

        # 深度伪彩色（近=暖色，远=冷色，用 plasma）
        depth_color = to_colormap(depth, cmap='plasma', vmin=d_vmin, vmax=d_vmax)

        # 置信度伪彩色（绿=高置信）
        conf_color = to_colormap(conf, cmap='RdYlGn', vmin=c_vmin, vmax=c_vmax)

        # mask：0=静态(黑) / 255=动态(白)
        mask_vis = np.stack([mask, mask, mask], axis=-1)

        # overlay：动态区域染红
        overlay = rgb_small.copy()
        dyn = mask > 0
        overlay[dyn] = (overlay[dyn].astype(np.float32) * 0.4 + np.array([255, 0, 0]) * 0.6).clip(0, 255).astype(np.uint8)

        imageio.imwrite(dirs['rgb']     / f'{t:04d}.png', rgb_small)
        imageio.imwrite(dirs['depth']   / f'{t:04d}.png', depth_color)
        imageio.imwrite(dirs['conf']    / f'{t:04d}.png', conf_color)
        imageio.imwrite(dirs['mask']    / f'{t:04d}.png', mask_vis)
        imageio.imwrite(dirs['overlay'] / f'{t:04d}.png', overlay)

    print(f"  Saved frame images → {out_dir}/{{rgb,depth,conf,mask,overlay}}_frames/")

    # ── 5. 相机轨迹图 ──────────────────────────────────────────────────────────
    xyz = pose_to_xyz(traj)   # [T, 3]
    cam2world = traj_to_cam2world(traj)

    fig = plt.figure(figsize=(14, 5))

    # 5a. 3D 轨迹（颜色=时间进度）
    ax3d = fig.add_subplot(131, projection='3d')
    sc = ax3d.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2],
                      c=np.arange(T), cmap='rainbow', s=20)
    ax3d.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], 'k-', lw=0.5, alpha=0.4)
    ax3d.scatter(*xyz[0],  color='green', s=80, zorder=5, label='start')
    ax3d.scatter(*xyz[-1], color='red',   s=80, zorder=5, label='end')
    plt.colorbar(sc, ax=ax3d, label='frame idx', pad=0.1)
    ax3d.set_title('Camera trajectory (3D)')
    ax3d.set_xlabel('X'); ax3d.set_ylabel('Y'); ax3d.set_zlabel('Z')
    ax3d.legend(fontsize=7)

    # 5b. XYZ 分量随时间变化
    ax2 = fig.add_subplot(132)
    for i, (label, color) in enumerate(zip(['X', 'Y', 'Z'], ['r', 'g', 'b'])):
        ax2.plot(xyz[:, i], color=color, label=label)
    ax2.set_xlabel('frame'); ax2.set_ylabel('translation (m)')
    ax2.set_title('Camera translation over time')
    ax2.legend(); ax2.grid(True, alpha=0.3)

    # 5c. 相机朝向（Z 轴方向，即光轴方向）
    ax3 = fig.add_subplot(133)
    lookat = cam2world[:, :3, 2]   # Z column = optical axis in world
    for i, (label, color) in enumerate(zip(['Zx', 'Zy', 'Zz'], ['r', 'g', 'b'])):
        ax3.plot(lookat[:, i], color=color, label=label)
    ax3.set_xlabel('frame'); ax3.set_ylabel('direction')
    ax3.set_title('Camera optical axis (world)')
    ax3.legend(); ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    traj_path = out_dir / 'camera_traj.png'
    plt.savefig(traj_path, dpi=150)
    plt.close()
    print(f"  Saved camera trajectory → {traj_path}")

    # ── 6. 内参随时间变化 ──────────────────────────────────────────────────────
    # intr 每行: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
    fx = intr[:, 0]; fy = intr[:, 4]
    cx = intr[:, 2]; cy = intr[:, 5]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(fx, label='fx', color='r'); axes[0].plot(fy, label='fy', color='b')
    axes[0].set_title('Focal length over time'); axes[0].set_xlabel('frame')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(cx, label='cx', color='r'); axes[1].plot(cy, label='cy', color='b')
    axes[1].set_title('Principal point over time'); axes[1].set_xlabel('frame')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    intr_path = out_dir / 'intrinsics.png'
    plt.savefig(intr_path, dpi=150)
    plt.close()
    print(f"  Saved intrinsics plot  → {intr_path}")

    # ── 7. 深度统计图 ──────────────────────────────────────────────────────────
    depth_mean = depths.mean(axis=(1, 2))
    depth_p10  = np.percentile(depths, 10, axis=(1, 2))
    depth_p90  = np.percentile(depths, 90, axis=(1, 2))

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.fill_between(range(T), depth_p10, depth_p90, alpha=0.3, label='p10-p90')
    ax.plot(depth_mean, label='mean depth', color='navy')
    ax.set_xlabel('frame'); ax.set_ylabel('depth (scene units)')
    ax.set_title('Depth statistics over time')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    depth_stat_path = out_dir / 'depth_stats.png'
    plt.savefig(depth_stat_path, dpi=150)
    plt.close()
    print(f"  Saved depth stats      → {depth_stat_path}")

    # ── 8. Summary 拼图（首帧 / 中间帧 / 末帧） ────────────────────────────────
    key_frames = [0, T // 2, T - 1]
    fig, axes = plt.subplots(4, 3, figsize=(15, 14))
    row_labels = ['RGB', 'Depth', 'Conf', 'Mask+Overlay']

    for col, t in enumerate(key_frames):
        rgb_s  = cv2.resize(rgb_frames[t], (W, H))
        dep_c  = to_colormap(depths[t], cmap='plasma', vmin=d_vmin, vmax=d_vmax)
        conf_c = to_colormap(confs[t],  cmap='RdYlGn', vmin=c_vmin, vmax=c_vmax)

        overlay = rgb_s.copy()
        dyn = masks[t] > 0
        overlay[dyn] = (overlay[dyn].astype(np.float32) * 0.4 + np.array([255, 0, 0]) * 0.6).clip(0, 255).astype(np.uint8)

        for row, img in enumerate([rgb_s, dep_c, conf_c, overlay]):
            axes[row, col].imshow(img)
            axes[row, col].axis('off')
            if row == 0:
                axes[row, col].set_title(f'frame {t}', fontsize=11, fontweight='bold')
        axes[0, col].set_title(f'frame {t}  ({clip_start+t} in video)', fontsize=10)

    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label, fontsize=11, rotation=90, labelpad=4)

    plt.suptitle(f'{scene_dir.name} / {clip_name}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    summary_path = out_dir / 'summary.png'
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved summary grid     → {summary_path}")

    # ── 9. 动态 mask 覆盖率随时间 ─────────────────────────────────────────────
    mask_ratio = (masks > 0).mean(axis=(1, 2))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.fill_between(range(T), 0, mask_ratio * 100, alpha=0.4, color='tomato')
    ax.plot(mask_ratio * 100, color='tomato', lw=1.5, label='dynamic area %')
    ax.set_xlabel('frame'); ax.set_ylabel('% pixels')
    ax.set_title('Dynamic mask coverage over time')
    ax.set_ylim(0, 100); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    mask_stat_path = out_dir / 'mask_coverage.png'
    plt.savefig(mask_stat_path, dpi=150)
    plt.close()
    print(f"  Saved mask coverage    → {mask_stat_path}")

    print(f"\n  Done. All outputs in: {out_dir}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Visualize data/dynamic clip')
    parser.add_argument('--data_dir',  type=str, default='./data',
                        help='数据根目录，含 dynamic/ 和 raw/')
    parser.add_argument('--scene',     type=str, default=None,
                        help='场景 ID，如 00000028（默认取第一个）')
    parser.add_argument('--clip',      type=str, default=None,
                        help='clip 目录名，如 clip_000000-000053（默认取第一个）')
    parser.add_argument('--out_dir',   type=str, default='./vis_out',
                        help='可视化输出根目录')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    dyn_dir  = data_dir / 'dynamic'
    raw_dir  = data_dir / 'raw' / 'dynamic'

    if not dyn_dir.exists():
        print(f"ERROR: {dyn_dir} does not exist"); sys.exit(1)

    # 选择 scene
    scenes = sorted([d for d in dyn_dir.iterdir() if d.is_dir()])
    if not scenes:
        print(f"ERROR: no scene directories under {dyn_dir}"); sys.exit(1)
    if args.scene:
        scene_dir = dyn_dir / args.scene
        if not scene_dir.exists():
            print(f"ERROR: scene {args.scene} not found"); sys.exit(1)
    else:
        scene_dir = scenes[0]
        print(f"No --scene specified, using: {scene_dir.name}")

    # 选择 clip
    clips = sorted([d for d in scene_dir.iterdir() if d.is_dir()])
    if not clips:
        print(f"ERROR: no clip directories under {scene_dir}"); sys.exit(1)
    if args.clip:
        clip_dir = scene_dir / args.clip
        if not clip_dir.exists():
            print(f"ERROR: clip {args.clip} not found"); sys.exit(1)
    else:
        clip_dir = clips[0]
        print(f"No --clip specified, using: {clip_dir.name}")

    # raw video
    raw_video = raw_dir / f'{scene_dir.name}.mp4'
    if not raw_video.exists():
        print(f"ERROR: raw video not found: {raw_video}"); sys.exit(1)

    out_dir = Path(args.out_dir) / scene_dir.name / clip_dir.name
    print(f"\nVisualizing: {scene_dir.name}/{clip_dir.name}")
    print(f"Output dir : {out_dir}\n")

    visualize_clip(scene_dir, clip_dir, raw_video, out_dir)


if __name__ == '__main__':
    main()
