"""
rerun_vis.py — 4D 点云序列可视化

职责：
    将 pm_registration.py 生成的 *_reg.pkl（或其他数据源）中的
    逐帧点云、RGB 图像和相机位姿，记录到 Rerun SDK 的时序数据库中，
    保存为 .rrd 文件，可用 Rerun Viewer 离线或在线浏览。

数据来源（任选其一）：
    --pkl_dir       : *_reg.pkl 文件目录（pm_registration.py 的输出，最常用）
    --monst3r_dir   : MonST3R 重建目录（GT 数据验证用）
    --pointmap_mp4 + --rgb_mp4 : 双 MP4 格式（build_wan_dataset 中间产物）
    --pointmap_npy + --rgb_mp4 : NumPy 点云 + RGB 视频

完整可视化流程：
    python pm_registration.py --pkl_dir ./results     # 先注册
    python rerun_vis.py --pkl_dir ./results --rr_recording vis.rrd
    rerun vis.rrd --web-viewer                        # 浏览器打开

Rerun 坐标系约定：
    world 空间坐标由 T_world_camera（4×4 变换矩阵）描述每帧相机位姿。
    rr.set_time_sequence("frame", i) 为所有后续 log 调用设定时间轴索引，
    Viewer 中可以播放/暂停来查看动态序列。
"""

import glob
import time
import pickle
import itertools
from pathlib import Path

import numpy as np
import tyro
from tqdm.auto import tqdm

import rerun as rr
import imageio

from core.dataset import PointmapDataset
from core.annotation import Monst3RAnno


def log_image_plane_outline(rr, name, K, T_world_camera, H, W):
    """
    在 Rerun 中记录相机视锥体的图像平面轮廓线（4条线围成的矩形）。

    原理：将图像四个角点（像素坐标）通过 K⁻¹ 反投影到相机空间（z=1 平面），
    再通过 T_world_camera 变换到世界空间，最后记录为 3D 折线。

    Args:
        rr             : rerun 模块（注意：此函数接收的是模块而非 rr.log 调用）
        name           : Rerun 实体路径，如 "<pkl_name>"
        K              : (3, 3) 相机内参矩阵
        T_world_camera : (4, 4) 相机到世界的变换矩阵
        H, W           : 图像高度和宽度（像素）
    """
    z = 1.0  # 投影平面深度（单位：与点云尺度一致）

    # 图像四角像素坐标（顺时针），末尾重复起点以闭合线段
    corners_px = np.array([
        [0, 0],
        [W, 0],
        [W, H],
        [0, H],
        [0, 0],  # 闭合线段
    ])

    # 将像素坐标反投影到相机坐标系（z=1 的归一化平面）
    Kinv = np.linalg.inv(K)
    corners_cam = []
    for u, v in corners_px:
        vec = np.array([u, v, 1.0])
        xyz = Kinv @ vec
        xyz = xyz / xyz[2] * z   # 归一化到 z=1
        corners_cam.append(xyz)
    corners_cam = np.stack(corners_cam, axis=0)   # (5, 3)

    # 变换到世界坐标系
    R = T_world_camera[:3, :3]
    t = T_world_camera[:3, 3]
    corners_world = (R @ corners_cam.T).T + t     # (5, 3)

    rr.log(
        f"{name}/image_frustum_outline",
        rr.LineStrips3D([corners_world])  # 单条折线，形成相机视锥轮廓
    )


def main(
    downsample_factor: int = 1,       # 点云空间降采样倍率（1=不降采样，2=每隔2像素取1点）
    frame_gap: int = 1,               # 帧间隔，用于加速浏览（1=每帧都记录）
    max_frames: int = 100,            # 最多记录帧数
    use_mask: bool = False,           # 是否记录前景/背景分离的点云（需要 Pointmap 含 fg_pcd）
    pkl_dir: str = None,              # *_reg.pkl 目录（pm_registration.py 的输出）
    monst3r_dir: str = None,          # MonST3R 标注目录
    pointmap_mp4: str = None,         # 点云可视化视频路径
    rgb_mp4: str = None,              # RGB 视频路径
    pointmap_npy: str = None,         # 点云 NumPy 路径
    rr_recording: str = "pointmap_log",  # 输出 .rrd 文件名（不含扩展名或含 .rrd）
) -> None:
    """
    主函数：初始化 Rerun → 加载数据 → 按帧记录 → 保存 .rrd 文件。

    注意事项：
        - pkl_dir 模式读取 *_reg.pkl（而非 *.pkl），需先运行 pm_registration.py
        - downsample_factor 可有效减小 .rrd 文件体积（推荐 2 或 4）
        - rr.save() 在所有帧记录完成后一次性写入磁盘
    """
    # 初始化 Rerun 记录会话，名称也作为 Viewer 中的标题显示
    rr.init(rr_recording)

    # ── 数据加载调度 ──────────────────────────────────────────────────────────
    if pointmap_npy is not None and rgb_mp4 is not None:
        # 来源 1：NumPy 格式点云 + RGB 视频
        pointmap   = np.load(pointmap_npy)   # [F, H, W, 3]
        rgb_reader = imageio.get_reader(rgb_mp4, "ffmpeg")
        num_frames = min(pointmap.shape[0], len(rgb_reader), max_frames)

        def PointmapIterator():
            for i in range(num_frames):
                pm  = pointmap[i]
                rgb = rgb_reader.get_data(i).astype(np.float32) / 255.0
                H, W, _ = pm.shape
                yield type('Frame', (), {
                    'pcd':            pm.reshape(-1, 3),
                    'pcd_color':      rgb.reshape(-1, 3),
                    'rgb':            rgb,
                    'T_world_camera': np.eye(4),  # 无已知位姿，用单位矩阵占位
                }), "npy_frame"

    elif pointmap_mp4 is not None and rgb_mp4 is not None:
        # 来源 2：MP4 双视频格式（build_wan_dataset 输出的可视化视频）
        pointmap_reader = imageio.get_reader(pointmap_mp4, "ffmpeg")
        rgb_reader      = imageio.get_reader(rgb_mp4,      "ffmpeg")
        num_frames      = 81  # Wan 标准帧数

        def PointmapIterator():
            for i in range(num_frames):
                pm  = pointmap_reader.get_data(i).astype(np.float32) / 255.0
                rgb = rgb_reader.get_data(i).astype(np.float32) / 255.0
                H, W, _ = pm.shape
                yield type('Frame', (), {
                    'pcd':            pm.reshape(-1, 3),
                    'pcd_color':      rgb.reshape(-1, 3),
                    'rgb':            rgb,
                    'T_world_camera': np.eye(4),
                }), "mp4_frame"

    elif pkl_dir is not None:
        # 来源 3（最常用）：pm_registration.py 输出的注册点云 *_reg.pkl
        # 注意 glob 模式：只加载 _reg.pkl，跳过未注册的原始 *.pkl
        pkl_list = glob.glob(f'{pkl_dir}/*_reg.pkl')

        def PointmapIterator():
            for pkl_path in sorted(pkl_list):
                yield pickle.load(open(pkl_path, 'rb')), pkl_path

    elif monst3r_dir is not None:
        # 来源 4：MonST3R 重建目录（GT 数据验证）
        monst3r_list = glob.glob(f'{monst3r_dir}/*/')

        def PointmapIterator():
            for scene_path in monst3r_list:
                yield Monst3RAnno(scene_path, max_frames=max_frames).pointmap, scene_path

    else:
        raise NotImplementedError("No data source provided")

    # ── 按帧记录到 Rerun ─────────────────────────────────────────────────────
    pointmap_iterator = PointmapIterator()

    if rgb_mp4 is not None:
        # MP4/NumPy 模式：迭代器每次 yield 一帧（非 Pointmap 对象，无 num_frames()）
        idx = 0
        for (frame, name) in pointmap_iterator:
            position = frame.pcd        # (H*W, 3) 或 (N, 3)
            color    = frame.pcd_color  # (N, 3)，值域 [0,1]
            rgb      = frame.rgb        # (H, W, 3)
            H, W, _  = rgb.shape

            # 空间降采样：reshape 到 (H, W, 3) 后按步长采样，再展平
            position = position.reshape(H, W, 3)[::downsample_factor, ::downsample_factor].reshape([-1, 3])
            color    = color.reshape(H, W, 3)[::downsample_factor, ::downsample_factor].reshape([-1, 3])

            # 设置当前时间帧（影响本次 rr.log 之后所有记录的时间戳）
            rr.set_time_sequence("frame", idx)

            # 记录点云（3D 坐标 + 颜色）
            rr.log(
                f"{name}/point_cloud",
                rr.Points3D(
                    positions=position,
                    colors=(color * 255).astype(np.uint8),
                ),
            )

            # 记录 RGB 图像（可在 Viewer 中与点云联动显示）
            rr.log(f"{name}/rgb_image", rr.Image(rgb))

            # 记录相机位姿（平移 + 旋转，用于绘制相机坐标轴）
            rr.log(
                f"{name}/camera",
                rr.Transform3D(
                    translation=frame.T_world_camera[:3, 3],
                    mat3x3=frame.T_world_camera[:3, :3],
                ),
            )

            # 可选：分别记录前景/背景点云（需 pm 含 fg_pcd/bg_pcd 字段）
            if use_mask and hasattr(frame, 'fg_pcd'):
                rr.log(
                    f"{name}/fg_point_cloud",
                    rr.Points3D(
                        positions=frame.fg_pcd,
                        colors=(frame.fg_pcd_color * 255).astype(np.uint8),
                    ),
                )
                rr.log(
                    f"{name}/bg_point_cloud",
                    rr.Points3D(
                        positions=frame.bg_pcd,
                        colors=(frame.bg_pcd_color * 255).astype(np.uint8),
                    ),
                )
            idx += 1

    else:
        # PKL / MonST3R 模式：迭代器每次 yield 一个 Pointmap 对象（含多帧）
        for (loader, name) in pointmap_iterator:
            num_frames = min(max_frames, loader.num_frames())

            for i in tqdm(range(0, num_frames, frame_gap)):
                frame    = loader.get_frame(i)
                position = frame.pcd        # (H*W, 3)
                color    = frame.pcd_color  # (H*W, 3)
                rgb      = frame.rgb        # (H, W, 3)
                H, W, _  = rgb.shape

                # 空间降采样（降低 .rrd 文件体积，同时加速 Viewer 渲染）
                position = position.reshape(H, W, 3)[::downsample_factor, ::downsample_factor].reshape([-1, 3])
                color    = color.reshape(H, W, 3)[::downsample_factor, ::downsample_factor].reshape([-1, 3])

                # 设置时间轴帧索引（Viewer 中的"时间"维度）
                rr.set_time_sequence("frame", i)

                rr.log(
                    f"{name}/point_cloud",
                    rr.Points3D(
                        positions=position,
                        colors=(color * 255).astype(np.uint8),
                    ),
                )

                rr.log(f"{name}/rgb_image", rr.Image(rgb))

                # 记录相机位姿（T_world_camera 来自 Pointmap，pm_registration 后为优化值）
                rr.log(
                    f"{name}/camera",
                    rr.Transform3D(
                        translation=frame.T_world_camera[:3, 3],
                        mat3x3=frame.T_world_camera[:3, :3],
                    ),
                )

                # log_image_plane_outline(rr, name, frame.K, frame.T_world_camera, H, W)
                # 上行已注释：若要显示视锥线框，取消注释并确认 frame.K 存在

                if use_mask and hasattr(frame, 'fg_pcd'):
                    rr.log(
                        f"{name}/fg_point_cloud",
                        rr.Points3D(
                            positions=frame.fg_pcd,
                            colors=(frame.fg_pcd_color * 255).astype(np.uint8),
                        ),
                    )
                    rr.log(
                        f"{name}/bg_point_cloud",
                        rr.Points3D(
                            positions=frame.bg_pcd,
                            colors=(frame.bg_pcd_color * 255).astype(np.uint8),
                        ),
                    )

            print(f"Finished logging {name}. Open rerun viewer to inspect.")

    # 将所有记录一次性写入 .rrd 文件
    # 打开方式：rerun <rr_recording>.rrd   或   rerun <rr_recording>.rrd --web-viewer
    rr.save(rr_recording)


if __name__ == "__main__":
    tyro.cli(main)
