"""
pm_registration.py — 点云相机参数优化（后处理）

职责：
    对 inference.py 生成的原始 Pointmap（*.pkl）执行几何后处理：
    通过梯度下降优化一组相机内参（K）和各帧外参（R, t），
    使"将预测 XYZ 坐标反投影后再重投影"的误差最小化。
    优化结果替换原始 pcd，输出为 *_reg.pkl（registered）。

动机：
    Wan 模型直接生成像素级 XYZ 坐标，但生成的坐标序列不保证满足严格的
    针孔相机几何约束（各帧 XYZ 不一定与单一相机的投影模型自洽）。
    pm_registration 通过求解最优 K/R/t，使重投影结果尽量对应预测坐标，
    输出几何一致性更强的点云，更有利于后续新视角渲染和可视化。

后处理完整流水线：
    inference.py → *.pkl
        └─ pm_registration.py → *_reg.pkl
                └─ rerun_vis.py → *.rrd（可视化）

运行示例：
    python pm_registration.py --pkl_dir ./results
"""

import torch
import torch.nn.functional as F

import glob
import pickle
from pathlib import Path
import numpy as np
from tqdm.auto import tqdm
import imageio
from core.dataset import PointmapDataset
from core.annotation import Monst3RAnno
import tyro


# ─────────────────────────────────────────────────────────────────────────────
#  工具函数
# ─────────────────────────────────────────────────────────────────────────────

def rot6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    将"6D 连续旋转表示"转换为 SO(3) 旋转矩阵。

    基于 Zhou et al. 2019 "On the Continuity of Rotation Representations in Neural Networks"。
    相比欧拉角（万向节锁）、四元数（双覆盖），6D 表示在梯度优化时更稳定。

    算法（Gram-Schmidt 正交化）：
        a1, a2 = d6[:, 0:3],  d6[:, 3:6]   # 从 6D 向量提取两个原始列向量
        b1 = normalize(a1)                   # 第一列：直接单位化
        b2 = normalize(a2 - proj(a2, b1))   # 第二列：a2 去掉在 b1 方向的分量，再单位化
        b3 = cross(b1, b2)                   # 第三列：叉积保证右手系正交

    Args:
        d6 : (B, 6)  B 个 6D 旋转向量，每个元素是 nn.Parameter 可微分

    Returns:
        R  : (B, 3, 3)  合法的旋转矩阵
    """
    a1, a2 = d6[:, 0:3], d6[:, 3:6]
    b1 = F.normalize(a1, dim=1)
    # 正交化：去掉 a2 在 b1 方向的投影分量
    b2 = F.normalize(a2 - (a2 * b1).sum(-1, keepdim=True) * b1, dim=1)
    b3 = torch.cross(b1, b2, dim=1)
    return torch.stack([b1, b2, b3], dim=2)  # (B, 3, 3)，列向量为基


def build_K(fx, fy, cx, cy) -> torch.Tensor:
    """
    构造 3×3 针孔相机内参矩阵（共享于所有帧）。

    K = [[fx,  0, cx],
         [ 0, fy, cy],
         [ 0,  0,  1]]

    Args:
        fx, fy : 焦距（像素单位），可以是 nn.Parameter（可微）
        cx, cy : 主点（光心）坐标，通常初始化为 W/2, H/2

    Returns:
        K : (3, 3) tensor，在 fx.device 上
    """
    K = torch.zeros(3, 3, device=fx.device)
    K[0, 0], K[1, 1] = fx, fy
    K[0, 2], K[1, 2] = cx, cy
    K[2, 2] = 1.0
    return K


# ─────────────────────────────────────────────────────────────────────────────
#  主优化函数
# ─────────────────────────────────────────────────────────────────────────────

def optimise_xyz_batch(pred_xyz: torch.Tensor,
                       n_iters: int = 1500,
                       lr: float = 5e-3,
                       verbose: int = 100):
    """
    对 B 帧预测 XYZ 点云联合优化相机参数，使重投影误差最小化。

    优化目标：
        找到一组 K（共享内参）、R_i / t_i（各帧外参）使得：
            world_pts_i = R_i @ (K⁻¹ @ (depth_i * pixel_i)) + t_i
        尽量等于 pred_xyz_i（即生成模型的原始预测）。

        损失函数：L = mean_over_all_pixels_frames ||world_pts - pred_xyz||²

    可优化参数：
        depth   : (B, H*W)   各帧每像素深度（标量），初始化为 ||xyz||（预测向量的模长）
        fx, fy  : (scalar)   共享焦距，初始化为 1.2 倍图像维度（略宽于标准视角）
        cx, cy  : (scalar)   共享主点，初始化为图像中心
        rot6d   : (B, 6)     各帧旋转（6D 表示），初始化为单位阵
        trans   : (B, 3)     各帧平移，初始化为零向量

    Args:
        pred_xyz : (B, H, W, 3)  模型预测的 3D 坐标，值域通常在 [-1,1]
        n_iters  : 优化迭代次数（1500 次在 V100 上约需 10-30s）
        lr       : Adam 学习率
        verbose  : 每隔多少步打印一次 loss（0 表示不打印）

    Returns:
        depth_maps : (B, H, W)    优化后各帧像素深度
        K          : (3, 3)       优化后相机内参
        R          : (B, 3, 3)    优化后各帧旋转矩阵
        t          : (B, 3)       优化后各帧平移向量
    """
    device         = pred_xyz.device
    B, H, W, _     = pred_xyz.shape
    N              = H * W                         # 每帧像素数
    pred_xyz_flat  = pred_xyz.reshape(B, N, 3)     # (B, N, 3)

    # ── 构建像素坐标网格（一次性，不参与梯度）───────────────────────────────
    # pix_h[b, n] = [x_n, y_n, 1]，对所有帧共享
    ys, xs = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    pix_h  = torch.stack([xs, ys, torch.ones_like(xs)], dim=-1) \
                   .reshape(1, N, 3).to(device)    # (1, N, 3)

    # ── 初始化可优化参数 ──────────────────────────────────────────────────────
    # 深度：用预测 xyz 向量的模长初始化（保持尺度量级），允许轻微负值被 clamp
    depth = torch.nn.Parameter(pred_xyz_flat.norm(dim=-1))   # (B, N)

    # 共享内参：焦距初始化略大于标准视角（1.2x），主点为图像中心
    fx = torch.nn.Parameter(torch.tensor(W * 1.2, device=device))
    fy = torch.nn.Parameter(torch.tensor(H * 1.2, device=device))
    cx = torch.nn.Parameter(torch.tensor(W / 2,   device=device))
    cy = torch.nn.Parameter(torch.tensor(H / 2,   device=device))

    # 各帧外参：旋转初始化为单位矩阵（6D 表示），平移初始化为零
    rot6d = torch.nn.Parameter(
        torch.eye(3, device=device).repeat(B, 1, 1)[:, :, :2].reshape(B, 6)
    )
    trans = torch.nn.Parameter(torch.zeros(B, 3, device=device))

    optimiser = torch.optim.Adam([depth, fx, fy, cx, cy, rot6d, trans], lr=lr)

    # ── 优化主循环 ────────────────────────────────────────────────────────────
    for it in range(0, n_iters):
        optimiser.zero_grad()

        # 避免深度为负/零（保持 clamp 轻缓以让梯度流过）
        d = depth.clamp(min=1e-4).unsqueeze(-1)    # (B, N, 1)

        K    = build_K(fx, fy, cx, cy)             # (3, 3)
        Kinv = torch.inverse(K).unsqueeze(0)       # (1, 3, 3)

        # 反投影：像素坐标 × 深度 × K⁻¹ → 相机坐标系
        # d * pix_h: (B, N, 3)，乘深度后再乘 K⁻¹
        cam_pts = torch.matmul(
            Kinv,
            (d * pix_h).permute(0, 2, 1)          # (B, N, 3) → (B, 3, N)
        )                                           # → (B, 3, N)

        # 构建各帧的 cam→world 4×4 变换矩阵：[R|t; 0 0 0 1]
        R   = rot6d_to_matrix(rot6d)               # (B, 3, 3)
        t   = trans.unsqueeze(-1)                  # (B, 3, 1)
        ones = torch.tensor([0, 0, 0, 1], device=device).view(1, 1, 4).repeat(B, 1, 1)
        P_cam2world = torch.cat([torch.cat([R, t], dim=2), ones], dim=1)  # (B, 4, 4)

        # 相机坐标扩展齐次 → 变换到世界坐标
        cam_pts_h = torch.cat([cam_pts, torch.ones(B, 1, N, device=device)], dim=1)
        world_pts = (P_cam2world @ cam_pts_h)[:, :3]     # (B, 3, N)
        world_pts = world_pts.permute(0, 2, 1)            # (B, N, 3)

        # 重投影误差：世界坐标与预测 XYZ 的逐元素 L2 距离
        loss = (world_pts - pred_xyz_flat).pow(2).mean()
        loss.backward()
        optimiser.step()

        if verbose and it % verbose == 0:
            print(f"[{it:4d}/{n_iters}]  reprojection L2 = {loss.item():.6f}")

    depth_maps = depth.clamp(min=1e-4).detach().reshape(B, H, W)
    K_final    = build_K(fx, fy, cx, cy).detach()
    return depth_maps, K_final, R.detach(), trans.detach()


def depth_to_3d_points(depth_map: torch.Tensor,
                       K: torch.Tensor,
                       R: torch.Tensor,
                       t: torch.Tensor) -> np.ndarray:
    """
    将单帧深度图通过相机参数反投影为世界坐标系下的 3D 点。

    公式：
        P_cam   = K⁻¹ @ (depth * [x, y, 1])    （逐像素）
        P_world = R @ P_cam + t

    Args:
        depth_map : (H, W)    像素深度值
        K         : (3, 3)    相机内参（由 optimise_xyz_batch 输出）
        R         : (3, 3)    旋转矩阵（本帧，来自 optimise_xyz_batch 输出的 R[i]）
        t         : (3,)      平移向量（本帧）

    Returns:
        (H*W, 3)  世界坐标系下的 3D 点，numpy 格式
    """
    H, W   = depth_map.shape
    device = depth_map.device

    ys, xs = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    pix_h  = torch.stack([xs, ys, torch.ones_like(xs)], dim=-1).to(device)  # (H, W, 3)

    # 展平像素，统一做矩阵运算
    d       = depth_map.unsqueeze(-1)             # (H, W, 1)
    cam_pts = torch.matmul(
        torch.inverse(K),
        (d * pix_h).reshape(-1, 3).T              # (3, N)
    ).T                                            # (N, 3)

    # 变换到世界坐标系
    world_pts = (R @ cam_pts.T).T + t             # (N, 3)

    return world_pts.cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
#  入口
# ─────────────────────────────────────────────────────────────────────────────

def main(
    downsample_factor: int = 1,   # 空间降采样（预留，当前优化阶段未使用）
    frame_gap: int = 1,           # 帧间隔（优化时处理每帧，反投影时可按 gap 采样）
    max_frames: int = 100,        # 最多处理帧数
    use_mask: bool = False,       # 是否区分前景/背景（预留参数）
    pkl_dir: str = None,          # 推理输出目录，含 *.pkl 文件（inference.py 的输出）
    monst3r_dir: str = None,      # MonST3R 重建目录（GT 数据验证用）
    pointmap_mp4: str = None,     # 可视化点云 .mp4 路径（build_wan_dataset 的输出）
    rgb_mp4: str = None,          # 对应 RGB 视频路径
    pointmap_npy: str = None,     # pointmap numpy 数组路径
) -> None:
    """
    数据加载调度 + 批量相机优化。

    数据来源优先级（按条件逐一判断）：
        1. pointmap_npy + rgb_mp4   : NumPy 格式点云 + 视频
        2. pointmap_mp4 + rgb_mp4   : 可视化 MP4 格式（build_wan_dataset 输出）
        3. pkl_dir                  : inference.py 生成的 *.pkl（最常用路径）
        4. monst3r_dir              : MonST3R 标注目录（GT/评估用）

    输出：
        *_reg.pkl —— 替换了 pcd 字段的 Pointmap 对象，由 rerun_vis.py 加载可视化
    """
    # ── 数据加载调度 ──────────────────────────────────────────────────────────
    if pointmap_npy is not None and rgb_mp4 is not None:
        # NumPy 格式：直接从 .npy 文件加载点云
        pointmap   = np.load(pointmap_npy)    # [F, H, W, 3]
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
                    'T_world_camera': np.eye(4),  # 无已知位姿时用单位矩阵占位
                }), "npy_frame"

    elif pointmap_mp4 is not None and rgb_mp4 is not None:
        # MP4 格式：从训练数据可视化视频中加载
        # 注意：读取时已归一化到 [0,1]；优化循环中会减去 0.5 做零中心化
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
        # PKL 格式（最常用）：inference.py 输出的 Pointmap 对象
        pkl_list = glob.glob(f'{pkl_dir}/*.pkl')

        def PointmapIterator():
            for pkl_path in sorted(pkl_list):
                yield pickle.load(open(pkl_path, 'rb')), pkl_path

    elif monst3r_dir is not None:
        # MonST3R 目录：用于与 GT 标注对比验证优化效果
        monst3r_list = glob.glob(f'{monst3r_dir}/*/')

        def PointmapIterator():
            for scene_path in monst3r_list:
                yield Monst3RAnno(scene_path, max_frames=max_frames).pointmap, scene_path

    else:
        raise NotImplementedError("No data source provided")

    # ── 执行优化 ──────────────────────────────────────────────────────────────
    pointmap_iterator = PointmapIterator()

    if rgb_mp4 is not None:
        raise NotImplementedError("RGB MP4 is not supported yet")
    else:
        for (loader, name) in pointmap_iterator:
            num_frames = min(max_frames, loader.num_frames())

            # 1. 读取所有帧的 XYZ 坐标，堆叠为 [B, H, W, 3]
            pointmap_list = []
            for i in range(num_frames):
                frame   = loader.get_frame(i)
                H, W, _ = frame.rgb.shape
                pointmap_list.append(frame.pcd.reshape(H, W, 3))
            pointmap_list = np.stack(pointmap_list, axis=0)  # (B, H, W, 3)

            # 2. XY 坐标零中心化（[0,1] → [-0.5, 0.5]）
            #    减少偏置方向的梯度，有助于优化器更快收敛
            pointmap_list[..., :2] = pointmap_list[..., :2] - 0.5

            pointmap_list = torch.from_numpy(pointmap_list).cuda()

            # 3. 联合优化 K 和各帧 R、t（1500 步 Adam）
            depth_map, K, R, t = optimise_xyz_batch(pointmap_list)

            # 4. 用优化后参数重新反投影，替换原始 pcd
            updated_pointmap_list = []
            for i in tqdm(range(0, num_frames, frame_gap)):
                position = depth_to_3d_points(depth_map[i], K, R[i], t[i])
                updated_pointmap_list.append(position)
            updated_pointmap = np.stack(updated_pointmap_list, axis=0)  # (B, H*W, 3)

            loader.pcd = updated_pointmap  # 原地替换 Pointmap 对象的点云字段

            # 5. 保存 *_reg.pkl（区别于未注册的 *.pkl，供 rerun_vis.py 专门读取）
            save_path = name.replace('.pkl', '_reg.pkl')
            pickle.dump(loader, open(save_path, 'wb'))


if __name__ == "__main__":
    tyro.cli(main)
