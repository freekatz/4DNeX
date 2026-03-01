"""
inference.py — 4DNeX 推理入口

职责：
    从单张图像 + 文本 prompt 生成 4D 动态点云序列，输出：
      - <id>.mp4      : XYZ+RGB 拼接视频（可视化用）
      - <id>.pkl      : Pointmap 对象（用于 pm_registration.py / rerun_vis.py）

调用链：
    python inference.py --prompt prompts.txt --image images.txt --out ./results ...
        └─ main()
             └─ generate_video()          # core/inference/wan.py
                    └─ VAE decode ─→ save_pointmap()
                            └─ pickle.dump(Pointmap)

6D 统一表示（xyzrgb 模式）：
    模型输出宽度为 720*2=1440 的"双宽"视频潜变量：
      左半部分（像素列 0~719）  → RGB 视频帧
      右半部分（像素列 720~1439）→ XYZ 点云坐标（归一化到 [-1,1]）
    两路数据共享同一个 Wan VAE 编解码，但归一化统计量不同（见 save_pointmap）。
"""

import argparse
import os
import torch
import numpy as np
import imageio
import pickle
from core.inference.wan import generate_video
from core.dataclass import Pointmap
from core.tokenizer.wan import WanTokenizer

# ── 模块级全局 tokenizer ──────────────────────────────────────────────────────
# 在模块导入时即加载 VAE，避免每次调用 save_pointmap 都重新加载。
# 注意：此路径为硬编码，若模型存放位置不同需同步修改。
model_path = 'pretrained/Wan2.1-I2V-14B-480P-Diffusers/vae/'
tokenizer = WanTokenizer(model_path=model_path)


def save_pointmap(latents, save_path, image_path=None, mode='xyz'):
    """
    将 generate_video 返回的原始潜变量解码并保存为 Pointmap。

    Args:
        latents   : generate_video 返回的潜变量，形状 [C, T, H_lat, W_lat]
                    （无批次维度，W_lat 对应像素宽 1440 的 VAE 压缩结果）
        save_path : 输出路径，扩展名必须是 .pkl；同时自动保存同名 .mp4
        image_path: 仅在 mode='xyz' 时使用，用于从原始图像读取 RGB 颜色
        mode      : 'xyz'    — 仅输出 XYZ 点云坐标，颜色从 image_path 读取
                    'xyzrgb' — 从双宽潜变量中同时解码 RGB 和 XYZ

    归一化说明：
        生成模型对 XYZ 和 RGB 两路信号使用不同的潜变量统计量。
        - RGB 部分：在 generate_video 内部已经过标准 Wan VAE 归一化，无需额外处理。
        - XYZ 部分（潜变量右半 W/2 列）：使用点云专用统计量（均值 -0.13，标准差 1.70）
          进行反归一化，公式为：latent_xyz = latent_xyz * std + mean
        此处仅对右半部分执行这一步，左半（RGB）保持原样传入 VAE decode。
    """
    # 补充批次维度：[C, T, H_lat, W_lat] → [1, C, T, H_lat, W_lat]
    latents = latents[None]

    # XYZ 专用反归一化统计量（从训练数据中统计得出）
    encoded_pm_mean = -0.13
    encoded_pm_std  = 1.70

    # 仅对潜变量的右半（XYZ 对应的 W/2 列）做反归一化
    # latents.shape[4] 是潜变量宽度，对应原始像素宽 1440 的 1/8 = 180
    # 右半 = 后 90 列 = 解码后的右 720 像素 = XYZ 通道
    latents[:, :, :, :, latents.shape[4]//2:] = (
        latents[:, :, :, :, latents.shape[4]//2:] * encoded_pm_std + encoded_pm_mean
    )

    # VAE 解码：latents [1, C, T, H_lat, W_lat] → pointmap [T, H, W*2, 3]，值域 [0,1]
    pointmap = tokenizer.decode(latents)

    # ── 保存可视化 .mp4 ────────────────────────────────────────────────────────
    if '.mp4' in save_path:
        mp4_save_path = save_path
    else:
        _, ext = os.path.splitext(save_path)
        mp4_save_path = save_path.replace(ext, ".mp4")
    # pointmap 值域 [0,1]，乘以 255 后写视频
    imageio.mimwrite(mp4_save_path, (pointmap * 255).clip(0, 255).astype(np.uint8), fps=24)

    # ── 构造 Pointmap 对象并保存 .pkl ─────────────────────────────────────────
    pm = Pointmap()

    if mode == 'xyzrgb':
        # 双宽帧：W 是全宽（1440），左半 [0:720] = RGB，右半 [720:] = XYZ
        W = pointmap.shape[2] // 2
        rgb      = pointmap[..., :W, :]   # [T, H, 720, 3]，RGB 颜色，值域 [0,1]
        pointmap = pointmap[..., W:, :]   # [T, H, 720, 3]，XYZ 坐标，已反归一化

    # 用实际帧数和空间分辨率初始化 Pointmap 的占位数组
    pm.init_dummy(pointmap.shape[0], pointmap.shape[1], pointmap.shape[2])

    # 将 [T, H, W, 3] 展平为 [T, H*W, 3]（Pointmap.pcd 的标准格式）
    pointmap = pointmap.reshape(*pm.pcd.shape)
    pm.pcd = pointmap

    if mode == 'xyzrgb':
        # 将解码得到的 RGB 同步存入 pm.rgb（[T,H,W,3]）和 pm.colors（[T,H*W,3]）
        pm.rgb    = rgb.clip(min=0, max=1)
        pm.colors = pm.rgb.reshape(*pm.colors.shape)
    elif image_path is not None:
        # xyz 模式：从原始图像文件读取颜色，复制到所有时间步
        rgb       = imageio.imread(image_path) / 255.
        pm.rgb    = np.stack([rgb for _ in range(pm.rgb.shape[0])], 0)
        pm.colors = pm.rgb.reshape(*pm.colors.shape)

    pickle.dump(pm, open(save_path, 'wb'))


def main(args):
    """
    批量推理：从 prompt 列表和图像列表生成对应的 4D 点云序列。

    输入文件格式（每行一条路径/文本）：
        --prompt : prompts.txt，每行一条文本描述
        --image  : images.txt，每行一条图像路径
        二者行数必须相同，i 行 prompt 对应 i 行图像。

    输出（保存至 --out 目录）：
        <i>.mp4  : XYZ+RGB 双宽视频（仅供调试可视化）
        <i>.pkl  : Pointmap 对象，输入至 pm_registration.py 进行相机标定
    """
    prompt_list = []
    with open(args.prompt, 'r') as f:
        for line in f.readlines():
            prompt_list.append(line.strip())

    image_list = []
    with open(args.image, 'r') as f:
        for line in f.readlines():
            image_list.append(line.strip())

    assert len(prompt_list) == len(image_list)

    os.makedirs(args.out, exist_ok=True)

    for i in range(len(prompt_list)):
        prompt, image_path = prompt_list[i], image_list[i]

        # 在用户 prompt 末尾附加触发词 'POINTMAP_STYLE.'
        # 该词在训练时对所有样本统一添加，使模型学会将其与"生成点云"任务关联
        suffix = 'POINTMAP_STYLE.'
        prompt = prompt + ' ' + suffix

        output_path = os.path.join(args.out, f'{i:05d}.mp4')

        # --idx=-1 表示处理全部样本；否则仅处理指定索引，用于单样本调试
        if args.idx == -1 or i == args.idx:
            latent = generate_video(
                prompt=prompt,
                image_or_video_path=image_path,
                model_path='pretrained/Wan2.1-I2V-14B-480P-Diffusers',
                sft_path=args.sft_path,
                lora_path=args.lora_path,
                lora_rank=args.lora_rank,
                output_path=output_path,
                num_frames=49,              # 推理帧数（训练时为 81，推理可减少以节省时间）
                width=720 * 2,              # 双宽：左720=RGB，右720=XYZ（对应 xyzrgb 模式）
                height=480,
                generate_type=args.type,    # 模型推理分支，如 "i2vwbw-demb-samerope"
                num_inference_steps=50,
                guidance_scale=5.0,
                fps=24,
                num_videos_per_prompt=1,
                dtype=torch.bfloat16,
                seed=42,
                mode=args.mode              # 'xyz' 或 'xyzrgb'
            )
            # 将原始潜变量解码并以 .pkl 格式保存 Pointmap
            save_pointmap(latent, output_path.replace('.mp4', '.pkl'), image_path, args.mode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a video from a text prompt using Wan")
    parser.add_argument("--prompt",    type=str, required=True,          help="prompt 列表文件，每行一条")
    parser.add_argument("--image",     type=str, required=True,          help="图像列表文件，每行一条路径")
    parser.add_argument("--idx",       type=int, default=-1,             help="指定处理单条样本的索引；-1 表示全部")
    parser.add_argument("--sft_path",  type=str, default=None,           help="SFT（全参微调）权重路径")
    parser.add_argument("--out",       type=str, default="results/output", help="结果输出目录")
    parser.add_argument("--mode",      type=str, default="xyzrgb",       help="'xyz' 或 'xyzrgb'")
    parser.add_argument("--type",      type=str, default="condpm-i2dpm", help="推理分支类型，如 'i2vwbw-demb-samerope'")
    parser.add_argument("--lora_path", type=str, default=None,           help="LoRA 权重路径")
    parser.add_argument("--lora_rank", type=int, default=64,             help="LoRA 秩，需与训练时一致")
    args = parser.parse_args()
    main(args)
