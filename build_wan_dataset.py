"""
build_wan_dataset.py — 训练数据预处理流水线

职责：
    将原始 4D 标注数据（MonST3R 格式的 RGB 视频 + 深度/点云 + 相机参数）
    转换为 Wan2.1 训练所需的标准化格式：
      - videos/          : 分辨率对齐后的 RGB 视频 (.mp4)
      - first_frames/    : 第一帧图像 (.png)，用作 I2V 条件
      - pointmap/        : 归一化后的 XYZ 点云视频 (.mp4)，供调试可视化
      - pointmap_latents/: 点云 VAE 潜变量 (.pt)，训练时直接加载
      - cache/
        ├─ video_latent/ : RGB 视频 VAE 潜变量 + CLIP 图像嵌入 (.safetensors)
        └─ prompt_embeddings/ : UMT5 文本嵌入 (.safetensors)，以 prompt hash 为文件名
      - prompts.txt      : caption 列表，供 finetune.py --caption_column 使用
      - videos.txt       : 相对路径列表，供 finetune.py --video_column 使用

并行预处理：
    通过 --num_tasks / --task_idx 将 datalist 均匀切分，支持多进程/多节点并行。

数据源目录结构（--data_dir 下）：
    dynamic/   — 动态场景，MonST3R 格式（含 .mp4 视频 + .npz 点云/相机文件）
    static/    — 静态场景，仅含 .npz 文件

运行示例：
    python build_wan_dataset.py \\
        --data_dir ./data \\
        --out ./data/wan21 \\
        --model_path ./pretrained/Wan2.1-I2V-14B-480P-Diffusers
"""

import os
import hashlib
import numpy as np
import imageio
import torch
from pathlib import Path
import torchvision.transforms.functional as F
import torch.nn.functional as nnf
import argparse
from core.dataset import PointmapDataset
from transformers import AutoTokenizer, UMT5EncoderModel, CLIPVisionModel, CLIPImageProcessor
from diffusers import AutoencoderKLWan
from safetensors.torch import save_file
import PIL


def transform_pointclouds_to_first_camera(pointclouds, cams2world):
    """
    将所有帧的点云从世界坐标系变换到第一帧相机坐标系。

    动机：Wan 模型生成的是以视频起始帧为参考的相对运动，
    将点云归一化到第一帧相机坐标系后，网络只需学习相对变化，
    不需要学习绝对的世界坐标，降低了泛化难度。

    Args:
        pointclouds : [F, N, 3]  世界坐标系下的点云（F 帧，N 个点）
        cams2world  : [F, 4, 4]  各帧相机到世界的变换矩阵

    Returns:
        [F, N, 3]  第一帧相机坐标系下的点云

    变换公式：
        world2first = inv(cams2world[0])
        new_pts = world2first @ [pts; 1]    （齐次坐标乘法）
    """
    # 第一帧相机到世界的变换取逆 → 世界到第一帧相机
    world2first = np.linalg.inv(cams2world[0])  # [4, 4]

    # 扩展为齐次坐标：[F, N, 3] → [F, N, 4]
    F, N, _ = pointclouds.shape
    ones = np.ones((F, N, 1), dtype=pointclouds.dtype)
    points_homo = np.concatenate([pointclouds, ones], axis=-1)  # [F, N, 4]

    # 批量矩阵乘：world2first(4×4) 作用于每帧每点
    # einsum 'ij,fkj->fki'：对每帧 f、每点 k，计算 world2first @ point
    points_transformed = np.einsum('ij,fkj->fki', world2first, points_homo)  # [F, N, 4]

    # 去掉齐次坐标的第 4 维
    return points_transformed[..., :3]


def split_list(datalist, num_segments):
    """
    将 datalist 均匀切分为 num_segments 段，用于多任务并行预处理。

    余数均匀分配给前几段（如 10 条数据分 3 段 → [4, 3, 3]），
    保证各段负载尽量均衡。

    Args:
        datalist     : 待切分的列表
        num_segments : 切分段数，对应 --num_tasks

    Returns:
        List[List]，长度为 num_segments
    """
    n = len(datalist)
    segment_size = n // num_segments
    remainder = n % num_segments  # 多余元素平均分配给前几段

    segments = []
    start = 0
    for i in range(num_segments):
        extra = 1 if i < remainder else 0
        end = start + segment_size + extra
        segments.append(datalist[start:end])
        start = end

    return segments


def encode_video(video, vae):
    """
    使用 Wan VAE 将 RGB 视频编码为潜变量。

    Args:
        video : [B, F, H, W, C]，值域必须在 [-1, 1]（断言检查）
        vae   : AutoencoderKLWan 实例

    Returns:
        latent : [B, z_dim, F', H', W']，标准化后的潜变量
                 F'=F/4（时间压缩），H'=H/8，W'=W/8（空间压缩）

    标准化公式（与训练时 sft_trainer.py 保持一致）：
        latent_scaled = (latent.sample() - latents_mean) / latents_std
        其中 latents_mean/std 来自 vae.config，是 Wan 预训练 VAE 的统计量。
    """
    assert torch.max(torch.abs(video)) <= 1.05, (
        "Input video values must be approximately in range [-1, 1]. Now it is "
        + str(torch.max(torch.abs(video)))
    )

    video = video.to(vae.device, dtype=vae.dtype)
    # BFHWC → BCFHW（VAE 期望的通道格式）
    video = video.permute(0, 4, 1, 2, 3)

    with torch.no_grad():
        latent_dist = vae.encode(video).latent_dist
        latent = latent_dist.sample()  # 从潜变量分布中采样一个点

        # 从 VAE config 中读取训练时的均值/方差，做标准化
        # view(1, z_dim, 1, 1, 1) 使其可以广播到 [B, z_dim, F', H', W']
        latents_mean = torch.tensor(vae.config.latents_mean).view(
            1, vae.config.z_dim, 1, 1, 1
        ).to(latent.device, latent.dtype)
        latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(
            1, vae.config.z_dim, 1, 1, 1
        ).to(latent.device, latent.dtype)

        latent = (latent - latents_mean) * latents_std

    return latent


def encode_text(prompt, tokenizer, text_encoder, max_text_seq_length, device):
    """
    使用 UMT5 编码文本，输出固定长度的嵌入序列。

    Args:
        prompt             : 单条文本字符串
        tokenizer          : UMT5 tokenizer
        text_encoder       : UMT5EncoderModel
        max_text_seq_length: 目标序列长度（Wan 中为 512）
        device             : 编码设备

    Returns:
        [1, max_text_seq_length, hidden_dim]  文本嵌入

    填充策略（动态 trim + 零填充）：
        1. 先 padding 到 max_length 做 token id 截断
        2. 计算每条文本的真实有效长度 seq_len（mask.sum）
        3. 取前 seq_len 个 token 的 hidden state，其余补零
        这种方式使有效 token 的嵌入向量不受 PAD token 影响。
    """
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_text_seq_length,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
    # 统计每条文本真正的 token 长度（mask=1 的位置数）
    seq_lens = mask.gt(0).sum(dim=1).long()

    prompt_embeds = text_encoder(
        text_input_ids.to(device),
        mask.to(device)
    ).last_hidden_state  # [B, max_length, hidden_dim]

    # 对每条文本：保留前 seq_len 个有效 token 嵌入，其余位置填零
    prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
    prompt_embeds = torch.stack(
        [torch.cat([u, u.new_zeros(max_text_seq_length - u.size(0), u.size(1))]) for u in prompt_embeds],
        dim=0
    )

    return prompt_embeds


def encode_image(image, image_processor, image_encoder, device):
    """
    使用 CLIP ViT 编码图像，用于 I2V 条件输入。

    Args:
        image          : PIL.Image
        image_processor: CLIPImageProcessor（负责 resize、归一化）
        image_encoder  : CLIPVisionModel

    Returns:
        [1, seq_len, 1024]  倒数第二层 hidden state

    为什么用 hidden_states[-2] 而非最后一层：
        CLIP 最后一层经过了有损的全局池化和对比学习投影，
        倒数第二层保留了更丰富的空间语义信息，
        被用作 Diffusion 模型 cross-attention 的视觉条件（通用做法，如 IP-Adapter）。
    """
    image = image_processor(images=image, return_tensors="pt").to(device)
    image_embeds = image_encoder(**image, output_hidden_states=True)
    return image_embeds.hidden_states[-2]  # 倒数第二层，[1, seq_len, 1024]


def main(args):
    """
    主预处理流程：遍历数据集，对每条样本执行以下步骤：
      1. 生成并缓存 UMT5 文本嵌入（以 SHA256(prompt) 为文件名，避免重复编码）
      2. Resize + Center Crop RGB 视频到目标分辨率（480×720），不足 81 帧则反向补帧
      3. 编码 RGB 视频为 VAE 潜变量 + 第一帧 CLIP 嵌入，合并保存为 .safetensors
      4. 将点云变换到第一帧相机坐标系 → 百分位截断 → 等比例归一化到 [-1,1]
      5. Resize 点云到目标分辨率 → 反向补帧 → VAE 编码为潜变量 (.pt)
      6. 写出 prompts.txt 和 videos.txt 索引文件
    """
    data_dir = Path(args.out)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(data_dir / "videos", exist_ok=True)
    os.makedirs(data_dir / "pointmap", exist_ok=True)
    os.makedirs(data_dir / "pointmap_latents", exist_ok=True)
    os.makedirs(data_dir / "first_frames", exist_ok=True)

    max_frames = 81  # Wan 时间压缩比为 4，81 帧 → 20 帧潜变量（(81-1)/4+1=21，满足 (N-1)%8==0）
    resolution_out  = (args.resolution_h, args.resolution_w)  # (480, 720)
    resolution_crop = resolution_out

    # ── 缓存目录结构 ──────────────────────────────────────────────────────────
    # 不同分辨率使用不同子目录，避免缓存污染
    cache_dir = data_dir / "cache"
    train_resolution_str  = f"{max_frames}x{args.resolution_h}x{args.resolution_w}"
    video_latent_dir      = cache_dir / "video_latent" / "wan-i2v" / train_resolution_str
    prompt_embeddings_dir = cache_dir / "prompt_embeddings"
    video_latent_dir.mkdir(parents=True, exist_ok=True)
    prompt_embeddings_dir.mkdir(parents=True, exist_ok=True)

    # ── 生成 datalist ─────────────────────────────────────────────────────────
    print(f"Generating datalist from directory: {args.data_dir}")
    pexel_datalist = generate_datalist_from_directory(args.data_dir)
    print(f"Found {len(pexel_datalist)} samples")

    # 保存 datalist 供复现/调试
    datalist_path = data_dir / "generated_datalist.txt"
    with open(datalist_path, 'w') as f:
        f.write('\n'.join(pexel_datalist))
    print(f"Saved generated datalist to {datalist_path}")

    # 按 task_idx 切分，支持多节点并行预处理（--num_tasks=N 启动 N 个进程，每个处理 1/N 数据）
    pexel_datalist_segment = split_list(pexel_datalist, num_segments=args.num_tasks)[args.task_idx]
    print(f'{args.task_idx}/{args.num_tasks} task | {len(pexel_datalist_segment)}/{len(pexel_datalist)} samples')

    # ── 初始化 Dataset 迭代器 ──────────────────────────────────────────────────
    cache_dir_tmp = f'.cache/{args.task_idx:05d}/'
    train_dataset_iterator = iter(
        PointmapDataset(
            datalist=pexel_datalist_segment,
            max_frames=max_frames,
            s3_conf_path='~/petreloss.conf',  # 对象存储配置（本地数据可忽略）
            debug=False,
            random_shuffle=False,
            cache_dir=cache_dir_tmp,
            skip_invalid=False
        )
    )

    # ── 加载编码模型 ──────────────────────────────────────────────────────────
    model_path = args.model_path

    # 1. Wan VAE：用于视频和点云的时空压缩
    vae = AutoencoderKLWan.from_pretrained(model_path, subfolder="vae")
    vae.to(args.device)

    # 2. UMT5 文本编码器：编码 prompt 为 512 维嵌入序列
    text_tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer")
    text_encoder   = UMT5EncoderModel.from_pretrained(model_path, subfolder="text_encoder")
    text_encoder.to(args.device)

    # 3. CLIP 图像编码器：编码第一帧图像为视觉嵌入（I2V 条件）
    image_processor = CLIPImageProcessor.from_pretrained(model_path, subfolder="image_processor")
    image_encoder   = CLIPVisionModel.from_pretrained(model_path, subfolder="image_encoder")
    image_encoder.to(args.device)

    # UMT5 最大文本序列长度（来自 Wan 官方配置）
    transformer_config = {"max_text_seq_length": 512}

    prompt_list = []
    data_list   = []
    center_list = []
    scale_list  = []

    for i in range(len(pexel_datalist_segment)):
        data = next(train_dataset_iterator)
        if data is None:
            continue

        # ── 组合 prompt ───────────────────────────────────────────────────────
        caption       = data.video.caption
        prompt_suffix = 'POINTMAP_STYLE.'  # 触发词，使模型在推理时能识别 4D 生成任务
        full_prompt   = caption + ' ' + prompt_suffix
        prompt_list.append(caption)  # 保存不含触发词的原始 caption

        # ── 生成视频文件名（含时间片段信息）─────────────────────────────────
        if 'dynamic' in data.video.path:
            video_name = os.path.basename(data.video.path).replace('.mp4', '')
        else:
            video_name = data.video.path.split('/')[-2]
        if data.clip_start > -1:
            # 同一视频有多个时间片段时，以 start-end 区分
            video_name = video_name + f'-{data.clip_start:04d}-{data.clip_start+data.length:04d}'

        # ── 定义输出路径 ──────────────────────────────────────────────────────
        video_path           = data_dir / "videos"           / f"{video_name}.mp4"
        pm_path              = data_dir / "pointmap"         / f"{video_name}.mp4"
        pointmap_latent_path = data_dir / "pointmap_latents" / f"{video_name}.pt"
        data_list.append(f'videos/{video_name}.mp4')

        # ── 文本嵌入缓存（SHA256 去重）────────────────────────────────────────
        # 同一 prompt 只编码一次，避免重复调用 UMT5（编码耗时较长）
        prompt_hash           = str(hashlib.sha256(full_prompt.encode()).hexdigest())
        prompt_embedding_path = prompt_embeddings_dir / (prompt_hash + ".safetensors")

        if not prompt_embedding_path.exists():
            prompt_embedding = encode_text(
                full_prompt,
                text_tokenizer,
                text_encoder,
                transformer_config["max_text_seq_length"],
                args.device
            )
            prompt_embedding = prompt_embedding[0].to("cpu")
            save_file({"prompt_embedding": prompt_embedding}, prompt_embedding_path)
            print(f"Saved prompt embedding to {prompt_embedding_path}")

        # ── RGB 视频预处理（Resize + Crop + 帧数补齐）────────────────────────
        if not video_path.exists():
            rgb = data.rgb_raw  # [T, H, W, C]，值域 [0,1]
            T, H, W, C = rgb.shape
            target_h, target_w = resolution_out

            # 等比缩放：先让短边对齐目标分辨率，再 center crop
            # 保持宽高比，避免拉伸失真
            if H / W < target_h / target_w:
                new_h = target_h
                new_w = int(W * (target_h / H))
            else:
                new_w = target_w
                new_h = int(H * (target_w / W))

            # THWC → TCHW（PyTorch 插值格式），完成后转回 THWC
            rgb = nnf.interpolate(
                torch.from_numpy(rgb.transpose(0, 3, 1, 2)),
                size=(new_h, new_w),
                mode='bilinear',
                align_corners=False,
                antialias=True
            ).numpy().transpose(0, 2, 3, 1)
            rgb = F.center_crop(
                torch.from_numpy(rgb.transpose(0, 3, 1, 2)),
                resolution_out
            ).numpy().transpose(0, 2, 3, 1)

            # 保存第一帧图像，用于 I2V 推理的条件输入
            first_frame = rgb[0]
            first_frame_path = data_dir / "first_frames" / f"{video_name}.png"
            imageio.imwrite(first_frame_path, (first_frame * 255).clip(0, 255).astype(np.uint8))
            print(f"Saved first frame to {first_frame_path}")

            # 视频帧数不足时，拼接反向帧（"弹跳" 扩充）以达到 81 帧
            # 此策略保持运动连续性，避免重复首帧造成的静止片段
            if rgb.shape[0] < max_frames:
                num_original = T
                num_needed   = max_frames - num_original
                reverse_rgb  = rgb[::-1]
                rgb = np.concatenate([rgb, reverse_rgb[:num_needed]], axis=0)

            imageio.mimwrite(video_path, (rgb * 255).clip(0, 255).astype(np.uint8), fps=24)
            print(f"Saved RGB video to {video_path}")
        else:
            print(f"RGB video already exists at {video_path}, skipping processing")
            rgb = None  # 标记为未加载，后续按需从磁盘读取

        # ── RGB VAE 潜变量 + CLIP 嵌入缓存 ───────────────────────────────────
        encoded_video_path = video_latent_dir / (video_name + ".safetensors")
        if not encoded_video_path.exists():
            if rgb is None:
                # 视频已存在但本轮未加载，用 decord 高效读取
                import decord
                decord.bridge.set_bridge("torch")
                video_reader = decord.VideoReader(uri=str(video_path))
                rgb = video_reader[:].float() / 255.0

                # 补存 first_frame（若之前跳过了 RGB 处理分支）
                first_frame_path = data_dir / "first_frames" / f"{video_name}.png"
                if not first_frame_path.exists():
                    first_frame = rgb[0].cpu().numpy().transpose(1, 2, 0)  # CHW → HWC
                    imageio.imwrite(first_frame_path, (first_frame * 255).clip(0, 255).astype(np.uint8))
                    print(f"Saved first frame to {first_frame_path}")

            # [T,H,W,C] numpy → [1,T,H,W,C] tensor，并归一化到 [-1,1]
            rgb_tensor = torch.from_numpy(rgb).float() if isinstance(rgb, np.ndarray) else rgb
            rgb_tensor = rgb_tensor * 2.0 - 1.0   # [0,1] → [-1,1]，VAE 期望此值域
            rgb_tensor = rgb_tensor.unsqueeze(0)   # 添加 batch 维度

            encoded_video = encode_video(rgb_tensor, vae)  # [1, z_dim, F', H', W']

            # 提取第一帧用于 CLIP 编码：反归一化 → uint8 → PIL
            first_frame = (rgb_tensor[:, 0] + 1) * 0.5   # [-1,1] → [0,1]
            first_frame = (first_frame[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            first_frame = PIL.Image.fromarray(first_frame, mode='RGB')
            image_embedding = encode_image(first_frame, image_processor, image_encoder, args.device)

            # 合并存储：训练时 WanI2VDataset 从同一文件加载两种嵌入
            encoded_video   = encoded_video[0].cpu()    # 去掉 batch 维度：[z_dim, F', H', W']
            image_embedding = image_embedding[0].cpu()  # [seq_len, 1024]
            save_file({
                "encoded_video":   encoded_video,
                "image_embedding": image_embedding
            }, encoded_video_path)
            print(f"Saved encoded video and image embedding to {encoded_video_path}")

        # ── 点云预处理 ────────────────────────────────────────────────────────
        if not pm_path.exists() or not pointmap_latent_path.exists():
            pointmap = data.pointmap.pcd  # [F, H*W, 3]，世界坐标系

            # 1. 坐标系归一化：转换到第一帧相机坐标系
            pointmap = transform_pointclouds_to_first_camera(pointmap, data.pointmap.cams2world)

            # 2. 百分位截断：去除离群点（最近 2% 和最远 2%），提升归一化稳定性
            pointmap = np.clip(
                pointmap,
                np.percentile(pointmap, 2,  axis=1, keepdims=True),
                np.percentile(pointmap, 98, axis=1, keepdims=True)
            )

            # 3. 各向同性 min-max 归一化到 [-1, 1]
            #    使用相同 scale 处理 X/Y/Z 三轴，保持几何形状不变形（等比缩放）
            pointmap_min = pointmap.min(axis=1, keepdims=True)
            pointmap_max = pointmap.max(axis=1, keepdims=True)
            center       = (pointmap_min + pointmap_max) / 2
            scale        = (pointmap_max - pointmap_min) / 2
            scale        = scale.max()          # 三轴取最大，保持各向同性
            pointmap_in  = (pointmap - center) / scale  # 值域 [-1, 1]

            center_list.append(center)
            scale_list.append(scale)

            # 4. reshape 为图像格式：[F, H*W, 3] → [F, H, W, 3]（rgb.shape）
            pointmap_in = pointmap_in.reshape(*data.pointmap.rgb.shape)

            # 5. 等比 Resize + Center Crop（与 RGB 保持一致的空间对齐）
            T, H, W, C = data.rgb_raw.shape
            target_h, target_w = resolution_out
            if H / W < target_h / target_w:
                new_h, new_w = target_h, int(W * (target_h / H))
            else:
                new_w, new_h = target_w, int(H * (target_w / W))

            # 点云使用 nearest 插值（避免 XYZ 坐标在相邻像素间插值产生无意义的混合值）
            pointmap_in = nnf.interpolate(
                torch.from_numpy(pointmap_in.transpose(0, 3, 1, 2)),
                size=(new_h, new_w),
                mode='nearest'
            ).numpy().transpose(0, 2, 3, 1)
            pointmap_in = F.center_crop(
                torch.from_numpy(pointmap_in.transpose(0, 3, 1, 2)),
                resolution_crop
            ).numpy().transpose(0, 2, 3, 1)

            # 6. 帧数不足时反向补帧（与 RGB 补帧策略一致）
            if pointmap_in.shape[0] < max_frames:
                num_needed       = max_frames - pointmap_in.shape[0]
                reverse_pointmap = pointmap_in[::-1]
                pointmap_in      = np.concatenate([pointmap_in, reverse_pointmap[:num_needed]], axis=0)

            # 7. 保存可视化视频：[-1,1] → [0,255]，(v+1)*127.5
            imageio.mimwrite(pm_path, ((pointmap_in + 1) * 127.5).clip(0, 255).astype(np.uint8), fps=24)

            # 8. VAE 编码点云：值域 [-1,1] 满足 encode_video 的断言要求
            pointmap_tensor = torch.from_numpy(pointmap_in).float()
            pointmap_tensor = pointmap_tensor.unsqueeze(0).to(args.device)  # [1, F, H, W, 3]
            pointmap_latents = encode_video(pointmap_tensor, vae)  # [1, z_dim, F', H', W']
            torch.save(pointmap_latents[0].cpu(), pointmap_latent_path)
            print(f"Saved pointmap video to {pm_path} and latents to {pointmap_latent_path}")
        else:
            print(f"Pointmap already exists at {pm_path}, skipping processing")
            if not center_list:
                # 尝试加载已保存的归一化参数（尽力恢复，不做强依赖）
                try:
                    center_path = data_dir / "center" / f"{args.task_idx:05d}.npy"
                    scale_path  = data_dir / "scale"  / f"{args.task_idx:05d}.npy"
                    if center_path.exists() and scale_path.exists():
                        centers     = np.load(center_path)
                        scales      = np.load(scale_path)
                        center_list = centers.tolist()
                        scale_list  = scales.tolist()
                except Exception as e:
                    print(f"Could not load center and scale: {e}")

        print(f'Finished processing sample {i+1}/{len(pexel_datalist_segment)}.')

    # ── 写出索引文件（供 finetune.py 的 --caption_column / --video_column 使用）──
    with open(data_dir / "prompts.txt", 'w') as f:
        f.write('\n'.join(prompt_list))

    with open(data_dir / "videos.txt", 'w') as f:
        f.write('\n'.join(data_list))

    print(f'All finished.')


def generate_datalist_from_directory(data_dir):
    """
    遍历数据源目录，生成样本路径列表。

    目录结构约定：
        data_dir/
          dynamic/   — 动态场景
            <scene>/   — 每个场景为一个子目录
              *.mp4    — 原始视频
              *.npz    — MonST3R 输出（深度、相机参数等），每帧一个
          static/    — 静态场景
            **/*.npz   — 直接包含重建结果

    Returns:
        List[str]：各样本的路径（dynamic 为目录路径+/，static 为 .npz 文件路径）
    """
    data_dir = Path(data_dir).absolute()
    datalist = []

    # ── 动态场景：以最深子目录（含视频或点云文件的目录）为单位 ─────────────
    dynamic_dir = data_dir / "dynamic"
    if dynamic_dir.exists():
        for root, dirs, files in os.walk(dynamic_dir):
            # 叶子目录（无子目录）或含视频文件的目录视为一个样本
            if not dirs or any(file.endswith('.mp4') for file in files):
                datalist.append(str(Path(root).absolute()) + '/')
            for file in files:
                if file.endswith('.npz'):  # 单帧重建结果也单独加入
                    datalist.append(str((Path(root) / file).absolute()))

    # ── 静态场景：所有 .npz 文件各自独立为一个样本 ──────────────────────────
    static_dir = data_dir / "static"
    if static_dir.exists():
        for root, dirs, files in os.walk(static_dir):
            for file in files:
                if file.endswith('.npz'):
                    datalist.append(str((Path(root) / file).absolute()))

    return datalist


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='build wan dataset')
    parser.add_argument('--data_dir',     type=str, default='./data',
                        help='原始数据根目录，需含 dynamic/ 和/或 static/ 子目录')
    parser.add_argument('--out',          type=str, default='./data/wan21',
                        help='预处理结果输出目录')
    parser.add_argument('--model_path',   type=str, default='./pretrained/Wan2.1-I2V-14B-480P-Diffusers',
                        help='Wan2.1 模型路径，用于加载 VAE / tokenizer / text_encoder / image_encoder')
    parser.add_argument('--num_tasks',    type=int, default=1,
                        help='并行任务总数，配合 --task_idx 实现多进程切分')
    parser.add_argument('--task_idx',     type=int, default=0,
                        help='当前任务编号（0-indexed），处理 datalist 的第 task_idx 段')
    parser.add_argument('--resolution_h', type=int, default=480,
                        help='目标视频高度（需与训练配置 --train_resolution 一致）')
    parser.add_argument('--resolution_w', type=int, default=720,
                        help='目标视频宽度（需与训练配置 --train_resolution 一致）')
    parser.add_argument('--device',       type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    # 与 finetune.sh 保持一致，避免 DataLoader 多进程 fork 时 tokenizer 死锁
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main(args)
