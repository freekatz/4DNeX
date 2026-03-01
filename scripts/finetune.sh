#!/bin/bash
# =============================================================================
# 4DNeX LoRA 微调启动脚本
#
# 功能：在 8 张 GPU 上，基于 Wan2.1-I2V-14B-480P 预训练模型，
#       通过 LoRA 微调训练图像到4D动态点云生成能力。
#
# 依赖：
#   - Accelerate 分布式后端（配置见 configs_acc/8gpu.yaml）
#   - DeepSpeed ZeRO-2 显存优化（配置见 configs_zero/zero2.yaml）
#   - 预处理完成的训练数据（由 build_wan_dataset.py 生成）
#
# 使用方法：
#   bash scripts/finetune.sh
# =============================================================================

# 禁用 HuggingFace tokenizer 的多进程并行，避免多 worker DataLoader 环境下
# 因 fork 引起的死锁问题（tokenizers 库的已知限制）
export TOKENIZERS_PARALLELISM=false

# ── 分布式通信配置 ────────────────────────────────────────────────────────────
# 单机训练，主进程地址固定为 localhost
export MASTER_ADDR=localhost
# NCCL/Gloo 进程组通信端口，与其他任务冲突时可修改
export MASTER_PORT=29500
# 节点数（单机训练固定为 1；多机训练时需同步修改 machine_rank 及 MASTER_ADDR）
export NNODES=1
# 总 GPU 进程数，需与 configs_acc/8gpu.yaml 中的 num_processes 保持一致
export NUM_PROCESSES=8

# ── Accelerate 启动器配置 ─────────────────────────────────────────────────────
# 使用 `accelerate launch` 拉起分布式训练，底层调用 DeepSpeed ZeRO-2
# --config_file    : 指定 Accelerate 元配置（GPU 数量、DeepSpeed 策略、精度）
# --machine_rank   : 当前节点编号（单机训练恒为 0）
# --num_processes  : 总进程数，与 NUM_PROCESSES 一致
export LAUNCHER="accelerate launch \
    --config_file configs_acc/8gpu.yaml \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank 0 \
    --num_processes $NUM_PROCESSES \
    --num_machines $NNODES \
    "

# ── 训练程序参数配置 ───────────────────────────────────────────────────────────
#
# [模型]
#   --model_path        : Wan2.1 图像到视频基础模型路径（HuggingFace Diffusers 格式）
#   --model_name        : 模型变体标识
#                         demb      = 可学习域嵌入（Learnable Domain Embeddings）
#                         samerope  = 共享旋转位置编码（Shared RoPE）
#                         对应实现  : core/finetune/models/wan_i2v/demb_samerope_trainer.py
#   --model_type        : 任务类型；wan-i2v = 图像条件视频生成（Image-to-Video）
#   --training_type     : 微调策略；lora = 参数高效微调，可选 sft = 全参数微调
#
# [LoRA 超参]
#   --rank              : 低秩矩阵的秩 r = 64；越大表达能力越强但显存占用越高
#   --lora_alpha        : 缩放系数 α = 32；实际输出缩放比 = α/r = 0.5
#                         默认作用层：Transformer 注意力层 to_q/to_k/to_v/to_out.0
#
# [输出 & 监控]
#   --output_dir        : Checkpoint 及日志保存目录
#   --report_to         : 训练指标上报后端（可选 wandb / all）
#
# [数据]
#   --data_root         : 预处理数据根目录（由 build_wan_dataset.py 生成）
#   --caption_column    : 文本描述索引文件（相对于 data_root），每行一条 prompt
#   --video_column      : 视频/点云数据索引文件，每行一条样本路径
#
# [训练分辨率]
#   --train_resolution  : 81帧 × 480高 × 720宽
#                         帧数须满足 (frames-1) % 8 == 0，即 (81-1)=80 ✓
#                         对应 Wan2.1 480P 推荐输入尺寸
#
# [训练超参]
#   --train_epochs      : 训练总轮数
#   --seed              : 随机种子，保证实验可复现
#   --batch_size        : 单 GPU 的 batch size；14B 模型受显存限制通常设为 1
#   --gradient_accumulation_steps
#                       : 梯度累积步数；等效全局 batch = batch_size × num_gpu × 此值
#   --mixed_precision   : 混合精度训练；bf16 对大模型数值稳定性优于 fp16
#   --num_workers       : DataLoader 数据加载子进程数
#   --pin_memory        : 锁页内存，加速 CPU→GPU 数据传输
#   --nccl_timeout      : NCCL 集合通信超时（秒）；14B 模型初始化较慢，建议 ≥1800
#
# [Checkpoint]
#   --checkpointing_steps  : 每 500 步保存一次 checkpoint
#   --checkpointing_limit  : 最多保留最新的 2 个 checkpoint，自动清理旧版本
#
# [验证]（当前关闭，do_validation=false）
#   --do_validation        : 是否在训练期间执行推理验证；false 时以下参数不生效
#   --validation_dir       : 验证集数据目录（do_validation=true 时必须指定）
#   --validation_steps     : 每 N 步执行一次验证（须为 checkpointing_steps 的整数倍）
#   --validation_prompts   : 验证用 prompt 索引文件
#   --validation_images    : 验证用条件图像索引文件（i2v 任务必须指定）
#   --gen_fps              : 验证生成视频的帧率（fps）
export PROGRAM="\
finetune.py \
    --model_path ./pretrained/Wan2.1-I2V-14B-480P-Diffusers \
    --model_name wan-i2v-demb-samerope \
    --model_type wan-i2v \
    --training_type lora \
    --rank 64 \
    --lora_alpha 32 \
    --output_dir training/4dnex \
    --report_to tensorboard \
    --data_root ./data/wan21 \
    --caption_column prompts.txt \
    --video_column videos.txt \
    --train_resolution 81x480x720 \
    --train_epochs 10 \
    --seed 42 \
    --batch_size 1 \
    --gradient_accumulation_steps 1 \
    --mixed_precision bf16 \
    --num_workers 8 \
    --pin_memory True \
    --nccl_timeout 1800 \
    --checkpointing_steps 500 \
    --checkpointing_limit 2 \
    --do_validation false  \
    --validation_dir ./data/wan21 \
    --validation_steps 500 \
    --validation_prompts prompts_val.txt \
    --validation_images images.txt \
    --gen_fps 24 \
"

# ── 拼接并执行完整命令 ─────────────────────────────────────────────────────────
export CMD="$LAUNCHER $PROGRAM"

"$CMD"

echo "END TIME: $(date)"
