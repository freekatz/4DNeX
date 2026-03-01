#!/bin/bash

# One4D Training Script - Single GPU with DeepSpeed ZeRO-2 Offload
# Uses optimizer CPU offload to reduce GPU memory usage

export TOKENIZERS_PARALLELISM=false

export LAUNCHER="accelerate launch \
    --config_file configs_acc/1gpu.yaml \
    --num_processes 1 \
    --num_machines 1 \
    "

export PROGRAM="\
finetune.py \
    --model_path ./pretrained/Wan2.1-I2V-14B-480P-Diffusers \
    --model_name wan-i2v-one4d \
    --model_type wan-i2v \
    --training_type lora \
    --rank 64 \
    --lora_alpha 32 \
    --output_dir training/one4d \
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
    --do_validation false \
"

export CMD="$LAUNCHER $PROGRAM"

$CMD

echo "END TIME: $(date)"
