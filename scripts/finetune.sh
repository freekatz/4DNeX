#!/bin/bash
# Training Script — auto-detects GPU count, supports specifying GPU IDs.
#
# Usage:
#   ./scripts/finetune.sh                           # all GPUs
#   ./scripts/finetune.sh --gpus 0,1                # GPU 0 and 1
#   ./scripts/finetune.sh --gpus 0                  # single GPU
#   ./scripts/finetune.sh --gpus 0,1 --zero offload # ZeRO-2 with CPU offload
#   ./scripts/finetune.sh --gpus 0 --zero offload   # single GPU + offload (low VRAM)

set -euo pipefail

# ---- Parse script args (before --) ----
GPUS=""
ZERO_MODE="auto"  # auto | offload | none

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)   GPUS="$2";     shift 2 ;;
        --zero)   ZERO_MODE="$2"; shift 2 ;;
        *)        break ;;  # remaining args passed to finetune.py
    esac
done

# ---- Detect GPUs ----
if [ -z "$GPUS" ]; then
    NUM_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 1)
    GPUS=$(seq -s, 0 $((NUM_GPUS - 1)))
else
    NUM_GPUS=$(echo "$GPUS" | tr ',' '\n' | wc -l | tr -d ' ')
fi

echo "[finetune] GPUs: $GPUS ($NUM_GPUS total), ZeRO mode: $ZERO_MODE"

# ---- Select DeepSpeed config ----
if [ "$ZERO_MODE" = "2" ]; then
    DS_CONFIG="configs/zero2.json"
elif [ "$ZERO_MODE" = "2_offload" ]; then
    DS_CONFIG="configs/zero2_offload.json"
elif [ "$ZERO_MODE" = "3" ]; then
    DS_CONFIG="configs/zero3.json"
elif [ "$ZERO_MODE" = "3_offload" ]; then
    DS_CONFIG="configs/zero3_offload.json"
elif [ "$ZERO_MODE" = "none" ]; then
    DS_CONFIG=""
else
    echo "Unknown --zero mode: $ZERO_MODE (use: 2, 2_offload, 3, 3_offload, none)"
    exit 1
fi

# ---- Generate accelerate config ----
ACCEL_CONFIG=$(mktemp /tmp/accel_config_XXXXXX.yaml)
trap "rm -f $ACCEL_CONFIG" EXIT

if [ -n "$DS_CONFIG" ]; then
    cat > "$ACCEL_CONFIG" <<EOF
compute_environment: LOCAL_MACHINE
gpu_ids: "$GPUS"
num_processes: $NUM_GPUS
debug: false
deepspeed_config:
  deepspeed_config_file: $DS_CONFIG
  zero3_init_flag: false
  deepspeed_multinode_launcher: standard
distributed_type: DEEPSPEED
downcast_bf16: "no"
enable_cpu_affinity: false
machine_rank: 0
main_training_function: main
num_machines: 1
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOF
else
    cat > "$ACCEL_CONFIG" <<EOF
compute_environment: LOCAL_MACHINE
gpu_ids: "$GPUS"
num_processes: $NUM_GPUS
debug: false
distributed_type: "NO"
downcast_bf16: "no"
enable_cpu_affinity: false
machine_rank: 0
main_training_function: main
num_machines: 1
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOF
fi

# ---- Launch ----
export TOKENIZERS_PARALLELISM=false

OUTPUT_DIR="./training"
RUN_TS=$(date +"%Y%m%d_%H%M%S")
# Shared run timestamp for all distributed workers.
export FINETRAINER_RUN_TS="$RUN_TS"

mkdir -p "$OUTPUT_DIR"

echo "[finetune] Persistent log: <output_dir>/finetune.log"

accelerate launch \
    --config_file "$ACCEL_CONFIG" \
    --num_processes "$NUM_GPUS" \
    finetune.py \
    --model_path ./pretrained/Wan2.1-I2V-14B-480P-Diffusers \
    --output_dir "$OUTPUT_DIR" \
    --report_to all \
    --rank 64 \
    --lora_alpha 32 \
    --zcl_layers 3,11,19,27,35 \
    --data_root ./data \
    --train_resolution 81x480x720 \
    --train_epochs 10 \
    --seed 42 \
    --batch_size 1 \
    --gradient_accumulation_steps 1 \
    --mixed_precision bf16 \
    --num_workers 8 \
    --pin_memory True \
    --nccl_timeout 1800 \
    --checkpointing_steps 200 \
    --checkpointing_limit 2 \
    --do_validation false \
    "$@"

echo "END TIME: $(date)"
