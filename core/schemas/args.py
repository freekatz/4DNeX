import argparse
import datetime
import os
from pathlib import Path
from typing import Any, List, Literal, Tuple

from pydantic import BaseModel, ValidationInfo, field_validator


class Args(BaseModel):
    ########## Model ##########
    model_path: Path
    training_type: Literal["lora", "sft"] = "lora"

    output_dir: Path | None = None
    report_to: Literal["tensorboard", "swanlab", "all"] | None = None
    tracker_name: str = "finetrainer"
    experiment_name: str | None = None

    ########## Data ###########
    data_root: Path
    index_file: Path | None = None

    ########## Training #########
    resume_from_checkpoint: Path | None = None

    seed: int | None = None
    train_epochs: int = 10
    train_steps: int | None = None
    checkpointing_steps: int = 200
    checkpointing_limit: int = 10

    batch_size: int = 1
    gradient_accumulation_steps: int = 1

    train_resolution: Tuple[int, int, int]  # (frames, height, width)

    mixed_precision: Literal["no", "fp16", "bf16"] = "bf16"

    learning_rate: float = 2e-5
    optimizer: str = "adamw"
    beta1: float = 0.9
    beta2: float = 0.95
    beta3: float = 0.98
    epsilon: float = 1e-8
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0

    lr_scheduler: str = "constant_with_warmup"
    lr_warmup_steps: int = 100
    lr_num_cycles: int = 1
    lr_power: float = 1.0

    num_workers: int = 8
    pin_memory: bool = True

    gradient_checkpointing: bool = True
    enable_slicing: bool = True
    enable_tiling: bool = True
    nccl_timeout: int = 1800

    ########## LoRA ##########
    rank: int = 64
    lora_alpha: int = 32
    target_modules: List[str] = ["to_q", "to_k", "to_v", "to_out.0", "ffn.net.0.proj", "ffn.net.2"]
    zcl_layers: List[int] = [3, 11, 19, 27, 35]

    ########## Validation ##########
    do_validation: bool = False
    validation_steps: int | None = None
    validation_dir: Path | None = None
    validation_prompts: str | None = None
    validation_images: str | None = None
    validation_videos: str | None = None
    gen_fps: int = 15

    @field_validator("validation_dir", "validation_prompts")
    def validate_validation_required_fields(cls, v: Any, info: ValidationInfo) -> Any:
        if info.data.get("do_validation") and not v:
            raise ValueError(f"{info.field_name} must be specified when do_validation is True")
        return v

    @field_validator("validation_images")
    def validate_validation_images(cls, v: str | None, info: ValidationInfo) -> str | None:
        if info.data.get("do_validation") and not v:
            raise ValueError("validation_images must be specified when do_validation is True")
        return v

    @field_validator("validation_steps")
    def validate_validation_steps(cls, v: int | None, info: ValidationInfo) -> int | None:
        values = info.data
        if values.get("do_validation"):
            if v is None:
                raise ValueError("validation_steps must be specified when do_validation is True")
            if values.get("checkpointing_steps") and v % values["checkpointing_steps"] != 0:
                raise ValueError("validation_steps must be a multiple of checkpointing_steps")
        return v

    @field_validator("train_resolution")
    def validate_train_resolution(cls, v: Tuple[int, int, int], info: ValidationInfo) -> Tuple[int, int, int]:
        frames, height, width = v
        if (frames - 1) % 8 != 0:
            raise ValueError(f"Number of frames - 1 must be a multiple of 8, got frames={frames}")
        return v

    @field_validator("zcl_layers")
    def validate_zcl_layers(cls, v: List[int], info: ValidationInfo) -> List[int]:
        if not v:
            raise ValueError("zcl_layers must not be empty")
        if any(layer < 0 for layer in v):
            raise ValueError(f"zcl_layers must be non-negative, got {v}")
        if any(layer >= 40 for layer in v):
            raise ValueError(f"zcl_layers must be in [0, 39] for Wan-40 blocks, got {v}")
        if len(set(v)) != len(v):
            raise ValueError(f"zcl_layers must not contain duplicates, got {v}")
        if v != sorted(v):
            raise ValueError(f"zcl_layers must be sorted in ascending order, got {v}")
        return v

    @classmethod
    def parse_args(cls):
        """Parse command line arguments and return Args instance."""
        p = argparse.ArgumentParser()

        # Model & output
        p.add_argument("--model_path", type=str, required=True)
        p.add_argument("--training_type", type=str, default="lora", choices=["lora", "sft"])
        p.add_argument("--output_dir", type=str, default="training")
        p.add_argument("--report_to", type=str, required=True)
        p.add_argument("--tracker_name", type=str, default="finetrainer")
        p.add_argument(
            "--experiment_name",
            type=str,
            default=None,
            help="Experiment name prefix; training will auto-append timestamp suffix",
        )

        # Data
        p.add_argument("--data_root", type=str, required=True)
        p.add_argument("--index_file", type=str, default=None)
        p.add_argument("--train_resolution", type=str, required=True)

        # Training
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--train_epochs", type=int, default=10)
        p.add_argument("--train_steps", type=int, default=None)
        p.add_argument("--gradient_accumulation_steps", type=int, default=1)
        p.add_argument("--batch_size", type=int, default=1)
        p.add_argument("--learning_rate", type=float, default=2e-5)
        p.add_argument("--optimizer", type=str, default="adamw")
        p.add_argument("--beta1", type=float, default=0.9)
        p.add_argument("--beta2", type=float, default=0.95)
        p.add_argument("--beta3", type=float, default=0.98)
        p.add_argument("--epsilon", type=float, default=1e-8)
        p.add_argument("--weight_decay", type=float, default=1e-4)
        p.add_argument("--max_grad_norm", type=float, default=1.0)
        p.add_argument("--mixed_precision", type=str, default="bf16")

        # LR scheduler
        p.add_argument("--lr_scheduler", type=str, default="constant_with_warmup")
        p.add_argument("--lr_warmup_steps", type=int, default=100)
        p.add_argument("--lr_num_cycles", type=int, default=1)
        p.add_argument("--lr_power", type=float, default=1.0)

        # Data loading & model config
        p.add_argument("--num_workers", type=int, default=8)
        p.add_argument("--pin_memory", type=bool, default=True)
        p.add_argument("--gradient_checkpointing", type=bool, default=True)
        p.add_argument("--enable_slicing", type=bool, default=True)
        p.add_argument("--enable_tiling", type=bool, default=True)
        p.add_argument("--nccl_timeout", type=int, default=1800)

        # LoRA
        p.add_argument("--rank", type=int, default=64)
        p.add_argument("--lora_alpha", type=int, default=32)
        p.add_argument(
            "--target_modules",
            type=str,
            nargs="+",
            default=["to_q", "to_k", "to_v", "to_out.0", "ffn.net.0.proj", "ffn.net.2"],
        )
        p.add_argument(
            "--zcl_layers",
            type=str,
            default="3,11,19,27,35",
            help="Comma-separated DiT block indices for inserting ZCL, e.g. 3,11,19,27,35",
        )

        # Checkpointing
        p.add_argument("--checkpointing_steps", type=int, default=200)
        p.add_argument("--checkpointing_limit", type=int, default=3)
        p.add_argument("--resume_from_checkpoint", type=str, default=None)

        # Validation
        p.add_argument("--do_validation", type=lambda x: x.lower() == "true", default=False)
        p.add_argument("--validation_steps", type=int, default=None)
        p.add_argument("--validation_dir", type=str, default=None)
        p.add_argument("--validation_prompts", type=str, default=None)
        p.add_argument("--validation_images", type=str, default=None)
        p.add_argument("--validation_videos", type=str, default=None)
        p.add_argument("--gen_fps", type=int, default=15)

        args = p.parse_args()

        # Always generate a timestamped experiment name for unique run identity.
        # Examples:
        #   my-exp -> my-exp-20260310_153000
        #   (missing) -> default-20260310_153000
        run_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_prefix = args.experiment_name if args.experiment_name else "default"
        args.experiment_name = f"{exp_prefix}-{run_ts}"

        # Convert train_resolution string "81x480x720" to tuple
        frames, height, width = args.train_resolution.split("x")
        args.train_resolution = (int(frames), int(height), int(width))

        # Convert zcl_layers string "3,11,19,27,35" to List[int]
        args.zcl_layers = [int(x.strip()) for x in args.zcl_layers.split(",") if x.strip()]

        args.output_dir = os.path.join(args.output_dir, args.tracker_name, args.experiment_name)
        return cls(**vars(args))
