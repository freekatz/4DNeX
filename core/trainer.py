import hashlib
import json
import logging
import math
import os
import time
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

import diffusers
import torch
import transformers
import swanlab
from swanlab.integration.accelerate import SwanLabTracker
from accelerate.accelerator import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import (
    DistributedDataParallelKwargs,
    InitProcessGroupKwargs,
    ProjectConfiguration,
    gather_object,
    set_seed,
)
from diffusers.optimization import get_scheduler
from diffusers.pipelines import DiffusionPipeline
from diffusers.utils.export_utils import export_to_video
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from core.constants import LOG_LEVEL, LOG_NAME
from core.datasets import DatasetWithResize
from core.datasets.utils import (
    load_prompts,
    load_images,
    load_videos,
    preprocess_image_with_resize,
    preprocess_video_with_resize,
)
from core.schemas import Args, Components, State
from core.utils import (
    cast_training_params,
    free_memory,
    get_intermediate_ckpt_path,
    get_latest_ckpt_path_to_resume_from,
    get_memory_statistics,
    get_optimizer,
    string_to_filename,
    unload_model,
    unwrap_model,
)


logger = get_logger(LOG_NAME, LOG_LEVEL)

_DTYPE_MAP = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


class Trainer:
    # If set, should be a list of components to unload (refer to `Components``)
    UNLOAD_LIST: List[str] = []

    def __init__(self, args: Args) -> None:
        self.args = args
        self.state = State(
            weight_dtype=self.__get_training_dtype(),
            train_frames=self.args.train_resolution[0],
            train_height=self.args.train_resolution[1],
            train_width=self.args.train_resolution[2],
        )

        self.components: Components = self.load_components()
        self.accelerator: Accelerator = None
        self.dataset: Dataset = None
        self.data_loader: DataLoader = None

        self.optimizer = None
        self.lr_scheduler = None
        self._use_swanlab_tracker: bool = False

        self._init_distributed()
        self._init_logging()
        self._init_directories()

        self.state.using_deepspeed = self.accelerator.state.deepspeed_plugin is not None

    def _init_distributed(self):
        logging_dir = Path(self.args.output_dir, "logs", "tensorboard")
        project_config = ProjectConfiguration(project_dir=self.args.output_dir, logging_dir=logging_dir)
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        init_process_group_kwargs = InitProcessGroupKwargs(
            backend="nccl", timeout=timedelta(seconds=self.args.nccl_timeout)
        )
        mixed_precision = "no" if torch.backends.mps.is_available() else self.args.mixed_precision
        report_to_arg = (self.args.report_to or "none").lower()
        log_with = []
        if report_to_arg in ("tensorboard", "all"):
            log_with.append("tensorboard")
        self._use_swanlab_tracker = report_to_arg in ("swanlab", "all")

        accelerator = Accelerator(
            project_config=project_config,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            mixed_precision=mixed_precision,
            log_with=log_with or None,
            kwargs_handlers=[ddp_kwargs, init_process_group_kwargs],
        )

        # SwanLabTracker must be created after Accelerator (needs AcceleratorState).
        if self._use_swanlab_tracker:
            swanlab_kwargs = {}
            if self.args.experiment_name:
                swanlab_kwargs["experiment_name"] = self.args.experiment_name
            swanlab_tracker = SwanLabTracker(self.args.tracker_name, **swanlab_kwargs)
            accelerator.trackers.append(swanlab_tracker)

        # Disable AMP for MPS.
        if torch.backends.mps.is_available():
            accelerator.native_amp = False

        self.accelerator = accelerator

        if self.args.seed is not None:
            set_seed(self.args.seed)

    def _init_logging(self) -> None:
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=LOG_LEVEL,
        )
        if self.accelerator.is_local_main_process:
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()

        logger.info("Initialized Trainer")
        logger.info(f"Accelerator state: \n{self.accelerator.state}", main_process_only=False)

    def _init_directories(self) -> None:
        if self.accelerator.is_main_process:
            self.args.output_dir = Path(self.args.output_dir)
            self.args.output_dir.mkdir(parents=True, exist_ok=True)
            (self.args.output_dir / "checkpoints").mkdir(exist_ok=True)
            (self.args.output_dir / "logs").mkdir(exist_ok=True)
            (self.args.output_dir / "validation").mkdir(exist_ok=True)

    def check_setting(self) -> None:
        # Check for unload_list
        if not self.UNLOAD_LIST:
            logger.warning(
                "\033[91mNo unload_list specified for this Trainer. All components will be loaded to GPU during training.\033[0m"
            )
        else:
            for name in self.UNLOAD_LIST:
                if name not in self.components.model_fields:
                    raise ValueError(f"Invalid component name in unload_list: {name}")

    def prepare_models(self) -> None:
        logger.info("Initializing models")

        if self.components.vae is not None:
            if self.args.enable_slicing:
                self.components.vae.enable_slicing()
            if self.args.enable_tiling:
                self.components.vae.enable_tiling()

        self.state.transformer_config = self.components.transformer.config

    def prepare_dataset(self) -> None:
        logger.info("Initializing dataset and dataloader")

        self.dataset = DatasetWithResize(
            **(self.args.model_dump()),
            device=self.accelerator.device,
            max_num_frames=self.state.train_frames,
            height=self.state.train_height,
            width=self.state.train_width,
            trainer=self,
        )

        self.data_loader = torch.utils.data.DataLoader(
            self.dataset,
            collate_fn=self.collate_fn,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_memory,
            shuffle=True,
        )

    def prepare_trainable_parameters(self):
        logger.info("Initializing trainable parameters")

        # For mixed precision training we cast all non-trainable weights to half-precision
        # as these weights are only used for inference, keeping weights in full precision is not required.
        weight_dtype = self.state.weight_dtype

        if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
            # due to pytorch#99272, MPS does not yet support bfloat16.
            raise ValueError(
                "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
            )

        # For LoRA, we freeze all the parameters
        # For SFT, we train all the parameters in transformer model
        for attr_name, component in vars(self.components).items():
            if hasattr(component, "requires_grad_"):
                if self.args.training_type == "sft" and attr_name == "transformer":
                    component.requires_grad_(True)
                else:
                    component.requires_grad_(False)

        if self.args.training_type == "lora":
            transformer_lora_config = LoraConfig(
                r=self.args.rank,
                lora_alpha=self.args.lora_alpha,
                init_lora_weights=True,
                target_modules=self.args.target_modules,
            )
            self.components.transformer.add_adapter(transformer_lora_config)
            self.__prepare_saving_loading_hooks(transformer_lora_config)
            for name, param in self.components.transformer.named_parameters():
                if 'learnable_domain_embeddings' in name:
                    param.requires_grad_(True)
                    logger.info(f"Training {name} after adding LoRA")

        # Load components needed for training to GPU (except transformer), and cast them to the specified data type
        ignore_list = ["transformer"] + self.UNLOAD_LIST
        self.__move_components_to_device(dtype=weight_dtype, ignore_list=ignore_list)

        if self.args.gradient_checkpointing:
            self.components.transformer.enable_gradient_checkpointing()

    def prepare_optimizer(self) -> None:
        logger.info("Initializing optimizer and lr scheduler")

        # Make sure the trainable params are in float32
        cast_training_params([self.components.transformer], dtype=torch.float32)

        # For LoRA, we only want to train the LoRA weights
        # For SFT, we want to train all the parameters
        trainable_parameters = list(filter(lambda p: p.requires_grad, self.components.transformer.parameters()))
        transformer_parameters_with_lr = {
            "params": trainable_parameters,
            "lr": self.args.learning_rate,
        }
        params_to_optimize = [transformer_parameters_with_lr]
        self.state.num_trainable_parameters = sum(p.numel() for p in trainable_parameters)

        use_deepspeed_opt = (
            self.accelerator.state.deepspeed_plugin is not None
            and "optimizer" in self.accelerator.state.deepspeed_plugin.deepspeed_config
        )
        optimizer = get_optimizer(
            params_to_optimize=params_to_optimize,
            optimizer_name=self.args.optimizer,
            learning_rate=self.args.learning_rate,
            beta1=self.args.beta1,
            beta2=self.args.beta2,
            beta3=self.args.beta3,
            epsilon=self.args.epsilon,
            weight_decay=self.args.weight_decay,
            use_deepspeed=use_deepspeed_opt,
        )

        num_update_steps_per_epoch = math.ceil(len(self.data_loader) / self.args.gradient_accumulation_steps)
        if self.args.train_steps is None:
            self.args.train_steps = self.args.train_epochs * num_update_steps_per_epoch
            self.state.overwrote_max_train_steps = True

        use_deepspeed_lr_scheduler = (
            self.accelerator.state.deepspeed_plugin is not None
            and "scheduler" in self.accelerator.state.deepspeed_plugin.deepspeed_config
        )
        total_training_steps = self.args.train_steps * self.accelerator.num_processes
        num_warmup_steps = self.args.lr_warmup_steps * self.accelerator.num_processes

        if use_deepspeed_lr_scheduler:
            from accelerate.utils import DummyScheduler

            lr_scheduler = DummyScheduler(
                name=self.args.lr_scheduler,
                optimizer=optimizer,
                total_num_steps=total_training_steps,
                num_warmup_steps=num_warmup_steps,
            )
        else:
            lr_scheduler = get_scheduler(
                name=self.args.lr_scheduler,
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=total_training_steps,
                num_cycles=self.args.lr_num_cycles,
                power=self.args.lr_power,
            )

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def prepare_for_training(self) -> None:
        self.components.transformer, self.optimizer, self.data_loader, self.lr_scheduler = self.accelerator.prepare(
            self.components.transformer, self.optimizer, self.data_loader, self.lr_scheduler
        )

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(len(self.data_loader) / self.args.gradient_accumulation_steps)
        if self.state.overwrote_max_train_steps:
            self.args.train_steps = self.args.train_epochs * num_update_steps_per_epoch
        # Afterwards we recalculate our number of training epochs
        self.args.train_epochs = math.ceil(self.args.train_steps / num_update_steps_per_epoch)
        self.state.num_update_steps_per_epoch = num_update_steps_per_epoch

    def prepare_for_validation(self):
        validation_prompts = load_prompts(self.args.validation_dir / self.args.validation_prompts)

        if self.args.validation_images is not None:
            validation_images = load_images(self.args.validation_dir / self.args.validation_images)
        else:
            validation_images = [None] * len(validation_prompts)

        if self.args.validation_videos is not None:
            validation_videos = load_videos(self.args.validation_dir / self.args.validation_videos)
        else:
            validation_videos = [None] * len(validation_prompts)

        self.state.validation_prompts = validation_prompts
        self.state.validation_images = validation_images
        self.state.validation_videos = validation_videos

    def prepare_trackers(self) -> None:
        logger.info("Initializing trackers")

        tracker_name = self.args.tracker_name or "finetrainers-experiment"
        # Filter config to only include types supported by tensorboard hparams
        raw_config = self.args.model_dump()
        config = {}
        for k, v in raw_config.items():
            if isinstance(v, (int, float, str, bool)):
                config[k] = v
            elif isinstance(v, Path):
                config[k] = str(v)
            elif isinstance(v, (list, tuple)):
                config[k] = str(v)
            elif v is None:
                config[k] = "None"
        self.accelerator.init_trackers(tracker_name, config=config)

        # Save run metadata (config snapshot + README)
        if self.accelerator.is_main_process:
            self._save_run_metadata(config)

    def _save_run_metadata(self, config: dict) -> None:
        """Write config.json and README.md to output_dir for self-documentation."""
        import datetime as _dt

        output_dir = self.args.output_dir

        # config.json
        config_path = output_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2, default=str)

        # README.md
        report_to = self.args.report_to or "none"
        readme_path = output_dir / "README.md"
        readme = (
            f"# Training Run\n\n"
            f"- **Started**: {_dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"- **Model**: `{self.args.model_path}`\n"
            f"- **Training type**: {self.args.training_type} (rank={self.args.rank})\n"
            f"- **Resolution**: {self.args.train_resolution}\n"
            f"- **Trackers**: {report_to}\n\n"
            f"## Directory Layout\n\n"
            f"```\n"
            f"{output_dir.name}/\n"
            f"  config.json          # Full training configuration snapshot\n"
            f"  README.md            # This file\n"
            f"  checkpoints/         # Model checkpoints (step-NNNNNN/)\n"
            f"    step-NNNNNN/\n"
            f"      weights.safetensors  # LoRA + DLC + patch_embedding_xyz\n"
            f"      optimizer.bin        # Optimizer state\n"
            f"      scheduler.bin        # LR scheduler state\n"
            f"      random_states_*.pkl  # RNG states for reproducibility\n"
            f"  logs/                # Training logs\n"
            f"    tensorboard/       # TensorBoard event files\n"
            f"    finetune_*.log     # Full console logs\n"
            f"    latest.log         # Symlink to most recent log\n"
            f"  validation/          # Validation outputs (step-NNNNNN/)\n"
            f"    step-NNNNNN/\n"
            f"      sample_NN_*.png  # Generated images\n"
            f"      sample_NN_*.mp4  # Generated videos\n"
            f"```\n"
        )
        with open(readme_path, "w") as f:
            f.write(readme)

    def train(self) -> None:
        logger.info("Starting training")

        memory_statistics = get_memory_statistics()
        logger.info(f"Memory before training start: {json.dumps(memory_statistics, indent=4)}")

        self.state.total_batch_size_count = (
            self.args.batch_size * self.accelerator.num_processes * self.args.gradient_accumulation_steps
        )
        info = {
            "trainable parameters": self.state.num_trainable_parameters,
            "total samples": len(self.dataset),
            "train epochs": self.args.train_epochs,
            "train steps": self.args.train_steps,
            "batches per device": self.args.batch_size,
            "total batches observed per epoch": len(self.data_loader),
            "train batch size total count": self.state.total_batch_size_count,
            "gradient accumulation steps": self.args.gradient_accumulation_steps,
        }
        logger.info(f"Training configuration: {json.dumps(info, indent=4)}")

        # Training start banner
        total_params = sum(p.numel() for p in self.components.transformer.parameters())
        trainable = self.state.num_trainable_parameters
        pct = trainable / total_params * 100 if total_params > 0 else 0
        report_to = self.args.report_to or "none"
        res = self.args.train_resolution
        banner = (
            "\n"
            "══════════════════════════════════════════════════════════\n"
            "  One4D Training\n"
            f"  Model: {self.args.model_path.name} | {self.args.training_type} rank={self.args.rank}\n"
            f"  Resolution: {res[0]}x{res[1]}x{res[2]}\n"
            f"  Params: {trainable:,} trainable / {total_params:,} total ({pct:.2f}%)\n"
            f"  Steps: {self.args.train_steps} | Epochs: {self.args.train_epochs} | "
            f"Batch: {self.args.batch_size} x {self.accelerator.num_processes} GPU x {self.args.gradient_accumulation_steps} accum\n"
            f"  LR: {self.args.learning_rate:.0e} ({self.args.lr_scheduler}, {self.args.lr_warmup_steps} warmup)\n"
            f"  Trackers: {report_to}\n"
            f"  Output: {self.args.output_dir}\n"
            "══════════════════════════════════════════════════════════"
        )
        logger.info(banner)

        global_step = 0
        first_epoch = 0
        initial_global_step = 0

        # Potentially load in the weights and states from a previous save
        (
            resume_from_checkpoint_path,
            initial_global_step,
            global_step,
            first_epoch,
        ) = get_latest_ckpt_path_to_resume_from(
            resume_from_checkpoint=self.args.resume_from_checkpoint,
            num_update_steps_per_epoch=self.state.num_update_steps_per_epoch,
        )
        if resume_from_checkpoint_path is not None:
            self.accelerator.load_state(resume_from_checkpoint_path)

        progress_bar = tqdm(
            range(0, self.args.train_steps),
            initial=initial_global_step,
            desc="Training steps",
            disable=not self.accelerator.is_local_main_process,
        )

        accelerator = self.accelerator
        generator = torch.Generator(device=accelerator.device)
        if self.args.seed is not None:
            generator = generator.manual_seed(self.args.seed)
        self.state.generator = generator

        free_memory()
        ema_loss = None
        ema_beta = 0.95
        for epoch in range(first_epoch, self.args.train_epochs):
            logger.debug(f"Starting epoch ({epoch + 1}/{self.args.train_epochs})")

            self.components.transformer.train()
            models_to_accumulate = [self.components.transformer]
            last_step_end_monotonic = time.monotonic()

            for step, batch in enumerate(self.data_loader):
                logger.debug(f"Starting step {step + 1}")
                logs = {}
                step_start_monotonic = time.monotonic()
                data_wait_sec = step_start_monotonic - last_step_end_monotonic

                with accelerator.accumulate(models_to_accumulate):
                    # These weighting schemes use a uniform timestep sampling and instead post-weight the loss
                    loss = self.compute_loss(batch)
                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        if accelerator.distributed_type == DistributedType.DEEPSPEED:
                            grad_norm = self.components.transformer.get_global_grad_norm()
                            # In some cases the grad norm may not return a float
                            if torch.is_tensor(grad_norm):
                                grad_norm = grad_norm.item()
                        else:
                            grad_norm = accelerator.clip_grad_norm_(
                                self.components.transformer.parameters(), self.args.max_grad_norm
                            )
                            if torch.is_tensor(grad_norm):
                                grad_norm = grad_norm.item()

                        logs["grad_norm"] = grad_norm
                        branch_grad_norms = self.__compute_one4d_branch_grad_norms()
                        logs.update(branch_grad_norms)

                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    self.__maybe_save_checkpoint(global_step)

                loss_scalar = float(loss.detach().item())
                step_time_sec = time.monotonic() - step_start_monotonic
                if ema_loss is None:
                    ema_loss = loss_scalar
                else:
                    ema_loss = ema_beta * ema_loss + (1.0 - ema_beta) * loss_scalar

                iter_per_sec = 1.0 / step_time_sec if step_time_sec > 0 else 0.0
                samples_per_sec_device = self.args.batch_size * iter_per_sec
                samples_per_sec_global = self.args.batch_size * self.accelerator.num_processes * iter_per_sec

                # Keep concise, backward-compatible keys for tqdm display.
                logs["loss"] = loss_scalar
                logs["lr"] = float(self.lr_scheduler.get_last_lr()[0])
                logs["step_time_sec"] = round(step_time_sec, 3)
                logs["idle_before_step_sec"] = round(data_wait_sec, 3)

                # Rich metrics for tracker backends (TensorBoard/SwanLab).
                logs["train/loss"] = loss_scalar
                logs["train/loss_ema"] = float(ema_loss)
                logs["train/epoch"] = float(epoch + 1)
                logs["optim/lr"] = logs["lr"]
                logs["perf/step_time_sec"] = float(step_time_sec)
                logs["perf/data_wait_sec"] = float(data_wait_sec)
                logs["perf/iter_per_sec"] = float(iter_per_sec)
                logs["perf/samples_per_sec_device"] = float(samples_per_sec_device)
                logs["perf/samples_per_sec_global"] = float(samples_per_sec_global)
                logs["progress/global_step"] = float(global_step)
                logs["progress/percent_complete"] = float(
                    100.0 * global_step / self.args.train_steps if self.args.train_steps else 0.0
                )

                if "grad_norm" in logs:
                    logs["optim/grad_norm"] = float(logs["grad_norm"])

                for key in ["optim/grad_norm_rgb_lora", "optim/grad_norm_xyz_lora", "optim/grad_norm_dlc"]:
                    if key in logs:
                        logs[key] = float(logs[key])

                if self.state.latest_loss_metrics:
                    one4d_metric_keys = [
                        "loss_rgb",
                        "loss_xyz",
                        "loss_ratio_rgb",
                        "loss_ratio_xyz",
                        "loss_gap_abs",
                        "loss_gap_rel",
                        "dlc_coupling_rgb",
                        "dlc_coupling_xyz",
                        "dlc_balance_gap",
                    ]
                    for metric_key in one4d_metric_keys:
                        if metric_key in self.state.latest_loss_metrics:
                            logs[f"train/{metric_key}"] = float(self.state.latest_loss_metrics[metric_key])

                if torch.cuda.is_available() and accelerator.device.type == "cuda":
                    mem_alloc = torch.cuda.memory_allocated(accelerator.device) / (1024**3)
                    mem_reserved = torch.cuda.memory_reserved(accelerator.device) / (1024**3)
                    mem_max_alloc = torch.cuda.max_memory_allocated(accelerator.device) / (1024**3)
                    logs["system/gpu_mem_allocated_gb"] = float(mem_alloc)
                    logs["system/gpu_mem_reserved_gb"] = float(mem_reserved)
                    logs["system/gpu_mem_max_allocated_gb"] = float(mem_max_alloc)

                # EMA divergence
                logs["train/loss_ema_divergence"] = float(
                    abs(ema_loss - loss_scalar) / (ema_loss + 1e-8)
                )

                # Grad clip indicator
                if "grad_norm" in logs:
                    logs["optim/grad_clipped"] = 1.0 if logs["grad_norm"] > self.args.max_grad_norm else 0.0

                # Sampled timestep (from compute_loss)
                if self.state.latest_loss_metrics:
                    if "sampled_timestep" in self.state.latest_loss_metrics:
                        logs["train/sampled_timestep"] = float(self.state.latest_loss_metrics["sampled_timestep"])
                    if "timestep_bucket" in self.state.latest_loss_metrics:
                        logs["train/timestep_bucket"] = float(self.state.latest_loss_metrics["timestep_bucket"])

                # LoRA weight norms (every 10 steps to avoid overhead)
                if accelerator.sync_gradients and global_step % 10 == 0:
                    lora_norms = self.__compute_lora_weight_norms()
                    logs.update(lora_norms)

                _mem_str = f"{mem_alloc:.1f}" if torch.cuda.is_available() and accelerator.device.type == "cuda" else "n/a"
                progress_bar.set_postfix(
                    {
                        "epoch": f"{epoch+1}/{self.args.train_epochs}",
                        "loss": f"{loss_scalar:.4f}",
                        "rgb": (
                            f"{logs['train/loss_rgb']:.4f}" if "train/loss_rgb" in logs else "n/a"
                        ),
                        "xyz": (
                            f"{logs['train/loss_xyz']:.4f}" if "train/loss_xyz" in logs else "n/a"
                        ),
                        "ratio": (
                            f"{logs['train/loss_ratio_rgb']:.2f}" if "train/loss_ratio_rgb" in logs else "n/a"
                        ),
                        "lr": f"{logs['lr']:.1e}",
                        "grad": f"{logs['grad_norm']:.2f}" if "grad_norm" in logs else "n/a",
                        "s/it": f"{step_time_sec:.1f}",
                        "mem": _mem_str,
                    }
                )
                last_step_end_monotonic = time.monotonic()

                if accelerator.is_main_process and accelerator.sync_gradients and global_step % 10 == 0:
                    _r = lambda k, d=0.0: round(float(logs.get(k, d)), 4)
                    _mem = f" mem={mem_alloc:.1f}G" if torch.cuda.is_available() and accelerator.device.type == "cuda" else ""
                    _gap_pct = f"{_r('train/loss_gap_rel') * 100:.1f}%"
                    console_line = (
                        f"[Step {global_step}/{self.args.train_steps}] "
                        f"loss={loss_scalar:.4f} ema={ema_loss:.4f} | "
                        f"rgb={_r('train/loss_rgb'):.4f} xyz={_r('train/loss_xyz'):.4f} "
                        f"ratio={_r('train/loss_ratio_rgb'):.2f}/{_r('train/loss_ratio_xyz'):.2f} gap={_gap_pct} | "
                        f"dlc={_r('train/dlc_coupling_rgb'):.3f}/{_r('train/dlc_coupling_xyz'):.3f} | "
                        f"lr={logs.get('lr', 0):.1e} "
                        f"grad={_r('optim/grad_norm'):.2f} "
                        f"(rgb={_r('optim/grad_norm_rgb_lora'):.2f} xyz={_r('optim/grad_norm_xyz_lora'):.2f} dlc={_r('optim/grad_norm_dlc'):.3f}) | "
                        f"{step_time_sec:.1f}s/it{_mem}"
                    )
                    logger.info(console_line)
                    # Also keep JSON for machine parsing at debug level
                    logger.debug(f"One4D metrics: {json.dumps({k: v for k, v in logs.items() if k.startswith(('train/', 'optim/'))}, ensure_ascii=True)}")

                # Maybe run validation
                should_run_validation = self.args.do_validation and global_step % self.args.validation_steps == 0
                if should_run_validation:
                    del loss
                    free_memory()
                    self.validate(global_step)

                # Only send prefixed metrics to trackers (filter out tqdm flat keys)
                tracker_logs = {k: v for k, v in logs.items() if "/" in k and isinstance(v, (int, float))}
                accelerator.log(tracker_logs, step=global_step)

                if global_step >= self.args.train_steps:
                    break

            memory_statistics = get_memory_statistics()
            logger.info(f"Memory after epoch {epoch + 1}: {json.dumps(memory_statistics, indent=4)}")

        accelerator.wait_for_everyone()
        self.__maybe_save_checkpoint(global_step, must_save=True)
        if self.args.do_validation:
            free_memory()
            self.validate(global_step)

        del self.components
        free_memory()
        memory_statistics = get_memory_statistics()
        logger.info(f"Memory after training end: {json.dumps(memory_statistics, indent=4)}")

        accelerator.end_training()

    def validate(self, step: int) -> None:
        logger.info("Starting validation")

        accelerator = self.accelerator
        num_validation_samples = len(self.state.validation_prompts)

        if num_validation_samples == 0:
            logger.warning("No validation samples found. Skipping validation.")
            return

        self.components.transformer.eval()
        torch.set_grad_enabled(False)

        memory_statistics = get_memory_statistics()
        logger.info(f"Memory before validation start: {json.dumps(memory_statistics, indent=4)}")

        #####  Initialize pipeline  #####
        pipe = self.initialize_pipeline()

        if self.state.using_deepspeed:
            # Can't using model_cpu_offload in deepspeed,
            # so we need to move all components in pipe to device
            # pipe.to(self.accelerator.device, dtype=self.state.weight_dtype)
            self.__move_components_to_device(dtype=self.state.weight_dtype, ignore_list=["transformer"])
        else:
            # if not using deepspeed, use model_cpu_offload to further reduce memory usage
            # Or use pipe.enable_sequential_cpu_offload() to further reduce memory usage
            pipe.enable_model_cpu_offload(device=self.accelerator.device)

            # Convert all model weights to training dtype
            # Note, this will change LoRA weights in self.components.transformer to training dtype, rather than keep them in fp32
            pipe = pipe.to(dtype=self.state.weight_dtype)

        #################################

        all_processes_artifacts = []
        for i in range(num_validation_samples):
            if self.state.using_deepspeed and self.accelerator.deepspeed_plugin.zero_stage != 3:
                # Skip current validation on all processes but one
                if i % accelerator.num_processes != accelerator.process_index:
                    continue

            prompt = self.state.validation_prompts[i]
            image = self.state.validation_images[i]
            video = self.state.validation_videos[i]

            if image is not None:
                image = preprocess_image_with_resize(image, self.state.train_height, self.state.train_width)
                # Convert image tensor (C, H, W) to PIL images
                image = image.to(torch.uint8)
                image = image.permute(1, 2, 0).cpu().numpy()
                image = Image.fromarray(image)

            if video is not None:
                video = preprocess_video_with_resize(
                    video, self.state.train_frames, self.state.train_height, self.state.train_width
                )
                # Convert video tensor (F, C, H, W) to list of PIL images
                video = video.round().clamp(0, 255).to(torch.uint8)
                video = [Image.fromarray(frame.permute(1, 2, 0).cpu().numpy()) for frame in video]

            logger.debug(
                f"Validating sample {i + 1}/{num_validation_samples} on process {accelerator.process_index}. Prompt: {prompt}",
                main_process_only=False,
            )
            validation_artifacts = self.validation_step({"prompt": prompt, "image": image, "video": video}, pipe)

            if (
                self.state.using_deepspeed
                and self.accelerator.deepspeed_plugin.zero_stage == 3
                and not accelerator.is_main_process
            ):
                continue

            prompt_filename = string_to_filename(prompt)[:25]
            # Calculate hash of reversed prompt as a unique identifier
            reversed_prompt = prompt[::-1]
            hash_suffix = hashlib.md5(reversed_prompt.encode()).hexdigest()[:5]

            artifacts = {
                "image": {"type": "image", "value": image},
                "video": {"type": "video", "value": video},
            }
            for i, (artifact_type, artifact_value) in enumerate(validation_artifacts):
                artifacts.update({f"artifact_{i}": {"type": artifact_type, "value": artifact_value}})
            logger.debug(
                f"Validation artifacts on process {accelerator.process_index}: {list(artifacts.keys())}",
                main_process_only=False,
            )

            for key, value in list(artifacts.items()):
                artifact_type = value["type"]
                artifact_value = value["value"]
                if artifact_type not in ["image", "video"] or artifact_value is None:
                    continue

                extension = "png" if artifact_type == "image" else "mp4"
                filename = f"sample_{i:02d}_{prompt_filename}_{hash_suffix}.{extension}"
                validation_path = self.args.output_dir / "validation" / f"step-{step:06d}"
                validation_path.mkdir(parents=True, exist_ok=True)
                filename = str(validation_path / filename)

                if artifact_type == "image":
                    logger.debug(f"Saving image to {filename}")
                    artifact_value.save(filename)
                    artifact_value = swanlab.Image(filename)
                elif artifact_type == "video":
                    logger.debug(f"Saving video to {filename}")
                    export_to_video(artifact_value, filename, fps=self.args.gen_fps)
                    artifact_value = swanlab.Video(filename, caption=prompt)

                all_processes_artifacts.append(artifact_value)

        all_artifacts = gather_object(all_processes_artifacts)

        if accelerator.is_main_process:
            for tracker in accelerator.trackers:
                if tracker.name == "swanlab":
                    tracker_key = "validation"
                    image_artifacts = [a for a in all_artifacts if isinstance(a, swanlab.Image)]
                    video_artifacts = [a for a in all_artifacts if isinstance(a, swanlab.Video)]
                    if image_artifacts or video_artifacts:
                        tracker.log(
                            {tracker_key: {"images": image_artifacts, "videos": video_artifacts}},
                            step=step,
                        )
                    break

        ##########  Clean up  ##########
        if self.state.using_deepspeed:
            del pipe
            # Unload models except those needed for training
            self.__move_components_to_cpu(unload_list=self.UNLOAD_LIST)
        else:
            pipe.remove_all_hooks()
            del pipe
            # Load models except those not needed for training
            self.__move_components_to_device(dtype=self.state.weight_dtype, ignore_list=self.UNLOAD_LIST)
            self.components.transformer.to(self.accelerator.device, dtype=self.state.weight_dtype)

            # Change trainable weights back to fp32 to keep with dtype after prepare the model
            cast_training_params([self.components.transformer], dtype=torch.float32)

        free_memory()
        accelerator.wait_for_everyone()
        ################################

        memory_statistics = get_memory_statistics()
        logger.info(f"Memory after validation end: {json.dumps(memory_statistics, indent=4)}")
        torch.cuda.reset_peak_memory_stats(accelerator.device)

        torch.set_grad_enabled(True)
        self.components.transformer.train()

    def fit(self):
        self.check_setting()
        self.prepare_models()
        self.prepare_dataset()
        self.prepare_trainable_parameters()
        self.prepare_optimizer()
        self.prepare_for_training()
        if self.args.do_validation:
            self.prepare_for_validation()
        self.prepare_trackers()
        self.train()

    def collate_fn(self, examples: List[Dict[str, Any]]):
        raise NotImplementedError

    def load_components(self) -> Components:
        raise NotImplementedError

    def initialize_pipeline(self) -> DiffusionPipeline:
        raise NotImplementedError

    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        # shape of input video: [B, C, F, H, W], where B = 1
        # shape of output video: [B, C', F', H', W'], where B = 1
        raise NotImplementedError

    def encode_text(self, text: str) -> torch.Tensor:
        # shape of output text: [batch size, sequence length, embedding dimension]
        raise NotImplementedError

    def compute_loss(self, batch) -> torch.Tensor:
        raise NotImplementedError

    def validation_step(self, eval_data, pipe) -> List[Tuple[str, Image.Image | List[Image.Image]]]:
        raise NotImplementedError

    def __get_training_dtype(self) -> torch.dtype:
        if self.args.mixed_precision == "no":
            return _DTYPE_MAP["fp32"]
        elif self.args.mixed_precision == "fp16":
            return _DTYPE_MAP["fp16"]
        elif self.args.mixed_precision == "bf16":
            return _DTYPE_MAP["bf16"]
        else:
            raise ValueError(f"Invalid mixed precision: {self.args.mixed_precision}")

    def __compute_one4d_branch_grad_norms(self) -> Dict[str, float]:
        """Compute grouped grad norms for One4D trainable groups.

        Groups:
        - RGB LoRA adapter params
        - XYZ LoRA adapter params
        - DLC params
        """
        try:
            model = unwrap_model(self.accelerator, self.components.transformer)
        except Exception:
            return {}

        group_sum_sq = {
            "rgb_lora": 0.0,
            "xyz_lora": 0.0,
            "dlc": 0.0,
        }

        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            name_lower = name.lower()
            grad_sq = float(torch.sum(param.grad.detach().float() ** 2).item())

            if "dlc_" in name_lower:
                group_sum_sq["dlc"] += grad_sq
                continue

            if "lora" not in name_lower:
                continue

            if "rgb" in name_lower:
                group_sum_sq["rgb_lora"] += grad_sq
            elif "xyz" in name_lower:
                group_sum_sq["xyz_lora"] += grad_sq

        grad_metrics: Dict[str, float] = {}
        for key, sum_sq in group_sum_sq.items():
            if sum_sq > 0:
                grad_metrics[f"optim/grad_norm_{key}"] = math.sqrt(sum_sq)

        return grad_metrics

    def __compute_lora_weight_norms(self) -> Dict[str, float]:
        """Compute L2 norms of LoRA adapter weights (how far they've moved from init)."""
        try:
            model = unwrap_model(self.accelerator, self.components.transformer)
        except Exception:
            return {}

        group_sum_sq = {"rgb_lora": 0.0, "xyz_lora": 0.0, "dlc": 0.0}

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            name_lower = name.lower()
            w_sq = float(torch.sum(param.detach().float() ** 2).item())

            if "dlc_" in name_lower:
                group_sum_sq["dlc"] += w_sq
            elif "lora" in name_lower:
                if "rgb" in name_lower:
                    group_sum_sq["rgb_lora"] += w_sq
                elif "xyz" in name_lower:
                    group_sum_sq["xyz_lora"] += w_sq

        norms: Dict[str, float] = {}
        for key, sum_sq in group_sum_sq.items():
            if sum_sq > 0:
                norms[f"optim/weight_norm_{key}"] = math.sqrt(sum_sq)
        return norms

    def __move_components_to_device(self, dtype, ignore_list: List[str] = []):
        ignore_list = set(ignore_list)
        components = self.components.model_dump()
        for name, component in components.items():
            if not isinstance(component, type) and hasattr(component, "to"):
                if name not in ignore_list:
                    setattr(self.components, name, component.to(self.accelerator.device, dtype=dtype))

    def __move_components_to_cpu(self, unload_list: List[str] = []):
        unload_list = set(unload_list)
        components = self.components.model_dump()
        for name, component in components.items():
            if not isinstance(component, type) and hasattr(component, "to"):
                if name in unload_list:
                    setattr(self.components, name, component.to("cpu"))

    def __prepare_saving_loading_hooks(self, transformer_lora_config):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if self.accelerator.is_main_process:
                transformer_lora_layers_to_save = None

                for model in models:
                    if isinstance(
                        unwrap_model(self.accelerator, model),
                        type(unwrap_model(self.accelerator, self.components.transformer)),
                    ):
                        model = unwrap_model(self.accelerator, model)
                        transformer_lora_layers_to_save = get_peft_model_state_dict(model)
                    else:
                        raise ValueError(f"Unexpected save model: {model.__class__}")

                    # make sure to pop weight so that corresponding model is not saved again
                    if weights:
                        weights.pop()

                self.components.pipeline_cls.save_lora_weights(
                    output_dir,
                    transformer_lora_layers=transformer_lora_layers_to_save,
                )

                # Save extra trainable parameters (e.g. learnable_domain_embeddings)
                model = unwrap_model(self.accelerator, self.components.transformer)
                extra_state = {
                    k: v for k, v in model.state_dict().items()
                    if 'learnable_domain_embeddings' in k
                }
                if extra_state:
                    torch.save(
                        extra_state,
                        os.path.join(output_dir, "learnable_domain_embeddings.pt"),
                    )

        def load_model_hook(models, input_dir):
            if not self.accelerator.distributed_type == DistributedType.DEEPSPEED:
                while len(models) > 0:
                    model = models.pop()
                    if isinstance(
                        unwrap_model(self.accelerator, model),
                        type(unwrap_model(self.accelerator, self.components.transformer)),
                    ):
                        transformer_ = unwrap_model(self.accelerator, model)
                    else:
                        raise ValueError(f"Unexpected save model: {unwrap_model(self.accelerator, model).__class__}")
            else:
                transformer_ = unwrap_model(self.accelerator, self.components.transformer).__class__.from_pretrained(
                    self.args.model_path, subfolder="transformer"
                )
                transformer_.add_adapter(transformer_lora_config)

            lora_state_dict = self.components.pipeline_cls.lora_state_dict(input_dir)
            transformer_state_dict = {
                f'{k.replace("transformer.", "")}': v
                for k, v in lora_state_dict.items()
                if k.startswith("transformer.")
            }
            incompatible_keys = set_peft_model_state_dict(transformer_, transformer_state_dict, adapter_name="default")
            if incompatible_keys is not None:
                # check only for unexpected keys
                unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
                if unexpected_keys:
                    logger.warning(
                        f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                        f" {unexpected_keys}. "
                    )

        self.accelerator.register_save_state_pre_hook(save_model_hook)
        self.accelerator.register_load_state_pre_hook(load_model_hook)

    def __maybe_save_checkpoint(self, global_step: int, must_save: bool = False):
        if self.accelerator.distributed_type == DistributedType.DEEPSPEED or self.accelerator.is_main_process:
            if must_save or global_step % self.args.checkpointing_steps == 0:
                # for training
                save_path = get_intermediate_ckpt_path(
                    checkpointing_limit=self.args.checkpointing_limit,
                    step=global_step,
                    output_dir=self.args.output_dir,
                )
                self.accelerator.save_state(save_path, safe_serialization=True)
