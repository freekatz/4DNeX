"""
finetune.py — 训练入口

职责：
    解析命令行参数 → 按 model_name / training_type 分派对应 Trainer 子类 → 执行训练。

调用链：
    scripts/finetune.sh
        └─ accelerate launch finetune.py --model_name wan-i2v-demb-samerope ...
               └─ main()
                    ├─ Args.parse_args()           # core/finetune/schemas/args.py
                    ├─ get_model_cls(...)           # core/finetune/models/utils.py
                    │       └─ 返回 WanI2VLoraTrainer / WanI2VDembSameRopeTrainer 等
                    └─ trainer.fit()               # core/finetune/trainer.py
                            ├─ prepare_models()
                            ├─ prepare_dataset()
                            ├─ prepare_trainable_parameters()
                            ├─ prepare_optimizer()
                            ├─ prepare_for_training()   ← accelerator.prepare() 在此调用
                            └─ train()
"""

from core.finetune.models.utils import get_model_cls
from core.finetune.schemas import Args


def main():
    # 解析所有 CLI 超参数，返回经 Pydantic 验证的 Args 实例
    # 参数定义见 core/finetune/schemas/args.py
    args = Args.parse_args()

    # 根据 model_name（如 "wan-i2v-demb-samerope"）和 training_type（"lora"/"sft"）
    # 从注册表中查找对应的 Trainer 子类
    trainer_cls = get_model_cls(args.model_name, args.training_type)

    # 实例化 Trainer：内部完成 Accelerator 初始化、模型加载、分布式环境配置
    trainer = trainer_cls(args)

    # 执行完整训练流程：数据预处理 → LoRA注入 → 优化器 → 训练循环 → checkpoint 保存
    trainer.fit()


if __name__ == "__main__":
    main()
