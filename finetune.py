from core.models.trainer import WanDualTrainer
from core.schemas import Args


def main():
    args = Args.parse_args()
    trainer = WanDualTrainer(args)
    trainer.fit()


if __name__ == "__main__":
    main()
