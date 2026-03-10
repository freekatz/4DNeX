from core.models.trainer import WanTrainer
from core.schemas import Args


def main():
    args = Args.parse_args()
    trainer = WanTrainer(args)
    trainer.fit()


if __name__ == "__main__":
    main()
