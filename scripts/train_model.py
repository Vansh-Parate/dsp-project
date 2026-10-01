"""Train UA-HSN from prepared clean/noisy manifests."""

import argparse
from pathlib import Path

from uahsn.train import train_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/uahsn.pt"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()
    print(train_model(args.train_manifest, args.validation_manifest, args.checkpoint,
                      args.epochs, args.learning_rate))


if __name__ == "__main__":
    main()
