"""Train UA-HSN from prepared manifests or run the fixed two-batch overfit check."""

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uahsn.config import Config
from uahsn.train import train_model, two_batch_overfit_test


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", "--train-manifest", dest="manifest", type=Path)
    parser.add_argument("--val-manifest", "--validation-manifest", dest="val_manifest", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/uahsn.pt"))
    parser.add_argument("--curves-csv", type=Path)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-train-files", type=int)
    parser.add_argument("--device", default="cpu", help="cpu, cuda, cuda:0, or auto")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overfit-test", action="store_true",
                        help="Run fixed two-batch synthetic overfit check without manifests")
    args = parser.parse_args()
    if args.overfit_test:
        print(two_batch_overfit_test(args.device, args.seed))
        return
    if args.manifest is None or args.val_manifest is None:
        parser.error("--manifest and --val-manifest are required for training")
    print(train_model(args.manifest, args.val_manifest, args.checkpoint,
                      epochs=args.epochs, learning_rate=args.learning_rate,
                      config=Config(seed=args.seed), batch_size=args.batch_size,
                      max_train_files=args.max_train_files, device=args.device,
                      curves_csv=args.curves_csv,
                      early_stopping_patience=args.early_stopping_patience))


if __name__ == "__main__":
    main()
