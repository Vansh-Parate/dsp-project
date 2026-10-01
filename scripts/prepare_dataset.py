"""Create deterministic noisy mixtures and JSONL train/validation manifests."""

import argparse
import json
from pathlib import Path

import numpy as np

from uahsn.audio import load_audio, mix_at_snr, save_audio
from uahsn.config import Config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-dir", type=Path, required=True)
    parser.add_argument("--noise-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/prepared"))
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--snr-db", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    clean_paths = sorted(args.clean_dir.rglob("*.wav"))
    noise_paths = sorted(args.noise_dir.rglob("*.wav"))
    if not clean_paths or not noise_paths:
        parser.error("Both directories must contain WAV files")
    if len(clean_paths) < 2 or not 0 < args.validation_fraction < 1:
        parser.error("Need at least two clean files and a validation fraction between 0 and 1")
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(clean_paths))
    validation_count = max(1, round(len(clean_paths) * args.validation_fraction))
    validation_ids = set(order[:validation_count])
    rows = {"train": [], "validation": []}
    for i, path in enumerate(clean_paths):
        clean = load_audio(path)
        noise = load_audio(noise_paths[i % len(noise_paths)])
        noisy = mix_at_snr(clean, noise, args.snr_db)
        split = "validation" if i in validation_ids else "train"
        stem = f"{i:05d}"
        clean_output = args.output_dir / split / f"{stem}_clean.wav"
        noisy_output = args.output_dir / split / f"{stem}_noisy.wav"
        save_audio(clean_output, clean)
        save_audio(noisy_output, noisy)
        rows[split].append({"clean": str(clean_output.resolve()), "noisy": str(noisy_output.resolve())})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, records in rows.items():
        with (args.output_dir / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for row in records:
                handle.write(json.dumps(row) + "\n")
    print({split: len(records) for split, records in rows.items()})


if __name__ == "__main__":
    main()
