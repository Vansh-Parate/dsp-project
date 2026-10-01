"""Evaluate all five methods and preserve waveforms and gain operators."""

import argparse
import json
from pathlib import Path

import numpy as np

from uahsn.audio import load_audio, save_audio
from uahsn.baselines import Noisy, SpectralSubtraction, DecisionDirectedWiener, LinearPhaseSpectralSubtraction
from uahsn.benchmark import benchmark
from uahsn.inference import UAHSNEnhancer
from uahsn.metrics import evaluate_pair
from uahsn.operator_metrics import operator_metrics
from uahsn.train import read_manifest


def evaluate(manifest: Path, checkpoint: Path, output_dir: Path) -> list[dict[str, object]]:
    methods = [Noisy(), SpectralSubtraction(), DecisionDirectedWiener(),
               LinearPhaseSpectralSubtraction(), UAHSNEnhancer.from_checkpoint(checkpoint)]
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for index, row in enumerate(read_manifest(manifest)):
        clean = load_audio(row["clean"])
        noisy = load_audio(row["noisy"])
        for method in methods:
            result = method.enhance(noisy)
            prefix = output_dir / f"{index:05d}_{method.name}"
            save_audio(prefix.with_suffix(".wav"), result.audio)
            arrays = {"gains": result.gains}
            if result.band_gains is not None:
                arrays["band_gains"] = result.band_gains
            if result.standard_deviation is not None:
                arrays["standard_deviation"] = result.standard_deviation
            np.savez_compressed(prefix.with_suffix(".npz"), **arrays)
            record = {"item": index, "method": method.name,
                      **evaluate_pair(clean, result.audio), **operator_metrics(result),
                      **benchmark(method, noisy)}
            records.append(record)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/evaluation"))
    args = parser.parse_args()
    records = evaluate(args.manifest, args.checkpoint, args.output_dir)
    print(f"Saved {len(records)} method outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
