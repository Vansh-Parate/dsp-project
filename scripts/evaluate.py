"""Evaluate all five methods and preserve waveforms and gain operators."""

import argparse
import json
from pathlib import Path

import numpy as np

from uahsn.audio import load_audio, save_audio
from uahsn.baselines import Noisy, SpectralSubtraction, DecisionDirectedWiener, ReconstructedLinearPhaseBaseline
from uahsn.config import Config
from uahsn.inference import UAHSNEnhancer
from uahsn.metrics import evaluate_pair
from uahsn.operator_metrics import operator_metrics
from uahsn.train import read_manifest


def evaluate(
    manifest: Path, checkpoint: Path, output_dir: Path,
    oversubtraction: float = 1.0, linear_phase_taps: int = 63,
    linear_phase_frame_length: int = 320,
) -> list[dict[str, object]]:
    linear_config = Config(win_length=linear_phase_frame_length,
                           hop_length=linear_phase_frame_length // 2)
    methods = [Noisy(), SpectralSubtraction(oversubtraction=oversubtraction),
               DecisionDirectedWiener(),
               ReconstructedLinearPhaseBaseline(linear_config, linear_phase_taps, oversubtraction),
               UAHSNEnhancer.from_checkpoint(checkpoint)]
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for index, row in enumerate(read_manifest(manifest)):
        clean = load_audio(row.get("clean_path", row.get("clean", "")))
        noisy = load_audio(row.get("mixture_path") or row.get("noisy", ""))
        for method in methods:
            result = method.enhance(noisy)
            prefix = output_dir / f"{index:05d}_{method.name}"
            save_audio(prefix.with_suffix(".wav"), result.audio)
            arrays = {"gains": result.gains, "noise_psd": result.noise_psd,
                      "runtime_seconds": np.asarray(result.runtime_seconds),
                      "real_time_factor": np.asarray(result.real_time_factor)}
            if result.band_gains is not None:
                arrays["band_gains"] = result.band_gains
            if result.standard_deviation is not None:
                arrays["standard_deviation"] = result.standard_deviation
            if result.fir_coefficients is not None:
                arrays["fir_coefficients"] = result.fir_coefficients
            np.savez_compressed(prefix.with_suffix(".npz"), **arrays)
            record = {"item": index, "method": method.name,
                      **evaluate_pair(clean, result.audio), **operator_metrics(result),
                      "runtime_seconds": result.runtime_seconds,
                      "real_time_factor": result.real_time_factor}
            if method.name == "reconstructed_linear_phase_baseline":
                record.update({"filter_length": linear_phase_taps,
                               "frame_length": linear_phase_frame_length,
                               "oversubtraction": oversubtraction})
            records.append(record)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/evaluation"))
    parser.add_argument("--oversubtraction", type=float, default=1.0)
    parser.add_argument("--linear-phase-filter-length", type=int, default=63)
    parser.add_argument("--linear-phase-frame-length", type=int, choices=(256, 320, 512), default=320)
    args = parser.parse_args()
    records = evaluate(args.manifest, args.checkpoint, args.output_dir,
                       args.oversubtraction, args.linear_phase_filter_length,
                       args.linear_phase_frame_length)
    print(f"Saved {len(records)} method outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
