"""Run a synthetic pipeline smoke test or evaluate a prepared manifest."""

import argparse
from pathlib import Path

import numpy as np

from uahsn.audio import save_audio
from uahsn.baselines import Noisy, SpectralSubtraction, DecisionDirectedWiener, ReconstructedLinearPhaseBaseline
from uahsn.config import Config
from uahsn.inference import UAHSNEnhancer
from uahsn.model import UAHSN
from scripts.evaluate import evaluate


def synthetic_smoke(output_dir: Path) -> None:
    config = Config()
    rng = np.random.default_rng(config.seed)
    t = np.arange(config.sample_rate, dtype=np.float64) / config.sample_rate
    clean = (0.20 * np.sin(2 * np.pi * 220 * t) + 0.12 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    noisy = (clean + 0.08 * rng.standard_normal(len(clean))).astype(np.float32)
    import torch
    torch.manual_seed(config.seed)
    methods = [Noisy(config), SpectralSubtraction(config), DecisionDirectedWiener(config),
               ReconstructedLinearPhaseBaseline(config), UAHSNEnhancer(UAHSN(), 1.0, config)]
    output_dir.mkdir(parents=True, exist_ok=True)
    save_audio(output_dir / "clean.wav", clean)
    for method in methods:
        result = method.enhance(noisy)
        assert len(result.audio) == len(clean)
        assert result.gains.shape[1] == config.n_fft // 2 + 1
        assert np.all(np.isfinite(result.audio))
        assert np.all(np.isfinite(result.gains))
        assert np.all(np.isfinite(result.noise_psd))
        assert np.min(result.gains) >= -1e-10
        assert np.max(result.gains) <= 1 + 1e-6
        assert result.runtime_seconds > 0
        save_audio(output_dir / f"{method.name}.wav", result.audio)
        arrays = {"gains": result.gains, "noise_psd": result.noise_psd,
                  "runtime_seconds": np.asarray(result.runtime_seconds),
                  "real_time_factor": np.asarray(result.real_time_factor)}
        if result.band_gains is not None:
            arrays["band_gains"] = result.band_gains
        if result.standard_deviation is not None:
            arrays["standard_deviation"] = result.standard_deviation
        if result.fir_coefficients is not None:
            arrays["fir_coefficients"] = result.fir_coefficients
        np.savez_compressed(output_dir / f"{method.name}.npz", **arrays)
    print(f"Synthetic pipeline smoke test passed; outputs: {output_dir}")
    print("UA-HSN output uses untrained weights and sigma0=1 for pipeline validation only.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic-smoke", action="store_true")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/synthetic_smoke"))
    args = parser.parse_args()
    if args.synthetic_smoke:
        synthetic_smoke(args.output_dir)
    elif args.manifest and args.checkpoint:
        evaluate(args.manifest, args.checkpoint, args.output_dir)
    else:
        parser.error("Use --synthetic-smoke or supply --manifest and --checkpoint")


if __name__ == "__main__":
    main()
