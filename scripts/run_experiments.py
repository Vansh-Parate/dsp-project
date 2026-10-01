"""Run the five-method final evaluation and write publication tables."""

import argparse
import csv
import math
from pathlib import Path
import re
import sys
from time import perf_counter
from typing import Any
import warnings as python_warnings

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from uahsn.audio import load_audio, save_audio
from uahsn.baselines import (DecisionDirectedWiener, Noisy,
                             ReconstructedLinearPhaseBaseline, SpectralSubtraction)
from uahsn.config import Config
from uahsn.inference import UAHSNEnhancer
from uahsn.metrics import evaluate_pair, si_sdr
from uahsn.model import UAHSN, parameter_count
from uahsn.operator_metrics import operator_metrics
from uahsn.train import read_manifest


SNRS = (-5, 0, 5, 10, 15)
METHOD_NAMES = ("noisy", "spectral_subtraction", "wiener",
                "reconstructed_linear_phase_baseline", "uahsn")
PER_UTTERANCE_COLUMNS = (
    "utterance", "speaker", "noise_type", "snr_db", "method", "sisdr",
    "sisdr_improvement", "segsnr", "stoi", "pesq", "V_G", "RPD",
    "rho_alias", "NR", "params", "macs_per_second", "mean_ms_per_frame",
    "p99_9_ms_per_frame", "max_ms_per_frame", "RTF", "algorithmic_latency_ms",
)
SNR_TABLE_COLUMNS = ("Method", "-5 dB", "0 dB", "5 dB", "10 dB", "15 dB")
OPERATOR_COLUMNS = ("Method", "V_G", "RPD", "rho_alias", "NR")
NOISE_COLUMNS = ("Noise Type", "Method", "SI-SDR Improvement (dB)", "N")
COMPUTE_COLUMNS = (
    "Method", "params", "thop_macs_per_second", "analytic_macs_per_second",
    "mean_ms_per_frame", "p99_9_ms_per_frame", "max_ms_per_frame", "RTF",
    "algorithmic_latency_ms", "frames_timed",
)


def _number(value: Any) -> str | Any:
    if value is None or isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return "NaN"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.10g}"
    return value


def _write_csv(path: Path, columns: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows([{key: _number(row.get(key)) for key in columns} for row in rows])


def _mean(values: list[Any]) -> float:
    finite = [float(value) for value in values if isinstance(value, (int, float, np.number))
              and np.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def _latency_ms(method: Any) -> float:
    if isinstance(method, Noisy):
        return 0.0
    block_delay = method.config.hop_length / method.config.sample_rate * 1000
    if isinstance(method, ReconstructedLinearPhaseBaseline):
        return block_delay + (method.taps - 1) / (2 * method.config.sample_rate) * 1000
    return block_delay


def _sync(method: Any) -> None:
    if isinstance(method, UAHSNEnhancer) and method.device.type == "cuda":
        torch.cuda.synchronize(method.device)


def _profile_network(method: UAHSNEnhancer, warnings: list[str]) -> tuple[float, float]:
    """THOP and weight-matrix analytic MACs per second for the neural network."""
    config = method.config
    frames_per_second = config.sample_rate / config.hop_length
    analytic_per_frame = 96 * 64 + 2 * (3 * (64 * 64 + 64 * 64)) + 2 * (64 * 32)
    analytic = float(analytic_per_frame * frames_per_second)
    try:
        with torch.no_grad(), python_warnings.catch_warnings():
            python_warnings.filterwarnings("ignore", message="This API is being deprecated", category=UserWarning)
            python_warnings.filterwarnings("ignore", message="distutils Version classes are deprecated", category=DeprecationWarning)
            from thop import profile

            model = method.model.eval()
            sample_frames = 100
            example = torch.zeros(1, sample_frames, 96, device=method.device)
            macs, _ = profile(model, inputs=(example,), verbose=False)
        measured = float(macs / sample_frames * frames_per_second)
    except (ImportError, RuntimeError, ValueError) as error:
        warnings.append(f"UA-HSN THOP MACs unavailable: {error}")
        measured = float("nan")
    return measured, analytic


def _benchmark(
    method: Any, noisy: np.ndarray, repeats: int,
) -> tuple[Any, dict[str, float | np.ndarray]]:
    frame_times: list[np.ndarray] = []
    runtimes = []
    result = None
    duration = len(noisy) / method.config.sample_rate
    with torch.no_grad():
        method.model.eval() if isinstance(method, UAHSNEnhancer) else None
        for _ in range(repeats):
            _sync(method)
            tick = perf_counter()
            current = method.enhance(noisy)
            _sync(method)
            runtimes.append(perf_counter() - tick)
            if current.frame_times_ms is None:
                raise RuntimeError(f"{method.name} did not return frame timings")
            frame_times.append(np.asarray(current.frame_times_ms[:-1], dtype=np.float64))
            if result is None:
                result = current
    times = np.concatenate(frame_times)
    if not len(times):
        raise ValueError("Benchmark needs at least one processed frame")
    return result, {
        "mean_ms_per_frame": float(np.mean(times)),
        "p99_9_ms_per_frame": float(np.percentile(times, 99.9)),
        "max_ms_per_frame": float(np.max(times)),
        "RTF": float(np.sum(runtimes) / (repeats * duration)),
        "frame_times": times,
        "runtime_seconds": float(np.sum(runtimes)),
        "duration_seconds": float(repeats * duration),
    }


def _save_result(output: Path, utterance: str, name: str, result: Any) -> None:
    safe_utterance = re.sub(r"[^A-Za-z0-9_.+-]", "_", utterance)
    stem = f"{safe_utterance}__{name}"
    save_audio(output / "audio" / f"{stem}.wav", result.audio)
    arrays = {"gains": result.gains, "noise_psd": result.noise_psd,
              "frame_times_ms": result.frame_times_ms,
              "runtime_seconds": np.asarray(result.runtime_seconds),
              "real_time_factor": np.asarray(result.real_time_factor)}
    if result.band_gains is not None:
        arrays["band_gains"] = result.band_gains
    if result.standard_deviation is not None:
        arrays["standard_deviation"] = result.standard_deviation
    if result.fir_coefficients is not None:
        arrays["fir_coefficients"] = result.fir_coefficients
    path = output / "gains" / f"{stem}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _markdown_table(columns: tuple[str, ...], rows: list[dict[str, Any]]) -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(str(_number(row.get(column))) for column in columns) + " |"
            for row in rows]
    return "\n".join([header, divider, *body])


def _tables(
    rows: list[dict[str, Any]], method_names: tuple[str, ...],
    compute: list[dict[str, Any]], output: Path, warnings: list[str],
) -> None:
    tables: list[tuple[str, tuple[str, ...], list[dict[str, Any]]]] = []
    missing_cells: dict[str, int] = {}
    for filename, metric in (("table1_sisdr.csv", "sisdr"),
                             ("table2_stoi.csv", "stoi"),
                             ("table3_pesq.csv", "pesq")):
        table_rows = []
        for method in method_names:
            table_row: dict[str, Any] = {"Method": method}
            for snr in SNRS:
                values = [row[metric] for row in rows if row["method"] == method
                          and row["snr_db"] == snr]
                value = _mean(values)
                table_row[f"{snr} dB"] = value
                if not math.isfinite(value):
                    missing_cells[metric] = missing_cells.get(metric, 0) + 1
            table_rows.append(table_row)
        tables.append((filename, SNR_TABLE_COLUMNS, table_rows))
    for metric, count in sorted(missing_cells.items()):
        warnings.append(f"{metric}: {count} SNR table cells have no finite value and are NaN")
    operator_rows = [{"Method": method,
                      **{metric: _mean([row[metric] for row in rows if row["method"] == method])
                         for metric in ("V_G", "RPD", "rho_alias", "NR")}}
                     for method in method_names]
    tables.append(("table5_operator_metrics.csv", OPERATOR_COLUMNS, operator_rows))
    noise_types = sorted({str(row["noise_type"]) for row in rows if row["snr_db"] == 5})
    if not noise_types:
        warnings.append("No 5 dB mixtures were available for Table 6")
    noise_rows = []
    for noise_type in noise_types:
        for method in method_names:
            values = [row["sisdr_improvement"] for row in rows
                      if row["snr_db"] == 5 and row["noise_type"] == noise_type
                      and row["method"] == method]
            noise_rows.append({"Noise Type": noise_type, "Method": method,
                               "SI-SDR Improvement (dB)": _mean(values),
                               "N": sum(np.isfinite(value) for value in values)})
    tables.append(("table6_noise_type.csv", NOISE_COLUMNS, noise_rows))
    tables.append(("table7_compute.csv", COMPUTE_COLUMNS, compute))
    for filename, columns, table_rows in tables:
        _write_csv(output / filename, columns, table_rows)
    unique_utterances = len({row["utterance"] for row in rows})
    group_counts = {snr: len({row["utterance"] for row in rows if row["snr_db"] == snr})
                    for snr in SNRS}
    lines = ["# Final UA-HSN experiment summary", "",
             f"Evaluated {unique_utterances} test mixtures across five methods. "
             + ", ".join(f"{snr} dB: {count}" for snr, count in group_counts.items()) + ".",
             "", "Each table cell is the arithmetic mean of finite per-mixture values. "
             "The full per-utterance values are in `per_utterance.csv`. `NaN` means no valid measurement was available.",
             "", "## Results in plain English", ""]
    for method in method_names:
        method_rows = [row for row in rows if row["method"] == method]
        gain_db = _mean([row["sisdr_improvement"] for row in method_rows])
        stoi = _mean([row["stoi"] for row in method_rows])
        pesq = _mean([row["pesq"] for row in method_rows])
        lines.append(f"- {method}: mean SI-SDR improvement {_number(gain_db)} dB, "
                     f"mean STOI {_number(stoi)}, and mean wideband PESQ {_number(pesq)} "
                     "over available mixtures. The group-specific values appear below.")
    lines += ["", "## Table 1: SI-SDR (dB)", "", _markdown_table(SNR_TABLE_COLUMNS, tables[0][2]),
             "", "## Table 2: STOI", "", _markdown_table(SNR_TABLE_COLUMNS, tables[1][2]),
             "", "## Table 3: PESQ wideband", "", _markdown_table(SNR_TABLE_COLUMNS, tables[2][2]),
             "", "## Table 5: Operator metrics", "",
             "V_G is gain-change variance on clean-reference noise-only frames; RPD is isolated residual peaks per noise-only second; rho_alias is the circular IDFT energy ratio outside the central region; NR is shadow-filtered noise reduction in dB.",
             "", _markdown_table(OPERATOR_COLUMNS, operator_rows),
             "", "## Table 6: 5 dB SI-SDR improvement by noise type", "",
             "Improvement is enhanced SI-SDR minus the noisy-mixture SI-SDR for the same mixture.",
             "", _markdown_table(NOISE_COLUMNS, noise_rows),
             "", "## Table 7: Computation", "",
             "Frame times measure per-frame PSD/gain/model work after warmup; RTF measures complete enhancement, including STFT and reconstruction. Percentile and maximum use all timed frames across repeats. UA-HSN MACs from THOP count the neural network only; the analytic estimate counts weight-matrix products. Classical NumPy methods have no THOP MAC count. Parameter counts are trainable PyTorch parameters. Algorithmic latency is one input hop for framed methods, plus FIR group delay for the reconstructed baseline; pass-through noisy has zero buffering delay.",
             "", _markdown_table(COMPUTE_COLUMNS, compute),
             "", "## Missing values and warnings", ""]
    lines += [f"- {warning}" for warning in sorted(set(warnings))] if warnings else ["None."]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_final_experiments(
    test_manifest: Path, checkpoint: Path, output: Path,
    device: str = "cpu", benchmark_repeats: int = 3,
    oversubtraction: float = 1.0, linear_phase_taps: int = 63,
    linear_phase_frame_length: int = 320,
) -> list[dict[str, Any]]:
    if benchmark_repeats < 1:
        raise ValueError("benchmark_repeats must be positive")
    rows = read_manifest(test_manifest)
    if any(row.get("split", "test") != "test" for row in rows):
        raise ValueError("Final runner requires a test manifest")
    mixture_ids = [row.get("mixture_id") or f"{index:06d}" for index, row in enumerate(rows)]
    if len(set(mixture_ids)) != len(mixture_ids):
        raise ValueError("Test manifest contains duplicate mixture_id values")
    linear_config = Config(win_length=linear_phase_frame_length,
                           hop_length=linear_phase_frame_length // 2)
    methods: list[tuple[str, Any]] = [
        ("noisy", Noisy()),
        ("spectral_subtraction", SpectralSubtraction(oversubtraction=oversubtraction)),
        ("wiener", DecisionDirectedWiener()),
        ("reconstructed_linear_phase_baseline",
         ReconstructedLinearPhaseBaseline(linear_config, linear_phase_taps, oversubtraction)),
        ("uahsn", UAHSNEnhancer.from_checkpoint(checkpoint, device=device)),
    ]
    output.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    first_noise = load_audio(rows[0].get("mixture_path") or rows[0].get("noisy", ""))
    warmup = first_noise[:min(len(first_noise), 16000)]
    with torch.no_grad():
        for _, method in methods:
            if isinstance(method, UAHSNEnhancer):
                method.model.eval()
            _sync(method)
            method.enhance(warmup)
            _sync(method)
    thop_macs, analytic_macs = _profile_network(methods[-1][1], warnings)
    params = parameter_count(methods[-1][1].model)
    timings: dict[str, dict[str, Any]] = {
        name: {"frames": [], "runtime_seconds": 0.0, "duration_seconds": 0.0}
        for name in METHOD_NAMES
    }
    records: list[dict[str, Any]] = []
    unavailable: dict[str, int] = {}
    for index, manifest_row in enumerate(rows):
        snr = float(manifest_row["snr_db"])
        if snr not in SNRS:
            raise ValueError(f"Unsupported input SNR {snr} for row {index}")
        clean = load_audio(manifest_row.get("clean_path", manifest_row.get("clean", "")))
        noisy = load_audio(manifest_row.get("mixture_path") or manifest_row.get("noisy", ""))
        if len(clean) != len(noisy):
            raise ValueError(f"Clean/mixture length mismatch in test row {index}")
        utterance = mixture_ids[index]
        noisy_sisdr = si_sdr(clean, noisy)
        for name, method in methods:
            result, timing = _benchmark(method, noisy, benchmark_repeats)
            timings[name]["frames"].append(timing["frame_times"])
            timings[name]["runtime_seconds"] += timing["runtime_seconds"]
            timings[name]["duration_seconds"] += timing["duration_seconds"]
            _save_result(output, f"{index:06d}_{utterance}", name, result)
            signal = evaluate_pair(clean, result.audio)
            operator = operator_metrics(result, clean, noisy, method.config)
            raw = {"sisdr": signal["si_sdr_db"], "segsnr": signal["segmental_snr_db"],
                   "stoi": signal["stoi"], "pesq": signal["pesq_wb"],
                   "V_G": operator["gain_variance"],
                   "RPD": operator["residual_peak_density_per_second"],
                   "rho_alias": operator["aliasing_ratio"],
                   "NR": operator["noise_reduction_db"]}
            values = {}
            for key, value in raw.items():
                if value is None or not np.isfinite(value):
                    unavailable[key] = unavailable.get(key, 0) + 1
                    values[key] = float("nan")
                else:
                    values[key] = float(value)
            speaker = manifest_row.get("speaker") or "NaN"
            noise_type = manifest_row.get("noise_type") or "NaN"
            if speaker == "NaN":
                unavailable["speaker"] = unavailable.get("speaker", 0) + 1
            if noise_type == "NaN":
                unavailable["noise_type"] = unavailable.get("noise_type", 0) + 1
            records.append({
                "utterance": utterance, "speaker": speaker,
                "noise_type": noise_type, "snr_db": int(snr),
                "method": name, "sisdr": values["sisdr"],
                "sisdr_improvement": values["sisdr"] - noisy_sisdr,
                "segsnr": values["segsnr"], "stoi": values["stoi"],
                "pesq": values["pesq"], "V_G": values["V_G"], "RPD": values["RPD"],
                "rho_alias": values["rho_alias"], "NR": values["NR"],
                "params": params if name == "uahsn" else 0,
                "macs_per_second": thop_macs if name == "uahsn" else float("nan"),
                "mean_ms_per_frame": timing["mean_ms_per_frame"],
                "p99_9_ms_per_frame": timing["p99_9_ms_per_frame"],
                "max_ms_per_frame": timing["max_ms_per_frame"], "RTF": timing["RTF"],
                "algorithmic_latency_ms": _latency_ms(method),
            })
    for metric, count in sorted(unavailable.items()):
        warnings.append(f"{metric}: {count} per-utterance values were unavailable and written as NaN")
    warnings.append("Classical NumPy methods have no THOP MAC measurement; their macs_per_second cells are NaN")
    warnings.append("Classical FFT implementations have no comparable analytic MAC estimate; their analytic_macs_per_second cells are NaN")
    _write_csv(output / "per_utterance.csv", PER_UTTERANCE_COLUMNS, records)
    compute_rows = []
    for name, method in methods:
        samples = np.concatenate(timings[name]["frames"])
        if len(samples) < 1000:
            warnings.append(f"{name}: only {len(samples)} frames were timed; use more test audio for stable tail latency")
        compute_rows.append({
            "Method": name, "params": params if name == "uahsn" else 0,
            "thop_macs_per_second": thop_macs if name == "uahsn" else float("nan"),
            "analytic_macs_per_second": analytic_macs if name == "uahsn" else float("nan"),
            "mean_ms_per_frame": float(np.mean(samples)),
            "p99_9_ms_per_frame": float(np.percentile(samples, 99.9)),
            "max_ms_per_frame": float(np.max(samples)),
            "RTF": timings[name]["runtime_seconds"] / timings[name]["duration_seconds"],
            "algorithmic_latency_ms": _latency_ms(method), "frames_timed": len(samples),
        })
    _tables(records, METHOD_NAMES, compute_rows, output, warnings)
    for warning in sorted(set(warnings)):
        print(f"Warning: {warning}", file=sys.stderr)
    return records


def synthetic_smoke(output_dir: Path) -> None:
    config = Config()
    rng = np.random.default_rng(config.seed)
    t = np.arange(config.sample_rate, dtype=np.float64) / config.sample_rate
    clean = (0.20 * np.sin(2 * np.pi * 220 * t) + 0.12 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    noisy = (clean + 0.08 * rng.standard_normal(len(clean))).astype(np.float32)
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
        _save_result(output_dir, "synthetic", method.name, result)
    print(f"Synthetic pipeline smoke test passed; outputs: {output_dir}")
    print("UA-HSN output uses untrained weights and sigma0=1 for pipeline validation only.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-manifest", "--manifest", dest="test_manifest", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", "--output-dir", dest="output", type=Path, default=Path("results"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--benchmark-repeats", type=int, default=3)
    parser.add_argument("--oversubtraction", type=float, default=1.0)
    parser.add_argument("--linear-phase-filter-length", type=int, default=63)
    parser.add_argument("--linear-phase-frame-length", type=int, choices=(256, 320, 512), default=320)
    parser.add_argument("--synthetic-smoke", action="store_true")
    args = parser.parse_args()
    if args.synthetic_smoke:
        synthetic_smoke(args.output)
    elif args.test_manifest and args.checkpoint:
        records = run_final_experiments(args.test_manifest, args.checkpoint, args.output,
                                        args.device, args.benchmark_repeats,
                                        args.oversubtraction, args.linear_phase_filter_length,
                                        args.linear_phase_frame_length)
        print(f"Saved {len(records)} rows and six tables to {args.output}")
    else:
        parser.error("Use --synthetic-smoke or supply --test-manifest and --checkpoint")


if __name__ == "__main__":
    main()
