"""Prepare 16 kHz speech/noise files and reproducible CSV manifests."""

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from uahsn.audio import load_audio, mix_components, random_noise_segment, save_audio


FIELDS = ("mixture_id", "clean_path", "noise_path", "mixture_path", "speaker",
          "noise_type", "snr_db", "split", "seed")
TEST_SNRS = (-5, 0, 5, 10, 15)


def speaker_id(path: Path, root: Path, pattern: re.Pattern[str] | None = None) -> str:
    relative = path.relative_to(root)
    if pattern is not None:
        match = pattern.search(str(relative))
        if not match:
            raise ValueError(f"Speaker regex did not match {relative}")
        return match.groupdict().get("speaker") or match.group(1)
    if len(relative.parts) > 1:
        return relative.parts[0]
    match = re.match(r"^(.+?)[_-](?:\d+|utt\d+)(?:[_-].*)?$", path.stem, re.IGNORECASE)
    return match.group(1) if match else path.stem


def split_by_speaker(
    paths: list[Path], root: Path, rng: np.random.Generator,
    validation_fraction: float, test_fraction: float,
    pattern: re.Pattern[str] | None = None,
) -> dict[Path, tuple[str, str]]:
    speakers = {path: speaker_id(path, root, pattern) for path in paths}
    distinct = sorted(set(speakers.values()))
    if len(distinct) < 3:
        raise ValueError("At least three distinguishable speakers/groups are required for leakage-free train/validation/test splits")
    shuffled = np.asarray(distinct)[rng.permutation(len(distinct))]
    n_test = max(1, round(len(distinct) * test_fraction))
    n_validation = max(1, round(len(distinct) * validation_fraction))
    if n_test + n_validation >= len(distinct):
        raise ValueError("Split fractions leave no speakers for training")
    assignment = {speaker: "test" for speaker in shuffled[:n_test]}
    assignment.update({speaker: "validation" for speaker in shuffled[n_test:n_test + n_validation]})
    return {path: (assignment.get(speaker, "train"), speaker) for path, speaker in speakers.items()}


def _valid_sources(root: Path, output: Path, kind: str) -> list[tuple[Path, Path]]:
    files = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() == ".wav")
    accepted = []
    for index, path in enumerate(files):
        try:
            audio = load_audio(path)
            if np.mean(np.asarray(audio, dtype=np.float64) ** 2) <= 1e-12:
                raise ValueError("silent or near-silent recording")
            processed = output / "audio" / kind / f"{index:06d}.wav"
            save_audio(processed, audio)
            accepted.append((path, processed.resolve()))
        except (OSError, RuntimeError, ValueError, sf.SoundFileError) as error:
            print(f"Rejected {kind} file {path}: {error}", file=sys.stderr)
    return accepted


def prepare_dataset(
    clean_dir: Path, noise_dir: Path, output_dir: Path,
    seed: int = 0, validation_fraction: float = 0.1, test_fraction: float = 0.1,
    speaker_regex: str | None = None,
) -> dict[str, int]:
    if not 0 < validation_fraction < 1 or not 0 < test_fraction < 1:
        raise ValueError("Validation and test fractions must be between zero and one")
    clean_dir, noise_dir, output_dir = clean_dir.resolve(), noise_dir.resolve(), output_dir.resolve()
    missing = [str(path) for path in (clean_dir, noise_dir) if not path.is_dir()]
    if missing:
        raise ValueError("Input directory not found: " + ", ".join(missing)
                         + ". Supply existing WAV folders with --clean-dir and --noise-dir.")
    if output_dir.is_relative_to(clean_dir) or output_dir.is_relative_to(noise_dir):
        raise ValueError("Output directory must be outside the source directories")
    pattern = re.compile(speaker_regex) if speaker_regex else None
    if pattern and not (pattern.groups or "speaker" in pattern.groupindex):
        raise ValueError("Speaker regex needs a capture group")
    rng = np.random.default_rng(seed)
    clean_sources = _valid_sources(clean_dir, output_dir, "clean")
    noise_sources = _valid_sources(noise_dir, output_dir, "noise")
    if not clean_sources or not noise_sources:
        raise ValueError("No usable clean or noise WAV files")
    assignments = split_by_speaker([source for source, _ in clean_sources], clean_dir, rng,
                                   validation_fraction, test_fraction, pattern)
    manifests: dict[str, list[dict[str, str]]] = {key: [] for key in ("train", "validation", "test")}
    for index, (source, clean_path) in enumerate(clean_sources):
        split, speaker = assignments[source]
        snrs = (None,) if split == "train" else TEST_SNRS
        clean = load_audio(clean_path)
        for snr in snrs:
            noise_source, noise_path = noise_sources[int(rng.integers(len(noise_sources)))]
            noise_type = noise_source.relative_to(noise_dir).parts[0] if noise_source.parent != noise_dir else noise_source.stem
            row_seed = int(rng.integers(0, 2**32, dtype=np.uint32))
            label = "dynamic" if snr is None else f"{snr:+03d}db"
            mixture_id = f"{split}_{index:06d}_{label}"
            row = {"mixture_id": mixture_id, "clean_path": str(clean_path),
                   "noise_path": str(noise_path), "mixture_path": "", "speaker": speaker,
                   "noise_type": noise_type, "snr_db": "" if snr is None else str(snr),
                   "split": split, "seed": str(row_seed)}
            if snr is not None:
                noise = load_audio(noise_path)
                segment = random_noise_segment(noise, len(clean), np.random.default_rng(row_seed))
                aligned_clean, _, mixture = mix_components(clean, segment, snr)
                clean_output = output_dir / "mixtures" / split / f"{mixture_id}_clean.wav"
                mixture_output = output_dir / "mixtures" / split / f"{mixture_id}.wav"
                save_audio(clean_output, aligned_clean)
                save_audio(mixture_output, mixture)
                row["clean_path"] = str(clean_output.resolve())
                row["mixture_path"] = str(mixture_output.resolve())
            manifests[split].append(row)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in manifests.items():
        with (output_dir / f"{split}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
    counts = {split: len(rows) for split, rows in manifests.items()}
    print(f"Prepared {counts}; rejected clean/noise files are listed on stderr")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-dir", type=Path, required=True)
    parser.add_argument("--noise-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/prepared"))
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--speaker-regex", help="Capture speaker ID from relative clean path; use group 1 or (?P<speaker>...)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    prepare_dataset(args.clean_dir, args.noise_dir, args.output_dir, args.seed,
                    args.validation_fraction, args.test_fraction, args.speaker_regex)


if __name__ == "__main__":
    main()
