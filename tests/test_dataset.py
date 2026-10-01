"""End-to-end dataset preparation checks on small local WAV fixtures."""

import csv
import re
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from scripts.prepare_dataset import prepare_dataset, speaker_id
from uahsn.audio import load_audio
from uahsn.bands import Bands
from uahsn.config import Config
from uahsn.train import make_example, read_manifest


class TestDataset(unittest.TestCase):
    def test_speaker_inference(self) -> None:
        root = Path("/clean")
        self.assertEqual(speaker_id(root / "p225_001.wav", root), "p225")
        self.assertEqual(speaker_id(root / "p225" / "chapter" / "001.wav", root), "p225")
        self.assertEqual(speaker_id(root / "folder" / "A1_001.wav", root,
                                    re.compile(r"(?P<speaker>A1)_")), "A1")

    def test_saved_snr_splits_resampling_and_corrupt_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clean_dir = root / "clean"
            noise_dir = root / "noise"
            rng = np.random.default_rng(7)
            for speaker in ("a", "b", "c", "d"):
                folder = clean_dir / speaker
                folder.mkdir(parents=True)
                for utterance in (0, 1):
                    t = np.arange(1200) / 8000
                    mono = 0.7 * np.sin(2 * np.pi * (170 + 30 * utterance) * t)
                    stereo = np.stack((mono, 0.5 * mono), axis=1)
                    sf.write(folder / f"{utterance}.wav", stereo, 8000, subtype="FLOAT")
            noise_dir.mkdir()
            sf.write(noise_dir / "white.wav", rng.standard_normal(5500).astype(np.float32), 22050, subtype="FLOAT")
            (clean_dir / "bad.wav").write_bytes(b"not a WAV file")
            output = root / "prepared"
            counts = prepare_dataset(clean_dir, noise_dir, output, seed=19,
                                     validation_fraction=0.25, test_fraction=0.25)
            self.assertEqual(counts, {"train": 4, "validation": 10, "test": 10})
            manifests = {}
            for split in ("train", "validation", "test"):
                with (output / f"{split}.csv").open(newline="") as handle:
                    manifests[split] = list(csv.DictReader(handle))
                self.assertEqual(len(manifests[split]), counts[split])
            speakers = [{row["speaker"] for row in manifests[split]}
                        for split in ("train", "validation", "test")]
            self.assertTrue(speakers[0].isdisjoint(speakers[1]))
            self.assertTrue(speakers[0].isdisjoint(speakers[2]))
            self.assertTrue(speakers[1].isdisjoint(speakers[2]))
            for row in manifests["train"]:
                self.assertEqual(row["mixture_path"], "")
                self.assertEqual(row["snr_db"], "")
                self.assertEqual(sf.info(row["clean_path"]).samplerate, 16000)
                self.assertEqual(sf.info(row["noise_path"]).samplerate, 16000)
                self.assertEqual(sf.info(row["clean_path"]).channels, 1)
            training_row = read_manifest(output / "train.csv")[0]
            config = Config()
            first = make_example(training_row, config, Bands(config), np.random.default_rng(12))[0]
            repeated = make_example(training_row, config, Bands(config), np.random.default_rng(12))[0]
            different = make_example(training_row, config, Bands(config), np.random.default_rng(13))[0]
            self.assertTrue(np.array_equal(first.numpy(), repeated.numpy()))
            self.assertFalse(np.array_equal(first.numpy(), different.numpy()))
            for split in ("validation", "test"):
                by_speaker = {}
                for row in manifests[split]:
                    by_speaker.setdefault(row["speaker"], set()).add(float(row["snr_db"]))
                    speech = load_audio(row["clean_path"])
                    mixture = load_audio(row["mixture_path"])
                    disturbance = mixture.astype(np.float64) - speech.astype(np.float64)
                    measured = 10 * np.log10(np.mean(speech.astype(np.float64) ** 2)
                                             / np.mean(disturbance ** 2))
                    self.assertLess(abs(measured - float(row["snr_db"])), 0.1)
                    self.assertLessEqual(np.max(np.abs(mixture)), 0.99 + 1e-6)
                    self.assertEqual(sf.info(row["mixture_path"]).samplerate, 16000)
                for snrs in by_speaker.values():
                    self.assertEqual(snrs, {-5.0, 0.0, 5.0, 10.0, 15.0})
            first_manifest = (output / "test.csv").read_bytes()
            first_mix = load_audio(manifests["test"][0]["mixture_path"])
            prepare_dataset(clean_dir, noise_dir, output, seed=19,
                            validation_fraction=0.25, test_fraction=0.25)
            self.assertEqual(first_manifest, (output / "test.csv").read_bytes())
            np.testing.assert_array_equal(first_mix, load_audio(manifests["test"][0]["mixture_path"]))


if __name__ == "__main__":
    unittest.main()
