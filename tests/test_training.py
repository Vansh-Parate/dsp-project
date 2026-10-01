"""Training crops, checkpoint calibration, and data-split guards."""

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from uahsn.config import Config
from uahsn.inference import UAHSNEnhancer
from uahsn.model import UAHSN
from uahsn.train import (TrainingCrops, calculate_feature_stats, read_manifest,
                         train_model, validation_sigma0)
from uahsn.bands import Bands


class TestTraining(unittest.TestCase):
    def test_crops_best_checkpoint_and_validation_only_sigma0(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sr = 16000
            t = np.arange(4000) / sr
            train_clean = (0.2 * np.sin(2 * np.pi * 210 * t)).astype(np.float32)
            validation_clean = (0.2 * np.sin(2 * np.pi * 270 * t)).astype(np.float32)
            noise = np.random.default_rng(7).standard_normal(60000).astype(np.float32)
            files = {"train_clean": train_clean, "validation_clean": validation_clean,
                     "noise": noise, "validation_mixture": validation_clean + 0.04 * noise[:len(t)]}
            for name, samples in files.items():
                sf.write(root / f"{name}.wav", samples, sr, subtype="FLOAT")
            fieldnames = ("mixture_id", "clean_path", "noise_path", "mixture_path",
                          "speaker", "noise_type", "snr_db", "split", "seed")
            train_row = {"mixture_id": "train_0", "clean_path": str(root / "train_clean.wav"),
                         "noise_path": str(root / "noise.wav"), "mixture_path": "",
                         "speaker": "a", "noise_type": "white", "snr_db": "",
                         "split": "train", "seed": "7"}
            validation_row = {"mixture_id": "validation_0",
                              "clean_path": str(root / "validation_clean.wav"),
                              "noise_path": str(root / "noise.wav"),
                              "mixture_path": str(root / "validation_mixture.wav"),
                              "speaker": "b", "noise_type": "white", "snr_db": "5",
                              "split": "validation", "seed": "8"}
            for name, row in (("train", train_row), ("validation", validation_row)):
                with (root / f"{name}.csv").open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerow(row)
            config = Config(seed=11)
            rows = read_manifest(root / "train.csv")
            mean, std = calculate_feature_stats(rows, config, 3 * sr, config.seed)
            dataset = TrainingCrops(rows, config, mean, std, config.seed)
            first = dataset[0][0]
            self.assertEqual(first.shape, (301, 96))
            self.assertTrue(torch.equal(first, dataset[0][0]))
            dataset.epoch = 1
            self.assertFalse(torch.equal(first, dataset[0][0]))
            checkpoint = root / "best.pt"
            result = train_model(root / "train.csv", root / "validation.csv", checkpoint,
                                 epochs=1, batch_size=16, max_train_files=1,
                                 config=config, device="cpu")
            self.assertEqual(result["best_epoch"], 1)
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            np.testing.assert_allclose(saved["feature_mean"], mean)
            np.testing.assert_allclose(saved["feature_std"], std)
            self.assertTrue(np.all(saved["feature_std"] > 0))
            model = UAHSN()
            model.load_state_dict(saved["model"])
            expected_sigma0 = validation_sigma0(model, read_manifest(root / "validation.csv"),
                                                config, Bands(config), mean, std, torch.device("cpu"))
            self.assertAlmostEqual(saved["sigma0"], expected_sigma0)
            self.assertTrue(np.isfinite(saved["best_validation_si_sdr_db"]))
            with checkpoint.with_suffix(".curves.csv").open(newline="") as handle:
                curves = list(csv.DictReader(handle))
            self.assertEqual(len(curves), 1)
            self.assertIn("validation_si_sdr_db", curves[0])
            enhancer = UAHSNEnhancer.from_checkpoint(checkpoint)
            self.assertTrue(np.all(enhancer.feature_std > 0))
            repeated_checkpoint = root / "repeated.pt"
            repeated = train_model(root / "train.csv", root / "validation.csv",
                                   repeated_checkpoint, epochs=1, batch_size=16,
                                   max_train_files=1, config=config, device="cpu")
            self.assertEqual(result["best_validation_si_sdr_db"],
                             repeated["best_validation_si_sdr_db"])
            repeated_saved = torch.load(repeated_checkpoint, map_location="cpu", weights_only=False)
            for key in saved["model"]:
                self.assertTrue(torch.equal(saved["model"][key], repeated_saved["model"][key]))
            validation_row["split"] = "test"
            with (root / "validation.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(validation_row)
            with self.assertRaisesRegex(ValueError, "test data cannot tune"):
                train_model(root / "train.csv", root / "validation.csv", root / "wrong.pt",
                            epochs=1, batch_size=1, config=config)


if __name__ == "__main__":
    unittest.main()
