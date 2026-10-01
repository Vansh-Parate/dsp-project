"""Final experiment tables and measured compute fields."""

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.run_experiments import (_benchmark, _profile_network, _tables,
                                     METHOD_NAMES, PER_UTTERANCE_COLUMNS, SNRS)
from uahsn.inference import UAHSNEnhancer
from uahsn.model import UAHSN, parameter_count


class TestRunner(unittest.TestCase):
    def test_tables_group_means_and_missing_nan(self) -> None:
        rows = []
        for index, method in enumerate(METHOD_NAMES):
            for snr in SNRS:
                rows.append({"utterance": f"{snr}_{index}", "noise_type": "white",
                             "snr_db": snr, "method": method, "sisdr": float(snr + index),
                             "sisdr_improvement": float(index), "stoi": float("nan") if snr == -5 else 0.5 + index / 100,
                             "pesq": 2 + index / 10, "V_G": index / 10,
                             "RPD": float(index), "rho_alias": index / 100,
                             "NR": float(index * 2)})
        compute = [{"Method": name, "params": 0, "thop_macs_per_second": float("nan"),
                    "analytic_macs_per_second": float("nan"), "mean_ms_per_frame": 1.0,
                    "p99_9_ms_per_frame": 2.0, "max_ms_per_frame": 3.0,
                    "RTF": 0.1, "algorithmic_latency_ms": 10.0,
                    "frames_timed": 1000} for name in METHOD_NAMES]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            warnings = []
            _tables(rows, METHOD_NAMES, compute, root, warnings)
            for filename in ("table1_sisdr.csv", "table2_stoi.csv", "table3_pesq.csv",
                             "table5_operator_metrics.csv", "table6_noise_type.csv",
                             "table7_compute.csv", "summary.md"):
                self.assertTrue((root / filename).exists())
            with (root / "table1_sisdr.csv").open(newline="") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(reader.fieldnames, ["Method", "-5 dB", "0 dB", "5 dB", "10 dB", "15 dB"])
                table1 = list(reader)
            self.assertEqual(float(table1[4]["5 dB"]), 9.0)
            with (root / "table2_stoi.csv").open(newline="") as handle:
                table2 = list(csv.DictReader(handle))
            self.assertEqual(table2[0]["-5 dB"], "NaN")
            with (root / "table6_noise_type.csv").open(newline="") as handle:
                table6 = list(csv.DictReader(handle))
            self.assertEqual(len(table6), 5)
            self.assertEqual(float(table6[4]["SI-SDR Improvement (dB)"]), 4.0)
            self.assertEqual(table6[4]["N"], "1")
            self.assertIn("NaN", (root / "summary.md").read_text())
            self.assertTrue(any("SNR table cells" in warning for warning in warnings))
        self.assertEqual(len(PER_UTTERANCE_COLUMNS), 21)

    def test_model_profile_and_frame_benchmark(self) -> None:
        model = UAHSN()
        method = UAHSNEnhancer(model, sigma0=1.0)
        warnings = []
        measured, analytic = _profile_network(method, warnings)
        self.assertTrue(np.isfinite(measured))
        self.assertGreater(measured, 0)
        self.assertEqual(analytic, 5_939_200)
        self.assertEqual(parameter_count(model), 60_289)
        audio = np.random.default_rng(4).standard_normal(3200).astype(np.float32) * 0.05
        result, timing = _benchmark(method, audio, repeats=2)
        self.assertEqual(result.audio.shape, audio.shape)
        self.assertEqual(len(timing["frame_times"]), 40)
        self.assertGreater(timing["mean_ms_per_frame"], 0)
        self.assertGreaterEqual(timing["max_ms_per_frame"], timing["p99_9_ms_per_frame"])
        self.assertGreater(timing["RTF"], 0)


if __name__ == "__main__":
    unittest.main()
