"""Known-signal checks for reference, operator, and statistical metrics."""

import unittest

import numpy as np

from uahsn.baselines import EnhancementResult, Noisy, ReconstructedLinearPhaseBaseline
from uahsn.config import Config
from uahsn.metrics import evaluate_pair, segmental_snr, si_sdr, stoi_score
from uahsn.operator_metrics import (aliasing_ratio, gain_variance, isolated_peak_count,
                                    noise_only_frames, noise_reduction_db, operator_metrics,
                                    shadow_filter_noise)
from uahsn.statistics import METRICS, summarize_results
from uahsn.stft import stft


class TestEvaluation(unittest.TestCase):
    def test_projection_and_segmental_clipping(self) -> None:
        t = np.arange(16000) / 16000
        clean = 0.2 * np.sin(2 * np.pi * 220 * t)
        noise = 0.02 * np.sin(2 * np.pi * 997 * t)
        self.assertAlmostEqual(si_sdr(clean, clean + noise), 20.0, delta=0.1)
        self.assertAlmostEqual(si_sdr(clean, 2 * (clean + noise)), 20.0, delta=0.1)
        self.assertAlmostEqual(segmental_snr(clean, clean), 35.0, delta=1e-8)
        self.assertAlmostEqual(segmental_snr(clean, clean + 100 * noise), -10.0, delta=1e-8)

    def test_stoi_and_wideband_pesq(self) -> None:
        rng = np.random.default_rng(2)
        t = np.arange(32000) / 16000
        clean = (0.25 * np.sin(2 * np.pi * 180 * t)
                 + 0.1 * np.sin(2 * np.pi * 580 * t)).astype(np.float32)
        degraded = clean + 0.03 * rng.standard_normal(len(clean)).astype(np.float32)
        values = evaluate_pair(clean, degraded)
        self.assertTrue(np.isfinite(values["stoi"]))
        self.assertIsNotNone(values["pesq_wb"])
        self.assertTrue(np.isfinite(values["pesq_wb"]))
        self.assertLessEqual(values["stoi"], 1.0 + 1e-6)
        self.assertIsNone(stoi_score(clean[:1600], degraded[:1600]))

    def test_noise_only_gain_variance_and_isolated_peak(self) -> None:
        clean = np.zeros(1600)
        clean[800:] = 0.1
        mask = noise_only_frames(clean)
        self.assertTrue(mask[:4].all())
        self.assertFalse(mask[-1])
        gains = np.ones((len(mask), 257))
        gains[2] = 0.5
        eligible = mask[1:]
        expected = np.mean(np.diff(gains, axis=0)[eligible] ** 2)
        self.assertAlmostEqual(gain_variance(gains, mask), float(expected))
        self.assertIsNone(gain_variance(gains, np.zeros_like(mask)))
        power = np.ones((5, 5))
        power[2, 2] = 10.0
        local_floor = 2 * np.ones_like(power)
        self.assertEqual(isolated_peak_count(power, local_floor, np.ones(5, dtype=bool)), 1)
        self.assertEqual(isolated_peak_count(power, local_floor, np.ones(5, dtype=bool), theta_db=12), 0)

    def test_alias_ratio_and_shadow_noise_reduction(self) -> None:
        rng = np.random.default_rng(3)
        noise = 0.1 * rng.standard_normal(3200).astype(np.float32)
        clean = np.zeros_like(noise)
        spectra = stft(noise)
        gains = np.full(spectra.shape, 0.5)
        result = EnhancementResult(np.zeros_like(noise), gains, np.ones_like(gains))
        self.assertAlmostEqual(aliasing_ratio(np.ones_like(gains)), 0.0, delta=1e-12)
        self.assertAlmostEqual(aliasing_ratio(gains), 0.0, delta=1e-12)
        self.assertAlmostEqual(noise_reduction_db(clean, noise, result), 10 * np.log10(4), delta=1e-5)
        random_gain = rng.uniform(0.1, 1, spectra.shape)
        self.assertGreater(aliasing_ratio(random_gain), 0)
        fir = ReconstructedLinearPhaseBaseline(Config(), taps=31, oversubtraction=0).enhance(noise)
        np.testing.assert_allclose(shadow_filter_noise(noise, fir), fir.audio, atol=2e-6)

    def test_operator_metrics_with_clean_reference_silence(self) -> None:
        rng = np.random.default_rng(6)
        clean = np.zeros(16000, dtype=np.float32)
        t = np.arange(8000) / 16000
        clean[8000:] = 0.2 * np.sin(2 * np.pi * 300 * t)
        mixture = clean + 0.03 * rng.standard_normal(len(clean)).astype(np.float32)
        result = Noisy().enhance(mixture)
        metrics = operator_metrics(result, clean, mixture)
        self.assertAlmostEqual(metrics["gain_variance"], 0.0)
        self.assertIsNotNone(metrics["residual_peak_density_per_second"])
        self.assertGreaterEqual(metrics["residual_peak_density_per_second"], 0)
        self.assertAlmostEqual(metrics["aliasing_ratio"], 0.0, delta=1e-12)
        self.assertAlmostEqual(metrics["noise_reduction_db"], 0.0, delta=1e-5)

    def test_bootstrap_and_paired_wilcoxon(self) -> None:
        records = []
        for i in range(5):
            for method, offset in (("uahsn", 2.0), ("spectral_subtraction", 0.0)):
                row = {"mixture_id": f"mix{i}", "method": method}
                row.update({metric: float(i + offset) for metric in METRICS})
                records.append(row)
        first = summarize_results(records, seed=9, bootstrap_samples=300)
        repeated = summarize_results(records, seed=9, bootstrap_samples=300)
        self.assertEqual(first, repeated)
        mean = first["summary"]["uahsn"]["si_sdr_db"]
        self.assertEqual(mean["n"], 5)
        self.assertAlmostEqual(mean["mean"], 4.0)
        self.assertLessEqual(mean["ci95"][0], 4.0)
        self.assertGreaterEqual(mean["ci95"][1], 4.0)
        comparison = first["paired_wilcoxon"]["spectral_subtraction"]["si_sdr_db"]
        self.assertEqual(comparison["n_pairs"], 5)
        self.assertIsNotNone(comparison["p_value"])


if __name__ == "__main__":
    unittest.main()
