"""Classical formula checks and FIR overlap-add experiments."""

import unittest

import numpy as np

from uahsn.baselines import (DecisionDirectedWiener, ReconstructedLinearPhaseBaseline,
                             SpectralSubtraction, symmetric_fir_from_gain)
from uahsn.config import Config
from uahsn.stft import istft, stft


class TestBaselines(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(23)
        t = np.arange(4096) / 16000
        self.audio = (0.2 * np.sin(2 * np.pi * 240 * t)
                      + 0.04 * rng.standard_normal(len(t))).astype(np.float32)

    def test_spectral_subtraction_gain_psd_and_phase(self) -> None:
        config = Config()
        unity = SpectralSubtraction(config, oversubtraction=0).enhance(self.audio)
        ordinary = SpectralSubtraction(config, oversubtraction=1).enhance(self.audio)
        stronger = SpectralSubtraction(config, oversubtraction=2).enhance(self.audio)
        np.testing.assert_allclose(unity.gains, 1, atol=0)
        np.testing.assert_allclose(unity.audio, self.audio, atol=2e-6)
        np.testing.assert_allclose(ordinary.noise_psd, stronger.noise_psd)
        self.assertTrue(np.all(stronger.gains <= ordinary.gains + 1e-12))
        self.assertGreaterEqual(float(stronger.gains.min()), config.gain_floor - 1e-12)
        self.assertLessEqual(float(stronger.gains.max()), 1.0)
        original_spec = stft(self.audio)
        filtered_spec = original_spec * ordinary.gains
        np.testing.assert_allclose(np.angle(filtered_spec[ordinary.gains > 0]),
                                   np.angle(original_spec[ordinary.gains > 0]), atol=1e-12)

    def test_wiener_decision_directed_formula(self) -> None:
        config = Config()
        result = DecisionDirectedWiener(config).enhance(self.audio)
        spectra = stft(self.audio)
        previous_gain = np.ones(257)
        previous_gamma = np.ones(257)
        expected = []
        for spectrum, psd in zip(spectra, result.noise_psd):
            gamma = np.abs(spectrum) ** 2 / psd
            xi = config.dd_beta * previous_gain ** 2 * previous_gamma + (1 - config.dd_beta) * np.maximum(gamma - 1, 0)
            gain = xi / (1 + xi)
            expected.append(gain)
            previous_gain, previous_gamma = gain, gamma
        np.testing.assert_allclose(result.gains, expected, atol=1e-12)
        self.assertGreaterEqual(float(result.gains.min()), 0)
        self.assertLessEqual(float(result.gains.max()), 1)

    def test_fir_lengths_frames_and_overlap_add(self) -> None:
        for frame_length in (256, 320, 512):
            config = Config(win_length=frame_length, hop_length=frame_length // 2)
            np.testing.assert_allclose(istft(stft(self.audio, config), len(self.audio), config),
                                       self.audio, atol=2e-6)
            for taps in (31, 63, 127):
                with self.subTest(frame_length=frame_length, taps=taps):
                    method = ReconstructedLinearPhaseBaseline(config, taps, oversubtraction=0)
                    result = method.enhance(self.audio)
                    delay = (taps - 1) // 2
                    self.assertEqual(method.name, "reconstructed_linear_phase_baseline")
                    self.assertEqual(result.audio.shape, self.audio.shape)
                    self.assertEqual(result.gains.shape, stft(self.audio, config).shape)
                    self.assertEqual(result.noise_psd.shape, result.gains.shape)
                    self.assertEqual(result.fir_coefficients.shape, (len(result.gains), taps))
                    self.assertTrue(np.isfinite(result.audio).all())
                    self.assertTrue(np.isfinite(result.gains).all())
                    self.assertTrue(np.isfinite(result.noise_psd).all())
                    self.assertGreaterEqual(float(result.gains.min()), 0)
                    self.assertLessEqual(float(result.gains.max()), 1 + 1e-12)
                    np.testing.assert_allclose(result.fir_coefficients,
                                               result.fir_coefficients[:, ::-1], atol=1e-12)
                    np.testing.assert_allclose(result.audio[:delay], 0, atol=2e-6)
                    np.testing.assert_allclose(result.audio[delay:], self.audio[:-delay], atol=2e-6)
                    self.assertGreater(result.runtime_seconds, 0)
                    ordinary = ReconstructedLinearPhaseBaseline(config, taps).enhance(self.audio)
                    self.assertTrue(np.isfinite(ordinary.audio).all())
                    self.assertTrue(np.isfinite(ordinary.gains).all())
                    self.assertGreaterEqual(float(ordinary.gains.min()), 0)
                    self.assertLessEqual(float(ordinary.gains.max()), 1 + 1e-6)

    def test_fir_construction_symmetry(self) -> None:
        rng = np.random.default_rng(8)
        gain = rng.uniform(0.1, 1.0, 257)
        for taps in (31, 63, 127):
            coefficients = symmetric_fir_from_gain(gain, taps, 512)
            np.testing.assert_allclose(coefficients, coefficients[::-1], atol=1e-12)
            self.assertEqual(len(coefficients), taps)
            response = np.fft.rfft(coefficients, n=512)
            delay = (taps - 1) // 2
            zero_phase = np.real(response * np.exp(2j * np.pi * np.arange(257) * delay / 512))
            self.assertGreaterEqual(float(zero_phase.min()), float(gain.min()) - 1e-10)
            self.assertLessEqual(float(zero_phase.max()), float(gain.max()) + 1e-10)


if __name__ == "__main__":
    unittest.main()
