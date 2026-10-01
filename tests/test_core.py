"""Signal, architecture, and shared-interface checks."""

import unittest

import numpy as np
import torch

from uahsn.bands import Bands
from uahsn.baselines import Noisy, SpectralSubtraction, DecisionDirectedWiener, ReconstructedLinearPhaseBaseline
from uahsn.config import Config
from uahsn.features import extract_features
from uahsn.inference import UAHSNEnhancer
from uahsn.losses import enhancement_loss
from uahsn.model import UAHSN, parameter_count
from uahsn.stft import istft, stft


class TestCore(unittest.TestCase):
    def setUp(self) -> None:
        self.config = Config()

    def test_stft_reconstruction(self) -> None:
        rng = np.random.default_rng(0)
        for length in (160, 321, 16000):
            audio = rng.standard_normal(length).astype(np.float32)
            reconstructed = istft(stft(audio), length)
            np.testing.assert_allclose(reconstructed, audio, atol=2e-6)

    def test_band_partition(self) -> None:
        bands = Bands()
        self.assertEqual(bands.weights.shape, (257, 32))
        np.testing.assert_allclose(bands.weights.sum(axis=1), 1, atol=1e-12)
        self.assertTrue(np.all(bands.weights >= 0))
        self.assertTrue(np.all(bands.weights <= 1))
        self.assertTrue(np.all(bands.analysis_weights.sum(axis=0) > 0.999999))

    def test_model_dimensions_and_parameter_count(self) -> None:
        model = UAHSN()
        count = parameter_count(model)
        print(f"UA-HSN PyTorch parameter count: {count}")
        self.assertEqual(count, 60_289)
        gain, log_variance, state = model(torch.zeros(2, 5, 96))
        self.assertEqual(tuple(gain.shape), (2, 5, 32))
        self.assertEqual(tuple(log_variance.shape), (2, 5, 32))
        self.assertEqual(tuple(state[0].shape), (1, 2, 64))
        self.assertEqual(tuple(state[1].shape), (1, 2, 64))
        self.assertTrue(torch.all((gain > 0) & (gain < 1)))

    def test_feature_and_method_output_shapes(self) -> None:
        rng = np.random.default_rng(1)
        audio = (0.1 * np.sin(2 * np.pi * 440 * np.arange(800) / 16000)
                 + 0.02 * rng.standard_normal(800)).astype(np.float32)
        spectra = stft(audio)
        self.assertEqual(extract_features(spectra).shape, (6, 96))
        torch.manual_seed(0)
        methods = [Noisy(), SpectralSubtraction(), DecisionDirectedWiener(),
                   ReconstructedLinearPhaseBaseline(), UAHSNEnhancer(UAHSN(), 1.0)]
        for method in methods:
            with self.subTest(method=method.name):
                result = method.enhance(audio)
                self.assertEqual(result.audio.shape, audio.shape)
                self.assertEqual(result.gains.shape, (6, 257))
                self.assertEqual(result.noise_psd.shape, (6, 257))
                self.assertTrue(np.isfinite(result.audio).all())
                self.assertTrue(np.isfinite(result.gains).all())
                self.assertTrue(np.isfinite(result.noise_psd).all())
                self.assertGreater(result.runtime_seconds, 0)
        result = methods[-1].enhance(audio)
        self.assertEqual(result.band_gains.shape, (6, 32))
        self.assertEqual(result.standard_deviation.shape, (6, 32))

    def test_causality_of_features(self) -> None:
        rng = np.random.default_rng(2)
        original = rng.standard_normal(1600).astype(np.float32)
        modified = original.copy()
        modified[800:] *= 3
        a = extract_features(stft(original))
        b = extract_features(stft(modified))
        np.testing.assert_allclose(a[:5], b[:5], atol=1e-6)

    def test_temporal_consistency_uses_only_adjacent_noise_units(self) -> None:
        gain = torch.tensor([[[0.0] * 32, [1.0] * 32, [0.0] * 32]])
        target = torch.tensor([[[0.2] * 32, [0.2] * 32, [0.8] * 32]])
        magnitude = torch.ones(1, 3, 257)
        weights = torch.tensor(Bands().weights, dtype=torch.float32)
        _, parts = enhancement_loss(gain, torch.zeros_like(gain), target,
                                    magnitude, magnitude, weights)
        self.assertAlmostEqual(float(parts["tc"]), 1.0)


if __name__ == "__main__":
    unittest.main()
