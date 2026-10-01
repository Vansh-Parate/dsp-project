"""Deterministic, crop-batched UA-HSN training and validation selection."""

import csv
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .audio import load_audio, mix_components, random_noise_segment
from .bands import Bands
from .config import Config
from .features import extract_features
from .inference import smooth_band_gains
from .losses import enhancement_loss, ideal_ratio_gain
from .metrics import si_sdr
from .model import UAHSN
from .stft import istft, stft


CROP_SECONDS = 3
CURVE_FIELDS = ("epoch", "train_loss", "validation_loss", "validation_si_sdr_db",
                "validation_sigma0", "learning_rate", "is_best")


def set_seed(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    return torch.device(device)


def read_manifest(path: str | Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        rows = (list(csv.DictReader(handle)) if Path(path).suffix.lower() == ".csv"
                else [json.loads(line) for line in handle if line.strip()])
    if not rows or any(not (("clean_path" in row and "noise_path" in row) or
                            ("clean" in row and "noisy" in row)) for row in rows):
        raise ValueError("Manifest needs clean/noise paths or legacy clean/noisy paths")
    return rows


def load_pair(
    row: dict[str, str], config: Config, rng: np.random.Generator | None = None,
    crop_samples: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Training rows draw a new crop, noise excerpt, and uniform SNR on each call."""
    clean = load_audio(row.get("clean_path", row.get("clean", "")), config.sample_rate)
    start = 0
    if crop_samples is not None:
        if rng is None:
            raise ValueError("Random crops require a seeded generator")
        if len(clean) > crop_samples:
            start = int(rng.integers(0, len(clean) - crop_samples + 1))
        crop = clean[start:start + crop_samples]
        clean = np.pad(crop, (0, max(0, crop_samples - len(crop))))
    if row.get("mixture_path"):
        noisy = load_audio(row["mixture_path"], config.sample_rate)
        if crop_samples is not None:
            noisy = np.pad(noisy[start:start + crop_samples],
                           (0, max(0, crop_samples - len(noisy[start:start + crop_samples]))))
    elif row.get("split") == "train":
        if rng is None:
            raise ValueError("Dynamic training mixtures require a seeded generator")
        noise = load_audio(row["noise_path"], config.sample_rate)
        segment = random_noise_segment(noise, len(clean), rng)
        snr_db = float(rng.uniform(-5, 20))
        clean, _, noisy = mix_components(clean, segment, snr_db)
    else:
        noisy = load_audio(row["noisy"], config.sample_rate)
        if crop_samples is not None:
            noisy = np.pad(noisy[start:start + crop_samples],
                           (0, max(0, crop_samples - len(noisy[start:start + crop_samples]))))
    if len(clean) != len(noisy):
        raise ValueError("Clean and noisy pair lengths differ")
    return clean, noisy


def example_from_signals(
    clean: np.ndarray, noisy: np.ndarray, config: Config, bands: Bands,
) -> tuple[torch.Tensor, ...]:
    clean_spec = stft(clean, config)
    noisy_spec = stft(noisy, config)
    features = extract_features(noisy_spec, config)
    clean_mag = np.abs(clean_spec).astype(np.float32)
    noisy_mag = np.abs(noisy_spec).astype(np.float32)
    clean_energy = bands.energy(clean_mag ** 2).astype(np.float32)
    noisy_energy = bands.energy(noisy_mag ** 2).astype(np.float32)
    target = ideal_ratio_gain(torch.from_numpy(clean_energy), torch.from_numpy(noisy_energy), config.epsilon)
    return (torch.from_numpy(features)[None], target[None],
            torch.from_numpy(noisy_mag)[None], torch.from_numpy(clean_mag)[None])


def make_example(
    row: dict[str, str], config: Config, bands: Bands,
    rng: np.random.Generator | None = None, crop_samples: int | None = None,
) -> tuple[torch.Tensor, ...]:
    clean, noisy = load_pair(row, config, rng, crop_samples)
    return example_from_signals(clean, noisy, config, bands)


def calculate_feature_stats(
    rows: list[dict[str, str]], config: Config, crop_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """One deterministic dynamic mixture per training row; validation is never read."""
    bands = Bands(config)
    total = np.zeros(96, dtype=np.float64)
    squared = np.zeros(96, dtype=np.float64)
    frames = 0
    for index, row in enumerate(rows):
        rng = np.random.default_rng(np.random.SeedSequence([seed, 0, index]))
        features = make_example(row, config, bands, rng, crop_samples)[0][0].numpy()
        total += features.sum(axis=0, dtype=np.float64)
        squared += np.square(features.astype(np.float64)).sum(axis=0)
        frames += len(features)
    mean = total / frames
    std = np.sqrt(np.maximum(squared / frames - mean ** 2, 0))
    return mean.astype(np.float32), np.maximum(std, 1e-5).astype(np.float32)


class TrainingCrops(Dataset[tuple[torch.Tensor, ...]]):
    def __init__(
        self, rows: list[dict[str, str]], config: Config,
        mean: np.ndarray, std: np.ndarray, seed: int,
    ) -> None:
        self.rows, self.config = rows, config
        self.bands = Bands(config)
        self.mean = torch.from_numpy(mean)
        self.std = torch.from_numpy(std)
        self.seed = seed
        self.epoch = 0
        self.crop_samples = CROP_SECONDS * config.sample_rate

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch + 1, index]))
        features, target, noisy_mag, clean_mag = make_example(
            self.rows[index], self.config, self.bands, rng, self.crop_samples,
        )
        return ((features[0] - self.mean) / self.std,
                target[0], noisy_mag[0], clean_mag[0])


def validation_sigma0(
    model: UAHSN, rows: list[dict[str, str]], config: Config,
    bands: Bands, mean: np.ndarray, std: np.ndarray, device: torch.device,
) -> float:
    model.eval()
    deviations = []
    with torch.no_grad():
        for row in rows:
            features = make_example(row, config, bands)[0]
            normalized = (features - torch.from_numpy(mean)) / torch.from_numpy(std)
            _, log_variance, _ = model(normalized.to(device))
            deviations.append(torch.exp(0.5 * log_variance.clamp(-40, 40)).reshape(-1).cpu().numpy())
    return float(np.median(np.concatenate(deviations)))


def validate(
    model: UAHSN, rows: list[dict[str, str]], config: Config,
    bands: Bands, mean: np.ndarray, std: np.ndarray, device: torch.device,
    weights: torch.Tensor,
) -> tuple[float, float, float]:
    """Calibrate provisional sigma0, then score the actual smoothed output."""
    sigma0 = validation_sigma0(model, rows, config, bands, mean, std, device)
    model.eval()
    losses, scores = [], []
    mean_tensor = torch.from_numpy(mean)
    std_tensor = torch.from_numpy(std)
    with torch.no_grad():
        for row in rows:
            clean, noisy = load_pair(row, config)
            features, target, noisy_mag, clean_mag = example_from_signals(clean, noisy, config, bands)
            normalized = ((features - mean_tensor) / std_tensor).to(device)
            gain, log_variance, _ = model(normalized)
            loss, _ = enhancement_loss(gain, log_variance, target.to(device),
                                       noisy_mag.to(device), clean_mag.to(device), weights, config)
            losses.append(float(loss))
            predicted = gain[0].cpu().numpy()
            predicted_std = torch.exp(0.5 * log_variance[0].clamp(-40, 40)).cpu().numpy()
            band_gain = smooth_band_gains(predicted, predicted_std, sigma0, config)
            bin_gain = np.maximum(config.gain_floor, bands.interpolate(band_gain))
            enhanced = istft(stft(noisy, config) * bin_gain, len(noisy), config)
            scores.append(si_sdr(clean, enhanced))
    return float(np.mean(losses)), float(np.mean(scores)), sigma0


def _write_curves(path: Path, history: list[dict[str, float | int | bool]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CURVE_FIELDS)
        writer.writeheader()
        writer.writerows(history)


def train_model(
    train_manifest: str | Path, validation_manifest: str | Path,
    checkpoint_path: str | Path, epochs: int = 30, learning_rate: float = 1e-3,
    config: Config = Config(), batch_size: int = 16, max_train_files: int | None = None,
    device: str = "cpu", curves_csv: str | Path | None = None,
    early_stopping_patience: int = 5,
) -> dict[str, float | int | str]:
    if epochs < 1 or batch_size < 1 or early_stopping_patience < 1:
        raise ValueError("epochs, batch size, and early stopping patience must be positive")
    if max_train_files is not None and max_train_files < 1:
        raise ValueError("max_train_files must be positive")
    set_seed(config.seed)
    torch_device = resolve_device(device)
    train_rows = read_manifest(train_manifest)
    validation_rows = read_manifest(validation_manifest)
    if any(row.get("split", "train") != "train" for row in train_rows):
        raise ValueError("Training manifest includes non-training rows")
    if any(row.get("split", "validation") != "validation" for row in validation_rows):
        raise ValueError("Validation manifest includes non-validation rows; test data cannot tune the model")
    if max_train_files is not None:
        selection = np.random.default_rng(config.seed).permutation(len(train_rows))[:max_train_files]
        train_rows = [train_rows[int(index)] for index in selection]
    crop_samples = CROP_SECONDS * config.sample_rate
    mean, std = calculate_feature_stats(train_rows, config, crop_samples, config.seed)
    dataset = TrainingCrops(train_rows, config, mean, std, config.seed)
    loader_generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        generator=loader_generator, num_workers=0)
    bands = Bands(config)
    weights = torch.tensor(bands.weights, dtype=torch.float32, device=torch_device)
    model = UAHSN().to(torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    checkpoint = Path(checkpoint_path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    curve_path = Path(curves_csv) if curves_csv is not None else checkpoint.with_suffix(".curves.csv")
    history: list[dict[str, float | int | bool]] = []
    best_score = -np.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    for epoch in range(1, epochs + 1):
        dataset.epoch = epoch - 1
        model.train()
        training_losses = []
        for features, target, noisy_mag, clean_mag in loader:
            features, target = features.to(torch_device), target.to(torch_device)
            noisy_mag, clean_mag = noisy_mag.to(torch_device), clean_mag.to(torch_device)
            optimizer.zero_grad(set_to_none=True)
            gain, log_variance, _ = model(features)
            loss, _ = enhancement_loss(gain, log_variance, target,
                                       noisy_mag, clean_mag, weights, config)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            training_losses.append(float(loss.detach()))
        val_loss, val_si_sdr, provisional_sigma0 = validate(
            model, validation_rows, config, bands, mean, std, torch_device, weights,
        )
        scheduler.step(val_si_sdr)
        improved = val_si_sdr > best_score
        if improved:
            best_score, best_epoch, stale_epochs = val_si_sdr, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            torch.save({"model": best_state, "feature_mean": mean, "feature_std": std,
                        "sigma0": provisional_sigma0, "config": config.__dict__,
                        "best_epoch": best_epoch, "best_validation_si_sdr_db": best_score}, checkpoint)
        else:
            stale_epochs += 1
        curve = {"epoch": epoch, "train_loss": float(np.mean(training_losses)),
                 "validation_loss": val_loss, "validation_si_sdr_db": val_si_sdr,
                 "validation_sigma0": provisional_sigma0,
                 "learning_rate": float(optimizer.param_groups[0]["lr"]), "is_best": improved}
        history.append(curve)
        _write_curves(curve_path, history)
        print(json.dumps(curve))
        if stale_epochs >= early_stopping_patience:
            break
    if best_state is None:
        raise RuntimeError("No finite validation SI-SDR was available to select a checkpoint")
    model.load_state_dict(best_state)
    # Recompute with the selected model only after training has ended.
    sigma0 = validation_sigma0(model, validation_rows, config, bands, mean, std, torch_device)
    torch.save({"model": best_state, "feature_mean": mean, "feature_std": std,
                "sigma0": sigma0, "config": config.__dict__, "history": history,
                "best_epoch": best_epoch, "best_validation_si_sdr_db": best_score}, checkpoint)
    return {"best_epoch": best_epoch, "best_validation_si_sdr_db": best_score,
            "sigma0": sigma0, "checkpoint": str(checkpoint), "curves_csv": str(curve_path),
            "train_files": len(train_rows)}


def two_batch_overfit_test(
    device: str = "cpu", seed: int = 0, max_rounds: int = 40,
) -> dict[str, float | int]:
    """Fit two fixed synthetic 3-second batches before any long training."""
    set_seed(seed)
    torch_device = resolve_device(device)
    config = Config(seed=seed)
    bands = Bands(config)
    rng = np.random.default_rng(seed)
    length = CROP_SECONDS * config.sample_rate
    t = np.arange(length, dtype=np.float64) / config.sample_rate
    examples = []
    for index in range(4):
        envelope = 0.5 + 0.5 * np.sin(2 * np.pi * (1.3 + 0.2 * index) * t) ** 2
        clean = (envelope * (0.2 * np.sin(2 * np.pi * (170 + 40 * index) * t)
                             + 0.08 * np.sin(2 * np.pi * (350 + 50 * index) * t))).astype(np.float32)
        noise = rng.standard_normal(length).astype(np.float32)
        clean, _, noisy = mix_components(clean, noise, float(index * 3 - 3))
        examples.append(example_from_signals(clean, noisy, config, bands))
    all_features = torch.cat([example[0] for example in examples], dim=0)
    mean = all_features.mean(dim=(0, 1))
    std = all_features.std(dim=(0, 1), unbiased=False).clamp_min(1e-5)
    batches = []
    for start in (0, 2):
        batch = tuple(torch.cat([examples[index][field] for index in (start, start + 1)], dim=0)
                      for field in range(4))
        normalized = (batch[0] - mean) / std
        batches.append((normalized.to(torch_device), *(item.to(torch_device) for item in batch[1:])))
    weights = torch.tensor(bands.weights, dtype=torch.float32, device=torch_device)
    model = UAHSN().to(torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    def measure() -> tuple[float, float]:
        model.eval()
        totals, magnitudes = [], []
        with torch.no_grad():
            for features, target, noisy_mag, clean_mag in batches:
                gain, log_variance, _ = model(features)
                loss, pieces = enhancement_loss(gain, log_variance, target,
                                                noisy_mag, clean_mag, weights, config)
                totals.append(float(loss))
                magnitudes.append(float(pieces["mag"]))
        return float(np.mean(totals)), float(np.mean(magnitudes))

    initial, initial_mag = measure()
    final, final_mag = initial, initial_mag
    rounds = 0
    for rounds in range(1, max_rounds + 1):
        model.train()
        for features, target, noisy_mag, clean_mag in batches:
            optimizer.zero_grad(set_to_none=True)
            gain, log_variance, _ = model(features)
            loss, _ = enhancement_loss(gain, log_variance, target,
                                       noisy_mag, clean_mag, weights, config)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        if rounds % 5 == 0 or rounds == max_rounds:
            final, final_mag = measure()
            if final <= 0.7 * initial and final_mag < initial_mag:
                break
    if not (final <= 0.7 * initial and final_mag < initial_mag):
        raise AssertionError(
            f"Two-batch overfit failed: total {initial:.4f} -> {final:.4f}, "
            f"magnitude {initial_mag:.4f} -> {final_mag:.4f}"
        )
    return {"initial_loss": initial, "final_loss": final,
            "initial_magnitude_loss": initial_mag, "final_magnitude_loss": final_mag,
            "rounds": rounds, "updates": rounds * len(batches)}
