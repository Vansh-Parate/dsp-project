"""Utterance-level training with deterministic ordering and validation sigma0."""

import json
import csv
import random
from pathlib import Path

import numpy as np
import torch

from .audio import load_audio, mix_components, random_noise_segment
from .bands import Bands
from .config import Config
from .features import extract_features
from .losses import enhancement_loss, ideal_ratio_gain
from .model import UAHSN
from .stft import stft


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def read_manifest(path: str | Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        if Path(path).suffix.lower() == ".csv":
            rows = list(csv.DictReader(handle))
        else:
            rows = [json.loads(line) for line in handle if line.strip()]
    if not rows or any(not (("clean_path" in row and "noise_path" in row) or
                            ("clean" in row and "noisy" in row)) for row in rows):
        raise ValueError("Manifest needs clean/noise paths or legacy clean/noisy paths")
    return rows


def make_example(
    row: dict[str, str], config: Config, bands: Bands,
    rng: np.random.Generator | None = None,
) -> tuple[torch.Tensor, ...]:
    clean = load_audio(row.get("clean_path", row.get("clean", "")), config.sample_rate)
    if row.get("mixture_path"):
        noisy = load_audio(row["mixture_path"], config.sample_rate)
    elif row.get("split") == "train":
        if rng is None:
            raise ValueError("Dynamic training mixtures require a seeded random generator")
        noise = load_audio(row["noise_path"], config.sample_rate)
        segment = random_noise_segment(noise, len(clean), rng)
        snr_db = float(rng.uniform(-5, 20))
        clean, _, noisy = mix_components(clean, segment, snr_db)
    else:
        noisy = load_audio(row["noisy"], config.sample_rate)
    if len(clean) != len(noisy):
        raise ValueError("Clean and noisy pair lengths differ")
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


def validation_sigma0(model: UAHSN, rows: list[dict[str, str]], config: Config, bands: Bands) -> float:
    model.eval()
    stds = []
    with torch.no_grad():
        for row in rows:
            features = make_example(row, config, bands)[0]
            _, log_variance, _ = model(features)
            stds.append(torch.exp(0.5 * log_variance).reshape(-1).numpy())
    return float(np.median(np.concatenate(stds)))


def train_model(
    train_manifest: str | Path,
    validation_manifest: str | Path,
    checkpoint_path: str | Path,
    epochs: int = 1,
    learning_rate: float = 1e-3,
    config: Config = Config(),
) -> dict[str, float]:
    set_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    train_rows = read_manifest(train_manifest)
    validation_rows = read_manifest(validation_manifest)
    bands = Bands(config)
    weights = torch.tensor(bands.weights, dtype=torch.float32)
    model = UAHSN()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history: dict[str, float] = {}
    for epoch in range(epochs):
        model.train()
        losses = []
        for index in rng.permutation(len(train_rows)):
            features, target, noisy_mag, clean_mag = make_example(train_rows[index], config, bands, rng)
            optimizer.zero_grad()
            gain, log_variance, _ = model(features)
            loss, _ = enhancement_loss(gain, log_variance, target, noisy_mag, clean_mag, weights, config)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            validation_losses = []
            for row in validation_rows:
                features, target, noisy_mag, clean_mag = make_example(row, config, bands)
                gain, log_variance, _ = model(features)
                loss, _ = enhancement_loss(gain, log_variance, target, noisy_mag, clean_mag, weights, config)
                validation_losses.append(float(loss))
        history = {"epoch": float(epoch + 1), "train_loss": float(np.mean(losses)),
                   "validation_loss": float(np.mean(validation_losses))}
        print(json.dumps(history))
    sigma0 = validation_sigma0(model, validation_rows, config, bands)
    checkpoint = Path(checkpoint_path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "sigma0": sigma0,
                "config": config.__dict__, "history": history}, checkpoint)
    return {**history, "sigma0": sigma0}
