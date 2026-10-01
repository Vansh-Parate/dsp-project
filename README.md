# UA-HSN research prototype

CPU-compatible, causal, single-channel 16 kHz speech enhancement. The five methods share `enhance(audio) -> EnhancementResult`, which contains enhanced audio and a per-frame, 257-bin gain operator. UA-HSN also returns smoothed band gains and predicted standard deviations.

## Signal path

- 320-sample periodic sqrt-Hann window, 160-sample hop, 512-point FFT; input is left padded by one hop. A zero block flushes the overlap-add tail. No frame reads beyond its current 160-sample hop.
- 32 ERB-rate-spaced triangular bands. Analysis weights sum to one within each band; synthesis weights sum to one across bands at each FFT bin.
- Noise estimation smooths power with `alpha_p=0.9` and takes the minimum over the current and seven completed 12-frame subwindows (up to 96 frames). The decision-directed Wiener recurrence uses its own previous Wiener gain, never the network gain.
- The network has Linear(96,64), PReLU, two GRU(64,64) layers, and separate 32-dimensional sigmoid-gain and log-variance heads. PyTorch parameter count: **60,289**.
- Training uses band ideal ratio gains, compressed magnitude exponent 0.3, Gaussian NLL, and temporal consistency only when both adjacent oracle gains are below 0.3. Loss weights are 1, 0.1, 0.1.
- `sigma0` is the median predicted standard deviation over the validation manifest and is saved in the checkpoint. Inference uses it for uncertainty-adaptive recursive gain smoothing.
- The `reconstructed_linear_phase_baseline` is a reconstruction from a prose description, **not an exact reproduction of an external paper**. Each spectral-subtraction gain is inverse transformed, circularly symmetrized, tapered with a triangular Fejer window to an odd-length Type-I FIR, delayed by `(L-1)/2`, and applied to hop blocks with FFT overlap-add. The Fejer window keeps the frequency response bounded by the original gain range. Defaults are `L=63` and a 320-sample frame; experiments also support `L=31,63,127` and 256-, 320-, and 512-sample frames.

## Run

Use Python 3.10+ with dependencies in `requirements.txt`:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m scripts.run_experiments --synthetic-smoke
```

The smoke test generates sine waves plus seeded white noise and writes audio and `.npz` gain arrays under `results/synthetic_smoke`. Its UA-HSN network is **untrained**, so the files validate execution only and are not research results.

For clean speech and noise WAV folders, prepare deterministic evaluation mixtures, train with dynamic SNRs, and evaluate:

```bash
python -m scripts.prepare_dataset --clean-dir data/clean --noise-dir data/noise --output-dir data/prepared --seed 0
python -m scripts.train_model --train-manifest data/prepared/train.csv --validation-manifest data/prepared/validation.csv --epochs 1 --seed 0 --checkpoint checkpoints/uahsn.pt
python -m scripts.evaluate --manifest data/prepared/test.csv --checkpoint checkpoints/uahsn.pt --output-dir results/evaluation
```

Evaluation saves each method's enhanced `.wav`, plus `.npz` arrays for per-frame gains, estimated noise PSD, runtime seconds, and real-time factor. The reconstructed baseline also saves each FIR coefficient vector. `metrics.json` contains SI-SDR, log-spectral distance, operator measures, and wall-clock timing. Spectral subtraction uses configurable `--oversubtraction`; the reconstructed baseline accepts `--linear-phase-filter-length` and `--linear-phase-frame-length`. These are measurement implementations, not claims of quality. A trained checkpoint requires real paired data; no dataset is bundled or downloaded.

`prepare_dataset` accepts stereo or mono input and resamples it to 16 kHz mono, writing float WAV files to avoid quantization changing the requested SNR. It reports and skips unreadable or silent files. Clean speech is split by speaker: the first subdirectory is treated as the speaker ID; flat names such as `p225_001.wav` use the prefix. For other layouts, pass `--speaker-regex '(?P<speaker>...)'` against the relative clean path. At least three distinct speaker groups are required. The CSV fields are `mixture_id,clean_path,noise_path,mixture_path,speaker,noise_type,snr_db,split,seed`. Training rows leave `mixture_path` and `snr_db` empty; each training epoch draws a new noise excerpt and a uniform SNR in `[-5, 20]` dB. Validation and test rows contain fixed mixtures at −5, 0, 5, 10, and 15 dB. For each mixture, power-based SNR scaling is applied exactly, then a single shared scale is applied to clean speech, scaled noise, and mixture if needed to prevent clipping.
