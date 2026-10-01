# UA-HSN research prototype

CPU/CUDA-compatible, causal, single-channel 16 kHz speech enhancement. The five methods share `enhance(audio) -> EnhancementResult`, which contains enhanced audio and a per-frame, 257-bin gain operator. UA-HSN also returns smoothed band gains and predicted standard deviations.

## Signal path

- 320-sample periodic sqrt-Hann window, 160-sample hop, 512-point FFT; input is left padded by one hop. A zero block flushes the overlap-add tail. No frame reads beyond its current 160-sample hop.
- 32 ERB-rate-spaced triangular bands. Analysis weights sum to one within each band; synthesis weights sum to one across bands at each FFT bin.
- Noise estimation smooths power with `alpha_p=0.9` and takes the minimum over the current and seven completed 12-frame subwindows (up to 96 frames). The decision-directed Wiener recurrence uses its own previous Wiener gain, never the network gain.
- The network has Linear(96,64), PReLU, two GRU(64,64) layers, and separate 32-dimensional sigmoid-gain and log-variance heads. PyTorch parameter count: **60,289**.
- Training uses band ideal ratio gains, compressed magnitude exponent 0.3, Gaussian NLL, and temporal consistency only when both adjacent oracle gains are below 0.3. Loss weights are 1, 0.1, 0.1. It draws random 3-second training crops and dynamically mixes noise at a uniform SNR in `[-5,20]` dB.
- Training-only feature means and standard deviations are saved in the best checkpoint and applied at inference. `sigma0` is recomputed after training as the median predicted standard deviation over validation data using the selected model. Inference uses it for uncertainty-adaptive recursive gain smoothing.
- The `reconstructed_linear_phase_baseline` is a reconstruction from a prose description, **not an exact reproduction of an external paper**. Each spectral-subtraction gain is inverse transformed, circularly symmetrized, tapered with a triangular Fejer window to an odd-length Type-I FIR, delayed by `(L-1)/2`, and applied to hop blocks with FFT overlap-add. The Fejer window keeps the frequency response bounded by the original gain range. Defaults are `L=63` and a 320-sample frame; experiments also support `L=31,63,127` and 256-, 320-, and 512-sample frames.

## Run

Use Python 3.10+ with dependencies in `requirements.txt`:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m scripts.run_experiments --synthetic-smoke
python scripts/train_model.py --overfit-test --device cpu --seed 0
```

The smoke test generates sine waves plus seeded white noise and writes audio and `.npz` gain arrays under `results/synthetic_smoke`. Its UA-HSN network is **untrained**, so the files validate execution only and are not research results.

For clean speech and noise WAV folders, prepare deterministic evaluation mixtures, train with dynamic SNRs, and evaluate:

```bash
python -m scripts.prepare_dataset --clean-dir data/clean --noise-dir data/noise --output-dir data/prepared --seed 0
python scripts/train_model.py --manifest data/prepared/train.csv --val-manifest data/prepared/validation.csv --max-train-files 16 --epochs 2 --batch-size 16 --device cpu --seed 0 --checkpoint checkpoints/uahsn.pt
python scripts/train_model.py --manifest data/prepared/train.csv --val-manifest data/prepared/validation.csv --epochs 30 --batch-size 16 --device cuda --seed 0 --checkpoint checkpoints/uahsn.pt
python -m scripts.evaluate --manifest data/prepared/test.csv --checkpoint checkpoints/uahsn.pt --output-dir results/evaluation
```

Run the fixed two-batch overfit check before a long training run. The trainer uses Adam at `1e-3`, clips gradient norm at 5, schedules learning rate with `ReduceLROnPlateau`, and stops early when validation SI-SDR does not improve for five epochs. The default batch size is 16. Shorter speech files are zero-padded to 3 seconds for training crops. The best checkpoint is chosen by validation SI-SDR and stores feature mean, feature standard deviation, `sigma0`, model weights, and selected epoch. The epoch-by-epoch CSV is written beside the checkpoint as `uahsn.curves.csv` by default. `--device` accepts `cpu`, `cuda`, or `auto`.

Evaluation saves each method's enhanced `.wav`, plus `.npz` arrays for per-frame gains, estimated noise PSD, runtime seconds, and real-time factor. The reconstructed baseline also saves each FIR coefficient vector. `metrics.json` contains one record per mixture and method; `summary.json` contains per-method means, sample standard deviations, seeded bootstrap 95% confidence intervals, and paired two-sided Wilcoxon signed-rank comparisons of UA-HSN with each baseline. Pairing uses `mixture_id`. Spectral subtraction uses configurable `--oversubtraction`; the reconstructed baseline accepts `--linear-phase-filter-length` and `--linear-phase-frame-length`. These are measurement implementations, not claims of quality. A trained checkpoint requires real paired data; no dataset is bundled or downloaded.

## Evaluation definitions

- Reference metrics: zero-mean SI-SDR via projection onto the clean reference; segmental SNR on 320-sample frames with 160-sample hop, clipped per frame to `[−10,35]` dB; classical [pystoi STOI](https://github.com/mpariente/pystoi); and 16 kHz wideband [PESQ](https://github.com/ludlows/PESQ). STOI and PESQ are recorded as `null` when a clip is too short or unsupported.
- Noise-only frames have full clean-reference window power at or below −40 dBFS by default (`--noise-only-threshold-dbfs`). Gain variance averages squared changes at entries whose current frame is noise-only.
- Residual peak density counts an interior TF bin only if its enhanced power exceeds each of its eight neighbors by 6 dB by default (`--rpd-theta-db`) and exceeds the local predicted residual noise floor `noise_psd × gain²`. It divides by noise-only duration and reports peaks per second. It is `null` when there are no noise-only frames.
- Aliasing ratio averages IDFT gain energy outside circular lags `[-31,31]` by default (`--alias-central-radius`).
- Noise reduction shadow-filters the known additive noise `mixture − clean` with the saved gain sequence. For the reconstructed baseline it uses its saved time-varying FIR sequence with the same overlap-add implementation. It reports `10 log10(||v||² / ||v_processed||²)`.
- Bootstrap resamples mixture IDs within each method; `--bootstrap-samples` and `--seed` control reproducibility. Undefined metrics are retained as `null` per utterance and excluded from summaries. Wilcoxon results are `null` unless at least three paired nonzero differences are available.

`prepare_dataset` accepts stereo or mono input and resamples it to 16 kHz mono, writing float WAV files to avoid quantization changing the requested SNR. It reports and skips unreadable or silent files. Clean speech is split by speaker: the first subdirectory is treated as the speaker ID; flat names such as `p225_001.wav` use the prefix. For other layouts, pass `--speaker-regex '(?P<speaker>...)'` against the relative clean path. At least three distinct speaker groups are required. The CSV fields are `mixture_id,clean_path,noise_path,mixture_path,speaker,noise_type,snr_db,split,seed`. Training rows leave `mixture_path` and `snr_db` empty; each training epoch draws a new noise excerpt and a uniform SNR in `[-5, 20]` dB. Validation and test rows contain fixed mixtures at −5, 0, 5, 10, and 15 dB. For each mixture, power-based SNR scaling is applied exactly, then a single shared scale is applied to clean speech, scaled noise, and mixture if needed to prevent clipping.
