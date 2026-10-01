# UA-HSN: uncertainty-adaptive speech denoising

UA-HSN is a research prototype for causal, single-channel speech enhancement at 16 kHz. It compares noisy speech, spectral subtraction, decision-directed Wiener filtering, a reconstructed linear-phase spectral-subtraction baseline, and an uncertainty-adaptive neural model. The repository includes dataset preparation, training, inference, evaluation, operator metrics, benchmarking, and CSV result tables.

**No dataset or published research result is bundled.** Supply your own clean-speech and noise WAV files. The synthetic smoke test checks execution with an untrained network; it is not a quality result.

## Project file structure

```text
uahsn/
├── README.md                     # Project guide
├── requirements.txt              # Python packages
├── uahsn/                        # Importable implementation
│   ├── __init__.py
│   ├── audio.py                   # Loading, resampling, SNR mixing, WAV output
│   ├── bands.py                   # 32 ERB-like bands and interpolation weights
│   ├── baselines.py               # Common enhancement interface and baselines
│   ├── benchmark.py               # Runtime and complexity helpers
│   ├── config.py                  # Signal and model constants
│   ├── features.py                # Causal 96-dimensional features
│   ├── inference.py               # UA-HSN enhancement and gain smoothing
│   ├── losses.py                  # Ideal gains and training loss
│   ├── metrics.py                 # SI-SDR, segmental SNR, STOI, PESQ
│   ├── minstats.py                # Minimum-statistics noise PSD
│   ├── model.py                   # Linear/PReLU/GRU network and two heads
│   ├── operator_metrics.py        # V_G, RPD, aliasing ratio, shadow NR
│   ├── statistics.py              # Bootstrap and Wilcoxon summaries
│   ├── stft.py                    # Causal STFT and inverse overlap-add
│   └── train.py                   # Crops, normalization, training, checkpoint
├── scripts/                      # Command-line programs
│   ├── __init__.py
│   ├── prepare_dataset.py         # Source validation and manifest creation
│   ├── train_model.py             # Overfit check and model training
│   ├── evaluate.py                # Detailed JSON evaluation
│   └── run_experiments.py         # Final five-method CSV tables
├── tests/                        # Unit and integration tests
│   ├── test_baselines.py
│   ├── test_core.py
│   ├── test_dataset.py
│   ├── test_evaluation.py
│   ├── test_runner.py
│   └── test_training.py
├── data/                         # User-provided sources and generated data
│   ├── clean/<speaker>/*.wav      # Clean source WAVs
│   ├── noise/<noise_type>/*.wav   # Noise source WAVs
│   └── prepared/                  # Generated manifests/audio
│       ├── train.csv
│       ├── validation.csv
│       ├── test.csv
│       ├── audio/
│       └── mixtures/
├── checkpoints/                  # Model checkpoints and training curves
└── results/                      # Enhanced audio, gains, tables, summary
```

The directory tree shows the expected layout. Input WAVs are not created by the program. Paths are configurable, so you can keep sources elsewhere and pass their actual locations to `--clean-dir` and `--noise-dir`.

## Install and test

Use Python 3.10 or newer. CPU works; CUDA is optional. Run commands from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`. Dependencies include NumPy, SciPy, SoundFile, PyTorch, pystoi, pesq, and THOP. Install a PyTorch build appropriate for your CUDA setup if training on a GPU.

## Signal path and methods

| Item | Value |
| --- | --- |
| Audio | Mono, 16,000 samples/second |
| Window | Periodic sqrt-Hann, 320 samples (20 ms) |
| Hop | 160 samples (10 ms) |
| FFT | 512 points, 257 positive-frequency bins |
| Frequency representation | 32 ERB-like triangular bands |
| Network input | 96 values: log10 band energy, a-posteriori SNR dB, decision-directed a-priori SNR dB |
| Noise estimator | PSD smoothing `alpha_p=0.9`; 8 minimum-statistics subwindows of 12 frames |
| Wiener recurrence | `beta=0.98`, using the previous Wiener gain |
| Gain floor | 0.1 |
| Network | Linear(96,64), PReLU, GRU(64,64), GRU(64,64), two 32-unit heads |
| PyTorch parameter count | **60,289** |

The network gain head uses a sigmoid; its second head predicts log variance. Validation-derived `sigma0` converts predicted standard deviation into a band-specific recursive smoothing factor between 0.2 and 0.9. Smoothed band gains are interpolated to FFT bins with partition-of-unity weights. All spectral methods reuse noisy phase. Processing does not read future input hops.

Each method implements `enhance(audio) -> EnhancementResult`. The result contains enhanced waveform, per-frame gain matrix, estimated noise PSD, and runtime information. UA-HSN additionally exposes band gains and predicted standard deviations. Method names in the result files are `noisy`, `spectral_subtraction`, `wiener`, `reconstructed_linear_phase_baseline`, and `uahsn`.

The linear-phase method is a **reconstruction from a prose specification, not an exact reproduction of an external paper**. It converts spectral-subtraction gains into causal Type-I symmetric FIR filters and applies them with overlap-add. Use odd filter lengths of 31, 63, or 127, and frame lengths of 256, 320, or 512 samples. The defaults are 63 and 320.

## 1. Obtain and arrange WAV files

`prepare_dataset.py` reads local WAVs; it does not download them. A starting source is [Microsoft MS-SNSD](https://github.com/microsoft/MS-SNSD), with [clean speech WAVs](https://github.com/microsoft/MS-SNSD/tree/master/clean_train) and [noise WAVs](https://github.com/microsoft/MS-SNSD/tree/master/noise_train). [DEMAND](https://zenodo.org/records/1227121) offers environmental noise; select 16 kHz recordings if using it. Check each source's license before redistributing files.

Example layout:

```text
data/
├── clean/
│   ├── p234/
│   │   ├── p234_001.wav
│   │   └── p234_002.wav
│   ├── p237/
│   │   └── p237_016.wav
│   └── p241/
│       └── p241_197.wav
└── noise/
    ├── air_conditioner/
    │   └── AirConditioner_2.wav
    └── babble/
        └── Babble_2.wav
```

These filenames illustrate the layout; use files present in your collection. The first subfolder below `data/clean` is treated as the speaker ID. Flat clean filenames such as `p234_001.wav` also work: the filename prefix identifies the speaker. Use `--speaker-regex` with a capture group for other conventions. At least **three distinct speakers** are required to make separate train, validation, and test splits. The first noise subfolder identifies `noise_type` in Table 6; a flat noise filename uses its stem.

Stereo or other-rate WAVs are accepted and converted to 16 kHz mono. Corrupt, unreadable, and near-silent files are reported and skipped. The preparer does not peak-normalize clean and noise independently.

## 2. Prepare dataset and manifests

```bash
python3 -m scripts.prepare_dataset \
    --clean-dir data/clean \
    --noise-dir data/noise \
    --output-dir data/prepared \
    --seed 0
```

If you see `Input directory not found`, create and populate those folders or pass the absolute locations of your WAV folders. The command writes `data/prepared/train.csv`, `validation.csv`, and `test.csv`, plus resampled audio and fixed validation/test mixtures. Manifest fields are `mixture_id`, `clean_path`, `noise_path`, `mixture_path`, `speaker`, `noise_type`, `snr_db`, `split`, and `seed`. Generated paths are absolute.

Training rows intentionally have empty `mixture_path` and `snr_db`: the trainer draws a new 3-second clean crop, random noise segment, and SNR uniformly from **−5 to 20 dB**. Validation and test rows contain fixed mixtures at **−5, 0, 5, 10, and 15 dB**. The seed controls source selection and mixing reproducibly. Mixture scaling follows:

```text
Ps = mean(clean²)
Pv = mean(noise_segment²)
noise_scale = sqrt(Ps / (Pv × 10^(SNR_dB / 10)))
mixture = clean + noise_scale × noise_segment
```

When clipping needs prevention, one common factor scales clean, scaled noise, and mixture together, preserving SNR.

## 3. Run smoke and overfit checks

The synthetic smoke test needs no external WAVs:

```bash
python3 -m scripts.run_experiments --synthetic-smoke --output results/synthetic_smoke
```

It synthesizes sine-wave speech and white noise and saves all five methods' enhanced WAVs and `.npz` gain arrays. Its UA-HSN weights are random. Before a long training job, run the fixed two-batch overfit check:

```bash
python3 -m scripts.train_model --overfit-test --device cpu --seed 0
```

The check passes only if the model substantially reduces training loss on the tiny fixed synthetic set.

## 4. Train UA-HSN

Short real-data sanity run:

```bash
python3 -m scripts.train_model \
    --manifest data/prepared/train.csv \
    --val-manifest data/prepared/validation.csv \
    --max-train-files 16 \
    --epochs 2 \
    --batch-size 16 \
    --device cpu \
    --seed 0 \
    --checkpoint checkpoints/sanity.pt
```

Longer run, using all training rows:

```bash
python3 -m scripts.train_model \
    --manifest data/prepared/train.csv \
    --val-manifest data/prepared/validation.csv \
    --epochs 30 \
    --batch-size 16 \
    --device cuda \
    --seed 0 \
    --checkpoint checkpoints/best.pt
```

Use `--device cpu` or `--device auto` if CUDA is unavailable. The trainer uses Adam at `1e-3`, `ReduceLROnPlateau`, gradient norm clipping at 5, and early stopping on validation SI-SDR after five unimproved epochs by default. It trains ideal band ratio gains with compressed-magnitude exponent 0.3 and loss weights `1` (magnitude), `0.1` (NLL), and `0.1` (temporal consistency). Temporal consistency applies only to consecutive units with oracle gain below 0.3.

The selected checkpoint stores weights, best epoch, **training-only** feature mean and standard deviation, and `sigma0`, the median predicted standard deviation on validation data after training. Test data are not used for normalization, tuning, or `sigma0`. Epoch curves are saved next to the checkpoint, for example `checkpoints/best.curves.csv`.

## 5. Generate final CSV results

```bash
python3 -m scripts.run_experiments \
    --test-manifest data/prepared/test.csv \
    --checkpoint checkpoints/best.pt \
    --output results \
    --device cpu
```

The runner requires a test manifest and trained checkpoint. It evaluates all five methods on every mixture, warms up methods, and benchmarks three enhancement repeats by default; set `--benchmark-repeats N` to change that. Neural inference runs in evaluation mode without gradients. CUDA timing synchronizes before and after calls.

```text
results/
├── per_utterance.csv             # One row per mixture and method
├── table1_sisdr.csv              # Mean SI-SDR at each input SNR
├── table2_stoi.csv               # Mean STOI at each input SNR
├── table3_pesq.csv               # Mean wideband PESQ at each input SNR
├── table5_operator_metrics.csv   # Mean V_G, RPD, rho_alias, NR
├── table6_noise_type.csv         # 5 dB SI-SDR improvement by noise type
├── table7_compute.csv            # Parameters, MACs, timing, RTF, latency
├── summary.md                    # Plain-English table guide and warnings
├── audio/*.wav                   # Enhanced audio per mixture and method
└── gains/*.npz                  # Gains, PSD, per-frame timing, optional FIR
```

Tables 1–3 use the exact columns `Method | -5 dB | 0 dB | 5 dB | 10 dB | 15 dB`. Table 6 reports **enhanced SI-SDR minus noisy SI-SDR** for the same 5 dB mixture, grouped by noise type. `per_utterance.csv` includes signal metrics, operator metrics, parameters, MACs per second, mean/99.9th/max frame time, RTF, and algorithmic latency. `.npz` files preserve each method's gain sequence and noise PSD, with optional UA-HSN band gains/uncertainty or reconstructed FIR coefficients. Missing measurements are `NaN` and explained in `summary.md` and stderr; no values are fabricated.

For per-utterance JSON, bootstrap confidence intervals, and paired Wilcoxon tests, use the separate detailed evaluator:

```bash
python3 -m scripts.evaluate \
    --manifest data/prepared/test.csv \
    --checkpoint checkpoints/best.pt \
    --output-dir results/evaluation \
    --seed 0
```

This writes `metrics.json`, `summary.json`, enhanced WAVs, and gain/PSD NPZs. Pairing uses `mixture_id`. Use the final runner for the requested CSV tables and the detailed evaluator for bootstrap/Wilcoxon summaries.

## Metrics and interpretation

| Metric | Definition |
| --- | --- |
| SI-SDR | Projection of enhanced speech onto the clean reference, in dB |
| SI-SDR improvement | Enhanced SI-SDR minus noisy-mixture SI-SDR for the same utterance |
| Segmental SNR | Frame SNR clipped to [−10, 35] dB before averaging |
| STOI | `pystoi`, against clean reference |
| PESQ | `pesq` wideband mode at 16 kHz, against clean reference |
| `V_G` | Mean squared gain change in clean-reference noise-only frames |
| RPD | Isolated residual TF peaks per noise-only second |
| `rho_alias` | Fraction of IDFT gain energy outside a central circular-lag region |
| NR | Reduction of known additive noise after shadow filtering with the saved operator |

The detailed evaluator's default clean-reference noise-only threshold is −40 dBFS. RPD requires power to exceed all eight TF neighbors by at least 6 dB and to exceed the local residual noise floor. NR filters the known additive noise with the same gain sequence used for the mixture; for the FIR baseline, it uses the saved FIR sequence. The detailed evaluator exposes `--noise-only-threshold-dbfs`, `--rpd-theta-db`, and `--alias-central-radius`.

The compute table uses actual PyTorch model parameters, THOP neural-network MACs per second, and an analytic neural-network matrix-product estimate. Classical NumPy methods have no comparable THOP MAC measurement, so those cells are `NaN`. RTF includes the full enhancement call; frame-time statistics describe timed per-frame work. Short test sets give unstable 99.9th-percentile timing, and the summary warns when few frames were measured.

## Baseline experiment options

The final runner and detailed evaluator support `--oversubtraction FLOAT` (default 1), `--linear-phase-filter-length INT` (default 63), and `--linear-phase-frame-length {256,320,512}` (default 320). For filter lengths 31, 63, and 127, run three separate experiments with separate `--output` directories. The UA-HSN signal configuration remains 320/160/512.

## Troubleshooting

- **`Input directory not found`**: confirm paths contain WAV files, for example `find data/clean data/noise -type f -iname '*.wav'`.
- **Too few speaker groups**: add at least three distinct speakers, put each in its own subfolder, or supply `--speaker-regex`.
- **No usable WAV files**: only `.wav` is scanned. Convert other formats first; corrupt and silent files are skipped.
- **CUDA unavailable**: use `--device cpu` or `--device auto`.
- **`NaN` metric**: read `results/summary.md`. Very short audio, no noise-only frames, or a metric dependency failure can make a value undefined.
- **Timing varies**: use enough test audio and keep hardware and system load comparable. The data split and training use seeds, but runtime still varies with host load.

After modifying code, rerun `python3 -m unittest discover -s tests -v`. Tests cover STFT reconstruction, band weights, model shapes and parameter count, exact mixture SNR, baseline output bounds, known-signal metrics, training behavior, and table generation.
