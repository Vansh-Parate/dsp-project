# UA-HSN research prototype

CPU-compatible, causal, single-channel 16 kHz speech enhancement. The five methods share `enhance(audio) -> EnhancementResult`, which contains enhanced audio and a per-frame, 257-bin gain operator. UA-HSN also returns smoothed band gains and predicted standard deviations.

## Signal path

- 320-sample periodic sqrt-Hann window, 160-sample hop, 512-point FFT; input is left padded by one hop. A zero block flushes the overlap-add tail. No frame reads beyond its current 160-sample hop.
- 32 ERB-rate-spaced triangular bands. Analysis weights sum to one within each band; synthesis weights sum to one across bands at each FFT bin.
- Noise estimation smooths power with `alpha_p=0.9` and takes the minimum over the current and seven completed 12-frame subwindows (up to 96 frames). The decision-directed Wiener recurrence uses its own previous Wiener gain, never the network gain.
- The network has Linear(96,64), PReLU, two GRU(64,64) layers, and separate 32-dimensional sigmoid-gain and log-variance heads. PyTorch parameter count: **60,289**.
- Training uses band ideal ratio gains, compressed magnitude exponent 0.3, Gaussian NLL, and temporal consistency only when both adjacent oracle gains are below 0.3. Loss weights are 1, 0.1, 0.1.
- `sigma0` is the median predicted standard deviation over the validation manifest and is saved in the checkpoint. Inference uses it for uncertainty-adaptive recursive gain smoothing.
- The linear-phase baseline projects each spectral-subtraction gain onto a symmetric 63-tap FIR. Its samples are generated causally with a fixed 31-sample group delay. The FIR response magnitude is saved as its gain operator.

## Run

Use Python 3.10+ with dependencies in `requirements.txt`:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m scripts.run_experiments --synthetic-smoke
```

The smoke test generates sine waves plus seeded white noise and writes audio and `.npz` gain arrays under `results/synthetic_smoke`. Its UA-HSN network is **untrained**, so the files validate execution only and are not research results.

For paired WAV data, prepare deterministic mixtures, train, and evaluate:

```bash
python -m scripts.prepare_dataset --clean-dir data/clean --noise-dir data/noise --output-dir data/prepared
python -m scripts.train_model --train-manifest data/prepared/train.jsonl --validation-manifest data/prepared/validation.jsonl --epochs 1 --checkpoint checkpoints/uahsn.pt
python -m scripts.evaluate --manifest data/prepared/validation.jsonl --checkpoint checkpoints/uahsn.pt --output-dir results/evaluation
```

Evaluation saves each method's enhanced `.wav`, `.npz` gain arrays, and `metrics.json` containing SI-SDR, log-spectral distance, operator measures, and wall-clock real-time factor. These are measurement implementations, not claims of quality. A trained checkpoint requires real paired data; no dataset is bundled or downloaded.
