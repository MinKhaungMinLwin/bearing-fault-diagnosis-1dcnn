# Bearing Fault Diagnosis from Vibration Signals with a 1D CNN

A 1D convolutional neural network classifies raw vibration windows from the Case Western Reserve University (CWRU) Bearing Data Center into four classes: **normal**, **inner-race fault**, **ball fault** and **outer-race fault**. The project runs a controlled comparison of a small first-layer kernel (3 samples) against a wide one (64 samples, after Zhang et al., 2017). Models are tested on bearings with a fault size they never saw during training.

All the work is in [`deep_learning.ipynb`](deep_learning.ipynb).

## Results

| Held-out fault size | Kernel 3 macro-F1 | Kernel 64 macro-F1 |
| :--- | :---: | :---: |
| 0.007" | 0.88 ± 0.06 | 1.00 ± 0.00 |
| 0.014" | 0.31 ± 0.01 | 0.36 ± 0.04 |
| 0.021" | 0.55 ± 0.09 | 0.43 ± 0.01 |

*Mean ± sd over 3 seeds on unseen bearings (normal test data: load 3).*

- Both models get every validation window right on the training bearings, but on unseen bearings macro-F1 ranges from 0.31 to 1.00 depending on which fault size is held out.
- The kernel size has no consistent effect: the wide kernel is better for 0.007", worse for 0.021", and both are near chance for 0.014".
- With 0.014" held out, the models predict "ball" for almost every faulty window. At that fault size the inner- and outer-race signals have ball-like RMS, which suggests the models learn amplitude cues tied to specific bearings rather than fault signatures that transfer.

## Repository layout

```
deep_learning.ipynb    # full analysis: data checks, model, comparison, error analysis, summary
data/cwru/manifest.csv # file number -> label, fault size, motor load
data/cwru/*.mat        # raw CWRU files (not in git, see below)
figures/               # figures saved by the notebook
requirements.txt       # pinned Python dependencies
```

## Setup

Tested with Python 3.15.2. Package versions are pinned in [`requirements.txt`](requirements.txt).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The notebook uses CUDA or Apple MPS when available and falls back to the CPU otherwise.

## Getting the data

The `.mat` files are **not** in this repository (`data/cwru/*.mat` is in `.gitignore`). Download them from the CWRU Bearing Data Center (https://engineering.case.edu/bearingdatacenter) before running the notebook. Each file is served at `https://engineering.case.edu/sites/default/files/<number>.mat`.

The project uses 40 files: drive-end bearing, 12 kHz fault data, motor loads 0–3 hp.

| Class | 0.007" | 0.014" | 0.021" |
| :--- | :--- | :--- | :--- |
| Normal baseline | 97, 98, 99, 100 (no fault size) | | |
| Inner race | 105, 106, 107, 108 | 169, 170, 171, 172 | 209, 210, 211, 212 |
| Ball | 118, 119, 120, 121 | 185, 186, 187, 188 | 222, 223, 224, 225 |
| Outer race (centred, 6:00) | 130, 131, 132, 133 | 197, 198, 199, 200 | 234, 235, 236, 237 |

Within each group of four, the files are in load order 0, 1, 2, 3 hp. [`data/cwru/manifest.csv`](data/cwru/manifest.csv) lists the same mapping.

Download loop (run from the repository root):

```bash
mkdir -p data/cwru
for n in 97 98 99 100 \
         105 106 107 108  169 170 171 172  209 210 211 212 \
         118 119 120 121  185 186 187 188  222 223 224 225 \
         130 131 132 133  197 198 199 200  234 235 236 237; do
  curl -fL -o "data/cwru/$n.mat" "https://engineering.case.edu/sites/default/files/$n.mat"
done
ls data/cwru/*.mat | wc -l   # expect 40
```

### Two quirks in the data

- **The normal baseline is recorded at 48 kHz**, not 12 kHz like the fault files. The notebook checks this from the spectrum (the shaft-speed peak only lands on about 30 Hz if 48 kHz is assumed) and decimates the normal signals by 4 to 12 kHz.
- **File 99 contains two recordings**: its own `X099_*` signals and a copy of file 98's `X098_*` signals. The notebook selects the drive-end channel whose number matches the file name (`X099_DE_time`) instead of taking the first `*_DE_time` key.

## Running

```bash
jupyter lab deep_learning.ipynb
```

Run all cells from top to bottom. The comparison trains 6 models, and the leave-one-fault-size-out experiment trains 18 more, each for 30 epochs. Figures are written to `figures/`:

| File | Content |
| :--- | :--- |
| `fig1_samples.png` | One training window per class |
| `fig2_training_curves.png` | Training and validation curves, kernel 3 vs 64 |
| `fig3_confusion.png` | Confusion matrices on unseen 0.014" bearings |
| `fig4_loso.png` | Test macro-F1 by held-out fault size and kernel |

## References

- Case Western Reserve University Bearing Data Center. https://engineering.case.edu/bearingdatacenter
- Zhang, W., Peng, G., Li, C., Chen, Y., & Zhang, Z. (2017). A new deep learning model for fault diagnosis with good anti-noise and domain adaptation ability on raw vibration signals. *Sensors*, 17(2), 425.
