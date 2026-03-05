# 3D Facial Expression Recognition

**Affective Computing – Spring 2026**
Bryan Baldie (U44374228), Twinkle Markana (U55075888)

Classifies 6 facial expressions from 3D landmark data using Leave-One-Subject-Out (LOSO) cross-validation on the BU-4DFE dataset.

---

## Expressions

Angry, Disgust, Fear, Happy, Sad, Surprise

## Dataset

BU-4DFE (`BU4DFE_BND_V1.1/`) — 101 subjects (58 female F001–F058, 43 male M001–M043), 83 3D facial landmarks per frame (`.bnd` format).

## Usage

### Classifier (Project1.py)

```
python Project1.py <mode> <data_dir>
```

| Mode | Description                                               |
| ---- | --------------------------------------------------------- |
| `o`  | Original – raw x, y, z coordinates                        |
| `t`  | Translated – centroid subtracted (face centred at origin) |
| `x`  | Rotated 180° around x-axis                                |
| `y`  | Rotated 180° around y-axis                                |
| `z`  | Rotated 180° around z-axis                                |

**Example:**

```
python Project1.py t ./BU4DFE_BND_V1.1
```

### Sample Plots (plot_samples.py)

Generates 5 3D scatter plots (one per mode) from a single sample face:

```
python plot_samples.py <data_dir>
```

Saves `plot_o.png`, `plot_t.png`, `plot_x.png`, `plot_y.png`, `plot_z.png`.

## Classifier

Random Forest (100 trees, `n_jobs=-1`)

- StandardScaler fit on training split only (no data leakage)
- 101-fold LOSO cross-validation (one fold per subject)

## Output

Results are printed to the screen and written to `results_<mode>.txt` alongside `Project1.py`, containing:

- Per-fold accuracy table (subject, test sample count, accuracy)
- Overall LOSO accuracy and elapsed time
- Per-class classification report (precision, recall, F1)
- Confusion matrix (rows = true label, cols = predicted label)

## Configuration

At the top of `Project1.py`:

```python
PEAK_ONLY = False  # True = last frame only (~601 samples, fast)
                   # False = all frames (default, better accuracy)
```

## Packages

| Package        | Version tested | Purpose                                         |
| -------------- | -------------- | ----------------------------------------------- |
| `numpy`        | ≥ 1.24         | Array operations, matrix multiplication         |
| `scikit-learn` | ≥ 1.3          | RandomForestClassifier, StandardScaler, metrics |
| `matplotlib`   | ≥ 3.7          | 3D scatter plots (`plot_samples.py` only)       |

Standard library: `os`, `sys`, `glob`, `time`, `warnings`, `math`
