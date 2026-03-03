# 3D Facial Expression Recognition

**Affective Computing – Spring 2026**
Bryan Baldie, Twinkle Markana

Classifies 6 facial expressions from 3D landmark data using Leave-One-Subject-Out (LOSO) cross-validation on the BU-4DFE dataset.

---

## Expressions

Angry, Disgust, Fear, Happy, Sad, Surprise

## Dataset

BU-4DFE (`BU4DFE_BND_V1.1/`) — 58 subjects, 83 3D facial landmarks per frame (`.bnd` format).

## Usage

```
python Project1.py <mode> <data_dir>
```

| Mode | Description                                               |
| ---- | --------------------------------------------------------- |
| `o`  | Original – raw x, y, z coordinates                        |
| `t`  | Translated – centroid subtracted (face centred at origin) |

**Example:**

```
python Project1.py t ./BU4DFE_BND_V1.1
```

## Classifier

Random Forest (100 trees, `n_jobs=-1`)

- StandardScaler fit on training split only (no data leakage)
- 58-fold LOSO cross-validation

## Output

Results are written to `results_<mode>.txt` alongside `Project1.py`, containing:

- Per-fold accuracy table (subject, test sample count, accuracy)
- Overall LOSO accuracy and elapsed time
- Per-class classification report (precision, recall, F1)
- Confusion matrix (rows = true label, cols = predicted label)

## Configuration

At the top of `Project1.py`:

```python
PEAK_ONLY = False  # True = last frame only (~343 samples, fast)
                   # False = all frames (~34,697 samples, default)
```
