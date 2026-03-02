"""
Project1.py
Affective Computing - Spring 2026
Bryan Baldie  U44374228
Twinkle Markana  U########

3D Facial Expression Recognition using Leave-One-Subject-Out (LOSO)
cross validation on the BU-4DFE dataset.

Usage:
    python Project1.py <mode> <data_dir>

Modes (feature representations):
    o  - original    : raw x, y, z coordinates (no transformation)
    t  - translated  : face centered at the origin (subtract centroid)
    x  - rotate x    : (coming soon)
    y  - rotate y    : (coming soon)
    z  - rotate z    : (coming soon)

Example:
    python Project1.py t ./BU4DFE_BND_V1.1

Classifiers:
    1. Support Vector Machine (linear SVM via LinearSVC)
    2. k-Nearest Neighbours (k=5)
    3. Random Forest (100 trees)

Each .bnd file: 83 rows of  <index>  <x>  <y>  <z>
                (optional empty 84th row is ignored)
"""

import os
import sys
import glob
import time
import warnings

# Force UTF-8 output so Unicode box-drawing characters print on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import matplotlib
matplotlib.use("Agg")           # non-interactive backend, safe on all platforms
import matplotlib.pyplot as plt

from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)

warnings.filterwarnings("ignore")

# ── Constants ─────────────────────────────────────────────────────────────────

EXPRESSIONS  = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise"]
N_LANDMARKS  = 83
VALID_MODES  = {"o", "t", "x", "y", "z"}

# Set to True to use only the last frame per sequence (much faster, fewer samples).
# False (default) uses all frames for more robust LOSO estimates.
PEAK_ONLY = False


# ── .bnd file parsing ─────────────────────────────────────────────────────────

def parse_bnd(filepath):
    """
    Read one .bnd file and return a (N_LANDMARKS, 3) float32 array,
    or None if the file does not contain exactly N_LANDMARKS valid rows.

    Row format:  <index>  <x>  <y>  <z>
    The optional empty 84th row is automatically skipped.
    """
    pts = []
    with open(filepath, "r") as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) >= 4:
                try:
                    pts.append([float(parts[1]),
                                float(parts[2]),
                                float(parts[3])])
                except ValueError:
                    continue
    if len(pts) != N_LANDMARKS:
        return None
    return np.array(pts, dtype=np.float32)


# ── Landmark transformations ──────────────────────────────────────────────────

def transform_landmarks(landmarks, mode):
    """
    Apply a geometric transformation to the (N_LANDMARKS, 3) array and
    return a transformed (N_LANDMARKS, 3) float32 array.

    The transformed array is later flattened to a 1-D feature vector.

    Parameters
    ----------
    landmarks : ndarray, shape (N_LANDMARKS, 3)
    mode      : str  one of 'o', 't', 'x', 'y', 'z'

    Modes
    -----
    'o'  original   – raw x, y, z  (no change)
    't'  translated – subtract the centroid so the face is centred at (0,0,0)
                      centroid = mean of all 83 landmark positions
    'x'  rotate-x   – (not yet implemented)
    'y'  rotate-y   – (not yet implemented)
    'z'  rotate-z   – (not yet implemented)
    """
    if mode == "o":
        # Return raw coordinates unchanged
        return landmarks.copy()

    elif mode == "t":
        # --- Translation to origin ---
        # 1. Compute the average x, y, z across all 83 landmarks
        #    This gives the approximate centre of the face.
        centroid = landmarks.mean(axis=0)   # shape (3,): [mean_x, mean_y, mean_z]

        # 2. Subtract the centroid from every landmark so the face sits at (0,0,0)
        return landmarks - centroid

    elif mode == "x":
        raise NotImplementedError("Rotation around x-axis not yet implemented.")

    elif mode == "y":
        raise NotImplementedError("Rotation around y-axis not yet implemented.")

    elif mode == "z":
        raise NotImplementedError("Rotation around z-axis not yet implemented.")

    else:
        raise ValueError(f"Unknown mode '{mode}'. Valid modes: {sorted(VALID_MODES)}")


# ── Dataset loader ────────────────────────────────────────────────────────────

def load_dataset(data_dir, mode, peak_only=False):
    """
    Walk data_dir, parse every .bnd file, apply transform_landmarks(mode),
    and flatten each result to a 1-D feature vector.

    Returns
    -------
    X        : ndarray (n_samples, N_LANDMARKS * 3)  – feature matrix
    y        : ndarray (n_samples,)                  – integer class labels
    subjects : ndarray (n_samples,)                  – subject ID strings
    """
    X_list, y_list, subj_list = [], [], []

    subject_dirs = sorted(glob.glob(os.path.join(data_dir, "F*")))
    if not subject_dirs:
        raise FileNotFoundError(
            f"No subject directories found under: {data_dir}\n"
            "Expected structure:  <data_dir>/F001/Angry/*.bnd  etc."
        )

    for subj_dir in subject_dirs:
        subject = os.path.basename(subj_dir)

        for expr_idx, expr in enumerate(EXPRESSIONS):
            expr_dir = os.path.join(subj_dir, expr)
            if not os.path.isdir(expr_dir):
                continue

            bnd_files = sorted(glob.glob(os.path.join(expr_dir, "*.bnd")))
            if not bnd_files:
                continue

            if peak_only:
                bnd_files = bnd_files[-1:]      # last frame ~ peak expression

            for bnd_path in bnd_files:
                lm = parse_bnd(bnd_path)
                if lm is None:
                    continue

                # Apply the mode-specific transformation, then flatten to 1-D
                feature_vec = transform_landmarks(lm, mode).flatten()

                X_list.append(feature_vec)
                y_list.append(expr_idx)
                subj_list.append(subject)

    return (np.array(X_list,    dtype=np.float32),
            np.array(y_list,    dtype=np.int32),
            np.array(subj_list))


# ── LOSO cross-validation ─────────────────────────────────────────────────────

def loso_eval(X, y, subjects, clf_name, clf):
    """
    Leave-One-Subject-Out cross-validation.

    For each unique subject:
      - Train on all samples NOT from that subject.
      - StandardScaler is fit on train split only (prevents data leakage).
      - Test on the held-out subject's samples.

    Returns (overall_accuracy, all_true_labels, all_predicted_labels).
    """
    unique_subjects = np.unique(subjects)
    all_true, all_preds = [], []

    bar = "=" * 64
    print(f"\n{bar}")
    print(f"  Classifier : {clf_name}")
    print(bar)

    t0 = time.time()

    for test_subj in unique_subjects:
        train_mask = subjects != test_subj
        test_mask  = subjects == test_subj

        X_train, y_train = X[train_mask], y[train_mask]
        X_test,  y_test  = X[test_mask],  y[test_mask]

        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)

        fold_acc = accuracy_score(y_test, preds)
        all_true.extend(y_test)
        all_preds.extend(preds)

        print(f"  {test_subj}  |  test samples: {len(y_test):4d}  |  "
              f"acc = {fold_acc:.4f}")

    elapsed = time.time() - t0
    overall = accuracy_score(all_true, all_preds)

    print(f"\n  Elapsed : {elapsed:.1f}s")
    print(f"  Overall LOSO Accuracy : {overall:.4f}")
    print(f"\n  Classification Report :")
    print(classification_report(all_true, all_preds,
                                target_names=EXPRESSIONS, digits=4))
    print("  Confusion Matrix (rows = true, cols = predicted) :")
    print(confusion_matrix(all_true, all_preds))

    return overall, all_true, all_preds


# ── Confusion-matrix plot ─────────────────────────────────────────────────────

def plot_confusion_matrix(true, preds, clf_name, mode, save_dir):
    """Save a colour-coded confusion matrix PNG.  Filename includes the mode."""
    cm  = confusion_matrix(true, preds)
    fig, ax = plt.subplots(figsize=(7, 6))
    im  = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)

    ticks = np.arange(len(EXPRESSIONS))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(EXPRESSIONS, rotation=45, ha="right")
    ax.set_yticklabels(EXPRESSIONS)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j],
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=9)

    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_title(f"Confusion Matrix [{mode}] – {clf_name}")
    fig.tight_layout()

    safe_clf = (clf_name.replace(" ", "_")
                        .replace("(", "").replace(")", "")
                        .replace(",", "").replace("=", ""))
    out_path = os.path.join(save_dir, f"cm_{mode}_{safe_clf}.png")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  Confusion matrix saved -> {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # ── 1. Parse command-line arguments ───────────────────────────────────────
    if len(sys.argv) < 3:
        print("Usage: python Project1.py <mode> <data_dir>")
        print(f"  mode     : one of {sorted(VALID_MODES)}")
        print("  data_dir : path to BU4DFE_BND_V1.1 directory")
        print("\nExample:")
        print("  python Project1.py t ./BU4DFE_BND_V1.1")
        sys.exit(1)

    mode     = sys.argv[1].lower()
    data_dir = sys.argv[2]

    if mode not in VALID_MODES:
        print(f"Error: unknown mode '{mode}'. Valid modes: {sorted(VALID_MODES)}")
        sys.exit(1)

    if not os.path.isdir(data_dir):
        print(f"Error: data directory not found: {data_dir}")
        sys.exit(1)

    save_dir = os.path.dirname(os.path.abspath(__file__))

    # ── 2. Load and transform data ────────────────────────────────────────────
    mode_labels = {
        "o": "original (raw coordinates)",
        "t": "translated (centred at origin)",
        "x": "rotated around x-axis",
        "y": "rotated around y-axis",
        "z": "rotated around z-axis",
    }
    frame_str = "peak frame only" if PEAK_ONLY else "all frames"
    print(f"\nMode     : {mode}  –  {mode_labels[mode]}")
    print(f"Frames   : {frame_str}")
    print(f"Data dir : {data_dir}")
    print(f"\nLoading dataset ...")

    t_load = time.time()
    X, y, subjects = load_dataset(data_dir, mode, peak_only=PEAK_ONLY)
    print(f"  Done in {time.time() - t_load:.1f}s")
    print(f"  Samples     : {len(X)}")
    print(f"  Subjects    : {len(np.unique(subjects))}")
    print(f"  Feature dim : {X.shape[1]}  ({N_LANDMARKS} landmarks x 3 coords)")
    counts = np.bincount(y)
    print("  Per-class   : "
          + "  ".join(f"{e}={n}" for e, n in zip(EXPRESSIONS, counts)))

    # ── 3. Classifiers ────────────────────────────────────────────────────────
    classifiers = {
        "SVM (linear, C=1)":
            LinearSVC(C=1, max_iter=5000, random_state=42),

        "k-NN (k=5, Euclidean)":
            KNeighborsClassifier(n_neighbors=5, metric="euclidean", n_jobs=-1),

        "Random Forest (100 trees)":
            RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    }

    # ── 4. LOSO cross-validation ──────────────────────────────────────────────
    summary = {}
    for clf_name, clf in classifiers.items():
        acc, true, preds = loso_eval(X, y, subjects, clf_name, clf)
        summary[clf_name] = acc
        plot_confusion_matrix(true, preds, clf_name, mode, save_dir)

    # ── 5. Summary ────────────────────────────────────────────────────────────
    bar = "=" * 64
    print(f"\n{bar}")
    print(f"  SUMMARY  –  mode={mode}  ({mode_labels[mode]})")
    print(bar)
    best = max(summary, key=summary.get)
    for clf_name, acc in summary.items():
        tag = "  <- best" if clf_name == best else ""
        print(f"  {clf_name:<32}  {acc:.4f}{tag}")
    print(bar)


if __name__ == "__main__":
    main()
