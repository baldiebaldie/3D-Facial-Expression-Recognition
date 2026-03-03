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
    x  - rotate x    : 180-degree rotation around the x-axis
    y  - rotate y    : 180-degree rotation around the y-axis
    z  - rotate z    : 180-degree rotation around the z-axis

Example:
    python Project1.py t ./BU4DFE_BND_V1.1

Classifier:
    Random Forest (100 trees)

Each .bnd file: 83 rows of  <index>  <x>  <y>  <z>
                (optional empty 84th row is ignored)
"""

import os
import sys
import glob
import time
import warnings
from math import acos, cos, sin

# Force UTF-8 output so Unicode box-drawing characters print on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

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
    'x'  rotate-x   – 180-degree rotation around the x-axis using the
                      rotation matrix specified in the project spec
    'y'  rotate-y   – 180-degree rotation around the y-axis
    'z'  rotate-z   – 180-degree rotation around the z-axis

    PI is approximated per the project spec: round(2 * acos(0.0), 3)
    Each landmark [x, y, z] is treated as a column vector and multiplied
    by the appropriate 3x3 rotation matrix (R @ point).
    """
    # PI approximation as required by the project specification
    PI = round(2 * acos(0.0), 3)

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
        # --- 180-degree rotation around the x-axis ---
        # R_x = [[1,      0,       0    ],
        #         [0,  cos(π),  sin(π)  ],
        #         [0, -sin(π),  cos(π)  ]]
        c, s = cos(PI), sin(PI)
        R_x = np.array([[1,  0,  0],
                         [0,  c,  s],
                         [0, -s,  c]], dtype=np.float32)
        # Apply rotation: each landmark is a row vector; transpose, multiply, transpose back
        return (R_x @ landmarks.T).T

    elif mode == "y":
        # --- 180-degree rotation around the y-axis ---
        # R_y = [[ cos(π),  0, -sin(π)],
        #         [   0,     1,    0   ],
        #         [ sin(π),  0,  cos(π)]]
        c, s = cos(PI), sin(PI)
        R_y = np.array([[ c,  0, -s],
                         [ 0,  1,  0],
                         [ s,  0,  c]], dtype=np.float32)
        return (R_y @ landmarks.T).T

    elif mode == "z":
        # --- 180-degree rotation around the z-axis ---
        # R_z = [[ cos(π),  sin(π),  0],
        #         [-sin(π),  cos(π),  0],
        #         [   0,       0,     1]]
        c, s = cos(PI), sin(PI)
        R_z = np.array([[ c,  s,  0],
                         [-s,  c,  0],
                         [ 0,  0,  1]], dtype=np.float32)
        return (R_z @ landmarks.T).T

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

    subject_dirs = sorted(glob.glob(os.path.join(data_dir, "F*")) +
                          glob.glob(os.path.join(data_dir, "M*")))
    if not subject_dirs:
        raise FileNotFoundError(
            f"No subject directories found under: {data_dir}\n"
            "Expected structure:  <data_dir>/F001/Angry/*.bnd  or  M001/Angry/*.bnd  etc."
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

    Returns (overall_accuracy, all_true_labels, all_predicted_labels,
             fold_rows, elapsed_seconds).
    fold_rows is a list of (subject, n_test_samples, fold_accuracy) tuples.
    """
    unique_subjects = np.unique(subjects)
    all_true, all_preds, fold_rows = [], [], []

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
        fold_rows.append((test_subj, len(y_test), fold_acc))

        print(f"  {test_subj}  |  test samples: {len(y_test):4d}  |  "
              f"acc = {fold_acc:.4f}")

    elapsed = time.time() - t0
    overall = accuracy_score(all_true, all_preds)

    print(f"\n  Elapsed : {elapsed:.1f}s")
    print(f"  Overall LOSO Accuracy : {overall:.4f}")

    return overall, all_true, all_preds, fold_rows, elapsed


# ── Text results writer ───────────────────────────────────────────────────────

def write_results(true, preds, overall_acc, elapsed, mode, mode_label,
                  fold_rows, save_dir):
    """Write per-fold accuracy and confusion matrix to a plain-text table file."""
    cm = confusion_matrix(true, preds)
    report = classification_report(true, preds,
                                   target_names=EXPRESSIONS, digits=4)

    col_w = 10          # width of each expression column in the confusion matrix
    label_w = 10        # width of the row-label column

    out_path = os.path.join(save_dir, f"results_{mode}.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        bar = "=" * 64
        f.write(f"{bar}\n")
        f.write(f"  Random Forest (100 trees)  –  mode={mode}  ({mode_label})\n")
        f.write(f"{bar}\n\n")

        # Per-fold table
        f.write(f"  {'Subject':<10}  {'Test Samples':>12}  {'Accuracy':>10}\n")
        f.write(f"  {'-'*10}  {'-'*12}  {'-'*10}\n")
        for subj, n_test, acc in fold_rows:
            f.write(f"  {subj:<10}  {n_test:>12}  {acc:>10.4f}\n")

        f.write(f"\n  {'Elapsed':<24}: {elapsed:.1f}s\n")
        f.write(f"  {'Overall LOSO Accuracy':<24}: {overall_acc:.4f}\n\n")

        # Classification report
        f.write(f"  Classification Report:\n")
        for line in report.splitlines():
            f.write(f"  {line}\n")

        # Confusion matrix
        f.write(f"\n  Confusion Matrix (rows = true, cols = predicted):\n\n")
        header = f"  {'':{label_w}}" + "".join(
            f"{e:>{col_w}}" for e in EXPRESSIONS)
        f.write(header + "\n")
        f.write(f"  {'-' * (label_w + col_w * len(EXPRESSIONS))}\n")
        for i, expr in enumerate(EXPRESSIONS):
            row = f"  {expr:{label_w}}" + "".join(
                f"{cm[i, j]:>{col_w}}" for j in range(len(EXPRESSIONS)))
            f.write(row + "\n")

        f.write(f"\n{bar}\n")

    print(f"  Results saved -> {out_path}")


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

    # ── 3. LOSO cross-validation (Random Forest) ──────────────────────────────
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    acc, true, preds, fold_rows, elapsed = loso_eval(
        X, y, subjects, "Random Forest (100 trees)", clf)

    # ── 4. Write results to text file ─────────────────────────────────────────
    write_results(true, preds, acc, elapsed, mode, mode_labels[mode],
                  fold_rows, save_dir)


if __name__ == "__main__":
    main()
