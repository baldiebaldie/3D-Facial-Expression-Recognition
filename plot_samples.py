"""
plot_samples.py
Affective Computing - Spring 2026
Bryan Baldie  U44374228
Twinkle Markana  U55075888

Generates 5 3D scatter plots (one per data mode) using a single sample face
from the BU-4DFE dataset. Required for Report Question 4.

Modes plotted:
    o  - original    : raw x, y, z coordinates
    t  - translated  : centred at origin
    x  - rotated x   : 180-degree rotation around x-axis
    y  - rotated y   : 180-degree rotation around y-axis
    z  - rotated z   : 180-degree rotation around z-axis

Usage:
    python plot_samples.py <data_dir>

Example:
    python plot_samples.py ./BU4DFE_BND_V1.1

Output:
    Five PNG files saved alongside this script:
        plot_o.png, plot_t.png, plot_x.png, plot_y.png, plot_z.png
"""

import os
import sys
import glob
from math import acos, cos, sin

import numpy as np
import matplotlib
matplotlib.use("Agg")       # non-interactive backend – safe on all platforms
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – registers 3D projection

# ── Constants (must match Project1.py) ────────────────────────────────────────

N_LANDMARKS = 83
VALID_MODES  = ["o", "t", "x", "y", "z"]

MODE_LABELS = {
    "o": "Original (raw coordinates)",
    "t": "Translated (centred at origin)",
    "x": "Rotated 180° around x-axis",
    "y": "Rotated 180° around y-axis",
    "z": "Rotated 180° around z-axis",
}


# ── .bnd file parsing (matches Project1.py) ───────────────────────────────────

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


# ── Landmark transformations (mirrors Project1.py) ────────────────────────────

def transform_landmarks(landmarks, mode):
    """
    Apply the mode-specific geometric transformation to a (N_LANDMARKS, 3)
    array and return the transformed array (same shape).

    PI is approximated as required by the project spec: round(2*acos(0.0), 3)
    """
    PI = round(2 * acos(0.0), 3)

    if mode == "o":
        return landmarks.copy()

    elif mode == "t":
        # Subtract centroid so the face is centred at (0, 0, 0)
        centroid = landmarks.mean(axis=0)
        return landmarks - centroid

    elif mode == "x":
        # 180-degree rotation around x-axis
        c, s = cos(PI), sin(PI)
        R = np.array([[1,  0,  0],
                      [0,  c,  s],
                      [0, -s,  c]], dtype=np.float32)
        return (R @ landmarks.T).T

    elif mode == "y":
        # 180-degree rotation around y-axis
        c, s = cos(PI), sin(PI)
        R = np.array([[ c,  0, -s],
                      [ 0,  1,  0],
                      [ s,  0,  c]], dtype=np.float32)
        return (R @ landmarks.T).T

    elif mode == "z":
        # 180-degree rotation around z-axis
        c, s = cos(PI), sin(PI)
        R = np.array([[ c,  s,  0],
                      [-s,  c,  0],
                      [ 0,  0,  1]], dtype=np.float32)
        return (R @ landmarks.T).T

    else:
        raise ValueError(f"Unknown mode '{mode}'")


# ── Find one sample .bnd file ─────────────────────────────────────────────────

def find_sample_bnd(data_dir):
    """
    Return the path to the first .bnd file found under data_dir.
    Searches F* then M* subject directories, first expression folder found.
    """
    for prefix in ("F*", "M*"):
        for subj_dir in sorted(glob.glob(os.path.join(data_dir, prefix))):
            for expr_dir in sorted(os.scandir(subj_dir),
                                   key=lambda e: e.name):
                if not expr_dir.is_dir():
                    continue
                bnd_files = sorted(glob.glob(
                    os.path.join(expr_dir.path, "*.bnd")))
                if bnd_files:
                    return bnd_files[0]
    return None


# ── 3D scatter plot ───────────────────────────────────────────────────────────

def plot_landmarks_3d(landmarks, mode, save_dir):
    """
    Create and save a 3D scatter plot of the 83 facial landmarks.

    Parameters
    ----------
    landmarks : ndarray, shape (N_LANDMARKS, 3)  – already transformed
    mode      : str  – one of 'o', 't', 'x', 'y', 'z'
    save_dir  : str  – directory where the PNG is saved
    """
    x, y, z = landmarks[:, 0], landmarks[:, 1], landmarks[:, 2]

    fig = plt.figure(figsize=(7, 6))
    ax  = fig.add_subplot(111, projection="3d")

    ax.scatter(x, y, z, c="steelblue", s=30, depthshade=True)

    # Annotate each landmark with its 1-based index
    for idx in range(len(x)):
        ax.text(x[idx], y[idx], z[idx], str(idx + 1),
                fontsize=5, color="dimgray")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Mode '{mode}' – {MODE_LABELS[mode]}\n"
                 f"(83 facial landmarks, 1 sample)")

    out_path = os.path.join(save_dir, f"plot_{mode}.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved -> {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_samples.py <data_dir>")
        print("Example: python plot_samples.py ./BU4DFE_BND_V1.1")
        sys.exit(1)

    data_dir = sys.argv[1]
    if not os.path.isdir(data_dir):
        print(f"Error: data directory not found: {data_dir}")
        sys.exit(1)

    save_dir = os.path.dirname(os.path.abspath(__file__))

    # Find one raw sample to use for all plots
    sample_path = find_sample_bnd(data_dir)
    if sample_path is None:
        print("Error: no .bnd files found under data_dir.")
        sys.exit(1)

    raw_landmarks = parse_bnd(sample_path)
    if raw_landmarks is None:
        print(f"Error: could not parse {sample_path}")
        sys.exit(1)

    print(f"Sample file : {sample_path}")
    print(f"Saving plots to : {save_dir}\n")

    # Generate one plot per mode
    for mode in VALID_MODES:
        transformed = transform_landmarks(raw_landmarks, mode)
        plot_landmarks_3d(transformed, mode, save_dir)

    print("\nDone. 5 plots saved.")


if __name__ == "__main__":
    main()
