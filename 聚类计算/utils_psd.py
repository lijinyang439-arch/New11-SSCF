#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Utility functions for:
- PSD computation
- spectral centroid
- outlier removal (IQR-based)
"""

import numpy as np
import os
from sklearn.decomposition import PCA


# =========================================================
#                   PSD COMPUTATION
# =========================================================

def compute_psd(x, fs=100):
    """
    Compute PSD of one epoch (1D signal)
    """
    x = np.asarray(x, dtype=np.float64)
    win = np.hanning(len(x))
    X = np.fft.rfft(x * win)
    Pxx = (1/(fs*np.sum(win**2))) * (np.abs(X)**2)
    freqs = np.fft.rfftfreq(len(x), 1/fs)
    return freqs, Pxx


def compute_centroid(freqs, Pxx):
    """
    Spectral centroid
    """
    if np.sum(Pxx) == 0:
        return 0.0
    return float(np.sum(freqs * Pxx) / np.sum(Pxx))


# =========================================================
#                OUTLIER REMOVAL (IQR)
# =========================================================
def remove_outliers(vectors, items, labels,
                    save_dir,
                    factor=2.5,
                    max_remove_ratio=0):
    """
    高维 IQR outlier removal
    返回：vectors_clean, items_clean, labels_clean, keep_mask
    """

    print("[INFO] Removing outliers (HIGH-DIM IQR)...")
    os.makedirs(save_dir, exist_ok=True)

    mean_vec = np.mean(vectors, axis=0)
    dist = np.sqrt(np.sum((vectors - mean_vec) ** 2, axis=1))

    Q1 = np.percentile(dist, 25)
    Q3 = np.percentile(dist, 75)
    IQR = Q3 - Q1
    upper = Q3 + factor * IQR

    candidate = dist > upper

    max_remove = int(len(dist) * max_remove_ratio)
    idx_sorted = np.argsort(dist)[::-1]

    outlier_idx = np.zeros_like(candidate, dtype=bool)
    removed = 0
    for idx in idx_sorted:
        if candidate[idx] and removed < max_remove:
            outlier_idx[idx] = True
            removed += 1

    print(f"[INFO] Outliers removed: {removed} / {len(items)}")

    out_path = os.path.join(save_dir, "outliers.txt")
    with open(out_path, "w") as f:
        for i, flag in enumerate(outlier_idx):
            if flag:
                f.write(f"{i}\t{items[i]}\n")

    keep = ~outlier_idx

    vectors_clean = vectors[keep]
    items_clean = [items[i] for i in range(len(items)) if keep[i]]
    labels_clean = [labels[i] for i in range(len(labels)) if keep[i]]

    return vectors_clean, items_clean, labels_clean, keep


