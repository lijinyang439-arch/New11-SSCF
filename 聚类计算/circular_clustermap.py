#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Circular Clustermap (NO recursion, stable for large N)

This version:
- avoids `dendrogram()` → prevents RecursionError
- uses linkage + leaves_list for ordering
- plots PSD vectors in a circular sorted layout
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist


def plot_circular_clustermap(save_path, data_matrix, row_labels=None):
    """
    data_matrix: [N, F] PSD vectors
    row_labels : N optional labels (unused in plot, only for debugging)
    """

    N = data_matrix.shape[0]
    print(f"[INFO] Circular-clustermap: N={N}")

    # 1. Compute distance
    dist = pdist(data_matrix, metric="euclidean")

    # 2. Hierarchical clustering (NO recursion)
    Z = linkage(dist, method="ward")

    # 3. Get leaf order (NO recursion)
    order = leaves_list(Z)

    # 4. Sort PSD by hierarchical order
    sorted_data = data_matrix[order]

    # 5. Normalize for plotting
    sorted_norm = sorted_data / (np.max(sorted_data) + 1e-8)

    # 6. Circular ring positions
    theta = np.linspace(0, 2*np.pi, N, endpoint=False)
    x = np.cos(theta)
    y = np.sin(theta)

    plt.figure(figsize=(8, 8))
    plt.scatter(x, y, c=sorted_norm.mean(axis=1), cmap="viridis", s=10)
    plt.title("Circular Clustermap (Ward linkage)")
    plt.axis("off")

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("[INFO] Saved circular clustermap:", save_path)
