#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualization utilities for:
- dataset → marker mapping
- cluster → color mapping
- grayscale mapping for ICML black-white figures
- PSD ± IQR shading
- PCA / t-SNE interactive visualizations (Plotly)
"""

import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import os


# =========================================================
#                 MARKER MAP (dataset → marker)
# =========================================================

def build_marker_map(datasets):
    """
    Assign distinct markers to each dataset.
    """
    markers = ['o', 's', '^', 'D', 'P', 'X', '*', 'v', '<', '>']
    marker_map = {ds: markers[i % len(markers)] for i, ds in enumerate(datasets)}
    return marker_map


# =========================================================
#                 COLOR MAP (cluster → color)
# =========================================================

def build_color_map(n_clusters):
    """
    cluster → color (tab10)
    """
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    color_map = {cid: colors[cid] for cid in range(n_clusters)}
    return color_map


# =========================================================
#          GRAYSCALE MAP (cluster → black-white)
# =========================================================

def build_grayscale_map(n_clusters):
    """
    For ICML-style black-and-white print-friendly figures.
    cluster0 = darkest → clusterN = lightest
    """
    gray_levels = np.linspace(0.15, 0.85, n_clusters)  # avoid pure white/black
    gray_map = {cid: (gray_levels[cid], gray_levels[cid], gray_levels[cid])
                for cid in range(n_clusters)}
    return gray_map


# =========================================================
#              PSD ± IQR Plot for each cluster
# =========================================================

def plot_psd_iqr_per_cluster(save_dir,
                             freqs,
                             psd_vectors,
                             cluster_id,
                             n_clusters,
                             max_freq=40):
    """
    psd_vectors : [N, F]
    cluster_id  : [N]
    Generates psd_cluster_iqr_plot_clusterX.png
    """

    os.makedirs(save_dir, exist_ok=True)

    freq_mask = freqs <= max_freq

    for cid in range(n_clusters):
        idx = (cluster_id == cid)
        if np.sum(idx) == 0:
            continue

        # 自动对齐 PSD 频率维度与 freq_mask 长度
        freq_dim_psd = psd_vectors.shape[1]
        freq_dim_mask = freq_mask.shape[0]

        if freq_dim_psd != freq_dim_mask:
            print(f"[警告] PSD 和 freq_mask 维度不一致: PSD={freq_dim_psd}, mask={freq_dim_mask}")
            min_dim = min(freq_dim_psd, freq_dim_mask)
            psd_vectors = psd_vectors[:, :min_dim]
            freq_mask = freq_mask[:min_dim]

        idx = np.where(cluster_id == cid)[0]
        psd_c = psd_vectors[idx][:, freq_mask]

        median = np.median(psd_c, axis=0)
        q1 = np.percentile(psd_c, 25, axis=0)
        q3 = np.percentile(psd_c, 75, axis=0)

        plt.figure(figsize=(7,5))
        plt.title(f"Cluster {cid}: PSD Median ± IQR")
        plt.fill_between(freqs[freq_mask], q1, q3, alpha=0.3, label='IQR')
        plt.plot(freqs[freq_mask], median, linewidth=2, label='Median PSD')

        plt.xlabel("Freq (Hz)")
        plt.ylabel("Power")
        plt.yscale("log")
        plt.tight_layout()

        out_path = os.path.join(save_dir, f"psd_cluster_{cid}_iqr.png")
        plt.savefig(out_path, dpi=300)
        plt.close()

        print(f"[INFO] PSD-IQR figure saved:", out_path)


# =========================================================
#                  PCA Interactive Plot (Plotly)
# =========================================================
def plotly_pca_interactive(save_dir,
                           pca_coords,
                           items,
                           dataset_labels,
                           cluster_id,
                           n_clusters):

    import pandas as pd
    os.makedirs(save_dir, exist_ok=True)

    N = pca_coords.shape[0]
    assert len(items) == N, f"items mismatch: {len(items)} != {N}"
    assert len(dataset_labels) == N, f"dataset mismatch: {len(dataset_labels)} != {N}"
    assert len(cluster_id) == N, f"cluster mismatch: {len(cluster_id)} != {N}"

    df = pd.DataFrame({
        "PC1": pca_coords[:,0],
        "PC2": pca_coords[:,1],
        "dataset": dataset_labels,
        "cluster": cluster_id,
        "item": items
    })

    fig = px.scatter(
        df,
        x="PC1",
        y="PC2",
        color="cluster",
        symbol="dataset",
        hover_data=["item", "dataset", "cluster"],
        title="Interactive PCA (dataset=marker, cluster=color)",
        width=800,
        height=650
    )

    out_path = os.path.join(save_dir, "pca_interactive.html")
    fig.write_html(out_path)
    print("[INFO] Saved interactive PCA:", out_path)



# =========================================================
#             t-SNE Interactive Plot (Plotly)
# =========================================================

def plotly_tsne_interactive(save_dir,
                            tsne_coords,
                            items,
                            dataset_labels,
                            cluster_id):

    import pandas as pd

    os.makedirs(save_dir, exist_ok=True)

    # ---- 长度一致性检查 ----
    N = tsne_coords.shape[0]
    assert len(items) == N, f"items length mismatch: {len(items)} != {N}"
    assert len(dataset_labels) == N, f"dataset_labels length mismatch: {len(dataset_labels)} != {N}"
    assert len(cluster_id) == N, f"cluster_id length mismatch: {len(cluster_id)} != {N}"

    # ---- 构建 DataFrame ----
    df = pd.DataFrame({
        "TSNE1": tsne_coords[:,0],
        "TSNE2": tsne_coords[:,1],
        "dataset": dataset_labels,
        "cluster": cluster_id,
        "item": items
    })

    fig = px.scatter(
        df,
        x="TSNE1",
        y="TSNE2",
        color="cluster",
        symbol="dataset",
        hover_data=["item", "dataset", "cluster"],
        title="Interactive t-SNE (dataset=marker, cluster=color)",
        width=800,
        height=650
    )

    out_path = os.path.join(save_dir, "tsne_interactive.html")
    fig.write_html(out_path)
    print("[INFO] Saved interactive t-SNE:", out_path)



# =========================================================
#               PCA Black-White ICML-friendly
# =========================================================

def plot_pca_blackwhite(save_dir,
                        pca_coords,
                        items,
                        dataset_labels,
                        cluster_id,
                        n_clusters):
    """
    dataset → marker
    cluster → grayscale
    """
    os.makedirs(save_dir, exist_ok=True)

    datasets = sorted(list(set(dataset_labels)))

    # marker map
    marker_map = build_marker_map(datasets)

    # grayscale map
    gray_map = build_grayscale_map(n_clusters)

    plt.figure(figsize=(6,5))

    for i in range(len(items)):
        ds = dataset_labels[i]
        cid = cluster_id[i]
        plt.scatter(
            pca_coords[i,0], pca_coords[i,1],
            c=[gray_map[cid]],
            marker=marker_map[ds],
            s=25,
            edgecolor='black',
            linewidth=0.3
        )

    # legend for dataset (marker only)
    for ds in datasets:
        plt.scatter([],[],c='white',marker=marker_map[ds],edgecolor='black',label=f"{ds}")

    # legend for cluster (grayscale only)
    for cid in range(n_clusters):
        plt.scatter([],[],c=[gray_map[cid]],marker='o',label=f"Cluster{cid}")

    plt.legend(loc='best', fontsize=7)
    plt.title("PCA (Black-White ICML Style)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()

    out_path = os.path.join(save_dir, "pca_blackwhite.png")
    plt.savefig(out_path, dpi=300)
    plt.close()

    print("[INFO] Saved black-white PCA:", out_path)
