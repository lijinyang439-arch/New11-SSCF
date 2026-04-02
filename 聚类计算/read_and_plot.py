#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
read_and_plot.py

Reads output from main_clustering.py and generates:
- PCA before clustering
- PCA after clustering
- ICML black-white PCA
- t-SNE
- Plotly interactive PCA
- Plotly interactive t-SNE
- PSD ± IQR per cluster
- cluster summary table
- Circular Clustermap  ← 新增
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.manifold import TSNE

from utils_visual import (
    build_marker_map,
    build_color_map,
    build_grayscale_map,
    plot_psd_iqr_per_cluster,
    plotly_pca_interactive,
    plotly_tsne_interactive
)

from circular_clustermap import plot_circular_clustermap


# =========================================================
#                  MAIN VISUALIZATION LOGIC
# =========================================================

def main():

    # 你的聚类结果输出路径
    save_dir = "/data/lijinyang/PSG_ALL_DATA/聚类计算/聚类结果/K_02"

    # ------------ Load saved arrays ------------
    pca_coords = np.load(os.path.join(save_dir, "pca_coords.npy"))
    items = np.load(os.path.join(save_dir, "items.npy"), allow_pickle=True)
    dataset_labels = np.load(os.path.join(save_dir, "dataset_labels.npy"), allow_pickle=True)
    subject_labels = np.load(os.path.join(save_dir, "subject_labels.npy"), allow_pickle=True)
    cluster_id = np.load(os.path.join(save_dir, "cluster_id.npy"))
    psd_vectors = np.load(os.path.join(save_dir, "psd_vectors.npy"))
    freqs = np.load(os.path.join(save_dir, "freqs.npy"))

    n_clusters = len(set(cluster_id))
    datasets = sorted(list(set(dataset_labels)))

    marker_map = build_marker_map(datasets)
    color_map = build_color_map(n_clusters)

    # =========================================================
    #    PCA BEFORE CLUSTERING
    # =========================================================

    plt.figure(figsize=(6, 5))
    for i in range(len(items)):
        ds = dataset_labels[i]
        plt.scatter(
            pca_coords[i,0], pca_coords[i,1],
            c=[[0.6,0.6,0.6]], marker=marker_map[ds],
            s=20, edgecolor='black', linewidth=0.3
        )
    plt.title("PCA Before Clustering\n(dataset=marker, color=gray)")
    plt.xlabel("PC1"); plt.ylabel("PC2")

    for ds in datasets:
        plt.scatter([], [], c='gray', marker=marker_map[ds], label=ds, s=35)
    plt.legend(fontsize=7)
    plt.tight_layout()

    out1 = os.path.join(save_dir, "pca_before_clustering.png")
    plt.savefig(out1, dpi=300)
    plt.close()
    print("[INFO] Saved:", out1)

    # =========================================================
    #    PCA AFTER CLUSTERING
    # =========================================================

    plt.figure(figsize=(6, 5))
    for i in range(len(items)):
        cid = cluster_id[i]; ds = dataset_labels[i]
        plt.scatter(
            pca_coords[i,0], pca_coords[i,1],
            c=[color_map[cid]], marker=marker_map[ds],
            s=20, edgecolor='black', linewidth=0.3
        )

    plt.title("PCA After Clustering\n(dataset=marker, cluster=color)")
    plt.xlabel("PC1"); plt.ylabel("PC2")
    for ds in datasets:
        plt.scatter([],[],c='gray',marker=marker_map[ds],label=ds,s=35)
    for cid in range(n_clusters):
        plt.scatter([],[],c=[color_map[cid]],marker='o',label=f"Cluster{cid}",s=35)

    plt.legend(fontsize=7)
    plt.tight_layout()

    out2 = os.path.join(save_dir, "pca_after_clustering.png")
    plt.savefig(out2, dpi=300)
    plt.close()
    print("[INFO] Saved:", out2)

    # =========================================================
    #    ICML BLACK & WHITE PCA
    # =========================================================

    gray_map = build_grayscale_map(n_clusters)

    plt.figure(figsize=(6, 5))
    for i in range(len(items)):
        cid = cluster_id[i]; ds = dataset_labels[i]
        plt.scatter(
            pca_coords[i,0], pca_coords[i,1],
            c=[gray_map[cid]], marker=marker_map[ds],
            s=22, edgecolor='black', linewidth=0.3
        )

    plt.title("ICML Black-White PCA")
    plt.xlabel("PC1"); plt.ylabel("PC2")

    for ds in datasets:
        plt.scatter([],[],c='white',marker=marker_map[ds],edgecolor='black',label=f"{ds}",s=35)
    for cid in range(n_clusters):
        plt.scatter([],[],c=[gray_map[cid]],marker='o',label=f"Cluster{cid}",s=35)

    plt.legend(fontsize=7)
    plt.tight_layout()

    out3 = os.path.join(save_dir, "pca_blackwhite.png")
    plt.savefig(out3, dpi=300)
    plt.close()
    print("[INFO] Saved:", out3)

    # =========================================================
    #    NEW: CIRCULAR CLUSTERMAP
    # =========================================================

    circular_path = os.path.join(save_dir, "circular_clustermap.png")
    plot_circular_clustermap(
        save_path=circular_path,
        data_matrix=psd_vectors,
        row_labels=subject_labels
    )

    # =========================================================
    #    t-SNE
    # =========================================================

    print("[INFO] Running t-SNE... (~10-30 sec)")

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate=200,
        max_iter=1000,
        init="random",
        random_state=0,
    )
    tsne_coords = tsne.fit_transform(psd_vectors)
    np.save(os.path.join(save_dir, "tsne_coords.npy"), tsne_coords)

    plt.figure(figsize=(6,5))
    for i in range(len(items)):
        cid = cluster_id[i]; ds = dataset_labels[i]
        plt.scatter(
            tsne_coords[i,0], tsne_coords[i,1],
            c=[color_map[cid]], marker=marker_map[ds],
            s=20, edgecolor='black', linewidth=0.3
        )

    plt.title("t-SNE (dataset=marker, cluster=color)")
    plt.xlabel("TSNE1"); plt.ylabel("TSNE2")
    plt.tight_layout()

    out4 = os.path.join(save_dir, "tsne_plot.png")
    plt.savefig(out4, dpi=300)
    plt.close()
    print("[INFO] Saved:", out4)

    # =========================================================
    #    INTERACTIVE PLOTS
    # =========================================================

    plotly_pca_interactive(save_dir, pca_coords, items, dataset_labels, cluster_id, n_clusters)
    plotly_tsne_interactive(save_dir, tsne_coords, items, dataset_labels, cluster_id)

    # =========================================================
    #    PSD IQR PLOTS
    # =========================================================

    plot_psd_iqr_per_cluster(
        save_dir,
        freqs,
        psd_vectors,
        cluster_id,
        n_clusters,
        max_freq=40
    )

    # =========================================================
    #    SUMMARY TABLE
    # =========================================================

    summary = defaultdict(lambda: [0] * n_clusters)
    for ds, cid in zip(dataset_labels, cluster_id):
        summary[ds][cid] += 1

    table_path = os.path.join(save_dir, "cluster_summary.txt")
    with open(table_path, "w") as f:
        header = "dataset\t" + "\t".join([f"cluster{c}" for c in range(n_clusters)])
        f.write(header + "\n")
        for ds in datasets:
            row = "\t".join(str(summary[ds][c]) for c in range(n_clusters))
            f.write(ds + "\t" + row + "\n")

    print("[INFO] Summary table saved:", table_path)

    # dataset × cluster CSV
    import pandas as pd

    df = pd.DataFrame({"dataset": dataset_labels, "cluster": cluster_id})
    table = df.groupby(["dataset", "cluster"]).size().unstack(fill_value=0)
    table.to_csv(os.path.join(save_dir, "dataset_cluster_distribution.csv"))

    print("[INFO] Saved dataset × cluster distribution → dataset_cluster_distribution.csv")


# =========================================================
#                       ENTRY
# =========================================================

if __name__ == "__main__":
    main()
