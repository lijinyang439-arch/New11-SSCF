#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
read_and_plot_batch_parallel_no_tsne.py

批量并行处理多个 K_XX 聚类结果文件夹。
- 不使用 t-SNE
- 支持根据 outliers.txt 和 PCA 极端点裁剪
- 包含：全局图例 PCA、黑白 PCA、数据集聚焦分布图、交互式绘图、PSD 统计图
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from joblib import Parallel, delayed

from utils_visual import (
    build_marker_map,
    build_color_map,
    build_grayscale_map,
    plot_psd_iqr_per_cluster,
    plotly_pca_interactive
)
from circular_clustermap import plot_circular_clustermap


# =========================================================
#            单个 K 文件夹处理逻辑 (完整版)
# =========================================================

def process_single_folder(save_dir):
    print(f"\n[INFO] Processing folder → {save_dir}")

    # 1. 加载数据
    pca_coords = np.load(os.path.join(save_dir, "pca_coords.npy"))
    items = np.load(os.path.join(save_dir, "items_clean.npy"), allow_pickle=True)
    dataset_labels = np.load(os.path.join(save_dir, "dataset_labels_clean.npy"), allow_pickle=True)
    cluster_id = np.load(os.path.join(save_dir, "cluster_id.npy"))
    psd_vectors = np.load(os.path.join(save_dir, "psd_vectors_clean.npy"))
    freqs = np.load(os.path.join(save_dir, "freqs.npy"))

    # 由 items 重建 subject_labels（永远保持对齐）
    subject_labels = np.array([item.split("/", 1)[1] for item in items])

    # 2. 处理剔除逻辑 (Outliers + PCA 极端点)
    outlier_path = os.path.join(save_dir, "outliers.txt")
    outlier_set = set()
    if os.path.exists(outlier_path):
        with open(outlier_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2: outlier_set.add(parts[1])

    vis_mask = np.ones(len(items), dtype=bool)
    if len(outlier_set) > 0:
        vis_mask &= np.array([item not in outlier_set for item in items])

    PCA_TRIM_PERCENT = 99.5
    pca_dist = np.linalg.norm(pca_coords, axis=1)
    vis_mask &= (pca_dist <= np.percentile(pca_dist, PCA_TRIM_PERCENT))

    # 应用 Mask 过滤数据
    items = items[vis_mask]
    dataset_labels = dataset_labels[vis_mask]
    subject_labels = subject_labels[vis_mask]
    cluster_id = cluster_id[vis_mask]
    pca_coords = pca_coords[vis_mask]
    psd_vectors = psd_vectors[vis_mask]

    # 聚类标签重映射 (保证 0,1,2... 连续)
    unique_cids = np.unique(cluster_id)
    cid_map = {old: new for new, old in enumerate(unique_cids)}
    cluster_id = np.array([cid_map[c] for c in cluster_id], dtype=int)

    n_clusters = len(unique_cids)
    datasets = sorted(list(set(dataset_labels)))

    # 3. 绘图通用配置
    marker_map = build_marker_map(datasets)
    color_map = build_color_map(n_clusters)
    gray_map = build_grayscale_map(n_clusters)
    POINT_SIZE = 6
    EDGE_WIDTH = 0.2
    FIGSIZE = (10, 8)

    # -----------------------------------------------------
    # A. PCA Before Clustering (Gray by default)
    # -----------------------------------------------------
    plt.figure(figsize=FIGSIZE)
    for i in range(len(items)):
        plt.scatter(pca_coords[i, 0], pca_coords[i, 1], c=[[0.6, 0.6, 0.6]],
                    marker=marker_map[dataset_labels[i]], s=POINT_SIZE, edgecolor='black', linewidth=EDGE_WIDTH)
    for ds in datasets:
        plt.scatter([], [], c='gray', marker=marker_map[ds], label=ds, s=POINT_SIZE + 20)
    plt.title("PCA Before Clustering\n(dataset=marker, color=gray)")
    plt.legend(fontsize=8, loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "pca_before_clustering.png"), dpi=300)
    plt.close()

    # -----------------------------------------------------
    # B. PCA After Clustering (With Full Legends)
    # -----------------------------------------------------
    plt.figure(figsize=FIGSIZE)
    for i in range(len(items)):
        plt.scatter(pca_coords[i, 0], pca_coords[i, 1], c=[color_map[cluster_id[i]]],
                    marker=marker_map[dataset_labels[i]], s=POINT_SIZE, edgecolor='black', linewidth=EDGE_WIDTH)
    # 形状图例
    for ds in datasets:
        plt.scatter([], [], c='white', edgecolor='black', marker=marker_map[ds], label=ds, s=POINT_SIZE + 20)
    # 颜色图例
    for cid in range(n_clusters):
        plt.scatter([], [], c=[color_map[cid]], marker='o', label=f"Cluster {cid}", s=POINT_SIZE + 20)
    plt.title("PCA After Clustering\n(dataset=marker, cluster=color)")
    plt.legend(fontsize=8, loc='best', ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "pca_after_clustering.png"), dpi=300)
    plt.close()

    # -----------------------------------------------------
    # C. ICML Black & White PCA
    # -----------------------------------------------------
    plt.figure(figsize=FIGSIZE)
    for i in range(len(items)):
        plt.scatter(pca_coords[i, 0], pca_coords[i, 1], c=[gray_map[cluster_id[i]]],
                    marker=marker_map[dataset_labels[i]], s=POINT_SIZE, edgecolor='black', linewidth=EDGE_WIDTH)
    for ds in datasets:
        plt.scatter([], [], c='white', edgecolor='black', marker=marker_map[ds], label=ds, s=POINT_SIZE + 20)
    for cid in range(n_clusters):
        plt.scatter([], [], c=[gray_map[cid]], marker='o', label=f"Cluster {cid}", s=POINT_SIZE + 20)
    plt.title("ICML Black-White PCA")
    plt.legend(fontsize=8, loc='best', ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "pca_blackwhite.png"), dpi=300)
    plt.close()

    # -----------------------------------------------------
    # D. Dataset-Specific Focused PCA (Focus on distribution)
    # -----------------------------------------------------
    focus_dir = os.path.join(save_dir, "dataset_focus")
    os.makedirs(focus_dir, exist_ok=True)
    for target_ds in datasets:
        plt.figure(figsize=FIGSIZE)
        # 背景层
        bg_mask = (dataset_labels != target_ds)
        plt.scatter(pca_coords[bg_mask, 0], pca_coords[bg_mask, 1], c='#E0E0E0', marker='o', s=POINT_SIZE - 2,
                    alpha=0.15)
        # 前景层
        fg_mask = (dataset_labels == target_ds)
        fg_coords = pca_coords[fg_mask];
        fg_clusters = cluster_id[fg_mask]
        ds_stats = []
        for c_idx in range(n_clusters):
            cnt = np.sum(fg_clusters == c_idx)
            ds_stats.append(f"C{c_idx}:{cnt}")
            sub_mask = (fg_clusters == c_idx)
            plt.scatter(fg_coords[sub_mask, 0], fg_coords[sub_mask, 1], c=[color_map[c_idx]],
                        marker=marker_map[target_ds], s=POINT_SIZE + 4, alpha=0.9, edgecolor='black',
                        linewidth=EDGE_WIDTH, label=f"Cluster {c_idx} (n={cnt})")
        plt.title(f"Focus: {target_ds}\nDistribution: {' | '.join(ds_stats)}")
        plt.legend(loc='upper right', fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(focus_dir, f"focus_{target_ds}.png"), dpi=200)
        plt.close()

    # -----------------------------------------------------
    # E. Circular Clustermap & Interactive & PSD
    # -----------------------------------------------------
    plot_circular_clustermap(os.path.join(save_dir, "circular_clustermap.png"), psd_vectors, subject_labels)
    plotly_pca_interactive(save_dir, pca_coords, items, dataset_labels, cluster_id, n_clusters)
    plot_psd_iqr_per_cluster(save_dir, freqs, psd_vectors, cluster_id, n_clusters, max_freq=40)

    # -----------------------------------------------------
    # F. Summary Table
    # -----------------------------------------------------
    summary = defaultdict(lambda: [0] * n_clusters)
    for ds, cid in zip(dataset_labels, cluster_id):
        summary[ds][cid] += 1
    with open(os.path.join(save_dir, "cluster_summary.txt"), "w") as f:
        f.write("dataset\t" + "\t".join([f"cluster{c}" for c in range(n_clusters)]) + "\n")
        for ds in datasets:
            row = "\t".join(str(summary[ds][c]) for c in range(n_clusters))
            f.write(ds + "\t" + row + "\n")

    return f"[DONE] {save_dir}"


# =========================================================
#                     并行主入口
# =========================================================

def main():
    ROOT = "/data/lijinyang/3_聚类分析/聚类计算/全部数据集聚类结果-新11个大数据集-新方法先log再中心"
    N_JOBS = 64
    K_dirs = sorted([os.path.join(ROOT, d) for d in os.listdir(ROOT)
                     if os.path.isdir(os.path.join(ROOT, d)) and d.startswith("K_")])

    print(f"\n[INFO] Starting parallel processing for {len(K_dirs)} folders...")
    results = Parallel(n_jobs=N_JOBS)(delayed(process_single_folder)(kdir) for kdir in K_dirs)
    print("\n============ ALL JOBS DONE ============")


if __name__ == "__main__":
    main()