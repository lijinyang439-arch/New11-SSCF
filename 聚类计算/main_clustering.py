#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main clustering script:
- subject-level PSD extraction (EEG only)
- outlier removal (IQR-based)
- PCA + KMeans clustering
- save all data for visualization

Outputs include:
    pca_coords.npy
    items.npy
    dataset_labels.npy
    subject_labels.npy
    cluster_id.npy
    psd_vectors.npy
    freqs.npy
    cluster_map.json
"""

import os
import json
import numpy as np
from glob import glob
from types import SimpleNamespace

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from datasets.dataset import LoadDataset
from utils_psd import compute_psd, remove_outliers


# =========================================================
#     SUBJECT-LEVEL PSD EXTRACTION (EEG ONLY)
# =========================================================

def compute_psd_subject_level(datasets_dir, dataset_list, max_files=999999):
    """
    Return:
        subject_items = list of "Dataset/SubjectID"
        psd_vectors   = array [N_subjects, F]
        dataset_labels = per subject
        subject_labels = per subject
        freqs
    """

    subject_items = []
    psd_vectors = []
    dataset_labels = []
    subject_labels = []

    fs = 100
    freqs_ref = None

    for name in dataset_list:
        print(f"[INFO] Computing subject-level EEG PSD for: {name}")

        root = os.path.join(datasets_dir, name, "seq")
        subject_folders = sorted(os.listdir(root))

        for subject_id in subject_folders:
            subject_path = os.path.join(root, subject_id)
            npy_files = sorted(glob(os.path.join(subject_path, "*.npy")))
            npy_files = npy_files[:max_files]

            all_psd = []

            for fpath in npy_files:
                # arr shape: (T, C, L)
                arr = np.load(fpath)

                # --------- 🚀 只保留 EEG 通道 (channel 0) ---------
                eeg_segments = arr[:, 0, :]          # shape (T, L)

                for seg in eeg_segments:
                    freqs, Pxx = compute_psd(seg, fs)

                    if freqs_ref is None:
                        freqs_ref = freqs

                    all_psd.append(Pxx)

            if len(all_psd) == 0:
                continue

            all_psd = np.stack(all_psd, axis=0)
            avg_psd = np.mean(all_psd, axis=0)

            # store
            subject_items.append(f"{name}/{subject_id}")
            psd_vectors.append(avg_psd)
            dataset_labels.append(name)
            subject_labels.append(subject_id)

    return (
        subject_items,
        np.array(psd_vectors),
        np.array(dataset_labels),
        np.array(subject_labels),
        freqs_ref
    )


# =========================================================
#                     MAIN PIPELINE
# =========================================================

def main():

    datasets_dir = "/data/lijinyang/PSG_ALL_DATA/datasets_dir2"

    selected_datasets = [
        "sleep-edfx", "HMC", "ISRUC", "SHHS1", "P2018",
        "ABC", "CCSHS", "MROS1", "MROS2", "CFS", "MESA"
    ]

    save_dir = "./11个数据集全部被试的9类聚类结果"
    os.makedirs(save_dir, exist_ok=True)

    n_clusters = 9

    # -----------------------------------------------------
    # Step 1: Extract EEG-only PSD per subject
    # -----------------------------------------------------
    (items,
     vectors,
     dataset_labels,
     subject_labels,
     freqs) = compute_psd_subject_level(
        datasets_dir,
        selected_datasets
    )

    print(f"[INFO] Total subjects loaded: {len(items)}")

    np.save(os.path.join(save_dir, "psd_vectors.npy"), vectors)
    np.save(os.path.join(save_dir, "freqs.npy"), freqs)

    # -----------------------------------------------------
    # Step 2: Remove outliers (IQR)
    # -----------------------------------------------------
    vectors_clean, items_clean, labels_clean = \
        remove_outliers(vectors, items, dataset_labels, save_dir)

    dataset_labels_clean = []
    subject_labels_clean = []

    for it in items_clean:
        ds, sid = it.split("/")
        dataset_labels_clean.append(ds)
        subject_labels_clean.append(sid)

    dataset_labels_clean = np.array(dataset_labels_clean)
    subject_labels_clean = np.array(subject_labels_clean)

    print(f"[INFO] After outlier removal: {len(items_clean)} subjects")

    # -----------------------------------------------------
    # Step 3: PCA
    # -----------------------------------------------------
    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(vectors_clean)

    # -----------------------------------------------------
    # Step 4: KMeans clustering
    # -----------------------------------------------------
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    cluster_id = kmeans.fit_predict(vectors_clean)

    # -----------------------------------------------------
    # Step 5: Save all results
    # -----------------------------------------------------
    np.save(os.path.join(save_dir, "pca_coords.npy"), pca_coords)
    np.save(os.path.join(save_dir, "items.npy"), np.array(items_clean))
    np.save(os.path.join(save_dir, "dataset_labels.npy"), dataset_labels_clean)
    np.save(os.path.join(save_dir, "subject_labels.npy"), subject_labels_clean)
    np.save(os.path.join(save_dir, "cluster_id.npy"), cluster_id)

    print("[INFO] Saved PCA, labels, clusters.")

    # Save cluster map JSON
    cluster_map = {it: int(cid) for it, cid in zip(items_clean, cluster_id)}
    with open(os.path.join(save_dir, "cluster_map.json"), "w") as f:
        json.dump(cluster_map, f, indent=2)

    # Save cluster log
    with open(os.path.join(save_dir, "cluster_log.txt"), "w") as f:
        for cid in range(n_clusters):
            f.write(f"\n=== Cluster {cid} ===\n")
            for it, c in zip(items_clean, cluster_id):
                if c == cid:
                    f.write(f"{it}\n")

    print("\n=== DONE: EEG-only clustering complete ===")


if __name__ == "__main__":
    main()
