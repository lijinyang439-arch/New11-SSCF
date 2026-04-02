#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
generate_cluster_datasets_parallel_v5.py

特点：
- 所有参数直接写在代码里，不用 argparse
- 支持多核并行
- 输出 cluster_0 / cluster_1 / ...，并复制 seq + label
"""

import os
import json
import shutil
from glob import glob
from joblib import Parallel, delayed


# =========================================================
#                直接在这里配置参数（手动模式）
# =========================================================

CLUSTER_DIR = "/data/lijinyang/3_聚类分析/聚类计算/全部数据集聚类结果/K_09"
DATASETS_DIR = "/fdata/lijinyang/datasets_dir2_all"
OUTPUT_DIR = "/fdata/lijinyang/D11_K_08"
N_JOBS = 150


# =========================================================
#            单个 subject 的拷贝任务（可并行）
# =========================================================
def process_subject(item, cid, datasets_dir, out_root):

    dataset_name, subject_id = item.split("/")

    # 原始路径
    seq_dir = os.path.join(datasets_dir, dataset_name, "seq", subject_id)
    label_dir = os.path.join(datasets_dir, dataset_name, "labels", subject_id)

    if not os.path.exists(seq_dir):
        print(f"[WARN] Missing seq: {seq_dir}")
        return

    # 输出路径
    out_subject_dir = os.path.join(out_root, f"cluster_{cid}", f"{dataset_name}_{subject_id}")
    out_seq_dir = os.path.join(out_subject_dir, "seq")
    out_label_dir = os.path.join(out_subject_dir, "labels")

    os.makedirs(out_seq_dir, exist_ok=True)
    os.makedirs(out_label_dir, exist_ok=True)

    # 拷贝 seq
    seq_files = sorted(glob(os.path.join(seq_dir, "*.npy")))
    if len(seq_files) == 0:
        print(f"[WARN] No seq npy: {seq_dir}")
        return

    for f in seq_files:
        shutil.copy(f, os.path.join(out_seq_dir, os.path.basename(f)))

    # 拷贝 label
    if not os.path.exists(label_dir):
        print(f"[WARN] Missing label: {label_dir}")
        return

    label_files = sorted(glob(os.path.join(label_dir, "*.npy")))
    if len(label_files) == 0:
        print(f"[WARN] No label npy: {label_dir}")
        return

    for f in label_files:
        shutil.copy(f, os.path.join(out_label_dir, os.path.basename(f)))

    print(f"[INFO] Copied subject {item} → cluster_{cid}")


# =========================================================
#                           MAIN
# =========================================================
def main():

    print("\n========== SleepHDG Cluster Dataset Generator v5.1 ==========")

    map_path = os.path.join(CLUSTER_DIR, "cluster_map.json")
    if not os.path.exists(map_path):
        raise FileNotFoundError(f"cluster_map.json not found: {map_path}")

    with open(map_path, "r") as f:
        cluster_map = json.load(f)

    print(f"[INFO] Loaded cluster_map.json")
    print(f"[INFO] Total subjects = {len(cluster_map)}")

    cluster_ids = list(cluster_map.values())
    max_cluster = max(cluster_ids)
    print(f"[INFO] Number of clusters = {max_cluster + 1}")

    for cid in range(max_cluster + 1):
        os.makedirs(os.path.join(OUTPUT_DIR, f"cluster_{cid}"), exist_ok=True)

    print(f"[INFO] Output root: {OUTPUT_DIR}")
    print(f"[INFO] Parallel workers = {N_JOBS}")

    Parallel(n_jobs=N_JOBS, backend="multiprocessing")(
        delayed(process_subject)(item, cid, DATASETS_DIR, OUTPUT_DIR)
        for item, cid in cluster_map.items()
    )

    print("\n========== ALL DONE: Cluster datasets generated ==========\n")


if __name__ == "__main__":
    main()
