#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cluster-Builder v6.3 (Direct Compute, No Copy, Legacy Output Compatible)
-----------------------------------------------------------------------
功能：
1) 使用 cluster_map.json 进行逻辑筛选（不 copy 数据）
2) 直接在 datasets_dir2_all 上提取特征
3) 全局一次 KMeans
4) 输出文件格式与旧版完全一致：
   - ours-11_centers.npy
   - ours-11_map.json
"""

# =========================================================
#                 【参数区：只改这里】
# =========================================================

CONFIG_PATH = "/data/lijinyang/1_SleepHDG主要运行代码/wujiHDG/results/sleepdg_original_psd_run_2025-12-17_12-02-45/mian/configs/config_ours8.yaml"

CHECKPOINT_PATH = "/data/lijinyang/1_SleepHDG主要运行代码/wujiHDG/results/sleepdg_original_psd_run_2025-12-17_12-02-45/fold0/best_val_ep168_vacc_0.81515_vf1_0.78063.pth"

DATA_ROOT = "/fdata/lijinyang/datasets_dir2_all"

DATASETS = [
    "ABC", "CCSHS", "CFS", "HMC", "ISRUC",
    "MESA", "SHHS1", "MROS1", "MROS2", "P2018", "sleep-edfx"
]

CLUSTER_MAP_PATH = "/data/lijinyang/3_聚类分析/聚类计算/全部数据集聚类结果/K_09/cluster_map.json"

N_CLUSTERS = 8
OUTPUT_DIR = "./cluster_output_direct"
PREFIX = "ours-8"

BATCH_SIZE = 64
NUM_WORKERS = 8
DEVICE = "cuda"

# =========================================================
#                    环境 & 依赖
# =========================================================

import os
import json
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans
from types import SimpleNamespace
import importlib
from torch.utils.data import Dataset, DataLoader

os.environ["OMP_NUM_THREADS"] = "1"
import threadpoolctl
threadpoolctl.threadpool_limits(limits=1)


# =========================================================
#                         Dataset
# =========================================================

class CustomDataset(Dataset):
    def __init__(self, seqs_labels_path_pair, cluster_map):
        super(CustomDataset, self).__init__()
        self.file_list = seqs_labels_path_pair
        self.cluster_map = cluster_map
        self.samples_map = []

        print(f"[Dataset] Scanning {len(self.file_list)} files for indexing...")
        for file_idx, (seq_path, _, _) in enumerate(self.file_list):
            try:
                data_ref = np.load(seq_path, mmap_mode='r')
                if data_ref.ndim == 3:
                    self.samples_map.append((file_idx, -1))
                else:
                    for inner_idx in range(data_ref.shape[0]):
                        self.samples_map.append((file_idx, inner_idx))
            except Exception:
                pass

        print(f"[Dataset] Valid samples: {len(self.samples_map)}")

    def __len__(self):
        return len(self.samples_map)

    def __getitem__(self, idx):
        file_idx, inner_idx = self.samples_map[idx]
        seq_path, label_path, subject_key = self.file_list[file_idx]

        try:
            data_mmap = np.load(seq_path, mmap_mode='r')
            label_mmap = np.load(label_path, mmap_mode='r')

            if inner_idx == -1:
                seq = data_mmap.copy()
                label = label_mmap.copy()
            else:
                seq = data_mmap[inner_idx].copy()
                label = label_mmap[inner_idx].copy()

            if np.any(label == 5):
                return None

            if subject_key not in self.cluster_map:
                return None

            return seq.astype(np.float32)

        except Exception:
            return None

    def collate(self, batch):
        batch = [b for b in batch if b is not None]
        if not batch:
            return None
        return torch.stack([torch.from_numpy(b) for b in batch]).float()


# =========================================================
#                     数据扫描
# =========================================================

def scan_files_recursively(data_root, datasets_list):
    all_pairs = []

    for ds_name in datasets_list:
        ds_dir = os.path.join(data_root, ds_name)
        if not os.path.exists(ds_dir):
            continue

        for root, dirs, _ in os.walk(ds_dir):
            if 'seq' in dirs and 'labels' in dirs:
                seq_base = os.path.join(root, 'seq')
                label_base = os.path.join(root, 'labels')

                for sub_root, _, files in os.walk(seq_base):
                    for f in files:
                        if not f.endswith(".npy"):
                            continue
                        seq_path = os.path.join(sub_root, f)
                        rel = os.path.relpath(seq_path, seq_base)
                        label_path = os.path.join(label_base, rel)
                        if not os.path.exists(label_path):
                            continue

                        subject_id = os.path.basename(os.path.dirname(seq_path))
                        subject_key = f"{ds_name}/{subject_id}"
                        all_pairs.append((seq_path, label_path, subject_key))

    return all_pairs


# =========================================================
#                   模型 & 特征
# =========================================================

def load_trained_model(config_path, checkpoint_path, device):
    with open(config_path, 'r') as f:
        params = SimpleNamespace(**yaml.safe_load(f))

    if hasattr(params, 'align'):
        params.align = {}
    params.external_centers = None

    model = importlib.import_module('original.models.model').Model(params)

    ckpt = torch.load(
        checkpoint_path,
        map_location='cpu',
        weights_only=False
    )

    state = ckpt.get('model_state', ckpt)
    model.load_state_dict(
        {k.replace("module.", ""): v for k, v in state.items()},
        strict=False
    )

    model.to(device).eval()
    return model


@torch.no_grad()
def extract_conv1_psd_batch(model, x_batch):
    B, T, C, L = x_batch.shape
    x = x_batch.view(B * T, C, L)
    enc = model.ae.encoder.epoch_encoder
    feat = F.gelu(enc.conv1(x))
    feat = feat - feat.mean(dim=-1, keepdim=True)
    psd = enc.norm1._encode_spectral_structure(feat)
    psd = psd.view(B, T, psd.shape[1], psd.shape[2]).mean(dim=1)
    return psd.reshape(B, -1)


# =========================================================
#                         MAIN
# =========================================================

def main():
    with open(CLUSTER_MAP_PATH, "r") as f:
        cluster_map = json.load(f)

    model = load_trained_model(CONFIG_PATH, CHECKPOINT_PATH, DEVICE)

    file_list = scan_files_recursively(DATA_ROOT, DATASETS)
    dataset = CustomDataset(file_list, cluster_map)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=dataset.collate,
        pin_memory=True
    )

    features = []

    for x in tqdm(loader, desc="Extracting"):
        if x is None:
            continue
        feats = extract_conv1_psd_batch(
            model,
            x.to(DEVICE, non_blocking=True)
        )
        features.append(feats.cpu().numpy())

    X = np.concatenate(features, axis=0).astype(np.float32)
    print(f"[Info] Feature Matrix Shape: {X.shape}")

    print(f"[Step] KMeans K={N_CLUSTERS}")
    kmeans = MiniBatchKMeans(
        n_clusters=N_CLUSTERS,
        batch_size=4096,
        n_init=10,
        random_state=42
    ).fit(X)

    centers = kmeans.cluster_centers_.astype(np.float32)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    centers_path = os.path.join(OUTPUT_DIR, f"{PREFIX}_centers.npy")
    np.save(centers_path, centers)

    feat_dim = centers.shape[1]
    expected_channels = 64
    f_r = feat_dim // expected_channels

    meta = {
        "source_checkpoint": CHECKPOINT_PATH,
        "n_clusters": N_CLUSTERS,
        "feature_dim": feat_dim,
        "structure": [expected_channels, f_r],
        "datasets": DATASETS
    }

    map_path = os.path.join(OUTPUT_DIR, f"{PREFIX}_map.json")
    with open(map_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[Done] Saved:")
    print(f"  {centers_path}")
    print(f"  {map_path}")


if __name__ == "__main__":
    main()
