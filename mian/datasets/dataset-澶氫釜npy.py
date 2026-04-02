import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import random
from types import SimpleNamespace
from pathlib import Path

# 尝试导入工具函数，失败则使用本地定义
try:
    from utils.util import to_tensor
except ImportError:
    def to_tensor(array):
        return torch.from_numpy(np.array(array)).float()


class CustomDataset(Dataset):
    def __init__(self, seqs_labels_path_pair):
        super(CustomDataset, self).__init__()
        # 原始的文件路径对列表 [(seq_path, label_path, subject_id), ...]
        self.file_list = seqs_labels_path_pair

        # --- 核心修复：建立全局样本索引映射 ---
        # map_index[i] = (file_idx, inner_idx)
        # inner_idx = -1 表示该文件本身就是一个完整的单序列样本 (20, 2, 3000)
        # inner_idx >= 0 表示该文件包含多个样本，取第 inner_idx 个
        self.samples_map = []

        print(f"[Info] Scanning {len(self.file_list)} files to build index map (Lazy Loading)...")

        # 预扫描所有文件以确定样本总数和位置
        for file_idx, (seq_path, _, _) in enumerate(self.file_list):
            try:
                # 使用 mmap_mode='r' 快速读取 shape
                data_ref = np.load(seq_path, mmap_mode='r')

                # --- 修复逻辑开始 ---
                if data_ref.ndim == 3:
                    # 情况A: 文件形状为 (20, 2, 3000)，是一个单独的序列
                    # 我们标记 inner_idx 为 -1，表示读取整个文件
                    self.samples_map.append((file_idx, -1))
                else:
                    # 情况B: 文件形状为 (N, 20, 2, 3000)，包含 N 个序列
                    num_samples = data_ref.shape[0]
                    for inner_idx in range(num_samples):
                        self.samples_map.append((file_idx, inner_idx))
                # --- 修复逻辑结束 ---

            except Exception as e:
                print(f"[Error] Failed to read shape of {seq_path}: {e}")

        print(f"[Info] Index built. Total valid samples: {len(self.samples_map)}")

    def __len__(self):
        return len(self.samples_map)

    def __getitem__(self, idx):
        # 1. 查表
        file_idx, inner_idx = self.samples_map[idx]
        seq_path, label_path, subject_id = self.file_list[file_idx]

        try:
            # 2. mmap 打开
            data_mmap = np.load(seq_path, mmap_mode='r')
            label_mmap = np.load(label_path, mmap_mode='r')

            # 3. 读取数据 (区分单样本文件和多样本文件)
            if inner_idx == -1:
                # 单序列文件 (20, 2, 3000) -> 直接读取全部
                seq = data_mmap.copy()  # (20, 2, 3000)
                label = label_mmap.copy()  # (20,)
            else:
                # 多序列文件 (N, 20, 2, 3000) -> 读取切片
                seq = data_mmap[inner_idx].copy()
                label = label_mmap[inner_idx].copy()

            # 4. 过滤标签 5 (如果标签包含 5，直接跳过)
            if np.any(label == 5):
                return None, None, None

            # 5. 类型转换
            seq = seq.astype(np.float32)
            label = label.astype(np.int64)

            return seq, label, subject_id

        except Exception as e:
            print(f"[Error] loading sample {idx} from {seq_path}: {e}")
            return None, None, None

    def collate(self, batch):
        # 1. 过滤掉加载失败 或 包含标签5 (None) 的样本
        batch = [item for item in batch if item[0] is not None]

        if not batch:
            return None, None, None

        # 2. 使用 torch.stack 高效堆叠
        # 现在 x[0] 保证是 (20, 2, 3000)，Stack 后变成 (Batch, 20, 2, 3000)
        x_tensor = torch.stack([torch.from_numpy(x[0]) for x in batch]).float()
        y_tensor = torch.stack([torch.from_numpy(x[1]) for x in batch]).long()
        z_tensor = torch.tensor([x[2] for x in batch]).long()

        return x_tensor, y_tensor, z_tensor


class LoadDataset(object):
    def __init__(self, params: SimpleNamespace):
        self.params = params
        self.datasets_map = {
            'cluster_0': 0, 'cluster_1': 1, 'cluster_2': 2, 'cluster_3': 3, 'cluster_4': 4,
            'cluster_5': 5, 'cluster_6': 6, 'cluster_7': 7
        }
        self.unknown_dataset_base_id = max(self.datasets_map.values()) + 1
        self.max_samples_per_dataset = getattr(params, 'max_samples_per_dataset', None)

        self.prefetch_factor = getattr(params, 'prefetch_factor', 2)
        self.persistent_workers = getattr(params, 'persistent_workers', True)

        if hasattr(params, 'test_dataset') and params.test_dataset:
            self.mode = 'test'
            self.test_dataset_name = params.test_dataset
            self.test_dir = f'{self.params.datasets_dir}/{self.test_dataset_name}'
            print(f"[Info] Mode: Test on {self.test_dataset_name}")
            if not os.path.isdir(self.test_dir):
                raise FileNotFoundError(f"Test dataset dir not found: {self.test_dir}")
            self.source_dirs = []
            self.targets_dirs = []
        else:
            self.mode = 'lodo'
            target_domains_list = params.target_domains.split(',') if isinstance(params.target_domains, str) else [
                params.target_domains]
            self.targets_dirs = [f'{self.params.datasets_dir}/{item}' for item in self.datasets_map.keys() if
                                 item in target_domains_list]
            self.source_dirs = [f'{self.params.datasets_dir}/{item}' for item in self.datasets_map.keys() if
                                item not in target_domains_list]
            print("[Info] Mode: LODO")
            print(f"  Target: {self.targets_dirs}")
            print(f"  Source: {self.source_dirs}")

    def get_data_loader(self):
        loader_args = {
            'batch_size': self.params.batch_size,
            'num_workers': self.params.num_workers,
            'pin_memory': True,
            'prefetch_factor': self.prefetch_factor if self.params.num_workers > 0 else None,
            'persistent_workers': self.persistent_workers if self.params.num_workers > 0 else False,
        }

        if self.mode == 'test':
            print(f"[Info] Loading Test Data from: {self.test_dir}")
            test_pairs, _ = self.load_path([self.test_dir], start_subject_id=0)
            if not test_pairs:
                test_set = CustomDataset([])
            else:
                test_set = CustomDataset(test_pairs)

            return {
                'train': None, 'val': None,
                'test': DataLoader(test_set, collate_fn=test_set.collate, shuffle=False, **loader_args)
            }, -1

        elif self.mode == 'lodo':
            print("[Info] Loading Source Data...")
            source_domains_pairs, next_subject_id = self.load_path(self.source_dirs, start_subject_id=0)

            print("[Info] Loading Target Data...")
            target_domains_pairs, final_subject_id = self.load_path(self.targets_dirs, start_subject_id=next_subject_id)

            if not source_domains_pairs:
                train_pairs, val_pairs = [], []
            else:
                train_pairs, val_pairs = self.split_dataset(source_domains_pairs)

            print(f"  Split: {len(train_pairs)} Train, {len(val_pairs)} Val")

            train_set = CustomDataset(train_pairs)
            val_set = CustomDataset(val_pairs)
            test_set = CustomDataset(target_domains_pairs)

            return {
                'train': DataLoader(train_set, collate_fn=train_set.collate, shuffle=True, **loader_args),
                'val': DataLoader(val_set, collate_fn=val_set.collate, shuffle=False, **loader_args),
                'test': DataLoader(test_set, collate_fn=test_set.collate, shuffle=False, **loader_args),
            }, final_subject_id

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def load_path(self, domains_dirs, start_subject_id):
        """
        递归搜索所有子目录，寻找成对的 'seq' 和 'labels' 文件夹。
        兼容:
        1. cluster_X/seq/file.npy
        2. cluster_X/subject/seq/file.npy
        """
        all_pairs = []
        current_subject_id = start_subject_id

        for dataset_dir in domains_dirs:
            dataset_name = Path(dataset_dir).name
            dataset_pairs = []

            # ID 分配
            if dataset_name in self.datasets_map:
                subject_id = self.datasets_map[dataset_name]
                current_subject_id = max(current_subject_id, max(self.datasets_map.values()) + 1)
            else:
                subject_id = current_subject_id
                current_subject_id += 1

            # 使用 os.walk 遍历整个目录树，寻找含有 seq 和 labels 的目录
            # print(f"    Scanning {dataset_dir} ...")

            for root, dirs, _ in os.walk(dataset_dir):
                if 'seq' in dirs and 'labels' in dirs:
                    # 找到了包含数据的目录对
                    seq_base = os.path.join(root, 'seq')
                    label_base = os.path.join(root, 'labels')

                    # 遍历 seq 文件夹
                    for sub_root, _, files in os.walk(seq_base):
                        for f in sorted(files):
                            if f.endswith('.npy'):
                                seq_path = os.path.join(sub_root, f)
                                # 构造对应的 label 路径 (保持相对路径一致)
                                rel_path = os.path.relpath(seq_path, seq_base)
                                label_path = os.path.join(label_base, rel_path)

                                if os.path.exists(label_path):
                                    dataset_pairs.append((seq_path, label_path, subject_id))

            # 采样限制
            limit = None
            if hasattr(self, 'max_samples_per_dataset') and self.max_samples_per_dataset is not None:
                if isinstance(self.max_samples_per_dataset, dict):
                    limit = self.max_samples_per_dataset.get(dataset_name)
                elif isinstance(self.max_samples_per_dataset, int):
                    limit = self.max_samples_per_dataset

            if limit is not None and len(dataset_pairs) > limit:
                # random.shuffle(dataset_pairs)
                dataset_pairs = dataset_pairs[:limit]
                print(f"    [{dataset_name}] Limited to {limit} files.")

            if len(dataset_pairs) == 0:
                # 兼容旧逻辑：如果根目录下没有 seq/labels，看看是否直接在 root 下找
                pass

            all_pairs.extend(dataset_pairs)

        return all_pairs, current_subject_id

    def split_dataset(self, source_domain_pairs, val_ratio=0.2, seed=None):
        if seed is not None:
            random.seed(seed)

        shuffled = source_domain_pairs[:]
        random.shuffle(shuffled)

        split = int(len(shuffled) * (1 - val_ratio))
        return shuffled[:split], shuffled[split:]