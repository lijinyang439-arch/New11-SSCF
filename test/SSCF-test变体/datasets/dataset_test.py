import torch
from torch.utils.data import Dataset, DataLoader
import torch.utils.data.distributed as dist_utils
import torch.distributed as dist
import numpy as np
import os
import random
from types import SimpleNamespace
from pathlib import Path
from collections import OrderedDict

# 尝试导入工具函数
try:
    from utils.util import to_tensor
except ImportError:
    def to_tensor(array):
        return torch.from_numpy(np.array(array)).float()


class LRUFileCache:
    """简单的 LRU 缓存"""

    def __init__(self, capacity=128):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, path):
        if path in self.cache:
            self.cache.move_to_end(path)
            return self.cache[path]
        return None

    def put(self, path, item):
        if path in self.cache:
            self.cache.move_to_end(path)
        self.cache[path] = item
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)


class CustomDataset(Dataset):
    def __init__(self, seqs_labels_path_pair):
        super(CustomDataset, self).__init__()
        # 注意：这里的 file_list 必须在所有 Rank 上顺序一致，依赖于 LoadDataset 中的 sort
        self.file_list = seqs_labels_path_pair
        self.samples_map = []
        self.file_cache = LRUFileCache(capacity=64)

        # 默认形状，用于 Dummy Data (防止死锁)
        self.default_seq_shape = (20, 2, 3000)

        # === DDP 同步机制 ===
        # 1. 仅 Rank 0 扫描文件构建索引
        # 2. 将索引广播给所有 Rank
        # 这确保了即使文件列表顺序有细微差异（虽然我们已经sort了），索引也是绝对统一的

        is_master = (not dist.is_initialized()) or (dist.get_rank() == 0)
        broadcast_data = [None, None]  # [samples_map, default_seq_shape]

        if is_master:
            print(f"[信息] (主进程) 正在扫描 {len(self.file_list)} 个文件以构建索引...")
            local_map = []
            local_shape = self.default_seq_shape
            valid_count = 0

            for file_idx, file_info in enumerate(self.file_list):
                try:
                    seq_path = file_info[0]
                    label_path = file_info[1]

                    data_ref = np.load(seq_path, mmap_mode='r')
                    label_ref = np.load(label_path, mmap_mode='r')

                    # 更新默认形状 (取第一个成功的)
                    if valid_count == 0 and data_ref.ndim >= 2:
                        if data_ref.ndim == 3:
                            local_shape = data_ref.shape
                        elif data_ref.ndim == 4:
                            local_shape = data_ref.shape[1:]

                    if data_ref.ndim == 3:
                        # 【修改核心点】：严格过滤，只有包含在 [0, 1, 2, 3, 4] 范围内的标签才有效
                        if np.all((label_ref >= 0) & (label_ref <= 4)):
                            local_map.append((file_idx, -1))
                            valid_count += 1
                    else:
                        num_samples = data_ref.shape[0]
                        for inner_idx in range(num_samples):
                            # 【修改核心点】：同理，严格过滤内部标签
                            if np.all((label_ref[inner_idx] >= 0) & (label_ref[inner_idx] <= 4)):
                                local_map.append((file_idx, inner_idx))
                                valid_count += 1
                except Exception as e:
                    print(f"[警告] 文件扫描失败: {seq_path}, {e}")

            print(f"[信息] 索引构建完成。有效样本数: {len(local_map)}")
            broadcast_data = [local_map, local_shape]

        # === 广播索引 ===
        if dist.is_initialized():
            # 这里可能会花费几秒钟
            dist.broadcast_object_list(broadcast_data, src=0)

        self.samples_map = broadcast_data[0]
        self.default_seq_shape = broadcast_data[1]

    def __len__(self):
        return len(self.samples_map)

    def _get_mmap(self, path):
        memmap = self.file_cache.get(path)
        if memmap is None:
            try:
                memmap = np.load(path, mmap_mode='r')
                self.file_cache.put(path, memmap)
            except Exception as e:
                return None
        return memmap

    def __getitem__(self, idx):
        file_idx, inner_idx = self.samples_map[idx]

        file_info = self.file_list[file_idx]

        if len(file_info) == 3:
            seq_path, label_path, subject_id = file_info
        elif len(file_info) == 4:
            seq_path, label_path, domain_id, real_subject_id = file_info
            subject_id = real_subject_id
        else:
            raise ValueError(f"file_list 中的样本项长度异常: {len(file_info)}")

        try:
            data_mmap = self._get_mmap(seq_path)
            label_mmap = self._get_mmap(label_path)

            if data_mmap is None or label_mmap is None:
                raise IOError(f"Failed to load mmap for {seq_path}")

            if inner_idx == -1:
                seq = data_mmap.copy()
                label = label_mmap.copy()
            else:
                seq = data_mmap[inner_idx].copy()
                label = label_mmap[inner_idx].copy()

            seq = seq.astype(np.float32)
            label = label.astype(np.int64)

            return seq, label, subject_id

        except Exception as e:
            # === DDP 兜底机制 ===
            # 如果加上 sort 后还有这个错误，说明文件系统有严重延迟或损坏
            # 返回假数据，保证训练不中断，不死锁
            if dist.is_initialized() and dist.get_rank() == 0:
                # 只有主进程打印日志，防止刷屏
                print(f"[DDP 异常] Rank {dist.get_rank()} 无法读取样本 {idx}。返回 Dummy Data。错误: {e}")

            dummy_seq = np.zeros(self.default_seq_shape, dtype=np.float32)
            dummy_label = np.zeros((self.default_seq_shape[0],), dtype=np.int64)
            return dummy_seq, dummy_label, subject_id

    def collate(self, batch):
        batch = [item for item in batch if item[0] is not None]
        if not batch: return None, None, None
        x_tensor = torch.stack([torch.from_numpy(x[0]) for x in batch]).float()
        y_tensor = torch.stack([torch.from_numpy(x[1]) for x in batch]).long()
        z_tensor = torch.tensor([x[2] for x in batch]).long()
        return x_tensor, y_tensor, z_tensor


class LoadDataset(object):
    def __init__(self, params: SimpleNamespace):
        self.params = params
        self.datasets_map = {
            'MROS1': 0, 'MROS2': 1, 'MESA': 2, 'SHHS1': 3, 'P2018': 4,
        }
        self.max_samples_per_dataset = getattr(params, 'max_samples_per_dataset', None)
        self.prefetch_factor = getattr(params, 'prefetch_factor', 2)
        self.persistent_workers = True
        self.datasets_dir = getattr(params, 'datasets_dir', './data')

        if hasattr(params, 'test_dataset') and params.test_dataset:
            self.mode = 'test'
            self.test_dataset_name = params.test_dataset
            self.test_dir = f'{self.datasets_dir}/{self.test_dataset_name}'
            if not os.path.isdir(self.test_dir):
                self.test_dir = str(Path(self.datasets_dir) / self.test_dataset_name)
            self.source_dirs = []
            self.targets_dirs = []
        else:
            self.mode = 'lodo'
            target_domains = getattr(params, 'target_domains', 'sleep-edfx')
            target_domains_list = target_domains.split(',') if isinstance(target_domains, str) else [target_domains]

            self.targets_dirs = [f'{self.datasets_dir}/{item}' for item in self.datasets_map.keys() if
                                 item in target_domains_list]
            self.source_dirs = [f'{self.datasets_dir}/{item}' for item in self.datasets_map.keys() if
                                item not in target_domains_list]

    def _make_loader(self, dataset, shuffle, loader_args):
        if dataset is None or len(dataset) == 0:
            return None
        if dist.is_initialized():
            sampler = dist_utils.DistributedSampler(dataset, shuffle=shuffle)
            return DataLoader(dataset, collate_fn=dataset.collate, shuffle=False,
                              sampler=sampler, **loader_args)
        else:
            return DataLoader(dataset, collate_fn=dataset.collate, shuffle=shuffle, **loader_args)

    def get_data_loader(self):
        default_workers = max(4, os.cpu_count() // 4)
        num_workers = getattr(self.params, 'num_workers', default_workers)

        if dist.is_initialized():
            num_workers = min(num_workers, 8)

        batch_size = getattr(self.params, 'batch_size', 32)
        loader_args = {
            'batch_size': batch_size,
            'num_workers': num_workers,
            'pin_memory': True,
            'prefetch_factor': self.prefetch_factor if num_workers > 0 else None,
            'persistent_workers': self.persistent_workers if num_workers > 0 else False,
        }

        if self.mode == 'test':
            if (not dist.is_initialized()) or (dist.get_rank() == 0):
                print(f"[信息] 加载测试数据: {self.test_dir}")
            test_pairs, _ = self.load_path([self.test_dir], 0)
            test_set = CustomDataset(test_pairs) if test_pairs else CustomDataset([])
            return {'train': None, 'val': None, 'test': self._make_loader(test_set, False, loader_args)}, -1

        elif self.mode == 'lodo':
            source_pairs, next_sid = self.load_path(self.source_dirs, 0)
            target_pairs, final_sid = self.load_path(self.targets_dirs, next_sid)

            if not source_pairs:
                train_pairs, val_pairs = [], []
            else:
                train_pairs, val_pairs = self.split_dataset(source_pairs)

            return {
                'train': self._make_loader(CustomDataset(train_pairs), True, loader_args),
                'val': self._make_loader(CustomDataset(val_pairs), False, loader_args),
                'test': self._make_loader(CustomDataset(target_pairs), False, loader_args),
            }, final_sid

        return {}, -1

    def load_path(self, domains_dirs, start_subject_id):
        all_pairs = []
        current_subject_id = start_subject_id

        # 仅用于 test mode：把真实受试者 key 映射成 int
        real_subject_id_map = {}
        next_real_subject_id = 0

        for dataset_dir in domains_dirs:
            dataset_name = Path(dataset_dir).name
            dataset_pairs = []
            if dataset_name in self.datasets_map:
                subject_id = self.datasets_map[dataset_name]
                current_subject_id = max(current_subject_id, max(self.datasets_map.values()) + 1)
            else:
                subject_id = current_subject_id
                current_subject_id += 1

            if not os.path.exists(dataset_dir): continue

            for root, dirs, _ in os.walk(dataset_dir):
                if 'seq' in dirs and 'labels' in dirs:
                    seq_base = os.path.join(root, 'seq')
                    label_base = os.path.join(root, 'labels')
                    for sub_root, _, files in os.walk(seq_base):
                        for f in sorted(files):
                            if f.endswith('.npy'):
                                seq_path = os.path.join(sub_root, f)
                                rel_path = os.path.relpath(seq_path, seq_base)
                                label_path = os.path.join(label_base, rel_path)
                                if os.path.exists(label_path):
                                    if self.mode == 'test':
                                        # test 模式下：第三项改成“真实受试者 id”
                                        # 这里按相对路径去掉后缀后的 stem 作为 subject key
                                        rel_stem = str(Path(rel_path).with_suffix(''))
                                        if rel_stem not in real_subject_id_map:
                                            real_subject_id_map[rel_stem] = next_real_subject_id
                                            next_real_subject_id += 1
                                        real_subject_id = real_subject_id_map[rel_stem]

                                        # 保存四元组，后面 __getitem__ 在 test 时返回 real_subject_id
                                        dataset_pairs.append((seq_path, label_path, subject_id, real_subject_id))
                                    else:
                                        # 训练 / lodo 模式保持原样，继续返回域 id
                                        dataset_pairs.append((seq_path, label_path, subject_id))

            # 【核心修复】：必须对文件列表进行排序！
            # 否则不同 Rank 的 os.walk 顺序可能不同，导致索引对应错误的文件
            dataset_pairs.sort(key=lambda x: x[0])

            limit = None
            if self.max_samples_per_dataset is not None:
                if isinstance(self.max_samples_per_dataset, dict):
                    limit = self.max_samples_per_dataset.get(dataset_name)
                elif isinstance(self.max_samples_per_dataset, int):
                    limit = self.max_samples_per_dataset
            if limit is not None and len(dataset_pairs) > limit: dataset_pairs = dataset_pairs[:limit]
            all_pairs.extend(dataset_pairs)
        return all_pairs, current_subject_id

    def split_dataset(self, source_domain_pairs, val_ratio=0.2, seed=None):
        if seed is not None: random.seed(seed)
        shuffled = source_domain_pairs[:]
        random.shuffle(shuffled)
        split = int(len(shuffled) * (1 - val_ratio))
        return shuffled[:split], shuffled[split:]