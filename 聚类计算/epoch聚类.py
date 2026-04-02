#!/usr/bin/env python3
# -*- coding: utf-8 -*-



# =========================================================
# 【关键】防止 BLAS / MKL / OpenBLAS 抢核
# 必须在 numpy / sklearn 之前设置
# =========================================================
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import json
import numpy as np
import torch
from pathlib import Path
from types import SimpleNamespace
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from tqdm import tqdm
from collections import defaultdict, Counter
from multiprocessing import Pool, cpu_count

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.io import savemat
from scipy.signal import welch

# =========================================================
# utils_psd 兼容导入 + remove_outliers 统一为返回4项
# =========================================================
_remove_outliers_impl = None

try:
    from utils_psd import remove_outliers as _remove_outliers_impl
except ImportError:
    _remove_outliers_impl = None


def remove_outliers(vectors, items, ds_labels, save_dir):
    if _remove_outliers_impl is None:
        keep_mask = np.ones(len(vectors), dtype=bool)
        return vectors, items, ds_labels, keep_mask

    out = _remove_outliers_impl(vectors, items, ds_labels, save_dir)
    if isinstance(out, tuple) and len(out) == 4:
        return out
    if isinstance(out, tuple) and len(out) == 3:
        vec_clean, items_clean, ds_clean = out
        keep_mask = np.ones(len(vectors), dtype=bool)
        return vec_clean, items_clean, ds_clean, keep_mask
    raise ValueError(f"remove_outliers 返回值数量异常: {type(out)}")

# =========================================================
# 1. 配置部分
# =========================================================

DATASETS_DIR = "/fdata/lijinyang/datasets_dir2_all_Merged"

SELECTED_DATASETS = [
    "SHHS1","P2018","MROS1","MROS2","MESA"
]

MAX_SUBJECTS_PER_DATASET = 53333333333333333333333333333333333333333333

# 改这里：从 2 聚到 30
K_LIST = list(range(2, 31))

PSD_CACHE_DIR = "./epoch聚类"
RESULTS_ROOT = "./epoch聚类"

os.makedirs(PSD_CACHE_DIR, exist_ok=True)
os.makedirs(RESULTS_ROOT, exist_ok=True)

# =========================================================
# [新增] 输入 epoch 长度控制
# =========================================================
FS = 100
FORCE_FIXED_EPOCH_LEN = True   # True: 强制过滤不一致长度；False: 只统计打印但不丢弃
EXPECTED_EPOCH_LEN = None      # 稍后自动推断（点数）

# =========================================================
# [新增] Welch + 分段参数（严格按你描述）
# =========================================================
EPOCH_SEC = 30
CHUNK_SEC = 5
NUM_CHUNKS = EPOCH_SEC // CHUNK_SEC  # 6

# 5 秒 = 500 点。为了 welch 真正“平均”，nperseg < 500
WELCH_NPERSEG = 256
WELCH_NOVERLAP = 128
WELCH_NFFT = None  # None -> 等于 nperseg

# 频段：0.5Hz~35Hz
BAND_LOW = 0.5
BAND_HIGH = 35.0

LOG_EPS = 1e-12


# =========================================================
# 2. 数据路径扫描（按你当前工程风格）
# =========================================================

class LoadDataset(object):
    def __init__(self, params: SimpleNamespace):
        self.params = params
        self.datasets_map = {
            'SHHS1': 0,'P2018': 0,'MROS1': 0,'MROS2': 0,'MESA': 0,
        }
        target_domains_list = params.target_domains.split(',')
        self.targets_dirs = [f'{self.params.datasets_dir}/{item}' for item in self.datasets_map.keys()
                             if item in target_domains_list]

        for item in target_domains_list:
            path = f'{self.params.datasets_dir}/{item}'
            if item not in self.datasets_map and os.path.isdir(path):
                self.targets_dirs.append(path)

    def load_path(self, domains_dirs, start_subject_id):
        """
        兼容“目录层级不固定”的数据结构：
        - 在 dataset_dir 内递归寻找同时包含 'seq' 和 'labels' 的父目录
        - seq 内文件相对路径匹配 labels
        """
        all_pairs = []
        current_subject_id = start_subject_id

        for dataset_dir in domains_dirs:
            dataset_dir = str(dataset_dir)
            dataset_name = Path(dataset_dir).name

            if dataset_name in self.datasets_map:
                subject_id = self.datasets_map[dataset_name]
                current_subject_id = max(current_subject_id, max(self.datasets_map.values()) + 1)
            else:
                subject_id = current_subject_id
                current_subject_id += 1

            if not os.path.exists(dataset_dir):
                continue

            dataset_pairs = []
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
                                    dataset_pairs.append((seq_path, label_path, subject_id))

            dataset_pairs.sort(key=lambda x: x[0])
            all_pairs.extend(dataset_pairs)

        return all_pairs, current_subject_id

# =========================================================
# 3. 归类 / 预检 / PSD 计算
# =========================================================
from scipy.io import savemat

def export_for_matlab(save_dir, centers_psd, freqs, K):
    row_names = [f"Class-{i + 1}" for i in range(K)]
    col_names = [f"{f:.2f}Hz" for f in freqs]
    mat_dict = {
        "Data": centers_psd,
        "rowName": row_names,
        "colName": col_names,
        "K_value": K
    }
    out_path = os.path.join(save_dir, "viz_matlab.mat")
    savemat(out_path, mat_dict)
    print(f"[MATLAB 导出完成] 数据已保存至 -> {out_path}")


def infer_metadata_from_path(seq_path, dataset_list):
    path_obj = Path(seq_path)
    parts = path_obj.parts
    dataset_name = "Unknown"
    subject_name = path_obj.stem

    for ds in dataset_list:
        if ds in parts:
            dataset_name = ds
            break

    if 'seq' in parts:
        seq_idx = parts.index('seq')
        if seq_idx + 1 < len(parts) - 1:
            subject_name = parts[seq_idx + 1]

    return dataset_name, subject_name


def organize_files_by_subject(all_pairs, dataset_list):
    print("[信息] 正在将文件按受试者归类...")
    structure = defaultdict(lambda: defaultdict(list))
    count = 0

    for seq_path, label_path, _ in tqdm(all_pairs, desc="归类进度"):
        ds_name, subj_name = infer_metadata_from_path(seq_path, dataset_list)
        if ds_name in dataset_list:
            structure[ds_name][subj_name].append((seq_path, label_path))
            count += 1

    print(f"[信息] 归类完成。共保留了 {count} 个相关文件。")
    for ds in dataset_list:
        n_sub = len(structure[ds])
        n_files = sum(len(files) for files in structure[ds].values())
        print(f"  -> {ds}: {n_sub} 个受试者 ({n_files} 个文件)")

    return structure


def _as_numpy(x):
    if x is None:
        return None
    try:
        return np.asarray(x)
    except Exception:
        return None


def _iter_valid_epochs_ct(data, label, skip_label_value=5):
    """
    yield epoch -> (C,T)
    支持：
      - 2D: (C,T) 或 (T,C)
      - 3D: (E,C,T)
      - 4D: (N,E,C,T)
    """
    data = _as_numpy(data)
    label = _as_numpy(label)

    if data is None:
        return

    if label is not None:
        label_s = np.squeeze(label)
    else:
        label_s = None

    # 2D：单 epoch
    if data.ndim == 2:
        if data.shape[0] <= 128 and data.shape[1] > 128:
            ep = data
        elif data.shape[1] <= 128 and data.shape[0] > 128:
            ep = data.T
        else:
            ep = data if data.shape[0] <= data.shape[1] else data.T
        yield np.asarray(ep)
        return

    # 3D：(E,C,T)
    if data.ndim == 3:
        E = data.shape[0]
        for e in range(E):
            if label_s is not None:
                try:
                    lab_e = label_s[e]
                    if np.any(lab_e == skip_label_value):
                        continue
                except Exception:
                    pass
            ep = data[e]
            yield np.asarray(ep)
        return

    # 4D：(N,E,C,T)
    if data.ndim == 4:
        N, E = data.shape[0], data.shape[1]
        for n in range(N):
            for e in range(E):
                if label_s is not None:
                    try:
                        lab_ne = label_s[n, e]
                        if np.any(lab_ne == skip_label_value):
                            continue
                    except Exception:
                        pass
                ep = data[n, e]
                yield np.asarray(ep)
        return

    return


def infer_expected_epoch_len(structure, dataset_list, fs=100):
    """
    推断 epoch 点数 T
    """
    print("\n" + "=" * 50)
    print(" >>> 正在推断输入 epoch 长度 (Expected Epoch Length) <<<")
    print("=" * 50)

    for ds in dataset_list:
        if not structure[ds]:
            continue
        for subj in structure[ds]:
            files = structure[ds][subj]
            for seq_path, label_path in files:
                try:
                    data = np.load(seq_path, mmap_mode='r')
                    label = np.load(label_path, mmap_mode='r')
                    for ep in _iter_valid_epochs_ct(data, label, skip_label_value=5):
                        if ep is None:
                            continue
                        ep = np.asarray(ep)
                        if ep.ndim != 2:
                            continue
                        T = int(ep.shape[-1])
                        if T < 16:
                            continue
                        sec = float(T) / float(fs)
                        print(f"[信息] 推断得到 EXPECTED_EPOCH_LEN={T} 点 (约 {sec:.3f} 秒), 来自 {ds}/{subj}")
                        print("=" * 50 + "\n")
                        return T
                except Exception:
                    pass

    print("[警告] 未能从数据中推断 epoch 长度（可能数据读取失败或全部被过滤）。")
    print("=" * 50 + "\n")
    return None


def run_sanity_check(structure, dataset_list, fs=100):
    print("\n" + "=" * 50)
    print(" >>> 正在执行数据读取预检 (Sanity Check) <<<")
    print("=" * 50)

    all_passed = True

    for ds in dataset_list:
        if not structure[ds]:
            print(f"[警告] 数据集 {ds} 为空！")
            continue

        subj = next(iter(structure[ds]))
        seq_path, label_path = structure[ds][subj][0]

        print(f"\n[检查] 数据集: {ds} | 样本受试者: {subj}")
        try:
            data = np.load(seq_path, mmap_mode='r')
            label = np.load(label_path, mmap_mode='r')

            ep0 = None
            for ep in _iter_valid_epochs_ct(data, label, skip_label_value=5):
                ep0 = ep
                break

            if ep0 is None:
                print(f"       [失败] 没有找到可用 epoch（可能全部 label=5 或数据/标签维度不匹配）")
                all_passed = False
            else:
                ep0 = np.asarray(ep0)
                if ep0.ndim != 2:
                    print(f"       [失败] epoch 维度异常: {ep0.ndim}, shape={ep0.shape}")
                    all_passed = False
                else:
                    C, T = int(ep0.shape[0]), int(ep0.shape[1])
                    sec = float(T) / float(fs)
                    print(f"       data.shape={tuple(data.shape)} | label.shape={tuple(label.shape)}")
                    print(f"       epoch形状: (C,T)=({C},{T}) | 约 {sec:.3f} 秒 | fs={fs}Hz")
                    print(f"       统计: Min={ep0.min():.4f}, Max={ep0.max():.4f}, Mean={ep0.mean():.4f}")
                    if float(ep0.min()) == 0.0 and float(ep0.max()) == 0.0:
                        all_passed = False

        except Exception as e:
            print(f"[错误] {e}")
            all_passed = False

    if all_passed:
        print("[成功] 所有选定数据集的抽样检查均通过。")
    else:
        print("[提示] 预检发现潜在问题，请留意。")
    print("=" * 50 + "\n")


def _welch_psd_1d(x, fs):
    f, Pxx = welch(
        x,
        fs=fs,
        window="hann",
        nperseg=WELCH_NPERSEG,
        noverlap=WELCH_NOVERLAP,
        nfft=WELCH_NFFT,
        detrend="constant",
        return_onesided=True,
        scaling="density",
        axis=-1
    )
    return f, Pxx


def compute_psd30_from_epoch(ep_ct, fs):
    """
    30秒 epoch -> 切 6 个 5秒 -> 每个5秒 welch -> 平均得到 PSD30
    返回：
      freqs: (F,)
      psd30: (C, F)  (线性域 PSD，不log，不归一化)
    """
    ep_ct = np.asarray(ep_ct, dtype=np.float64)
    if ep_ct.ndim != 2:
        return None, None

    C, T = ep_ct.shape
    chunk_len = int(round(fs * CHUNK_SEC))      # 500
    need_len = int(round(fs * EPOCH_SEC))       # 3000

    # 截断/补零（是否强制过滤由外层控制，这里只保证切片不越界）
    if T > need_len:
        ep_ct = ep_ct[:, :need_len]
        T = need_len
    elif T < need_len:
        pad = need_len - T
        ep_ct = np.pad(ep_ct, ((0, 0), (0, pad)), mode="constant")
        T = need_len

    freqs_ref = None
    psd_chunks = []

    for i in range(NUM_CHUNKS):
        s = i * chunk_len
        e = s + chunk_len
        chunk = ep_ct[:, s:e]  # (C, 500)

        psd_c = []
        for ch in range(C):
            f, Pxx = _welch_psd_1d(chunk[ch], fs)
            if freqs_ref is None:
                freqs_ref = f
            psd_c.append(np.asarray(Pxx).reshape(-1))
        psd_c = np.stack(psd_c, axis=0)  # (C, F)
        psd_chunks.append(psd_c)

    psd_stack = np.stack(psd_chunks, axis=0)   # (6, C, F)
    psd30 = np.mean(psd_stack, axis=0)         # (C, F)

    return freqs_ref, psd30


def _process_one_subject(args):
    ds, subj, files, expected_len, force_fixed, fs = args
    all_psd30 = []
    freqs_ref = None

    # [新增] 长度统计
    cnt_total = 0
    cnt_used = 0
    cnt_skipped_len = 0
    len_counter = Counter()

    for seq_path, label_path in files:
        try:
            data = np.load(seq_path, mmap_mode='r')
            label = np.load(label_path, mmap_mode='r')

            for ep in _iter_valid_epochs_ct(data, label, skip_label_value=5):
                if ep is None:
                    continue
                ep = np.asarray(ep)
                if ep.ndim != 2 or ep.shape[-1] < 16:
                    continue

                cnt_total += 1
                T = int(ep.shape[-1])
                len_counter[T] += 1

                if expected_len is not None and force_fixed:
                    if T != int(expected_len):
                        cnt_skipped_len += 1
                        continue

                f, psd30 = compute_psd30_from_epoch(ep, fs)
                if f is None or psd30 is None:
                    continue

                if freqs_ref is None:
                    freqs_ref = f

                all_psd30.append(psd30)
                cnt_used += 1

        except Exception:
            pass

    if not all_psd30:
        return (None, None, None, ds, subj, None, cnt_total, cnt_used, cnt_skipped_len, dict(len_counter))

    # PSD-subj（线性域）：所有 epoch 的 PSD30 均值
    psd_subj_linear = np.mean(np.stack(all_psd30, axis=0), axis=0)  # (C, F)

    # PSD-subj（log）：用于聚类特征
    psd_subj_log = np.log(psd_subj_linear + LOG_EPS)

    return (f"{ds}/{subj}", psd_subj_linear, psd_subj_log, ds, subj, freqs_ref,
            cnt_total, cnt_used, cnt_skipped_len, dict(len_counter))


def compute_psd_main(structure, dataset_list, expected_len=None, force_fixed=True, fs=100):
    subject_items = []
    psd_subj_linear_list = []   # (N,C,F) 线性域
    psd_subj_log_list = []      # (N,C,F) log域
    dataset_labels = []
    subject_labels = []
    freqs_ref = None

    # [新增] 全局统计
    stat_total = 0
    stat_used = 0
    stat_skipped_len = 0
    stat_len_counter_all = Counter()
    stat_len_counter_by_ds = {ds: Counter() for ds in dataset_list}

    tasks = []
    for ds in dataset_list:
        subjects = list(structure[ds].keys())
        if len(subjects) > MAX_SUBJECTS_PER_DATASET:
            subjects = subjects[:MAX_SUBJECTS_PER_DATASET]
        for subj in subjects:
            tasks.append((ds, subj, structure[ds][subj], expected_len, force_fixed, fs))

    n_workers = min(64, cpu_count())
    print(f"\n[信息] 使用 {n_workers} 个 CPU 核心并行计算 PSD-subj (log)")

    if expected_len is not None:
        sec = float(expected_len) / float(fs)
        print(f"[信息] 输入 epoch 期望长度: {expected_len} 点 (约 {sec:.3f} 秒) | 强制过滤={force_fixed}")
    else:
        print(f"[警告] expected_len=None，将无法保证 PSD 频率轴一致。建议先推断后再跑。")

    with Pool(processes=n_workers) as pool:
        for out in tqdm(pool.imap_unordered(_process_one_subject, tasks),
                        total=len(tasks), desc="PSD并行计算"):
            item, psd_subj_linear, psd_subj_log, ds, subj, f, cnt_total, cnt_used, cnt_skipped_len, len_counter = out

            # 汇总统计
            stat_total += int(cnt_total)
            stat_used += int(cnt_used)
            stat_skipped_len += int(cnt_skipped_len)

            if isinstance(len_counter, dict):
                stat_len_counter_by_ds[ds].update(len_counter)
                stat_len_counter_all.update(len_counter)

            if item is None:
                continue

            subject_items.append(item)
            psd_subj_linear_list.append(psd_subj_linear)
            psd_subj_log_list.append(psd_subj_log)
            dataset_labels.append(ds)
            subject_labels.append(subj)
            if freqs_ref is None and f is not None:
                freqs_ref = f

    # [新增] 打印长度统计
    print("\n" + "=" * 60)
    print(">>> 输入 epoch 长度统计 (在过滤前的原始统计) <<<")
    print(f"[总体] epoch总数={stat_total} | 实际用于PSD={stat_used} | 因长度不一致跳过={stat_skipped_len}")
    if stat_total > 0:
        print(f"[总体] 用于PSD比例={(stat_used/stat_total)*100:.2f}% | 跳过比例={(stat_skipped_len/stat_total)*100:.2f}%")

    if len(stat_len_counter_all) > 0:
        top_all = stat_len_counter_all.most_common(10)
        top_str = " | ".join([f"{L}:{c}" for L, c in top_all])
        print(f"[总体] 最常见长度Top10: {top_str}")
        L_min = min(stat_len_counter_all.keys())
        L_max = max(stat_len_counter_all.keys())
        print(f"[总体] 长度范围: min={L_min}, max={L_max} (点数) | 对应秒数: {L_min/fs:.3f}~{L_max/fs:.3f} 秒")

    for ds in dataset_list:
        cnt = stat_len_counter_by_ds[ds]
        if len(cnt) == 0:
            continue
        top_ds = cnt.most_common(5)
        top_str = " | ".join([f"{L}:{c}" for L, c in top_ds])
        L_min = min(cnt.keys())
        L_max = max(cnt.keys())
        print(f"[{ds}] Top5: {top_str} | 范围: {L_min}~{L_max} 点 ({L_min/fs:.3f}~{L_max/fs:.3f} 秒)")
    print("=" * 60 + "\n")

    if len(psd_subj_log_list) == 0:
        return (np.array([]), np.array([]), np.array([]), np.array([]), None, None, None)

    psd_linear = np.stack(psd_subj_linear_list, axis=0)  # (N,C,F)
    psd_log = np.stack(psd_subj_log_list, axis=0)        # (N,C,F)

    if freqs_ref is None:
        freqs_ref = np.arange(psd_log.shape[-1])

    freqs_ref = np.asarray(freqs_ref).reshape(-1)
    band_mask = (freqs_ref >= BAND_LOW) & (freqs_ref <= BAND_HIGH)
    idx = np.where(band_mask)[0]
    if idx.size == 0:
        raise ValueError(f"频段选择失败：freqs 不覆盖 [{BAND_LOW},{BAND_HIGH}] Hz")

    freqs_band = freqs_ref[idx]               # (Fband,)
    psd_linear_band = psd_linear[:, :, idx]   # (N,C,Fband)
    psd_log_band = psd_log[:, :, idx]         # (N,C,Fband)

    # KMeans 输入：log(PSD-subj) flatten
    vectors_flat = psd_log_band.reshape(psd_log_band.shape[0], -1)

    return (
        np.array(subject_items),
        vectors_flat,            # (N, C*Fband) ——KMeans 用这个（log域）
        np.array(dataset_labels),
        np.array(subject_labels),
        freqs_band,
        psd_linear_band,         # (N,C,Fband) ——算“几何中心”用（线性域）
        psd_log_band             # (N,C,Fband) ——备用
    )


def _sqrt_mean_square_center(psd_linear_cxf, eps=1e-12):
    """
    你给的几何中心定义：
      P_center = ( mean( sqrt(P_i) ) )^2
    输入：psd_linear_cxf: (N,C,F) 线性域 PSD
    输出：center_linear: (C,F) 线性域 PSD
    """
    x = np.asarray(psd_linear_cxf, dtype=np.float64)
    x = np.clip(x, a_min=0.0, a_max=None)
    amp = np.sqrt(x + eps)             # (N,C,F)
    amp_mean = np.mean(amp, axis=0)    # (C,F)
    center = amp_mean ** 2             # (C,F)
    return center


def _plot_cluster_centers_meanch(save_dir, centers_cxf, freqs, title_prefix):
    """
    centers_cxf: (K,C,F)
    默认画“跨通道平均曲线”
    """
    K = centers_cxf.shape[0]
    mean_centers = centers_cxf.mean(axis=1)  # (K,F)

    plt.figure()
    for k in range(K):
        plt.plot(freqs, mean_centers[k], label=f"Cluster-{k}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("log(PSD)")
    plt.title(f"{title_prefix} (channel-mean)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "cluster_centers_psd_meanch.png"), dpi=220)
    plt.close()

    for k in range(K):
        plt.figure()
        plt.plot(freqs, mean_centers[k])
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("log(PSD)")
        plt.title(f"{title_prefix} - Cluster {k} (channel-mean)")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"cluster_{k:02d}_psd_meanch.png"), dpi=220)
        plt.close()


def _plot_dataset_composition(save_dir, cluster_id, ds_labels, dataset_list):
    cluster_id = np.asarray(cluster_id).astype(int)
    ds_labels = np.asarray(ds_labels).astype(str)

    K = int(cluster_id.max()) + 1
    ds_to_idx = {ds: i for i, ds in enumerate(dataset_list)}
    mat = np.zeros((K, len(dataset_list)), dtype=np.int64)

    for cid, ds in zip(cluster_id, ds_labels):
        if ds in ds_to_idx:
            mat[cid, ds_to_idx[ds]] += 1

    mat_sum = mat.sum(axis=1, keepdims=True)
    mat_ratio = mat / np.maximum(mat_sum, 1)

    plt.figure()
    bottom = np.zeros((K,), dtype=np.float64)
    x = np.arange(K)
    for j, ds in enumerate(dataset_list):
        plt.bar(x, mat_ratio[:, j], bottom=bottom, label=ds)
        bottom += mat_ratio[:, j]

    plt.xticks(x, [f"C{k}" for k in range(K)])
    plt.ylim(0, 1.0)
    plt.ylabel("Proportion")
    plt.title("Dataset composition per cluster")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "cluster_dataset_composition.png"), dpi=220)
    plt.close()

    np.save(os.path.join(save_dir, "cluster_dataset_counts.npy"), mat)
    np.save(os.path.join(save_dir, "cluster_dataset_ratio.npy"), mat_ratio)


# =========================================================
# 4. 聚类流程（按你要求：KMeans 用 log(PSD-subj)，中心按 sqrt-mean-square 定义）
# =========================================================
def run_clustering_for_k(K, items, vectors_flat_log, dataset_labels, subject_labels,
                         freqs_band, psd_linear_band, psd_log_band):
    print(f"\n[聚类] === 开始执行 K={K} ===")

    save_dir = os.path.join(RESULTS_ROOT, f"K_{K:02d}")
    os.makedirs(save_dir, exist_ok=True)

    if len(vectors_flat_log) < K:
        print(f"[跳过] 样本数 ({len(vectors_flat_log)}) 少于 K={K}")
        return

    # ========= 1. 去离群点（在 log 向量空间做 mask，同步到线性/其它数组） =========
    vec_clean, items_clean, ds_clean, keep_mask = remove_outliers(
        vectors_flat_log, items, dataset_labels, save_dir
    )

    vec_clean = np.asarray(vec_clean)
    if vec_clean.ndim != 2:
        raise ValueError(f"vec_clean 维度异常: {vec_clean.ndim}, shape={vec_clean.shape}")

    keep_mask = np.asarray(keep_mask).astype(bool)
    items_clean = list(items_clean)
    ds_clean = np.asarray(ds_clean)

    subj_clean = [subject_labels[i] for i in range(len(subject_labels)) if keep_mask[i]]

    psd_linear_clean = psd_linear_band[keep_mask]  # (Nclean,C,F)
    psd_log_clean = psd_log_band[keep_mask]        # (Nclean,C,F)

    if len(vec_clean) < K:
        print(f"[跳过] 去离群点后样本数 ({len(vec_clean)}) 少于 K={K}")
        return

    # ========= 2. KMeans：用 log(PSD-subj) (0.5~35Hz) =========
    kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
    cluster_id = kmeans.fit_predict(vec_clean)

    print(f"[结果] K={K} 聚类分布: {np.bincount(cluster_id)}")

    # ========= 3. 代表性 PSD-cluster（关键：按你给的几何中心定义算） =========
    # 先在线性域算中心：P_center = (mean sqrt(P))^2

    centers_linear_cxf = []
    for k in range(K):
        centers_linear_cxf.append(_sqrt_mean_square_center(psd_linear_clean[cluster_id == k], eps=LOG_EPS))
    centers_linear_cxf = np.stack(centers_linear_cxf, axis=0)  # (K,C,F)

    # 再把中心转成 log（保存用）：log(PSD_center)
    centers_log_cxf = np.log(centers_linear_cxf + LOG_EPS)

    """
    # ========= 3. 代表性 PSD-cluster（关键：先 log 再算“几何中心”） =========
    # 先转到 log 域：log(PSD)
    psd_log_clean = np.log(psd_linear_clean + LOG_EPS)  # (N,C,F)

    # 在 log 域计算中心：log_center = mean(log(PSD))
    centers_log_cxf = []
    for k in range(K):
        mask = (cluster_id == k)
        if np.any(mask):
            centers_log_cxf.append(np.mean(psd_log_clean[mask], axis=0))  # (C,F)
        else:
            # 极端情况：该簇为空，给一个可控的兜底（用全局均值）
            centers_log_cxf.append(np.mean(psd_log_clean, axis=0))  # (C,F)

    centers_log_cxf = np.stack(centers_log_cxf, axis=0)  # (K,C,F)

    # 如需线性域中心（用于后续线性计算/可视化）：PSD_center = exp(log_center)
    centers_linear_cxf = np.exp(centers_log_cxf)  # (K,C,F)
    """

    # flatten 两种都存
    centers_linear_flat = centers_linear_cxf.reshape(K, -1)
    centers_log_flat = centers_log_cxf.reshape(K, -1)

    # ========= 4. 保存 =========
    cluster_map = {items_clean[i]: int(cluster_id[i]) for i in range(len(items_clean))}
    with open(os.path.join(save_dir, "cluster_map.json"), "w") as f:
        json.dump(cluster_map, f, indent=2)

    # MATLAB 导出：导出“中心log曲线（跨通道均值）”
    export_for_matlab(save_dir, centers_log_cxf.mean(axis=1), freqs_band, K)

    np.save(os.path.join(save_dir, "items_clean.npy"), np.array(items_clean))
    np.save(os.path.join(save_dir, "vectors_clean_flat_log.npy"), vec_clean)               # (Nclean,C*F) log域
    np.save(os.path.join(save_dir, "psd_linear_clean_cxf.npy"), psd_linear_clean)          # (Nclean,C,F) 线性域
    np.save(os.path.join(save_dir, "psd_log_clean_cxf.npy"), psd_log_clean)                # (Nclean,C,F) log域
    np.save(os.path.join(save_dir, "dataset_labels_clean.npy"), np.array(ds_clean))
    np.save(os.path.join(save_dir, "subject_labels_clean.npy"), np.array(subj_clean))
    np.save(os.path.join(save_dir, "cluster_id.npy"), cluster_id)
    np.save(os.path.join(save_dir, "freqs_band.npy"), freqs_band)

    # 关键：几何中心（你要求“保存 log 之后，不归一化”）
    np.save(os.path.join(save_dir, "psd_cluster_centers_cxf_log.npy"), centers_log_cxf)    # (K,C,F) log
    np.save(os.path.join(save_dir, "psd_cluster_centers_flat_log.npy"), centers_log_flat)  # (K,C*F) log

    # 同时保存线性中心（训练端如果要直接用比值滤波更方便）
    np.save(os.path.join(save_dir, "psd_cluster_centers_cxf_linear.npy"), centers_linear_cxf)    # (K,C,F) linear
    np.save(os.path.join(save_dir, "psd_cluster_centers_flat_linear.npy"), centers_linear_flat)  # (K,C*F) linear

    # ========= 5. 可视化 =========
    _plot_cluster_centers_meanch(save_dir, centers_log_cxf, freqs_band,
                                 title_prefix="PSD-cluster center (log, sqrt-mean-square, 0.5-35Hz)")
    _plot_dataset_composition(save_dir, cluster_id, ds_clean, SELECTED_DATASETS)

    # ========= 6. PCA（只可视化：用 log 向量） =========
    try:
        n_components = 2
        if vec_clean.shape[0] >= 2 and vec_clean.shape[1] >= 2:
            pca = PCA(n_components=n_components)
            pca_coords = pca.fit_transform(vec_clean)
            np.save(os.path.join(save_dir, "pca_coords.npy"), pca_coords)

            plt.figure()
            for k in range(K):
                idxk = (cluster_id == k)
                plt.scatter(pca_coords[idxk, 0], pca_coords[idxk, 1], s=8, label=f"C{k}")
            plt.title("PCA of PSD-subj (log, 0.5-35Hz)")
            plt.xlabel("PC1")
            plt.ylabel("PC2")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "pca_scatter.png"), dpi=220)
            plt.close()
    except Exception:
        pass

    print(f"[保存完成] Clean 聚类结果已保存至 → {save_dir}")

# =========================================================
# 主程序
# =========================================================
def main():
    print("=== 开始运行 Cluster Builder (Welch 5s -> PSD30 -> PSD-subj(log) + KMeans) ===")

    params = SimpleNamespace(
        datasets_dir=DATASETS_DIR,
        target_domains=",".join(SELECTED_DATASETS),
        batch_size=1, num_workers=0, max_samples_per_dataset=None
    )

    loader = LoadDataset(params)
    all_pairs, _ = loader.load_path(loader.targets_dirs, start_subject_id=0)
    print(f"[信息] 原始扫描共发现 {len(all_pairs)} 个 .npy 文件片段。")

    structure = organize_files_by_subject(all_pairs, SELECTED_DATASETS)
    run_sanity_check(structure, SELECTED_DATASETS, fs=FS)

    # [新增] 推断 expected epoch length
    global EXPECTED_EPOCH_LEN
    EXPECTED_EPOCH_LEN = infer_expected_epoch_len(structure, SELECTED_DATASETS, fs=FS)

    print("\n[步骤 3/4] 计算 PSD-subj 特征 ...")
    items, vectors_flat_log, ds_labels, sb_labels, freqs_band, psd_linear_band, psd_log_band = compute_psd_main(
        structure,
        SELECTED_DATASETS,
        expected_len=EXPECTED_EPOCH_LEN,
        force_fixed=FORCE_FIXED_EPOCH_LEN,
        fs=FS
    )

    print(f"[信息] 特征计算完成。共得到 {len(items)} 个受试者的特征向量。")
    if len(vectors_flat_log) > 0:
        print(f"[信息] vectors_flat_log.shape = {vectors_flat_log.shape} (应为 [N, C*Fband])")
        if freqs_band is not None and len(freqs_band) > 1:
            df = float(freqs_band[1] - freqs_band[0])
            print(f"[信息] freqs_band点数={len(freqs_band)} | 频率分辨率≈{df:.6f} Hz | 频率范围≈[{freqs_band[0]:.3f}, {freqs_band[-1]:.3f}] Hz")

    print("\n[步骤 4/4] 执行多 K 值聚类...")
    for K in K_LIST:
        run_clustering_for_k(K, items, vectors_flat_log, ds_labels, sb_labels, freqs_band, psd_linear_band, psd_log_band)

    print("\n=== 所有任务已完成 ===")


if __name__ == "__main__":
    main()