#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute geometric centers from clustering assignment (cluster_map.json),
WITHOUT requiring cluster_x folder split.

输入：
- 源数据集根目录 datasets_dir（包含各数据集，内部递归寻找 seq/labels）
- 聚类结果目录 cluster_result_dir（例如 RESULTS_ROOT/K_08），其中应包含 cluster_map.json
  cluster_map.json: {"DS/SUBJ": cluster_id, ...}

输出：
- <output_dir>/<prefix>_centers.npy              (K, C*Fr) 线性域 PSD 中心（sqrt-mean-square）
- <output_dir>/<prefix>_centers_cxf.npy          (K, C, Fr) 线性域 PSD 中心
- <output_dir>/<prefix>_freqs_band.npy           (Fr,)
- <output_dir>/<prefix>_map.json                 meta 信息（不是逐样本 map）
- <output_dir>/<prefix>_cluster_subjects.json    每个簇包含哪些 subject（ds/subj）

PSD 定义：与 SSA.py 中 SpectralStructureAlignment1d 完全一致
- 对每个 epoch (C, L) 做分段：段长 F_len，hop 默认 F_len//2
- 每段乘归一化 Hann（w / ||w||）
- rfft(n=F_len) -> P=|X|^2
- 对所有段的 P 做均值得到 (C, Fr)，其中 Fr = F_len//2 + 1
- subject PSD = 所有 epoch 的 (C,Fr) 均值（线性域）
- 几何中心：P_center = (mean(sqrt(P_i)))^2 （在线性域）

注意：
- 这里不再做 0.5~35Hz band 截取，因为 SSA 的外部中心维度必须是 C*Fr 才能和训练时 P_hat 对齐。
"""

# =========================================================
# 【关键】防止 BLAS / MKL / OpenBLAS 抢核（先设 1）
# 必须在 numpy / sklearn 之前设置
# =========================================================
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import json
import numpy as np
from pathlib import Path
from types import SimpleNamespace
from collections import defaultdict, Counter
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


# =========================================================
# 1) SSA 一致的频谱参数
# =========================================================
FS = 100

# SSA 里的 F_len 默认就是 5（你训练报错里 64*3=192 对应 F_len=5）
SSA_F_LEN = 5
SSA_HOP = None          # None 表示 hop = F_len//2
LOG_EPS = 1e-12


# =========================================================
# 2) 数据路径扫描（复用你第二脚本风格）
# =========================================================
class LoadDataset(object):
    def __init__(self, params: SimpleNamespace):
        self.params = params
        self.datasets_map = {
            'SHHS1': 0, 'P2018': 0, 'MROS1': 0, 'MROS2': 0, 'MESA': 0,
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


# =========================================================
# 3) SSA 一致的谱编码：分段 + 归一化Hann + rfft(n=F_len) + 段均值
# =========================================================
def _hann_window_norm(F_len: int):
    w = np.hanning(F_len).astype(np.float64)
    nrm = np.linalg.norm(w)
    if nrm <= 0:
        return w
    return w / nrm


def _segment_indices(L: int, F_len: int, hop: int):
    if L < F_len:
        return np.array([0], dtype=np.int64), 1
    num = 1 + (L - F_len) // hop
    starts = np.arange(0, num * hop, hop, dtype=np.int64)
    return starts, hop


def compute_ssa_psd_from_epoch(ep_ct, fs, F_len=SSA_F_LEN, hop=SSA_HOP):
    """
    返回：
      freqs: (Fr,)
      psd_epoch: (C, Fr) 线性域 PSD（SSA一致）
    """
    ep_ct = np.asarray(ep_ct, dtype=np.float64)
    if ep_ct.ndim != 2:
        return None, None

    C, L = ep_ct.shape
    F_len = int(F_len)
    if hop is None:
        hop = max(1, F_len // 2)
    hop = int(hop)

    w = _hann_window_norm(F_len)
    starts, hop = _segment_indices(L, F_len, hop)

    # SSA: L < F_len 时 pad
    if starts.size == 1 and L < F_len:
        pad = F_len - L
        ep_pad = np.pad(ep_ct, ((0, 0), (0, pad)), mode="constant")
    else:
        ep_pad = ep_ct

    seg_psd = []
    for s in starts.tolist():
        seg = ep_pad[:, s:s + F_len]
        if seg.shape[-1] < F_len:
            seg = np.pad(seg, ((0, 0), (0, F_len - seg.shape[-1])), mode="constant")

        seg = seg * w[None, :]
        Xf = np.fft.rfft(seg, n=F_len, axis=-1)
        P = (np.abs(Xf) ** 2)  # (C, Fr)
        seg_psd.append(P)

    P_stack = np.stack(seg_psd, axis=0)      # (num_seg, C, Fr)
    psd_epoch = np.mean(P_stack, axis=0)     # (C, Fr)

    freqs = np.fft.rfftfreq(F_len, d=1.0 / float(fs)).astype(np.float64)  # (Fr,)
    return freqs, psd_epoch


def _sqrt_mean_square_center(psd_linear_ncf, eps=1e-12):
    """
    几何中心定义：
      P_center = ( mean( sqrt(P_i) ) )^2
    输入：psd_linear_ncf: (N,C,F) 线性域 PSD
    输出：center_linear: (C,F) 线性域 PSD
    """
    x = np.asarray(psd_linear_ncf, dtype=np.float64)
    x = np.clip(x, a_min=0.0, a_max=None)
    amp = np.sqrt(x + eps)             # (N,C,F)
    amp_mean = np.mean(amp, axis=0)    # (C,F)
    center = amp_mean ** 2             # (C,F)
    return center


# =========================================================
# 4) 仅针对 “指定 subject 列表” 计算 PSD-subj（线性域）
# =========================================================
def _process_one_subject_for_list(args):
    ds, subj, files, expected_len, force_fixed, fs, ssa_f_len, ssa_hop = args

    all_psd_epoch = []
    freqs_ref = None

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

                f, psd_ep = compute_ssa_psd_from_epoch(ep, fs, F_len=ssa_f_len, hop=ssa_hop)
                if f is None or psd_ep is None:
                    continue

                if freqs_ref is None:
                    freqs_ref = f

                all_psd_epoch.append(psd_ep)
                cnt_used += 1
        except Exception:
            pass

    if not all_psd_epoch or freqs_ref is None:
        return (None, None, None, ds, subj, None, cnt_total, cnt_used, cnt_skipped_len, dict(len_counter))

    # subject PSD = epoch 均值（SSA一致）
    psd_subj_linear = np.mean(np.stack(all_psd_epoch, axis=0), axis=0)  # (C,Fr)

    item = f"{ds}/{subj}"
    return (item, psd_subj_linear, freqs_ref, ds, subj, freqs_ref,
            cnt_total, cnt_used, cnt_skipped_len, dict(len_counter))


# =========================================================
# 5) 主流程：读 cluster_map.json -> 回源数据算 PSD-subj -> 按 cluster 聚合算中心
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets_dir", type=str, required=True,
                        help="源数据集根目录，例如 /fdata/lijinyang/datasets_dir2_all_Merged")
    parser.add_argument("--datasets", type=str, nargs="+", required=True,
                        help="例如 SHHS1 P2018 MROS1 ...")
    parser.add_argument("--cluster_result_dir", type=str, required=True,
                        help="聚类结果目录，例如 ./全部数据集聚类结果/.../K_08 (里面应有 cluster_map.json)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="输出目录")
    parser.add_argument("--prefix", type=str, default="ours-8",
                        help="输出前缀")
    parser.add_argument("--num_cores", type=int, default=64,
                        help="CPU 核心数量")
    parser.add_argument("--force_fixed_epoch_len", action="store_true",
                        help="强制过滤不一致长度 epoch（建议开）")
    parser.add_argument("--fs", type=int, default=FS)

    # 保留参数名（但不再用于 band 截取），避免你之前的调用脚本崩
    parser.add_argument("--band_low", type=float, default=0.5)
    parser.add_argument("--band_high", type=float, default=35.0)

    # SSA 对齐参数
    parser.add_argument("--ssa_f_len", type=int, default=SSA_F_LEN,
                        help="SSA 频谱编码段长 F_len（必须和训练 SSA 的 F_len 一致）")
    parser.add_argument("--ssa_hop", type=int, default=-1,
                        help="SSA hop；-1 表示 None -> hop=F_len//2（必须和训练 SSA 的 hop 一致）")

    args = parser.parse_args()

    # 尽量设置多进程/BLAS线程（注意 numpy 已导入，部分 BLAS 不一定完全生效，但至少对后续子进程有意义）
    os.environ["OMP_NUM_THREADS"] = str(args.num_cores)
    os.environ["MKL_NUM_THREADS"] = str(args.num_cores)
    os.environ["OPENBLAS_NUM_THREADS"] = str(args.num_cores)
    os.environ["NUMEXPR_NUM_THREADS"] = str(args.num_cores)

    ssa_hop = None if int(args.ssa_hop) < 0 else int(args.ssa_hop)

    Fr = (int(args.ssa_f_len) // 2) + 1
    print("=== 按聚类索引回源数据计算几何中心（SSA一致版）===")
    print(f"[Info] datasets_dir      = {args.datasets_dir}")
    print(f"[Info] datasets          = {args.datasets}")
    print(f"[Info] cluster_result_dir= {args.cluster_result_dir}")
    print(f"[Info] output_dir        = {args.output_dir}")
    print(f"[Info] prefix            = {args.prefix}")
    print(f"[Info] CPU Threads       = {args.num_cores}")
    print(f"[Info] force_fixed_epoch_len = {args.force_fixed_epoch_len}")
    print(f"[Info] fs               = {args.fs}")
    print(f"[Info] SSA F_len         = {int(args.ssa_f_len)} -> Fr={Fr}")
    print(f"[Info] SSA hop           = {ssa_hop} (None means F_len//2)")
    print("[Info] band_low/high 参数保留但不再用于截频（否则维度无法和 SSA 对齐）")

    # 1) 读取 cluster_map.json
    cluster_map_path = os.path.join(args.cluster_result_dir, "cluster_map.json")
    if not os.path.exists(cluster_map_path):
        raise FileNotFoundError(f"找不到 {cluster_map_path}（你给的 cluster_result_dir 不对，或该 K 目录未生成 map）")

    with open(cluster_map_path, "r") as f:
        cluster_map = json.load(f)

    items_all = list(cluster_map.keys())
    cluster_ids_all = np.array([int(cluster_map[k]) for k in items_all], dtype=np.int64)
    K = int(cluster_ids_all.max()) + 1
    print(f"[Info] cluster_map items = {len(items_all)} | inferred K = {K}")

    # 2) 扫描源数据集，建立 structure[ds][subj] -> files
    params = SimpleNamespace(
        datasets_dir=args.datasets_dir,
        target_domains=",".join(args.datasets),
        batch_size=1, num_workers=0, max_samples_per_dataset=None
    )
    loader = LoadDataset(params)
    all_pairs, _ = loader.load_path(loader.targets_dirs, start_subject_id=0)
    print(f"[信息] 原始扫描共发现 {len(all_pairs)} 个 .npy 文件片段。")
    structure = organize_files_by_subject(all_pairs, args.datasets)

    # 3) 推断 expected epoch len
    expected_len = infer_expected_epoch_len(structure, args.datasets, fs=args.fs)

    # 4) 只对 cluster_map 里出现的 subject 建任务
    tasks = []
    miss = 0
    for item in items_all:
        if "/" not in item:
            miss += 1
            continue
        ds, subj = item.split("/", 1)
        if ds not in structure or subj not in structure[ds]:
            miss += 1
            continue
        files = structure[ds][subj]
        if not files:
            miss += 1
            continue
        tasks.append((ds, subj, files, expected_len, args.force_fixed_epoch_len,
                      args.fs, int(args.ssa_f_len), ssa_hop))

    print(f"[Info] tasks(subjects) = {len(tasks)} | missing_in_source = {miss}")
    if len(tasks) == 0:
        raise RuntimeError("没有任何 subject 能在源数据中匹配到（ds/subj 对不上，或 datasets 选错）")

    # 5) 并行算 PSD-subj(linear)
    n_workers = min(args.num_cores, cpu_count())
    print(f"\n[信息] 使用 {n_workers} 个 CPU 核心并行计算 PSD-subj(linear, SSA-encoding)")

    subject_items = []
    psd_subj_linear_list = []
    freqs_ref = None

    # 统计
    stat_total = 0
    stat_used = 0
    stat_skipped_len = 0
    stat_len_counter_all = Counter()

    with Pool(processes=n_workers) as pool:
        for out in tqdm(pool.imap_unordered(_process_one_subject_for_list, tasks),
                        total=len(tasks), desc="PSD-subj并行计算"):
            item, psd_subj_linear, fref, ds, subj, fref2, cnt_total, cnt_used, cnt_skipped_len, len_counter = out

            stat_total += int(cnt_total)
            stat_used += int(cnt_used)
            stat_skipped_len += int(cnt_skipped_len)
            if isinstance(len_counter, dict):
                stat_len_counter_all.update(len_counter)

            if item is None or psd_subj_linear is None:
                continue

            subject_items.append(item)
            psd_subj_linear_list.append(psd_subj_linear)
            if freqs_ref is None and fref is not None:
                freqs_ref = fref

    print("\n" + "=" * 60)
    print(">>> 输入 epoch 长度统计（过滤前） <<<")
    print(f"[总体] epoch总数={stat_total} | 实际用于PSD={stat_used} | 因长度不一致跳过={stat_skipped_len}")
    if stat_total > 0:
        print(f"[总体] 用于PSD比例={(stat_used/stat_total)*100:.2f}% | 跳过比例={(stat_skipped_len/stat_total)*100:.2f}%")
    if len(stat_len_counter_all) > 0:
        top_all = stat_len_counter_all.most_common(10)
        top_str = " | ".join([f"{L}:{c}" for L, c in top_all])
        print(f"[总体] 最常见长度Top10: {top_str}")
        L_min = min(stat_len_counter_all.keys())
        L_max = max(stat_len_counter_all.keys())
        print(f"[总体] 长度范围: min={L_min}, max={L_max} (点) | 秒: {L_min/args.fs:.3f}~{L_max/args.fs:.3f}")
    print("=" * 60 + "\n")

    if len(psd_subj_linear_list) == 0 or freqs_ref is None:
        raise RuntimeError("所有 subject 的 PSD-subj 计算都失败了（可能全被 label=5 或读不到数据）")

    psd_linear = np.stack(psd_subj_linear_list, axis=0)  # (N,C,Fr)
    subject_items = np.array(subject_items)

    # 6) 把当前成功计算的 subject 映射回 cluster_id
    cluster_id_for_subject_items = []
    dropped = 0
    for it in subject_items:
        if it in cluster_map:
            cluster_id_for_subject_items.append(int(cluster_map[it]))
        else:
            dropped += 1
            cluster_id_for_subject_items.append(-1)
    cluster_id_for_subject_items = np.array(cluster_id_for_subject_items, dtype=np.int64)

    ok_mask = (cluster_id_for_subject_items >= 0)
    psd_linear = psd_linear[ok_mask]
    subject_items = subject_items[ok_mask]
    cluster_id_for_subject_items = cluster_id_for_subject_items[ok_mask]

    if psd_linear.shape[0] == 0:
        raise RuntimeError("subject_items 全部无法在 cluster_map 中找到对应 cluster（理论上不该发生）")

    # 7) 按 cluster 算几何中心（SSA一致）
    centers_cxf = []
    cluster_subjects = {}
    for k in range(K):
        mask = (cluster_id_for_subject_items == k)
        cluster_subjects[str(k)] = subject_items[mask].tolist()
        if not np.any(mask):
            # 空簇兜底：用全局几何中心（避免 NaN）
            centers_cxf.append(_sqrt_mean_square_center(psd_linear, eps=LOG_EPS))
        else:
            centers_cxf.append(_sqrt_mean_square_center(psd_linear[mask], eps=LOG_EPS))
    centers_cxf = np.stack(centers_cxf, axis=0)  # (K,C,Fr)
    centers_flat = centers_cxf.reshape(K, -1).astype(np.float32)

    # 8) 保存
    os.makedirs(args.output_dir, exist_ok=True)
    out_centers = os.path.join(args.output_dir, f"{args.prefix}_centers.npy")
    out_centers_cxf = os.path.join(args.output_dir, f"{args.prefix}_centers_cxf.npy")
    out_freqs = os.path.join(args.output_dir, f"{args.prefix}_freqs_band.npy")
    out_meta = os.path.join(args.output_dir, f"{args.prefix}_map.json")
    out_cluster_subjects = os.path.join(args.output_dir, f"{args.prefix}_cluster_subjects.json")

    np.save(out_centers, centers_flat)
    np.save(out_centers_cxf, centers_cxf.astype(np.float32))
    np.save(out_freqs, np.asarray(freqs_ref, dtype=np.float64))

    meta = {
        "datasets": args.datasets,
        "n_clusters": int(K),
        "feature_dim": int(centers_flat.shape[1]),
        "center_shape_flat": list(centers_flat.shape),
        "center_shape_cxf": list(centers_cxf.shape),
        "description": "Geometric centers computed by reading raw source datasets with cluster_map.json assignment (SSA-consistent encoding)",
        "cluster_result_dir": args.cluster_result_dir,
        "cluster_map_path": cluster_map_path,
        "fs": int(args.fs),
        "ssa_encoding": {
            "F_len": int(args.ssa_f_len),
            "hop": None if ssa_hop is None else int(ssa_hop),
            "Fr": int((int(args.ssa_f_len) // 2) + 1),
            "window": "hann_norm(w/||w||)",
            "rfft_n": int(args.ssa_f_len),
            "segment_mean": True
        },
        "expected_epoch_len": None if expected_len is None else int(expected_len),
        "force_fixed_epoch_len": bool(args.force_fixed_epoch_len),
        "subjects_used": int(psd_linear.shape[0]),
        "note": "band_low/high 参数保留但不用于截频；外部中心维度必须与 SSA 的 C*Fr 对齐，否则训练会报 cdist 维度错误"
    }
    with open(out_meta, "w") as f:
        json.dump(meta, f, indent=2)

    with open(out_cluster_subjects, "w") as f:
        json.dump(cluster_subjects, f, indent=2)

    print("\n===================== DONE =====================")
    print(f"[Saved] {out_centers}")
    print(f"[Saved] {out_centers_cxf}")
    print(f"[Saved] {out_freqs}")
    print(f"[Saved] {out_meta}")
    print(f"[Saved] {out_cluster_subjects}")
    print("================================================")


if __name__ == "__main__":
    main()
