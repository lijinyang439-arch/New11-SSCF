import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import seaborn as sns
import random
from sklearn.cluster import MiniBatchKMeans
import warnings
import multiprocessing
from tqdm import tqdm

# 忽略警告
warnings.filterwarnings("ignore")

# ==========================================
# 0. 配色方案 (NPG - Nature 风格)
# ==========================================
# 这是一组精心挑选的科研绘图常用色，对比度高且不刺眼
NPG_COLORS = [
    "#E64B35",  # 朱红 (热烈，Mode 1)
    "#4DBBD5",  # 湖蓝 (清新，Mode 2)
    "#00A087",  # 翡翠绿 (沉稳，Mode 3)
    "#3C5488",  # 藏青 (专业，Mode 4)
    "#F39B7F",  # 珊瑚粉 (柔和，Mode 5)
    "#8491B4",  # 灰蓝 (中性，Mode 6)
    "#91D1C2",  # 浅青 (明快，Mode 7)
    "#DC0000",  # 深红 (警示，Mode 8)
    "#7E6148",  # 褐色 (备用)
    "#B09C85"  # 米色 (备用)
]


# ==========================================
# 1. 核心计算逻辑
# ==========================================
def compute_psd_fixed(x, target_length=128, fs=100.0):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float()

    dims = x.shape
    max_dim_idx = torch.argmax(torch.tensor(dims)).item()
    if max_dim_idx != x.ndim - 1:
        if x.ndim == 2:
            x = x.permute(1, 0)
        elif x.ndim == 3:
            perm = list(range(x.ndim))
            perm.pop(max_dim_idx)
            perm.append(max_dim_idx)
            x = x.permute(*perm)

    L = x.shape[-1]
    if L < 64: return None, None

    x = x - x.mean(dim=-1, keepdim=True)
    w = torch.hann_window(L, periodic=True)
    x_w = x * w
    fft = torch.fft.rfft(x_w, dim=-1).abs() ** 2

    if x.ndim >= 3:
        psd = fft.mean(dim=0).mean(dim=0)
    elif x.ndim == 2:
        psd = fft.mean(dim=0)
    else:
        psd = fft

    limit = max(32, psd.shape[0] // 4)
    psd_crop = psd[:limit]

    # 该样本低频段最高频率 (Hz)
    # rFFT 第 k 个点对应频率：k * fs / L
    if limit >= 2:
        f_max = (limit - 1) * float(fs) / float(L)
    else:
        f_max = 0.0

    if psd_crop.shape[0] != target_length:
        psd_in = psd_crop.view(1, 1, -1)
        psd_out = F.interpolate(psd_in, size=target_length, mode='linear', align_corners=False)
        psd_final = psd_out.view(-1)
    else:
        psd_final = psd_crop

    return psd_final.numpy(), float(f_max)


# ==========================================
# 2. 多进程处理
# ==========================================
def process_file_wrapper(args_tuple):
    fpath, fs = args_tuple
    try:
        data = np.load(fpath, mmap_mode='r')
        if data.ndim == 3:
            x = data[0].copy()
        elif data.ndim == 2:
            x = data.copy()
        else:
            return None, None
        return compute_psd_fixed(x, target_length=128, fs=fs)
    except:
        return None, None


def scan_files(data_root, datasets, max_files=30000):
    file_list = []
    print(f"[Info] Scanning datasets: {datasets} ...")
    for ds_name in datasets:
        ds_path = os.path.join(data_root, ds_name)
        if not os.path.exists(ds_path): continue

        search_paths = [os.path.join(ds_path, 'seq'), ds_path]
        found = False
        for sp in search_paths:
            if os.path.exists(sp):
                files = []
                for root, _, fs in os.walk(sp):
                    for f in fs:
                        if f.endswith('.npy'):
                            files.append(os.path.join(root, f))
                if files:
                    file_list.extend(files)
                    print(f"  - {ds_name}: {len(files)} files found")
                    found = True
                    break

    if len(file_list) > max_files:
        print(f"[Info] Sampling {max_files} files...")
        return random.sample(file_list, max_files)
    return file_list


# ==========================================
# 3. 排序算法
# ==========================================
def sort_clusters_by_centroid(X, labels, n_clusters):
    centroids = []
    for i in range(n_clusters):
        indices = np.where(labels == i)[0]
        if len(indices) == 0:
            centroids.append(0)
            continue

        mode_samples = X[indices]
        consensus = (np.mean(np.sqrt(np.maximum(mode_samples, 1e-8)), axis=0)) ** 2

        freq_idx = np.arange(len(consensus))
        spectral_centroid = np.sum(freq_idx * consensus) / np.sum(consensus)
        centroids.append(spectral_centroid)

    sorted_indices = np.argsort(centroids)
    mapping = {old: new for new, old in enumerate(sorted_indices)}
    new_labels = np.array([mapping[l] for l in labels])
    return new_labels


# ==========================================
# 4. 绘图功能 (Nature 风格配色)
# ==========================================
def plot_grid_view(X_psd, labels, n_clusters, output_path, freq_axis_hz):
    print(f"[Plotting] Generating Grid View (NPG Colors) -> {output_path}")

    # 设置全局字体
    sns.set(style="whitegrid", context="paper")
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']

    fig, axes = plt.subplots(2, 4, figsize=(18, 7), sharey=True, sharex=True)
    axes = axes.flatten()

    # 使用自定义 NPG 色板
    colors = NPG_COLORS[:n_clusters]

    x_axis = freq_axis_hz
    x_new = np.linspace(float(x_axis.min()), float(x_axis.max()), 300)

    for i in range(n_clusters):
        ax = axes[i]
        color = colors[i]
        indices = np.where(labels == i)[0]
        count = len(indices)

        if count == 0: continue

        mode_samples = X_psd[indices]

        # 归一化 (最高点为1)
        consensus_raw = (np.mean(np.sqrt(np.maximum(mode_samples, 1e-8)), axis=0)) ** 2
        max_val = np.max(consensus_raw) if np.max(consensus_raw) > 0 else 1.0

        mode_samples_norm = mode_samples / max_val
        consensus_norm = consensus_raw / max_val

        # 1. 范围带 (10-90th)
        lower_bound = np.percentile(mode_samples_norm, 10, axis=0)
        upper_bound = np.percentile(mode_samples_norm, 90, axis=0)

        spl_low = make_interp_spline(x_axis, lower_bound, k=3)
        spl_high = make_interp_spline(x_axis, upper_bound, k=3)
        y_low = np.maximum(spl_low(x_new), 0)
        y_high = np.maximum(spl_high(x_new), 0)

        # 绘制半透明带 (alpha=0.3 配合 NPG 颜色很显质感)
        ax.fill_between(x_new, y_low, y_high, color=color, alpha=0.3, linewidth=0)

        # 2. 几何中心
        spl_c = make_interp_spline(x_axis, consensus_norm, k=3)
        y_c = np.maximum(spl_c(x_new), 0)

        # 白边更粗一点，增强立体感
        ax.plot(x_new, y_c, color='white', linewidth=4.0, alpha=1.0)
        ax.plot(x_new, y_c, color=color, linewidth=2.5, label='Geometric Consensus')

        # 标题和 N 值
        ax.set_title(f"Mode $\mathcal{{K}}_{{{i + 1}}}$", fontsize=13, fontweight='bold', color='#333333')
        ax.text(0.95, 0.90, f"N={count}", transform=ax.transAxes, ha='right', va='top',
                fontsize=10, style='italic',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='#cccccc', boxstyle='round,pad=0.2'))

        # 美化边框
        sns.despine(ax=ax, left=True, bottom=False)
        ax.grid(True, linestyle='--', alpha=0.4)  # 网格线淡一点

        if i >= 4: ax.set_xlabel("Frequency (Hz)", fontsize=11)
        if i % 4 == 0: ax.set_ylabel("Norm. Magnitude", fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')


def plot_overlay_view(X_psd, labels, n_clusters, output_path, freq_axis_hz):
    print(f"[Plotting] Generating Overlay View (NPG Colors) -> {output_path}")
    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid", context="paper")
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']

    colors = NPG_COLORS[:n_clusters]
    x_axis = freq_axis_hz
    x_new = np.linspace(float(x_axis.min()), float(x_axis.max()), 300)

    for i in range(n_clusters):
        indices = np.where(labels == i)[0]
        if len(indices) == 0: continue

        mode_samples = X_psd[indices]

        # 归一化
        consensus_raw = (np.mean(np.sqrt(np.maximum(mode_samples, 1e-8)), axis=0)) ** 2
        max_val = np.max(consensus_raw) if np.max(consensus_raw) > 0 else 1.0

        mode_samples_norm = mode_samples / max_val
        consensus_norm = consensus_raw / max_val

        # IQR 范围 (25-75th)
        lower_bound = np.percentile(mode_samples_norm, 25, axis=0)
        upper_bound = np.percentile(mode_samples_norm, 75, axis=0)

        spl_low = make_interp_spline(x_axis, lower_bound, k=3)
        spl_high = make_interp_spline(x_axis, upper_bound, k=3)
        y_low = np.maximum(spl_low(x_new), 0)
        y_high = np.maximum(spl_high(x_new), 0)

        # 叠加时透明度再低一点，避免糊在一起
        plt.fill_between(x_new, y_low, y_high, color=colors[i], alpha=0.08)

        # 几何中心
        spl_c = make_interp_spline(x_axis, consensus_norm, k=3)
        y_c = np.maximum(spl_c(x_new), 0)

        # 同样加白边区分
        plt.plot(x_new, y_c, color='white', linewidth=2.5, alpha=0.8)
        plt.plot(x_new, y_c, color=colors[i], linewidth=2.0, label=f"Mode $\mathcal{{K}}_{{{i + 1}}}$")

    plt.title("Structural Modes Comparison (Normalized Topology)", fontsize=15, fontweight='bold', pad=15)
    plt.xlabel("Frequency (Hz)", fontsize=12)
    plt.ylabel("Normalized Magnitude (a.u.)", fontsize=12)

    # 图例优化
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False, fontsize=10)
    sns.despine()
    plt.grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')


# ==========================================
# 5. 主程序
# ==========================================
def main(args):
    files = scan_files(args.data_root, args.datasets, max_files=args.num_samples)
    if not files: return

    print(f"[Info] Loading data with {args.num_workers} workers...")
    with multiprocessing.Pool(processes=args.num_workers) as pool:
        # 传 (path, fs)
        results = list(tqdm(pool.imap(process_file_wrapper, [(fp, args.fs) for fp in files]), total=len(files)))

    features = []
    fmax_list = []
    for psd, fmax in results:
        if psd is not None and fmax is not None:
            features.append(psd)
            fmax_list.append(fmax)

    if len(features) < args.n_clusters: return

    X_psd = np.array(features)
    print(f"[Info] Feature Matrix Shape: {X_psd.shape}")

    # 用所有样本的 f_max 中位数构造统一 Hz 轴
    fmax_median = float(np.median(np.array(fmax_list))) if len(fmax_list) > 0 else 0.0
    freq_axis_hz = np.linspace(0.0, fmax_median, X_psd.shape[1])

    print(f"[Info] fs = {args.fs} Hz, f_max(median) = {fmax_median:.4f} Hz")

    print(f"[Info] Running K-Means (K={args.n_clusters})...")
    kmeans = MiniBatchKMeans(n_clusters=args.n_clusters, random_state=42, batch_size=2048, n_init=3)
    raw_labels = kmeans.fit_predict(X_psd)

    labels = sort_clusters_by_centroid(X_psd, raw_labels, args.n_clusters)

    plot_grid_view(X_psd, labels, args.n_clusters, "Figure3_A_Final_Colors.pdf", freq_axis_hz)
    plot_overlay_view(X_psd, labels, args.n_clusters, "Figure3_B_Final_Colors.pdf", freq_axis_hz)

    print("\n[Success] Generated optimized figures with Hz x-axis!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/fdata/lijinyang/datasets_dir2_all_Merged",
                        help="Data root directory")
    parser.add_argument("--datasets", nargs='+', default=['sleep-edfx', 'ISRUC', 'HMC', 'SHHS1', 'P2018','ABC','CCSHS','CFS','MESA','MROS1','MROS2'],
                        help="Datasets list")
    parser.add_argument("--n_clusters", type=int, default=8, help="Number of structural modes")
    parser.add_argument("--num_samples", type=int, default=50000, help="Total samples")
    parser.add_argument("--num_workers", type=int, default=64, help="CPU cores")
    parser.add_argument("--fs", type=float, default=100.0, help="Sampling rate (Hz)")

    args = parser.parse_args()
    main(args)
