import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import wasserstein_distance
from tqdm import tqdm
import warnings
import sys
from types import SimpleNamespace
import random
import pandas as pd

# 1. 导入模型
sys.path.append(os.getcwd())
try:
    from original.models.model import Model
except ImportError:
    sys.path.append(os.path.join(os.getcwd(), 'wujiHDGDDP'))
    try:
        from original.models.model import Model
    except ImportError:
        print("[Error] Could not import Model.")

warnings.filterwarnings("ignore")

# ==========================================
# [配置区]
# ==========================================
CHECKPOINT_PATH = "/data/lijinyang/1_SleepHDG主要运行代码/wujiHDGDDP/results/SleepHDG_all数据集_5轮test一次_用8类算好的几何中心——test指标_2025-12-29_13-00-44/fold0/best_test_acc_a0.8320_f0.7750_ep44.pth"
DATA_ROOT = "/fdata/lijinyang/datasets_dir2_all_Merged"
SOURCE_DOMAIN = 'sleep-edfx'

TARGET_DOMAINS = [
    'ISRUC',
    'ABC',
    'MESA',
    'MROS1',
    'CFS',
    'CCSHS',
]

N_CLUSTERS = 8
FIXED_SEQ_LEN = 20
FIXED_CHANNELS = 2
FIXED_TIME_LEN = 3000
LATENT_DIM = 128


# ==========================================
# 数据处理
# ==========================================
def robust_standardize_input(x):
    if isinstance(x, np.ndarray): x = torch.from_numpy(x.copy()).float()
    if x.ndim == 1:
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.ndim == 2:
        x = x.shape[0] == FIXED_SEQ_LEN and x.unsqueeze(1) or x.shape[0] == FIXED_CHANNELS and x.unsqueeze(0) or \
            x.shape[1] == FIXED_CHANNELS and x.T.unsqueeze(0) or x.T.unsqueeze(1)
    if x.shape[0] != FIXED_SEQ_LEN:
        if x.shape[0] == 1:
            x = x.repeat(FIXED_SEQ_LEN, 1, 1)
        else:
            x = x.repeat((FIXED_SEQ_LEN // x.shape[0]) + 1, 1, 1)[:FIXED_SEQ_LEN]
    if x.shape[1] != FIXED_CHANNELS:
        if x.shape[1] == 1:
            x = x.repeat(1, FIXED_CHANNELS, 1)
        else:
            x = x[:, :FIXED_CHANNELS, :]
    if x.shape[2] != FIXED_TIME_LEN:
        x = x.reshape(-1, x.shape[1], x.shape[2])
        x = F.interpolate(x, size=FIXED_TIME_LEN, mode='linear', align_corners=False)
        x = x.view(FIXED_SEQ_LEN, FIXED_CHANNELS, FIXED_TIME_LEN)
    if x.shape == (FIXED_SEQ_LEN, FIXED_CHANNELS, FIXED_TIME_LEN): return x
    return None


def compute_spectral_feature(x, target_dim=128):
    if isinstance(x, np.ndarray): x = torch.from_numpy(x).float()
    x_flat = x.view(-1, x.shape[-1])
    x_flat = x_flat - x_flat.mean(dim=-1, keepdim=True)
    w = torch.hann_window(x_flat.shape[-1], device=x.device)
    fft = torch.fft.rfft(x_flat * w, dim=-1).abs() ** 2
    psd = fft.mean(dim=0)
    limit = max(32, psd.shape[0] // 4)
    psd = psd[:limit]
    psd = psd.view(1, 1, -1)
    psd_proj = F.interpolate(psd, size=target_dim, mode='linear', align_corners=False)
    return psd_proj.view(-1).numpy()


def extract_features(data_root, dataset_name, model, device, max_samples=800):
    print(f"[Info] Extracting {dataset_name}...")
    ds_path = os.path.join(data_root, dataset_name)
    files = []
    for root, _, fs in os.walk(ds_path):
        for f in fs:
            if f.endswith('.npy'): files.append(os.path.join(root, f))
    if len(files) > max_samples: files = random.sample(files, max_samples)

    raw_feats = []
    latent_feats = []
    model.eval()
    for f in tqdm(files, leave=False):
        try:
            raw = np.load(f, mmap_mode='r')
            x_std = robust_standardize_input(raw)
            if x_std is None: continue

            raw_f = compute_spectral_feature(x_std, target_dim=LATENT_DIM)
            raw_feats.append(raw_f)

            with torch.no_grad():
                batch_x = x_std.unsqueeze(0).to(device)
                _, mu = model.ae(batch_x)
                feat = mu.mean(dim=1).squeeze(0).cpu()
                if feat.shape[0] != LATENT_DIM:
                    feat = F.interpolate(
                        feat.view(1, 1, -1),
                        size=LATENT_DIM,
                        mode='linear',
                        align_corners=False
                    ).view(-1)
                latent_feats.append(feat.numpy())
        except:
            continue
    return np.array(raw_feats), np.array(latent_feats)


def fit_anchors(s_feats, n_clusters=8):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(s_feats)
    return kmeans.cluster_centers_


def compute_similarity_to_anchors(t_feats, anchors):
    sim_matrix = cosine_similarity(t_feats, anchors)
    max_sims = np.max(sim_matrix, axis=1)
    return max_sims


# ==========================================
# 绘图逻辑：ECDF + Wasserstein Table (修复版)
# ==========================================
def compute_ecdf(data):
    """Aux function to get x, y for ECDF plotting manually"""
    x = np.sort(data)
    n = x.size
    y = np.arange(1, n + 1) / n
    return x, y


def plot_ecdf_with_metrics(results, sim_src_ref, out_path="Figure4_ECDF_Wasserstein.pdf"):
    print("[Plotting] ECDF with Wasserstein Distance Table...")

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Liberation Serif']
    plt.rcParams['axes.linewidth'] = 0.8

    # 调色板
    palette = sns.color_palette("tab10", n_colors=max(10, len(results)))
    color_map = {r['name']: palette[i] for i, r in enumerate(results)}

    # 创建 Figure
    fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=300)

    # -------------------------------------------------------
    # 1. 绘制 Source Reference (带填充)
    # -------------------------------------------------------
    src_x, src_y = compute_ecdf(sim_src_ref)

    ax.step(src_x, src_y, where='post', color='gray', linestyle='-', linewidth=2, label='Source Reference', zorder=1)
    # 仅填充 Source
    ax.fill_between(src_x, src_y, step='post', color='gray', alpha=0.15, zorder=0)

    # -------------------------------------------------------
    # 2. 绘制 Targets (Before/After) 并计算指标
    # -------------------------------------------------------
    table_data = []  # 存储指标数据：[Dataset, W-Dist(Pre), W-Dist(Post), Gain%]

    # 收集所有数据以确定 x 轴范围
    all_data = [src_x]

    for r in results:
        name = r['name']
        sim_b = r['before']
        sim_a = r['after']
        c = color_map[name]

        all_data.append(sim_b)
        all_data.append(sim_a)

        # 计算 Wasserstein 距离
        wd_before = wasserstein_distance(sim_b, sim_src_ref)
        wd_after = wasserstein_distance(sim_a, sim_src_ref)
        improv = (wd_before - wd_after) / wd_before * 100.0 if wd_before > 0 else 0.0

        table_data.append([name, f"{wd_before:.4f}", f"{wd_after:.4f}", f"{improv:+.1f}%"])

        # 绘制 Before (虚线, 无填充)
        bx, by = compute_ecdf(sim_b)
        ax.step(bx, by, where='post', color=c, linestyle='--', linewidth=1.2, alpha=0.7, zorder=2)

        # 绘制 After (实线, 无填充, 更加醒目)
        # [修复] 变量名从 ax, ay 改为 ax_vals, ay_vals，避免覆盖 matplotlib 的 ax 对象
        ax_vals, ay_vals = compute_ecdf(sim_a)
        ax.step(ax_vals, ay_vals, where='post', color=c, linestyle='-', linewidth=2.5, alpha=0.9, zorder=3)

    # -------------------------------------------------------
    # 3. 设置轴和图例
    # -------------------------------------------------------
    flat_data = np.concatenate(all_data)
    xmin = np.percentile(flat_data, 5.0)
    xmax = 1.002
    ax.set_xlim(left=max(0.0, xmin - 0.02), right=xmax)
    ax.set_ylim(-0.02, 1.02)

    ax.set_xlabel("Cosine Similarity to Structural Center", fontsize=12)
    ax.set_ylabel("Cumulative Probability (ECDF)", fontsize=12)
    ax.grid(True, which='major', linestyle=':', linewidth=0.5, color='gray', alpha=0.3)

    # 构造自定义图例
    custom_lines = [
        plt.Line2D([0], [0], color='gray', lw=2, linestyle='-'),
        plt.Line2D([0], [0], color='black', lw=1.2, linestyle='--', alpha=0.7),
        plt.Line2D([0], [0], color='black', lw=2.5, linestyle='-', alpha=0.9),
    ]
    ax.legend(custom_lines, ['Source (Reference)', 'Target Before (Raw)', 'Target After (Aligned)'],
              loc='lower right', frameon=True, fontsize=10, fancybox=False, edgecolor='k')

    # -------------------------------------------------------
    # 4. 嵌入量化表格 (B+C 的核心)
    # -------------------------------------------------------
    col_labels = ["Dataset", "WD(Pre)", "WD(Post)↓", "Gain%"]

    avg_gain = np.mean([float(x[3].strip('%')) for x in table_data])
    table_data.append(["MEAN", "-", "-", f"{avg_gain:+.1f}%"])

    # 绘制表格
    the_table = ax.table(cellText=table_data,
                         colLabels=col_labels,
                         loc='upper left',
                         bbox=[0.02, 0.45, 0.45, 0.5])

    the_table.auto_set_font_size(False)
    the_table.set_fontsize(8)
    the_table.scale(1, 1.3)

    for (row, col), cell in the_table.get_celld().items():
        cell.set_edgecolor('white')
        cell.set_facecolor('none')
        cell.set_text_props(color='black')
        if row == 0:
            cell.set_text_props(weight='bold', color='#333333')
        if row == len(table_data):
            cell.set_text_props(weight='bold')
        if col == 3 and row > 0:
            cell.set_text_props(color='darkgreen', weight='bold')

    sns.despine(ax=ax, trim=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"[Success] Saved {out_path} with metric table.")

    print("\n[Wasserstein Distance Table]")
    print(f"{'Dataset':<10} | {'WD(Pre)':<10} | {'WD(Post)':<10} | {'Gain':<10}")
    print("-" * 50)
    for row in table_data[:-1]:
        print(f"{row[0]:<10} | {row[1]:<10} | {row[2]:<10} | {row[3]:<10}")
    print("-" * 50)
    print(f"Average Gain: {avg_gain:+.2f}%")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    params = SimpleNamespace(num_of_classes=5, d_model=128, nhead=4, num_layers=1, dim_feedforward=512, dropout=0.1,
                             activation="relu")
    try:
        model = Model(params).to(device)
        state_dict = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(state_dict, strict=False)
    except Exception as e:
        print(f"[Error] {e}")
        return

    s_raw, s_lat = extract_features(DATA_ROOT, SOURCE_DOMAIN, model, device)
    if s_raw is None or s_lat is None or len(s_raw) == 0 or len(s_lat) == 0:
        print("[Error] Source features empty.")
        return

    print("[Metric] Fitting Source Anchors...")
    anchors_raw = fit_anchors(s_raw, N_CLUSTERS)
    anchors_lat = fit_anchors(s_lat, N_CLUSTERS)

    print("[Metric] Computing Source reference...")
    sim_src = compute_similarity_to_anchors(s_lat, anchors_lat)

    results = []
    for td in TARGET_DOMAINS:
        t_raw, t_lat = extract_features(DATA_ROOT, td, model, device)
        if t_raw is None or t_lat is None or len(t_raw) == 0 or len(t_lat) == 0:
            print(f"[Warn] Skip {td}: empty features.")
            continue

        print(f"[Metric] Computing {td}...")
        sim_before = compute_similarity_to_anchors(t_raw, anchors_raw)
        sim_after = compute_similarity_to_anchors(t_lat, anchors_lat)

        results.append({
            'name': td,
            'before': sim_before,
            'after': sim_after,
        })

    if len(results) == 0:
        print("[Error] No valid target results.")
        return

    plot_ecdf_with_metrics(results, sim_src, out_path="Figure4_ECDF_Wasserstein.pdf")


if __name__ == "__main__":
    main()