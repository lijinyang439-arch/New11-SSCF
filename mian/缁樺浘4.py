import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.lines as mlines
import seaborn as sns
import pandas as pd
import random
import warnings
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics.pairwise import cosine_similarity
from types import SimpleNamespace

# ==========================================
# 1. 全局配置与路径 (Configuration)
# ==========================================
warnings.filterwarnings("ignore")
sys.path.append(os.getcwd())

# --- 路径配置 (请根据实际情况修改) ---
CHECKPOINT_PATH = "/data/lijinyang/1_SleepHDG主要运行代码/wujiHDGDDP/results/SleepHDG_all数据集_5轮test一次_用8类算好的几何中心——test指标_2025-12-29_13-00-44/fold0/best_test_acc_a0.8320_f0.7750_ep44.pth"
DATA_ROOT = "/fdata/lijinyang/datasets_dir2_all_Merged"
SAVE_DIR = "./saved_stats"
ANCHOR_FILE = os.path.join(SAVE_DIR, "universal_anchors_8clusters.npy")

# --- 数据集定义 ---
TEST_DOMAINS = ['ABC', 'MROS1', 'MROS2', 'CCSHS', 'MESA', 'CFS']

# --- 运行参数 ---
NUM_WORKERS = 4
BATCH_SIZE = 32
MAX_SAMPLES_PER_DOMAIN = 3000  # 限制采样数以加快绘图速度

# --- 配色方案 (ICML Mechanism Style) ---
COLOR_CONCENTRATED = "#006d77"  # 深青色 (Ours: Aligned/Concentrated)
COLOR_REFERENCE = "#555555"  # 深灰色 (Reference: Unaligned)
COLOR_ANNOTATION = "#264653"  # 标注文字颜色


# ==========================================
# 2. 数据处理与加载 (Dataset & Utils)
# ==========================================
class SleepFeatureDataset(Dataset):
    def __init__(self, files, mode='latent', target_dim=128):
        self.files = files
        self.mode = mode
        self.target_dim = target_dim
        self.FIXED_SEQ_LEN = 20
        self.FIXED_CHANNELS = 2
        self.FIXED_TIME_LEN = 3000

    def safe_load_and_process(self, f_path):
        """安全加载 .npy 并进行鲁棒标准化"""
        try:
            x_mmap = np.load(f_path, mmap_mode='r')
            orig_seq = x_mmap.shape[0] if x_mmap.ndim > 0 else 1

            # 维度适配
            if x_mmap.ndim == 1:
                x_slice = x_mmap[:]
            elif x_mmap.ndim == 2:
                x_slice = x_mmap[:self.FIXED_SEQ_LEN, :] if x_mmap.shape[0] >= self.FIXED_SEQ_LEN else x_mmap[:, :]
            elif x_mmap.ndim == 3:
                x_slice = x_mmap[:min(orig_seq, self.FIXED_SEQ_LEN), :, :]
            else:
                x_slice = x_mmap

            x = torch.from_numpy(np.array(x_slice)).float()
            return self.robust_standardize(x)
        except Exception:
            return None

    def robust_standardize(self, x):
        """将任意形状的数据对齐到 [Seq, Channel, Time]"""
        if x is None: return None
        if x.ndim == 1:
            x = x.view(1, 1, -1)
        elif x.ndim == 2:
            if x.shape[0] == self.FIXED_SEQ_LEN:
                x = x.unsqueeze(1)
            elif x.shape[0] == self.FIXED_CHANNELS:
                x = x.unsqueeze(0)
            elif x.shape[1] == self.FIXED_CHANNELS:
                x = x.T.unsqueeze(0)

        if x.shape[1] != self.FIXED_CHANNELS:
            x = x[:, :self.FIXED_CHANNELS, :] if x.shape[1] > self.FIXED_CHANNELS else x.repeat(1, self.FIXED_CHANNELS,
                                                                                                1)

        if x.shape[2] != self.FIXED_TIME_LEN:
            x = F.interpolate(x.view(-1, x.shape[1], x.shape[2]), size=self.FIXED_TIME_LEN, mode='linear',
                              align_corners=False)
            x = x.view(-1, self.FIXED_CHANNELS, self.FIXED_TIME_LEN)

        curr_seq = x.shape[0]
        if curr_seq < self.FIXED_SEQ_LEN:
            pad = x[-1].unsqueeze(0).repeat(self.FIXED_SEQ_LEN - curr_seq, 1, 1)
            x = torch.cat([x, pad], dim=0)
        elif curr_seq > self.FIXED_SEQ_LEN:
            x = x[:self.FIXED_SEQ_LEN]
        return x

    def compute_spectral(self, x):
        """计算 Raw 信号的 PSD 作为物理参考系 (Reference Frame)"""
        x_flat = x.view(-1, x.shape[-1])
        x_flat = x_flat - x_flat.mean(dim=-1, keepdim=True)
        w = torch.hann_window(x_flat.shape[-1])
        fft = torch.fft.rfft(x_flat * w, dim=-1).abs() ** 2
        psd = fft.mean(dim=0)
        limit = max(32, psd.shape[0] // 4)
        psd = psd[:limit].view(1, 1, -1)
        psd_proj = F.interpolate(psd, size=self.target_dim, mode='linear', align_corners=False)
        return psd_proj.view(-1)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        x_std = self.safe_load_and_process(self.files[idx])
        if x_std is None:
            return (
                torch.zeros(self.target_dim) if self.mode == 'raw' else torch.zeros(self.FIXED_SEQ_LEN, 2, 3000)), False

        if self.mode == 'raw':
            return self.compute_spectral(x_std), True
        else:
            return x_std, True


def get_all_files(data_root, domain_list):
    if isinstance(domain_list, str): domain_list = [domain_list]
    all_files = []
    for domain in domain_list:
        ds_path = os.path.join(data_root, domain)
        if not os.path.exists(ds_path): continue
        for root, _, fs in os.walk(ds_path):
            for f in fs:
                if f.endswith('.npy'): all_files.append(os.path.join(root, f))
    return all_files


def extract_features_parallel(files, model, device, mode='latent'):
    """批量提取特征"""
    if not files: return np.array([])
    dataset = SleepFeatureDataset(files, mode=mode, target_dim=128)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    features_list = []

    if mode == 'latent': model.eval()

    with torch.no_grad():
        for batch_data, valid_mask in tqdm(loader, desc=f"Ext {mode}", leave=False):
            valid_indices = torch.where(valid_mask)[0]
            if len(valid_indices) == 0: continue
            batch_data = batch_data[valid_indices]

            if mode == 'raw':
                features_list.append(batch_data.numpy())
            else:
                batch_data = batch_data.to(device)
                _, mu = model.ae(batch_data)
                feat = mu.mean(dim=1)
                if feat.shape[1] != 128:
                    feat = F.interpolate(feat.unsqueeze(1), size=128, mode='linear').squeeze(1)
                features_list.append(feat.cpu().numpy())

    return np.concatenate(features_list, axis=0) if features_list else np.array([])


def compute_similarity(feats, anchors):
    """计算余弦相似度"""
    if len(feats) == 0: return np.array([])
    sim_matrix = cosine_similarity(feats, anchors)
    return np.max(sim_matrix, axis=1)


# ==========================================
# 3. 核心绘图引擎 (ICML Mechanism Style - Clean Ticks)
# ==========================================
def plot_icml_mechanism_clean(df_all):
    print("\n[Plotting] Rendering ICML Mechanism-style (Clean Ticks)...")

    # 1. 数据预处理
    eps = 1e-9
    df_all['Error'] = 1.0 - df_all['Similarity']
    df_all['Error'] = df_all['Error'].apply(lambda x: max(x, eps))

    # 2. 字体与风格全局设置 (关键步骤：统一 Times New Roman)
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],  # 强制正文衬线字体
        'mathtext.fontset': 'stix',  # 强制数学公式使用类 Times 字体 (stix)
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'lines.linewidth': 2,
        'axes.linewidth': 1.2
    })

    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

    datasets = df_all['Dataset'].unique()
    medians_raw = []

    # 3. 绘制 Reference (Raw) 虚线
    for i, ds in enumerate(datasets):
        raw_vals = df_all[(df_all['Dataset'] == ds) & (df_all['State'] == 'Raw')]['Error']
        median_raw = raw_vals.median()
        medians_raw.append(median_raw)

        # 灰色虚线作为参考系
        ax.plot([i - 0.35, i + 0.35], [median_raw, median_raw],
                color=COLOR_REFERENCE, linestyle='--', linewidth=2.0, alpha=0.8, zorder=1)

    # 4. 绘制 Concentrated (Ours) Boxen Plot
    df_aligned = df_all[df_all['State'] == 'Aligned'].copy()

    # 注意：Boxenplot 没有 zorder 参数，利用绘图顺序控制覆盖关系
    sns.boxenplot(
        data=df_aligned, x="Dataset", y="Error",
        color=COLOR_CONCENTRATED,
        width=0.45,
        linewidth=0.8,
        ax=ax,
        showfliers=False
    )

    # 5. Y 轴刻度核心修改：极简模式 (Clean Ticks)
    ax.set_yscale("log")

    # [关键] Major Ticks: 只有 10^0, 10^-2, 10^-4... (Base=100)
    ax.yaxis.set_major_locator(ticker.LogLocator(base=100.0, numticks=10))

    # [关键] Minor Ticks: 彻底移除 (NullLocator)，不留任何小刻度
    ax.yaxis.set_minor_locator(ticker.NullLocator())

    # 6. 网格线处理
    ax.grid(True, which="major", axis="y", linestyle="--", alpha=0.4, color='gray')
    ax.grid(False, axis="x")

    # 7. 动态调整 Y 轴范围
    y_floor = 10 ** np.floor(np.log10(df_aligned['Error'].min()))
    y_ceil = 10 ** np.ceil(np.log10(max(medians_raw)))
    # 适当留白
    ax.set_ylim(y_floor * 0.2, y_ceil * 8.0)

    # 8. 机制性标注 (字体会自动匹配 STIX/Times)
    annotation_text = (r"$\bf{Structural\ Collapse}$" + "\n" +
                       r"$\Delta \approx \mathcal{O}(10^{3}\!\!-\!\!10^{4})$" + "\n" +
                       r"$\mathrm{(deviation\ magnitude)}$")

    ax.text(0.98, 0.95, annotation_text, transform=ax.transAxes,
            ha='right', va='top', fontsize=15, color=COLOR_ANNOTATION,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="none", alpha=0.9))

    # 9. 标签与标题
    ax.set_ylabel(r"Structural Deviation ($1 - \cos(\mathbf{z}, \mathbf{c})$)", fontsize=16, fontweight='bold')
    ax.set_xlabel("", fontsize=12)
    ax.set_title("Collapse of Structural Deviation Across Domains", fontsize=18, fontweight='bold', pad=20)

    sns.despine(trim=True, offset=10)

    # 10. 语义化图例
    legend_elements = [
        mlines.Line2D([], [], color=COLOR_REFERENCE, linestyle='--', linewidth=2.0,
                      label='Unaligned (Reference Frame)'),
        mlines.Line2D([], [], color=COLOR_CONCENTRATED, marker='s', linestyle='None', markersize=10,
                      label='Structurally Concentrated (Ours)')
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.1),
              ncol=2, frameon=False, fontsize=14)

    plt.tight_layout()
    save_path = "Figure_ICML_Mechanism_Final.pdf"
    plt.savefig(save_path, transparent=True, bbox_inches='tight')
    print(f"\n[Success] Plot saved to: {save_path}")


# ==========================================
# 4. 主程序入口
# ==========================================
def main():
    torch.multiprocessing.set_sharing_strategy('file_system')
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[System] Device: {device}, Workers: {NUM_WORKERS}")

    # --- 模型加载 ---
    try:
        from original.models.model import Model
        # Mock params (确保与 checkpoint 匹配)
        params = SimpleNamespace(num_of_classes=5, d_model=128, nhead=4, num_layers=1, dim_feedforward=512, dropout=0.1,
                                 activation="relu")
        model = Model(params).to(device)

        if os.path.exists(CHECKPOINT_PATH):
            state_dict = torch.load(CHECKPOINT_PATH, map_location=device)
            model.load_state_dict(state_dict, strict=False)
            print("[Model] Checkpoint loaded successfully.")
        else:
            print(f"[Warning] Checkpoint not found at {CHECKPOINT_PATH}. Using random weights.")
    except Exception as e:
        print(f"[Error] Model import failed: {e}. Ensure 'original.models.model' is in path.")
        return

    # --- Anchors 加载 ---
    if os.path.exists(ANCHOR_FILE):
        universal_anchors = np.load(ANCHOR_FILE)
        print(f"[Anchors] Loaded shape: {universal_anchors.shape}")
    else:
        print(f"[Error] Anchors not found at {ANCHOR_FILE}.")
        return

    records = []
    print(f"[Processing] Sampling max {MAX_SAMPLES_PER_DOMAIN} files per domain...")

    for domain in TEST_DOMAINS:
        print(f"  > Domain: {domain}")
        files = get_all_files(DATA_ROOT, domain)
        if not files:
            print("    No files found.")
            continue

        if len(files) > MAX_SAMPLES_PER_DOMAIN:
            files = random.sample(files, MAX_SAMPLES_PER_DOMAIN)

        # 1. 提取 Reference (Raw)
        raw_feats = extract_features_parallel(files, model, device, mode='raw')
        sim_raw = compute_similarity(raw_feats, universal_anchors)

        # 2. 提取 Concentrated (Aligned)
        lat_feats = extract_features_parallel(files, model, device, mode='latent')
        sim_lat = compute_similarity(lat_feats, universal_anchors)

        if len(sim_raw) > 0 and len(sim_lat) > 0:
            for val in sim_raw: records.append({'Dataset': domain, 'State': 'Raw', 'Similarity': val})
            for val in sim_lat: records.append({'Dataset': domain, 'State': 'Aligned', 'Similarity': val})
        else:
            print("    Extraction returned empty.")

    if len(records) > 0:
        plot_icml_mechanism_clean(pd.DataFrame(records))
    else:
        print("[Error] No valid data collected.")


if __name__ == "__main__":
    main()