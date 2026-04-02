import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.interpolate import make_interp_spline
import seaborn as sns

# 设置字体，模拟论文排版
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['mathtext.fontset'] = 'cm'  # 使用 LaTeX 风格数学字体


# ==========================================
# 1. 模拟数据生成 (保持逻辑不变)
# ==========================================
def generate_structural_data():
    x = np.linspace(0, 10, 100)

    # Dataset A (Source): 基准结构
    y_a = 1.0 / (x + 1) + 0.6 * np.exp(-((x - 3.0) ** 2) / 0.5)

    # Dataset B (Target): 结构扭曲
    # 为了绘图，我们在 A 的坐标系下画 B 的数据，但呈现出“错位”的形态
    y_b_plotted_on_a = 0.8 / (x + 1) + 0.5 * np.exp(-((x - 4.2) ** 2) / 0.8)

    return x, y_a, y_b_plotted_on_a


# ==========================================
# 2. 绘图逻辑 (防御增强版)
# ==========================================
def draw_figure_v3():
    x, y_a, y_b = generate_structural_data()

    # 平滑处理
    x_smooth = np.linspace(x.min(), x.max(), 300)
    spl_a = make_interp_spline(x, y_a, k=3)
    spl_b = make_interp_spline(x, y_b, k=3)
    y_a_smooth = spl_a(x_smooth)
    y_b_smooth = spl_b(x_smooth)

    # 归一化 (但 Label 不叫 Normalization)
    y_a_smooth /= y_a_smooth.max()
    y_b_smooth /= y_b_smooth.max()

    fig, ax = plt.subplots(figsize=(10, 6.5))  # 稍微加高一点给注脚留位置

    # --- 绘制主曲线 ---
    color_a = '#1f77b4'  # Source Blue
    color_b = '#d62728'  # Target Red

    ax.plot(x_smooth, y_a_smooth, color=color_a, linewidth=3, label=r'Observation $\mathcal{O}_A$ (Source)')
    ax.plot(x_smooth, y_b_smooth, color=color_b, linewidth=3, linestyle='--',
            label=r'Observation $\mathcal{O}_B$ (Target)')

    ax.fill_between(x_smooth, y_a_smooth, alpha=0.1, color=color_a)
    ax.fill_between(x_smooth, y_b_smooth, alpha=0.05, color=color_b)

    # --- 关键修改 1: 双坐标轴 + 符号隔离 ($f$ vs $f'$) ---

    # 隐藏原生 X 轴
    ax.spines['bottom'].set_visible(False)
    ax.xaxis.set_visible(False)

    # >>> Axis A (Source) <<<
    y_axis_pos = -0.05
    ax.annotate('', xy=(0, y_axis_pos), xytext=(10, y_axis_pos), arrowprops=dict(arrowstyle='-', color='black', lw=1.5))
    for i in range(1, 10, 2):
        # 使用 f 符号
        ax.text(i, y_axis_pos - 0.08, f"$f_{{{i}}}$", ha='center', color=color_a, fontsize=12, fontweight='bold')
        ax.plot([i, i], [y_axis_pos, y_axis_pos + 0.02], color=color_a, lw=2)
    ax.text(10.5, y_axis_pos - 0.05, r"Basis $\mathcal{B}_A$", ha='left', va='center', fontsize=11, fontstyle='italic',
            color=color_a)

    # >>> Axis B (Target - Misaligned) <<<
    y_axis_pos_b = -0.18
    ax.annotate('', xy=(0, y_axis_pos_b), xytext=(10, y_axis_pos_b),
                arrowprops=dict(arrowstyle='-', color='black', lw=1.5))

    # 模拟非线性扭曲的刻度位置
    shifts = [1.2, 3.5, 5.8, 8.2, 10.0]
    for i, pos in enumerate(shifts):
        idx = i * 2 + 1
        # 【关键修改】使用 f' (prime) 符号，视觉上宣告“这不是同一个东西”
        ax.text(pos, y_axis_pos_b - 0.08, f"$f'_{{{idx}}}$", ha='center', color=color_b, fontsize=12, fontweight='bold')
        ax.plot([pos, pos], [y_axis_pos_b, y_axis_pos_b + 0.02], color=color_b, lw=2)
    ax.text(10.5, y_axis_pos_b - 0.05, r"Basis $\mathcal{B}_B$ (Warped)", ha='left', va='center', fontsize=11,
            fontstyle='italic', color=color_b)

    # --- 关键修改 2: 显式注脚 (Non-equivalence Note) ---
    # 在图的底部添加防御性声明
    note_text = r"$\bf{Note:}$ Indices $f$ and $f'$ denote non-equivalent structural components."
    ax.text(5, y_axis_pos_b - 0.15, note_text, ha='center', fontsize=10, color='#555555',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2))

    # --- 冲突示意 (保持原样) ---
    peak_a = np.argmax(y_a_smooth)
    peak_b = np.argmax(y_b_smooth)
    x_peak_a, x_peak_b = x_smooth[peak_a], x_smooth[peak_b]

    ax.annotate('', xy=(x_peak_b, y_b_smooth[peak_b] + 0.05), xytext=(x_peak_a, y_b_smooth[peak_b] + 0.05),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1.5, ls='--'))
    ax.text((x_peak_a + x_peak_b) / 2, y_b_smooth[peak_b] + 0.08, "Structural Conflict",
            ha='center', fontsize=10, fontweight='bold', color='#333333', backgroundcolor='white')

    # --- 标题与标签 (术语更新) ---
    ax.set_title("Structural Reference Frame Misalignment\n(Same Latent State, Distinct Spectral Topologies)",
                 fontsize=14, fontweight='bold', pad=25)

    # 【关键修改】换掉 "Normalized Spectral Density"
    ax.set_ylabel("Relative Spectral Magnitude", fontsize=12)

    sns.despine(ax=ax, bottom=True, trim=True)
    ax.set_xlim(0, 11)  # 留多一点空间给右边的 Label
    ax.set_ylim(-0.35, 1.15)  # 底部留更多空间给双轴和注脚
    ax.set_yticks([])

    plt.tight_layout()
    plt.savefig('figure2_structural_misalignment_v3.pdf', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    draw_figure_v3()