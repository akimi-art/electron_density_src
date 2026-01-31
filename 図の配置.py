import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import rcParams

# ==================================================
# rcParams（グローバル設定）
# ==================================================
plt.rcParams.update({
    # --- 図全体 ---
    "figure.figsize": (12, 6),       # 図サイズ
    "font.size": 16,                 # 全体フォントサイズ
    "axes.labelsize": 18,            # 軸ラベルのサイズ
    "axes.titlesize": 18,            # タイトルのサイズ
    "axes.grid": False,              # グリッドOFF

    # --- 目盛り設定 (ticks) ---
    "xtick.direction": "in",         # x軸目盛りの向き
    "ytick.direction": "in",         # y軸目盛りの向き
    "xtick.top": True,               # 上にも目盛り
    "ytick.right": True,             # 右にも目盛り

    # 主目盛り（major ticks）
    "xtick.major.size": 16,          # 長さ
    "ytick.major.size": 16,
    "xtick.major.width": 2,          # 太さ
    "ytick.major.width": 2,

    # 補助目盛り（minor ticks）
    "xtick.minor.visible": True,     # 補助目盛りON
    "ytick.minor.visible": True,
    "xtick.minor.size": 8,           # 長さ
    "ytick.minor.size": 8,
    "xtick.minor.width": 1.5,        # 太さ
    "ytick.minor.width": 1.5,

    # --- 目盛りラベル ---
    "xtick.labelsize": 18,           # x軸ラベルサイズ
    "ytick.labelsize": 18,           # y軸ラベルサイズ
})

# ==================================================
# tick 制御用ヘルパー関数
# ==================================================
def configure_ticks(ax, *, show_left=False, show_bottom=False):
    """
    rcParams で minor / top / right tick が有効でも
    Axes ごとに完全制御するための関数
    """

    # まず全てオフ（major / minor / 全辺 / ラベル）
    ax.tick_params(
        which="both",
        top=False, bottom=False,
        left=False, right=False,
        labeltop=False, labelbottom=False,
        labelleft=False, labelright=False,
    )

    # 必要な辺だけオン
    if show_bottom:
        ax.tick_params(
            which="both",
            bottom=True,
            labelbottom=True
        )

    if show_left:
        ax.tick_params(
            which="both",
            left=True,
            labelleft=True
        )

# ==================================================
# ダミーデータ
# ==================================================
rng = np.random.default_rng(42)

# ==================================================
# Figure / GridSpec
# ==================================================
fig = plt.figure(figsize=(12, 6))

gs = GridSpec(
    2, 4,
    figure=fig,
    left=0.08, right=0.98,
    bottom=0.12, top=0.9,
    wspace=0, hspace=0
)

# ==================================================
# 8 個の散布図を配置
# ==================================================
for k in range(8):
    i = k // 4   # 行
    j = k % 4    # 列

    ax = fig.add_subplot(gs[i, j])

    x = rng.normal(size=100)
    y = rng.normal(size=100)
    ax.scatter(x, y, s=15, alpha=0.7)

    # ---- 軸表示ルール ----
    show_left   = (j == 0)
    show_bottom = (i == 1)

    configure_ticks(
        ax,
        show_left=show_left,
        show_bottom=show_bottom
    )

    # ラベルは外周のみ
    if show_left:
        ax.set_ylabel("y")

    if show_bottom:
        ax.set_xlabel("x")

    # ax.set_title(f"Panel {k+1}", fontsize=10)
    # === 枠線 (spines) の設定 ===
    # 線の太さ・色・表示非表示などを個別に制御
    for spine in ax.spines.values():
        spine.set_linewidth(2)       # 枠線の太さ
        spine.set_color("black")     # 枠線の色

# ==================================================
# 表示
# ==================================================
plt.show()
