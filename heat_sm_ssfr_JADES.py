#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
スクリプトの概要:
横軸M*, 縦軸sSFRのヒートマップを作成します。

使用方法:
    heat_sm_ssfr_JADES.py [オプション]

著者: A. M.
作成日: 2026-06-22

参考文献:
    - PEP 8:                  https://peps.python.org/pep-0008/
    - PEP 257 (Docstring規約): https://peps.python.org/pep-0257/
    - Python公式ドキュメント:    https://docs.python.org/ja/3/
"""

from astropy.io import fits
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d
import os
from astropy.table import Table


# -----------------------
# 軸の設定
# -----------------------
# 軸の設定
plt.rcParams.update({
    # --- 図全体 ---
    "figure.figsize": (12, 6),       # 図サイズ
    "font.size": 20,                 # 全体フォントサイズ
    "axes.labelsize": 24,            # 軸ラベルのサイズ
    "axes.titlesize": 20,            # タイトルのサイズ
    "axes.grid": False,              # グリッドOFF

    # --- 目盛り設定 (ticks) ---
    "xtick.direction": "in",         # x軸目盛りの向き
    "ytick.direction": "in",         # y軸目盛りの向き
    "xtick.top": True,               # 上にも目盛り
    "ytick.right": True,             # 右にも目盛り

    # 主目盛り（major ticks）
    "xtick.major.size": 12,          # 長さ
    "ytick.major.size": 12,
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
    "xtick.labelsize": 20,           # x軸ラベルサイズ
    "ytick.labelsize": 20,           # y軸ラベルサイズ

    # --- フォント ---
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
})


# ========= CSV読み込み =========
csv_path = "results/JADES/JADES_DR3/data_from_Nishigaki/jades_info_with_HA_plus_logSFR_with_Reff_sii_gn_gs_fit.csv"
df = pd.read_csv(csv_path)

# ========= 必要列 =========
logM = df["logM"].values
logSFR = df["log10_SFR_hb"].values
logsSFR = logSFR - logM
df["logsSFR"] = logsSFR



ratio = df["ratio"].values

# ========= 品質マスク =========
mask = (
    np.isfinite(logM) &
    np.isfinite(logSFR) &
    np.isfinite(ratio) &
    (ratio > 0) 
)

# ========= ビニング =========
xbins = np.arange(8, 12.1, 0.1)
ybins = np.arange(-14, -6.9, 0.1)

ratio_map, xedge, yedge, _ = binned_statistic_2d(
    logM[mask],
    logsSFR[mask],
    ratio[mask],
    statistic="median",
    bins=[xbins, ybins]
)

# ========= 描画 =========
fig, ax = plt.subplots(figsize=(8,6))

# vmin = np.nanpercentile(ratio_map, 5)
# vmax = np.nanpercentile(ratio_map, 95)

pcm = ax.pcolormesh(
    xedge,
    yedge,
    ratio_map.T,
    shading="auto",
    cmap="viridis",
    vmin=1.0,
    vmax=1.5,
)

plt.colorbar(pcm, ax=ax)
ax.set_xlabel(r'$\log M_*$')
plt.ylabel(r'$\log sSFR$')

for spine in ax.spines.values():
    spine.set_linewidth(2)

plt.tight_layout()

# 保存
fig_dir = "results/JADES/figure"
os.makedirs(fig_dir, exist_ok=True)

save_path = os.path.join(fig_dir, "heat_sm_ssfr_sii_ratio_jades.png")
plt.savefig(save_path)

print(f"Saved heatmap to: {save_path}")
plt.show()