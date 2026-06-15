#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
スクリプトの概要:
横軸M*, 縦軸SFRのヒートマップを作成します。

使用方法:
    heat_sm_sfr.py [オプション]

著者: A. M.
作成日: 2026-06-14

参考文献:
    - PEP 8:                  https://peps.python.org/pep-0008/
    - PEP 257 (Docstring規約): https://peps.python.org/pep-0257/
    - Python公式ドキュメント:    https://docs.python.org/ja/3/
"""

# 1. FITS読込に必要なパッケージのインストール
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


current_dir = os.getcwd()

fits_path = os.path.join(
    current_dir,
    "results/fits/mpajhu_dr7_v5_2_merged_zlt0.2_Lgt1e+39_radius.fits"
)

with fits.open(fits_path) as hdul:

    data = hdul[1].data

tab = Table.read(fits_path, hdu=1)
df = tab.to_pandas()


# 2. 必要な列を取得
UNIT_FLUX = 1e-17

F6716 = df["SII_6717_FLUX"].values * UNIT_FLUX
F6731 = df["SII_6731_FLUX"].values * UNIT_FLUX

err6716 = df["SII_6717_FLUX_ERR"].values * UNIT_FLUX
err6731 = df["SII_6731_FLUX_ERR"].values * UNIT_FLUX

logM = df["sm_MEDIAN"].values
logSFR = df["sfr_MEDIAN"].values

# 3. ratio作成
ratio = F6716 / F6731

# 4. 品質カット
sn6716 = F6716 / err6716
sn6731 = F6731 / err6731

mask = (
    np.isfinite(logM)
    &
    np.isfinite(logSFR)
    &
    np.isfinite(F6716)
    &
    np.isfinite(F6731)
    &
    np.isfinite(ratio)
    &
    (F6716 > 0)
    &
    (F6731 > 0)
    &
    (sn6716 > 3)
    &
    (sn6731 > 3)
)

# 5. ratioヒートマップ
xbins = np.arange(8.0, 11.6, 0.01)
ybins = np.arange(3.5, 4.5, 0.01)

ratio_map, xedge, yedge, _ = (
    binned_statistic_2d(
        logM[mask],
        logSFR[mask],
        ratio[mask],
        statistic="mean",
        bins=[xbins, ybins]
    )
)

fig, ax = plt.subplots(figsize=(8,6))
fig.subplots_adjust(left=0.15, right=0.85, bottom=0.15, top=0.85)

plt.pcolormesh(
    xedge,
    yedge,
    ratio_map.T,
    shading="auto",
    cmap="viridis"
)

plt.colorbar()

plt.xlabel(r'$\log M_*$')
plt.ylabel(r'$\log SFR$')

for spine in ax.spines.values():
    spine.set_linewidth(2)
plt.tight_layout()
# 保存
fig_dir = os.path.join(current_dir, "results/figure")
os.makedirs(fig_dir, exist_ok=True)
save_path_heat = os.path.join(
    fig_dir,
    "heat_sm_sfr_sii_ratio_sdss.png"
)

plt.savefig(save_path_heat)
print(f"Saved heatmap to: {save_path_heat}")
plt.show()


# 6. 各セルの銀河数も確認
count_map, _, _, _ = (
    binned_statistic_2d(
        logM[mask],
        logSFR[mask],
        ratio[mask],
        statistic="count",
        bins=[xbins, ybins]
    )
)

fig, ax = plt.subplots(figsize=(8,6))
fig.subplots_adjust(left=0.15, right=0.85, bottom=0.15, top=0.85)

plt.pcolormesh(
    xedge,
    yedge,
    count_map.T,
    shading="auto",
    cmap="viridis"
)

plt.colorbar()

plt.xlabel(r'$\log M_*$')
plt.ylabel(r'$\log SFR$')

for spine in ax.spines.values():
    spine.set_linewidth(2)
plt.tight_layout()

save_path_number = os.path.join(
    fig_dir,
    "heat_sm_sfr_galaxy_number_sdss.png"
)

plt.savefig(save_path_number)
print(f"Saved galaxy count heatmap to: {save_path_number}")
plt.show()