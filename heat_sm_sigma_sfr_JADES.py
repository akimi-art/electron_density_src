#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
スクリプトの概要:
横軸M*, 縦軸ΣSFRのヒートマップを作成します。

使用方法:
    heat_sm_sigma_sfr_JADES.py [オプション]

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
from astropy.cosmology import Planck18 as cosmo


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


# Re（arcsec → kpc）
z = df["z_spec"].values

arcsec_to_kpc = cosmo.kpc_proper_per_arcmin(z).value / 60.0
Re_kpc = df["ReffOpt"].values * arcsec_to_kpc

df["Re_kpc"] = Re_kpc
logRe = np.log10(Re_kpc)
df["logRe"] = logRe

# ΣSFRの計算
logSigma = (
    logSFR
    -
    np.log10(2*np.pi*Re_kpc**2)
)

df["logSigma_SFR"] = logSigma



ratio = df["ratio"].values

# ========= 品質マスク =========
mask = (
    np.isfinite(logM) &
    np.isfinite(logSFR) &
    np.isfinite(z) &
    np.isfinite(ratio) &
    (Re_kpc > 0) &
    (ratio > 0) 
)

# ========= ビニング =========
xbins = np.arange(7.0, 12.1, 0.1)
ybins = np.arange(-3.0, 3.1, 0.1)

ratio_map, xedge, yedge, _ = binned_statistic_2d(
    logM[mask],
    logSFR[mask],
    ratio[mask],
    statistic="median",
    bins=[xbins, ybins]
)

# ========= 描画 =========
fig, ax = plt.subplots(figsize=(8,6))

vmin = np.nanpercentile(ratio_map, 5)
vmax = np.nanpercentile(ratio_map, 95)

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
ax.set_ylabel(r'$\log(\Sigma_{\rm SFR})\ [{\rm M_\odot\ yr^{-1}\ kpc^{-2}}]$')

for spine in ax.spines.values():
    spine.set_linewidth(2)

plt.tight_layout()

# 保存
fig_dir = "results/JADES/figure"
os.makedirs(fig_dir, exist_ok=True)

save_path = os.path.join(fig_dir, "heat_sm_sigma_sfr_sii_ratio_jades.png")
plt.savefig(save_path)

print(f"Saved heatmap to: {save_path}")
plt.show()