#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
スクリプトの概要:
このスクリプトは
JADESのカタログを使って
スペクトル（1d, 2d）を描画するものです。

使用方法:
    JADES_spectra_draw.py [オプション]

著者: A. M.
作成日: 2026-02-02

参考文献:
    - PEP 8: https://peps.python.org/pep-0008/
    - PEP 257 (Docstring規約): https://peps.python.org/pep-0257/
    - Python公式ドキュメント: https://docs.python.org/ja/3/
    - Elbaz et al. (2007)
    - Chen et al. (2016)
"""


# == 必要なパッケージのインストール == #
import re
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import ZScaleInterval
from matplotlib.gridspec import GridSpec

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
    "xtick.major.size": 20,          # 長さ
    "ytick.major.size": 20,
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

# =========================
# 1. CSV を読む
# =========================
current_dir = os.getcwd()
csv_path = os.path.join(current_dir, "results/csv/JADES_ne_candidates.csv")
df = pd.read_csv(csv_path)

# df.iloc[0]: 「0行目（最初の1天体）」を取り出す
nir_id = df.iloc[0]["NIRSpec_ID"] 
nir_id_str = f"{int(nir_id):08d}"
z_spec = df.iloc[0]["z_Spec"]
z_spec_str = f"{float(z_spec):.3f}"

# =========================
# 2. スペクトルを探す
# =========================
base = os.path.join(current_dir, "results/JADES/JADES_spectra_G395H/HLSP")

x1d = glob.glob(f"{base}/**/*{nir_id_str}*_x1d.fits", recursive=True)[0]
s2d = glob.glob(f"{base}/**/*{nir_id_str}*_s2d.fits", recursive=True)[0]

print("1D:", x1d)
print("2D:", s2d)

# =========================
# 3. 1D スペクトル
# =========================
with fits.open(x1d) as hdul:
    tab = hdul["EXTRACT1D"].data
    wave_1d = tab["WAVELENGTH"]
    flux_1d = tab["FLUX"]
    flux_1d = flux_1d * 1e19 # 縦軸のスケーリング
    err_1d  = tab["FLUX_ERR"]
    err_1d = err_1d * 1e19   # 縦軸のスケーリング

# =========================
# 4. 2D スペクトル
# =========================
with fits.open(s2d) as hdul:
    flux_2d = hdul["FLUX"].data
    wave_2d = hdul["WAVELENGTH"].data

# ZScale
zscale = ZScaleInterval()
vmin, vmax = zscale.get_limits(flux_2d[np.isfinite(flux_2d)])

# =========================
# 5. プロット（GridSpec）
# =========================
fig = plt.figure(figsize=(12, 6))
gs = GridSpec(
    2, 1,
    height_ratios=[1, 5],
    hspace=0.0
)

ax2d = fig.add_subplot(gs[0])
ax1d = fig.add_subplot(gs[1], sharex=ax2d)

# ---- 2D ----
im = ax2d.imshow(
    flux_2d,
    origin="lower",
    aspect="auto",
    cmap="plasma",
    vmin=vmin,
    vmax=vmax,
    extent=[wave_2d.min(), wave_2d.max(), 0, flux_2d.shape[0]]
)

ax2d.set_ylabel("")
ax2d.set_title(f"JADES NIRSpec G395H  |  ID {nir_id_str} | z_spec = {z_spec_str}")
ax2d.tick_params(axis='both', which='both',
               bottom=False, top=False, left=False, right=False,
               labelbottom=False, labeltop=False,
               labelleft=False, labelright=False)

# === 枠線 (spines) の設定 ===
# 線の太さ・色・表示非表示などを個別に制御
for spine in ax2d.spines.values():
    spine.set_linewidth(2)       # 枠線の太さ
    spine.set_color("black")     # 枠線の色

# ---- 1D ----
ax1d.step(
    wave_1d,
    flux_1d,
    where="mid",
    color='black',
    lw=1.0,
)

ax1d.fill_between(
    wave_1d,
    flux_1d - err_1d,
    flux_1d + err_1d,
    step="mid",
    color="gray",
    alpha=0.4,
)

ax1d.set_xlabel(r'$\lambda$ (Å)')
ax1d.set_ylabel(r'F$_{\lambda}$ ($10^{-19}$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$)')
# === 枠線 (spines) の設定 ===
# 線の太さ・色・表示非表示などを個別に制御
for spine in ax1d.spines.values():
    spine.set_linewidth(2)       # 枠線の太さ
    spine.set_color("black")     # 枠線の色

save_path = os.path.join(current_dir, f"results/figure/JADES/JADES_NIRSpec_G395H_ID{nir_id_str}")
plt.savefig(save_path)
print(f"Saved as {save_path}")
plt.show()
