#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
スクリプトの概要:
このスクリプトは
JADESのカタログを使って
スペクトル（1d, 2d）を描画するものです。
同じディレクトリに入っている銀河を一挙に描画します。

使用方法:
    JADES_spectra_draw_v1.py [オプション]

著者: A. M.
作成日: 2026-02-08

参考文献:
    - PEP 8: https://peps.python.org/pep-0008/
    - PEP 257 (Docstring規約): https://peps.python.org/pep-0257/
    - Python公式ドキュメント: https://docs.python.org/ja/3/
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
    "font.size": 16,                 # 全体フォントサイズ
    "axes.labelsize": 16,            # 軸ラベルのサイズ
    "axes.titlesize": 16,            # タイトルのサイズ
    "axes.grid": False,              # グリッドOFF

    # --- 目盛り設定 (ticks) ---
    "xtick.direction": "in",         # x軸目盛りの向き
    "ytick.direction": "in",         # y軸目盛りの向き
    "xtick.top": False,               # 上にも目盛り
    "ytick.right": False,             # 右にも目盛り

    # 主目盛り（major ticks）
    "xtick.major.size": 16,          # 長さ
    "ytick.major.size": 16,
    "xtick.major.width": 2,          # 太さ
    "ytick.major.width": 2,

    # 補助目盛り（minor ticks）
    "xtick.minor.visible": False,     # 補助目盛りON
    "ytick.minor.visible": False,
    "xtick.minor.size": 8,           # 長さ
    "ytick.minor.size": 8,
    "xtick.minor.width": 1.5,        # 太さ
    "ytick.minor.width": 1.5,

    # --- 目盛りラベル ---
    "xtick.labelsize": 16,           # x軸ラベルサイズ
    "ytick.labelsize": 16,           # y軸ラベルサイズ

    # --- フォント ---
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
})


# =====================================================
# 0. 参照波長
# =====================================================
wavelength_o2_rest  = np.array([3727.092, 3729.875])
wavelength_s2_rest  = np.array([6716.440, 6730.820])
wavelength_o3_rest  = np.array([4960.295, 5008.240])

# =====================================================
# 1. カタログ読み込み
# =====================================================
current_dir = os.getcwd()
csv_path = os.path.join(current_dir, "results/csv/JADES_ne_candidates_dr4_oii.csv")
df = pd.read_csv(csv_path) 

# =====================================================
# 2. clear-prism ファイル取得
# =====================================================
base_dir = os.path.join(current_dir, "results/JADES/JADES_DR4/JADES_DR4_GOODS-N_G395H_OII")

x1d_files = glob.glob(
    os.path.join(base_dir, "**", "*f290lp-g395h*_x1d.fits"),
    recursive=True
)

x1d_files.sort()

print(f"Found {len(x1d_files)} f290lp-g395h spectra")

# =====================================================
# 3. レイアウト設定
# =====================================================
n_cols = 5
n_obj = len(x1d_files)
n_rows = int(np.ceil(n_obj / n_cols))

fig = plt.figure(figsize=(4*n_cols, 4*n_rows))
# fig = plt.figure(figsize=(12, 6))

outer_gs = GridSpec(n_rows, n_cols, hspace=0.0, wspace=0.0)

# =====================================================
# 4. 各天体を描画
# =====================================================
for idx, x1d in enumerate(x1d_files):

    row = idx // n_cols
    col = idx % n_cols

    filename = os.path.basename(x1d)

    match = re.search(r"(\d{8})", filename)
    if match is None:
        print("IDが見つからない:", filename)
        continue

    nir_id = int(match.group(1))
    nir_id_str = f"{nir_id:08d}"

    # z取得
    z_match = df[df["NIRSpec_ID"] == nir_id]
    if len(z_match) == 0:
        continue
    z_spec = z_match.iloc[0]["z_Spec"]

    # 対応する2D探す
    s2d_list = glob.glob(x1d.replace("_x1d.fits", "_s2d.fits"))
    if len(s2d_list) == 0:
        continue
    s2d = s2d_list[0]

    # ---- スペクトル読み込み ----
    with fits.open(x1d) as hdul:
        tab = hdul["EXTRACT5PIX1D"].data
        wave_1d = tab["WAVELENGTH"] * 10000
        flux_1d = tab["FLUX"] * 1e19
        err_1d  = tab["FLUX_ERR"] * 1e19

    with fits.open(s2d) as hdul:
        flux_2d = hdul["FLUX"].data
        wave_2d = hdul["WAVELENGTH"].data * 10000

    # 観測波長
    o2_obs = wavelength_o2_rest * (1 + z_spec)
    s2_obs = wavelength_s2_rest * (1 + z_spec)
    o3_obs = wavelength_o3_rest * (1 + z_spec)

    # =================================================
    # 内側2段構造（2D上・1D下）
    # =================================================
    inner_gs = outer_gs[row, col].subgridspec(2, 1, height_ratios=[1, 5], hspace=0)

    ax2d = fig.add_subplot(inner_gs[0])
    ax1d = fig.add_subplot(inner_gs[1], sharex=ax2d)

    # ---- 2D ----
    zscale = ZScaleInterval()
    vmin, vmax = zscale.get_limits(flux_2d[np.isfinite(flux_2d)])

    ax2d.imshow(
        flux_2d,
        origin="lower",
        aspect="auto",
        cmap="plasma",
        vmin=vmin,
        vmax=vmax,
        extent=[wave_2d.min(), wave_2d.max(), 0, flux_2d.shape[0]]
    )

    ax2d.tick_params(axis='both', which='both',
               bottom=False, top=False, left=False, right=False,
               labelbottom=False, labeltop=False,
               labelleft=False, labelright=False)

    # ---- 1D ----
    ax1d.step(wave_1d, flux_1d, where="mid", color="black", lw=0.8)
    ax1d.fill_between(
        wave_1d,
        flux_1d - err_1d,
        flux_1d + err_1d,
        step="mid",
        color="gray",
        alpha=0.4,
        label=f"ID {nir_id_str}\nz={z_spec:.3f}",
    )
    ax1d.legend(fontsize=8)

    # 縦線
    for w in o2_obs:
        ax1d.axvline(w, color='blue', ls='--', lw=0.8)
    for w in s2_obs:
        ax1d.axvline(w, color='red', ls='-.', lw=0.8)
    for w in o3_obs:
        ax1d.axvline(w, color='green', ls=':', lw=0.8)

    if col != 0:
        ax1d.tick_params(labelleft=False)
    # ===============================
    # 軸ラベル整理
    # ===============================

    # --- 縦軸は全て非表示 ---
    ax1d.tick_params(axis="y", left=False, labelleft=False)

    # --- 横軸は最下段のみ表示 ---
    if row != n_rows - 1:
        ax1d.tick_params(axis="x", bottom=False, labelbottom=False)
    else:
        ax1d.set_xlabel("Observed wavelength (Å)", fontsize=14)


# =====================================================
# 5. 表示
# =====================================================
save_path = os.path.join(current_dir, f"results/figure/JADES/JADES_NIRSpec_spectra_dr4_oii.png")
plt.savefig(save_path)
print(f"Saved as {save_path}")
plt.show()
