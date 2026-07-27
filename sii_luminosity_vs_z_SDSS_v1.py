#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
スクリプトの概要:
SIIのLumiosityとzの関係を描画します。
これはSDSS用です。

v0との変更点
・SIIの暗い側のみをサンプル選択に含める
・フラックス一定の曲線とSDSSの観測装置の特性をつなげる
・フラックス一定の曲線とLuminosity一定の曲線の交点をサンプル選択に使用
・使用するファイル(merged)にReの情報をあらかじめ入れておく

使用方法:
    sii_luminoisity_vs_z_SDSS_v1.py [オプション]

著者: A. M.
作成日: 2026-07-27

参考文献:
    - フラックス一定の曲線とSDSSの観測装置の特性
"""


# === 必要なモジュールのインポート ===
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from astropy.io import fits
import astropy.units as u
from astropy.cosmology import Planck18 as cosmo
from astropy.table import Table
from astropy.table import Column
from matplotlib.colors import TwoSlopeNorm
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize


# 軸の設定
plt.rcParams.update({
    # --- 図全体 ---
    "figure.figsize": (12, 6),       # 図サイズ
    "font.size": 28,                 # 全体フォントサイズ
    "axes.labelsize": 28,            # 軸ラベルのサイズ
    "axes.titlesize": 28,            # タイトルのサイズ
    "axes.grid": False,              # グリッドOFF

    # --- 目盛り設定 (ticks) ---
    "xtick.direction": "in",         # x軸目盛りの向き
    "ytick.direction": "in",         # y軸目盛りの向き
    "xtick.top": True,               # 上にも目盛り
    "ytick.right": True,             # 右にも目盛り

    # 主目盛り（major ticks）
    "xtick.major.size": 28,          # 長さ
    "ytick.major.size": 28,
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
    "xtick.labelsize": 24,           # x軸ラベルサイズ
    "ytick.labelsize": 24,           # y軸ラベルサイズ

    # --- フォント ---
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
})


# =====================================
# 基本量の設定
# =====================================
# ここを変更する
Z_MAX = 0.20
L_MIN = 1e39          # erg s^-1
UNIT_FLUX = 1e-17     # MPA-JHU flux unit

# =====================================
# FITS 読み込み
# =====================================
current_dir = os.getcwd()
# 先にfitsファイルにReの情報を入れておく
fits_path = os.path.join(current_dir, "results/fits/mpajhu_dr7_v5_2_merged.fits")

t = Table.read(fits_path, format="fits")
df = t.to_pandas()

# =====================================
# データ抽出
# =====================================
z = df["Z"].values

F6716 = df["SII_6717_FLUX"].values * UNIT_FLUX
err6716 = df["SII_6717_FLUX_ERR"].values * UNIT_FLUX

F6731 = df["SII_6731_FLUX"].values * UNIT_FLUX
err6731 = df["SII_6731_FLUX_ERR"].values * UNIT_FLUX

# SII輝線のSN比を設定する
sn6716 = F6716 / err6716
sn6731 = F6731 / err6731

# =====================================
# Luminosity 計算
# =====================================
d_L = cosmo.luminosity_distance(z).to(u.cm).value
L6716 = 4 * np.pi * d_L**2 * F6716
L6731 = 4 * np.pi * d_L**2 * F6731

# =====================================
# 数値健全性マスク
# =====================================
# ここも変更する
mask_finite = (
    np.isfinite(z) &
    np.isfinite(L6716) &
    np.isfinite(L6731)
)

# =====================================
# 完全サンプル条件
# =====================================
# ここを変更する
mask_complete = (
    mask_finite &
    (z < Z_MAX) &
    (L6716 > L_MIN) &
    (L6731 > L_MIN)   # ★ 追加
)

print(f"[INFO] 抽出件数: {mask_complete.sum()} / {len(mask_complete)}")

# =====================================
# 図の描画
# =====================================
# 6731の方
fig, ax = plt.subplots(figsize=(12, 6))
fig.subplots_adjust(left=0.10, right=0.95, bottom=0.15, top=0.95)

# 全体
ax.scatter(z, L6731, s=6, alpha=0.3, color="gray")

# 完全サンプル
ax.scatter(
    z[mask_complete],
    L6731[mask_complete],
    s=2,
    alpha=1,
    color="C1",
)

# カット線
ax.axvline(Z_MAX, color="k", linestyle="-", linewidth=2.0)
ax.axhline(L_MIN, color="k", linestyle="-", linewidth=2.0)

#  L([S II] 6731) の一定フラックス線
z_grid = np.linspace(0.0, 0.4, 200)
d_L_grid = cosmo.luminosity_distance(z_grid).to(u.cm).value 
L_const = 4 * np.pi * d_L_grid**2 * 1e-17
ax.plot(z_grid, L_const, color="black", linestyle="-", linewidth=2.0)

# 軸設定
ax.set_yscale("log")
ax.set_xlim(0, 0.4)
ax.set_ylim(1e37, 1e42)

ax.set_xlabel("z")
ax.set_ylabel(r"L([S II] 6731) [erg s$^{-1}$]")


# 枠線強調
for spine in ax.spines.values():
    spine.set_linewidth(2)

# 保存
fig_dir = os.path.join(current_dir, "results/figure")
os.makedirs(fig_dir, exist_ok=True)

save_path = os.path.join(fig_dir, "sii6731_luminosity_vs_z_v1.png")
plt.savefig(save_path, dpi=200, bbox_inches="tight")
plt.show()

print(f"[DONE] 図を保存: {save_path}")


# =====================================
# FITS 抽出
# =====================================
t_sel = t[mask_complete]

out_dir = os.path.join(current_dir, "results/fits")
os.makedirs(out_dir, exist_ok=True)

out_path = os.path.join(
    out_dir,
    f"mpajhu_dr7_v5_2_merged_zlt{Z_MAX}_Lgt{L_MIN:.0e}.fits"
)

t_sel.write(out_path, format="fits", overwrite=True)

print(f"[DONE] 書き出し完了: {out_path}")
# [INFO] 抽出件数: 526970 / 927552


# =====================================
# 抽出された銀河の赤方偏位のヒストグラム描画
# =====================================