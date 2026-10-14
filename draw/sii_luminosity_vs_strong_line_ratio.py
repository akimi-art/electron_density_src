#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
スクリプトの概要:
SIIのLuminosityとStrong line ratioの
関係性をプロットします。
Completeなサンプルだと思われるもののみを使用しています。

使用方法:
    sii_luminoisity_vs_sm_sfr_v1.py [オプション]

著者: A. M.
作成日: 2026-02-17

参考文献:
    - PEP 8:                  https://peps.python.org/pep-0008/
    - PEP 257 (Docstring規約): https://peps.python.org/pep-0257/
    - Python公式ドキュメント:    https://docs.python.org/ja/3/
    - Curti+17
"""


# === 必要なモジュール ===
import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u


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
    "xtick.top": False,               # 上にも目盛り
    "ytick.right": False,             # 右にも目盛り

    # 主目盛り（major ticks）
    "xtick.major.size": 20,          # 長さ
    "ytick.major.size": 20,
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
    "xtick.labelsize": 20,           # x軸ラベルサイズ
    "ytick.labelsize": 20,           # y軸ラベルサイズ

    # --- フォント ---
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
})

# === 入力ファイル（サブセット FITS） ===
current_dir = os.getcwd()
fits_path = os.path.join(
    current_dir,
    "results/fits/mpajhu_dr7_v5_2_merged_L6717_ge_4pi_dL2_1e-17_L6731_ge_4pi_dL2_1e-17_z0.00-0.40.fits"
)

# === 読み込み（列構造はそのまま） ===
t = Table.read(fits_path, format="fits")


# === 必要列を取り出し ===
# MPA-JHU: sm_MEDIAN, sfr_MEDIAN は log(M*/Msun), log(SFR/Msun/yr)
z = np.array(t["Z"], dtype=float)
F6717_raw = np.array(t["SII_6717_FLUX"], dtype=float)
F6731_raw = np.array(t["SII_6731_FLUX"], dtype=float)

# === 単位スケール（MPA-JHUは 1e-17 cgs）→ cgs へ ===
UNIT_FLUX = 1e-17
F6717 = F6717_raw * UNIT_FLUX
F6731 = F6731_raw * UNIT_FLUX

# === 光度計算（L = 4π d_L^2 F） ===
m_all = np.isfinite(z) & np.isfinite(F6717) & np.isfinite(F6731) 
dL = np.full_like(z, np.nan, dtype=float)
dL[m_all] = cosmo.luminosity_distance(z[m_all]).to(u.cm).value

L6716 = 4 * np.pi * dL**2 * F6717
L6731 = 4 * np.pi * dL**2 * F6731

# === strong line ratios ===
ratios = ["R2", "R3", "O32", "R23", "N2", "O3N2"]

ratio_data = {}
for r in ratios:
    ratio_data[r] = np.array(t[r], dtype=float)

# === L6717 有効マスク ===
m6717 = (
    np.isfinite(L6716) &
    (L6716 > 0)
)

# === 図設定 ===
xlim_default = (-3, 3)  # ← 後で変更可
ylim_default = (1e35, 1e42)

fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(2, 3, 
                      left=0.10, right=0.90, bottom=0.06, top=0.98,  # マージンを詰める
                      wspace=0.00, hspace=0.00)

axes = []
for i in range(2):
    for j in range(3):
        axes.append(fig.add_subplot(gs[i, j]))

for ax, r in zip(axes, ratios):
    m = m6717 & np.isfinite(ratio_data[r])
    ax.scatter(
        ratio_data[r][m],
        L6716[m],
        s=1,
        alpha=0.6,
        label=r
    )

    ax.text(
        0.05, 0.95,
        r,
        transform=ax.transAxes,
        fontsize=20,
        verticalalignment='top'
    )

    ax.set_yscale("log")
    ax.set_xlim(*xlim_default)
    ax.set_ylim(*ylim_default)

    # 枠線
    for spine in ax.spines.values():
        spine.set_linewidth(2)
        spine.set_color("black")

# ラベル（左列のみ y）
axes[0].set_ylabel("L([S II] 6717) [erg/s]")
axes[3].set_ylabel("L([S II] 6717) [erg/s]")

# # 下段のみ xラベル
# for ax, r in zip(axes[3:], ratios[3:]):
#     ax.set_xlabel(r)

# 上段 xラベル消去
for ax in axes[:3]:
    ax.tick_params(labelbottom=False)

# 右側 yラベル消去
for ax in (axes[1], axes[2], axes[4], axes[5]):
    ax.tick_params(labelleft=False)

savepath = os.path.join(current_dir, "results/figure/L6717_vs_strongline.png")
plt.savefig(savepath, dpi=200)
print(f"Saved: {savepath}")
plt.show()


# === L6731 有効マスク ===
m6731 = (
    np.isfinite(L6731) &
    (L6731 > 0)
)

fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(2, 3,
                      left=0.10, right=0.90, bottom=0.06, top=0.98, 
                      wspace=0.00, hspace=0.00)

axes = []
for i in range(2):
    for j in range(3):
        axes.append(fig.add_subplot(gs[i, j]))

for ax, r in zip(axes, ratios):
    m = m6731 & np.isfinite(ratio_data[r])
    ax.scatter(
        ratio_data[r][m],
        L6731[m],
        s=1,
        alpha=0.6,
        label=r
    )

    ax.text(
        0.05, 0.95,
        r,
        transform=ax.transAxes,
        fontsize=20,
        verticalalignment='top'
    )

    ax.set_yscale("log")
    ax.set_xlim(*xlim_default)
    ax.set_ylim(*ylim_default)

    for spine in ax.spines.values():
        spine.set_linewidth(2)
        spine.set_color("black")

# ラベル
axes[0].set_ylabel("L([S II] 6731) [erg/s]")
axes[3].set_ylabel("L([S II] 6731) [erg/s]")

# for ax, r in zip(axes[3:], ratios[3:]):
#     ax.set_xlabel(r)

for ax in axes[:3]:
    ax.tick_params(labelbottom=False)

for ax in (axes[1], axes[2], axes[4], axes[5]):
    ax.tick_params(labelleft=False)

savepath = os.path.join(current_dir, "results/figure/L6731_vs_strongline.png")
plt.savefig(savepath, dpi=200)
print(f"Saved: {savepath}")
plt.show()
