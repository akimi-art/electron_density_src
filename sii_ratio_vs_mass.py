#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
スクリプトの概要:
SIIのratioとMassの関係を見て、
Binning Artifactを評価します。

使用方法:
    sii_ratio_vs_mass.py [オプション]

著者: A. M.
作成日: 2026-07-06

参考文献:
    - PEP 8:                  https://peps.python.org/pep-0008/
    - PEP 257 (Docstring規約): https://peps.python.org/pep-0257/
    - Python公式ドキュメント:    https://docs.python.org/ja/3/
"""

# === 必要なパッケージのインストール === #
# 必要なモジュールのインポート
from astropy.io import fits
# リンク:https://docs.astropy.org/en/stable/io/fits/index.html
import numpy as np
import os
import matplotlib.pyplot as plt
import re
import math
from astropy.table import Table
import importlib.util
import sys
import emcee
import psutil
from scipy.stats import kendalltau
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np
import importlib.util
from scipy.stats import binned_statistic_2d
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u
from statsmodels.nonparametric.smoothers_lowess import lowess


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


# ==========================================
# 入出力
# ==========================================
base_dir = os.getcwd()
fits_path = os.path.join(base_dir, "results/fits/mpajhu_dr7_v5_2_merged_zlt0.2_Lgt1e+39.fits")

# ==========================================
# パラメータ
# ==========================================
BIN_WIDTH = 0.2
NMIN = 100
N_MC = 5000
N_BS = 1000   # ← 追加（mean/median用）

UNIT_FLUX = 1e-17        # MPA-JHU flux単位

# ==========================================
# 読み込み
# ==========================================
tab = Table.read(fits_path, hdu=1)
df = tab.to_pandas()
# df = pd.read_csv(csv_path)

# ==========================================
# 基本量の計算
# ==========================================
z = df["Z"].values

F6716 = df["SII_6717_FLUX"].values * UNIT_FLUX
F6731 = df["SII_6731_FLUX"].values * UNIT_FLUX
err6716 = df["SII_6717_FLUX_ERR"].values * UNIT_FLUX
err6731 = df["SII_6731_FLUX_ERR"].values * UNIT_FLUX
# Hα
FHa = df["H_ALPHA_FLUX"].values * UNIT_FLUX
errHa = df["H_ALPHA_FLUX_ERR"].values * UNIT_FLUX


sn6716 = F6716 / err6716
sn6731 = F6731 / err6731

# luminosity
d_L = cosmo.luminosity_distance(z).to(u.cm).value
L6716 = 4 * np.pi * d_L**2 * F6716
L6731 = 4 * np.pi * d_L**2 * F6731

df["R_SII"] = F6716 / F6731

# ==========================================
# マスク定義
# ==========================================
def valid_mass(x):
    x = np.asarray(x, float)

    m = np.isfinite(x)

    # 変更
    m &= (x >= 6.0)
    m &= (x <= 12.0) 

    return m

# 追加
def valid_sfr(x):

    x = np.asarray(x, float)

    m = np.isfinite(x)

    m &= (x > -3)
    m &= (x < 3)

    return m

m_sii = (
    np.isfinite(F6716) & np.isfinite(F6731) &
    np.isfinite(err6716) & np.isfinite(err6731) &
    np.isfinite(FHa) & np.isfinite(errHa) &
    (err6716 > 0) & (err6731 > 0) &
    (errHa > 0)
)

m_sm = valid_mass(df["sm_MEDIAN"])
m_sfr = valid_sfr(df["sfr_MEDIAN"]) # 追加
m_ratio = np.isfinite(df["R_SII"])

# 変更
mask_all = (
    m_sii
    & m_sm
    & m_sfr
    & m_ratio
)
m_complete = mask_all

# ------------------------------------------
# 使用データ
# ------------------------------------------

mass_all = df.loc[m_complete, "sm_MEDIAN"].values
ratio_all = df.loc[m_complete, "R_SII"].values

# ------------------------------------------
# 試す bin幅
# ------------------------------------------

bin_widths = [
    0.01,
    0.10,
    0.20,
    0.30,
    0.50,
    1.00,
    1.20,
    1.40,
    1.50,
]

# ------------------------------------------
# plot
# ------------------------------------------

fig, ax = plt.subplots(figsize=(8,6))

colors = [
    "C0",
    "C1",
    "C2",
    "C3",
    "C4",
    "C5",
    "C6",
    "C7",
    "C8",
]

for bw, color in zip(bin_widths, colors):

    edges = np.arange(
        np.floor(np.nanmin(mass_all)/bw)*bw,
        np.ceil(np.nanmax(mass_all)/bw)*bw + bw,
        bw
    )

    xvals = []
    med50 = []
    med16 = []
    med84 = []

    for lo, hi in zip(edges[:-1], edges[1:]):

        m = (
            (mass_all >= lo)
            &
            (mass_all < hi)
        )

        N = np.sum(m)

        # if N < 50:
        if N < 1:
            continue

        ratio_bin = ratio_all[m]

        xvals.append(
            0.5*(lo+hi)
        )

        med50.append(
            np.nanmedian(ratio_bin)
        )

        med16.append(
            np.nanpercentile(ratio_bin,16)
        )

        med84.append(
            np.nanpercentile(ratio_bin,84)
        )

    xvals = np.array(xvals)

    med50 = np.array(med50)
    med16 = np.array(med16)
    med84 = np.array(med84)

    # 中央値
    ax.plot(
        xvals,
        med50,
        color=color,
        lw=2,
        label=f"{bw:.1f} dex"
    )

    # scatterを重ねると分かりやすい
    ax.scatter(
        xvals,
        med50,
        color=color,
        s=20
    )

    # # 16-84%
    # ax.fill_between(
    #     xvals,
    #     med16,
    #     med84,
    #     color=color,
    #     alpha=0.15
    # )

# ------------------------------------------
# style
# ------------------------------------------

ax.set_xlabel(
    r"$\log(M_\star/M_\odot)$"
)

ax.set_ylabel(
    r"Median [SII]6717/6731"
)

ax.set_xlim(8,12)
ax.set_ylim(1.2, 1.6)

# ax.legend(
#     title="Bin width"
# )

for spine in ax.spines.values():
    spine.set_linewidth(2)

plt.tight_layout()

save_path = os.path.join(
    base_dir,
    "results/figure/binwidth_test_median_ratio_mass.png"
)

plt.savefig(
    save_path,
    dpi=200
)

print(
    "Saved:",
    save_path
)

plt.show()

from statsmodels.nonparametric.smoothers_lowess import lowess

mass = df.loc[m_complete, "sm_MEDIAN"].values
ratio = df.loc[m_complete, "R_SII"].values

# ----------------------------------
# LOWESS用サブサンプル
# ----------------------------------

N_LOWESS = 50000

rng = np.random.default_rng(12345)

idx = rng.choice(
    len(mass),
    size=min(N_LOWESS, len(mass)),
    replace=False
)

mass_sub = mass[idx]
ratio_sub = ratio[idx]

# LOWESSの安定化のためMass順に並べる
order = np.argsort(mass_sub)

mass_sub = mass_sub[order]
ratio_sub = ratio_sub[order]

# ----------------------------------
# plot
# ----------------------------------

fig, ax = plt.subplots(figsize=(8,6))

# 生データ（重ければこちらもsubに変えてよい）
ax.scatter(
    mass,
    ratio,
    s=1,
    alpha=0.02,
    color="0.8"
)

fracs = [
    0.01, # 全体の1%のデータを使って局所フィットする
    0.02,
    0.03,
]

for frac in fracs:

    curve = lowess(
        ratio_sub,
        mass_sub,
        frac=frac,
        return_sorted=True
    )

    ax.plot(
        curve[:,0],
        curve[:,1],
        lw=2,
        label=f"frac={frac}"
    )

ax.set_xlabel(
    r"$\log(M_\star/M_\odot)$"
)

ax.set_ylabel(
    r"[SII]6717 / 6731"
)

ax.set_xlim(8,11)
ax.set_ylim(1.2,1.6)

save_path = os.path.join(
    base_dir,
    "results/figure/binwidth_test_median_ratio_mass_lowess.png"
)

plt.savefig(
    save_path,
    dpi=200
)

plt.tight_layout()
plt.show()