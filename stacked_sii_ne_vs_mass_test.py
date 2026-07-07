#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
スクリプトの概要:
logM* ビンごとに
  [SII]6717,6731 フラックスをスタック
→ MCで ratio 分布
→ PyNebで ne 分布
→ P16, P50, P84 を保存・描画
→ 完全なサンプルのみを対象とする
→ mean, medianの結果も追加
→ Haで規格化したweighted stackも追加（ただし、HaのS/Nが十分なものに限定する必要あり）

* v4との変更点: 
1. Mass, SFRのratioヒートマップを使いながら数の少ないピクセルを除去

使用方法:
    stacked_sii_ne_vs_mass_test.py [オプション]

著者: A. M.
作成日: 2026-07-07

参考文献:
    - PEP 8:                  https://peps.python.org/pep-0008/
    - PEP 257 (Docstring規約): https://peps.python.org/pep-0257/
    - Python公式ドキュメント:    https://docs.python.org/ja/3/
    - Curti+17
"""

# === 必要なモジュールのインポート ===
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.table import Table
import re
import matplotlib.gridspec as gridspec
from pathlib import Path
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u
from scipy.stats import binned_statistic_2d


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
current_dir = os.getcwd()
fits_path = os.path.join(current_dir, "results/fits/mpajhu_dr7_v5_2_merged_zlt0.2_Lgt1e+39.fits")

out_csv = os.path.join(current_dir, "results/csv/stacked_sii_ratio_vs_mass_COMPLETE_v5.csv")
out_png = os.path.join(current_dir, "results/figure/stacked_sii_ratio_vs_mass_COMPLETE_v5.png")

os.makedirs(os.path.dirname(out_csv), exist_ok=True)
os.makedirs(os.path.dirname(out_png), exist_ok=True)

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

# ==========================================
# ビン作成
# ==========================================

logM_valid = df.loc[
    m_complete,
    "sm_MEDIAN"
].values

edges = np.arange(
    np.floor(logM_valid.min()/BIN_WIDTH)*BIN_WIDTH,
    np.ceil(logM_valid.max()/BIN_WIDTH)*BIN_WIDTH
    + BIN_WIDTH,
    BIN_WIDTH
)

# ==========================================
# スタック用関数
# ==========================================
def mean_stack(flux):
    return np.nanmean(flux)

def median_stack(flux):
    return np.nanmedian(flux)

def weighted_stack(flux, err):

    w = 1.0/err**2

    return np.sum(w*flux)/np.sum(w)

rng = np.random.default_rng(12345)

rows = []


# ==========================================
# Mass-SFR count map作成
# ==========================================

xbins_sfr = np.arange(7.0, 12.1, 0.1)
ybins_sfr = np.arange(-3.0, 3.1, 0.1)

count_map_sfr, xedge_sfr, yedge_sfr, _ = (
    binned_statistic_2d(
        df.loc[m_complete, "sm_MEDIAN"],
        df.loc[m_complete, "sfr_MEDIAN"],
        values=None,
        statistic="count",
        bins=[xbins_sfr, ybins_sfr]
    )
)

# ==========================================
# 各銀河に cell_count を付与
# ==========================================
ix = np.digitize(
    df["sm_MEDIAN"],
    xbins_sfr
) - 1

iy = np.digitize(
    df["sfr_MEDIAN"],
    ybins_sfr
) - 1

df["cell_count"] = np.nan

valid_cell = (
    (ix >= 0)
    &
    (ix < count_map_sfr.shape[0])
    &
    (iy >= 0)
    &
    (iy < count_map_sfr.shape[1])
)

df.loc[valid_cell, "cell_count"] = (
    count_map_sfr[
        ix[valid_cell],
        iy[valid_cell]
    ]
)


NMIN_CELL = 10


rows = []


# ==========================================
# メインstack（meanのみ）
# ==========================================
for lo, hi in zip(edges[:-1], edges[1:]):

    m_bin = (
        m_complete
        &
        (df["cell_count"] >= NMIN_CELL)
        &
        (df["sm_MEDIAN"] >= lo)
        &
        (df["sm_MEDIAN"] < hi)
    )

    N = np.sum(m_bin)

    if N < NMIN:
        continue

    f1 = F6716[m_bin]
    e1 = err6716[m_bin]

    f2 = F6731[m_bin]
    e2 = err6731[m_bin]


    # ==========================================
    # percentile cut
    # ==========================================

    f1_lo, f1_hi = np.percentile(
        f1,
        [2, 98] # 1, 99でも可
    )

    f2_lo, f2_hi = np.percentile(
        f2,
        [2, 98]
    )

    good = (
        (f1 >= f1_lo)
        &
        (f1 <= f1_hi)
        &
        (f2 >= f2_lo)
        &
        (f2 <= f2_hi)
    )

    print(
        f"before cut : {len(f1)}"
    )

    print(
        f"after cut  : {np.sum(good)}"
    )

    f1 = f1[good]
    e1 = e1[good]

    f2 = f2[good]
    e2 = e2[good]

    N = len(f1)


    print("\n========================")
    print(f"N(f1 > 1e-10)={np.sum(f1 > 1e-10)}")
    print(f"N(f2 > 1e-10)={np.sum(f2 > 1e-10)}")

    # df_bin = df.loc[m_bin].copy()

    # bad = (
    #     (f1 > 1e-10)
    #     |
    #     (f2 > 1e-10)
    # )

    # print(
    #     df_bin.loc[bad,[
    #         "sm_MEDIAN",
    #         "sfr_MEDIAN",
    #         "SII_6717_FLUX",
    #         "SII_6731_FLUX",
    #         "SII_6717_FLUX_ERR",
    #         "SII_6731_FLUX_ERR"
    #     ]]
    # )

    print(f"logM = [{lo:.1f}, {hi:.1f})")

    # ratioの確認
    print("mean(f1)/mean(f2) =",
          np.mean(f1)/np.mean(f2))

    print("median(f1/f2) =",
          np.median(f1/f2))

    print("N =", N)

    # f1分布
    print("f1 min/max")
    print(np.min(f1))
    print(np.max(f1))

    print("f1 percentiles")
    print(
        np.percentile(
            f1,
            [0,1,50,99,100]
        )
    )

    # f2分布
    print("f2 min/max")
    print(np.min(f2))
    print(np.max(f2))

    print("f2 percentiles")
    print(
        np.percentile(
            f2,
            [0,1,50,99,100]
        )
    )

    # mean vs median
    print("f1 mean median")
    print(
        np.mean(f1),
        np.median(f1)
    )

    print("f2 mean median")
    print(
        np.mean(f2),
        np.median(f2)
    )

    # ←ここに追加

    f1_mc = rng.normal(
        f1[:,None],
        e1[:,None],
        (len(f1),N_MC)
    )

    f2_mc = rng.normal(
        f2[:,None],
        e2[:,None],
        (len(f2),N_MC)
    )


    # # ----------------------------
    # # 発散対策（あまり意味がないかもしれない）
    # # ----------------------------
    # valid = (
    #     (F1_mean_mc > 0)
    #     &
    #     (F2_mean_mc > 0)
    #     &
    #     np.isfinite(F1_mean_mc)
    #     &
    #     np.isfinite(F2_mean_mc)
    # )

    # R_mean_mc = np.full(
    #     len(F1_mean_mc),
    #     np.nan
    # )

    # R_mean_mc[valid] = (
    #     F1_mean_mc[valid]
    #     /
    #     F2_mean_mc[valid]
    # )

    # R50 = np.nanmedian(R_mean_mc[valid])
    # R16 = np.nanpercentile(R_mean_mc[valid],16)
    # R84 = np.nanpercentile(R_mean_mc[valid],84)

    F1_mean_mc = np.nanmean(
        f1_mc,
        axis=0
    )

    F2_mean_mc = np.nanmean(
        f2_mc,
        axis=0
    )

    R_mean_mc = (
        F1_mean_mc
        /
        F2_mean_mc
    )

    R50 = np.nanmedian(R_mean_mc)
    R16 = np.nanpercentile(R_mean_mc,16)
    R84 = np.nanpercentile(R_mean_mc,84)

    print(
        "R50(MC mean stack) =",
        R50
    )
    print(
        np.percentile(
            R_mean_mc,
            [1,5,16,50,84,95,99]
        )
    )

    rows.append(dict(

        logM_lo=lo,
        logM_hi=hi,

        logM_cen=0.5*(lo+hi),

        N=N,

        R_mean=R50,

        R_mean_err_lo=R50-R16,
        R_mean_err_hi=R84-R50

    ))


# 保存
res = pd.DataFrame(rows)

res.to_csv(
    out_csv,
    index=False
)


# Mean stack 図
fig, ax = plt.subplots(
    figsize=(6,6)
)

mask_plot = (
    m_complete
    &
    (df["cell_count"] >= NMIN_CELL)
)

ax.scatter(
    df.loc[mask_plot,"sm_MEDIAN"],
    df.loc[mask_plot,"R_SII"],
    s=0.01,
    alpha=0.1
)


ax.errorbar(

    res["logM_cen"],

    res["R_mean"],

    yerr=np.vstack([
        res["R_mean_err_lo"],
        res["R_mean_err_hi"]
    ]),

    fmt="o",
    color="red",
    capsize=3,
    label=f"Mean (Ncell ≥ {NMIN_CELL})"
)

ax.set_xlim(8,12)
ax.set_ylim(1.0,1.6)

ax.legend()

plt.show()


# Count map（残す）
fig, ax = plt.subplots(
    figsize=(8,6)
)

plt.pcolormesh(

    xedge_sfr,
    yedge_sfr,

    count_map_sfr.T,

    shading="auto",
    cmap="viridis"
)

plt.colorbar(
    label="Count"
)

plt.xlabel(
    r"log M$_*$"
)

plt.ylabel(
    r"log SFR"
)

plt.show()