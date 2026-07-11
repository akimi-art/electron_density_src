#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
スクリプトの概要:
logΣSFR ビンごとに
  [SII]6717,6731 フラックスをスタック
→ MCで ratio 分布
→ PyNebで ne 分布
→ P16, P50, P84 を保存・描画
→ 完全なサンプルのみを対象とする
→ mean, medianの結果も追加
→ Haで規格化したweighted stackも追加（ただし、HaのS/Nが十分なものに限定する必要あり）

* v6との変更点: 
1. フラックスの平均をとった後にratioをだす
2. meanスタックを追加

使用方法:
    stacked_sii_ne_vs_sigma_sfr_v3.py [オプション]

著者: A. M.
作成日: 2026-07-09

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
fits_path = os.path.join(current_dir, "results/fits/mpajhu_dr7_v5_2_merged_zlt0.2_Lgt1e+39_radius.fits")

out_csv = os.path.join(current_dir, "results/csv/stacked_sii_ratio_vs_sigma_sfr_COMPLETE_v3.csv")
out_png = os.path.join(current_dir, "results/figure/stacked_sii_ratio_vs_sigma_sfr_COMPLETE_v3.png")

os.makedirs(os.path.dirname(out_csv), exist_ok=True)
os.makedirs(os.path.dirname(out_png), exist_ok=True)

# ==========================================
# パラメータ
# ==========================================
BIN_WIDTH = 0.3
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

# Re（arcsec → kpc）
arcsec_to_kpc = cosmo.kpc_proper_per_arcmin(z).value / 60.0
Re_kpc = df["Re"].values * arcsec_to_kpc

df["Re_kpc"] = Re_kpc
logRe = np.log10(Re_kpc)
df["logRe"] = logRe


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

m_re = np.isfinite(Re_kpc) & (Re_kpc > 0)

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
    & m_re  # ← ここに追加
)

logSFR = df["sfr_MEDIAN"].values
logSigma_SFR = logSFR - np.log10(2*np.pi*Re_kpc**2)
df["logSigma_SFR"] = logSigma_SFR

m_complete = m_sii & m_sfr & m_ratio & m_re  # ← ここにも追加


# ==========================================
# ビン作成
# ==========================================
df["sigma_sfr_MEDIAN"] = np.nan  # 先に列を作る

df.loc[m_complete, "sigma_sfr_MEDIAN"] = df.loc[m_complete, "logSigma_SFR"] # 新しい列をmask付きで追加

logSigma_SFR_all = df.loc[m_complete, "sigma_sfr_MEDIAN"].values

edges = np.arange(
    np.floor(logSigma_SFR_all.min()/BIN_WIDTH)*BIN_WIDTH,
    np.ceil(logSigma_SFR_all.max()/BIN_WIDTH)*BIN_WIDTH + BIN_WIDTH,
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

rng = np.random.default_rng()

rows = []


# ==========================================
# メインstack（完全サンプルのみ）
# ==========================================
for lo, hi in zip(edges[:-1], edges[1:]):

    m_bin = (
        m_complete &
        (df["sigma_sfr_MEDIAN"] >= lo) &
        (df["sigma_sfr_MEDIAN"] < hi)
    )

    N = np.sum(m_bin)
    if N < NMIN:
        continue



    f1 = F6716[m_bin]
    e1 = err6716[m_bin]
    f2 = F6731[m_bin]
    e2 = err6731[m_bin]
    fHa = FHa[m_bin]
    eHa = errHa[m_bin]

    # ==========================================
    # percentile cut（追加）
    # ==========================================

    f1_lo, f1_hi = np.percentile(
        f1,
        [2, 98] # 1, 99でも可
    )

    f2_lo, f2_hi = np.percentile(
        f2,
        [2, 98]
    )

    fHa_lo, fHa_hi = np.percentile(
        fHa,
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
        &
        (fHa >= fHa_lo)
        &
        (fHa <= fHa_hi)
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

    fHa = fHa[good]
    eHa = eHa[good]

    # MCで各銀河を揺らす

    f1_mc = rng.normal(
        f1[:, None],
        e1[:, None],
        (len(f1), N_MC)
    )

    f2_mc = rng.normal(
        f2[:, None],
        e2[:, None],
        (len(f2), N_MC)
    )

    # ==========================================
    # Hα規格化 stack（正しいMC）
    # ==========================================

    # MCで各銀河を揺らす
    f1_i_mc = rng.normal(f1[:, None], e1[:, None], (len(f1), N_MC))
    f2_i_mc = rng.normal(f2[:, None], e2[:, None], (len(f2), N_MC))
    fHa_i_mc = rng.normal(fHa[:, None], eHa[:, None], (len(fHa), N_MC))

    # valid（ゼロ除算防止）
    valid_mc = (f1_i_mc > 0) & (f2_i_mc > 0) & (fHa_i_mc > 0)

    # Hα規格化（各MC）
    r1_mc = np.where(valid_mc, f1_i_mc / fHa_i_mc, np.nan)
    r2_mc = np.where(valid_mc, f2_i_mc / fHa_i_mc, np.nan)

    # 各MCでstack
    w1 = 1.0 / (e1[:, None]**2)
    w2 = 1.0 / (e2[:, None]**2)

    R1_stack_mc = np.nansum(w1 * r1_mc, axis=0) / np.nansum(w1, axis=0)
    R2_stack_mc = np.nansum(w2 * r2_mc, axis=0) / np.nansum(w2, axis=0)

    # ratio計算（各MC）
    valid_ratio_mc = (R2_stack_mc > 0)

    R_Ha_w_mc = np.full_like(R1_stack_mc, np.nan)
    R_Ha_w_mc[valid_ratio_mc] = (
        R1_stack_mc[valid_ratio_mc] /
        R2_stack_mc[valid_ratio_mc]
    )

    R1_mean_Ha_mc = np.nanmean(r1_mc, axis=0)
    R2_mean_Ha_mc = np.nanmean(r2_mc, axis=0)

    R_Ha_mean_mc = (
        R1_mean_Ha_mc
        /
        R2_mean_Ha_mc
    )

    R1_med_Ha_mc = np.nanmedian(r1_mc, axis=0)
    R2_med_Ha_mc = np.nanmedian(r2_mc, axis=0)

    R_Ha_med_mc = (
        R1_med_Ha_mc
        /
        R2_med_Ha_mc
    )

    # 統計
    # Mean(Ha)

    R_Ha_mean_50 = np.nanmedian(R_Ha_mean_mc)
    R_Ha_mean_16 = np.nanpercentile(R_Ha_mean_mc,16)
    R_Ha_mean_84 = np.nanpercentile(R_Ha_mean_mc,84)

    # Median(Ha)

    R_Ha_med_50 = np.nanmedian(R_Ha_med_mc)
    R_Ha_med_16 = np.nanpercentile(R_Ha_med_mc,16)
    R_Ha_med_84 = np.nanpercentile(R_Ha_med_mc,84)

    # Weighted(Ha)

    R_Ha_w_50 = np.nanmedian(R_Ha_w_mc)
    R_Ha_w_16 = np.nanpercentile(R_Ha_w_mc,16)
    R_Ha_w_84 = np.nanpercentile(R_Ha_w_mc,84)
    

    # ===========================
    # Mean stack
    # ===========================

    F1_mean_mc = np.nanmean(f1_mc, axis=0)
    F2_mean_mc = np.nanmean(f2_mc, axis=0)

    R_mean_mc = F1_mean_mc / F2_mean_mc

    # ===========================
    # Median stack
    # ===========================

    F1_med_mc = np.nanmedian(f1_mc, axis=0)
    F2_med_mc = np.nanmedian(f2_mc, axis=0)

    R_med_mc = F1_med_mc / F2_med_mc

    # ===========================
    # Weighted stack
    # ===========================

    w1 = 1.0 / e1[:, None]**2
    w2 = 1.0 / e2[:, None]**2

    F1_w_mc = (
        np.sum(w1 * f1_mc, axis=0)
        /
        np.sum(w1, axis=0)
    )

    F2_w_mc = (
        np.sum(w2 * f2_mc, axis=0)
        /
        np.sum(w2, axis=0)
    )

    R_w_mc = F1_w_mc / F2_w_mc


    # Mean

    R_mean_50 = np.nanmedian(R_mean_mc)
    R_mean_16 = np.nanpercentile(R_mean_mc, 16)
    R_mean_84 = np.nanpercentile(R_mean_mc, 84)

    # Median

    R_med_50 = np.nanmedian(R_med_mc)
    R_med_16 = np.nanpercentile(R_med_mc, 16)
    R_med_84 = np.nanpercentile(R_med_mc, 84)

    # Weighted

    R_w_50 = np.nanmedian(R_w_mc)
    R_w_16 = np.nanpercentile(R_w_mc, 16)
    R_w_84 = np.nanpercentile(R_w_mc, 84)


    rows.append(dict(
        logSigma_SFR_lo=lo,
        logSigma_SFR_hi=hi,
        logSigma_SFR_cen = 0.5*(lo+hi),
        N=N,

        R_mean=R_mean_50,
        R_mean_err_lo=R_mean_50-R_mean_16,
        R_mean_err_hi=R_mean_84-R_mean_50,

        R_med=R_med_50,
        R_med_err_lo=R_med_50-R_med_16,
        R_med_err_hi=R_med_84-R_med_50,

        R_w=R_w_50,
        R_w_err_lo=R_w_50-R_w_16,
        R_w_err_hi=R_w_84-R_w_50,

        R_Ha_mean=R_Ha_mean_50,
        R_Ha_mean_err_lo=R_Ha_mean_50-R_Ha_mean_16,
        R_Ha_mean_err_hi=R_Ha_mean_84-R_Ha_mean_50,

        R_Ha_med=R_Ha_med_50,
        R_Ha_med_err_lo=R_Ha_med_50-R_Ha_med_16,
        R_Ha_med_err_hi=R_Ha_med_84-R_Ha_med_50,

        R_Ha_w=R_Ha_w_50,
        R_Ha_w_err_lo=R_Ha_w_50-R_Ha_w_16,
        R_Ha_w_err_hi=R_Ha_w_84-R_Ha_w_50,
    ))

res = pd.DataFrame(rows)
res.to_csv(out_csv, index=False)
print("Saved:", out_csv)

# ==========================================
# 描画
# ==========================================
fig, ax = plt.subplots(figsize=(6,6))

# 描画前に必ず定義
df["R_SII"] = F6716 / F6731

# 完全（青）
ax.scatter(
    df.loc[m_complete, "sigma_sfr_MEDIAN"],
    df.loc[m_complete, "R_SII"],
    s=0.01,
    marker='.',
    alpha=0.8,
    color="C0",
)

x = res["logSigma_SFR_cen"].values


# mean
y_mean = res["R_mean"].values

yerr_mean = np.vstack([
    res["R_mean_err_lo"],
    res["R_mean_err_hi"]
])

# median
y_med = res["R_med"].values
yerr_med = np.vstack([
    res["R_med_err_lo"], 
    res["R_med_err_hi"]
])

# weighted mean
y_w = res["R_w"].values
yerr_w = np.vstack([
    res["R_w_err_lo"],
    res["R_w_err_hi"]
])


# Hα normalized mean
y_Ha_mean = res["R_Ha_mean"].values

yerr_Ha_mean = np.vstack([
    res["R_Ha_mean_err_lo"],
    res["R_Ha_mean_err_hi"]
])

# Hα normalized median
y_Ha_med = res["R_Ha_med"].values
yerr_Ha_med = np.vstack([
    res["R_Ha_med_err_lo"],
    res["R_Ha_med_err_hi"]
])

# Hα normalized weighted mean
y_Ha_w = res["R_Ha_w"].values

yerr_Ha_w = np.vstack([
    res["R_Ha_w_err_lo"],
    res["R_Ha_w_err_hi"]
])

# ---------------------------
# mean
# ---------------------------
ax.errorbar(
    x,
    y_mean,
    yerr=yerr_mean,
    fmt="s",
    color="red",
    mfc="red",
    capsize=3,
    label="Mean"
)


# ---------------------------
# median
# ---------------------------
ax.errorbar(
    x,
    y_med,
    yerr=yerr_med,
    fmt="^",
    color="red",
    mfc="red",
    capsize=3,
    label="Median"
)

# ---------------------------
# weighted mean
# ---------------------------
ax.errorbar(
    x,
    y_w,
    yerr=yerr_w,
    fmt="D",
    color="red",
    mfc="red",
    capsize=3,
    label="Weighted"
)

# ---------------------------
# Hα normalized mean
# ---------------------------
ax.errorbar(
    x,
    y_Ha_mean,
    yerr=yerr_Ha_mean,
    fmt="s",
    color="red",
    mfc="white",
    capsize=3,
    label="Mean (Hα norm)"
)

# ---------------------------
# Hα normalized median
# ---------------------------
ax.errorbar(
    x,
    y_Ha_med,
    yerr=yerr_Ha_med,
    fmt="^",
    color="red",
    mfc="white",
    capsize=3,
    label="Median (Hα norm)"
)

# ---------------------------
# Hα normalized weighted mean
# ---------------------------
ax.errorbar(
    x,
    y_Ha_w,
    yerr=yerr_Ha_w,
    fmt="D",
    color="red",
    mfc="white",
    capsize=3,
    label="Weighted (Hα norm)"
)

ax.set_xlabel(r"$\log(\Sigma_{\rm SFR})\ [{\rm M_\odot\ yr^{-1}\ kpc^{-2}}]$")
ax.set_ylabel(r"[SII] 6717 / 6731")
ax.set_xlim(-5, 1.1)
ax.set_ylim(0.5,2.0)

for spine in ax.spines.values():
    spine.set_linewidth(2)

plt.tight_layout()
plt.savefig(out_png, dpi=200)
plt.show()

print("Saved:", out_png)


# ==========================================
# countヒートマップを作成
# ==========================================
xbins = np.arange(-5, 1.1, 0.01)
ybins = np.arange(0.5, 2.1, 0.01)

count_map, xedge, yedge, _ = (
    binned_statistic_2d(
        df.loc[m_complete, "sigma_sfr_MEDIAN"],
        df.loc[m_complete, "R_SII"],
        values=None,
        statistic="count",
        bins=[xbins, ybins]
    )
)

fig, ax = plt.subplots(figsize=(8,6))
fig.subplots_adjust(left=0.15, right=0.85, bottom=0.15, top=0.85)

vmin = 0 # 下限を5パーセンタイルに設定（必要に応じて調整）
vmax = 20 # 上限を95パーセンタイルに設定（必要に応じて調整）

plt.pcolormesh(
    xedge,
    yedge,
    count_map.T,
    vmin=vmin, vmax=vmax,  # カラーマップの範囲を固定（必要に応じて調整）
    shading="auto",
    cmap="viridis" # 必要に応じて調整
)

plt.colorbar()

# mean
ax.errorbar(
    x, y_mean,
    yerr=yerr_mean,
    fmt="s",
    color="#ff5eaa", # ローズピンク:#ff5eaa, マゼンタピンク:#ff3399
    mfc="#ff5eaa",
    capsize=3,
    label="Mean"
)

# median
ax.errorbar(
    x, y_med,
    yerr=yerr_med,
    fmt="^",
    color="#ffa500", # ライトオレンジ:#ffa500, 鮮やかなオレンジ:#ff9900
    mfc="#ffa500",
    capsize=3,
    label="Median"
)

# weighted mean
ax.errorbar(
    x, y_w,
    yerr=yerr_w,
    fmt="D",
    color="#33ffcc", # ミントシアン:#33ffcc, 明るいシアン:00e5ff
    mfc="#33ffcc",
    capsize=3,
    label="Weighted"
)

# Ha normalized mean
ax.errorbar(
    x, y_Ha_mean,
    yerr=yerr_Ha_mean,
    fmt="s",
    color="#ff5eaa",
    mfc="white",
    capsize=3,
    label="Mean (Hα)"
)

# Ha normalized median
ax.errorbar(
    x, y_Ha_med,
    yerr=yerr_Ha_med,
    fmt="^",
    color="#ff9900",
    mfc="white",
    capsize=3,
    label="Median (Hα)"
)

# Ha normalized weighted mean
ax.errorbar(
    x, y_Ha_w,
    yerr=yerr_Ha_w,
    fmt="D",
    color="#33ffcc",
    mfc="white",
    capsize=3,
    label="Weighted (Hα)"
)

# # mean
# ax.errorbar(
#     x, y_mean,
#     yerr=yerr_mean,
#     fmt="s",
#     color="red",
#     mfc="red",
#     capsize=3,
#     label="Mean"
# )

# # median
# ax.errorbar(
#     x, y_med,
#     yerr=yerr_med,
#     fmt="^",
#     color="red",
#     mfc="red",
#     capsize=3,
#     label="Median"
# )

# # weighted mean
# ax.errorbar(
#     x, y_w,
#     yerr=yerr_w,
#     fmt="D",
#     color="red",
#     mfc="red",
#     capsize=3,
#     label="Weighted"
# )

# # Ha normalized mean
# ax.errorbar(
#     x, y_Ha_mean,
#     yerr=yerr_Ha_mean,
#     fmt="s",
#     color="red",
#     mfc="white",
#     capsize=3,
#     label="Mean (Hα)"
# )

# # Ha normalized median
# ax.errorbar(
#     x, y_Ha_med,
#     yerr=yerr_Ha_med,
#     fmt="^",
#     color="red",
#     mfc="white",
#     capsize=3,
#     label="Median (Hα)"
# )

# # Ha normalized weighted mean
# ax.errorbar(
#     x, y_Ha_w,
#     yerr=yerr_Ha_w,
#     fmt="D",
#     color="red",
#     mfc="white",
#     capsize=3,
#     label="Weighted (Hα)"
# )

ax.set_xlabel(r"$\log(\Sigma_{\rm SFR})\ [{\rm M_\odot\ yr^{-1}\ kpc^{-2}}]$")
ax.set_ylabel(r"[SII] 6717 / 6731")
ax.set_xlim(-5, 1.1)
ax.set_ylim(1.0,1.6)

for spine in ax.spines.values():
    spine.set_linewidth(2)
plt.tight_layout()
# 保存
fig_dir = os.path.join(current_dir, "results/figure")
os.makedirs(fig_dir, exist_ok=True)
save_path_count = os.path.join(
    fig_dir,
    "heat_sigma_sfr_sii_ratio_count_sdss_v3.png"
)

plt.savefig(save_path_count)
print(f"Saved heatmap to: {save_path_count}")
plt.show()


# ==========================================
# ヒストグラム＋代表値プロット
# ==========================================

nbins_plot = len(res)
ncols = 4
nrows = int(np.ceil(nbins_plot / ncols))

fig, axes = plt.subplots(
    nrows, ncols,
    figsize=(4*ncols, 3*nrows),
    sharex=True, sharey=True
)
axes = axes.flatten()

for i, row in res.iterrows():

    lo = row["logSigma_SFR_lo"]
    hi = row["logSigma_SFR_hi"]

    ax = axes[i]

    # 同じbinのデータ取り出し
    m_bin = (
        m_complete &
        (df["sigma_sfr_MEDIAN"] >= lo) &
        (df["sigma_sfr_MEDIAN"] < hi)
    )

    f1 = F6716[m_bin]
    f2 = F6731[m_bin]

    valid = (f1 > 0) & (f2 > 0)
    R_ind = f1[valid] / f2[valid]

    # ===============================
    # ヒストグラム用データ整形
    # ===============================
    R_ind = f1[valid] / f2[valid]

    # 外れ値を軽く除去（重要）
    lo_cut = np.nanpercentile(R_ind, 1)
    hi_cut = np.nanpercentile(R_ind, 99)

    R_plot = R_ind[(R_ind > lo_cut) & (R_ind < hi_cut)]

    # ===============================
    # 動的bin（Freedman–Diaconis）
    # ===============================
    q75, q25 = np.percentile(R_plot, [75, 25])
    iqr = q75 - q25

    if iqr > 0:
        bin_width = 2 * iqr / (len(R_plot) ** (1/3))
        bins = int((hi_cut - lo_cut) / bin_width)
    else:
        bins = 30

    # 安定化
    bins = max(20, min(bins, 120))

    # ===============================
    # ヒストグラム描画
    # ===============================
    ax.hist(
        R_plot,
        bins=bins,
        density=True,
        histtype="step",
        color="black",
        linewidth=1.5
    )

    # x範囲はデータに合わせる
    ax.set_xlim(lo_cut, hi_cut)


    # 縦線
    ax.axvline(row["R_mean"], c="#d0116a", ls="--", lw="1") # チェリーピンク
    ax.axvline(row["R_med"],  c="#d96b00", ls="--", lw="1") # ディープオレンジ
    ax.axvline(row["R_w"],    c="#008b8b", ls="--", lw="1") # 深みのあるティール

    ax.axvline(row["R_Ha_mean"], c="#d0116a", ls=":", lw="1")
    ax.axvline(row["R_Ha_med"],  c="#d96b00", ls=":", lw="1")
    ax.axvline(row["R_Ha_w"],    c="#008b8b", ls=":", lw="1")

    ax.text(
        0.02, 0.95,
        f"{lo:.1f}–{hi:.1f}\nN={int(row['N'])}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=20
    )

    ax.set_xlim(1.0, 1.8)
    
    for spine in ax.spines.values():
        spine.set_linewidth(2)


# 余ったsubplot消す
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

# 軸ラベル
fig.supxlabel("[SII] 6717 / 6731")
fig.supylabel("Count")

# 凡例（1つだけ）
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper right")

plt.subplots_adjust(
    wspace=0.0,
    hspace=0.0
)

# 保存
hist_path = os.path.join(
    current_dir,
    "results/figure/stacked_sii_sigma_sfr_histograms_v3.png"
)
plt.savefig(hist_path, dpi=200)

print("Saved:", hist_path)

plt.show()