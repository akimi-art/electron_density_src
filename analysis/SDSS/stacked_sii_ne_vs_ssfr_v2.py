#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
スクリプトの概要:
logsSFR ビンごとに
  [SII]6717,6731 フラックスをスタック
→ MCで ratio 分布
→ PyNebで ne 分布
→ P16, P50, P84 を保存・描画
→ 完全なサンプルのみを対象とする
→ mean, medianの結果も追加
→ Haで規格化したweighted stackも追加（ただし、HaのS/Nが十分なものに限定する必要あり）

使用方法:
    stacked_sii_ne_vs_ssfr_v2.py [オプション]

著者: A. M.
作成日: 2026-06-17

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
# Imports
# ==========================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u

# ==========================================
# 入出力
# ==========================================
current_dir = os.getcwd()
fits_path = os.path.join(current_dir, "results/fits/mpajhu_dr7_v5_2_merged_zlt0.2_Lgt1e+39.fits")

out_csv = os.path.join(current_dir, "results/csv/stacked_sii_ratio_vs_ssfr_COMPLETE_v2.csv")
out_png = os.path.join(current_dir, "results/figure/stacked_sii_ratio_vs_ssfr_COMPLETE_v2.png")

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
    m &= (x > 0) & (x < 13)
    return m
    
def valid_sfr(x):

    x = np.asarray(x, float)

    m = np.isfinite(x)
    m &= (x > -5) & (x < 3)
    m &= (x != -1.0)

    return m

m_sii = (
    np.isfinite(F6716) & np.isfinite(F6731) &
    np.isfinite(err6716) & np.isfinite(err6731) &
    np.isfinite(FHa) & np.isfinite(errHa) &
    (err6716 > 0) & (err6731 > 0) &
    (errHa > 0)
)

m_sm = valid_mass(df["sm_MEDIAN"])
m_sfr = valid_sfr(df["sfr_MEDIAN"])
m_ratio = np.isfinite(df["R_SII"])

mask_all = m_sii & m_sfr & m_ratio
m_complete = mask_all

# ==========================================
# ビン作成
# ==========================================
logM = df.loc[m_complete, "sm_MEDIAN"].values
logSFR = df.loc[m_complete, "sfr_MEDIAN"].values
logsSFR = logSFR - logM
df["ssfr_MEDIAN"] = np.nan  # 先に列を作る

df.loc[m_complete, "ssfr_MEDIAN"] = logsSFR # 新しい列をmask付きで追加

edges = np.arange(
    np.floor(logsSFR.min()/BIN_WIDTH)*BIN_WIDTH,
    np.ceil(logsSFR.max()/BIN_WIDTH)*BIN_WIDTH + BIN_WIDTH,
    BIN_WIDTH
)

# ==========================================
# スタック用関数
# ==========================================
def weighted_mean(flux, err):
    w = 1.0 / err**2
    mu = np.sum(w * flux) / np.sum(w)
    sigma = np.sqrt(1.0 / np.sum(w))
    return mu, sigma

rng = np.random.default_rng()

rows = []


# ==========================================
# メインstack（完全サンプルのみ）
# ==========================================
for lo, hi in zip(edges[:-1], edges[1:]):

    ssfr_bin = (
        m_complete &
        (df["ssfr_MEDIAN"] >= lo) &
        (df["ssfr_MEDIAN"] < hi)
    )

    N = np.sum(ssfr_bin)
    if N < NMIN:
        continue

    f1 = F6716[ssfr_bin]
    e1 = err6716[ssfr_bin]
    f2 = F6731[ssfr_bin]
    e2 = err6731[ssfr_bin]
    fHa = FHa[ssfr_bin]
    eHa = errHa[ssfr_bin]

    # ===========================
    # simple ratio（各銀河）
    # ===========================
    valid_ind = (f1 > 0) & (f2 > 0)
    R_individual = f1[valid_ind] / f2[valid_ind]
    
    # 平均と中央値
    R_mean_simple = np.nanmean(R_individual)
    R_med_simple  = np.nanmedian(R_individual)

    # ===========================
    # Hαで正規化した individual ratio
    # ===========================
    valid_Ha = (f1 > 0) & (f2 > 0) & (fHa > 0)

    r1_ind = f1[valid_Ha] / fHa[valid_Ha]
    r2_ind = f2[valid_Ha] / fHa[valid_Ha]

    # median (Hα正規化後)
    R_Ha_med = np.nanmedian(r1_ind / r2_ind)
    
    # ===========================
    # bootstrap誤差
    # ===========================
    bs_mean = []
    bs_med  = []
    bs_Ha_med = []
    
    for _ in range(N_BS):
        idx = rng.integers(0, len(R_individual), len(R_individual))
        sample = R_individual[idx]
    
        bs_mean.append(np.nanmean(sample))
        bs_med.append(np.nanmedian(sample))

        # Hα median bootstrap
        idx = rng.integers(0, len(r1_ind), len(r1_ind))
        sample_r1 = r1_ind[idx]
        sample_r2 = r2_ind[idx]

        bs_Ha_med.append(np.nanmedian(sample_r1 / sample_r2))
    
    bs_mean = np.array(bs_mean)
    bs_med  = np.array(bs_med)
    bs_Ha_med = np.array(bs_Ha_med)

    mean16, mean84 = np.percentile(bs_mean, [16, 84])
    med16,  med84  = np.percentile(bs_med,  [16, 84])
    Ha_med16, Ha_med84 = np.percentile(bs_Ha_med, [16, 84])


    F1, e1_stack = weighted_mean(f1, e1)
    F2, e2_stack = weighted_mean(f2, e2)

    # Monte Carlo for ratio
    f1_mc = rng.normal(F1, e1_stack, N_MC)
    f2_mc = rng.normal(F2, e2_stack, N_MC)

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

    R_Ha_mc = np.full_like(R1_stack_mc, np.nan)
    R_Ha_mc[valid_ratio_mc] = (
        R1_stack_mc[valid_ratio_mc] /
        R2_stack_mc[valid_ratio_mc]
    )

    # 統計
    R_Ha_50 = np.nanmedian(R_Ha_mc)
    R_Ha_16 = np.nanpercentile(R_Ha_mc, 16)
    R_Ha_84 = np.nanpercentile(R_Ha_mc, 84)


    valid = (f1_mc > 0) & (f2_mc > 0)
    R_mc = f1_mc[valid] / f2_mc[valid]

    R50 = np.nanmedian(R_mc)
    R16 = np.nanpercentile(R_mc, 16)
    R84 = np.nanpercentile(R_mc, 84)

    rows.append(dict(
        logsSFR_lo=lo,
        logsSFR_hi=hi,
        logsSFR_cen = 0.5*(lo+hi),
        N = N,

        # ----- weighted stack -----
        R_wmed = R50,
        R_werr_lo = R50 - R16,
        R_werr_hi = R84 - R50,

        # ----- simple mean -----
        R_mean = R_mean_simple,
        R_mean_err_lo = R_mean_simple - mean16,
        R_mean_err_hi = mean84 - R_mean_simple,

        # ----- median -----
        R_med = R_med_simple,
        R_med_err_lo = R_med_simple - med16,
        R_med_err_hi = med84 - R_med_simple,

        # --- Hα normalized ratio ---
        R_Ha = R_Ha_50,
        R_Ha_err_lo = R_Ha_50 - R_Ha_16,
        R_Ha_err_hi = R_Ha_84 - R_Ha_50,
        
        # --- Hα normalized median ---
        R_Ha_med = R_Ha_med,
        R_Ha_med_err_lo = R_Ha_med - Ha_med16,
        R_Ha_med_err_hi = Ha_med84 - R_Ha_med,

        N_MC_valid=int(valid.sum()),
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
    df.loc[m_complete, "ssfr_MEDIAN"],
    df.loc[m_complete, "R_SII"],
    s=0.01,
    marker='.',
    alpha=0.8,
    color="C0",
)

x = res["logsSFR_cen"].values

# weighted
y_w = res["R_wmed"].values
yerr_w = np.vstack([res["R_werr_lo"], res["R_werr_hi"]])

# mean
y_mean = res["R_mean"].values
yerr_mean = np.vstack([res["R_mean_err_lo"], res["R_mean_err_hi"]])

# median
y_med = res["R_med"].values
yerr_med = np.vstack([res["R_med_err_lo"], res["R_med_err_hi"]])

# Hα normalized weighted mean
y_Ha = res["R_Ha"].values
yerr_Ha = np.vstack([
    res["R_Ha_err_lo"],
    res["R_Ha_err_hi"]
])

# Ha normalized median
y_Ha_med = res["R_Ha_med"].values
yerr_Ha_med = np.vstack([
    res["R_Ha_med_err_lo"],
    res["R_Ha_med_err_hi"]
])

# ---------------------------
# weighted mean（白四角）
# ---------------------------
ax.errorbar(
    x, y_w,
    yerr=yerr_w,
    fmt="s",
    mec="white", mfc="none",
    ecolor="white", color="white",
    capsize=3,
    label="Weighted stack"
)

# # ---------------------------
# # simple mean（赤丸）: 削除した方が良い(値が壊れるため)
# # ---------------------------
# ax.errorbar(
#     x, y_mean,
#     yerr=yerr_mean,
#     fmt="o",
#     mec="red", mfc="none",
#     ecolor="red", color="red",
#     capsize=3,
#     label="Mean"
# )

# ---------------------------
# median（シアン三角）
# ---------------------------
ax.errorbar(
    x, y_med,
    yerr=yerr_med,
    fmt="^",
    mec="white", mfc="none",
    ecolor="white", color="white",
    capsize=3,
    label="Median"
)


# ---------------------------
# [SII]/Hα（緑ダイヤ）
# ---------------------------
ax.errorbar(
    x, y_Ha,
    yerr=yerr_Ha,
    fmt="D",
    mec="white", mfc="none",
    ecolor="white", color="white",
    capsize=3,
    label="[SII]6717 / Hα"
)

# ---------------------------
# Hα normalized median（黄色などにすると見やすい）
# ---------------------------
ax.errorbar(
    x, y_Ha_med,
    yerr=yerr_Ha_med,
    fmt="v",
    mec="white", mfc="none",
    ecolor="white", color="white",
    capsize=3,
    label="Median ([SII]/Hα)"
)

ax.set_xlabel(r"$\log(sSFR) [\mathrm{yr^{-1}}]$") 
ax.set_ylabel(r"[SII] 6717 / 6731")
ax.set_xlim(-14, -7)
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
xbins = np.arange(-14, -6.9, 0.01)
ybins = np.arange(0.5, 2.1, 0.01)

count_map, xedge, yedge, _ = (
    binned_statistic_2d(
        df.loc[m_complete, "ssfr_MEDIAN"],
        df.loc[m_complete, "R_SII"],
        values=None,
        statistic="count",
        bins=[xbins, ybins]
    )
)

fig, ax = plt.subplots(figsize=(8,6))
fig.subplots_adjust(left=0.15, right=0.85, bottom=0.15, top=0.85)

vmin = 0 # 下限を5パーセンタイルに設定（必要に応じて調整）
vmax = 50 # 上限を95パーセンタイルに設定（必要に応じて調整）

plt.pcolormesh(
    xedge,
    yedge,
    count_map.T,
    vmin=vmin, vmax=vmax,  # カラーマップの範囲を固定（必要に応じて調整）
    shading="auto",
    cmap="viridis" # 必要に応じて調整
)

plt.colorbar()

# weighted
ax.errorbar(
    x, y_w, yerr=yerr_w,
    fmt="s", mec="white", mfc="None",
    ecolor="white", color="white",
    capsize=3, label="Weighted"
)

# # mean: 削除した方が良い(値が壊れるため)
# ax.errorbar(
#     x, y_mean, yerr=yerr_mean,
#     fmt="o", mec="red", mfc="none",
#     ecolor="red", color="red",
#     capsize=3, label="Mean"
# )

# median
ax.errorbar(
    x, y_med, yerr=yerr_med,
    fmt="^", mec="white", mfc="none",
    ecolor="white", color="white",
    capsize=3, label="Median"
)

# [SII]/Hα weighted mean
ax.errorbar(
    x, y_Ha, yerr=yerr_Ha,
    fmt="D", mec="white", mfc="none",
    ecolor="white", color="white",
    capsize=3, label="[SII]6717 / Hα"
)

# [SII]/Hα median
ax.errorbar(
    x, y_Ha_med, yerr=yerr_Ha_med,
    fmt="v",
    mec="white", mfc="none",
    ecolor="white", color="white",
    capsize=3
)

ax.set_xlabel(r"$\log(sSFR) [\mathrm{yr^{-1}}]$") 
ax.set_ylabel(r"[SII] 6717 / 6731")
ax.set_xlim(-14, -7.0)
ax.set_ylim(1.0,1.6)

for spine in ax.spines.values():
    spine.set_linewidth(2)
plt.tight_layout()
# 保存
fig_dir = os.path.join(current_dir, "results/figure")
os.makedirs(fig_dir, exist_ok=True)
save_path_count = os.path.join(
    fig_dir,
    "heat_ssfr_sii_ratio_count_sdss.png"
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

    lo = row["logsSFR_lo"]
    hi = row["logsSFR_hi"]

    ax = axes[i]

    # 同じbinのデータ取り出し
    ssfr_bin = (
        m_complete &
        (df["ssfr_MEDIAN"] >= lo) &
        (df["ssfr_MEDIAN"] < hi)
    )

    f1 = F6716[ssfr_bin]
    f2 = F6731[ssfr_bin]

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
    ax.axvline(row["R_wmed"], color="k", linestyle="-", lw=2, label="Weighted")
    ax.axvline(row["R_med"], color="k", linestyle="--", lw=2, label="Median")
    ax.axvline(row["R_Ha"], color="k", linestyle=":", lw=2, label="Weighted (Hα norm)")
    ax.axvline(row["R_Ha_med"], color="k", linestyle="-.", lw=2, label="Median (Hα norm)")

    ax.text(
        0.02, 0.95,
        f"{lo:.1f}–{hi:.1f} N={int(row['N'])}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=20
    )

    ax.set_xlim(1.0, 1.8)

    ax.tick_params(
        top=False, right=False,   # 上・右を消す
        bottom=True, left=True    # 下・左を有効
    )
    ax.tick_params(
        axis='both',      # x, y 両方
        which='major',    # major ticks
        length=8,         # 長さ
        width=1.5,        # 線の太さ
        direction='in'   # 方向（out / in / inout）
    )
    
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
    "results/figure/stacked_sii_histograms_ssfr.png"
)
plt.savefig(hist_path, dpi=200)

print("Saved:", hist_path)

plt.show()