#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
スクリプトの概要:
logSFR ビンごとに
  [SII]6717,6731 フラックスをスタック
→ MCで ratio 分布
→ PyNebで ne 分布
→ P16, P50, P84 を保存・描画
→ 完全なサンプルのみを対象とする


使用方法:
    stacked_sii_ne_vs_sfr_v2.py [オプション]

著者: A. M.
作成日: 2026-02-26

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
# fits_path = os.path.join(current_dir, "results/JADES/JADES_DR3/data_from_Nishigaki/jades_info_crossmatch.csv")
csv_path = "./results/JADES/JADES_DR3/data_from_Nishigaki/jades_info_crossmatch_with_logSFR.csv"

out_csv = os.path.join(current_dir, "results/table/stacked_sii_ratio_vs_sfr_JADES_DR3.csv")
out_png = os.path.join(current_dir, "results/figure/stacked_sii_ratio_vs_sfr_JADES_DR3.png")

os.makedirs(os.path.dirname(out_csv), exist_ok=True)
os.makedirs(os.path.dirname(out_png), exist_ok=True)

# ==========================================
# パラメータ
# ==========================================
BIN_WIDTH = 0.1 # 変更
NMIN = 1          # スタックに含める最小データ数
N_MC = 5000
# Lcut = 1e39     # 完全サンプル条件（使うなら下で有効化）

# 単位スケール（カタログの単位に合わせて調整）
# UNIT_FLUX = 1e-17  # 例: MPA-JHU などの慣例
UNIT_FLUX = 1e-20    # いまの設定を踏襲

# z-bin の定義（左開右閉ではなく、ここでは「(lo, hi]」を採用しないよう注意）
# 今回は (1,4), (4,7), (>7) を色分け
Z_BINS = [
    dict(name="1<z<4", color="tab:blue",  lo=1.0, hi=4.0,  inclusive="(,)"),
    dict(name="4<z<7", color="tab:green", lo=4.0, hi=7.0,  inclusive="(,)"),
    dict(name="z>7",   color="tab:red",   lo=7.0, hi=np.inf, inclusive="(,]"),  # hi=inf
]
# 参考：inclusive の意味
# "(,)"   :   lo <  z <  hi
# "[,)"   :  lo <= z <  hi
# "(,]"   :   lo <  z <= hi
# ここは慣例的に開区間 "(,)" を使用（境界の二重カウント回避）

# 質量ビンの作り方：
#   'per_z'  : z-bin ごとに最小～最大から edges を作る（ユーザの意図）
#   'global' : 全体の最小～最大から共通 edges を作る（比較しやすい）
MASS_BIN_MODE = 'per_z'  # 'per_z' または 'global'

# ==========================================
# 読み込み
# ==========================================
# tab = Table.read(fits_path, hdu=1)
# df = tab.to_pandas()
df = pd.read_csv(csv_path)

# ==========================================
# 基本量の計算
# ==========================================
z = df["z_spec"].values

# SII line fluxes（列名はユーザの現状に合わせる）
# F6716 = df["S2_6718_flux"].values * UNIT_FLUX
# F6731 = df["S2_6733_flux"].values * UNIT_FLUX
# err6716 = df["S2_6718_err"].values * UNIT_FLUX
# err6731 = df["S2_6733_err"].values * UNIT_FLUX
F6716 = df["S2_6718_flux"].values * UNIT_FLUX
F6731 = df["S2_6733_flux"].values * UNIT_FLUX
err6716 = df["S2_6718_err"].values * UNIT_FLUX
err6731 = df["S2_6733_err"].values * UNIT_FLUX
sn6716 = F6716 / err6716
sn6731 = F6731 / err6731

# luminosity
d_L = cosmo.luminosity_distance(z).to(u.cm).value
L6716 = 4 * np.pi * d_L**2 * F6716
L6731 = 4 * np.pi * d_L**2 * F6731

# [SII] ratio
df["S2_ratio"] = F6716 / F6731

# ==========================================
# マスク定義
# ==========================================
def valid_sfr(x):
    x = np.asarray(x, float)
    m = np.isfinite(x)
    # 極端な outlier だけ落とす（-5<log(SFR)<3ならまず間違いない→-1が異常値になっていないか？）
    m &= (x > -5) & (x < 3) & (x != -1.0)
    return m

m_sii = (
    np.isfinite(F6716) & np.isfinite(F6731) &
    np.isfinite(err6716) & np.isfinite(err6731) &
    (err6716 > 0) & (err6731 > 0)
)

# m_sfr = valid_sfr(df["sfr_MEDIAN"])
# m_ratio = np.isfinite(df["R_SII"])
# m_sfr = valid_sfr(df["sfr_MEDIAN"])
m_sfr = valid_sfr(df["logSFR_hb"])
m_ratio = np.isfinite(df["logSFR_hb_err_high"])
m_sfr = valid_sfr(df["logSFR_hb_err_low"])

mask_all = m_sii & m_sfr & m_ratio

# 完全サンプル
# m_complete = mask_all & (L6716 >= Lcut) & (L6731 >= Lcut)
m_complete = mask_all 
# m_incomplete = mask_all & (~m_complete)

# ==========================================
# ビン作成
# ==========================================
# logSFR = df.loc[m_complete, "sfr_MEDIAN"].values
logSFR = df.loc[m_complete, "SFR_hb"].values

edges = np.arange(
    np.floor(logSFR.min()/BIN_WIDTH)*BIN_WIDTH,
    np.ceil(logSFR.max()/BIN_WIDTH)*BIN_WIDTH + BIN_WIDTH,
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

    m_bin = (
        m_complete &
        (df["logSFR_hb"] >= lo) &
        (df["logSFR_hb"] < hi)
    )

    N = np.sum(m_bin)
    if N < NMIN:
        continue

    f1 = F6716[m_bin]
    e1 = err6716[m_bin]
    f2 = F6731[m_bin]
    e2 = err6731[m_bin]

    F1, e1_stack = weighted_mean(f1, e1)
    F2, e2_stack = weighted_mean(f2, e2)

    # Monte Carlo for ratio
    f1_mc = rng.normal(F1, e1_stack, N_MC)
    f2_mc = rng.normal(F2, e2_stack, N_MC)

    valid = (f2_mc > 0)
    R_mc = f1_mc[valid] / f2_mc[valid]

    R50 = np.nanmedian(R_mc)
    R16 = np.nanpercentile(R_mc, 16)
    R84 = np.nanpercentile(R_mc, 84)

    rows.append(dict(
        logSFR_lo=lo,
        logSFR_hi=hi,
        logSFR_cen = 0.5*(lo+hi),
        N = N,
        # F6717=F1, # 構文が違う
        # F6717_err=e1,
        # F6731=F2,
        # F6731_err=e2,
        R_med = R50,
        R_err_lo = R50 - R16,
        R_err_hi = R84 - R50,
        N_MC_valid=int(valid.sum())
    ))

res = pd.DataFrame(rows)
res.to_csv(out_csv, index=False)
print("Saved:", out_csv)

# ==========================================
# 描画
# ==========================================
fig, ax = plt.subplots(figsize=(6,6))

# # 不完全（薄グレー）
# ax.scatter(
#     df.loc[m_incomplete, "sfr_MEDIAN"],
#     df.loc[m_incomplete, "R_SII"],
#     s=0.01,
#     marker='.',
#     alpha=0.8,
#     color="gray",
# )

# 完全（青）
ax.scatter(
    # df.loc[m_complete, "sfr_MEDIAN"],
    df.loc[m_complete, "logSFR_hb"],
    df.loc[m_complete, "S2_ratio"],
    s=0.01,
    marker='.',
    alpha=0.8,
    color="C0",
)

# stack結果（完全なものとそうでないものの色を分ける）
thr = 0.0

x = res["logSFR_cen"].values
y = res["R_med"].values
yerr = np.vstack([res["R_err_lo"].values, res["R_err_hi"].values])

mask_lt = x < thr
mask_ge = ~mask_lt

# x < 0（白四角・黒縁）
ax.errorbar(
    x[mask_lt], y[mask_lt],
    yerr=yerr[:, mask_lt],
    fmt="s", mec="black", mfc="white",
    ecolor="k", color="k",  # 誤差線色/線色（同時指定）
    capsize=3, label=f"x < {thr}"
)

# x >= 0（黒四角）
ax.errorbar(
    x[mask_ge], y[mask_ge],
    yerr=yerr[:, mask_ge],
    fmt="s", mec="black", mfc="black",
    ecolor="k", color="k",
    capsize=3, label=f"x ≥ {thr}"
)


ax.set_xlabel(r"$\log(SFR)\ [M_{\odot}\mathrm{yr^{-1}}]$")
ax.set_ylabel(r"[SII] 6717 / 6731")
ax.set_xlim(0, 2)
ax.set_ylim(0.5,2.0)

for spine in ax.spines.values():
    spine.set_linewidth(2)

# ax.legend(frameon=False)

plt.tight_layout()
plt.savefig(out_png, dpi=200)
plt.show()

print("Saved:", out_png)