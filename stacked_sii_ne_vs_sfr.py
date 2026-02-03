#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
スクリプトの概要:
logSFR ビンごとに
  [SII]6717,6731 フラックスをスタック
→ MCで ratio 分布
→ PyNebで ne 分布
→ P16, P50, P84 を保存・描画


使用方法:
    stacked_sii_ne_vs_sfr.py [オプション]

著者: A. M.
作成日: 2026-02-03

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


# -----------------------
# 入出力
# -----------------------
current_dir = os.getcwd()
fits_path = os.path.join(current_dir, "results/fits/mpajhu_dr7_v5_2_merged.fits")
out_csv   = os.path.join(current_dir, "results/table/stacked_sii_ratio_vs_sfr.csv")
out_png   = os.path.join(current_dir, "results/figure/stacked_sii_ratio_vs_sfr.png")

os.makedirs(os.path.dirname(out_csv), exist_ok=True)
os.makedirs(os.path.dirname(out_png), exist_ok=True)

# -----------------------
# パラメータ
# -----------------------
BIN_WIDTH = 0.1      # dex
NMIN      = 100
N_MC      = 5000

# -----------------------
# 読み込み
# -----------------------
tab = Table.read(fits_path, hdu=1)
df  = tab.to_pandas()

# -----------------------
# マスク
# -----------------------
def valid_sfr(x):
    x = np.asarray(x, float)
    m = np.isfinite(x)
    # 極端な outlier だけ落とす（-5<log(SFR)<5ならまず間違いない）
    m &= (x > -5) & (x < 5)
    return m


m_sii = (
    np.isfinite(df["SII_6717_FLUX"]) &
    np.isfinite(df["SII_6731_FLUX"]) &
    np.isfinite(df["SII_6717_FLUX_ERR"]) &
    np.isfinite(df["SII_6731_FLUX_ERR"]) &
    (df["SII_6717_FLUX_ERR"] > 0) &
    (df["SII_6731_FLUX_ERR"] > 0)
)

m_sfr = valid_sfr(df["sfr_MEDIAN"])
mask = m_sii & m_sfr

# -----------------------
# ビン作成
# -----------------------
logSFR = df.loc[mask, "sfr_MEDIAN"].to_numpy()

edges = np.arange(
    np.floor(logSFR.min()/BIN_WIDTH)*BIN_WIDTH,
    np.ceil(logSFR.max()/BIN_WIDTH)*BIN_WIDTH + BIN_WIDTH,
    BIN_WIDTH
)

# -----------------------
# ユーティリティ
# -----------------------
def weighted_mean(flux, err, err_floor_frac=0.05):
    flux = np.asarray(flux, float)
    err  = np.asarray(err,  float)

    m = np.isfinite(flux) & np.isfinite(err) & (err > 0)
    flux = flux[m]
    err  = err[m]

    if flux.size == 0:
        return np.nan, np.nan

    # error floor（SDSSでは必須）
    floor = err_floor_frac * np.nanmedian(err)
    err = np.maximum(err, floor)

    w = 1.0 / err**2
    mu = np.sum(w * flux) / np.sum(w)
    sigma = np.sqrt(1.0 / np.sum(w))
    return mu, sigma

def percentile_summary(x):
    x = np.asarray(x, float)
    # NaN/inf を落とす（All-NaN 対策）
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan, np.nan
    p16, p50, p84 = np.nanpercentile(x, [16, 50, 84])
    return p50, p50 - p16, p84 - p50

# 乱数（再現性が欲しいなら seed を固定）
rng = np.random.default_rng()

# -----------------------
# メインループ（ne 関連は一切なし）
# -----------------------
rows = []

for lo, hi in zip(edges[:-1], edges[1:]):

    m_bin = (
        mask &
        (df["sfr_MEDIAN"] >= lo) &
        (df["sfr_MEDIAN"] <  hi)
    )

    N = int(np.sum(m_bin))
    if N < NMIN:
        continue

    f6717 = df.loc[m_bin, "SII_6717_FLUX"].to_numpy()
    e6717 = df.loc[m_bin, "SII_6717_FLUX_ERR"].to_numpy()
    f6731 = df.loc[m_bin, "SII_6731_FLUX"].to_numpy()
    e6731 = df.loc[m_bin, "SII_6731_FLUX_ERR"].to_numpy()

    # フラックススタック
    F1, e1 = weighted_mean(f6717, e6717)
    F2, e2 = weighted_mean(f6731, e6731)

    if not (np.isfinite(F1) and np.isfinite(F2) and np.isfinite(e1) and np.isfinite(e2)):
        continue
    if (e1 <= 0) or (e2 <= 0):
        continue

    # MC（比 R のみ）
    f1_mc = rng.normal(F1, e1, N_MC)
    f2_mc = rng.normal(F2, e2, N_MC)

    # f2 が 0/負になると比が発散するので除外
    valid = np.isfinite(f1_mc) & np.isfinite(f2_mc) & (f2_mc > 0)
    if valid.sum() < 10:
        # 有効サンプルが少なすぎるビンはスキップ
        continue

    R_mc = f1_mc[valid] / f2_mc[valid]

    # 要約統計（比）
    R50, Rlo, Rhi = percentile_summary(R_mc)

    rows.append(dict(
        logSFR_lo=lo,
        logSFR_hi=hi,
        logSFR_cen=0.5*(lo+hi),
        N=N,
        F6717=F1,
        F6717_err=e1,
        F6731=F2,
        F6731_err=e2,
        R_med=R50,
        R_err_lo=Rlo,
        R_err_hi=Rhi,
        N_MC_valid=int(valid.sum())
    ))

# -----------------------
# 保存
# -----------------------
res = pd.DataFrame(rows)
res.to_csv(out_csv, index=False)
print("Saved:", out_csv)

# -----------------------
# プロット（比のみ）
# -----------------------
fig, ax = plt.subplots(figsize=(10, 6))
ax.errorbar(
    res["logSFR_cen"],
    res["R_med"],
    yerr=[res["R_err_lo"], res["R_err_hi"]],
    fmt="o",
    capsize=2
)

ax.set_xlabel(r"$\log(SFR)\ [M_{\odot}\mathrm{yr^{-1}}]$")
ax.set_ylabel(r"[SII] 6717 / 6731 (stacked)")
# === 枠線 (spines) の設定 ===
# 線の太さ・色・表示非表示などを個別に制御
for spine in ax.spines.values():
    spine.set_linewidth(2)       # 枠線の太さ
    spine.set_color("black")     # 枠線の色
plt.tight_layout()
plt.savefig(out_png, dpi=200)
plt.show()

print("Saved:", out_png)