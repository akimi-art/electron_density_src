#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
スクリプトの概要:
logM* ビンごとに
  [SII]6717,6731 フラックスをスタック
→ MCで ratio 分布
→ PyNebで ne 分布
→ P16, P50, P84 を保存・描画


使用方法:
    stacked_sii_ne_vs_mass.py [オプション]

著者: A. M.
作成日: 2026-02-01

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

# -----------------------
# 軸の設定
# -----------------------
plt.rcParams.update({
    "figure.figsize": (12, 6),
    "font.size": 16,
    "axes.labelsize": 18,
    "axes.titlesize": 18,
    "axes.grid": False,

    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,

    "xtick.major.size": 16,
    "ytick.major.size": 16,
    "xtick.major.width": 2,
    "ytick.major.width": 2,

    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.minor.size": 8,
    "ytick.minor.size": 8,
    "xtick.minor.width": 1.5,
    "ytick.minor.width": 1.5,

    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
})

# -----------------------
# 入出力
# -----------------------
current_dir = os.getcwd()
fits_path = os.path.join(current_dir, "results/fits/mpajhu_dr7_v5_2_merged.fits")
out_csv   = os.path.join(current_dir, "results/table/stacked_sii_ratio_vs_mass.csv")
out_png   = os.path.join(current_dir, "results/figure/stacked_sii_ratio_vs_mass_1.1_1.5.png")

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
def valid_mass(x):
    x = np.asarray(x, float)
    m = np.isfinite(x)
    m &= (x > 0) & (x < 13)
    return m

m_sii = (
    np.isfinite(df["SII_6717_FLUX"]) &
    np.isfinite(df["SII_6731_FLUX"]) &
    np.isfinite(df["SII_6717_FLUX_ERR"]) &
    np.isfinite(df["SII_6731_FLUX_ERR"]) &
    (df["SII_6717_FLUX_ERR"] > 0) &
    (df["SII_6731_FLUX_ERR"] > 0)
)

m_sm = valid_mass(df["sm_MEDIAN"])
mask = m_sii & m_sm

# -----------------------
# ビン作成
# -----------------------
logM = df.loc[mask, "sm_MEDIAN"].to_numpy()

edges = np.arange(
    np.floor(logM.min()/BIN_WIDTH)*BIN_WIDTH,
    np.ceil(logM.max()/BIN_WIDTH)*BIN_WIDTH + BIN_WIDTH,
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
        (df["sm_MEDIAN"] >= lo) &
        (df["sm_MEDIAN"] <  hi)
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
        logM_lo=lo,
        logM_hi=hi,
        logM_cen=0.5*(lo+hi),
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
    res["logM_cen"],
    res["R_med"],
    yerr=[res["R_err_lo"], res["R_err_hi"]],
    fmt="o",
    capsize=2
)

ax.set_xlabel(r"log $M_\star$ [M$_\odot$]")
ax.set_ylabel(r"[SII] 6717 / 6731 (stacked)")
ax.set_ylim(1.1, 1.5)
# === 枠線 (spines) の設定 ===
# 線の太さ・色・表示非表示などを個別に制御
for spine in ax.spines.values():
    spine.set_linewidth(2)       # 枠線の太さ
    spine.set_color("black")     # 枠線の色
plt.tight_layout()
plt.savefig(out_png, dpi=200)
plt.show()

print("Saved:", out_png)