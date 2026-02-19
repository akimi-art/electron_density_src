#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
スクリプトの概要:
Strong line ratioビンごとに
  [SII]6717,6731 フラックスをスタック
→ MCで ratio 分布
→ PyNebで ne 分布
→ P16, P50, P84 を保存・描画


使用方法:
    stacked_sii_ne_vs_strong_line_ratio.py [オプション]

著者: A. M.
作成日: 2026-02-18

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
# 入出力
# ==========================================
current_dir = os.getcwd()
fits_path = os.path.join(
    current_dir,
    "results/fits/mpajhu_dr7_v5_2_merged_L6717_ge_4pi_dL2_1e-17_L6731_ge_4pi_dL2_1e-17_z0.00-0.40.fits"
)
out_csv = os.path.join(
    current_dir,
    "results/table/stacked_sii_ratio_vs_strong_line_ratio.csv"
)
out_png = os.path.join(
    current_dir,
    "results/figure/stacked_sii_ratio_vs_strong_line_ratio.png"
)

os.makedirs(os.path.dirname(out_csv), exist_ok=True)
os.makedirs(os.path.dirname(out_png), exist_ok=True)

# ==========================================
# 読み込み
# ==========================================
tab = Table.read(fits_path, hdu=1)
df = tab.to_pandas()

UNIT_FLUX = 1e-17

F6716 = df["SII_6717_FLUX"].values * UNIT_FLUX
F6731 = df["SII_6731_FLUX"].values * UNIT_FLUX
err6716 = df["SII_6717_FLUX_ERR"].values * UNIT_FLUX
err6731 = df["SII_6731_FLUX_ERR"].values * UNIT_FLUX

df["R_SII"] = F6716 / F6731

# ==========================================
# マスク
# ==========================================
m_sii = (
    np.isfinite(F6716) & np.isfinite(F6731) &
    np.isfinite(err6716) & np.isfinite(err6731) &
    (err6716 > 0) & (err6731 > 0)
)

m_ratio = np.isfinite(df["R_SII"])
mask_base = m_sii & m_ratio

# ==========================================
# スタック用関数
# ==========================================
def weighted_mean(flux, err):
    flux = np.asarray(flux)
    err = np.asarray(err)
    w = 1.0 / err**2
    mu = np.sum(w * flux) / np.sum(w)
    sigma = np.sqrt(1.0 / np.sum(w))
    return mu, sigma

rng = np.random.default_rng()

# ==========================================
# 設定
# ==========================================
strong_lines = ["R2", "R3", "O32", "R23", "N2", "O3N2"]

BIN_WIDTH = 0.1
NMIN = 100
N_MC = 5000
XMIN, XMAX = -3.0, 3.0
edges = np.arange(XMIN, XMAX + BIN_WIDTH, BIN_WIDTH)
mask_all = m_sii & m_ratio

# ==========================================
# 解析 & 保存用
# ==========================================
all_results = []

fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(2, 3, wspace=0.00, hspace=0.00)
axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]

for ax, line in zip(axes, strong_lines):

    m_line = np.isfinite(df[line])
    m_complete = mask_all & m_line

    # =========================
    # ★ ここに入れる ★
    # =========================
    ax.scatter(
        df.loc[m_complete, line],        # x = strong line
        df.loc[m_complete, "R_SII"],     # y = 個別SII ratio
        s=0.01,
        alpha=0.8,
        color="C0",
        rasterized=True
    )

    m_line = np.isfinite(df[line])
    mask = mask_base & m_line

    stack_results = []

    for lo, hi in zip(edges[:-1], edges[1:]):

        m_bin = mask & (df[line] >= lo) & (df[line] < hi)
        N = int(np.sum(m_bin))
        if N < NMIN:
            continue

        f6717 = F6716[m_bin]
        e6717 = err6716[m_bin]
        f6731 = F6731[m_bin]
        e6731 = err6731[m_bin]

        F1, e1 = weighted_mean(f6717, e6717)
        F2, e2 = weighted_mean(f6731, e6731)

        if not np.isfinite(F1) or not np.isfinite(F2):
            continue

        # MC
        f1_mc = rng.normal(F1, e1, N_MC)
        f2_mc = rng.normal(F2, e2, N_MC)

        valid = np.isfinite(f1_mc) & np.isfinite(f2_mc) & (f2_mc > 0)
        if valid.sum() < 50:
            continue

        R_mc = f1_mc[valid] / f2_mc[valid]
        R16, R50, R84 = np.percentile(R_mc, [16, 50, 84])

        stack_results.append(dict(
            line=line,
            xcen=0.5*(lo+hi),
            N=N,
            R_med=R50,
            R_lo=R50-R16,
            R_hi=R84-R50
        ))

    if len(stack_results) == 0:
        continue

    res = pd.DataFrame(stack_results)

    # 保存用に追加
    all_results.extend(stack_results)

    # 描画（stackのみ）
    ax.errorbar(
        res["xcen"],
        res["R_med"],
        yerr=[res["R_lo"], res["R_hi"]],
        fmt="s",
        capsize=2,
        color="black"
    )

    ax.set_xlim(XMIN, XMAX)
    ax.set_ylim(0.5, 2.0)

    for spine in ax.spines.values():
        spine.set_linewidth(2)
        spine.set_color("black")

    ax.text(
        0.05, 0.95,
        line,
        transform=ax.transAxes,
        verticalalignment="top"
    )

# 軸整理
for ax in axes[:3]:
    ax.tick_params(labelbottom=False)

for ax in (axes[1], axes[2], axes[4], axes[5]):
    ax.tick_params(labelleft=False)

axes[0].set_ylabel("[SII] 6717 / 6731")
axes[3].set_ylabel("[SII] 6717 / 6731")

plt.tight_layout()
plt.savefig(out_png, dpi=200)
plt.show()

# ==========================================
# CSV保存（全lineまとめて）
# ==========================================
df_out = pd.DataFrame(all_results)
df_out.to_csv(out_csv, index=False)
print("Saved:", out_csv)