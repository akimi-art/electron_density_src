#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
スクリプトの概要:
統合fitsファイルを用いて、
SII6716/6731のratioと
Stellar Mass, SFR, Metallcity, sSFR（logスケール, 
おそらくファイル内ですでにlogになっている）
のプロットをします。


使用方法:
    sii_ratio_vs_sm_sfr_metallicity_ssfr_plot.py [オプション]

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
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u


# 軸の設定
plt.rcParams.update({
    # --- 図全体 ---
    "figure.figsize": (12, 6),       # 図サイズ
    "font.size": 16,                 # 全体フォントサイズ
    "axes.labelsize": 18,            # 軸ラベルのサイズ
    "axes.titlesize": 18,            # タイトルのサイズ
    "axes.grid": False,              # グリッドOFF

    # --- 目盛り設定 (ticks) ---
    "xtick.direction": "in",         # x軸目盛りの向き
    "ytick.direction": "in",         # y軸目盛りの向き
    "xtick.top": False,               # 上にも目盛り
    "ytick.right": False,             # 右にも目盛り

    # 主目盛り（major ticks）
    "xtick.major.size": 16,          # 長さ
    "ytick.major.size": 16,
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
    "xtick.labelsize": 14,           # x軸ラベルサイズ
    "ytick.labelsize": 14,           # y軸ラベルサイズ
})

# === 必要なモジュールのインポート ===
import os
import numpy as np
import pandas as pd
from astropy.table import Table
import matplotlib.pyplot as plt


current_dir = os.getcwd()
fits_path = os.path.join(current_dir, "results/fits/mpajhu_dr7_v5_2_merged.fits")   # 適宜パスを変更
out_png   = os.path.join(current_dir, "results/figure/sii_ratio_vs_properties.png")

# ---- 読み込み（Table->pandas） ----
tab = Table.read(fits_path, hdu=1)
df = tab.to_pandas()

# ---- 欠損・無効値マスク ----
# SII flux は 0以下や NaN を除外（ratioにするので）
m_sii = (
    np.isfinite(df["SII_6717_FLUX"]) & np.isfinite(df["SII_6731_FLUX"]) &
    np.isfinite(df["SII_6717_FLUX_ERR"]) & np.isfinite(df["SII_6731_FLUX_ERR"]) 
    # (df["SII_6717_FLUX"] > 0) & (df["SII_6731_FLUX"] > 0)
)

# # 物理量の欠損（-99.9や-1）を除外するマスクを作る関数
# def valid_col(x, bad_threshold=-90):
#     return np.isfinite(x) & (x > bad_threshold)

# m_sm  = valid_col(df["sm_MEDIAN"], bad_threshold=-90)
# m_sfr = valid_col(df["sfr_MEDIAN"], bad_threshold=-90)
# m_oh  = valid_col(df["oh_MEDIAN"], bad_threshold=-90)

# m_ssfr = m_sm & m_sfr


def valid_mass(x):
    x = np.asarray(x, dtype=float)
    m = np.isfinite(x)
    m &= (x != -1.0) & (x != -99.9)
    m &= (x > 0) & (x < 13)   # logM★は0〜13の範囲のみ採用
    return m

def valid_oh(x):
    x = np.asarray(x, dtype=float)
    m = np.isfinite(x)
    m &= (x != -99.9)
    m &= (x > 7) & (x < 10) # ここがバイアスになるかもしれないので注意
    return m

def valid_sfr(x, bad_values=(-1.0, -99.9)):
    x = np.asarray(x, dtype=float)
    m = np.isfinite(x)
    for bv in bad_values:
        m &= (x != bv)

    # ロバスト外れ値除去：巨大負値（-1e5など）を弾く
    xm = x[m]
    med = np.nanmedian(xm)
    mad = np.nanmedian(np.abs(xm - med))
    sigma = 1.4826 * mad
    m &= (np.abs(x - med) < 8 * sigma)  # ゆるい 8σ クリップ（実質欠損だけ除去）

    return m

m_sm  = valid_mass(df["sm_MEDIAN"])
m_sfr = valid_sfr(df["sfr_MEDIAN"])
m_oh  = valid_oh(df["oh_MEDIAN"])

m_ssfr = m_sm & m_sfr


# ---- ratio と誤差（単純誤差伝播：まず全体像用） ----
R = df["SII_6717_FLUX"] / df["SII_6731_FLUX"]
# error propagation for ratio:
Rerr = R * np.sqrt(
    (df["SII_6717_FLUX_ERR"] / df["SII_6717_FLUX"])**2 +
    (df["SII_6731_FLUX_ERR"] / df["SII_6731_FLUX"])**2
)

df["R_SII"] = R
df["R_SII_ERR"] = Rerr

# ---- 描画用：極端値を抑える（任意） ----
# SII比の物理的な典型範囲は ~0.4-1.45 付近（Te~1e4K）なので、見やすさのために範囲で表示
# しかし, 最初の段階では強く制限しない。
ymin, ymax = -40, 40

# ---- ビン中央値を計算する関数（ロバスト） ----
def binned_median(x, y, bins=12, x_min=None, x_max=None):
    x = np.asarray(x); y = np.asarray(y)
    if x_min is None: x_min = np.nanpercentile(x, 1)
    if x_max is None: x_max = np.nanpercentile(x, 99)
    edges = np.linspace(x_min, x_max, bins+1)
    xc, ym, y16, y84, n = [], [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x < hi) & np.isfinite(y)
        if np.sum(m) < 30:   # 少なすぎるbinはスキップ（好みで調整）
            continue
        yy = y[m]
        xc.append(0.5*(lo+hi))
        ym.append(np.nanmedian(yy))
        y16.append(np.nanpercentile(yy, 16))
        y84.append(np.nanpercentile(yy, 84))
        n.append(np.sum(m))
    return np.array(xc), np.array(ym), np.array(y16), np.array(y84), np.array(n)

# ---- 4パネルプロット ----
fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharey=True)
axes = axes.ravel()

# Panel 1: vs stellar mass (logM*)
m = m_sii & m_sm & np.isfinite(df["R_SII"])
x = df.loc[m, "sm_MEDIAN"].to_numpy()
y = df.loc[m, "R_SII"].to_numpy()
axes[0].scatter(x, y, s=2, alpha=0.08, rasterized=True)
xc, ym, y16, y84, n = binned_median(x, y, bins=14)
# axes[0].errorbar(xc, ym, yerr=[ym-y16, y84-ym], fmt="o", ms=4, color="crimson", capsize=2, label="binned median (16–84%)")
axes[0].set_xlabel(r"log $M_\star$ [M$_\odot$]")
axes[0].set_ylabel(r"[S II] 6717/6731")
# axes[0].legend(frameon=False, fontsize=9)

# Panel 2: vs SFR (logSFR)
m = m_sii & m_sfr & np.isfinite(df["R_SII"])
x = df.loc[m, "sfr_MEDIAN"].to_numpy()
y = df.loc[m, "R_SII"].to_numpy()
axes[1].scatter(x, y, s=2, alpha=0.08, rasterized=True)
xc, ym, y16, y84, n = binned_median(x, y, bins=14)
# axes[1].errorbar(xc, ym, yerr=[ym-y16, y84-ym], fmt="o", ms=4, color="crimson", capsize=2)
axes[1].set_xlabel(r"log SFR [M$_\odot$ yr$^{-1}$]")

# Panel 3: vs metallicity (12+log(O/H) assumed)
m = m_sii & m_oh & np.isfinite(df["R_SII"])
x = df.loc[m, "oh_MEDIAN"].to_numpy()
y = df.loc[m, "R_SII"].to_numpy()
axes[2].scatter(x, y, s=2, alpha=0.08, rasterized=True)
xc, ym, y16, y84, n = binned_median(x, y, bins=14)
# axes[2].errorbar(xc, ym, yerr=[ym-y16, y84-ym], fmt="o", ms=4, color="crimson", capsize=2)
axes[2].set_xlabel(r"Metallicity (fiber)")

# Panel 4: vs sSFR (log sSFR = logSFR - logM*)
m = m_sii & m_ssfr & np.isfinite(df["R_SII"])
x = (df.loc[m, "sfr_MEDIAN"] - df.loc[m, "sm_MEDIAN"]).to_numpy()
y = df.loc[m, "R_SII"].to_numpy()
axes[3].scatter(x, y, s=2, alpha=0.08, rasterized=True)
xc, ym, y16, y84, n = binned_median(x, y, bins=14)
# axes[3].errorbar(xc, ym, yerr=[ym-y16, y84-ym], fmt="o", ms=4, color="crimson", capsize=2)
axes[3].set_xlabel(r"log sSFR [yr$^{-1}$]")

# y-axis settings
for ax in axes:
    ax.set_ylim(ymin, ymax)
    ax.grid(alpha=0.2)

plt.tight_layout()
plt.savefig(out_png, dpi=200)
plt.show()
# plt.close()
print("Saved:", out_png)


# col = "oh_MEDIAN"
# x = df[col].to_numpy()

# print("min/max:", np.nanmin(x), np.nanmax(x))
# print("percentiles:", np.nanpercentile(x[np.isfinite(x)], [0, 0.1, 1, 5, 50, 95, 99, 99.9, 100]))

# # 負値だけ見てみる
# neg = x[np.isfinite(x) & (x < 0)]
# print("n(neg) =", neg.size)
# print("unique negatives (first 20):", np.unique(neg)[:20])

m = m_sii & m_sm   # あなたが実際に使っているマスクに合わせる
x = df.loc[m, "sm_MEDIAN"].astype(float).to_numpy()

print("x min/max:", np.nanmin(x), np.nanmax(x))
print("x percentiles:", np.nanpercentile(x[np.isfinite(x)], [0.1, 1, 5, 50, 95, 99, 99.9]))
print("count <6:", np.sum(x < 6))
print("count >12:", np.sum(x > 12))