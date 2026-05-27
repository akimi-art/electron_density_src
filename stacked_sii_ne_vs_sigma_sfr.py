# -*- coding: utf-8 -*-
"""
スクリプトの概要:
Re ビンごとに
  [SII]6717,6731 フラックスをスタック
→ MCで ratio 分布
→ PyNebで ne 分布
→ P16, P50, P84 を保存・描画
→ 完全なサンプルのみを対象とする


使用方法:
    stacked_sii_ne_vs_sigma_sfr.py [オプション]

著者: A. M.
作成日: 2026-02-16

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
fits_path = os.path.join(current_dir, "results/fits/mpajhu_dr7_v5_2_merged_zlt0.2_Lgt1e+39_radius.fits")

out_csv = os.path.join(current_dir, "results/csv/stacked_sii_ratio_vs_sigma_sfr_COMPLET_v1.csv")
out_png = os.path.join(current_dir, "results/figure/sii_ratio_vs_sigma_sfr_v1.png")

os.makedirs(os.path.dirname(out_csv), exist_ok=True)
os.makedirs(os.path.dirname(out_png), exist_ok=True)

# ==========================================
# パラメータ
# ==========================================
BIN_WIDTH = 0.2 # 調整 (0.1がデフォルト)
NMIN = 10 # 調整 (100がデフォルト)

UNIT_FLUX = 1e-17

# ==========================================
# 読み込み
# ==========================================
tab = Table.read(fits_path, hdu=1)
df = tab.to_pandas()

# ==========================================
# 基本量
# ==========================================
F6716 = df["SII_6717_FLUX"].values * UNIT_FLUX
F6731 = df["SII_6731_FLUX"].values * UNIT_FLUX
err6716 = df["SII_6717_FLUX_ERR"].values * UNIT_FLUX
err6731 = df["SII_6731_FLUX_ERR"].values * UNIT_FLUX

df["R_SII"] = F6716 / F6731

# ==========================================
# Re（arcsec → kpc）
# ==========================================
z = df["Z"].values

arcsec_to_kpc = cosmo.kpc_proper_per_arcmin(z).value / 60.0
Re_kpc = df["Re"].values * arcsec_to_kpc

df["Re_kpc"] = Re_kpc
logRe = np.log10(Re_kpc)
df["logRe"] = logRe

# ==========================================
# ΣSFRの計算
# ==========================================
SFR = df["sfr_MEDIAN"].values
Sigma_SFR = SFR / (2 * np.pi * Re_kpc**2)

df["Sigma_SFR"] = Sigma_SFR
logSigma = np.log10(Sigma_SFR)
df["logSigma_SFR"] = logSigma

# ==========================================
# マスク
# ==========================================
mask = (
    np.isfinite(df["R_SII"]) &
    np.isfinite(logSigma) &
    (Sigma_SFR > 0)
)

# ==========================================
# ビン作成（logΣSFR）
# ==========================================
vals = df.loc[mask, "logSigma_SFR"].values

edges = np.arange(
    np.floor(vals.min()/BIN_WIDTH)*BIN_WIDTH,
    np.ceil(vals.max()/BIN_WIDTH)*BIN_WIDTH + BIN_WIDTH,
    BIN_WIDTH
)

# ==========================================
# bin内中央値（ratio）
# ==========================================
rows = []

for lo, hi in zip(edges[:-1], edges[1:]):

    m_bin = (
        mask &
        (df["logSigma_SFR"] >= lo) &
        (df["logSigma_SFR"] < hi)
    )

    N = np.sum(m_bin)
    if N < NMIN:
        continue

    ratio_vals = df.loc[m_bin, "R_SII"].values
    ratio_med = np.nanmedian(ratio_vals)

    rows.append(dict(
        logSigma_lo=lo,
        logSigma_hi=hi,
        logSigma_cen=0.5*(lo+hi),
        N=N,
        R_med=ratio_med
    ))

res = pd.DataFrame(rows)
res.to_csv(out_csv, index=False)

# ==========================================
# 描画
# ==========================================
fig, ax = plt.subplots(figsize=(6,6))

# scatter
ax.scatter(
    df.loc[mask, "logSigma_SFR"],
    df.loc[mask, "R_SII"],
    s=0.01,
    alpha=0.8,
    color="C0"
)

# median
ax.plot(
    res["logSigma_cen"],
    res["R_med"],
    "ks",
    label="median"
)

ax.set_xlabel(r"$\log(\Sigma_{\rm SFR})\ [{\rm M_\odot\ yr^{-1}\ kpc^{-2}}]$")
ax.set_ylabel(r"[SII] 6717 / 6731")
ax.set_xlim(-5.0, 0.0)
ax.set_ylim(0.5, 2.0)

for spine in ax.spines.values():
    spine.set_linewidth(2)

plt.tight_layout()
plt.savefig(out_png, dpi=200)
plt.show()
