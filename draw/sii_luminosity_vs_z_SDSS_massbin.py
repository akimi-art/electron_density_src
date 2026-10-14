#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
スクリプトの概要:
SIIのLumiosityとzの関係を描画します。
単にデータを描画するだけではなく、
フラックスのSN比の情報も含まれます。
これはSDSS用です。
Stellar Massごとにビンを分けて描画します。


使用方法:
    sii_luminoisity_vs_z_SDSS_massbin.py [オプション]

著者: A. M.
作成日: 2026-02-17

参考文献:
    - PEP 8:                  https://peps.python.org/pep-0008/
    - PEP 257 (Docstring規約): https://peps.python.org/pep-0257/
    - Python公式ドキュメント:    https://docs.python.org/ja/3/
    - Curti+17
    - Matplotlib colormap:    https://jp.matplotlib.net/stable/tutorials/colors/colormaps.html#google_vignette
"""

"""
luminosity_vs_z_v1.pyを参考に後でflux一定の線を入れよう
"""


# === 必要なモジュールのインポート ===
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
from astropy.table import Table
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u


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
    "xtick.top": False,               # 上にも目盛り
    "ytick.right": False,             # 右にも目盛り

    # 主目盛り（major ticks）
    "xtick.major.size": 20,          # 長さ
    "ytick.major.size": 20,
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
    "xtick.labelsize": 20,           # x軸ラベルサイズ
    "ytick.labelsize": 20,           # y軸ラベルサイズ

    # --- フォント ---
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
})


# === 入力 FITS ===
current_dir = os.getcwd()
fits_path = os.path.join(
    current_dir,
    "results/fits/mpajhu_dr7_v5_2_merged.fits"
)
t = Table.read(fits_path, format="fits")
df = t.to_pandas()

# === 列の読み込み ===
UNIT_FLUX = 1e-17

z       = df["Z"].to_numpy(float)
logM    = df["sm_MEDIAN"].to_numpy(float)
F6716   = df["SII_6717_FLUX"].to_numpy(float) * UNIT_FLUX
F6731   = df["SII_6731_FLUX"].to_numpy(float) * UNIT_FLUX
err6716 = df["SII_6717_FLUX_ERR"].to_numpy(float) * UNIT_FLUX
err6731 = df["SII_6731_FLUX_ERR"].to_numpy(float) * UNIT_FLUX

# === S/N と光度 ===
# 置き換え前：
# sn6716 = F6716 / err6716
# sn6731 = F6731 / err6731


# 置き換え後（安全版）：
with np.errstate(divide='ignore', invalid='ignore'):
    sn6716 = np.divide(F6716, err6716, out=np.full_like(F6716, np.nan), where=np.isfinite(err6716) & (err6716 > 0))
    sn6731 = np.divide(F6731, err6731, out=np.full_like(F6731, np.nan), where=np.isfinite(err6731) & (err6731 > 0))


dL = np.full_like(z, np.nan)
idx = np.isfinite(z) & (z >= 0)
dL[idx] = cosmo.luminosity_distance(z[idx]).to(u.cm).value

L6716 = 4 * np.pi * dL**2 * F6716
L6731 = 4 * np.pi * dL**2 * F6731

# === Mass bins ===
mass_edges = np.array([6,7,8,9,10,11,12])
n_cols = len(mass_edges) - 1

# === 描画パラメータ ===
zmin, zmax = 0.0, 0.40
Lmin, Lmax = 1e30, 1e50
flux_lines = (1e-19, 1e-18, 1e-17)

z_grid = np.linspace(zmin, zmax, 200)
dL_grid = cosmo.luminosity_distance(z_grid).to(u.cm).value
def Lfl(F):
    return 4*np.pi * dL_grid**2 * F

# === 図作成（GridSpec 2×6、隙間ゼロ） ===
fig = plt.figure(figsize=(20, 6))
gs  = gridspec.GridSpec(2, n_cols, figure=fig,
                        wspace=0.0, hspace=0.0)   # ← 隙間ゼロ

# カラーマップ（S/N 表示用、ただしカラーバーは付けない）
norm = TwoSlopeNorm(vcenter=0, vmin=-3, vmax=5)
cmap = "coolwarm"

# === 各ビンで上下に 6716 / 6731 を描く ===
for j in range(n_cols):

    mlo, mhi = mass_edges[j], mass_edges[j+1]

    # 最後のビンのみ上限含む
    if j < n_cols - 1:
        m_bin = (logM >= mlo) & (logM < mhi)
    else:
        m_bin = (logM >= mlo) & (logM <= mhi)

    # --- マスク（6716 と 6731 共通構造） ---
    mask6716 = (
        m_bin &
        np.isfinite(z) & np.isfinite(L6716) & (L6716 > 0) &
        np.isfinite(err6716) & (err6716 > 0) &    # ★ 追加
        np.isfinite(sn6716)
    )
    mask6731 = (
        m_bin &
        np.isfinite(z) & np.isfinite(L6731) & (L6731 > 0) &
        np.isfinite(err6731) & (err6731 > 0) &    # ★ 追加
        np.isfinite(sn6731)
    )

    # === 上段（6716） ===
    ax1 = fig.add_subplot(gs[0, j])
    ax1.scatter(z[mask6716], L6716[mask6716],
                c=sn6716[mask6716],
                cmap=cmap, norm=norm,
                s=10, alpha=0.75)
    # 上段（6716）の散布の直後に：
    ax1.plot([], [], ' ', label=(f"{mlo:.0f} ≤ logM < {mhi:.0f}" if j < n_cols-1 else f"{mlo:.0f} ≤ logM ≤ {mhi:.0f}"))
    leg1 = ax1.legend(loc='lower right', frameon=True, fontsize=10, handlelength=0, handletextpad=0.2, edgecolor='black')
    leg1.get_frame().set_alpha(0.8)

    # 一定フラックス線
    for i,F0 in enumerate(flux_lines):
        ax1.plot(z_grid, Lfl(F0),
                 color="black",
                 lw=1.0,
                 ls=["--","-.",":"][i % 3],
                 alpha=0.9)

    ax1.set_xlim(zmin, zmax)
    ax1.set_ylim(Lmin, Lmax)
    ax1.set_yscale("log")

    # y ラベルは左端だけ
    if j == 0:
        ax1.set_ylabel("L([S II] 6716)")
    else:
        ax1.set_yticklabels([])

    # x ラベルは上段は非表示
    ax1.set_xticklabels([])

    # 軸枠
    for spine in ax1.spines.values():
        spine.set_linewidth(2)

    # === 下段（6731） ===
    ax2 = fig.add_subplot(gs[1, j])
    ax2.scatter(z[mask6731], L6731[mask6731],
                c=sn6731[mask6731],
                cmap=cmap, norm=norm,
                s=10, alpha=0.75)

    # 下段（6731）の散布の直後に：
    ax2.plot([], [], ' ', label=(f"{mlo:.0f} ≤ logM < {mhi:.0f}" if j < n_cols-1 else f"{mlo:.0f} ≤ logM ≤ {mhi:.0f}"))
    leg2 = ax2.legend(loc='lower right', frameon=True, fontsize=10, handlelength=0, handletextpad=0.2, edgecolor='black')
    leg2.get_frame().set_alpha(0.8)

    for i,F0 in enumerate(flux_lines):
        ax2.plot(z_grid, Lfl(F0),
                 color="black",
                 lw=1.0,
                 ls=["--","-.",":"][i % 3],
                 alpha=0.9)

    ax2.set_xlim(zmin, zmax)
    ax2.set_ylim(Lmin, Lmax)
    ax2.set_yscale("log")

    if j == 0:
        ax2.set_ylabel("L([S II] 6731)")
    else:
        ax2.set_yticklabels([])

    ax2.set_xlabel("z")

    for spine in ax2.spines.values():
        spine.set_linewidth(2)

# === 保存 ===
save_path = os.path.join(current_dir, "results/figure/mpajhu_massbin_2x6_L6716_L6731.png")
plt.savefig(save_path, dpi=200, bbox_inches="tight")
print(f"Saved: {save_path}")

plt.show()