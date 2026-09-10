#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
スクリプトの概要:
SIIのLumiosityとzの関係を描画します。
これはSDSS用です。

v0との変更点
・SIIの暗い側のみをサンプル選択に含める
・フラックス一定の曲線とSDSSの観測装置の特性をつなげる
・フラックス一定の曲線とLuminosity一定の曲線の交点をサンプル選択に使用
・使用するファイル(merged)にReの情報をあらかじめ入れておく

使用方法:
    sii_luminoisity_vs_z_SDSS_v2.py [オプション]

著者: A. M.
作成日: 2026-08-30

参考文献:
    - フラックス一定の曲線とSDSSの観測装置の特性
"""


# === 必要なモジュールのインポート ===
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from astropy.io import fits
import astropy.units as u
from astropy.cosmology import Planck18 as cosmo
from astropy.table import Table
from astropy.table import Column
from matplotlib.colors import TwoSlopeNorm
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize


# 軸の設定
plt.rcParams.update({
    # --- 図全体 ---
    "figure.figsize": (12, 6),       # 図サイズ
    "font.size": 32,                 # 全体フォントサイズ
    "axes.labelsize": 32,            # 軸ラベルのサイズ
    "axes.titlesize": 32,            # タイトルのサイズ
    "axes.grid": False,              # グリッドOFF

    # --- 目盛り設定 (ticks) ---
    "xtick.direction": "in",         # x軸目盛りの向き
    "ytick.direction": "in",         # y軸目盛りの向き
    "xtick.top": True,               # 上にも目盛り
    "ytick.right": True,             # 右にも目盛り

    # 主目盛り（major ticks）
    "xtick.major.size": 32,          # 長さ
    "ytick.major.size": 32,
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
    "xtick.labelsize": 28,           # x軸ラベルサイズ
    "ytick.labelsize": 28,           # y軸ラベルサイズ

    # --- フォント ---
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
})


# =====================================
# 基本量の設定
# =====================================

L_MIN = 1e39          # erg s^-1
FLUX_LIMIT = 1e-17    # erg s^-1 cm^-2

UNIT_FLUX = 1e-17     # MPA-JHU flux unit

# =====================================
# FITS 読み込み
# =====================================
current_dir = os.getcwd()
# 先にfitsファイルにReの情報を入れておく
fits_path = os.path.join(current_dir, "results/fits/mpajhu_dr7_v5_2_merged.fits")

t = Table.read(fits_path, format="fits")
df = t.to_pandas()

# =====================================
# L_MIN と flux-limit の交点を計算
# =====================================

z_grid_cross = np.linspace(1e-5, 0.4, 10000)

dL_cross = (
    cosmo.luminosity_distance(z_grid_cross)
    .to(u.cm)
    .value
)

L_cross = 4 * np.pi * dL_cross**2 * FLUX_LIMIT

idx = np.argmin(np.abs(L_cross - L_MIN))

Z_MAX = z_grid_cross[idx]

print(f"[INFO] Adopted Z_MAX = {Z_MAX:.4f}")

# =====================================
# データ抽出
# =====================================
z = df["Z"].values

F6716 = df["SII_6717_FLUX"].values * UNIT_FLUX
err6716 = df["SII_6717_FLUX_ERR"].values * UNIT_FLUX

F6731 = df["SII_6731_FLUX"].values * UNIT_FLUX
err6731 = df["SII_6731_FLUX_ERR"].values * UNIT_FLUX

# SII輝線のSN比を設定する
sn6716 = F6716 / err6716
sn6731 = F6731 / err6731

# =====================================
# Luminosity 計算
# =====================================
d_L = cosmo.luminosity_distance(z).to(u.cm).value
L6716 = 4 * np.pi * d_L**2 * F6716
L6731 = 4 * np.pi * d_L**2 * F6731

# =====================================
# 数値健全性マスク
# =====================================
# ここも変更する
mask_finite = (
    np.isfinite(z) &
    np.isfinite(L6716) &
    np.isfinite(L6731) &
    (z > 0)
)

# =====================================
# 完全サンプル条件
# =====================================
# ここを変更する
mask_complete = (
    mask_finite &
    (z < Z_MAX) &
    (L6731 > L_MIN)  
)

print(f"[INFO] 抽出件数: {mask_complete.sum()} / {len(mask_complete)}")

# =====================================
# 基本統計量
# =====================================

z_sel = z[mask_complete]
L6731_sel = L6731[mask_complete]

print("\n===== Sample Statistics =====")

print(f"N(all)      = {len(df):,}")
print(f"N(selected) = {len(z_sel):,}")

print(
    f"Fraction    = "
    f"{len(z_sel)/len(df):.3f}"
)

print(
    f"z range     = "
    f"{np.min(z_sel):.4f} -- {np.max(z_sel):.4f}"
)

print(
    f"z median    = "
    f"{np.median(z_sel):.4f}"
)

print(
    f"logL median = "
    f"{np.log10(np.median(L6731_sel)):.3f}"
)

print(
    f"logL min    = "
    f"{np.log10(np.min(L6731_sel)):.3f}"
)

print(
    f"logL max    = "
    f"{np.log10(np.max(L6731_sel)):.3f}"
)

print("================================\n")

# =====================================
# 図の描画
# =====================================
# 6731の方
fig, ax = plt.subplots(figsize=(12, 6))
fig.subplots_adjust(left=0.10, right=0.95, bottom=0.15, top=0.95)

# 全体
ax.scatter(z, L6731, s=6, alpha=0.3, color="gray")

# 完全サンプル
ax.scatter(
    z[mask_complete],
    L6731[mask_complete],
    s=0.2,
    alpha=1,
    color="firebrick",
)

# カット線
ax.axvline(Z_MAX, color="k", linestyle="-", linewidth=2.0)
ax.axhline(L_MIN, color="k", linestyle="-", linewidth=2.0)

#  L([S II] 6731) の一定フラックス線
z_grid = np.linspace(0.0, 0.4, 200)
d_L_grid = cosmo.luminosity_distance(z_grid).to(u.cm).value 
L_const = 4 * np.pi * d_L_grid**2 * FLUX_LIMIT
ax.plot(z_grid, L_const, color="black", linestyle="-", linewidth=2.0)

# 軸設定
ax.set_yscale("log")
ax.set_xlim(0, 0.4)
ax.set_ylim(1e37, 1e42)

ax.set_xlabel("z")
ax.set_ylabel(r"L([S II] 6731) [erg s$^{-1}$]")


# 枠線強調
for spine in ax.spines.values():
    spine.set_linewidth(2)

# 保存
fig_dir = os.path.join(current_dir, "results/figure")
os.makedirs(fig_dir, exist_ok=True)

save_path = os.path.join(fig_dir, "sii6731_luminosity_vs_z_v2.png")
plt.savefig(save_path, dpi=200, bbox_inches="tight")
plt.show()

print(f"[DONE] 図を保存: {save_path}")


# =====================================
# FITS 抽出
# =====================================
t_sel = t[mask_complete]

out_dir = os.path.join(current_dir, "results/fits")
os.makedirs(out_dir, exist_ok=True)

out_path = os.path.join(
    out_dir,
    f"mpajhu_dr7_v5_2_merged_zlt{Z_MAX:.3f}_Lgt{L_MIN:.0e}.fits"
)

t_sel.write(out_path, format="fits", overwrite=True)

print(f"[DONE] 書き出し完了: {out_path}")
# [INFO] 抽出件数: 526970 / 927552


# =====================================
# 抽出された銀河の赤方偏位のヒストグラム描画
# =====================================
fig, ax = plt.subplots(figsize=(12, 6))

ax.hist(
    z[mask_complete],
    bins=50,
    color="gray",
    edgecolor="black",
    alpha=0.8,
)

z_median = np.median(z[mask_complete])
z_max = np.max(z[mask_complete])
ax.axvline(
    z_median,
    color="k",
    linestyle="--",
    linewidth=2,
)

# 横軸の0.00を消す
ticks = ax.get_xticks()
ticks = ticks[ticks > 0]
ax.set_xticks(ticks)

ax.set_xlabel("z")
ax.set_ylabel("Number of galaxies")
ax.set_xlim(0, z_max)

for spine in ax.spines.values():
    spine.set_linewidth(2)

hist_path = os.path.join(
    fig_dir,
    "selected_sample_redshift_histogram_v2.png"
)

plt.savefig(
    hist_path,
    dpi=200,
    bbox_inches="tight"
)

plt.show()

print(f"[DONE] Histogram saved: {hist_path}")


# =====================================
# L6731 vs M*
# =====================================

Mstar = df["sm_MEDIAN"].values

mask_plot = (
    mask_finite &
    np.isfinite(Mstar) &
    (L6731 > 0)
)

fig, ax = plt.subplots(figsize=(12, 6))

ax.scatter(
    Mstar[mask_plot],
    np.log10(L6731[mask_plot]),
    s=1,
    alpha=0.2,
    color="black",
)

ax.set_xlabel(
    r"$\log(M_\star/M_\odot)$"
)

ax.set_ylabel(
    r"$\log\,L([{\rm S\,II}]\,6731)\ [{\rm erg\ s^{-1}}]$"
)

ax.set_xlim(8, 12)
ax.set_ylim(36, 44)

for spine in ax.spines.values():
    spine.set_linewidth(2)

plt.tight_layout()

save_path = os.path.join(
    fig_dir,
    "L6731_vs_Mstar.png"
)

plt.savefig(
    save_path,
    dpi=200,
    bbox_inches="tight"
)

plt.show()

print(f"[DONE] Saved: {save_path}")



# =====================================
# L6731 vs M* (color = z)
# =====================================

mask_plot = (
    np.isfinite(Mstar) &
    np.isfinite(L6731) &
    np.isfinite(z) &
    (L6731 > 0)
)

fig, ax = plt.subplots(figsize=(12, 6))

sc = ax.scatter(
    Mstar[mask_plot],
    np.log10(L6731[mask_plot]),
    c=z[mask_plot],
    cmap="viridis",
    s=1,
    alpha=0.5,
)

cbar = plt.colorbar(sc)
cbar.set_label("z")

ax.set_xlabel(
    r"$\log(M_\star/M_\odot)$"
)

ax.set_ylabel(
    r"$\log\,L([{\rm S\,II}]\,6731)$"
)

ax.set_xlim(8, 12)
ax.set_ylim(36, 44)

for spine in ax.spines.values():
    spine.set_linewidth(2)

save_path = os.path.join(
    fig_dir,
    "L6731_vs_Mstar_cmap.png"
)

plt.savefig(
    save_path,
    dpi=200,
    bbox_inches="tight"
)


plt.tight_layout()
plt.show()

print(f"[DONE] Saved: {save_path}")


fig, ax = plt.subplots(figsize=(12, 6))
plt.scatter(
    Mstar[mask_plot],
    z[mask_plot],
    s=0.1
)
ax.set_xlabel(
    r"$\log(M_\star/M_\odot)$"
)

ax.set_ylabel(
    r"$z$"
)
ax.set_xlim(8, 12)
ax.set_ylim(0, 0.4)

save_path = os.path.join(
    fig_dir,
    "Mstar_vs_z.png"
)

plt.savefig(
    save_path,
    dpi=200,
    bbox_inches="tight"
)


plt.tight_layout()
plt.show()

print(f"[DONE] Saved: {save_path}")


# =====================================
# L6731 vs Stellar Mass
# =====================================

# 星質量
Mstar = t_sel["sm_MEDIAN"]

# 念のため有限値のみ
mask_mass = (
    np.isfinite(Mstar) &
    np.isfinite(L6731_sel) &
    (L6731_sel > 0)
)

# プロット
fig, ax = plt.subplots(figsize=(12, 6))

ax.scatter(
    Mstar[mask_mass],
    np.log10(L6731_sel[mask_mass]),
    s=1,
    alpha=0.2,
    color="black",
)

ax.set_xlabel(
    r"$\log(M_\star/M_\odot)$"
)

ax.set_ylabel(
    r"$\log\,L([{\rm S\,II}]\,6731)\ [{\rm erg\ s^{-1}}]$"
)
ax.set_xlim(8, 12)
ax.set_ylim(36, 44)

for spine in ax.spines.values():
    spine.set_linewidth(2)

plt.tight_layout()

save_path = os.path.join(
    fig_dir,
    "L6731_vs_Mstar_2.png"
)

plt.savefig(
    save_path,
    dpi=200,
    bbox_inches="tight"
)

plt.show()

print(f"[DONE] Saved: {save_path}")




# =====================================
# Mass-bin histogram of L6731
# =====================================

Mstar = df["sm_MEDIAN"].values

mask = (
    np.isfinite(Mstar) &
    np.isfinite(L6731) &
    (L6731 > 0)
)

# 質量ビン
mass_bins = [
    (8.0, 9.0),
    (9.0, 10.0),
    (10.0, 11.0),
    (11.0, 12.0),
]

fig, axes = plt.subplots(
    2, 2,
    figsize=(12, 10),
    sharex=True,
    sharey=True
)

axes = axes.flatten()

for ax, (mmin, mmax) in zip(axes, mass_bins):

    msk = (
        mask &
        (Mstar >= mmin) &
        (Mstar < mmax)
    )

    ax.hist(
        np.log10(L6731[msk]),
        bins=50,
        color="gray",
        edgecolor="black",
        alpha=0.8,
    )

    ax.set_title(
        rf"${mmin}\leq \log(M_\star/M_\odot)<{mmax}$"
    )

    ax.set_xlabel(
        r"$\log L([{\rm S\,II}]6731)$"
    )

    ax.set_ylabel("Number")

    for spine in ax.spines.values():
        spine.set_linewidth(2)

plt.tight_layout()
plt.show()


fig, ax = plt.subplots(figsize=(10,6))

colors = ["blue", "green", "orange", "red"]

for (mmin, mmax), color in zip(mass_bins, colors):

    msk = (
        mask &
        (Mstar >= mmin) &
        (Mstar < mmax)
    )

    ax.hist(
        np.log10(L6731[msk]),
        bins=60,
        density=True,   # 規格化
        histtype="step",
        linewidth=2,
        color=color,
        label=rf"${mmin}\leq \log(M_\star/M_\odot)<{mmax}$"
    )

ax.set_xlabel(
    r"$\log L([{\rm S\,II}]6731)$"
)

ax.set_ylabel("Normalized count")

ax.legend(fontsize=12)

for spine in ax.spines.values():
    spine.set_linewidth(2)

plt.tight_layout()
plt.show()


# =====================================
# LHa vs M* (color = z)
# =====================================
Fha = df["H_ALPHA_FLUX"].values * UNIT_FLUX
Lha = 4 * np.pi * d_L**2 * Fha

mask_plot = (
    np.isfinite(Mstar) &
    np.isfinite(Lha) &
    np.isfinite(z) &
    (Lha > 0)
)

fig, ax = plt.subplots(figsize=(12, 6))

sc = ax.scatter(
    Mstar[mask_plot],
    np.log10(Lha[mask_plot]),
    c=z[mask_plot],
    cmap="viridis",
    s=1,
    alpha=0.5,
)

cbar = plt.colorbar(sc)
cbar.set_label("z")

ax.set_xlabel(
    r"$\log(M_\star/M_\odot)$"
)

ax.set_ylabel(
    r"$\log L([{\rm H\,a}])$"
)

ax.set_xlim(8, 12)
ax.set_ylim(36, 44)

for spine in ax.spines.values():
    spine.set_linewidth(2)

save_path = os.path.join(
    fig_dir,
    "Lha_vs_Mstar_cmap.png"
)

plt.savefig(
    save_path,
    dpi=200,
    bbox_inches="tight"
)


plt.tight_layout()
plt.show()

print(f"[DONE] Saved: {save_path}")


# =====================================
# 図の描画(Ha)
# =====================================
fig, ax = plt.subplots(figsize=(12, 6))
fig.subplots_adjust(left=0.10, right=0.95, bottom=0.15, top=0.95)

# 全体
ax.scatter(z, Lha, s=6, alpha=0.3, color="gray")

mask_complete_ha = (
    mask_finite &
    (z < Z_MAX) &
    (Lha > L_MIN)  
)

# 完全サンプル
ax.scatter(
    z[mask_complete_ha],
    Lha[mask_complete_ha],
    s=0.2,
    alpha=1,
    color="firebrick",
)

# カット線
ax.axvline(Z_MAX, color="k", linestyle="-", linewidth=2.0)
ax.axhline(L_MIN, color="k", linestyle="-", linewidth=2.0)

#  L([H a]) の一定フラックス線
z_grid = np.linspace(0.0, 0.4, 200)
d_L_grid = cosmo.luminosity_distance(z_grid).to(u.cm).value 
L_const = 4 * np.pi * d_L_grid**2 * FLUX_LIMIT
ax.plot(z_grid, L_const, color="black", linestyle="-", linewidth=2.0)

# 軸設定
ax.set_yscale("log")
ax.set_xlim(0, 0.4)
ax.set_ylim(1e37, 1e44)

ax.set_xlabel("z")
ax.set_ylabel(r"L([H a]) [erg s$^{-1}$]")


# 枠線強調
for spine in ax.spines.values():
    spine.set_linewidth(2)

# 保存
fig_dir = os.path.join(current_dir, "results/figure")
os.makedirs(fig_dir, exist_ok=True)

save_path = os.path.join(fig_dir, "ha_luminosity_vs_z_v2.png")
plt.savefig(save_path, dpi=200, bbox_inches="tight")
plt.show()

print(f"[DONE] 図を保存: {save_path}")



# =====================================
# Mass-completeなsample(SII)
# =====================================
# 全サンプル
mask_all = (
    np.isfinite(Mstar) &
    np.isfinite(L6731) &
    (L6731 > 0)
)

# luminosity cut
mask_lcut = (
    mask_all &
    (L6731 > 1e39)
)

# 0.2 dex bin
mass_bins = np.arange(8.0, 12.2, 0.2)
mass_center = 0.5 * (mass_bins[1:] + mass_bins[:-1])

fraction = []
N_all_list = []
N_sel_list = []

for mmin, mmax in zip(mass_bins[:-1], mass_bins[1:]):

    msk_all = (
        mask_all &
        (Mstar >= mmin) &
        (Mstar < mmax)
    )

    msk_sel = (
        mask_lcut &
        (Mstar >= mmin) &
        (Mstar < mmax)
    )

    N_all = np.sum(msk_all)
    N_sel = np.sum(msk_sel)

    N_all_list.append(N_all)
    N_sel_list.append(N_sel)

    if N_all > 0:
        fraction.append(N_sel / N_all)
    else:
        fraction.append(np.nan)

fraction = np.array(fraction)


fig, ax = plt.subplots(figsize=(10,6))

ax.plot(
    mass_center,
    fraction,
    marker="o",
    lw=2,
)

ax.set_xlabel(
    r"$\log(M_\star/M_\odot)$"
)

ax.set_ylabel(
    r"$N(L_{6731}>10^{39})/N({\rm all})$"
)

ax.set_ylim(0, 1.05)

for spine in ax.spines.values():
    spine.set_linewidth(2)

plt.tight_layout()
plt.show()


for m, nall, nsel, frac in zip(
    mass_center,
    N_all_list,
    N_sel_list,
    fraction
):
    print(
        f"logM={m:.1f} : "
        f"Nall={nall:7d}, "
        f"Nsel={nsel:7d}, "
        f"f={frac:.3f}"
    )


# =====================================
# Mass-completeなsample (Ha)
# =====================================
# 全サンプル
mask_all = (
    np.isfinite(Mstar) &
    np.isfinite(Lha) &
    (Lha > 0)
)

# luminosity cut
mask_lcut = (
    mask_all &
    (L6731 > 1e39)
)

# 0.2 dex bin
mass_bins = np.arange(8.0, 12.2, 0.2)
mass_center = 0.5 * (mass_bins[1:] + mass_bins[:-1])

fraction = []
N_all_list = []
N_sel_list = []

for mmin, mmax in zip(mass_bins[:-1], mass_bins[1:]):

    msk_all = (
        mask_all &
        (Mstar >= mmin) &
        (Mstar < mmax)
    )

    msk_sel = (
        mask_lcut &
        (Mstar >= mmin) &
        (Mstar < mmax)
    )

    N_all = np.sum(msk_all)
    N_sel = np.sum(msk_sel)

    N_all_list.append(N_all)
    N_sel_list.append(N_sel)

    if N_all > 0:
        fraction.append(N_sel / N_all)
    else:
        fraction.append(np.nan)

fraction = np.array(fraction)


fig, ax = plt.subplots(figsize=(10,6))

ax.plot(
    mass_center,
    fraction,
    marker="o",
    lw=2,
)

ax.set_xlabel(
    r"$\log(M_\star/M_\odot)$"
)

ax.set_ylabel(
    r"$N(LHa>10^{39})/N({\rm all})$"
)

ax.set_ylim(0, 1.05)

for spine in ax.spines.values():
    spine.set_linewidth(2)

plt.tight_layout()
plt.show()


for m, nall, nsel, frac in zip(
    mass_center,
    N_all_list,
    N_sel_list,
    fraction
):
    print(
        f"logM={m:.1f} : "
        f"Nall={nall:7d}, "
        f"Nsel={nsel:7d}, "
        f"f={frac:.3f}"
    )