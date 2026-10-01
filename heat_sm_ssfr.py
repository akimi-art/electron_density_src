#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
スクリプトの概要:
横軸M*, 縦軸sSFRのヒートマップを作成します。

使用方法:
    heat_sm_ssfr.py [オプション]

著者: A. M.
作成日: 2026-06-17

参考文献:
    - PEP 8:                  https://peps.python.org/pep-0008/
    - PEP 257 (Docstring規約): https://peps.python.org/pep-0257/
    - Python公式ドキュメント:    https://docs.python.org/ja/3/
"""

# 1. FITS読込に必要なパッケージのインストール
from astropy.io import fits
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d
import os
from astropy.table import Table

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
    "xtick.major.size": 12,          # 長さ
    "ytick.major.size": 12,
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


# current_dir = os.getcwd()

# fits_path = os.path.join(
#     current_dir,
#     "results/fits/mpajhu_dr7_v5_2_merged_zlt0.2_Lgt1e+39_radius.fits"
# )

# with fits.open(fits_path) as hdul:

#     data = hdul[1].data

# tab = Table.read(fits_path, hdu=1)
# df = tab.to_pandas()


# # 2. 必要な列を取得
# UNIT_FLUX = 1e-17

# F6716 = df["SII_6717_FLUX"].values * UNIT_FLUX
# F6731 = df["SII_6731_FLUX"].values * UNIT_FLUX

# err6716 = df["SII_6717_FLUX_ERR"].values * UNIT_FLUX
# err6731 = df["SII_6731_FLUX_ERR"].values * UNIT_FLUX

# logM = df["sm_MEDIAN"].values
# logSFR = df["sfr_MEDIAN"].values
# logsSFR = logSFR - logM
# df["logsSFR"] = logsSFR

# # 3. ratio作成
# ratio = F6716 / F6731

# # 4. 品質カット
# sn6716 = F6716 / err6716
# sn6731 = F6731 / err6731

# mask = (
#     np.isfinite(logM)
#     &
#     np.isfinite(logSFR)
#     &
#     np.isfinite(F6716)
#     &
#     np.isfinite(F6731)
#     &
#     np.isfinite(ratio)
#     &
#     (F6716 > 0)
#     &
#     (F6731 > 0)
#     &
#     (sn6716 > 3)
#     &
#     (sn6731 > 3)
# )

# # 5. ratioヒートマップ
# xbins = np.arange(8, 12.1, 0.1)
# ybins = np.arange(-14, -6.9, 0.1)

# ratio_map, xedge, yedge, _ = (
#     binned_statistic_2d(
#         logM[mask],
#         logsSFR[mask],
#         ratio[mask],
#         statistic="median", # 必要に応じて "mean" や "count" に変更可能
#         bins=[xbins, ybins]
#     )
# )

# fig, ax = plt.subplots(figsize=(8,6))
# fig.subplots_adjust(left=0.15, right=0.85, bottom=0.15, top=0.85)

# # vmin = np.nanpercentile(ratio_map,5) # 下限を5パーセンタイルに設定（必要に応じて調整）
# # vmax = np.nanpercentile(ratio_map,95) # 上限を95パーセンタイルに設定（必要に応じて調整）
# vmin = 1.0
# vmax = 1.5

# plt.pcolormesh(
#     xedge,
#     yedge,
#     ratio_map.T,
#     vmin=vmin, vmax=vmax,  # カラーマップの範囲を固定（必要に応じて調整）
#     shading="auto",
#     cmap="viridis"
# )

# plt.colorbar()

# plt.xlabel(r'$\log M_*$')
# plt.ylabel(r'$\log sSFR$')

# for spine in ax.spines.values():
#     spine.set_linewidth(2)
# plt.tight_layout()
# # 保存
# fig_dir = os.path.join(current_dir, "results/figure")
# os.makedirs(fig_dir, exist_ok=True)
# save_path_heat = os.path.join(
#     fig_dir,
#     "heat_sm_ssfr_sii_ratio_sdss.png"
# )

# plt.savefig(save_path_heat)
# print(f"Saved heatmap to: {save_path_heat}")
# plt.show()


# # 6. 各セルの銀河数も確認
# count_map, _, _, _ = (
#     binned_statistic_2d(
#         logM[mask],
#         logsSFR[mask],
#         ratio[mask],
#         statistic="count",
#         bins=[xbins, ybins]
#     )
# )

# fig, ax = plt.subplots(figsize=(8,6))
# fig.subplots_adjust(left=0.15, right=0.85, bottom=0.15, top=0.85)

# plt.pcolormesh(
#     xedge,
#     yedge,
#     count_map.T,
#     shading="auto",
#     cmap="viridis"
# )

# plt.colorbar()

# plt.xlabel(r'$\log M_*$')
# plt.ylabel(r'$\log sSFR$')

# for spine in ax.spines.values():
#     spine.set_linewidth(2)
# plt.tight_layout()

# save_path_number = os.path.join(
#     fig_dir,
#     "heat_sm_ssfr_galaxy_number_sdss.png"
# )

# plt.savefig(save_path_number)
# print(f"Saved galaxy count heatmap to: {save_path_number}")
# plt.show()









"""
Claudeが作成した、「完全性をM*とSFRのヒートマップで可視化する」スクリプトをベースに、必要な部分を整理して再構築しました。
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u

# (rcParams の設定は今のものをそのまま使う)

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)   # 論文で採用する宇宙論に合わせる
UNIT_FLUX = 1e-17                        # erg s^-1 cm^-2
Z_MAX = 0.1825
L_CUT = 1e39                             # erg s^-1
M_MIN = 9.2                              # M*–SFR の完全性から決めた質量の下限

# main sequence の目安（Renzini & Peng 2015）：log SFR = 0.76 log M* - 7.64
MS_SLOPE, MS_ZP, MS_WIDTH = 0.76, -7.64, 0.6

# ------------------------------------------------------------
# 1. 光度のカット「前」のカタログを読む
# ------------------------------------------------------------
path = "results/fits/mpajhu_dr7_v5_2_merged_radius.fits"
df = Table.read(path, hdu=1).to_pandas()

z       = df["Z"].values
logM    = df["sm_MEDIAN"].values
logSFR  = df["sfr_MEDIAN"].values
logsSFR = logSFR - logM                  # log(sSFR / yr^-1)

def flux(name):
    return (df[f"{name}_FLUX"].values * UNIT_FLUX,
            df[f"{name}_FLUX_ERR"].values * UNIT_FLUX)

F_Hb,  E_Hb  = flux("H_BETA")
F_O3,  E_O3  = flux("OIII_5007")
F_Ha,  E_Ha  = flux("H_ALPHA")
F_N2,  E_N2  = flux("NII_6584")
F_S31, E_S31 = flux("SII_6731")

# ------------------------------------------------------------
# 2. 親サンプル：光度のカット以外の条件をすべて課す
# ------------------------------------------------------------
with np.errstate(divide="ignore", invalid="ignore"):
    sn_ok = ((F_Hb / E_Hb >= 3) & (F_O3 / E_O3 >= 3) &
             (F_Ha / E_Ha >= 3) & (F_N2 / E_N2 >= 3))
    x = np.log10(F_N2 / F_Ha)
    y = np.log10(F_O3 / F_Hb)
    sf_ka03 = (x < 0.05) & (y < 0.61 / (x - 0.05) + 1.3)

base = (np.isfinite(z) & (z > 0) & (z < Z_MAX) &
        np.isfinite(logM) & np.isfinite(logSFR) & np.isfinite(logsSFR))

parent = base & sn_ok & sf_ka03

# ------------------------------------------------------------
# 3. 選択後：親サンプル ＋ 光度のカット
# ------------------------------------------------------------
dL = cosmo.luminosity_distance(np.clip(z, 1e-4, None)).to(u.cm).value
with np.errstate(invalid="ignore"):
    L6731 = 4 * np.pi * dL**2 * F_S31

selected = parent & np.isfinite(L6731) & (L6731 > L_CUT)

print(f"parent  : {parent.sum():,}")
print(f"selected: {selected.sum():,}")

# ------------------------------------------------------------
# 4. 数の地図と完全性の地図（M*–sSFR）
# ------------------------------------------------------------
xbins = np.arange(7.0, 12.01, 0.1)
ybins = np.arange(-13.0, -7.99, 0.1)

N_par, _, _ = np.histogram2d(logM[parent],   logsSFR[parent],   bins=[xbins, ybins])
N_sel, _, _ = np.histogram2d(logM[selected], logsSFR[selected], bins=[xbins, ybins])

NMIN_CELL = 10
with np.errstate(divide="ignore", invalid="ignore"):
    comp = N_sel / N_par
comp[N_par < NMIN_CELL] = np.nan

fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
norm = LogNorm(vmin=1, vmax=N_par.max())

for ax, N, title in zip(axes[:2], [N_par, N_sel], ["Parent", "Selected"]):
    im = ax.pcolormesh(xbins, ybins, np.where(N > 0, N, np.nan).T,
                       norm=norm, cmap="viridis", shading="auto")
    ax.set_title(title)
fig.colorbar(im, ax=axes[:2], label="Number of galaxies")

im3 = axes[2].pcolormesh(xbins, ybins, comp.T, vmin=0, vmax=1,
                         cmap="magma", shading="auto")
axes[2].set_title("Completeness")
fig.colorbar(im3, ax=axes[2], label=r"$N_{\rm sel}/N_{\rm parent}$")

# main sequence（sSFR の平面では log sSFR = (0.76 - 1) log M* - 7.64）と採用した M* の下限
mm = np.linspace(7, 12, 100)
ms_ssfr = (MS_SLOPE - 1) * mm + MS_ZP
for ax in axes:
    ax.plot(mm, ms_ssfr, color="w", lw=1.5)
    ax.plot(mm, ms_ssfr - MS_WIDTH, color="w", lw=1, ls="--")
    ax.plot(mm, ms_ssfr + MS_WIDTH, color="w", lw=1, ls="--")
    ax.axvline(M_MIN, color="cyan", lw=1.5, ls=":")
    ax.set_xlabel(r"$\log(M_*/M_\odot)$")
    ax.set_xlim(xbins[0], xbins[-1])
    ax.set_ylim(ybins[0], ybins[-1])
axes[0].set_ylabel(r"$\log({\rm sSFR}/{\rm yr^{-1}})$")

fig_dir = "results/figure"
os.makedirs(fig_dir, exist_ok=True)
save_map = os.path.join(fig_dir, "completeness_map_sm_ssfr.png")
plt.savefig(save_map, bbox_inches="tight")
plt.show()
print(f"Saved completeness map to: {save_map}")

# ------------------------------------------------------------
# 5. 完全性を sSFR の関数にする（採用した M* の範囲の銀河について）
#    Figure 1 の sSFR パネルで、どの sSFR のビンが信頼できるかを見る
# ------------------------------------------------------------
in_mass = logM >= M_MIN

sbins = np.arange(-12.0, -7.99, 0.2)
scen = 0.5 * (sbins[1:] + sbins[:-1])
n_par, _ = np.histogram(logsSFR[parent & in_mass],   bins=sbins)
n_sel, _ = np.histogram(logsSFR[selected & in_mass], bins=sbins)
with np.errstate(divide="ignore", invalid="ignore"):
    frac = n_sel / n_par
frac[n_par < NMIN_CELL] = np.nan

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(scen, frac, "o-", color="k")
ax.axhline(0.9, ls="--", color="gray")
ax.set_xlabel(r"$\log({\rm sSFR}/{\rm yr^{-1}})$")
ax.set_ylabel(rf"Completeness ($\log M_* \geq {M_MIN}$)")
ax.set_ylim(0, 1.05)
save_curve = os.path.join(fig_dir, "completeness_vs_ssfr.png")
plt.savefig(save_curve, bbox_inches="tight")
plt.show()
print(f"Saved completeness vs sSFR plot to: {save_curve}")

for s, f, n in zip(scen, frac, n_par):
    print(f"log sSFR = {s:6.1f}: completeness = {f:.2f} (N_parent = {n})")