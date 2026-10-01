#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
スクリプトの概要:
横軸M*, 縦軸SFRのヒートマップを作成します。

使用方法:
    heat_sm_sfr.py [オプション]

著者: A. M.
作成日: 2026-06-14

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
#     "results/fits/mpajhu_dr7_v5_2_merged_radius_zlt0.182_Lgt1e+39.fits"
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

# # ==========================================
# # Flux histogram
# # ==========================================

# fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# # 0以下・NaN除去
# m6716 = np.isfinite(F6716) & (F6716 > 0)
# m6731 = np.isfinite(F6731) & (F6731 > 0)

# ax[0].hist(
#     np.log10(F6716[m6716]),
#     bins=100,
#     histtype="step",
#     linewidth=2,
#     color="C0"
# )

# ax[0].set_xlabel(
#     r"$\log F(6717)$"
# )
# ax[0].set_ylabel("Count")

# ax[1].hist(
#     np.log10(F6731[m6731]),
#     bins=100,
#     histtype="step",
#     linewidth=2,
#     color="C1"
# )

# ax[1].set_xlabel(
#     r"$\log F(6731)$"
# )
# ax[1].set_ylabel("Count")

# plt.tight_layout()
# plt.show()


# # ==========================================
# # Ratio histogram
# # ==========================================

# ratio = F6716 / F6731

# m_ratio = (
#     np.isfinite(ratio)
#     &
#     (ratio > 0)
# )

# m_low = (
#     (logM > 8.0)
#     &
#     (logM < 9.0)
#     &
#     m_ratio
# )

# plt.figure(figsize=(6,5))

# plt.hist(
#     ratio[m_low],
#     bins=2000,
#     histtype="step",
#     linewidth=2
# )

# plt.xlabel(r"[SII]6717/[SII]6731")
# plt.ylabel("Count")
# plt.xlim(0, 2)
# plt.title(r"$8 < \log M_* < 9$")

# plt.show()


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
# xbins = np.arange(7.0, 12.1, 0.1)
# ybins = np.arange(-3.0, 3.1, 0.1)

# ratio_map, xedge, yedge, _ = (
#     binned_statistic_2d(
#         logM[mask],
#         logSFR[mask],
#         ratio[mask],
#         statistic="median", # 必要に応じて "mean" や "count" に変更可能
#         bins=[xbins, ybins]
#     )
# )

# count_map, _, _, _ = (
#     binned_statistic_2d(
#         logM[mask],
#         logSFR[mask],
#         ratio[mask],
#         statistic="count",
#         bins=[xbins, ybins]
#     )
# )

# fig, ax = plt.subplots(figsize=(8,6))
# fig.subplots_adjust(left=0.15, right=0.85, bottom=0.15, top=0.85)

# # vmin = np.nanpercentile(ratio_map,5) # 下限を5パーセンタイルに設定（必要に応じて調整）
# # vmax = np.nanpercentile(ratio_map,95) # 上限を95パーセンタイルに設定（必要に応じて調整）

# vmin=1.0
# vmax=1.5

# # 少数セルを除去
# NMIN_CELL = 10

# ratio_map_masked = ratio_map.copy()
# ratio_map_masked[count_map < NMIN_CELL] = np.nan

# plt.pcolormesh(
#     xedge,
#     yedge,
#     ratio_map_masked.T,
#     vmin=vmin, vmax=vmax,  # カラーマップの範囲を固定（必要に応じて調整）
#     shading="auto",
#     cmap="viridis"
# )

# plt.colorbar()

# plt.xlabel(r'$\log M_*$')
# plt.ylabel(r'$\log SFR$')

# for spine in ax.spines.values():
#     spine.set_linewidth(2)
# plt.tight_layout()
# # 保存
# fig_dir = os.path.join(current_dir, "results/figure")
# os.makedirs(fig_dir, exist_ok=True)
# save_path_heat = os.path.join(
#     fig_dir,
#     "heat_sm_sfr_sii_ratio_sdss_median.png"
# )

# plt.savefig(save_path_heat)
# print(f"Saved heatmap to: {save_path_heat}")
# plt.show()


# xbins = np.arange(7.0, 12.1, 0.1)
# ybins = np.arange(-3.0, 3.1, 0.1)

# ratio_map, xedge, yedge, _ = (
#     binned_statistic_2d(
#         logM[mask],
#         logSFR[mask],
#         ratio[mask],
#         statistic="mean", # 必要に応じて "mean" や "count" に変更可能
#         bins=[xbins, ybins]
#     )
# )

# count_map, _, _, _ = (
#     binned_statistic_2d(
#         logM[mask],
#         logSFR[mask],
#         ratio[mask],
#         statistic="count",
#         bins=[xbins, ybins]
#     )
# )

# fig, ax = plt.subplots(figsize=(8,6))
# fig.subplots_adjust(left=0.15, right=0.85, bottom=0.15, top=0.85)

# # vmin = np.nanpercentile(ratio_map,5) # 下限を5パーセンタイルに設定（必要に応じて調整）
# # vmax = np.nanpercentile(ratio_map,95) # 上限を95パーセンタイルに設定（必要に応じて調整）

# vmin=1.0
# vmax=1.5

# # 少数セルを除去
# NMIN_CELL = 10 # 1: maskしない

# ratio_map_masked = ratio_map.copy()
# ratio_map_masked[count_map < NMIN_CELL] = np.nan

# plt.pcolormesh(
#     xedge,
#     yedge,
#     ratio_map_masked.T,
#     vmin=vmin, vmax=vmax,  # カラーマップの範囲を固定（必要に応じて調整）
#     shading="auto",
#     cmap="viridis"
# )

# plt.colorbar()

# plt.xlabel(r'$\log M_*$')
# plt.ylabel(r'$\log SFR$')

# for spine in ax.spines.values():
#     spine.set_linewidth(2)
# plt.tight_layout()
# # 保存
# fig_dir = os.path.join(current_dir, "results/figure")
# os.makedirs(fig_dir, exist_ok=True)
# save_path_heat = os.path.join(
#     fig_dir,
#     "heat_sm_sfr_sii_ratio_sdss_mean.png"
# )

# plt.savefig(save_path_heat)
# print(f"Saved heatmap to: {save_path_heat}")
# plt.show()


# # 6. 各セルの銀河数も確認

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
# plt.ylabel(r'$\log SFR$')

# for spine in ax.spines.values():
#     spine.set_linewidth(2)
# plt.tight_layout()

# save_path_number = os.path.join(
#     fig_dir,
#     "heat_sm_sfr_galaxy_number_sdss.png"
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

# ============================================================
# 0. 設定（結果を見る前に決めて固定する）
# ============================================================
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)   # 論文で採用する宇宙論
UNIT_FLUX = 1e-17                        # erg s^-1 cm^-2
Z_MAX = 0.1825
L_CUT = 1e39                             # erg s^-1

# main sequence の帯（Renzini & Peng 2015 の目安）
MS_SLOPE, MS_ZP, MS_WIDTH = 0.76, -7.64, 0.6

COMP_THRESH = 0.9                        # 完全性の基準
NMIN_CELL = 10                           # これより少ないビンは使わない

# Rₑ の条件：半径のカタログを取り直した後は True にして、
# 全パネル（SFR、sSFR、ΣSFR）で同じ親サンプルを使う
REQUIRE_RE = False

fig_dir = "results/figure/completeness"
os.makedirs(fig_dir, exist_ok=True)

# ============================================================
# 1. カタログを読む（光度のカット前のもの）
# ============================================================
path = "results/fits/mpajhu_dr7_v5_2_merged_radius.fits"
df = Table.read(path, hdu=1).to_pandas()

z      = df["Z"].values
logM   = df["sm_MEDIAN"].values
logSFR = df["sfr_MEDIAN"].values

def flux(name):
    return (df[f"{name}_FLUX"].values * UNIT_FLUX,
            df[f"{name}_FLUX_ERR"].values * UNIT_FLUX)

F_Hb,  E_Hb  = flux("H_BETA")
F_O3,  E_O3  = flux("OIII_5007")
F_Ha,  E_Ha  = flux("H_ALPHA")
F_N2,  E_N2  = flux("NII_6584")
F_S31, E_S31 = flux("SII_6731")

# ============================================================
# 2. 条件の定義
# ============================================================
# 2a. 親サンプル（輝線の条件は入れない）
base = (np.isfinite(z) & (z > 0) & (z < Z_MAX) &
        np.isfinite(logM) & np.isfinite(logSFR))

if REQUIRE_RE:
    deV, expR, fdev = (df[c].values for c in ["deVRad_r", "expRad_r", "fracDeV_r"])
    Re_arcsec = np.where(fdev > 0.5, deV, expR)       # 1.678 倍はしない
    base &= (np.isfinite(Re_arcsec) & (Re_arcsec > 0) &
             np.isfinite(fdev) & (fdev >= 0) & (fdev <= 1))

# 重複を除く列ができたら、ここで base に加える
# base &= df["is_primary"].values.astype(bool)

parent = base

# 2b. 輝線にもとづく3つの選択
with np.errstate(divide="ignore", invalid="ignore"):
    sn_ok = ((F_Hb / E_Hb >= 3) & (F_O3 / E_O3 >= 3) &
             (F_Ha / E_Ha >= 3) & (F_N2 / E_N2 >= 3))
    x = np.log10(F_N2 / F_Ha)                           # log([NII]/Ha)
    y = np.log10(F_O3 / F_Hb)                           # log([OIII]/Hb)
    sf_ka03 = (x < 0.05) & (y < 0.61 / (x - 0.05) + 1.3)

dL = cosmo.luminosity_distance(np.clip(z, 1e-4, None)).to(u.cm).value
with np.errstate(invalid="ignore"):
    L6731 = 4 * np.pi * dL**2 * F_S31
lum_ok = np.isfinite(L6731) & (L6731 > L_CUT)

cond_sn  = sn_ok                       # 1. S/N
cond_bpt = sn_ok & sf_ka03             # 2. + BPT
cond_all = sn_ok & sf_ka03 & lum_ok    # 3. + 光度のカット

selected = parent & cond_all

# 2c. main sequence の帯（完全性を数える領域。最終サンプルの選択には使わない）
dMS = logSFR - (MS_SLOPE * logM + MS_ZP)
on_ms = np.abs(dMS) < MS_WIDTH

# ============================================================
# 3. 完全性を M* の関数にする → M_MIN を決める
# ============================================================
def frac_in_bins(x, num_mask, den_mask, bins):
    n_den, _ = np.histogram(x[den_mask], bins=bins)
    n_num, _ = np.histogram(x[num_mask], bins=bins)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = n_num / n_den
    f = f.astype(float)
    f[n_den < NMIN_CELL] = np.nan
    return f, n_den

def lowest_complete_edge(bins, f, thresh):
    """そのビンより上のすべての有効なビンが基準を満たす、最も低いビンの下端"""
    for i in range(len(f)):
        if not np.isfinite(f[i]):
            continue
        rest = f[i:][np.isfinite(f[i:])]
        if np.all(rest >= thresh):
            return bins[i]
    return np.nan

mbins = np.arange(8.0, 11.51, 0.2)
mcen = 0.5 * (mbins[1:] + mbins[:-1])

den = parent & on_ms
f_sn,  n_den = frac_in_bins(logM, den & cond_sn,  den, mbins)
f_bpt, _     = frac_in_bins(logM, den & cond_bpt, den, mbins)
f_all, _     = frac_in_bins(logM, den & cond_all, den, mbins)

# 判断用：BPT を通った星形成銀河のうち、光度のカットも通った割合
f_L, n_bpt = frac_in_bins(logM, den & cond_all, den & cond_bpt, mbins)

M_MIN = lowest_complete_edge(mbins, f_L, COMP_THRESH)
print(f"\n採用する質量の下限 M_MIN = {M_MIN:.1f}"
      f"（光度のカットによる完全性 >= {COMP_THRESH}）")

# S/N による欠けが M_MIN より上で小さいかを確認
above = mcen >= M_MIN
if np.any(f_sn[above] < COMP_THRESH):
    print("注意：M_MIN より上に、S/N の条件で基準を下回るビンがあります")

# 選択による SFR の偏り（main sequence 上、BPT を通った銀河の中で）
dmed = []
for lo, hi in zip(mbins[:-1], mbins[1:]):
    inb = den & cond_bpt & (logM >= lo) & (logM < hi)
    if inb.sum() < NMIN_CELL or (inb & lum_ok).sum() < NMIN_CELL:
        dmed.append(np.nan); continue
    dmed.append(np.median(logSFR[inb & lum_ok]) - np.median(logSFR[inb]))
dmed = np.array(dmed)

print("\nlogM   N(MS)   S/N   +BPT  +L    L|BPT  ΔmedSFR")
for m, n, a, b, c, d, e in zip(mcen, n_den, f_sn, f_bpt, f_all, f_L, dmed):
    print(f"{m:4.1f} {n:7d}  {a:.2f}  {b:.2f}  {c:.2f}  {d:.2f}  {e:+.3f}")

fig, ax = plt.subplots(figsize=(8, 5.5))
ax.plot(mcen, f_sn,  "o-", label=r"S/N $\geq$ 3")
ax.plot(mcen, f_bpt, "o-", label="+ BPT (Ka03)")
ax.plot(mcen, f_all, "o-", label=r"+ $L$([S II]) > $10^{39}$")
ax.plot(mcen, f_L,   "s--", color="k", label=r"$L$ cut | BPT SF")
ax.axhline(COMP_THRESH, ls=":", color="gray")
ax.axvline(M_MIN, ls=":", color="c")
ax.set_xlabel(r"$\log(M_*/M_\odot)$")
ax.set_ylabel("Fraction of MS galaxies retained")
ax.set_ylim(0, 1.05)
ax.legend(fontsize=14, loc="lower right")
save = os.path.join(fig_dir, "completeness_vs_mass_steps.png")
plt.savefig(save, bbox_inches="tight"); plt.show()
print(f"Saved: {save}")

# ============================================================
# 4. 数の地図と完全性の地図（M*–SFR）
#    完全性 = 選択後 / 親サンプル（親サンプルは輝線の条件なし）
# ============================================================
xbins = np.arange(7.0, 12.01, 0.1)
ybins = np.arange(-3.0, 3.01, 0.1)

N_par, _, _ = np.histogram2d(logM[parent],   logSFR[parent],   bins=[xbins, ybins])
N_sel, _, _ = np.histogram2d(logM[selected], logSFR[selected], bins=[xbins, ybins])
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

mm = np.linspace(xbins[0], xbins[-1], 100)
ms = MS_SLOPE * mm + MS_ZP
for ax in axes:
    ax.plot(mm, ms, color="w", lw=1.5)
    ax.plot(mm, ms - MS_WIDTH, color="w", lw=1, ls="--")
    ax.plot(mm, ms + MS_WIDTH, color="w", lw=1, ls="--")
    ax.axvline(M_MIN, color="c", lw=1.5, ls=":")
    ax.set_xlim(xbins[0], xbins[-1]); ax.set_ylim(ybins[0], ybins[-1])
    ax.set_xlabel(r"$\log(M_*/M_\odot)$")
axes[0].set_ylabel(r"$\log({\rm SFR}/M_\odot\,{\rm yr^{-1}})$")

save = os.path.join(fig_dir, "completeness_map_sm_sfr.png")
plt.savefig(save, bbox_inches="tight"); plt.show()
print(f"Saved: {save}")

# ============================================================
# 5. 完全性を SFR の関数にする（M* >= M_MIN、main sequence 上）
#    段階ごとに分けて描く
# ============================================================
in_mass = logM >= M_MIN
den_s = parent & on_ms & in_mass

fbins = np.arange(-1.5, 2.51, 0.2)
fcen = 0.5 * (fbins[1:] + fbins[:-1])

fs_sn,  n_fs = frac_in_bins(logSFR, den_s & cond_sn,  den_s, fbins)
fs_bpt, _    = frac_in_bins(logSFR, den_s & cond_bpt, den_s, fbins)
fs_all, _    = frac_in_bins(logSFR, den_s & cond_all, den_s, fbins)
fs_L,   _    = frac_in_bins(logSFR, den_s & cond_all, den_s & cond_bpt, fbins)

fig, ax = plt.subplots(figsize=(8, 5.5))
ax.plot(fcen, fs_sn,  "o-", label=r"S/N $\geq$ 3")
ax.plot(fcen, fs_bpt, "o-", label="+ BPT (Ka03)")
ax.plot(fcen, fs_all, "o-", label=r"+ $L$([S II]) > $10^{39}$")
ax.plot(fcen, fs_L,   "s--", color="k", label=r"$L$ cut | BPT SF")
ax.axhline(COMP_THRESH, ls=":", color="gray")
ax.set_xlabel(r"$\log({\rm SFR}/M_\odot\,{\rm yr^{-1}})$")
ax.set_ylabel(rf"Fraction of MS galaxies retained ($\log M_* \geq {M_MIN:.1f}$)")
ax.set_ylim(0, 1.05)
ax.legend(fontsize=14, loc="lower left")
save = os.path.join(fig_dir, "completeness_vs_sfr_steps.png")
plt.savefig(save, bbox_inches="tight"); plt.show()
print(f"Saved: {save}")

print("\nlogSFR  N(MS)   S/N   +BPT  +L    L|BPT")
for f_, n_, a, b, c, d in zip(fcen, n_fs, fs_sn, fs_bpt, fs_all, fs_L):
    print(f"{f_:5.1f} {n_:6d}  {a:.2f}  {b:.2f}  {c:.2f}  {d:.2f}")

# ============================================================
# 6. 選択の流れの数（2章の表に使う）
# ============================================================
final = selected & (logM >= M_MIN)
flow = [
    ("親サンプル（z, M*, SFR" + (", Rₑ" if REQUIRE_RE else "") + "）", parent),
    ("+ 4本の S/N >= 3",          parent & cond_sn),
    ("+ BPT（Kauffmann+03）",     parent & cond_bpt),
    ("+ L([SII]6731) > 1e39",     parent & cond_all),
    (f"+ log M* >= {M_MIN:.1f}",  final),
]
print("\n選択の流れ")
for name, m in flow:
    print(f"  {name:32s}: {m.sum():,}")