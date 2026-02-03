#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
スクリプトの概要:
このスクリプトは
JADESのカタログを使って
スペクトル（1d, 2d）を描画するものです。

使用方法:
    JADES_spectra_draw.py [オプション]

著者: A. M.
作成日: 2026-02-02

参考文献:
    - PEP 8: https://peps.python.org/pep-0008/
    - PEP 257 (Docstring規約): https://peps.python.org/pep-0257/
    - Python公式ドキュメント: https://docs.python.org/ja/3/
"""


# == 必要なパッケージのインストール == #
import re
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import ZScaleInterval
from matplotlib.gridspec import GridSpec

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


###########################################################################################################################################################
# REFERENCE WAVELENGTHS, IN VACUUM, USED IN SPECTRO1D (ref. "Table of Spectral Lines Used in SDSS", NIST Atomic Spectra Database (ASD))  #
# SDSSのURL: https://classic.sdss.org/dr6/algorithms/linestable.php
# NISTのURL: https://physics.nist.gov/PhysRefData/ASD/lines_form.html
# 他、Berg et al. 2021 Table
# URL: https://physics.nist.gov/cgi-bin/ASD/lines1.pl?spectra=O+III&output_type=0&low_w=1600&upp_w=1700&unit=0&submit=Retrieve+Data&de=0&plot_out=0&I_scale_type=1&format=0&line_out=0&en_unit=0&output=0&bibrefs=1&page_size=15&show_obs_wl=1&show_calc_wl=1&unc_out=1&order_out=0&max_low_enrg=&show_av=2&max_upp_enrg=&tsb_value=0&min_str=&A_out=0&intens_out=on&max_str=&allowed_out=1&forbid_out=1&min_accur=&min_intens=&conf_out=on&term_out=on&enrg_out=on&J_out=on
###########################################################################################################################################################

# 禁制線, ne診断に用いる
wavelength_o2_rest  = np.array([3727.092, 3729.875]) # O II (Å)
wavelength_s2_rest  = np.array([6716.440, 6730.820]) # S II (Å)

# 禁制線, ne診断に用いない
wavelength_o3_rest_4363, wavelength_o3_rest_4959, wavelength_o3_rest_5007 = [
    np.array([4364.436]),
    np.array([4960.295]),
    np.array([5008.240])
]
# O III 4363, 4959, 5007 (Å)

# 許容線
wavelength_ha, wavelength_hb, wavelength_hc, wavelength_hd = [
    np.array([6564.61]),
    np.array([4862.68]),
    np.array([4341.68]),
    np.array([4102.89])
]
###########################################################################################################################################################



# =========================
# 1. CSV を読む
# =========================
current_dir = os.getcwd()
csv_path = os.path.join(current_dir, "results/csv/JADES_ne_candidates.csv")
df = pd.read_csv(csv_path)

# df.iloc[0]: 「1行目（最初の1天体）」を取り出す
nir_id = df.iloc[1]["NIRSpec_ID"] 
nir_id_str = f"{int(nir_id):08d}"
z_spec = df.iloc[1]["z_Spec"]
z_spec_str = f"{float(z_spec):.3f}"


# Observed wavelengths (Å)
wavelength_o2_obs = wavelength_o2_rest * (1 + z_spec)
wavelength_s2_obs = wavelength_s2_rest * (1 + z_spec)

wavelength_o3_obs_4363 = wavelength_o3_rest_4363 * (1 + z_spec)
wavelength_o3_obs_4959 = wavelength_o3_rest_4959 * (1 + z_spec)
wavelength_o3_obs_5007 = wavelength_o3_rest_5007 * (1 + z_spec)

wavelength_ha_obs = wavelength_ha * (1 + z_spec)
wavelength_hb_obs = wavelength_hb * (1 + z_spec)

# Display all observed wavelengths
print("Observed Wavelengths (Å):")
print("O II:", wavelength_o2_obs)
print("S II:", wavelength_s2_obs)
print("O III 4363:", wavelength_o3_obs_4363)
print("O III 4959:", wavelength_o3_obs_4959)
print("O III 5007:", wavelength_o3_obs_5007)
print("Hα:", wavelength_ha_obs)
print("Hβ:", wavelength_hb_obs)


# =========================
# 2. スペクトルを探す
# =========================
base = os.path.join(current_dir, "results/JADES/individual/JADES_00003892/HLSP")

x1d = glob.glob(f"{base}/**/*{nir_id_str}*_x1d.fits", recursive=True)[1]
s2d = glob.glob(f"{base}/**/*{nir_id_str}*_s2d.fits", recursive=True)[1]

print("1D:", x1d)
print("2D:", s2d)

# =========================
# 3. 1D スペクトル
# =========================
with fits.open(x1d) as hdul:
    tab = hdul["EXTRACT1D"].data
    wave_1d = tab["WAVELENGTH"]
    wave_1d = wave_1d * 10000 # 横軸のスケーリング
    flux_1d = tab["FLUX"]
    flux_1d = flux_1d * 1e19 # 縦軸のスケーリング
    err_1d  = tab["FLUX_ERR"]
    err_1d = err_1d * 1e19   # 縦軸のスケーリング

# =========================
# 4. 2D スペクトル
# =========================
with fits.open(s2d) as hdul:
    flux_2d = hdul["FLUX"].data
    wave_2d = hdul["WAVELENGTH"].data
    wave_2d = wave_2d * 10000 # 横軸のスケーリング

# ZScale
zscale = ZScaleInterval()
vmin, vmax = zscale.get_limits(flux_2d[np.isfinite(flux_2d)])

# =========================
# 5. プロット（GridSpec）
# =========================
fig = plt.figure(figsize=(12, 6))
gs = GridSpec(
    2, 1,
    height_ratios=[1, 5],
    hspace=0.0
)

ax2d = fig.add_subplot(gs[0])
ax1d = fig.add_subplot(gs[1], sharex=ax2d)

# ---- 2D ----
im = ax2d.imshow(
    flux_2d,
    origin="lower",
    aspect="auto",
    cmap="plasma",
    vmin=vmin,
    vmax=vmax,
    extent=[wave_2d.min(), wave_2d.max(), 0, flux_2d.shape[0]]
)

ax2d.set_ylabel("")
ax2d.set_title(f"JADES NIRSpec f290lp-g395m |  ID {nir_id_str} | z_spec = {z_spec_str}")
ax2d.tick_params(axis='both', which='both',
               bottom=False, top=False, left=False, right=False,
               labelbottom=False, labeltop=False,
               labelleft=False, labelright=False)

# === 枠線 (spines) の設定 ===
# 線の太さ・色・表示非表示などを個別に制御
for spine in ax2d.spines.values():
    spine.set_linewidth(2)       # 枠線の太さ
    spine.set_color("black")     # 枠線の色

# ---- 1D ----
ax1d.step(
    wave_1d,
    flux_1d,
    where="mid",
    color='black',
    lw=1.0,
)

ax1d.fill_between(
    wave_1d,
    flux_1d - err_1d,
    flux_1d + err_1d,
    step="mid",
    color="gray",
    alpha=0.4,
)

ax1d.axvline(x=wavelength_o2_obs[0],   color='gray', linestyle='--', label='x = 3726')
ax1d.axvline(x=wavelength_o2_obs[1],   color='gray', linestyle='--', label='x = 3729')
ax1d.axvline(x=wavelength_s2_obs[0],   color='gray', linestyle='-.', label='x = 6716')
ax1d.axvline(x=wavelength_s2_obs[1],   color='gray', linestyle='-.', label='x = 6730')
ax1d.axvline(x=wavelength_o3_obs_4959, color='gray', linestyle=':',  label='x = 4959')
ax1d.axvline(x=wavelength_o3_obs_5007, color='gray', linestyle=':',  label='x = 5007')
ax1d.legend(fontsize=12)
ax1d.set_xlabel(r'$\lambda (Å)$')
ax1d.set_ylabel(r'F$_{\lambda}$ ($10^{-19}$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$)')
wave_min = min(wave_1d)
wave_max = max(wave_1d)
print(f"wavelength min = {wave_min}, max = {wave_max}")
ax1d.set_xlim(wave_min, wave_max)
# === 枠線 (spines) の設定 ===
# 線の太さ・色・表示非表示などを個別に制御
for spine in ax1d.spines.values():
    spine.set_linewidth(2)       # 枠線の太さ
    spine.set_color("black")     # 枠線の色

save_path = os.path.join(current_dir, f"results/figure/JADES/JADES_NIRSpec_f290lp-g395m_ID{nir_id_str}.png")
plt.savefig(save_path)
print(f"Saved as {save_path}")
plt.show()

print(min(wave_1d), max(wave_1d))
print(min(wave_2d), max(wave_2d))
