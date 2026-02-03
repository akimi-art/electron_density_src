#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
スクリプトの概要:
このスクリプトは
JADESのカタログを使って
スペクトル (SII) をフィッティングするものです。

使用方法:
    JADES_spectra_fit_SII.py [オプション]

著者: A. M.
作成日: 2026-02-03

参考文献:
    - PEP 8: https://peps.python.org/pep-0008/
    - PEP 257 (Docstring規約): https://peps.python.org/pep-0257/
    - Python公式ドキュメント: https://docs.python.org/ja/3/
"""

# == 必要なパッケージのインストール == #
import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import ZScaleInterval
from scipy.optimize import curve_fit
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


# =========================
# CSV を読む
# =========================
current_dir = os.getcwd()
csv_path = os.path.join(current_dir, "results/csv/JADES_ne_candidates.csv")
df = pd.read_csv(csv_path)

# df.iloc[0]: 「1行目（最初の1天体）」を取り出す
nir_id = df.iloc[1]["NIRSpec_ID"] 
nir_id_str = f"{int(nir_id):08d}"
z_spec = df.iloc[1]["z_Spec"]
z_spec_str = f"{float(z_spec):.3f}"


# =========================
# 基本設定
# =========================
wave_length_6716 = 6716.440  # Å
wave_length_6730 = 6730.820  # Å
delta_lambda = 100.0          # fit 幅（Å）
sigma_instr = 10.0            # 固定（後で grating 依存にしてOK）
filter_grating = "f170lp-g235m" # ここにフィルターグレーティング情報を追加

# =========================
# Gaussian
# =========================
def gaussian(x, amp, mu, sigma):
    return amp * np.exp(-(x-mu)**2/(2*sigma**2)) / (np.sqrt(2*np.pi)*sigma)

# =========================
# SII model（z 固定）
# =========================
def s2_doublet_model(x, amp_6716, amp_6730,  z, sigma_int, bg):
    mu_6716 = wave_length_6716 * (1 + z)
    mu_6730 = wave_length_6730 * (1 + z)
    sigma_total = np.sqrt(sigma_int**2 + sigma_instr**2)
    f6716 = gaussian(x, amp_6716, mu_6716, sigma_total)
    f6730 = gaussian(x, amp_6730, mu_6730, sigma_total)

    return f6716 + f6730 + bg

def s2_doublet_model_6716(x, amp_6716, amp_6730, z, sigma_int, bg):
    mu_6716 = wave_length_6716 * (1 + z)
    mu_6730 = wave_length_6730 * (1 + z)
    sigma_total = np.sqrt(sigma_int**2 + sigma_instr**2)
    f6716 = gaussian(x, amp_6716, mu_6716, sigma_total)
    f6730 = gaussian(x, amp_6730, mu_6730, sigma_total)

    return f6716 + bg

def s2_doublet_model_6730(x, amp_6716, amp_6730, z, sigma_int, bg):
    mu_6716 = wave_length_6716 * (1 + z)
    mu_6730 = wave_length_6730 * (1 + z)
    sigma_total = np.sqrt(sigma_int**2 + sigma_instr**2)
    f6716 = gaussian(x, amp_6716, mu_6716, sigma_total)
    f6730 = gaussian(x, amp_6730, mu_6730, sigma_total)

    return f6730 + bg

# =========================
# 1. CSV 読み込み
# =========================
df = pd.read_csv("results/csv/JADES_ne_candidates.csv")

row = df.iloc[1]
nir_id = int(row["NIRSpec_ID"])
z_fix = row["z_Spec"]

nir_id_str = f"{nir_id:08d}"

# =========================
# 2. スペクトル取得
# =========================
base = f"results/JADES/individual/JADES_{nir_id_str}/HLSP"

x1d = glob.glob(f"{base}/**/*_x1d.fits", recursive=True)[2] # ここでフィルターグレーディングを調整する
s2d = glob.glob(f"{base}/**/*_s2d.fits", recursive=True)[2] # ここでフィルターグレーディングを調整する

# =========================
# 3. 1D スペクトル
# =========================
with fits.open(x1d) as hdul:
    tab = hdul["EXTRACT1D"].data
    wave_1d = tab["WAVELENGTH"] * 1e4
    flux_1d = tab["FLUX"] * 1e19
    err_1d  = tab["FLUX_ERR"] * 1e19

# =========================
# 4. 2D スペクトル
# =========================
with fits.open(s2d) as hdul:
    flux_2d = hdul["FLUX"].data
    wave_2d = hdul["WAVELENGTH"].data * 1e4

# =========================
# 5. mask 定義（z 使用）
# =========================
wave_center_s2 = 0.5 * (wave_length_6716 + wave_length_6730) * (1 + z_fix)

mask_1d = (
    (wave_1d > wave_center_s2 - delta_lambda) &
    (wave_1d < wave_center_s2 + delta_lambda)
)

print("SII center =", wave_center_s2)
print("wave_2d range =", wave_2d.min(), wave_2d.max())

x_fit = wave_1d[mask_1d]
y_fit = flux_1d[mask_1d]
yerr_fit = err_1d[mask_1d]

# =========================
# 6. フィッティング
# =========================
# === 最適化パラメータの初期値を設定する ===
######################################################
wave_center_s2 = 25600 # これも変更していく, Å, 目視で確認
######################################################
amplitude_6716_init = 20
amplitude_6730_init = 20
z_init = (wave_center_s2 / ((wave_length_6716 + wave_length_6730)/2)) - 1 # 変更
sigma_int_init = 10 # 適当, 目安がわからないのでLSFに合わせた
bgd_s2_mask_init = 0
p0 = [amplitude_6716_init, amplitude_6730_init, z_init, sigma_int_init, bgd_s2_mask_init]

popt, pcov = curve_fit(
    s2_doublet_model,
    x_fit, y_fit,
    p0=p0,
    sigma=yerr_fit,
    absolute_sigma=True
)

amp_6716, amp_6730, z, sigma_int, bg = popt

ratio = amp_6716 / amp_6730
print(f"[S II] 6716/6730 = {ratio:.3f}")

# =========================
# 7. プロット
# =========================
fig = plt.figure(figsize=(12,6))
gs = GridSpec(2,1,height_ratios=[1,5],hspace=0)

ax2d = fig.add_subplot(gs[0])
ax1d = fig.add_subplot(gs[1], sharex=ax2d)

# ---- 2D ----
flux_2d_cut = flux_2d[:, mask_1d]
wave_2d_cut = wave_2d[mask_1d]

zscale = ZScaleInterval()
vmin, vmax = zscale.get_limits(flux_2d_cut[np.isfinite(flux_2d_cut)])

ax2d.imshow(
    flux_2d_cut,
    origin="lower",
    aspect="auto",
    cmap="plasma",
    vmin=vmin, vmax=vmax,
    extent=[wave_2d_cut.min(), wave_2d_cut.max(), 0, flux_2d_cut.shape[0]]
)

# ---- 1D ----
ax1d.step(x_fit, y_fit, where="mid", color="black")
ax1d.fill_between(
    x_fit,
    y_fit-yerr_fit,
    y_fit+yerr_fit,
    step="mid",
    color="gray", alpha=0.4
)

x_model = np.linspace(x_fit.min(), x_fit.max(), 1000)
ax1d.plot(x_model, s2_doublet_model(x_model, *popt), color="red", lw=2)
ax1d.plot(x_model, s2_doublet_model_6716(x_model, *popt), color="red", lw=2, ls="--", label="SII 6716")
ax1d.plot(x_model, s2_doublet_model_6730(x_model, *popt), color="red", lw=2, ls="-.", label="SII 6730")
ax1d.legend(fontsize=16)
mu_6716 = wave_length_6716 * (1 + z)
mu_6730 = wave_length_6730 * (1 + z)
ax1d.axvline(mu_6716, color="red", ls="--")
ax1d.axvline(mu_6730, color="red", ls="-.")
ax1d.set_xlabel(r'$\lambda (Å)$')
ax1d.set_ylabel(r'F$_{\lambda}$ ($10^{-19}$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$)')
# === 枠線 (spines) の設定 ===
# 線の太さ・色・表示非表示などを個別に制御
for spine in ax1d.spines.values():
    spine.set_linewidth(2)       # 枠線の太さ
    spine.set_color("black")     # 枠線の色

ax2d.set_title(f"JADES NIRSpec {filter_grating} |  ID {nir_id_str} | z_spec = {z_spec_str}")
ax2d.tick_params(axis='both', which='both',
               bottom=False, top=False, left=False, right=False,
               labelbottom=False, labeltop=False,
               labelleft=False, labelright=False)
# === 枠線 (spines) の設定 ===
# 線の太さ・色・表示非表示などを個別に制御
for spine in ax2d.spines.values():
    spine.set_linewidth(2)       # 枠線の太さ
    spine.set_color("black")     # 枠線の色

save_path = os.path.join(current_dir, f"results/figure/JADES/JADES_NIRSpec_{filter_grating}_ID{nir_id_str}_fit.png")
plt.savefig(save_path)
print(f"Saved as {save_path}")
plt.show()
