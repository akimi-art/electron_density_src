#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
スクリプトの概要:
このスクリプトは
SDSSのスペクトルをフィットする
モジュールです。

使用方法:
    SDSS_spectra_fit_v1.py [オプション]

著者: A. M.
作成日: 2026-02-24

参考文献:
    - PEP 8: https://peps.python.org/pep-0008/
    - PEP 257 (Docstring規約): https://peps.python.org/pep-0257/
    - Python公式ドキュメント: https://docs.python.org/ja/3/
"""


# == 必要なパッケージのインポート == #
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.optimize import curve_fit
from matplotlib.gridspec import GridSpec


# 軸の設定
plt.rcParams.update({
    # --- 図全体 ---
    "figure.figsize": (12, 6),       # 図サイズ
    "font.size": 16,                 # 全体フォントサイズ
    "axes.labelsize": 16,            # 軸ラベルのサイズ
    "axes.titlesize": 16,            # タイトルのサイズ
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
    "xtick.labelsize": 16,           # x軸ラベルサイズ
    "ytick.labelsize": 16,           # y軸ラベルサイズ

    # --- フォント ---
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
})

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.optimize import curve_fit

# ===============================
# 入力
# ===============================
filename = "./results/SDSS/spectra/sdss_spectro_0275-51910-0141/spec-0275-51910-0141.fits"

z_init = 0.0818   # 初期値

sii_6716_vac = 6716.440
sii_6731_vac = 6730.820

# ===============================
# FITS読み込み
# ===============================
with fits.open(filename) as hdul:
    data = hdul[1].data

lam = 10**data['loglam']
flux = data['flux']
ivar = data['ivar']

sigma = np.zeros_like(ivar)
good = ivar > 0
sigma[good] = 1/np.sqrt(ivar[good])
sigma[~good] = np.inf

# ===============================
# フィット範囲（広めに）
# ===============================
lam_guess1 = sii_6716_vac * (1 + z_init)
lam_guess2 = sii_6731_vac * (1 + z_init)

window = 50
region = (
    (lam > lam_guess1 - window) &
    (lam < lam_guess2 + window) &
    good
)

lam_fit = lam[region]
flux_fit = flux[region]
sigma_fit = sigma[region]

# ===============================
# モデル（z含む）
# ===============================
def sii_model(lam, A1, A2, sig, z, c0, c1):
    lam1 = sii_6716_vac * (1 + z)
    lam2 = sii_6731_vac * (1 + z)

    g1 = A1 * np.exp(-(lam - lam1)**2 / (2*sig**2))
    g2 = A2 * np.exp(-(lam - lam2)**2 / (2*sig**2))

    return g1 + g2 + c0 + c1*lam

def sii_model_6716(lam, A1, A2, sig, z, c0, c1):
    lam1 = sii_6716_vac * (1 + z)
    lam2 = sii_6731_vac * (1 + z)

    g1 = A1 * np.exp(-(lam - lam1)**2 / (2*sig**2))
    g2 = A2 * np.exp(-(lam - lam2)**2 / (2*sig**2))

    return g1 + c0 + c1*lam

def sii_model_6731(lam, A1, A2, sig, z, c0, c1):
    lam1 = sii_6716_vac * (1 + z)
    lam2 = sii_6731_vac * (1 + z)

    g1 = A1 * np.exp(-(lam - lam1)**2 / (2*sig**2))
    g2 = A2 * np.exp(-(lam - lam2)**2 / (2*sig**2))

    return g2 + c0 + c1*lam

# ===============================
# 初期値
# ===============================
A1_init = np.max(flux_fit) - np.median(flux_fit)
A2_init = A1_init * 0.8
sig_init = 2.0
c0_init = np.median(flux_fit)
c1_init = 0.0

p0 = [A1_init, A2_init, sig_init, z_init, c0_init, c1_init]

# ===============================
# フィット
# ===============================
popt, pcov = curve_fit(
    sii_model,
    lam_fit,
    flux_fit,
    sigma=sigma_fit,
    p0=p0,
    absolute_sigma=True
)

A1, A2, sig, z_fit, c0, c1 = popt
perr = np.sqrt(np.diag(pcov))

# ===============================
# フラックス
# ===============================
F1 = A1 * sig * np.sqrt(2*np.pi)
F2 = A2 * sig * np.sqrt(2*np.pi)
ratio = F1 / F2

# ===============================
# 描画
# ===============================
fig, ax = plt.subplots(figsize=(8, 4))

ax.step(lam_fit, flux_fit, where='mid', color='black', lw=0.8)
ax.fill_between(
    lam_fit,
    flux_fit - sigma_fit,
    flux_fit + sigma_fit,
    step='mid',
    color='gray',
    alpha=0.3
)

lam_dense = np.linspace(lam_fit.min(), lam_fit.max(), 1000)
ax.plot(lam_dense, sii_model(lam_dense, *popt),
         color='red', lw=1.5)
ax.plot(lam_dense, sii_model_6716(lam_dense, *popt),
         color='red', lw=1.5, linestyle='--')
ax.plot(lam_dense, sii_model_6731(lam_dense, *popt),
         color='red', lw=1.5, linestyle='-.')
# フィットされた中心
lam1_fit = sii_6716_vac * (1 + z_fit)
lam2_fit = sii_6731_vac * (1 + z_fit)

plt.axvline(lam1_fit, color='red', linestyle='--')
plt.axvline(lam2_fit, color='red', linestyle='-.')

# plt.xlabel("Observed wavelength (Å)")
# plt.ylabel(r"F$_{\lambda}$")
# === 枠線 (spines) の設定 ===
# 線の太さ・色・表示非表示などを個別に制御
for spine in ax.spines.values():
    spine.set_linewidth(2)       # 枠線の太さ
    spine.set_color("black")     # 枠線の色
ax.set_xlabel(r'$\lambda (Å)$')
ax.set_ylabel(r'F$_{\lambda}$ ($10^{-19}$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$)')
# plt.title(f"[SII] doublet fit  z_fit = {z_fit:.6f}")

plt.tight_layout()
save_dir = "./results/SDSS/spectra/sdss_spectro_0275-51910-0141/spec-0275-51910-0141.png"
plt.savefig(save_dir)
print(f"Saved as {save_dir}")
plt.show()

# ===============================
# 出力
# ===============================
print("Fitted z =", z_fit)
print("SII 6716 flux =", F1)
print("SII 6731 flux =", F2)
print("Flux ratio (6716/6731) =", ratio)
print("Common sigma =", sig)
print("FWHM =", 2.355 * sig)