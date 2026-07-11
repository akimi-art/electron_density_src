#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
スクリプトの概要:
このスクリプトは
SDSSのスペクトルをフィットする
モジュールです。

使用方法:
    SDSS_spectra_fit.py [オプション]

著者: A. M.
作成日: 2026-02-20

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
filename = "data/data_SDSS/DR7/spectra/fit/spSpec-51609-0292-084.fit"
z = 0.0818
hb_vac = 4862.683  # vacuum Å

sii_6716_vac = 6716.440  # Å
sii_6731_vac = 6730.820  # Å

# ===============================
# FITS 読み込み
# ===============================
with fits.open(filename) as hdul:
    data = hdul[1].data

loglam = data['loglam']
flux   = data['flux']
ivar   = data['ivar']

lam = 10**loglam

# ===============================
# エラー
# ===============================
sigma = np.zeros_like(ivar)
mask_good = ivar > 0
sigma[mask_good] = 1/np.sqrt(ivar[mask_good])
sigma[~mask_good] = np.inf

# ===============================
# フィット範囲
# ===============================
lam0 = hb_vac * (1 + z)
window = 30

region = (lam > lam0 - window) & (lam < lam0 + window) & mask_good

lam_fit   = lam[region]
flux_fit  = flux[region]
sigma_fit = sigma[region]

# ===============================
# モデル（中心固定）
# ===============================
def hb_model(lam, A, sig, c0, c1):
    return (
        A * np.exp(-(lam - lam0)**2 / (2 * sig**2))
        + c0
        + c1 * lam
    )

# ===============================
# 初期値
# ===============================
A_init   = np.max(flux_fit) - np.median(flux_fit)
sig_init = 2.0
c0_init  = np.median(flux_fit)
c1_init  = 0.0

p0 = [A_init, sig_init, c0_init, c1_init]

# ===============================
# フィット
# ===============================
popt, pcov = curve_fit(
    hb_model,
    lam_fit,
    flux_fit,
    sigma=sigma_fit,
    p0=p0,
    absolute_sigma=True
)

A, sig, c0, c1 = popt
perr = np.sqrt(np.diag(pcov))

# フラックス
hb_flux = A * sig * np.sqrt(2*np.pi)
dA, dsig = perr[0], perr[1]
hb_flux_err = hb_flux * np.sqrt((dA/A)**2 + (dsig/sig)**2)

# ===============================
# 描画（単独プロット）
# ===============================
plt.figure(figsize=(16,8))

plt.step(lam_fit, flux_fit, where='mid',
         color='black', lw=0.8)

plt.fill_between(
    lam_fit,
    flux_fit - sigma_fit,
    flux_fit + sigma_fit,
    step='mid',
    color='gray',
    alpha=0.3
)

lam_dense = np.linspace(lam_fit.min(), lam_fit.max(), 1000)
plt.plot(lam_dense, hb_model(lam_dense, *popt),
         color='red', lw=1.5)

plt.axvline(lam0, color='blue', linestyle='--')

plt.xlabel(r'$\lambda (Å)$')
plt.ylabel(r'F$_{\lambda}$ ($10^{-19}$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$)')

plt.tight_layout()
plt.show()

# ===============================
# 出力
# ===============================
print("Hβ flux =", hb_flux)
print("Hβ flux error =", hb_flux_err)
print("sigma_g =", sig)
print("FWHM =", 2.355 * sig)