#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
スクリプトの概要:
このスクリプトは
SDSSのスペクトルをフィットする
モジュールです。

使用方法:
    SDSS_spectra_draw.py [オプション]

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

# ===============================
# 入力
# ===============================
filename = "./results/SDSS/spectra/sdss_spectro_0275-51910-0141/spec-0275-51910-0141.fits"
z=0.0818
hb_vac = 4862.683  # Å (vacuum)

# ===============================
# FITS 読み込み
# ===============================
hdul = fits.open(filename)
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
A_init  = np.max(flux_fit) - np.median(flux_fit)
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
# 描画
# ===============================
lam_dense = np.linspace(lam_fit.min(), lam_fit.max(), 1000)
model_dense = hb_model(lam_dense, *popt)
continuum_dense = c0 + c1 * lam_dense

residual = flux_fit - hb_model(lam_fit, *popt)

plt.figure(figsize=(12,6))
# =================================================
# 内側2段構造（Spectrum上・Residual下）
# =================================================
inner_gs = outer_gs[row, col].subgridspec(
    2, 1,
    height_ratios=[4, 1],
    hspace=0
)

ax_spec = fig.add_subplot(inner_gs[0])
ax_res  = fig.add_subplot(inner_gs[1], sharex=ax_spec)

# =================================================
# 上段：スペクトル + フィット
# =================================================

# データ
ax_spec.step(lam_fit, flux_fit, where='mid',
             color='black', lw=0.8)

ax_spec.fill_between(
    lam_fit,
    flux_fit - sigma_fit,
    flux_fit + sigma_fit,
    step='mid',
    color='gray',
    alpha=0.3
)

# フィット
ax_spec.plot(lam_dense, model_dense,
             color='red', lw=1.5)

# 中心線
ax_spec.axvline(lam0, color='blue',
                linestyle='--', lw=1)

# 目盛り整理
ax_spec.tick_params(axis='x',
                    bottom=False,
                    labelbottom=False)

# Y軸非表示にしたいなら
# ax_spec.tick_params(axis='y', left=False, labelleft=False)

ax_spec.set_ylabel(r'F$_{\lambda}$')

# =================================================
# 下段：残差
# =================================================

residual = flux_fit - hb_model(lam_fit, *popt)

ax_res.axhline(0, color='black', lw=0.8)
ax_res.step(lam_fit, residual,
            where='mid', color='black', lw=0.8)

# ±1σ目安線
ax_res.fill_between(
    lam_fit,
    -sigma_fit,
    sigma_fit,
    step='mid',
    color='gray',
    alpha=0.2
)

ax_res.set_xlabel("Observed wavelength (Å)")
ax_res.set_ylabel("Res.")

# 左端以外はY軸消すなら
if col != 0:
    ax_res.tick_params(labelleft=False)
    ax_spec.tick_params(labelleft=False)

# 最下段以外はX軸消すなら
if row != n_rows - 1:
    ax_res.tick_params(labelbottom=False)
