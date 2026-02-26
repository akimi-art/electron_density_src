#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
スクリプトの概要:
このスクリプトは
SDSSのスペクトルをフィットする
モジュールです。
2Dスペクトルを追加しています。

使用方法:
    SDSS_spectra_fit_v2.py [オプション]

著者: A. M.
作成日: 2026-02-26

参考文献:
    - PEP 8: https://peps.python.org/pep-0008/
    - PEP 257 (Docstring規約): https://peps.python.org/pep-0257/
    - Python公式ドキュメント: https://docs.python.org/ja/3/
"""


# == 必要なパッケージのインポート == #
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import ZScaleInterval
from scipy.optimize import curve_fit


# 軸の設定
plt.rcParams.update({
    # --- 図全体 ---
    "figure.figsize": (12, 6),       # 図サイズ
    "font.size": 20,                 # 全体フォントサイズ
    "axes.labelsize": 20,            # 軸ラベルのサイズ
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


# ===============================
# 入力
# ===============================
plate, mjd, fiber = 275, 51910, 141
spplate_fn = "./results/SDSS/spectra/2d/spPlate-0275-51910.fits"
specbs_fn  = "./results/SDSS/spectra/1d/sdss_spectro_0275-51910-0141/spec-0275-51910-0141.fits"

# [S II] 真空波長（Å）と z 初期値（表示窓）
sii_6716_vac = 6716.440
sii_6731_vac = 6730.820
z_init = 0.0818
win = 50.0  # Å（[S II] を挟む表示幅）

# ===============================
# 1) spPlate → fiber の 2D（Å）
# ===============================
with fits.open(spplate_fn) as hdul2d:
    flux2d = hdul2d[0].data              # (NFIBER, NPIX)
    hdr0   = hdul2d[0].header
    coeff0 = hdr0["COEFF0"]              # log10(λ) = COEFF0 + COEFF1 * pixel
    coeff1 = hdr0["COEFF1"]              # plate 内で等間隔（典型 1e-4）
    npix   = flux2d.shape[1]

    # 波長（Å, 真空）へ変換
    loglam = coeff0 + coeff1 * np.arange(npix)
    lam2d  = 10**loglam

    # 行 ↔ FIBERID を PLUGMAP で安全に一致
    fiber_ids = np.array(hdul2d[5].data["FIBERID"])
    row = int(np.where(fiber_ids == fiber)[0][0])

# fiber 1行だけを取り出す
cut2d = flux2d[row:row+1, :]

# [S II] の表示窓（Å）
lam1_init = sii_6716_vac * (1 + z_init)
lam2_init = sii_6731_vac * (1 + z_init)
lam_min = lam1_init - win
lam_max = lam2_init + win

# 2D を同じ窓でトリミング
mask_2d = (lam2d >= lam_min) & (lam2d <= lam_max)
lam2d_win = lam2d[mask_2d]
cut2d_win = cut2d[:, mask_2d]

# ===============================
# 2) 1D（spec-*.fits）を読み込み → 同じ窓に揃える
# ===============================
with fits.open(specbs_fn) as hdul1d:
    t = hdul1d[1].data                   # BINTABLE（loglam, flux, ivar）
    lam_1d  = 10**t["loglam"]            # Å（真空）
    flux_1d = t["flux"]
    ivar_1d = t["ivar"]

good = np.isfinite(ivar_1d) & (ivar_1d > 0)
sigma_1d = np.full_like(ivar_1d, np.inf, dtype=float)
sigma_1d[good] = 1.0 / np.sqrt(ivar_1d[good])

mask_1d = (lam_1d >= lam_min) & (lam_1d <= lam_max) & good
lam_fit   = lam_1d[mask_1d]
flux_fit  = flux_1d[mask_1d]
sigma_fit = sigma_1d[mask_1d]

# ===============================
# 3) [S II] ダブレット同時フィット（2ガウス＋線形連続光）
# ===============================
def sii_model(lam, A1, A2, sig, z, c0, c1):
    lam1 = sii_6716_vac * (1 + z)
    lam2 = sii_6731_vac * (1 + z)
    g1 = A1 * np.exp(-(lam - lam1)**2 / (2*sig**2))
    g2 = A2 * np.exp(-(lam - lam2)**2 / (2*sig**2))
    return g1 + g2 + c0 + c1*lam

def sii_model_6716(lam, A1, A2, sig, z, c0, c1):
    lam1 = sii_6716_vac * (1 + z)
    g1 = A1 * np.exp(-(lam - lam1)**2 / (2*sig**2))
    return g1 + c0 + c1*lam

def sii_model_6731(lam, A1, A2, sig, z, c0, c1):
    lam2 = sii_6731_vac * (1 + z)
    g2 = A2 * np.exp(-(lam - lam2)**2 / (2*sig**2))
    return g2 + c0 + c1*lam

# 初期値（素朴）
A1_init = np.max(flux_fit) - np.median(flux_fit)
A2_init = A1_init * 0.8
sig_init = 2.0
c0_init = np.median(flux_fit)
c1_init = 0.0
p0 = [A1_init, A2_init, sig_init, z_init, c0_init, c1_init]

popt, pcov = curve_fit(
    sii_model, lam_fit, flux_fit,
    sigma=sigma_fit, p0=p0, absolute_sigma=True, maxfev=20000
)
A1, A2, sig, z_fit, c0, c1 = popt
perr = np.sqrt(np.diag(pcov))

# フラックス積分（各ガウス）
F1 = A1 * sig * np.sqrt(2*np.pi)
F2 = A2 * sig * np.sqrt(2*np.pi)
ratio = F1 / F2

# ===============================
# 4) 図：上=2D（Å）/ 下=1D（Å, フィット重ね）
# ===============================
fig = plt.figure(figsize=(12, 6))
gs = fig.add_gridspec(2, 1, height_ratios=[1, 3], hspace=0.0)

# --- 上：2D ---
ax2d = fig.add_subplot(gs[0])
vmin, vmax = ZScaleInterval().get_limits(cut2d_win[np.isfinite(cut2d_win)])
ax2d.imshow(
    cut2d_win, origin="lower", aspect="auto", cmap="gray",
    vmin=vmin, vmax=vmax,
    extent=[lam2d_win.min(), lam2d_win.max(), fiber-0.5, fiber+0.5],
    interpolation="none",
)
# フィット中心（z_fit）に縦線
lam1_fit = sii_6716_vac * (1 + z_fit)
lam2_fit = sii_6731_vac * (1 + z_fit)
ax2d.axvline(lam1_fit, color="w", ls="--", lw=0.8, alpha=0.8)
ax2d.axvline(lam2_fit, color="w", ls="-.", lw=0.8, alpha=0.8)
ax2d.tick_params(axis="both", which="both",
                 bottom=False, top=False, left=False, right=False,
                 labelbottom=False, labelleft=False)

# --- 下：1D ---
ax1d = fig.add_subplot(gs[1], sharex=ax2d)
ax1d.step(lam_fit, flux_fit, where="mid", color="black", lw=0.9)
ax1d.fill_between(lam_fit, flux_fit - sigma_fit, flux_fit + sigma_fit,
                  step="mid", color="gray", alpha=0.35)

x_dense = np.linspace(lam_fit.min(), lam_fit.max(), 1200)
ax1d.plot(x_dense, sii_model(x_dense, *popt), color="red", lw=1.6, label=f"fit (z={z_fit:.5f})")
ax1d.plot(x_dense, sii_model_6716(x_dense, *popt), color="red", lw=1.1, ls="--")
ax1d.plot(x_dense, sii_model_6731(x_dense, *popt), color="red", lw=1.1, ls="-.")
ax1d.axvline(lam1_fit, color="red",  ls="--", lw=0.8, alpha=0.8)
ax1d.axvline(lam2_fit, color="red",  ls="-.", lw=0.8, alpha=0.8)

ax1d.set_xlabel(r"Observed wavelength $\lambda$ (Å)")
ax1d.set_ylabel(r"$F_{\lambda}$ (10$^{-17}$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$)")

save_dir = "./results/SDSS/figure/SDSS_0275-51910-0141_SII_fit.png"
plt.savefig(save_dir, dpi=300)
print(f"Saved as {save_dir}")
plt.tight_layout()
plt.show()

# --- 数値出力（任意） ---
print(f"z_fit = {z_fit:.6f}")
print(f"SII 6716 flux = {F1:.3g}")
print(f"SII 6731 flux = {F2:.3g}")
print(f"Flux ratio (6716/6731) = {ratio:.3f}")
print(f"sigma = {sig:.3f} Å,  FWHM = {2.355*sig:.3f} Å")