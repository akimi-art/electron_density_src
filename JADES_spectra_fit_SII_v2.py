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
import emcee
import corner
import pyneb as pn
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
# 1. データ読み込み
# =========================
name = "stack_G395M_SFR_0.01_5.95"
data = np.loadtxt(f"{name}.txt")

wave = data[:,0]   # Å (rest)
flux = data[:,1]
err  = data[:,2]
flux = flux * 1e19
err = err * 1e19

# =========================
# 2. フィット範囲
# =========================
w1 = 6716.44
w2 = 6730.82

mask = (wave > 6650) & (wave < 6800)

x = wave[mask]
y = flux[mask]
yerr = err[mask]

# =========================
# 3. モデル
# =========================
def gaussian(x, amp, mu, sigma):
    return amp * np.exp(-(x-mu)**2/(2*sigma**2))

# --- doublet model (shift付き) ---
def model(x, a1, a2, sigma, shift, bg):

    mu1 = w1 + shift
    mu2 = w2 + shift

    g1 = gaussian(x, a1, mu1, sigma)
    g2 = gaussian(x, a2, mu2, sigma)

    return g1 + g2 + bg


def model_6716(x, a1, a2, sigma, shift, bg):

    mu1 = w1 + shift

    g1 = gaussian(x, a1, mu1, sigma)

    return g1 + bg


def model_6731(x, a1, a2, sigma, shift, bg):

    mu2 = w2 + shift

    g2 = gaussian(x, a2, mu2, sigma)

    return g2 + bg

# =========================
# 4. curve_fit
# =========================
p0 = [1, 1, 2, 0, 0]

popt,_ = curve_fit(
    model,
    x,
    y,
    p0=p0,
    sigma=yerr,
    absolute_sigma=True
)

print("curve_fit ratio =", popt[0]/popt[1])

# =========================
# 5. MCMC
# =========================
def log_prior(theta):

    a1,a2,s,shift,b = theta

    if a1 <= 0: return -np.inf
    if a2 <= 0: return -np.inf
    if s <= 0: return -np.inf

    # shift 制限
    if not (-5 < shift < 5):
        return -np.inf

    return 0

def log_like(theta,x,y,yerr):
    return -0.5*np.sum(((y-model(x,*theta))/yerr)**2)

def log_prob(theta,x,y,yerr):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_like(theta,x,y,yerr)

ndim = 5
nwalkers=32

pos = popt + 1e-4*np.random.randn(nwalkers,ndim)

sampler = emcee.EnsembleSampler(
    nwalkers,ndim,log_prob,args=(x,y,yerr)
)

sampler.run_mcmc(pos,3000,progress=True)

# =========================
# 6. サンプル
# =========================
samples = sampler.get_chain(discard=500,thin=10,flat=True)

a1 = samples[:,0]
a2 = samples[:,1]

ratio = a1/a2

q16,q50,q84 = np.percentile(ratio,[16,50,84])

print("SII ratio = %.3f+%.3f-%.3f" %
      (q50,q84-q50,q50-q16))

# =========================
# 7. プロット
# =========================
x_model = np.linspace(6650,6800,1000)

fig, ax = plt.subplots(figsize=(12, 6))
ax.step(x, y, where="mid", color="black", lw=0.8)
ax.fill_between(
    x,
    y - yerr,
    y + yerr,
    step="mid",
    color="gray",
    alpha=0.4,
)
ax.plot(x_model,
         model(x_model,*popt),
         color="red")
ax.plot(x_model,
         model_6716(x_model,*popt),
         color="red", ls="--")
ax.plot(x_model,
         model_6731(x_model,*popt),
         color="red", ls="-.")

mu1 = w1 + popt[3]
mu2 = w2 + popt[3]

ax.axvline(mu1,ls="--", color="red")
ax.axvline(mu2,ls="-.", color="red")
# ax.set_xlim(6600, 6900)
ax.set_xlabel(r'$\lambda (Å)$')
ax.set_ylabel(r'F$_{\lambda}$ ($10^{-19}$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$)')
# ax.set_xlim(6700, 6750)
for spine in ax.spines.values():
    spine.set_linewidth(2)       # 枠線の太さ
    spine.set_color("black")     # 枠線の色
plt.savefig(f"{name}_fit.png", dpi=300)  # 画像保存
plt.show()

# =========================
# 8. corner
# =========================
corner.corner(samples,
              labels=["amp6716","amp6730","sigma","shift", "bg"])

plt.show()