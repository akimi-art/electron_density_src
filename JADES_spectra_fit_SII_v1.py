#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
スクリプトの概要:
このスクリプトは
JADESのカタログを使って
スペクトル (SII) をフィッティングするものです。
csvファイルに存在する全てのIDのスペクトルに対して
同時にフィッティングを行います。

使用方法:
    JADES_spectra_fit_SII_v1.py [オプション]

著者: A. M.
作成日: 2026-02-09

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


import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import ZScaleInterval
from scipy.optimize import curve_fit
import emcee
import corner
import pyneb as pn

# =====================================================
# 基本設定
# =====================================================
wave_6716 = 6716.440
wave_6730 = 6730.820
delta_lambda = 100.0
Te = 15000.0
nwalkers = 32
nsteps = 4000
burnin = 1000

# =====================================================
# Gaussian
# =====================================================
def gaussian(x, amp, mu, sigma):
    return amp * np.exp(-(x-mu)**2/(2*sigma**2)) / (np.sqrt(2*np.pi)*sigma)

def nirspec_sigma(wavelength_A, R=1000.0):
    fwhm_A = wavelength_A / R
    return fwhm_A / 2.355

# =====================================================
# SII model
# =====================================================
def s2_model(x, amp1, amp2, z, sigma_int, bg, sigma_instr):
    mu1 = wave_6716 * (1 + z)
    mu2 = wave_6730 * (1 + z)
    sigma_tot = np.sqrt(sigma_int**2 + sigma_instr**2)
    return (
        gaussian(x, amp1, mu1, sigma_tot) +
        gaussian(x, amp2, mu2, sigma_tot) +
        bg
    )

# =====================================================
# MCMC
# =====================================================
def run_mcmc(popt, x, y, yerr, sigma_instr, z_fix):

    def log_prior(theta):
        amp1, amp2, z, sigma_int, bg = theta
        if amp1 <= 0 or amp2 <= 0 or sigma_int <= 0:
            return -np.inf
        if not (z_fix-0.01 < z < z_fix+0.01):
            return -np.inf
        return 0.0

    def log_likelihood(theta):
        model = s2_model(x, *theta, sigma_instr)
        return -0.5 * np.sum(((y-model)/yerr)**2)

    def log_prob(theta):
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood(theta)

    ndim = 5
    pos = popt + 1e-3*np.random.randn(nwalkers, ndim)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
    sampler.run_mcmc(pos, nsteps, progress=False)

    samples = sampler.get_chain(discard=burnin, thin=10, flat=True)
    return samples

# =====================================================
# メイン処理
# =====================================================
current_dir = os.getcwd()
csv_path = os.path.join(current_dir, "results/csv/JADES_ne_candidates_GOODS_S_v1.1.csv")
df = pd.read_csv(csv_path)

base_dir = os.path.join(current_dir, "results/JADES/individual")

for _, row in df.iterrows():

    nir_id = int(row["NIRSpec_ID"])
    z_spec = float(row["z_Spec"])
    nir_id_str = f"{nir_id:08d}"

    print(f"\nProcessing ID {nir_id_str}")

    # =============================
    # フィルター自動判定
    # =============================
    wave_center = 0.5*(wave_6716+wave_6730)*(1+z_spec)

    if 7000 < wave_center < 18800:
        filter_grating = "f070lp-g140m"
        R = 1000
    else:
        filter_grating = "f170lp-g235m"
        R = 1000

    # =============================
    # スペクトル取得
    # =============================
    base = os.path.join(base_dir, f"JADES_{nir_id_str}")

    x1d_files = glob.glob(os.path.join(base, "**", f"*{filter_grating}*_x1d.fits"), recursive=True)
    s2d_files = glob.glob(os.path.join(base, "**", f"*{filter_grating}*_s2d.fits"), recursive=True)

    if len(x1d_files)==0 or len(s2d_files)==0:
        print("  Spectrum not found")
        continue

    x1d = x1d_files[0]
    s2d = s2d_files[0]

    # =============================
    # 読み込み
    # =============================
    with fits.open(x1d) as hdul:
        tab = hdul["EXTRACT1D"].data
        wave = tab["WAVELENGTH"]*1e4
        flux = tab["FLUX"]*1e19
        err  = tab["FLUX_ERR"]*1e19

    with fits.open(s2d) as hdul:
        flux2d = hdul["FLUX"].data
        wave2d = hdul["WAVELENGTH"].data*1e4

    sigma_instr = nirspec_sigma(wave_center, R=R)

    mask = (wave > wave_center-delta_lambda) & (wave < wave_center+delta_lambda)
    x_fit, y_fit, yerr_fit = wave[mask], flux[mask], err[mask]

    if len(x_fit)==0:
        print("  No valid region")
        continue

    # =============================
    # curve_fit
    # =============================
    p0 = [20,20,z_spec,10,0]
    popt,_ = curve_fit(
        lambda x,a1,a2,z,s,b: s2_model(x,a1,a2,z,s,b,sigma_instr),
        x_fit,y_fit,p0=p0,sigma=yerr_fit,absolute_sigma=True
    )

    ratio = popt[0]/popt[1]
    print(f"  Ratio initial = {ratio:.3f}")

    # =============================
    # MCMC
    # =============================
    samples = run_mcmc(popt,x_fit,y_fit,yerr_fit,sigma_instr,z_spec)

    amp1 = samples[:,0]
    amp2 = samples[:,1]
    ratio_samples = amp1/amp2

    q16,q50,q84 = np.percentile(ratio_samples,[16,50,84])

    # =============================
    # ne 計算
    # =============================
    S2 = pn.Atom("S",2)
    ne_median = S2.getTemDen(q50,tem=Te,wave1=6716,wave2=6731)

    # =============================
    # 保存
    # =============================
    save_dir = os.path.join(current_dir,f"results/JADES/parameters/{nir_id_str}")
    os.makedirs(save_dir,exist_ok=True)

    pd.DataFrame({
        "ratio_median":[q50],
        "ratio_minus":[q50-q16],
        "ratio_plus":[q84-q50],
        "ne":[ne_median]
    }).to_csv(os.path.join(save_dir,f"SII_results_{nir_id_str}.csv"),index=False)

    print(f"  Saved results for {nir_id_str}")
