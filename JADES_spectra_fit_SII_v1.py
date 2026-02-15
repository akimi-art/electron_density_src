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


# == 必要なパッケージ == #
import os
import glob
import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.optimize import curve_fit
import emcee

# =====================================================
# 基本物理定数
# =====================================================
wave_6716 = 6716.440
wave_6730 = 6730.820
delta_lambda = 120.0
nwalkers = 32
nsteps = 3000
burnin = 800

# =====================================================
# Gaussian
# =====================================================
def gaussian(x, amp, mu, sigma):
    return amp * np.exp(-(x-mu)**2/(2*sigma**2)) / (np.sqrt(2*np.pi)*sigma)

def nirspec_sigma(wavelength_A, R):
    return (wavelength_A / R) / 2.355

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
        a1,a2,z,sig,bg = theta
        if a1<=0 or a2<=0 or sig<=0:
            return -np.inf
        if not (z_fix-0.01 < z < z_fix+0.01):
            return -np.inf
        return 0.0

    def log_likelihood(theta):
        model = s2_model(x,*theta,sigma_instr)
        return -0.5*np.sum(((y-model)/yerr)**2)

    def log_prob(theta):
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood(theta)

    ndim = 5
    pos = popt + 1e-3*np.random.randn(nwalkers,ndim)
    sampler = emcee.EnsembleSampler(nwalkers,ndim,log_prob)
    sampler.run_mcmc(pos,nsteps,progress=False)

    samples = sampler.get_chain(discard=burnin,thin=10,flat=True)
    return samples

# =====================================================
# ディレクトリ設定
# =====================================================
current_dir = os.getcwd()

catalog_path = os.path.join(
    current_dir,
    "results/JADES/JADES_DR3/catalog/jades_dr3_medium_gratings_public_gs_v1.1.fits"
)

base_root = os.path.join(
    current_dir,
    "results/JADES/JADES_DR3/JADES_DR3_full_spectra"
)

grating_dirs = {
    "f070lp-g140m": "JADES_DR3_G140M",
    "f170lp-g235m": "JADES_DR3_G235M",
    "f290lp-g395m": "JADES_DR3_G395M",
}

# =====================================================
# カタログ読み込み
# =====================================================
with fits.open(catalog_path) as hdul:
    cat = hdul[1].data

df_cat = pd.DataFrame({
    "NIRSpec_ID": cat["NIRSpec_ID"],
    "z_Spec": cat["z_Spec"],
})

total_objects = len(df_cat)
print(f"\nTotal objects in catalog: {total_objects}")

results_all = []
n_success = 0
n_skip = 0

# =====================================================
# メインループ
# =====================================================
for i, row in enumerate(df_cat.itertuples(index=False), 1):

    nir_id = int(row.NIRSpec_ID)
    z_spec = float(row.z_Spec)
    nir_id_str = f"{nir_id:08d}"

    print(f"\n[{i}/{total_objects}] Processing {nir_id_str} (z={z_spec:.3f})")

    if not np.isfinite(z_spec):
        print("  -> skipped: z_spec not finite")
        n_skip += 1
        continue

    wave_center = 0.5*(wave_6716+wave_6730)*(1+z_spec)

    if 7000 < wave_center < 18800:
        filter_grating = "f070lp-g140m"
        R = 1000
    elif 18800 < wave_center < 31000:
        filter_grating = "f170lp-g235m"
        R = 1000
    elif 31000 < wave_center < 52000:
        filter_grating = "f290lp-g395m"
        R = 1000
    else:
        print("  -> skipped: SII out of wavelength range")
        n_skip += 1
        continue

    subdir = grating_dirs.get(filter_grating)
    base = os.path.join(base_root, subdir)
    pattern = f"*{nir_id_str}_{filter_grating}*_x1d.fits"
    files = glob.glob(os.path.join(base, pattern))

    if len(files)==0:
        print("  -> skipped: spectrum file not found")
        n_skip += 1
        continue

    x1d = files[0]

    try:
        with fits.open(x1d) as hdul:
            tab = hdul["EXTRACT1D"].data
            wave = tab["WAVELENGTH"]*1e4
            flux = tab["FLUX"]*1e19
            err  = tab["FLUX_ERR"]*1e19
    except Exception as e:
        print(f"  -> skipped: fits read error {e}")
        n_skip += 1
        continue

    sigma_instr = nirspec_sigma(wave_center,R)

    mask = (wave > wave_center-delta_lambda) & (wave < wave_center+delta_lambda)
    if np.sum(mask)==0:
        print("  -> skipped: no wavelength coverage")
        n_skip += 1
        continue

    x_fit = wave[mask]
    y_fit = flux[mask]
    yerr_fit = err[mask]

    p0 = [20,20,z_spec,10,0]

    try:
        popt,_ = curve_fit(
            lambda x,a1,a2,z,s,b: s2_model(x,a1,a2,z,s,b,sigma_instr),
            x_fit,y_fit,p0=p0,sigma=yerr_fit,absolute_sigma=True
        )
    except Exception as e:
        print(f"  -> skipped: fit failed {e}")
        n_skip += 1
        continue

    try:
        samples = run_mcmc(popt,x_fit,y_fit,yerr_fit,sigma_instr,z_spec)
    except Exception as e:
        print(f"  -> skipped: MCMC failed {e}")
        n_skip += 1
        continue

    amp1 = samples[:,0]
    amp2 = samples[:,1]

    ratio_samples = amp1/amp2
    r16,r50,r84 = np.percentile(ratio_samples,[16,50,84])

    results_all.append({
        "NIRSpec_ID": nir_id,
        "z_Spec": z_spec,
        "S2_6716_flux": np.median(amp1),
        "S2_6730_flux": np.median(amp2),
        "ratio_median": r50,
        "ratio_minus": r50-r16,
        "ratio_plus": r84-r50,
    })

    n_success += 1
    print(f"  ✓ success: ratio = {r50:.3f}")

# =====================================================
# 保存
# =====================================================
df_final = pd.DataFrame(results_all)

output_path = os.path.join(current_dir,"results/csv/JADES_DR3_GOODS-S_SII_ratio_only.csv")
df_final.to_csv(output_path,index=False)

print("\n========== SUMMARY ==========")
print(f"Total: {total_objects}")
print(f"Success: {n_success}")
print(f"Skipped: {n_skip}")
print("Saved to:",output_path)


