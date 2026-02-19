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
# delta_lambda = 120.0
delta_lambda = 50 
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

    # ---- 初期値をlog空間へ ----
    logA1_0 = np.log(popt[0])
    logA2_0 = np.log(popt[1])
    logSig_0 = np.log(popt[3])
    bg_0 = popt[4]

    # ---------------------------
    # Prior
    # ---------------------------
    def log_prior(theta):

        logA1, logA2, logSig, bg = theta

        # 振幅（広いが有限）
        if not (-50 < logA1 < 20):
            return -np.inf
        if not (-50 < logA2 < 20):
            return -np.inf

        # intrinsic sigma (Å)
        if not (-5 < logSig < 5):
            return -np.inf

        # 背景
        if not (-1e6 < bg < 1e6):
            return -np.inf

        return 0.0


    # ---------------------------
    # Likelihood
    # ---------------------------
    def log_likelihood(theta):

        logA1, logA2, logSig, bg = theta

        amp1 = np.exp(logA1)
        amp2 = np.exp(logA2)
        sigma_int = np.exp(logSig)

        model = s2_model(
            x,
            amp1,
            amp2,
            z_fix,        # z固定
            sigma_int,
            bg,
            sigma_instr
        )

        return -0.5*np.sum(((y-model)/yerr)**2)


    def log_prob(theta):
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood(theta)


    # ---------------------------
    # Sampler
    # ---------------------------
    ndim = 4
    pos0 = np.array([logA1_0, logA2_0, logSig_0, bg_0])
    # pos = pos0 + 1e-3*np.random.randn(nwalkers, ndim)
    pos = pos0 * (1 + 1e-4*np.random.randn(nwalkers, ndim))
    pos += 1e-4*np.random.randn(nwalkers, ndim)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
    sampler.run_mcmc(pos, nsteps, progress=False)

    flat = sampler.get_chain(discard=burnin, thin=10, flat=True)

    # 物理量に戻す
    amp1 = np.exp(flat[:,0])
    amp2 = np.exp(flat[:,1])
    sigma_int = np.exp(flat[:,2])
    bg = flat[:,3]

    return amp1, amp2, sigma_int, bg



# =====================================================
# ディレクトリ設定
# =====================================================
current_dir = os.getcwd()

catalog_path = os.path.join(
    current_dir,
    "results/JADES/JADES_DR3/catalog/jades_dr3_medium_gratings_public_gn_v1.1.fits" # DR3
    # "results/JADES/JADES_DR4/catalog/Combined_DR4_external_v1.2.1.fits" # DR4
)

base_root = os.path.join(
    current_dir,
    "results/JADES/JADES_DR3/JADES_DR3_full_spectra" # DR3
    # "results/JADES/JADES_DR4/JADES_DR4_full_spectra/GOODS-N" # DR4
    #  results/JADES/JADES_DR4/JADES_DR4_full_spectra/GOODS-N

)

grating_dirs = {
    # DR3
    "f070lp-g140m": "JADES_DR3_G140M",
    "f170lp-g235m": "JADES_DR3_G235M",
    "f290lp-g395m": "JADES_DR3_G395M",
    # # DR4
    # "f070lp-g140m": "JADES_DR4_G140M",
    # "f170lp-g235m": "JADES_DR4_G235M",
    # "f290lp-g395m": "JADES_DR4_G395M",
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
            tab = hdul["EXTRACT1D"].data # DR3
            # tab = hdul["EXTRACT5PIX1D"].data # DR4
            wave = tab["WAVELENGTH"]*1e4
            flux = tab["FLUX"]*1e19
            err  = tab["FLUX_ERR"]*1e19
    except Exception as e:
        print(f"  -> skipped: fits read error {e}")
        n_skip += 1
        continue

    sigma_instr = nirspec_sigma(wave_center,R)

    # mask = (wave > wave_center-delta_lambda) & (wave < wave_center+delta_lambda)
    mask = (
        (wave > wave_center-delta_lambda) &
        (wave < wave_center+delta_lambda) &
        np.isfinite(flux) &
        np.isfinite(err) &
        (err > 0)
    )
    if np.sum(mask)==0:
        print("  -> skipped: no wavelength coverage")
        n_skip += 1
        continue

    x_fit = wave[mask]
    y_fit = flux[mask]
    yerr_fit = err[mask]
    yerr_fit = np.clip(yerr_fit, 1e-30, None) # 追加


    p0 = [20,20,z_spec,10,0]

    try:
        # popt,_ = curve_fit(
        #     lambda x,a1,a2,z,s,b: s2_model(x,a1,a2,z,s,b,sigma_instr),
        #     x_fit,y_fit,p0=p0,sigma=yerr_fit,absolute_sigma=True
        # )
        popt,_ = curve_fit(
            lambda x,a1,a2,z,s,b: s2_model(x,a1,a2,z,s,b,sigma_instr),
            x_fit,
            y_fit,
            p0=p0,
            sigma=yerr_fit,
            absolute_sigma=True,
            bounds=(
                [0, 0, z_spec-1e-4, 0.1, -np.inf],
                [np.inf, np.inf, z_spec+1e-4, 50, np.inf]
            )
        )
    except Exception as e:
        print(f"  -> skipped: fit failed {e}")
        n_skip += 1
        continue

    try:
        amp1, amp2, sigma_int, bg_samples = run_mcmc(
            popt,
            x_fit,
            y_fit,
            yerr_fit,
            sigma_instr,
            z_spec
        )
    except Exception as e:
        print(f"  -> skipped: MCMC failed {e}")
        n_skip += 1
        continue

    # ==========================
    # 統計処理
    # ==========================

    f1_16,f1_50,f1_84 = np.percentile(amp1,[16,50,84])
    f2_16,f2_50,f2_84 = np.percentile(amp2,[16,50,84])

    ratio_samples = amp1 / amp2
    r16,r50,r84 = np.percentile(ratio_samples,[16,50,84])

    results_all.append({
        "NIRSpec_ID": nir_id,
        "z_Spec": z_spec,
        "S2_6716_flux": f1_50,
        "S2_6716_err_minus": f1_50 - f1_16,
        "S2_6716_err_plus":  f1_84 - f1_50,
        "S2_6730_flux": f2_50,
        "S2_6730_err_minus": f2_50 - f2_16,
        "S2_6730_err_plus":  f2_84 - f2_50,
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

output_path = os.path.join(current_dir,"results/csv/JADES_DR3_GOODS-N_SII_ratio_only.csv")
df_final.to_csv(output_path,index=False)

print("\n========== SUMMARY ==========")
print(f"Total: {total_objects}")
print(f"Success: {n_success}")
print(f"Skipped: {n_skip}")
print("Saved to:",output_path)

# ============================
#  基準線より上側の SII6717 を抽出し、新しい FITS を保存
# ============================
# ============================
#  Lベースの完全サンプル抽出（構造そのまま・行のみ削除）— 両線同時版
# ============================
# --- パラメータ ---
# 一定フラックス [erg s^-1 cm^-2]（図の基準線に対応）
F_CONST_6717_CGS = 1e-17
F_CONST_6731_CGS = 1e-17     # 6717と同じにしてよければ同値のままでOK（別々に設定可能）
Z_RANGE = (0.0, 0.40)        # 図と揃える場合。全 z を許容するなら None
REQUIRE_FINITE = True        # 数値の健全性チェック（NaN/inf除外）

# --- L_lim(z) を計算（一定フラックス線） ---
# すでに z が配列としてあり、cosmo は Planck18 想定
dL_each = cosmo.luminosity_distance(z).to(u.cm).value
Llim6717_each = 4 * np.pi * dL_each**2 * F_CONST_6717_CGS
Llim6731_each = 4 * np.pi * dL_each**2 * F_CONST_6731_CGS

# --- マスク作成（両線の L >= L_lim(z) を同時に満たす） ---
mask_L_6717 = (L6716 >= Llim6717_each)
mask_L_6731 = (L6731 >= Llim6731_each)
mask_L_both = mask_L_6717 & mask_L_6731

# z 範囲の適用（必要に応じて）
if Z_RANGE is not None:
    zmin, zmax = Z_RANGE
    mask_z = np.isfinite(z) & (z >= zmin) & (z <= zmax)
else:
    mask_z = np.ones_like(z, dtype=bool)

# 数値の健全性（NaN/inf の排除）
if REQUIRE_FINITE:
    mask_finite = (
        np.isfinite(z) &
        np.isfinite(L6716) & np.isfinite(L6731) &
        np.isfinite(Llim6717_each) & np.isfinite(Llim6731_each)
    )
else:
    mask_finite = np.ones_like(z, dtype=bool)

# --- 最終マスク（列構造は触らない） ---
select_mask = mask_L_both & mask_z & mask_finite

print(f"[INFO] 抽出件数（両線同時）: {select_mask.sum()} / {len(select_mask)}")
if select_mask.sum() == 0:
    print("[WARN] 0 件です。F_CONST_* や Z_RANGE を見直してください。")

# --- Table を行スライスのみで抽出（列・メタデータ保持） ---
t_sel = t[select_mask]  # ← 列構造は一切変更しない

# --- 書き出し（ファイル名に条件を明記） ---
def _sci_notation(x):
    return f"{x:.0e}".replace("+","")

suffix_parts = [
    f"L6717_ge_4pi_dL2_{_sci_notation(F_CONST_6717_CGS)}",
    f"L6731_ge_4pi_dL2_{_sci_notation(F_CONST_6731_CGS)}",
]
if Z_RANGE is not None:
    suffix_parts.append(f"z{zmin:.2f}-{zmax:.2f}")
suffix = "_".join(suffix_parts)

out_dir = os.path.join(current_dir, "results", "fits")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, f"mpajhu_dr7_v5_2_merged_{suffix}.fits")

t_sel.write(out_path, format="fits", overwrite=True)
print(f"[DONE] 書き出し完了: {out_path}")