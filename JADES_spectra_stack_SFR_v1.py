#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
スクリプトの概要:
JADESスペクトルスタックを作成します。
SFRのビンごとにスタックを作成します。
加えて、
ΣSFRのビンごとにスタックを作成します。

使用方法:
    JADES_spectra_stack_SFR_v1.py [オプション]

著者: A. M.
作成日: 2026-05-27

参考文献:
    - PEP 8:                  https://peps.python.org/pep-0008/
    - PEP 257 (Docstring規約): https://peps.python.org/pep-0257/
    - Python公式ドキュメント:    https://docs.python.org/ja/3/
"""

# ============================
# z-bin stacking only version
# （機能は維持しつつ、後半の z-bin 制御部分のみ残した版）
# ============================

import glob
import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from astropy.cosmology import Planck18

# ============================
# SETTINGS
# ============================

csv_file = "results/JADES/JADES_DR3/data_from_Nishigaki/jades_info_with_HA_plus_logSFR_with_Reff.csv" # 変更

spec_dir = "results/JADES/JADES_DR3/JADES_DR3_full_spectra"

gratings = {
    "G140M": f"{spec_dir}/JADES_DR3_G140M",
    "G235M": f"{spec_dir}/JADES_DR3_G235M",
    "G395M": f"{spec_dir}/JADES_DR3_G395M"
}

wave_grid = np.arange(6500, 6900, 0.5)

# ← ここだけ変えればOK
n_bins = 1 # 一番安定したスタックを得るためには、bin数は少なめ（1-3程度）が良いと思います。

# ============================
# CSV
# ============================

df = pd.read_csv(csv_file)

df = df[df["z_spec"].notna()]
df = df[df["HA_6563_flux"].notna()]
df = df[df["log10_SFR_hb"].notna()]
df = df[df["log10_SFR_hb"] >= 0]

print("usable rows after CSV filtering:", len(df))

# ============================
# FUNCTIONS
# ============================

def read_spectrum(file):

    with fits.open(file) as h:

        data = h["EXTRACT1D"].data

        wave = data["WAVELENGTH"] * 1e4
        flux = data["FLUX"]
        err  = data["FLUX_ERR"]

    return wave, flux, err


def restframe(wave, flux, err, z):

    wave = wave / (1 + z)

    return wave, flux, err


def resample(wave, flux, err):

    f = interp1d(wave, flux, bounds_error=False, fill_value=np.nan)
    e = interp1d(wave, err , bounds_error=False, fill_value=np.nan)

    return f(wave_grid), e(wave_grid)


def coverage_ok(wave, flux):

    ha_region  = (wave > 6550) & (wave < 6575)
    sii_region = (wave > 6705) & (wave < 6740)
    cont_region = (wave > 6600) & (wave < 6650)

    if np.sum(np.isfinite(flux[ha_region])) < 2:
        return False

    if np.sum(np.isfinite(flux[sii_region])) < 2:
        return False

    if np.sum(np.isfinite(flux[cont_region])) < 3:
        return False

    return True


def mask_artifact(flux):

    med = np.nanmedian(flux)
    std = np.nanstd(flux)

    mask = flux < med - 5 * std

    flux = flux.copy()
    flux[mask] = np.nan

    return flux


def stack(fluxes, errs):

    fluxes = np.array(fluxes)
    errs   = np.array(errs)

    floor = 0.05 * np.nanmedian(errs)

    errs = np.sqrt(errs**2 + floor**2)

    w = 1 / errs**2

    flux_stack = np.nansum(fluxes * w, axis=0) / np.nansum(w, axis=0)

    err_stack = np.sqrt(1 / np.nansum(w, axis=0))

    return flux_stack, err_stack


def median_stack(fluxes):

    fluxes = np.array(fluxes)

    return np.nanmedian(fluxes, axis=0)


def bootstrap_median_error(fluxes, nboot=500):

    fluxes = np.array(fluxes)

    n_spec = fluxes.shape[0]
    n_wave = fluxes.shape[1]

    med_boot = np.zeros((nboot, n_wave))

    for i in range(nboot):

        idx = np.random.randint(0, n_spec, n_spec)

        sample = fluxes[idx]

        med_boot[i] = np.nanmedian(sample, axis=0)

    return np.nanstd(med_boot, axis=0)


def weighted_mean_sfr(values, err_lo, err_hi):

    values = np.array(values)

    err_lo = np.array(err_lo)
    err_hi = np.array(err_hi)

    # 非対称誤差 → 対称化
    sigma = 0.5 * (err_lo + err_hi)

    mask = (
        np.isfinite(values)
        &
        np.isfinite(sigma)
        &
        (sigma > 0)
    )

    values = values[mask]
    sigma = sigma[mask]

    w = 1 / sigma**2

    mean = np.sum(w * values) / np.sum(w)

    err = np.sqrt(1 / np.sum(w))

    return mean, err


def compute_log_sigma_sfr(
    logSFR,
    err_lo,
    err_hi,
    Re_arcsec,
    eRe_arcsec,
    z
):

    if (
        (not np.isfinite(logSFR))
        or
        (not np.isfinite(Re_arcsec))
        or
        (not np.isfinite(eRe_arcsec))
        or
        (Re_arcsec <= 0)
    ):

        return np.nan, np.nan, np.nan

    # --------------------------------
    # arcsec -> kpc
    # --------------------------------

    kpc_per_arcsec = (
        Planck18.kpc_proper_per_arcmin(z).value
        / 60.0
    )

    Re_kpc = Re_arcsec * kpc_per_arcsec
    eRe_kpc = eRe_arcsec * kpc_per_arcsec

    # --------------------------------
    # central value
    # --------------------------------

    logSigma = (
        logSFR
        -
        np.log10(2 * np.pi * Re_kpc**2)
    )

    # --------------------------------
    # SFR uncertainty
    # --------------------------------

    sfr_sigma = 0.5 * (
        err_lo + err_hi
    )

    # --------------------------------
    # Re uncertainty
    # --------------------------------

    re_sigma = (
        2 * eRe_kpc
        /
        (Re_kpc * np.log(10))
    )

    total_sigma = np.sqrt(
        sfr_sigma**2
        +
        re_sigma**2
    )

    return (
        logSigma,
        total_sigma,
        total_sigma
    )

# ============================
# MAIN
# ============================

used_items_all = []

for gr in gratings:

    print("\nGRATING:", gr)

    spec_path = gratings[gr]

    if gr == "G140M":
        df_gr = df[(df["z_spec"] > 0.5) & (df["z_spec"] < 1.7)]

    elif gr == "G235M":
        df_gr = df[(df["z_spec"] > 1.5) & (df["z_spec"] < 3.6)]

    elif gr == "G395M":
        df_gr = df[(df["z_spec"] > 3.3) & (df["z_spec"] < 6.7)]

    print("candidates:", len(df_gr))

    ha_vals = df_gr["HA_6563_flux"].values
    ha_vals = ha_vals[np.isfinite(ha_vals) & (ha_vals > 0)]

    ha_p995 = np.nanpercentile(ha_vals, 99.5)

    for _, row in df_gr.iterrows():

        nid = int(row["NIRSpec_ID"])
        z   = row["z_spec"]
        ha  = row["HA_6563_flux"]

        if (not np.isfinite(ha)) or (ha <= 0) or (ha > ha_p995):
            continue

        sid = f"{nid:08d}"

        pattern = f"{spec_path}/*{sid}*_x1d.fits"

        files = glob.glob(pattern)

        if len(files) == 0:
            continue

        try:

            wave, flux, err = read_spectrum(files[0])

            wave, flux, err = restframe(wave, flux, err, z)

            if not coverage_ok(wave, flux):
                continue

            flux = mask_artifact(flux)

            flux = flux / ha
            err  = err  / ha

            flux_i, err_i = resample(wave, flux, err)

            if not np.isfinite(flux_i).any():
                continue

            # ↓ 追加
            logSigma, logSigma_err_lo, logSigma_err_hi = (
                compute_log_sigma_sfr(
                    row["log10_SFR_hb"],
                    row["log10_SFR_hb_err_lower"],
                    row["log10_SFR_hb_err_upper"],
                    row["ReffOpt"],
                    row["e_ReffOpt"],
                    z
                )
            )

            used_items_all.append({
                "z": z,
                "flux": flux_i,
                "err": err_i,
                "id": sid,
                "gr": gr,
                
                "sfr": row["log10_SFR_hb"],
                "sfr_err_lo": row["log10_SFR_hb_err_lower"],
                "sfr_err_hi": row["log10_SFR_hb_err_upper"],

                # ↓ 追加
                "sigma_sfr": logSigma,
                "sigma_sfr_err_lo": logSigma_err_lo,
                "sigma_sfr_err_hi": logSigma_err_hi,
                            })

        except Exception:
            continue

# ============================
# z-bin split
# ============================

if len(used_items_all) == 0:

    print("No usable spectra.")

else:

    used_z_all = np.array([it["z"] for it in used_items_all])

    N = len(used_z_all)

    print("\nTotal usable spectra:", N)

    # ============================
    # Sigma_SFR distribution
    # ============================

    vals = np.array([
        it["sigma_sfr"]
        for it in used_items_all
    ])

    vals = vals[np.isfinite(vals)]

    plt.figure(figsize=(6,4))

    plt.hist(
        vals,
        bins=30,
        color="0.7",
        edgecolor="black"
    )

    plt.xlabel(r'$\log \Sigma_{\rm SFR}$')
    plt.ylabel("count")

    plt.tight_layout()
    plt.show()

    print("median =", np.nanmedian(vals))
    print("std =", np.nanstd(vals))

    sort_idx = np.argsort(used_z_all)

    q, r = divmod(N, n_bins)

    counts = [q + 1 if i < r else q for i in range(n_bins)]

    cum = np.cumsum([0] + counts)

    print("equal-count bins:", counts)

    # ============================
    # histogram
    # ============================

    z_sorted = used_z_all[sort_idx]

    boundaries = []

    for i in range(1, n_bins):

        left_end = z_sorted[cum[i] - 1]
        right_start = z_sorted[cum[i]]

        boundaries.append(0.5 * (left_end + right_start))

    plt.figure(figsize=(7,4))

    plt.hist(
        used_z_all,
        bins="auto",
        color="0.7",
        edgecolor="black"
    )

    for bx in boundaries:

        plt.axvline(
            bx,
            color="tab:red",
            linestyle="--",
            linewidth=2
        )

    plt.xlabel("redshift")
    plt.ylabel("count")

    plt.title(
        f"usable galaxies (N={N})\n"
        f"equal-count {n_bins} bins"
    )

    plt.tight_layout()
    plt.show()

    # ============================
    # stack each z-bin
    # ============================

    for b in range(n_bins):

        s = cum[b]
        e = cum[b+1]

        sel_idx = sort_idx[s:e]

        selected = [used_items_all[i] for i in sel_idx]

        z_vals = np.array([it["z"] for it in selected])

        flux_list = [it["flux"] for it in selected]
        err_list  = [it["err"]  for it in selected]

        # ============================
        # Sigma_SFR subsample
        # ============================

        selected_sigma = [
        
            it for it in selected

            if np.isfinite(it["sigma_sfr"])
        ]

        flux_list_sigma = [
            it["flux"]
            for it in selected_sigma
        ]

        err_list_sigma = [
            it["err"]
            for it in selected_sigma
        ]

        # ============================
        # SFR weighted mean
        # ============================
        
        sfr_vals = [it["sfr"] for it in selected]
        
        sfr_err_lo = [it["sfr_err_lo"] for it in selected]
        
        sfr_err_hi = [it["sfr_err_hi"] for it in selected]
        
        sfr_mean, sfr_err = weighted_mean_sfr(
            sfr_vals,
            sfr_err_lo,
            sfr_err_hi
        )
        sigma_vals = [
            it["sigma_sfr"]
            for it in selected
        ]

        sigma_err_lo = [
            it["sigma_sfr_err_lo"]
            for it in selected
        ]

        sigma_err_hi = [
            it["sigma_sfr_err_hi"]
            for it in selected
        ]
        sigma_mean, sigma_err = weighted_mean_sfr(
            sigma_vals,
            sigma_err_lo,
            sigma_err_hi
        )

        print(
            f"log10(SFR) weighted mean = "
            f"{sfr_mean:.3f} ± {sfr_err:.3f}"
        )
        print(
            f"log10(Sigma_SFR) weighted mean = "
            f"{sigma_mean:.3f} ± {sigma_err:.3f}"
        )

        z_min = np.min(z_vals)
        z_max = np.max(z_vals)

        print(
            f"\nz-bin {b+1}: "
            f"N={len(selected)} "
            f"z=[{z_min:.3f}, {z_max:.3f}]"
        )

        flux_stack_w, err_stack_w = stack(flux_list, err_list)

        # ============================
        # Sigma_SFR stack
        # ============================

        flux_stack_sigma_w, err_stack_sigma_w = stack(
            flux_list_sigma,
            err_list_sigma
        )

        flux_stack_sigma_m = median_stack(
            flux_list_sigma
        )

        err_stack_sigma_m = bootstrap_median_error(
            flux_list_sigma
        )

        flux_stack_m = median_stack(flux_list)

        err_stack_m = bootstrap_median_error(flux_list)

        outname_w = f"results/JADES/JADES_DR3/spectra/stack_zbin{b+1}.txt"

        outname_m = f"results/JADES/JADES_DR3/spectra/stack_zbin{b+1}_median.txt"

        outname_sigma_w = (
            f"results/JADES/JADES_DR3/spectra/"
            f"stack_zbin{b+1}_sigmaSFR.txt"
        )

        outname_sigma_m = (
            f"results/JADES/JADES_DR3/spectra/"
            f"stack_zbin{b+1}_sigmaSFR_median.txt"
        )

        np.savetxt(
            outname_w,
            np.column_stack([
                wave_grid,
                flux_stack_w,
                err_stack_w
            ]),
            header=(
                f"wave flux err | "
                f"weighted stack | "
                f"z-bin {b+1}/{n_bins} | "
                f"N={len(selected)} | "
                f"z=[{z_min:.5f},{z_max:.5f}]"
            )
        )

        np.savetxt(
            outname_m,
            np.column_stack([
                wave_grid,
                flux_stack_m,
                err_stack_m
            ]),
            header=(
                f"wave flux err | "
                f"median stack | "
                f"z-bin {b+1}/{n_bins}"
            )
        )

        np.savetxt(
            outname_sigma_w,
            np.column_stack([
                wave_grid,
                flux_stack_sigma_w,
                err_stack_sigma_w
            ]),
            header=(
                f"wave flux err | "
                f"weighted stack | "
                f"Sigma_SFR subsample | "
                f"z-bin {b+1}/{n_bins} | "
                f"N={len(selected_sigma)} | "
                f"logSigmaSFR="
                f"{sigma_mean:.5f}+/-{sigma_err:.5f}"
            )
        )

        np.savetxt(
            outname_sigma_m,
            np.column_stack([
                wave_grid,
                flux_stack_sigma_m,
                err_stack_sigma_m
            ]),
            header=(
                f"wave flux err | "
                f"median stack | "
                f"Sigma_SFR subsample | "
                f"z-bin {b+1}/{n_bins}"
            )
        )

        print("saved:", outname_w)
        print("saved:", outname_m)
        print("saved:", outname_sigma_w)
        print("saved:", outname_sigma_m)

print("\nDone.")