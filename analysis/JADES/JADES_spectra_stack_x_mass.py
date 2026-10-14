#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
スクリプトの概要:
JADESスペクトルスタックを作成します。
z, 物理量（例: SFR, Mstar, sSFR, Sigma_SFR）を基に、スペクトルを複数のビンに分割してスタックします。

使用方法:
    JADES_spectra_stack_x_bin.py [オプション]

著者: A. M.
作成日: 2026-06-03
最終更新日: 2026-06-03

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
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from astropy.cosmology import Planck18
from scipy.optimize import curve_fit

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
# ↓追加
n_phys_bins = 1 # 変更

# ============================
# CSV
# ============================

df = pd.read_csv(csv_file)

df = df[df["z_spec"].notna()]
df = df[df["HA_6563_flux"].notna()]
df = df[df["log10_SFR_hb"].notna()]
df = df[df["logM"].notna()]
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

def compute_log_ssfr(
    logSFR,
    errSFR_lo,
    errSFR_hi,
    logM,
    errM_lo,
    errM_hi
):

    if (
        (not np.isfinite(logSFR))
        or
        (not np.isfinite(logM))
    ):

        return np.nan, np.nan, np.nan

    # -------------------------
    # central value
    # -------------------------

    log_sSFR = logSFR - logM

    # -------------------------
    # uncertainty
    # -------------------------

    sfr_sigma = 0.5 * (
        errSFR_lo + errSFR_hi
    )

    mass_sigma = 0.5 * (
        errM_lo + errM_hi
    )

    total_sigma = np.sqrt(
        sfr_sigma**2
        +
        mass_sigma**2
    )

    return (
        log_sSFR,
        total_sigma,
        total_sigma
    )

# ↓追加
def gaussian(x, amp, mu, sigma, bg):

    return (
        amp *
        np.exp(
            -(x-mu)**2 /
            (2*sigma**2)
        )
        +
        bg
    )

def fit_Ha_center(
    wave,
    flux
):

    mask = (
        (wave > 6550)
        &
        (wave < 6575)
    )

    x = wave[mask]
    y = flux[mask]

    if len(x) < 5:
        return np.nan

    try:

        p0 = [
            np.nanmax(y),
            6564.61, # 真空中
            2.0,
            np.nanmedian(y)
        ]

        popt, _ = curve_fit(
            gaussian,
            x,
            y,
            p0=p0,
            maxfev=10000
        )

        return popt[1]

    except:

        return np.nan



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

            # ↓ 追加
            log_sSFR, log_sSFR_err_lo, log_sSFR_err_hi = (
                compute_log_ssfr(
                    row["log10_SFR_hb"],
                    row["log10_SFR_hb_err_lower"],
                    row["log10_SFR_hb_err_upper"],

                    row["logM"],
                    row["err1_logM"],
                    row["err2_logM"]
                )
            )

            used_items_all.append({
                "z": z,
                "flux": flux_i,
                "err": err_i,
                "id": sid,
                "gr": gr,
                # ↓ 追加
                "wave_rest": wave,
                "flux_rest": flux,

                "sfr": row["log10_SFR_hb"],
                "sfr_err_lo": row["log10_SFR_hb_err_lower"],
                "sfr_err_hi": row["log10_SFR_hb_err_upper"],

                # ↓ 追加
                "sigma_sfr": logSigma,
                "sigma_sfr_err_lo": logSigma_err_lo,
                "sigma_sfr_err_hi": logSigma_err_hi,

                # ↓ 追加
                "logM": row["logM"],
                "logM_err_lo": row["err1_logM"],
                "logM_err_hi": row["err2_logM"],

                # ↓ 追加
                "ssfr": log_sSFR,
                "ssfr_err_lo": log_sSFR_err_lo,
                "ssfr_err_hi": log_sSFR_err_hi,
                })

        except Exception:
            continue

# ============================
# Sigma_SFR-bin split
# ============================
if len(used_items_all) == 0:

    print("No usable spectra.")

else:

    # =====================================
    # use Sigma_SFR as binning variable
    # =====================================

    used_sfr_all = np.array([
        it["sfr"]
        for it in used_items_all
    ])

    valid_mask = np.isfinite(used_sfr_all)

    used_sfr_all = used_sfr_all[valid_mask]

    used_items_valid = [
        used_items_all[i]
        for i in range(len(used_items_all))
        if valid_mask[i]
    ]

    N = len(used_sfr_all)

    print("\nTotal usable spectra:", N)

    # =====================================
    # histogram
    # =====================================

    plt.figure(figsize=(6,4))

    plt.hist(
        used_mass_all,
        bins=30,
        color="0.7",
        edgecolor="black"
    )

    plt.xlabel(r'$\log {\rm SFR}$')
    plt.ylabel("count")

    plt.tight_layout()
    plt.show()

    print("median =", np.nanmedian(used_sfr_all))
    print("std =", np.nanstd(used_sfr_all))

    # =====================================
    # equal-number bins
    # =====================================

    sort_idx = np.argsort(used_sfr_all)

    q, r = divmod(N, n_phys_bins)

    counts = [
        q + 1 if i < r else q
        for i in range(n_phys_bins)
    ]

    cum = np.cumsum([0] + counts)

    print("equal-count bins:", counts)

    # =====================================
    # stack each Sigma_SFR bin
    # =====================================

    for b in range(n_phys_bins):

        s = cum[b]
        e = cum[b+1]

        sel_idx = sort_idx[s:e]

        selected = [
            used_items_valid[i]
            for i in sel_idx
        ]

        # outlierを除去するために、以下のIDを除外します。
        bad_ids = [
        
            "00051209", # bin1

            "00028139", # bin2

            "00045967", # bin3

        ]

        selected = [
            it
            for it in selected
            if it["id"] not in bad_ids
        ]

        # ↓追加
        ha_offsets = []

        for it in selected:
        
            center = fit_Ha_center(
                it["wave_rest"],
                it["flux_rest"]
            )

            if np.isfinite(center):
            
                ha_offsets.append(
                    center - 6564.61 # 真空中
                )

        ha_offsets = np.array(
            ha_offsets
        )

        print(
            "Hα offset mean =",
            np.nanmean(ha_offsets)
        )

        print(
            "Hα offset std =",
            np.nanstd(ha_offsets)
        )

        plt.figure(figsize=(6,4))

        plt.hist(
            ha_offsets,
            bins=20,
            color="0.7",
            edgecolor="black"
        )

        plt.axvline(
            0,
            color="red",
            ls="--"
        )

        plt.xlabel(
            r"$\lambda_{H\alpha}-6564.61$ (Å)"
        )

        plt.ylabel("count")

        plt.title(
            f"Hα offset distribution bin {b+1}"
        )

        plt.show()



        # ↓ 追加
        plt.figure(figsize=(6,6))

        for it in selected:
        
            wave = it["wave_rest"]
            flux = it["flux_rest"]

            mask = (
                (wave > 6540)
                & (wave < 6585)
            )

            wave_sel = wave[mask]
            flux_sel = flux[mask]

            # ★ここで平滑化
            flux_smooth = gaussian_filter1d(flux_sel, sigma=2)

            plt.plot(
                wave_sel,
                flux_smooth,
                alpha=0.3,
                color="black"
            )

        plt.axvline(6564.61, color="red", ls="--")

        plt.xlabel("Rest wavelength (A)")
        plt.ylabel("Flux")
        plt.show()

        # ↓追加
        for it in selected:
        
            wave = it["wave_rest"]
            flux = it["flux_rest"]

            mask = (
                (wave > 6540)
                &
                (wave < 6585)
            )

            # 確認用のプロット（必要に応じてコメントアウト）
            # plt.figure(figsize=(6,4))

            # plt.plot(
            #     wave[mask],
            #     flux[mask]
            # )

            # plt.axvline(
            #     6562.8,
            #     color="red",
            #     ls="--"
            # )

            # plt.title(it["id"])

            # plt.show()


        sfr_vals = np.array([
            it["sfr"]
            for it in selected
        ])

        flux_list = [
            it["flux"]
            for it in selected
        ]

        err_list = [
            it["err"]
            for it in selected
        ]

        # =====================================
        # weighted mean Sigma_SFR
        # =====================================

        sfr_err_lo = [
            it["sfr_err_lo"]
            for it in selected
        ]

        sfr_err_hi = [
            it["sfr_err_hi"]
            for it in selected
        ]

        sfr_mean, sfr_err = weighted_mean_sfr(
            sfr_vals,
            sfr_err_lo,
            sfr_err_hi
        )

        print(
            f"\nSFR bin {b+1}"
        )

        print(
            f"N = {len(selected)}"
        )

        print(
            f"logSFR = "
            f"{sfr_mean:.3f} ± {sfr_err:.3f}"
        )

        print(
            f"range = "
            f"[{np.min(sfr_vals):.3f}, "
            f"{np.max(sfr_vals):.3f}]"
        )

        # =====================================
        # stack
        # =====================================

        flux_stack_w, err_stack_w = stack(
            flux_list,
            err_list
        )

        flux_stack_m = median_stack(
            flux_list
        )

        err_stack_m = bootstrap_median_error(
            flux_list
        )

        # =====================================
        # output names
        # =====================================

        outname_w = (
            f"results/JADES/JADES_DR3/spectra/"
            f"stack_SFR_bin{b+1}.txt"
        )

        outname_m = (
            f"results/JADES/JADES_DR3/spectra/"
            f"stack_SFR_bin{b+1}_median.txt"
        )

        # =====================================
        # save
        # =====================================

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
                f"SFR bin {b+1}/{n_phys_bins} | "
                f"N={len(selected)} | "
                f"logSFR="
                f"{sfr_mean:.5f}+/-{sfr_err:.5f}"
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
                f"SFR bin {b+1}/{n_phys_bins}"
            )
        )

        print("saved:", outname_w)
        print("saved:", outname_m)

print("\nDone.")