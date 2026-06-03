#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
スクリプトの概要:
JADESスペクトルスタックを作成します。
z, 物理量（例: SFR, Mstar, sSFR, Sigma_SFR）を基に、スペクトルを複数のビンに分割してスタックします。

使用方法:
    JADES_spectra_stack_x_sfr_equal width.py [オプション]

著者: A. M.
作成日: 2026-05-29
最終更新日: 2026-06-01

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
# 試金石
# sigma_bins = [
#     (-3.0, -2.0),
#     (-2.0, -1.0),
#     (-1.0, 0.0),
#     (0.0, 1.0),
# ]
sfr_bins = [
    (-1.0, 0.0),
    (0.0, 1.0),
    (1.0, 2.0),
]


# ============================
# CSV
# ============================

df = pd.read_csv(csv_file)

df = df[df["z_spec"].notna()]
df = df[df["HA_6563_flux"].notna()]
df = df[df["log10_SFR_hb"].notna()]
# df = df[df["log10_SFR_hb"] >= 0] # なぜ入っている?

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


def weighted_mean(values, err_lo, err_hi):

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
            logSFR = row["log10_SFR_hb"]

            logSFR_err_lo = row["log10_SFR_hb_err_lower"]
            logSFR_err_hi = row["log10_SFR_hb_err_upper"]

            used_items_all.append({

                "id": sid,

                "flux": flux_i,
                "err": err_i,

                "logSFR": logSFR,
                "logSFR_err_lo": logSFR_err_lo,
                "logSFR_err_hi": logSFR_err_hi,

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
        it["logSFR"]
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
        used_sfr_all,
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
    # stack each SFR bin
    # =====================================

    for b, (lo, hi) in enumerate(sfr_bins):

        # outlierを除去するために、以下のIDを除外します。
        # 確実に弾いてよいもの
        # Ha, SII領域に欠損(NaN)がある
        # 明らかなデータ落ち
        # SII波長域がスペクトル端にかかる
        # 極端な単一ピクセルスパイク
        bad_ids = [
            "00024958", 
            "00025356", 
            "00051209", 
            "00082961",

            "00004504", 
            "00028139", 

            "00045967", 

        ]

        selected = [
        
            it

            for it in used_items_valid

            if (
                (it["logSFR"] >= lo)
                and
                (it["logSFR"] < hi)
                and
                (it["id"] not in bad_ids)
            )
        ]

        if len(selected) == 0:

            print(
                f"logSFR [{lo},{hi}) : empty"
            )

            continue


        sfr_vals = np.array([
            it["logSFR"]
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
            it["logSFR_err_lo"]
            for it in selected
        ]

        sfr_err_hi = [
            it["logSFR_err_hi"]
            for it in selected
        ]

        sfr_mean, sfr_err = weighted_mean(
            sfr_vals,
            sfr_err_lo,
            sfr_err_hi
        )

        print(
            f"\nlogSFR [{lo},{hi})"
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

        # =====================================
        # output names
        # =====================================

        outname_w = (
            "results/JADES/JADES_DR3/spectra/"
            f"stack_sfr_{lo:+.1f}_{hi:+.1f}.txt"
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
                f"logSFR=[{lo},{hi}) | "
                f"N={len(selected)} | "
                f"logSFR="
                f"{sfr_mean:.5f}+/-{sfr_err:.5f}"
            )
        )

        print("saved:", outname_w)

print("\nDone.")