#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
スクリプトの概要:
SDSSスペクトルスタックを作成します。

使用方法:
    SDSS_spectra_stack_x_mass_equal_width.py [オプション]

著者: A. M.
作成日: 2026-07-10
最終更新日: 2026-07-10

参考文献:
    - PEP 8:                  https://peps.python.org/pep-0008/
    - PEP 257 (Docstring規約): https://peps.python.org/pep-0257/
    - Python公式ドキュメント:    https://docs.python.org/ja/3/
"""


import glob
import os
import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from astropy.cosmology import Planck18


def read_spectrum(filename):

    with fits.open(filename) as h:

        flux = h[0].data[0]

        # SDSS DR7
        ivar = h[0].data[2]

        hdr = h[0].header

        coeff0 = hdr["COEFF0"]
        coeff1 = hdr["COEFF1"]

    pix = np.arange(len(flux))

    wave = 10**(coeff0 + coeff1 * pix)

    err = np.full_like(flux, np.nan)

    good = ivar > 0

    err[good] = np.sqrt(1.0 / ivar[good])

    return wave, flux, err

wave_grid = np.arange(
    6500,
    6800,
    0.5
)

def resample(wave, flux, err):

    f = interp1d(
        wave,
        flux,
        bounds_error=False,
        fill_value=np.nan
    )

    e = interp1d(
        wave,
        err,
        bounds_error=False,
        fill_value=np.nan
    )

    return (
        f(wave_grid),
        e(wave_grid)
    )

# カタログ読込
data = fits.getdata("results/fits/mpajhu_dr7_v5_2_merged_zlt0.2_Lgt1e+39.fits")

sample = data[
    (data["sm_MEDIAN"] >= 9.5)
    &
    (data["sm_MEDIAN"] < 10.0)
]

used_items = []

for row in sample:

    plate = int(row["PLATEID"])
    mjd   = int(row["MJD"])
    fiber = int(row["FIBERID"])

    z  = row["Z"]
    ha = row["H_ALPHA_FLUX"]

    if (not np.isfinite(ha)) or (ha <= 0):
        continue

    filename = (
        f"data/data_SDSS/DR7/spectra/fit/"
        f"spSpec-{mjd}-{plate:04d}-{fiber:03d}.fit"
    )

    if not os.path.exists(filename):
        continue

    try:

        wave, flux, err = read_spectrum(filename)

        wave = wave / (1 + z)

        ha_region = (
            (wave > 6550)
            &
            (wave < 6580)
        )

        if np.sum(np.isfinite(flux[ha_region])) < 10:
            continue

        flux_i, err_i = resample(
            wave,
            flux,
            err
        )

        flux_norm_i = flux_i / ha
        err_norm_i  = err_i / ha

        used_items.append({

            "flux": flux_i,
            "err": err_i,

            "flux_norm": flux_norm_i,
            "err_norm": err_norm_i,

            "logM": row["sm_MEDIAN"],
            "z": z

        })

    except Exception:

        continue

print("usable spectra =", len(used_items))

if len(used_items) == 0:
    raise RuntimeError("No usable spectra.")

flux_raw = np.array(
    [it["flux"] for it in used_items]
)

err_raw = np.array(
    [it["err"] for it in used_items]
)

flux_norm = np.array(
    [it["flux_norm"] for it in used_items]
)

err_norm = np.array(
    [it["err_norm"] for it in used_items]
)

# =========================
# Monte Carlo
# =========================

N_MC = 500

rng = np.random.default_rng(42)

flux_raw_mc = rng.normal(

    flux_raw[:, :, None],

    err_raw[:, :, None],

    size=(
        flux_raw.shape[0],
        flux_raw.shape[1],
        N_MC
    )
)

flux_norm_mc = rng.normal(

    flux_norm[:, :, None],

    err_norm[:, :, None],

    size=(
        flux_norm.shape[0],
        flux_norm.shape[1],
        N_MC
    )
)

mean_raw_mc = np.nanmean(
    flux_raw_mc,
    axis=0
)

mean_raw = np.nanmedian(
    mean_raw_mc,
    axis=1
)

mean_raw_err = (
    np.nanpercentile(
        mean_raw_mc,
        84,
        axis=1
    )
    -
    np.nanpercentile(
        mean_raw_mc,
        16,
        axis=1
    )
)/2

mean_norm_mc = np.nanmean(
    flux_norm_mc,
    axis=0
)

mean_norm = np.nanmedian(
    mean_norm_mc,
    axis=1
)

mean_norm_err = (
    np.nanpercentile(
        mean_norm_mc,
        84,
        axis=1
    )
    -
    np.nanpercentile(
        mean_norm_mc,
        16,
        axis=1
    )
)/2

median_raw_mc = np.nanmedian(
    flux_raw_mc,
    axis=0
)

median_raw = np.nanmedian(
    median_raw_mc,
    axis=1
)

median_raw_err = (
    np.nanpercentile(
        median_raw_mc,
        84,
        axis=1
    )
    -
    np.nanpercentile(
        median_raw_mc,
        16,
        axis=1
    )
)/2

median_norm_mc = np.nanmedian(
    flux_norm_mc,
    axis=0
)

median_norm = np.nanmedian(
    median_norm_mc,
    axis=1
)

median_norm_err = (
    np.nanpercentile(
        median_norm_mc,
        84,
        axis=1
    )
    -
    np.nanpercentile(
        median_norm_mc,
        16,
        axis=1
    )
)/2

w_raw = np.zeros_like(err_raw)

good = (
    np.isfinite(err_raw)
    &
    (err_raw > 0)
)

w_raw[good] = (
    1.0 / err_raw[good]**2
)

weighted_raw_mc = (

    np.nansum(
        w_raw[:, :, None]
        *
        flux_raw_mc,
        axis=0
    )
    /
    np.nansum(
        w_raw[:, :, None],
        axis=0
    )
)

weighted_raw = np.nanmedian(
    weighted_raw_mc,
    axis=1
)

weighted_raw_err = (

    np.nanpercentile(
        weighted_raw_mc,
        84,
        axis=1
    )
    -
    np.nanpercentile(
        weighted_raw_mc,
        16,
        axis=1
    )
)/2

w_norm = np.zeros_like(err_norm)

good = (
    np.isfinite(err_norm)
    &
    (err_norm > 0)
)

w_norm[good] = (
    1.0 / err_norm[good]**2
)

weighted_norm_mc = (

    np.nansum(
        w_norm[:, :, None]
        *
        flux_norm_mc,
        axis=0
    )
    /
    np.nansum(
        w_norm[:, :, None],
        axis=0
    )
)

weighted_norm = np.nanmedian(
    weighted_norm_mc,
    axis=1
)

weighted_norm_err = (

    np.nanpercentile(
        weighted_norm_mc,
        84,
        axis=1
    )
    -
    np.nanpercentile(
        weighted_norm_mc,
        16,
        axis=1
    )
)/2

np.savetxt(
    "stack_mean_raw.txt",
    np.column_stack([
        wave_grid,
        mean_raw,
        mean_raw_err
    ])
)

np.savetxt(
    "stack_median_raw.txt",
    np.column_stack([
        wave_grid,
        median_raw,
        median_raw_err
    ])
)

np.savetxt(
    "stack_weighted_raw.txt",
    np.column_stack([
        wave_grid,
        weighted_raw,
        weighted_raw_err
    ])
)

np.savetxt(
    "stack_mean_norm.txt",
    np.column_stack([
        wave_grid,
        mean_norm,
        mean_norm_err
    ])
)

np.savetxt(
    "stack_median_norm.txt",
    np.column_stack([
        wave_grid,
        median_norm,
        median_norm_err
    ])
)

np.savetxt(
    "stack_weighted_norm.txt",
    np.column_stack([
        wave_grid,
        weighted_norm,
        weighted_norm_err
    ])
)

plt.figure(figsize=(6,4))

plt.hist(
    [it["logM"] for it in used_items],
    bins=30,
    edgecolor="black"
)

plt.xlabel(r"$\log M_\star$")
plt.ylabel("Count")

plt.tight_layout()

plt.savefig("hist_mass.png")

plt.figure(figsize=(6,4))

plt.hist(
    [it["z"] for it in used_items],
    bins=30,
    edgecolor="black"
)

plt.xlabel("z")
plt.ylabel("Count")

plt.tight_layout()

plt.savefig("hist_z.png")

plt.figure(figsize=(8,5))

for spec in flux_raw:

    plt.plot(
        wave_grid,
        spec,
        color="0.8",
        lw=0.4,
        alpha=0.2
    )

plt.plot(
    wave_grid,
    mean_raw,
    color="black",
    lw=3,
    label="Mean"
)

plt.plot(
    wave_grid,
    median_raw,
    color="red",
    lw=2,
    label="Median"
)

plt.plot(
    wave_grid,
    weighted_raw,
    color="blue",
    lw=2,
    label="Weighted Mean"
)

plt.axvline(6564.61, ls="--")
plt.axvline(6718.29, ls=":")
plt.axvline(6732.67, ls=":")

plt.xlim(6718.29-50, 6732.67+50)
plt.ylim(0,50)
plt.legend()
plt.tight_layout()

plt.savefig("SDSS_raw_stack.png")

plt.figure(figsize=(8,5))

for spec in flux_norm:

    plt.plot(
        wave_grid,
        spec,
        color="0.8",
        lw=0.4,
        alpha=0.2
    )

plt.plot(
    wave_grid,
    mean_norm,
    color="black",
    lw=3,
    label="Mean"
)

plt.plot(
    wave_grid,
    median_norm,
    color="red",
    lw=2,
    label="Median"
)

plt.plot(
    wave_grid,
    weighted_norm,
    color="blue",
    lw=2,
    label="Weighted Mean"
)

plt.axvline(6564.61, ls="--")
plt.axvline(6718.29, ls=":")
plt.axvline(6732.67, ls=":")

plt.xlim(6718.29-50, 6732.67+50)
plt.ylim(0,0.15)
plt.legend()
plt.tight_layout()

plt.savefig("SDSS_norm_stack.png")


print(
    "N spectra =",
    len(used_items)
)

print(
    "median logM =",
    np.median([it["logM"] for it in used_items])
)

print(
    "median z =",
    np.median([it["z"] for it in used_items])
)

with fits.open(
    "data/data_SDSS/DR7/spectra/fit/spSpec-51609-0292-084.fit"
) as h:

    print(h[0].data.shape)

    for i in range(4):

        arr = h[0].data[i]

        print(
            i,
            np.nanmin(arr),
            np.nanmax(arr),
            np.nanmedian(arr)
        )