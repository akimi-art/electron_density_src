#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
JADES spectral stacking pipeline

Produces:
    - weighted mean stack
    - median stack
    - bootstrap error (median)

Author: A.M.
"""

import numpy as np
import pandas as pd
import glob
from astropy.io import fits
from scipy.interpolate import interp1d


# =========================
# SETTINGS
# =========================

wave_grid = np.arange(6500, 6900, 0.5)

csv_file = "results/JADES/JADES_DR3/data_from_Nishigaki/jades_info_with_HA_plus_logSFR.csv"

spec_dir = "results/JADES/JADES_DR3/JADES_DR3_full_spectra"


# =========================
# READ SPECTRUM
# =========================

def read_spectrum(file):

    with fits.open(file) as h:

        data = h["EXTRACT1D"].data

        wave = data["WAVELENGTH"] * 1e4
        flux = data["FLUX"]
        err  = data["FLUX_ERR"]

    return wave, flux, err


# =========================
# RESTFRAME
# =========================

def restframe(wave, flux, err, z):

    wave = wave / (1 + z)

    return wave, flux, err


# =========================
# RESAMPLE
# =========================

def resample(wave, flux, err):

    f = interp1d(wave, flux, bounds_error=False, fill_value=np.nan)
    e = interp1d(wave, err , bounds_error=False, fill_value=np.nan)

    return f(wave_grid), e(wave_grid)


# =========================
# ARTIFACT MASK
# =========================

def mask_artifact(flux):

    med = np.nanmedian(flux)
    std = np.nanstd(flux)

    mask = flux < med - 5*std
    flux[mask] = np.nan

    return flux


# =========================
# WEIGHTED STACK
# =========================

def weighted_stack(fluxes, errs):

    fluxes = np.array(fluxes)
    errs   = np.array(errs)

    floor = 0.05 * np.nanmedian(errs)
    errs  = np.sqrt(errs**2 + floor**2)

    w = 1 / errs**2

    flux_stack = np.nansum(fluxes*w, axis=0) / np.nansum(w, axis=0)

    err_stack = np.sqrt(1 / np.nansum(w, axis=0))

    return flux_stack, err_stack


# =========================
# MEDIAN STACK
# =========================

def median_stack(fluxes):

    fluxes = np.array(fluxes)

    flux_stack = np.nanmedian(fluxes, axis=0)

    return flux_stack


# =========================
# BOOTSTRAP ERROR
# =========================

def bootstrap_median_error(fluxes, n_boot=1000):

    fluxes = np.array(fluxes)

    N = fluxes.shape[0]

    rng = np.random.default_rng(42)

    samples = []

    for i in range(n_boot):

        idx = rng.integers(0, N, N)

        sample = fluxes[idx]

        samples.append(np.nanmedian(sample, axis=0))

    samples = np.array(samples)

    err = np.nanstd(samples, axis=0)

    return err


# =========================
# SPECTRA MATRIX PLOT
# =========================
def plot_spectra_matrix(fluxes):

    import matplotlib.pyplot as plt

    fluxes = np.array(fluxes)

    plt.figure(figsize=(6,8))

    plt.imshow(
        fluxes,
        aspect="auto",
        vmin=np.nanpercentile(fluxes,5),
        vmax=np.nanpercentile(fluxes,95)
    )

    plt.xlabel("wavelength pixel")
    plt.ylabel("galaxy index")
    plt.title(f"spectra matrix (N={fluxes.shape[0]})")

    plt.colorbar(label="normalized flux")

    plt.tight_layout()

    plt.show()


# =========================
# CONTRIBUTING GALAXIES PLOT
# =========================
def plot_contributing_galaxies(fluxes, wave_grid):

    import numpy as np
    import matplotlib.pyplot as plt

    fluxes = np.array(fluxes)

    # finite な値を持つ銀河数
    n_contrib = np.sum(np.isfinite(fluxes), axis=0)

    plt.figure(figsize=(7,4))

    plt.plot(wave_grid, n_contrib)

    plt.xlabel("Rest wavelength [Å]")
    plt.ylabel("Number of galaxies contributing")

    plt.title("Contributing galaxies vs wavelength")

    plt.axvline(6563, linestyle="--")
    plt.axvline(6716, linestyle="--")
    plt.axvline(6731, linestyle="--")

    plt.tight_layout()

    plt.show()


# =========================
# MAIN
# =========================

df = pd.read_csv(csv_file)

flux_list = []
err_list  = []

for _, row in df.iterrows():

    nid = int(row["NIRSpec_ID"])
    z   = row["z_spec"]
    ha  = row["HA_6563_flux"]

    if not np.isfinite(ha) or ha <= 0:
        continue

    sid = f"{nid:08d}"

    files = glob.glob(f"{spec_dir}/*{sid}*_x1d.fits")

    if len(files) == 0:
        continue

    try:

        wave, flux, err = read_spectrum(files[0])

        wave, flux, err = restframe(wave, flux, err, z)

        flux = mask_artifact(flux)

        flux = flux / ha
        err  = err / ha

        flux_i, err_i = resample(wave, flux, err)

        if not np.isfinite(flux_i).any():
            continue

        flux_list.append(flux_i)
        err_list.append(err_i)

    except:
        continue


print("spectra used:", len(flux_list))


# convert to numpy
flux_list = np.array(flux_list)
err_list  = np.array(err_list)


# =========================
# VISUALIZATION
# =========================

plot_spectra_matrix(flux_list)

plot_contributing_galaxies(flux_list, wave_grid)


# =========================
# STACK
# =========================

flux_w, err_w = weighted_stack(flux_list, err_list)

flux_m = median_stack(flux_list)

err_m = bootstrap_median_error(flux_list)


# =========================
# SAVE
# =========================

np.savetxt(

    "stack_weighted.txt",

    np.column_stack([wave_grid, flux_w, err_w]),

    header="wave flux err | weighted mean stack"

)

np.savetxt(

    "stack_median.txt",

    np.column_stack([wave_grid, flux_m, err_m]),

    header="wave flux err | median stack"

)

print("Stacking complete.")