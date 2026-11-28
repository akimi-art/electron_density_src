#!/usr/bin/env python3
"""Mean [S II] stack of local DR7 spectra matched to the Ka03/Reu master."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parents[3]
MASTER = ROOT / "results/fits/mpajhu_dr7_v5_2_master_v2_Ka03_Reu.fits"
SPECTRA = ROOT / "data/data_SDSS/DR7/spectra/fit"
OUTPUT = ROOT / "results/figure/SDSS_DR7_SII_master_mean_stack.png"
FLUX_COLUMNS = ("SII_6717_FLUX", "SII_6731_FLUX", "H_ALPHA_FLUX")
LINE_CENTERS = np.array([6718.29, 6732.67])
WAVE_GRID = np.arange(6500, 6900, 0.5)
SII_WINDOW = (WAVE_GRID > LINE_CENTERS[0] - 100) & (
    WAVE_GRID < LINE_CENTERS[1] + 100
)


def matched_rows(paths, extra_fields=()):
    """Find master rows by the plate, MJD and fiber in each spSpec filename."""
    keys = [
        (int(plate), int(mjd), int(fiber))
        for mjd, plate, fiber in (path.stem.split("-")[1:] for path in paths)
    ]
    with fits.open(MASTER) as hdus:
        catalog = hdus[1].data
        ids = zip(catalog["PLATEID"], catalog["MJD"], catalog["FIBERID"])
        local_keys = set(keys)
        indices = {}
        for index, key in enumerate(ids):
            if key in local_keys:
                if key in indices:
                    raise ValueError(f"Duplicate master entry for {key}")
                indices[key] = index
        matched = [(path, indices[key]) for path, key in zip(paths, keys) if key in indices]
        if not matched:
            raise RuntimeError("No local spSpec files matched the master catalog.")
        matched_paths, row_indices = zip(*matched)
        fields = ("Z", *FLUX_COLUMNS, *(f"{name}_ERR" for name in FLUX_COLUMNS), *extra_fields)
        values = {name: np.asarray(catalog[name][list(row_indices)], float) for name in fields}
    return matched_paths, values


def resample(path, redshift):
    with fits.open(path) as hdus:
        header = hdus[0].header
        flux, noise = hdus[0].data[[0, 2]]
        wave = 10 ** (header["COEFF0"] + header["COEFF1"] * np.arange(flux.size))
        if not header["VACUUM"]:
            raise ValueError(f"Expected vacuum wavelengths in {path.name}")

    # Keep gaps at invalid pixels during interpolation.
    bad = ~np.isfinite(flux) | ~np.isfinite(noise) | (noise <= 0)
    flux[bad] = np.nan
    noise[bad] = np.nan
    rest_wave = wave / (1 + redshift)
    return (
        np.interp(WAVE_GRID, rest_wave, flux, left=np.nan, right=np.nan),
        np.interp(WAVE_GRID, rest_wave, noise, left=np.nan, right=np.nan),
    )


def mean_stack(flux, noise, counts):
    """Apply the JADES 500-draw mean-stack estimator at each wavelength."""
    return mean_stack_from_sums(np.nansum(flux, axis=0), np.nansum(noise**2, axis=0), counts)


def mean_stack_from_sums(flux_sum, variance_sum, counts):
    covered = counts > 0
    means = flux_sum[covered] / counts[covered]
    # The average of independent Gaussian draws is itself Gaussian.
    scale = np.sqrt(variance_sum[covered]) / counts[covered]
    rng = np.random.default_rng(42)
    draws = rng.normal(means[:, None], scale[:, None], size=(len(means), 500))
    p16, median, p84 = np.percentile(draws, [16, 50, 84], axis=1)
    stack = np.full(counts.shape, np.nan)
    stack_err = stack.copy()
    stack[covered] = median
    stack_err[covered] = (p84 - p16) / 2
    return stack, stack_err


def fit_doublet(wave, stack, stack_err, *, return_ratio=False):
    """Fit the JADES shared-width doublet plus linear continuum in vacuum."""
    fit_range = (wave > 6650) & (wave < 6800)
    use = fit_range & np.isfinite(stack) & np.isfinite(stack_err) & (stack_err > 0)
    x, y = wave[use], stack[use]
    pivot = x.mean()

    def model(wavelength, amp1, amp2, sigma, shift, offset, slope):
        center1, center2 = LINE_CENTERS + shift
        line1 = amp1 * np.exp(-0.5 * ((wavelength - center1) / sigma) ** 2)
        line2 = amp2 * np.exp(-0.5 * ((wavelength - center2) / sigma) ** 2)
        return line1 + line2 + offset + slope * (wavelength - pivot)

    continuum = np.median(y)
    amplitudes = np.interp(LINE_CENTERS, x, y) - continuum
    params, _ = curve_fit(
        model, x, y,
        p0=[*amplitudes, 2, 0, continuum, 0],
        sigma=stack_err[use],
        absolute_sigma=True,
    )
    print(f"[S II] fit: ratio={params[0] / params[1]:.3f}, "
          f"sigma={abs(params[2]):.3f} Å, shift={params[3]:+.3f} Å")
    model_wave = np.linspace(6650, 6800, 1000)
    result = model_wave, model(model_wave, *params), LINE_CENTERS + params[3]
    return (*result, params[0] / params[1]) if return_ratio else result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inner-percentile", type=float, default=2)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    low = args.inner_percentile
    if not 0 <= low < 50:
        parser.error("--inner-percentile must be in [0, 50)")

    paths = sorted(SPECTRA.glob("spSpec-*.fit"))
    matched_paths, catalog = matched_rows(paths)
    fluxes = np.column_stack([catalog[name] for name in FLUX_COLUMNS])
    errors = np.column_stack([catalog[f"{name}_ERR"] for name in FLUX_COLUMNS])
    redshifts = catalog["Z"]
    valid = (
        (np.isfinite(fluxes) & (fluxes > 0)).all(axis=1)
        & (np.isfinite(errors) & (errors > 0)).all(axis=1)
        & np.isfinite(redshifts)
        & (redshifts > -1)
    )

    # Estimate the cuts within the usable local sample, without binning.
    spectra = []
    spectrum_errors = []
    sample_fluxes = []
    line_pixels = np.abs(WAVE_GRID[:, None] - LINE_CENTERS).argmin(axis=0)
    for index in np.flatnonzero(valid):
        spectrum, noise = resample(matched_paths[index], redshifts[index])
        if np.isfinite(spectrum[line_pixels]).all():
            spectra.append(spectrum[SII_WINDOW])
            spectrum_errors.append(noise[SII_WINDOW])
            sample_fluxes.append(fluxes[index])

    if not spectra:
        raise RuntimeError("No matched spectra with usable pixels at both [S II] lines.")

    sample_fluxes = np.asarray(sample_fluxes)
    lower, upper = np.percentile(sample_fluxes, [low, 100 - low], axis=0)
    keep = ((sample_fluxes >= lower) & (sample_fluxes <= upper)).all(axis=1)
    if not keep.any():
        raise RuntimeError("No spectra survived the catalog flux cuts.")
    raw = np.asarray(spectra)[keep]
    wave = WAVE_GRID[SII_WINDOW]

    print(f"Local FITS: {len(paths)}; matched: {len(matched_paths)}")
    print(f"Valid catalog values: {valid.sum()}; usable spectra: {len(spectra)}")
    for name, lo, hi in zip(FLUX_COLUMNS, lower, upper):
        print(f"{name}: [{lo:.3f}, {hi:.3f}] (1e-17 erg s^-1 cm^-2)")
    counts = np.isfinite(raw).sum(axis=0)
    print(f"Stacked: {len(raw)}; contributors per wavelength: {counts.min()}–{counts.max()}")
    stack, stack_err = mean_stack(raw, np.asarray(spectrum_errors)[keep], counts)
    model_wave, model_flux, fitted_centers = fit_doublet(wave, stack, stack_err)

    plot_single_stack(wave, stack, stack_err, model_wave, model_flux, fitted_centers, args.output)


def plot_single_stack(wave, stack, stack_err, model_wave, model_flux, fitted_centers, output,
                      *, minor_tick_size=4, figsize=(12, 7.5),
                      ylabel_size=None,
                      ylabel=r"$f_\lambda$ [$10^{-17}$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$]"):
    """Save a stack, measured uncertainty band, and doublet fit."""
    plt.rcParams.update({
        "figure.figsize": figsize,
        "font.size": 38, "axes.labelsize": 38, "axes.titlesize": 38,
        "axes.grid": False, "axes.linewidth": 3,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "xtick.major.size": 16, "ytick.major.size": 16,
        "xtick.major.width": 2.8, "ytick.major.width": 2.8,
        "xtick.minor.visible": True, "ytick.minor.visible": True,
        "xtick.minor.size": minor_tick_size, "ytick.minor.size": minor_tick_size,
        "xtick.minor.width": 2, "ytick.minor.width": 2,
        "xtick.labelsize": 34, "ytick.labelsize": 34,
        "font.family": "STIXGeneral", "mathtext.fontset": "stix",
    })
    fig, ax = plt.subplots(layout="constrained")
    fig.get_layout_engine().set(w_pad=0.12, h_pad=0.16)
    lower_band, upper_band = stack - stack_err, stack + stack_err
    ax.fill_between(
        wave, lower_band, upper_band,
        step="mid", color="gray", alpha=0.4,
    )
    ax.step(wave, stack, where="mid", color="black", lw=1.6)
    ax.plot(model_wave, model_flux, color="red", lw=2.4)
    for center, linestyle in zip(fitted_centers, ("--", "-.")):
        ax.axvline(center, color="red", ls=linestyle, lw=2)
    ax.set(
        xlabel=r"$\lambda$ [Å]",
        xlim=(wave[0], wave[-1]),
    )
    ax.set_ylabel(ylabel, fontsize=ylabel_size)
    # Match the JADES plot: use the visible error envelope plus 5% padding.
    visible = np.isfinite(lower_band) & np.isfinite(upper_band)
    ymin, ymax = lower_band[visible].min(), upper_band[visible].max()
    pad = 0.05 * ((ymax - ymin) if ymax > ymin else max(abs(ymin), 1e-30))
    ax.set_ylim(ymin - pad, ymax + pad)
    print(f"Plot y limits: {ymin - pad:.3f} to {ymax + pad:.3f}")
    fig.savefig(output, dpi=220)
    plt.close(fig)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
