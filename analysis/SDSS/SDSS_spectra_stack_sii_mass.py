#!/usr/bin/env python3
"""Stack downloaded DR7 [S II] spectra in 0.2 dex stellar-mass bins."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table

from SDSS_spectra_stack_sii_master import (
    LINE_CENTERS, ROOT, SII_WINDOW, WAVE_GRID, fit_doublet, matched_rows,
    mean_stack_from_sums, plot_single_stack, resample,
)

SPECTRA = ROOT / "data/data_SDSS/DR7/spectra/master_v2_Ka03_Reu"
OUTPUT = ROOT / "results/figure/SDSS_DR7_SII_master_mass_stacks.png"
LINES = ("SII_6717_FLUX", "SII_6731_FLUX")
WIDTH = 0.2
NMIN = 100


def stack_bin(indices, paths, redshifts):
    flux_sum = np.zeros(WAVE_GRID.size)
    variance_sum = np.zeros(WAVE_GRID.size)
    count = np.zeros(WAVE_GRID.size, dtype=int)
    centers = np.abs(WAVE_GRID[:, None] - LINE_CENTERS).argmin(axis=0)
    used = []
    failures = 0
    for index in indices:
        try:
            flux, noise = resample(paths[index], redshifts[index])
        except (OSError, KeyError, ValueError) as exc:
            failures += 1
            if failures <= 3:
                print(f"Skipping {paths[index].name}: {exc}", flush=True)
            continue
        if not np.isfinite(flux[centers]).all():
            continue
        good = np.isfinite(flux) & np.isfinite(noise)
        flux_sum[good] += flux[good]
        variance_sum[good] += noise[good] ** 2
        count[good] += 1
        used.append(index)
    stack, error = mean_stack_from_sums(flux_sum, variance_sum, count)
    return np.asarray(used, dtype=int), failures, stack, error, count


def plot_stacks(rows, output):
    plt.rcParams.update({
        "font.family": "STIXGeneral", "mathtext.fontset": "stix",
        "font.size": 18, "axes.labelsize": 18, "axes.titlesize": 19,
        "xtick.labelsize": 17, "ytick.labelsize": 17,
        "axes.linewidth": 2.8, "xtick.direction": "in",
        "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
        "xtick.minor.visible": True, "ytick.minor.visible": True,
        "xtick.major.width": 2, "ytick.major.width": 2,
        "xtick.minor.width": 1.5, "ytick.minor.width": 1.5,
    })
    ncols = min(4, len(rows))
    nrows = (len(rows) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 3.2 * nrows),
                             sharex=True, squeeze=False, layout="constrained")
    for ax, row in zip(axes.flat, rows):
        wave, stack, err = (row[name][SII_WINDOW] for name in ("WAVE", "FLUX", "FLUX_ERR"))
        lower, upper = stack - err, stack + err
        ax.fill_between(wave, lower, upper, step="mid", color="gray", alpha=0.4)
        ax.step(wave, stack, where="mid", color="black", lw=1.5)
        ax.plot(row["FIT_WAVE"], row["FIT_FLUX"], color="red", lw=2.2)
        for center, style in zip(row["FIT_CENTERS"], ("--", "-.")):
            ax.axvline(center, color="red", ls=style, lw=1.6)
        lo, hi = np.nanmin(lower), np.nanmax(upper)
        pad = 0.05 * (hi - lo)
        ax.set(xlim=(wave[0], wave[-1]), ylim=(lo - pad, hi + pad))
        ax.set_title(f'{row["MASS_LO"]:.1f}–{row["MASS_HI"]:.1f}   '
                     f'($N={row["N_STACK"]:,}$)')
    for ax in list(axes.flat)[len(rows):]:
        ax.set_visible(False)
    for ax in list(axes.flat)[len(rows) - ncols:len(rows)]:
        ax.tick_params(labelbottom=True)
        ax.set_xlabel(r"$\lambda$ [Å]")
    fig.suptitle(r"SDSS DR7  $\log_{10}(M_\ast/M_\odot)$")
    fig.supylabel(r"$f_\lambda$ [$10^{-17}$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$]")
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_individual_stacks(stacks, directory):
    directory.mkdir(parents=True, exist_ok=True)
    for row in stacks:
        output = directory / f'SDSS_DR7_SII_mass_{row["MASS_LO"]:.1f}_{row["MASS_HI"]:.1f}.png'
        wave, flux, error = (row[name][SII_WINDOW] for name in ("WAVE", "FLUX", "FLUX_ERR"))
        plot_single_stack(wave, flux, error, row["FIT_WAVE"], row["FIT_FLUX"],
                          row["FIT_CENTERS"], output, minor_tick_size=8, figsize=(12, 6), ylabel_size=27)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--individual-plots", action="store_true",
                        help="Render one PNG per mass bin from the saved stack FITS.")
    args = parser.parse_args()
    if args.individual_plots:
        plot_individual_stacks(Table.read(args.output.with_suffix(".fits")),
                               args.output.with_suffix(""))
        return
    paths = sorted(SPECTRA.glob("spSpec-*.fit"))
    paths, cat = matched_rows(paths, ("sm_MEDIAN", "SELECTED_V2"))
    mass, redshift = cat["sm_MEDIAN"], cat["Z"]
    fluxes = np.column_stack([cat[name] for name in LINES])
    errors = np.column_stack([cat[f"{name}_ERR"] for name in LINES])
    valid = (cat["SELECTED_V2"].astype(bool) & np.isfinite(mass) & np.isfinite(redshift)
             & (redshift > -1) & np.isfinite(fluxes).all(axis=1)
             & (np.isfinite(errors) & (errors > 0)).all(axis=1))
    print(f"Matched FITS: {len(paths):,}; valid catalog rows: {valid.sum():,}", flush=True)
    bin_id = np.full(len(mass), -1, dtype=int)
    bin_id[valid] = np.floor(mass[valid] / WIDTH).astype(int)
    rows = []
    skipped_small_bins = []
    for i in np.unique(bin_id[valid]):
        lo, hi = round(i * WIDTH, 1), round((i + 1) * WIDTH, 1)
        idx = np.flatnonzero(valid & (bin_id == i))
        if len(idx) < NMIN:
            skipped_small_bins.append((lo, hi, len(idx)))
            continue
        p2, p98 = np.percentile(fluxes[idx], [2, 98], axis=0)
        keep = idx[((fluxes[idx] >= p2) & (fluxes[idx] <= p98)).all(axis=1)]
        if len(keep) < NMIN:
            skipped_small_bins.append((lo, hi, len(keep)))
            continue

        print(f"Mass {lo:.1f}–{hi:.1f}: {len(idx):,} -> {len(keep):,} after flux cuts",
              flush=True)
        used, failures, stack, error, count = stack_bin(keep, paths, redshift)
        print(f"  Stacked {len(used):,}; FITS read errors {failures:,}; "
              f"missing line coverage {len(keep) - len(used) - failures:,}", flush=True)
        if len(used) < NMIN:
            skipped_small_bins.append((lo, hi, len(used)))
            continue
        model_wave, model_flux, centers, ratio = fit_doublet(
            WAVE_GRID, stack, error, return_ratio=True
        )
        catalog_sums = fluxes[used].sum(axis=0)
        rows.append(dict(MASS_LO=lo, MASS_HI=hi, MASS_MEDIAN=np.median(mass[used]),
                         N_CATALOG=len(idx), N_AFTER_CUT=len(keep), N_STACK=len(used),
                         N_READ_ERRORS=failures, P2=p2, P98=p98, R_FIT=ratio,
                         R_CATALOG=catalog_sums[0] / catalog_sums[1],
                         WAVE=WAVE_GRID, FLUX=stack, FLUX_ERR=error, N_SPEC=count,
                         FIT_WAVE=model_wave, FIT_FLUX=model_flux, FIT_CENTERS=centers))
    print(f"Bins below {NMIN} spectra: {skipped_small_bins}", flush=True)
    if not rows:
        raise RuntimeError("No mass bins qualified for stacking.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    stacks = Table(rows=rows)
    stacks.meta.update(BINWIDTH=WIDTH, NMIN=NMIN, PCTL_LO=2, PCTL_HI=98,
                       WAVESYS="rest-frame vacuum", FLUXUNIT="1e-17 erg/s/cm^2/Angstrom")
    fits_path = args.output.with_suffix(".fits")
    stacks.write(fits_path, overwrite=True)
    plot_stacks(rows, args.output)
    print(f"Saved {len(rows)} bins with {sum(row['N_STACK'] for row in rows):,} spectra")
    print(f"Saved {args.output} and {fits_path}")


if __name__ == "__main__":
    main()
