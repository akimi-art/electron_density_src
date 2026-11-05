#!/usr/bin/env python3
"""SDSS M*–SFR maps: trimmed mean of individual SII ratios and ne(mean ratio).

Run from any directory with the project's .venv Python. Each cell applies the
existing SDSS analysis's joint 2–98 percentile cuts on the two SII fluxes,
BEFORE averaging individual ratios. This is not a color-scale percentile cut.
Density is inferred from the mean ratio, not the mean of individual densities.
No additional SII S/N cut or individual physical-ratio cut is imposed: those
would change the selected sample and bias ratios near the low-density limit.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
from astropy.table import Table
import pyneb as pn

ROOT = Path(__file__).resolve().parents[2]


def make_maps(path, width=0.2, nmin=30, percentiles=(2., 98.), temperature=1e4):
    tab = Table.read(path, hdu=1)
    selected = np.asarray(tab['SELECTED_V2'], bool)
    def col(name):
        return np.ma.filled(np.ma.asarray(tab[name], dtype=float), np.nan)
    mass, sfr = col('sm_MEDIAN'), col('sfr_MEDIAN')
    f1, f2 = col('SII_6717_FLUX'), col('SII_6731_FLUX')
    e1, e2 = col('SII_6717_FLUX_ERR'), col('SII_6731_FLUX_ERR')
    valid = selected & np.isfinite(mass) & np.isfinite(sfr) & (mass > -90) & (sfr > -90)
    for a in (f1, f2, e1, e2):
        valid &= np.isfinite(a) & (a > 0)
    xe = np.linspace(7, 12, int(round(5 / width)) + 1)
    ye = np.linspace(-3, 3, int(round(6 / width)) + 1)
    atom = pn.Atom('S', 2)
    # Use a monotonic density interval; never clamp nonphysical ratios.
    rlo = float(atom.getEmissivity(tem=temperature, den=1e5, wave=6716) /
                atom.getEmissivity(tem=temperature, den=1e5, wave=6731))
    rhi = float(atom.getEmissivity(tem=temperature, den=1e-2, wave=6716) /
                atom.getEmissivity(tem=temperature, den=1e-2, wave=6731))
    rows = []
    for i, (ml, mh) in enumerate(zip(xe[:-1], xe[1:])):
        for j, (sl, sh) in enumerate(zip(ye[:-1], ye[1:])):
            idx = np.flatnonzero(valid & (mass >= ml) & ((mass < mh) if i < len(xe)-2 else (mass <= mh)) &
                                 (sfr >= sl) & ((sfr < sh) if j < len(ye)-2 else (sfr <= sh)))
            row = dict(mass_lo=ml, mass_hi=mh, sfr_lo=sl, sfr_hi=sh,
                       n_before=len(idx), n_used=0, mean_ratio=np.nan, ne_cm3=np.nan,
                       status='insufficient_sample')
            if len(idx) >= nmin:
                keep = np.ones(len(idx), bool)
                for name, flux in (('6717', f1), ('6731', f2)):
                    low, high = np.percentile(flux[idx], percentiles)
                    row[f'flux_{name}_lo'], row[f'flux_{name}_hi'] = low, high
                    keep &= (flux[idx] >= low) & (flux[idx] <= high)
                idx = idx[keep]
                row['n_used'] = len(idx)
                if len(idx) >= nmin:
                    ratio = float(np.mean(f1[idx] / f2[idx]))
                    row.update(mean_ratio=ratio, status='ratio_outside_density_range')
                    if rlo < ratio < rhi:
                        ne = float(atom.getTemDen(int_ratio=ratio, tem=temperature, wave1=6716,
                                                 wave2=6731, maxError=1e-6,
                                                 start_x=-2, end_x=5))
                        if np.isfinite(ne) and ne > 0:
                            row.update(ne_cm3=ne, status='ok')
            rows.append(row)
    print(f'Master: {len(tab):,}; SELECTED_V2: {selected.sum():,}; valid: {valid.sum():,}')
    return pd.DataFrame(rows), xe, ye


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fits', type=Path, default=ROOT / 'results/fits/mpajhu_dr7_v5_2_master_v2_Ka03_Reu.fits')
    p.add_argument('--bin-width', type=float, default=0.2)
    p.add_argument('--min-count', type=int, default=30)
    p.add_argument('--percentiles', type=float, nargs=2, default=(2, 98))
    p.add_argument('--temperature', type=float, default=1e4)
    args = p.parse_args()
    if args.bin_width <= 0 or any(abs(v / args.bin_width - round(v / args.bin_width)) > 1e-7 for v in (5, 6)):
        p.error('bin-width must divide both 5 and 6 dex')
    if not 0 <= args.percentiles[0] < args.percentiles[1] <= 100 or args.min_count < 1 or args.temperature <= 0:
        p.error('invalid percentiles, min-count, or temperature')
    data, xe, ye = make_maps(args.fits, args.bin_width, args.min_count, args.percentiles, args.temperature)
    stem = 'heat_sm_sfr_sii_ratio_ne_sdss_mean'
    out = ROOT / 'results/figure' / f'{stem}.png'
    csv = ROOT / 'results/csv' / f'{stem}.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    csv.parent.mkdir(parents=True, exist_ok=True)
    data['trim_percentile_lo'], data['trim_percentile_hi'] = args.percentiles
    data['temperature_K'] = args.temperature
    data['input_fits'] = str(args.fits.resolve())
    data.to_csv(csv, index=False)
    plt.rcParams.update({'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix',
                         'font.size': 15, 'axes.labelsize': 19, 'xtick.direction': 'in',
                         'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True})
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True, layout='constrained')
    shape = (len(xe)-1, len(ye)-1)
    ratio = data.mean_ratio.to_numpy().reshape(shape)
    density = data.ne_cm3.to_numpy().reshape(shape)
    im = axes[0].pcolormesh(xe, ye, ratio.T, cmap='viridis', vmin=1.0, vmax=1.5, shading='flat')
    fig.colorbar(im, ax=axes[0], extend='both', label=r'Mean [S II] $\lambda6717/\lambda6731$')
    finite = density[np.isfinite(density)]
    if not len(finite):
        raise ValueError('No cells have a valid density')
    im = axes[1].pcolormesh(xe, ye, density.T, cmap='magma',
                          norm=LogNorm(vmin=10, vmax=1000), shading='flat')
    fig.colorbar(im, ax=axes[1], extend='both', label=r'$n_e(\langle R_{\rm SII}\rangle)$ [cm$^{-3}$]')
    axes[0].set_title('Mean S II ratio')
    axes[1].set_title(rf'Electron density ($T_e={args.temperature:,.0f}$ K)')
    for ax in axes:
        ax.set(xlim=(7, 12), ylim=(-3, 3), xlabel=r'$\log(M_*/M_\odot)$')
        ax.minorticks_on()
    axes[0].set_ylabel(r'$\log(\mathrm{SFR}/[M_\odot\,\mathrm{yr}^{-1}])$')
    fig.suptitle(f'SDSS selected sample | {args.bin_width:g} dex bins | '
                 f'flux trim {args.percentiles[0]:g}–{args.percentiles[1]:g}% per cell | '
                 f'N after trim ≥ {args.min_count}', fontsize=15)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f'Ratio cells: {np.isfinite(ratio).sum()}; density cells: {len(finite)}')
    print(f'Saved: {out}\nSaved: {csv}')


if __name__ == '__main__':
    main()
