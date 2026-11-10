#!/usr/bin/env python3
"""Fixed 9.4 <= logM < 9.65: one local stack and two equal-number high-z redshift stacks.
Reuses v1 selection, flux cuts, fitting and Gaussian Monte Carlo uncertainties.
"""
from pathlib import Path
import argparse
import json
import emcee
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.container import ErrorbarContainer
from astropy.table import Table
import stacked_sii_ne_vs_mass_sfr_ssfr_sigma_sfr_v1 as base

plt.rcParams.update({
    "figure.figsize": (12, 6), "font.size": 32, "axes.labelsize": 32,
    "axes.grid": False, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": False, "ytick.right": False, "xtick.minor.visible": False, "ytick.minor.visible": False,
    "xtick.major.size": 16, "ytick.major.size": 16, "xtick.major.width": 2, "ytick.major.width": 2,
    "xtick.minor.visible": True, "ytick.minor.visible": True,
    "xtick.minor.size": 4, "ytick.minor.size": 4, "xtick.minor.width": 1.5, "ytick.minor.width": 1.5,
    "xtick.labelsize": 28, "ytick.labelsize": 28,
    "font.family": "STIXGeneral", "mathtext.fontset": "stix",
})


MASS_RANGE = (9.5, 9.9)
HIGHZ_N_BINS = 2
OUT = base.ROOT / 'results/csv' / Path(__file__).stem
FIG = base.ROOT / 'results/figure' / Path(__file__).stem

# 最初に言葉（変数）とカラーコードを結びつける
madder_red = '#A41F35'
deep_red   = '#A62B2B'
forest     = 'forestgreen'  # matplotlib標準のフォレストグリーン
crimson    = '#DC143C'

def symbol_legend(ax, fontsize):
    """Show only the data marker for measurements; preserve model line styles."""
    handles, labels = ax.get_legend_handles_labels()
    handles = [handle.lines[0] if isinstance(handle, ErrorbarContainer) else handle
               for handle in handles]
    ax.legend(handles, labels, fontsize=fontsize, loc='upper left', frameon=False,
              numpoints=1)


def fit_and_plot(result):
    """MCMC in log density with asymmetric Gaussian measurement errors.

    Uniform priors: -2 < log10(c / cm^-3) < 6, -10 < p < 10.
    Redshift bars describe sample ranges; medians are fixed covariates.
    The grey band is the pointwise 68% credible interval of the mean relation.
    """
    data = result[result.status == 'ok']
    if len(data) < 3:
        raise ValueError('At least three constrained measurements are required')
    z = data.z_median.to_numpy()
    x = np.log10(1 + z)
    y = data.log_ne.to_numpy()
    lower = (data.log_ne-data.log_ne_lo).to_numpy()
    upper = (data.log_ne_hi-data.log_ne).to_numpy()
    if np.any(lower <= 0) or np.any(upper <= 0):
        raise ValueError('Measurement errors must be positive')

    def log_prob(theta):
        a, power = theta
        if not (-2 < a < 6 and -10 < power < 10):
            return -np.inf
        residual = a + power*x - y
        sigma = np.where(residual < 0, lower, upper)
        # Split-normal normalization is constant with respect to theta.
        return -0.5*np.sum((residual/sigma)**2)

    rng = np.random.default_rng(base.SEED)
    power, a = np.polyfit(x, y, 1)
    sampler = emcee.EnsembleSampler(32, 2, log_prob)
    sampler.random_state = np.random.RandomState(base.SEED).get_state()
    sampler.run_mcmc(np.array([a, power]) + rng.normal(0, 0.01, (32, 2)), 12000, progress=False)
    burn = 3000
    tau = sampler.get_autocorr_time(discard=burn)
    if np.any((12000-burn) < 50*tau):
        raise RuntimeError('MCMC chain is too short relative to autocorrelation time')
    chain = sampler.get_chain(discard=burn, flat=True)
    np.savez_compressed(OUT/'mcmc_chain.npz', chain=sampler.get_chain(), burn_in=burn,
                        parameter_names=np.array(['log10_c', 'p']))
    summaries = {}
    for name, values in [('log10_c', chain[:, 0]), ('c', 10**chain[:, 0]), ('p', chain[:, 1])]:
        low, median, high = np.percentile(values, [16, 50, 84])
        summaries[name] = dict(median=float(median), lower_16=float(low), upper_84=float(high),
                               minus=float(median-low), plus=float(high-median))
    summaries['diagnostics'] = dict(walkers=32, steps=12000, burn_in=burn,
                                    autocorrelation_time=tau.tolist(),
                                    effective_sample_size=(len(chain)/tau).tolist(),
                                    mean_acceptance_fraction=float(sampler.acceptance_fraction.mean()))
    summaries['method'] = dict(model='log10(ne) = log10(c) + p*log10(1+z)',
        likelihood='split normal in log10(ne); independent measurements; fixed z medians',
        priors='uniform log10(c) in (-2, 6), p in (-10, 10)',
        band='pointwise 16th-84th percentiles of mean relation; no intrinsic scatter')
    (OUT/'mcmc_fit.json').write_text(json.dumps(summaries, indent=2)+'\n')
    grid = np.linspace(0, max(data.bin_hi)*1.04, 300)
    prediction = chain[::20, 0, None] + chain[::20, 1, None]*np.log10(1+grid)
    low, median, high = np.percentile(prediction, [16, 50, 84], axis=0)
    pd.DataFrame(dict(z=grid, log_ne_p16=low, log_ne_median=median, log_ne_p84=high)).to_csv(OUT/'mcmc_curve.csv', index=False)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.fill_between(grid, low, high, color='0.8', alpha=0.7, linewidth=0)
    # A single curve evaluated at the posterior median parameters.
    ax.plot(grid, summaries['log10_c']['median']+summaries['p']['median']*np.log10(1+grid),
            color='black', lw=1.6,
            label=fr'$n_{{\rm e}}={summaries["c"]["median"]:.1f}(1+z)^{{{summaries["p"]["median"]:.2f}}}$')
    groups = [
        (data[data['sample'] == 'local'], 'o', 'black', 'SDSS'),
        (data[(data['sample'] == 'high-z') & (data.bin_index == 1)], '*', forest, 'JADES (low-z)'),
        (data[(data['sample'] == 'high-z') & (data.bin_index == 2)], '*', crimson, 'JADES (high-z)'),
    ]
    for sub, marker, color, label in groups:
        ax.errorbar(sub.z_median, sub.log_ne,
                    xerr=np.vstack([sub.z_median-sub.bin_lo, sub.bin_hi-sub.z_median]),
                    yerr=np.vstack([sub.log_ne-sub.log_ne_lo, sub.log_ne_hi-sub.log_ne]),
                    fmt=marker, color=color, capsize=13, capthick=2, elinewidth=1.8, ms=15 if marker == 'o' else 21, label=label, zorder=3)
    ax.set_xlabel(r'$z$')
    ax.set_ylabel(r'$\log(n_{\rm e}/{\rm cm}^{-3})$')
    ax.set_xlim(0, max(data.bin_hi)*1.04)
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    ax.tick_params(direction='in', top=True, right=True)
    symbol_legend(ax, fontsize=12)
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG.with_suffix('.png'), dpi=200)
    fig.savefig(FIG.with_suffix('.pdf'))
    # Companion figure for comparing redshift exponents at fixed normalization.
    for exponent, style in [(1, '--'), (2, '-.'), (3, ':')]:
        ax.plot(grid, summaries['log10_c']['median'] + exponent*np.log10(1+grid),
                color='black', ls=style, lw=1.5, label='_nolegend_', zorder=1)
        label_z = grid[-1]*0.86
        label_y = summaries['log10_c']['median'] + exponent*np.log10(1+label_z)
        ax.annotate(fr'$p={exponent}$', xy=(label_z, label_y),
                    xytext=(0, 8), textcoords='offset points', fontsize=16,
                    ha='center', va='bottom',
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=1))
    symbol_legend(ax, fontsize=16)
    fig.tight_layout()
    comparison = FIG.with_name(FIG.name + '_power_comparison')
    fig.savefig(comparison.with_suffix('.png'), dpi=200)
    fig.savefig(comparison.with_suffix('.pdf'))
    cfit, pfit = summaries['c'], summaries['p']
    mass_low, mass_high = float(data.mass_lo.iloc[0]), float(data.mass_hi.iloc[0])
    caption = (
        r'\caption{Electron density as a function of redshift at fixed stellar mass '
        + fr'(${mass_low:g} \leq \log(M_*/M_\odot) < {mass_high:g}$). '
        + r'The black circle shows the stacked SDSS measurement; the green and red stars show '
        + r'the lower- and higher-redshift JADES measurements, respectively. '
        + r'Horizontal bars span the redshift ranges of the contributing galaxies; '
        + r'vertical bars show the 16th--84th percentile measurement intervals. '
        + r'The solid black line shows the relation evaluated at the MCMC posterior median parameters, '
        + fr'$n_{{\rm e}} = ({cfit["median"]:.1f}^{{+{cfit["plus"]:.1f}}}_{{-{cfit["minus"]:.1f}}})'
        + fr'\,(1+z)^{{{pfit["median"]:.3f}^{{+{pfit["plus"]:.3f}}}_{{-{pfit["minus"]:.3f}}}}}\,{{\rm cm}}^{{-3}}$. '
        + r'The gray band indicates the pointwise 68\% credible interval of the mean relation. '
        + r'The dashed, dash-dotted, and dotted black lines show $n_{\rm e} \propto (1+z)^p$ '
        + r'with $p=1$, 2, and 3, respectively, normalized to the same posterior median $c$ as the solid curve.}'
    )
    (OUT/'power_comparison_caption.tex').write_text(caption + '\n')
    plt.close(fig)
    print(f'Fit results saved to {OUT}/mcmc_fit.json and {OUT}/mcmc_curve.csv')
    print(f'Figure saved to {FIG}.png and {FIG}.pdf')
    print(json.dumps(summaries, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local-n-mc', type=int, default=base.LOCAL_N_MC)
    parser.add_argument('--jades-n-mc', type=int, default=base.JADES_N_MC)
    parser.add_argument('--fit-only', action='store_true', help='Fit and plot existing density_results.csv')
    args = parser.parse_args()
    if args.fit_only:
        fit_and_plot(pd.read_csv(OUT/'density_results.csv'))
        return
    if min(args.local_n_mc, args.jades_n_mc) < 2:
        parser.error('MC counts must be at least 2')
    OUT.mkdir(parents=True, exist_ok=True)
    base.STACK_DIR = OUT / 'jades_stacks'
    base.STACK_DIR.mkdir(exist_ok=True)
    rng = np.random.default_rng(base.SEED)
    d = base.load_local()
    z = base.column(Table.read(base.LOCAL_PATH, hdu=1), 'Z')
    idx = np.flatnonzero(d['use'] & base.in_range(d['mass'], MASS_RANGE) & np.isfinite(z))
    before = len(idx)
    keep = np.ones(before, bool)
    for name in ('f1', 'f2'):
        low, high = np.percentile(d[name][idx], base.FLUX_PERCENTILES)
        keep &= (d[name][idx] >= low) & (d[name][idx] <= high)
    idx = idx[keep]
    if len(idx) < base.NMIN:
        raise ValueError('Insufficient local sample')
    draws = base.ratio_draws(d['f1'][idx], d['e1'][idx], d['f2'][idx], d['e2'][idx], args.local_n_mc, rng)
    row = dict(sample='local', N=len(idx), N_before_cut=before,
               z_median=np.median(z[idx]), bin_lo=z[idx].min(), bin_hi=z[idx].max(),
               n_mc=args.local_n_mc, Te=base.TE)
    row.update(base.density_summary(np.nanmedian(draws), draws))
    rows = [row]
    pd.DataFrame(dict(catalog_row=idx, mass=d['mass'][idx], z=z[idx])).to_csv(OUT/'local_members.csv', index=False)
    items = sorted([it for it in base.load_jades() if base.in_range(it['mass'], MASS_RANGE)
                    and np.isfinite(it['z']) and 0.5 <= it['z'] < 8], key=lambda it: (it['z'], it['id']))
    if len(items) < HIGHZ_N_BINS:
        raise ValueError('Insufficient high-z sample')
    members = []
    for number, indices in enumerate(np.array_split(np.arange(len(items)), HIGHZ_N_BINS), 1):
        selected = [items[i] for i in indices]
        low, high = selected[0]['z'], np.nextafter(selected[-1]['z'], np.inf)
        # A separate key gives each stack a unique file name. Membership is fixed by rank,
        # including when identical redshifts occur across the split.
        key = f'z_bin{number}'
        for it in selected:
            it[key] = it['z']
        results, used = base.stack_highz(selected, key, dict(highz_bins=[(low, high)]), args.jades_n_mc, rng)
        results[0].update(bin_index=number, binning='equal_number_redshift', parameter='z')
        rows.extend(results)
        for member in used:
            member.update(bin_index=number, binning='equal_number_redshift', parameter='z')
        members.extend(used)
    result = pd.DataFrame(rows)
    result['mass_lo'], result['mass_hi'] = MASS_RANGE
    for name in ('ne', 'ne_lo', 'ne_hi'):
        result['log_'+name] = np.log10(result[name])
    result.to_csv(OUT/'density_results.csv', index=False)
    pd.DataFrame(members).to_csv(OUT/'jades_members.csv', index=False)
    fit_and_plot(result)
    print(result[['sample', 'N', 'z_median', 'log_ne', 'log_ne_lo', 'log_ne_hi', 'status']].to_string(index=False))


if __name__ == '__main__':
    main()
