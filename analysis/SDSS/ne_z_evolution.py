#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""電子密度の赤方偏移進化 n_e = c (1+z)^p を MCMC で推定する。

入力は下記の固定値。log n_e の非対称誤差は上下の平均を用いる。
図は results/figure, パラメータの 16/50/84 パーセンタイルは results/csv に保存。

実行: .venv/bin/python electron_density_src/analysis/SDSS/ne_z_evolution.py [--no-show]
"""
from pathlib import Path
import argparse
import csv

import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner

plt.rcParams.update({
    'figure.figsize': (6, 6), 'font.size': 32, 'axes.labelsize': 32,
    'axes.titlesize': 32, 'axes.grid': False,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True,
    'xtick.major.size': 32, 'ytick.major.size': 32,
    'xtick.major.width': 2, 'ytick.major.width': 2,
    'xtick.minor.visible': True, 'ytick.minor.visible': True,
    'xtick.minor.size': 8, 'ytick.minor.size': 8,
    'xtick.minor.width': 1.5, 'ytick.minor.width': 1.5,
    'xtick.labelsize': 28, 'ytick.labelsize': 28,
    'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix',
})


# =====================================================================
# 設定
# =====================================================================
ROOT = Path(__file__).resolve().parents[3]
CSV_DIR = ROOT / 'results/csv'
FIG_DIR = ROOT / 'results/figure'
OUT = dict(summary_csv=CSV_DIR / 'ne_z_evolution.csv',
           corner_png=FIG_DIR / 'ne_z_evolution_corner.png',
           ne_png=FIG_DIR / 'ne_z_evolution.png')

Z = np.array([0.1, 2.117, 4.351])
LOGNE = np.array([1.64565992, 2.31153, 2.68482])
LOGNE_ERR = np.array([0.00951, (0.06200 + 0.06167) / 2,
                     (0.05707 + 0.05937) / 2])
POINT_COLORS = ('black', 'seagreen', 'crimson')
SEED = 42
N_WALKERS = 32
N_STEPS = 10000
BURN_IN = 2000
INITIAL = np.array([1.6, 1.4])  # log10(c), p
PRIOR_BOUNDS = ((-2, 4), (0, 5))
Z_FIT = np.linspace(0, 5, 500)


# =====================================================================
# モデルと MCMC
# =====================================================================
def model(theta, z):
    """log10(n_e / cm^-3) = log10(c) + p log10(1+z)。"""
    logc, p = theta
    return logc + p * np.log10(1 + z)


def log_prior(theta):
    return 0.0 if all(lo < v < hi for v, (lo, hi) in
                      zip(theta, PRIOR_BOUNDS)) else -np.inf


def log_likelihood(theta):
    return -0.5 * np.sum(((LOGNE - model(theta, Z)) / LOGNE_ERR) ** 2)


def log_probability(theta):
    lp = log_prior(theta)
    return lp + log_likelihood(theta) if np.isfinite(lp) else -np.inf


def run_mcmc(n_steps, burn_in, seed, progress=True):
    """固定 seed で初期位置と emcee の乱数を再現可能にする。"""
    rng = np.random.default_rng(seed)
    pos = INITIAL + 1e-4 * rng.standard_normal((N_WALKERS, len(INITIAL)))
    sampler = emcee.EnsembleSampler(N_WALKERS, len(INITIAL), log_probability)
    sampler.random_state = np.random.RandomState(seed).get_state()
    sampler.run_mcmc(pos, n_steps, progress=progress)
    print(f'[MCMC] mean acceptance fraction = {sampler.acceptance_fraction.mean():.3f}')
    return sampler.get_chain(discard=burn_in, flat=True)


def summarize(samples, n_steps, burn_in, seed):
    """元のパラメータと c のパーセンタイルを保存する。"""
    rows = []
    print('\n===== MCMC Result =====')
    for name, values in [('log10(c)', samples[:, 0]), ('p', samples[:, 1]),
                         ('c', 10 ** samples[:, 0])]:
        lo, med, hi = np.percentile(values, [16, 50, 84])
        rows.append(dict(parameter=name, p16=lo, median=med, p84=hi,
                         err_lo=med - lo, err_hi=hi - med,
                         n_walkers=N_WALKERS, n_steps=n_steps, burn_in=burn_in, seed=seed))
        print(f'{name} = {med:.4f} (+{hi - med:.4f}/-{med - lo:.4f})')
    path = OUT['summary_csv']
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print('Saved CSV:', path)


# =====================================================================
# 作図
# =====================================================================
def finish(fig, path, show=True):
    for ax in fig.axes:
        for spine in ax.spines.values():
            spine.set_linewidth(2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    print('Saved figure:', path)
    if show:
        plt.show()
    plt.close(fig)


def credible_band(samples, zfit):
    """全事後サンプルから区間を計算。赤方偏移を分割してメモリを抑える。"""
    band = np.empty((3, len(zfit)))
    for start in range(0, len(zfit), 16):
        zz = zfit[start:start + 16]
        curves = samples[:, 0, None] + samples[:, 1, None] * np.log10(1 + zz)
        band[:, start:start + len(zz)] = np.percentile(curves, [16, 50, 84], axis=0)
    return band


def plot_results(samples, show=True):
    with plt.rc_context({'font.size': 16, 'axes.labelsize': 22,
                         'axes.titlesize': 16, 'xtick.labelsize': 16,
                         'ytick.labelsize': 16, 'xtick.major.size': 8,
                         'ytick.major.size': 8}):
        fig = corner.corner(samples, labels=[r'$\log_{10}(c)$', r'$p$'],
                            show_titles=True, quantiles=[0.16, 0.5, 0.84])
        finish(fig, OUT['corner_png'], show)

    lower, median, upper = credible_band(samples, Z_FIT)
    fig, ax = plt.subplots(figsize=(6, 6))
    for z, ne, err, color in zip(Z, LOGNE, LOGNE_ERR, POINT_COLORS):
        ax.errorbar(z, ne, yerr=err, fmt='s', color=color, mfc=color,
                    mec=color, ecolor=color, ms=12, mew=1,capsize=10)
    ax.plot(Z_FIT, median, color='black', lw=2, label='MCMC median')
    ax.fill_between(Z_FIT, lower, upper, color='gray', alpha=0.5,
                    label='68% credible interval')
    logc_med = np.median(samples[:, 0])
    ax.set_xlim(0.0, 5)
    ax.set_xlabel(r'$z$')
    ax.set_ylabel(r'$\log(n_{\mathrm{e}}\,/\,\mathrm{cm}^{-3})$')
    finish(fig, OUT['ne_png'], show)


# =====================================================================
# main
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n-steps', type=int, default=N_STEPS)
    parser.add_argument('--burn-in', type=int, default=BURN_IN)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--no-show', action='store_true', help='図を表示せず保存する')
    parser.add_argument('--no-progress', action='store_true', help='進捗表示を省略する')
    args = parser.parse_args()
    if args.n_steps < 2 or not 0 <= args.burn_in < args.n_steps:
        parser.error('--n-steps >= 2 かつ 0 <= --burn-in < --n-steps が必要です')
    if not 0 <= args.seed < 2 ** 32:
        parser.error('--seed は 0 以上 2**32 未満にしてください')
    if args.no_show:
        plt.switch_backend('Agg')
    samples = run_mcmc(args.n_steps, args.burn_in, args.seed, not args.no_progress)
    summarize(samples, args.n_steps, args.burn_in, args.seed)
    plot_results(samples, show=not args.no_show)


if __name__ == '__main__':
    main()
