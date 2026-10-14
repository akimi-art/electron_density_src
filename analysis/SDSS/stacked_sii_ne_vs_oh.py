#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SDSS の 12+log(O/H) ビンごとに [SII] flux をスタックして ne を描く。

SFR v7 と同じく、各線の flux を平均で
集計してから比を取る。各ビンで両線の flux の 2–98% 範囲を共通選別。
誤差は固定サンプルの独立 Gaussian flux MC (bootstrap ではない)。
oh_MEDIAN でビン分けし、oh_P16/P84 は代表的な個別測定区間の表示に
使用する。金属量の誤差によるビン間移動は伝播しない。
local は Hα 規格化・追加の質量/SFR 選別を使用しない。
JADES は BASE_SAMPLE + 有効な N26_logOH の同一銀河群を1スタックにする。
横棒は JADES 金属量の16–84%分布であり、測定誤差ではない。

実行: python electron_density_src/analysis/SDSS/stacked_sii_ne_vs_oh.py
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.table import Table
import pyneb as pn
from scipy.stats import binned_statistic_2d
from scipy.optimize import curve_fit
from astropy.io import fits

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u

plt.rcParams.update({
    "figure.figsize": (12, 6), "font.size": 32, "axes.labelsize": 32, "axes.titlesize": 32,
    "axes.grid": False, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "xtick.major.size": 32, "ytick.major.size": 32, "xtick.major.width": 2, "ytick.major.width": 2,
    "xtick.minor.visible": True, "ytick.minor.visible": True,
    "xtick.minor.size": 8, "ytick.minor.size": 8, "xtick.minor.width": 1.5, "ytick.minor.width": 1.5,
    "xtick.labelsize": 28, "ytick.labelsize": 28,
    "font.family": "STIXGeneral", "mathtext.fontset": "stix",
})

ROOT = Path(__file__).resolve().parents[3]
FITS_PATH = ROOT / 'results/fits/sdss_sample_v2_zlt0.1879_Lgt1e+39_Ka03_Reu.fits'
OUT_CSV = ROOT / 'results/csv/stacked_sii_ne_vs_oh.csv'
OUT_PNG = ROOT / 'results/figure/stacked_sii_ne_vs_oh.png'
OUT_MASS_OH_PNG = ROOT / 'results/figure/stellar_mass_vs_metallicity_sfr_sdss.png'
FLAGS_PATH = ROOT / 'results/fits/mpajhu_dr7_v5_2_master_v2_Ka03_Reu.fits'   # 全行 + フラグ
OUT_COMPL_CSV = ROOT / 'results/csv/completeness_vs_oh.csv'
OUT_COMPL_PNG = ROOT / 'results/figure/completeness_vs_oh.png'
JADES_PATH = ROOT / 'results/JADES/sample/jades_all_with_flags.fits'
JADES_OH_COLUMN = 'N26_logOH'  # 既存推定値。R23からの再推定ではない。
JADES_N_MC = 500
OUT_JADES_CSV = ROOT / 'results/csv/stacked_sii_ne_vs_oh_jades.csv'
OUT_JADES_STACK = ROOT / 'results/figure/stacked_sii_ne_vs_oh_jades_stack.png'
OUT_JADES_OH_HIST = ROOT / 'results/figure/jades_metallicity_histogram.png'
JADES_OH_HIST_BIN_WIDTH = 0.1
BIN_WIDTH = 0.1
NMIN = 100
N_MC = 5000
SEED = 42
TE = 1e4
FLUX_PERCENTILES = (2, 98)
MASS_PLOT_BIN_WIDTH = 0.1
OH_PLOT_BIN_WIDTH = 0.05
MASS_OH_PLOT_NMIN = 5


def column(tab, name):
    return np.ma.filled(np.ma.asarray(tab[name], dtype=float), np.nan)


def ratio_draws(f1, e1, f2, e2, n_mc, rng):
    """メモリを抑えるため MC を64回ずつ生成。負の flux も保持する。"""
    out = np.full(n_mc, np.nan)
    for start in range(0, n_mc, 64):
        stop = min(start + 64, n_mc)
        a = rng.normal(f1[:, None], e1[:, None], (len(f1), stop-start))
        b = rng.normal(f2[:, None], e2[:, None], (len(f2), stop-start))
        num, den = a.mean(axis=0), b.mean(axis=0)
        valid = (num > 0) & (den > 0)
        values = np.full(stop-start, np.nan)
        values[valid] = num[valid] / den[valid]
        out[start:stop] = values
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n-mc', type=int, default=N_MC)
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()
    if args.n_mc < 2:
        parser.error('--n-mc must be at least 2')
    tab = Table.read(FITS_PATH, hdu=1)
    # 選択後のサンプルかどうかの確認
    assert 'SELECTED_V2' in tab.colnames, 'SELECTED_V2 列がありません（選択後のファイルではない可能性）'
    assert np.all(np.asarray(tab['SELECTED_V2'], bool)), '選択されていない行が含まれています'
    print(f'Selected sample confirmed: {len(tab)} rows (all SELECTED_V2)')
    oh, oh16, oh84 = [column(tab, k) for k in ('oh_MEDIAN', 'oh_P16', 'oh_P84')]
    f1, f2, e1, e2 = [column(tab, k) for k in (
        'SII_6717_FLUX', 'SII_6731_FLUX',
        'SII_6717_FLUX_ERR', 'SII_6731_FLUX_ERR')]
    # -99.9 などの欠損 sentinel と順序が不正な金属量区間を除外。
    valid = (np.isfinite(oh) & np.isfinite(oh16) & np.isfinite(oh84)
             & (oh16 > 0) & (oh16 <= oh) & (oh <= oh84)
             & np.isfinite(f1) & np.isfinite(f2)
             & np.isfinite(e1) & np.isfinite(e2) & (e1 > 0) & (e2 > 0))
    if not valid.any():
        raise ValueError('No valid metallicity and [SII] measurements')
    print(f'Valid measurements: {valid.sum()}/{len(tab)}')
    edges = np.arange(np.floor(oh[valid].min()/BIN_WIDTH)*BIN_WIDTH,
                      np.floor(oh[valid].max()/BIN_WIDTH)*BIN_WIDTH+1.5*BIN_WIDTH,
                      BIN_WIDTH)
    # ---- 光度カットの completeness（O/H のビンごと）----
    #   分母 = PARENT_V2、分子 = PARENT_V2 & LUM_OK_V2（build_sample.py 11c と同じ定義）
    #   O/H が有効な行だけで数える（O/H の有効条件は上の valid と同じ）
    flags = Table.read(FLAGS_PATH, hdu=1)
    p_oh, p_oh16, p_oh84 = [column(flags, k) for k in ('oh_MEDIAN', 'oh_P16', 'oh_P84')]
    parent = np.asarray(flags['PARENT_V2'], bool)
    lum = np.asarray(flags['LUM_OK_V2'], bool)
    oh_ok = (np.isfinite(p_oh) & np.isfinite(p_oh16) & np.isfinite(p_oh84)
             & (p_oh16 > 0) & (p_oh16 <= p_oh) & (p_oh <= p_oh84))
    print(f'Parent: {parent.sum()}  うち O/H 有効: {(parent & oh_ok).sum()}')
    n_den = np.histogram(p_oh[parent & oh_ok], edges)[0]
    n_num = np.histogram(p_oh[parent & oh_ok & lum], edges)[0]
    with np.errstate(invalid='ignore', divide='ignore'):
        frac = n_num / n_den
        ferr = np.sqrt(frac * (1 - frac) / n_den)
    centers = 0.5 * (edges[1:] + edges[:-1])
    OUT_COMPL_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_COMPL_PNG.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'oh_center': centers, 'f': frac, 'f_err': ferr,
                  'N_den': n_den, 'N_num': n_num}).to_csv(OUT_COMPL_CSV, index=False)
    v = n_den >= 30
    figc, axc = plt.subplots(figsize=(12, 6))
    axc.errorbar(centers[v], frac[v], yerr=ferr[v], fmt='o-', color='k', ms=5, lw=1.5)
    axc.set_ylim(0, 1.05)
    axc.set_xlabel(r'$12+\log_{10}(\mathrm{O/H})$')
    axc.set_ylabel(r'$N(L_{6731} > L_{\rm min})\,/\,N$')
    for sp in axc.spines.values():
        sp.set_linewidth(2)
    figc.tight_layout()
    figc.savefig(OUT_COMPL_PNG, dpi=200)
    plt.show()
    print('Saved:', OUT_COMPL_CSV, OUT_COMPL_PNG)
    atom = pn.Atom('S', 2)
    # 有効密度範囲外の線比を黙って捨て、条件付きの区間を作らない。
    r_max = float(atom.getEmissivity(tem=TE, den=1e-3, wave=6716)
                  / atom.getEmissivity(tem=TE, den=1e-3, wave=6731))
    r_min = float(atom.getEmissivity(tem=TE, den=1e8, wave=6716)
                  / atom.getEmissivity(tem=TE, den=1e8, wave=6731))

    def density(r):
        if not np.isfinite(r) or not r_min < r < r_max:
            return np.nan
        return float(atom.getTemDen(int_ratio=r, tem=TE, wave1=6716, wave2=6731,
                                    maxError=1e-6))

    rng = np.random.default_rng(SEED)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        idx = np.flatnonzero(valid & (oh >= lo) & (oh < hi))
        before = len(idx)
        if before < NMIN:
            continue
        good = np.ones(before, dtype=bool)
        for flux in (f1, f2):
            lower, upper = np.percentile(flux[idx], FLUX_PERCENTILES)
            good &= (flux[idx] >= lower) & (flux[idx] <= upper)
        idx = idx[good]
        if len(idx) < NMIN:
            continue
        x = np.median(oh[idx])
        row = dict(oh_lo=lo, oh_hi=hi, oh_median=x,
                   oh_p16_median=np.median(oh16[idx]),
                   oh_p84_median=np.median(oh84[idx]),
                   N_before_cut=before, N=len(idx), n_mc=args.n_mc, Te=TE)
        ratios = ratio_draws(f1[idx], e1[idx], f2[idx], e2[idx], args.n_mc, rng)
        if not np.isfinite(ratios).any():
            r16 = r50 = r84 = np.nan
        else:
            r16, r50, r84 = np.nanpercentile(ratios, [16, 50, 84])
        ne, low, high = density(r50), density(r84), density(r16)
        # R→ne は単調減少なので区間の上下を反転する。
        status = ('ok' if np.all(np.isfinite([ne, low, high])) else 'unconstrained_interval')
        row.update({'R_mean': r50, 'R_mean_p16': r16,
                    'R_mean_p84': r84, 'ne_mean': ne,
                    'ne_mean_p16': low, 'ne_mean_p84': high,
                    'status_mean': status,
                    'valid_ratio_fraction_mean': np.isfinite(ratios).mean(),
                    'physical_ratio_fraction_mean': np.mean(
                        (ratios > r_min) & (ratios < r_max))})
        rows.append(row)
        print(f'O/H {lo:.1f}–{hi:.1f}: {before} -> {len(idx)} galaxies')
    if not rows:
        raise ValueError('No bins meet NMIN after flux cuts')
    res = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT_CSV, index=False)
    plt.rcParams.update({'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix',
                         'font.size': 20, 'axes.labelsize': 24,
                         'xtick.direction': 'in', 'ytick.direction': 'in',
                         'xtick.top': True, 'ytick.right': True})
    fig, ax = plt.subplots(figsize=(8, 6))
    x = res['oh_median'].to_numpy()
    xerr = np.vstack([x-res['oh_p16_median'], res['oh_p84_median']-x])
    ok = res['status_mean'].eq('ok').to_numpy()
    # 密度と区間の両端を常用対数に変換し、線形軸に描画する。
    y = np.log10(res['ne_mean'].to_numpy())
    yerr = np.vstack([y-np.log10(res['ne_mean_p16']),
                      np.log10(res['ne_mean_p84'])-y])
    ax.errorbar(x[ok], y[ok], xerr=xerr[:, ok], yerr=yerr[:, ok],
                fmt='s', color='k', capsize=3, label='Mean')
    if (~ok).any():
        print(f'Mean: {(~ok).sum()} unconstrained bins omitted from plot; see CSV')
    ax.set_xlabel(r'$12+\log_{10}(\mathrm{O/H})$')
    ax.set_ylabel(r'$\log_{10}(n_e\,/\,\mathrm{cm}^{-3})$')
    # local の表示範囲に加え、JADES の点と区間が入るように拡張する。
    jades = calculate_jades_point()
    if jades['status'] == 'ok':
        jx = jades['oh_median']
        jy = np.log10(jades['ne'])
        jlo, jhi = np.log10(jades['ne_lo']), np.log10(jades['ne_hi'])
        ax.errorbar(jx, jy,
                    xerr=[[jx-jades['oh_p16']], [jades['oh_p84']-jx]],
                    yerr=[[jy-jlo], [jhi-jy]], fmt='*', ms=14,
                    color='crimson', capsize=4, label=f"JADES (N={jades['N']})")
        ax.set_xlim(min(8.5, jades['oh_p16']-0.1), max(9.25, jades['oh_p84']+0.1))
        ax.set_ylim(min(1.25, jlo-0.1), max(3, jhi+0.1))
    else:
        print('JADES interval is unconstrained; see JADES CSV (no misleading error bar drawn)')
        ax.set_xlim(8.5, 9.25)
        ax.set_ylim(1.25, 3)
    ax.legend(fontsize=12)
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=200)
    print('Saved:', OUT_CSV, OUT_PNG)
    if args.show:
        plt.show()
    plt.close(fig)
    plot_mass_metallicity(tab, show=args.show)


def plot_mass_metallicity(tab, show=False):
    """質量–金属量の2次元ビンを、各ビンの log SFR 中央値で色付け。"""
    mass = column(tab, 'sm_MEDIAN')
    metallicity = column(tab, 'oh_MEDIAN')
    sfr = column(tab, 'sfr_MEDIAN')
    # カタログの欠損値 (-99.9 等) を除外。flux のビン内カットは適用しない。
    valid = (np.isfinite(mass) & np.isfinite(metallicity) & np.isfinite(sfr)
             & (mass > 0) & (metallicity > 0) & (sfr > -90))
    if not valid.any():
        raise ValueError('No valid stellar mass, metallicity and SFR measurements')
    mass_edges = np.arange(
        np.floor(mass[valid].min()/MASS_PLOT_BIN_WIDTH)*MASS_PLOT_BIN_WIDTH,
        np.floor(mass[valid].max()/MASS_PLOT_BIN_WIDTH)*MASS_PLOT_BIN_WIDTH
        + 1.5*MASS_PLOT_BIN_WIDTH, MASS_PLOT_BIN_WIDTH)
    oh_edges = np.arange(
        np.floor(metallicity[valid].min()/OH_PLOT_BIN_WIDTH)*OH_PLOT_BIN_WIDTH,
        np.floor(metallicity[valid].max()/OH_PLOT_BIN_WIDTH)*OH_PLOT_BIN_WIDTH
        + 1.5*OH_PLOT_BIN_WIDTH, OH_PLOT_BIN_WIDTH)
    bins = [mass_edges, oh_edges]
    median_sfr, _, _, _ = binned_statistic_2d(
        mass[valid], metallicity[valid], sfr[valid], statistic='median', bins=bins)
    counts, _, _ = np.histogram2d(mass[valid], metallicity[valid], bins=bins)
    colors = np.ma.masked_where(
        (counts < MASS_OH_PLOT_NMIN) | ~np.isfinite(median_sfr), median_sfr)
    fig, ax = plt.subplots(figsize=(10, 7))
    points = ax.pcolormesh(mass_edges, oh_edges, colors.T,
                           shading='flat', cmap='viridis', rasterized=True)
    colorbar = fig.colorbar(points, ax=ax, pad=0.02)
    colorbar.set_label(r'Median $\log_{10}(\mathrm{SFR}\,/\,M_\odot\,\mathrm{yr}^{-1})$')
    ax.set_xlabel(r'$\log_{10}(M_\ast/M_\odot)$')
    ax.set_ylabel(r'$12+\log_{10}(\mathrm{O/H})$')
    ax.set_xlim(8, 12)
    ax.set_ylim(7.5, 9.5)
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    fig.tight_layout()
    OUT_MASS_OH_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_MASS_OH_PNG, dpi=200)
    print(f'Saved: {OUT_MASS_OH_PNG} (N={valid.sum()})')
    if show:
        plt.show()
    plt.close(fig)


def calculate_jades_point(oh_min=None):
    """overlap_box_ne と同じ5パラメータ[SII]モデルでJADESを1スタック。

    金属量はカタログ値の中央値。R23換算・金属量測定誤差のMCは行わない。
    スペクトル測定誤差は独立Gaussianと仮定。波長補間後の相関は未使用。
    """
    tab = Table.read(JADES_PATH)
    oh = column(tab, JADES_OH_COLUMN)
    z = column(tab, 'z_Spec')
    selected = (np.asarray(tab['BASE_SAMPLE'], bool) & np.isfinite(oh)
                & (oh > 0) & np.isfinite(z) & (z > 0))
    if oh_min is not None:
        selected &= oh >= oh_min
    if not selected.any():
        raise ValueError('No JADES BASE_SAMPLE galaxies with metallicity')
    wave_grid = np.arange(6500., 6900., 0.5)
    fit_win = (wave_grid >= 6650) & (wave_grid <= 6800)
    fluxes, errors = [], []
    for row in tab[selected]:
        path = Path(str(row['SPEC_FILE']).strip())
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file():
            raise FileNotFoundError(f'JADES SPEC_FILE not found: {path}; rebuild sample paths')
        with fits.open(path) as hdus:
            d = hdus['EXTRACT1D'].data
            w = np.asarray(d['WAVELENGTH'], float)*1e4/(1+float(row['z_Spec']))
            f = np.asarray(d['FLUX'], float)
            e = np.asarray(d['FLUX_ERR'], float)
        good = np.isfinite(w) & np.isfinite(f) & np.isfinite(e) & (e > 0)
        if good.sum() < 2:
            raise ValueError(f'Insufficient finite JADES pixels: {path}')
        order = np.argsort(w[good])
        wg, fg, eg = w[good][order], f[good][order], e[good][order]
        fr = np.interp(wave_grid, wg, fg, left=np.nan, right=np.nan)
        er = np.interp(wave_grid, wg, eg, left=np.nan, right=np.nan)
        bad = fr < np.nanmedian(fr)-5*np.nanstd(fr)
        fr[bad], er[bad] = np.nan, np.nan
        # 両線の中心で有効なスペクトルのみ。同じ銀河群の金属量を使う。
        centers = [np.argmin(abs(wave_grid-line)) for line in (6718.29, 6732.67)]
        if not np.all(np.isfinite(fr[centers]) & np.isfinite(er[centers])):
            raise ValueError(f'JADES BASE_SAMPLE lacks [SII] coverage: {path}')
        fluxes.append(fr)
        errors.append(er)
    fluxes, errors = np.asarray(fluxes), np.asarray(errors)
    n_pix = np.sum(np.isfinite(fluxes), axis=0)
    stack = np.nanmean(fluxes, axis=0)
    # 独立Gaussianの平均の分散は sum(sigma_i^2)/N^2。
    # 個別スペクトルを全MC分展開せず、同じ分布を直接サンプリングする。
    stack_err = np.divide(np.sqrt(np.nansum(errors**2, axis=0)), n_pix,
                          out=np.full(len(wave_grid), np.nan), where=n_pix > 0)

    def model(w, f1, f2, dv, sig, c0):
        m1, m2 = np.array([6718.29, 6732.67])*(1+dv/299792.458)
        return (f1*np.exp(-0.5*((w-m1)/sig)**2)
                + f2*np.exp(-0.5*((w-m2)/sig)**2))/(np.sqrt(2*np.pi)*sig)+c0

    def fit(y):
        valid = fit_win & np.isfinite(y) & np.isfinite(stack_err) & (stack_err > 0)
        if valid.sum() <= 5:
            raise ValueError('Insufficient JADES stack pixels for fitting')
        scale = np.max(abs(y[valid]))
        yy = y[valid]/scale
        cont = np.median(yy)
        amp = max(np.max(yy)-cont, 1e-8)*np.sqrt(2*np.pi)*2
        p, _ = curve_fit(model, wave_grid[valid], yy,
                         p0=[amp, amp, 0, 2, cont],
                         bounds=([0, 0, -300, 0.3, -np.inf],
                                 [np.inf, np.inf, 300, 15, np.inf]),
                         sigma=stack_err[valid]/scale, absolute_sigma=True, maxfev=20000)
        p[[0, 1, 4]] *= scale
        return p

    best = fit(stack)
    rng = np.random.default_rng(SEED+1)
    ratios = np.full(JADES_N_MC, np.nan)
    for k in range(JADES_N_MC):
        try:
            p = fit(rng.normal(stack, stack_err))
            if p[1] > 0:
                ratios[k] = p[0]/p[1]
        except RuntimeError:
            pass
    if np.isfinite(ratios).sum() < 0.9*JADES_N_MC:
        raise RuntimeError('More than 10% of JADES MC fits failed')
    r16, r84 = np.nanpercentile(ratios, [16, 84])
    ratio = best[0]/best[1]
    atom = pn.Atom('S', 2)

    def ne(r):
        return float(atom.getTemDen(int_ratio=r, tem=TE, wave1=6716, wave2=6731,
                                    maxError=1e-6))

    ne_best, ne_lo, ne_hi = ne(ratio), ne(r84), ne(r16)
    oh16, oh50, oh84 = np.percentile(oh[selected], [16, 50, 84])
    plot_jades_metallicity_histogram(oh[selected])
    status = ('ok' if np.all(np.isfinite([ne_best, ne_lo, ne_hi]))
              and 0 < ne_lo <= ne_best <= ne_hi else 'unconstrained_interval')
    result = dict(N=int(selected.sum()), z_median=np.median(z[selected]),
                  oh_column=JADES_OH_COLUMN, oh_median=oh50, oh_p16=oh16, oh_p84=oh84,
                  oh_interval_type='sample_percentiles_not_measurement_error',
                  R=ratio, R_p16=r16, R_p84=r84, ne=ne_best, ne_lo=ne_lo, ne_hi=ne_hi,
                  dv=best[2], sigma=best[3], Te=TE, n_mc=JADES_N_MC,
                  mc_success_fraction=np.isfinite(ratios).mean(), status=status)
    OUT_JADES_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([result]).to_csv(OUT_JADES_CSV, index=False)
    OUT_JADES_STACK.parent.mkdir(parents=True, exist_ok=True)
    Table(dict(WAVE=wave_grid, FLUX=stack, FLUX_ERR=stack_err, N_GAL=n_pix)).write(
        OUT_JADES_CSV.with_suffix('.fits'), overwrite=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.step(wave_grid, stack, where='mid', color='k', label=f"JADES mean (N={result['N']})")
    ax.fill_between(wave_grid, stack-stack_err, stack+stack_err, alpha=0.3, color='gray')
    wf = np.linspace(6650, 6800, 1000)
    ax.plot(wf, model(wf, *best), color='crimson', label=f'Fit R={ratio:.3f}')
    ax.set_xlim(6650, 6800)
    # 表示波長内の誤差帯の min/max に、上下各5%の余白を付ける。
    visible = ((wave_grid >= 6650) & (wave_grid <= 6800)
               & np.isfinite(stack) & np.isfinite(stack_err))
    if np.any(visible):
        ymin = np.min((stack-stack_err)[visible])
        ymax = np.max((stack+stack_err)[visible])
        span = ymax-ymin
        padding = 0.05 * (span if span > 0 else max(abs(ymin), 1e-30))
        ax.set_ylim(ymin-padding, ymax+padding)
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    ax.set_xlabel('Rest-frame wavelength [Å]')
    ax.set_ylabel('Mean flux')
    ax.legend(fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_JADES_STACK, dpi=200)
    plt.close(fig)
    print('Saved:', OUT_JADES_CSV, OUT_JADES_STACK, OUT_JADES_CSV.with_suffix('.fits'))
    print('JADES:', result)
    return result


def plot_jades_metallicity_histogram(metallicity):
    """電子密度スタックと同じ銀河群のカタログ金属量分布。"""
    metallicity = np.asarray(metallicity, float)
    p16, median, p84 = np.percentile(metallicity, [16, 50, 84])
    width = JADES_OH_HIST_BIN_WIDTH
    edges = np.arange(np.floor(metallicity.min()/width)*width,
                      np.floor(metallicity.max()/width)*width+1.5*width, width)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.axvspan(p16, p84, color='crimson', alpha=0.12,
               label=f'16–84%: {p16:.3f}–{p84:.3f}')
    ax.hist(metallicity, bins=edges, histtype='step', linewidth=2,
            color='crimson', label=f'JADES (N={len(metallicity)})')
    ax.axvline(median, color='k', linestyle='--', linewidth=1.5,
               label=f'Median: {median:.3f}')
    ax.set_xlabel(r'$12+\log_{10}(\mathrm{O/H})$')
    ax.set_ylabel('Number of galaxies')
    ax.set_title(JADES_OH_COLUMN, fontsize=16)
    ax.legend(fontsize=12)
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    fig.tight_layout()
    OUT_JADES_OH_HIST.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_JADES_OH_HIST, dpi=200)
    plt.close(fig)
    print('Saved:', OUT_JADES_OH_HIST)


if __name__ == '__main__':
    main()
