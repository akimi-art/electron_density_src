#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SDSS/local と JADES/high-z の [SII] 電子密度を隙間のない1行4列で比較。

JADES_spectra_stack_x_ssfr_equal_width_v3.py のサンプル選択・静止系補間・
非正規化 mean stack を踏襲。各軸は独立にビン選択する（4変数の同時カットではない）。
local: 選択済み SDSS v2、ビン内で両線の2–98%共通 flux カット後、平均 flux の比。
high-z: CSV + grating別スペクトル、5パラメータ [SII] フィットの積分 flux 比。
誤差: 固定サンプルの独立 Gaussian flux MC（bootstrap ではない）。
Te=10^4 K。横棒は使用銀河の16–84%分布であり、測定誤差ではない。
物理的な [SII] 比の範囲外をクリップせず、制約できない区間はCSVに残して描画しない。

実行（作業ディレクトリは任意）:
    python electron_density_src/analysis/SDSS/stacked_sii_ne_vs_mass_sfr_ssfr_sigma_sfr.py
    # 動作確認用: --local-n-mc 20 --jades-n-mc 20 --no-show
"""
from pathlib import Path
import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from astropy.io import fits
from astropy.table import Table
from astropy.cosmology import Planck18
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import pyneb as pn

ROOT = Path(__file__).resolve().parents[3]
LOCAL_PATH = ROOT / 'results/fits/sdss_sample_v2_zlt0.1879_Lgt1e+39_Ka03_Reu.fits'
JADES_CSV = ROOT / 'results/JADES/JADES_DR3/data_from_Nishigaki/jades_info_with_HA_plus_logSFR_with_Reff.csv'
SPEC_DIR = ROOT / 'data/data_JADES/JADES_DR3/JADES_DR3_full_spectra'
STEM = Path(__file__).stem
CSV_DIR = ROOT / 'results/csv' / STEM
STACK_DIR = CSV_DIR / 'jades_stacks'
FIG_PATH = ROOT / 'results/figure' / f'{STEM}.png'

# ビン境界は lo <= x < hi。以下はすべて log（底10）。
mass_bins = [(9.3, 10.0)]
sfr_bins = [(0.7, 0.9), (1.1, 1.3)]
ssfr_bins = [(-9.0, -8.4)]
sigma_bins = [(-1.0, 0.0)]
PANELS = {
    'mass': dict(bins=mass_bins, xlim=(9.0, 10.5), label=r'$\log(M_\ast/M_\odot)$'),
    'sfr': dict(bins=sfr_bins, xlim=(-1.0, 1.5), label=r'$\log(\mathrm{SFR}/[M_\odot\,\mathrm{yr}^{-1}])$'),
    'ssfr': dict(bins=ssfr_bins, xlim=(-10.5, -8.4), label=r'$\log(\mathrm{sSFR}/\mathrm{yr}^{-1})$'),
    'sigma': dict(bins=sigma_bins, xlim=(-2.0, 0.5), label=r'$\log(\Sigma_{\mathrm{SFR}}/[M_\odot\,\mathrm{yr}^{-1}\,\mathrm{kpc}^{-2}])$'),
}
LOCAL_BIN_WIDTH = 0.2
NMIN = 100
FLUX_PERCENTILES = (2, 98)
LOCAL_N_MC, JADES_N_MC = 5000, 500
SEED, TE = 42, 1e4
WAVE_GRID = np.arange(6500., 6900., 0.5)
FIT_WINDOW = (6650., 6800.)
SII_WAVES = np.array([6718.29, 6732.67])  # 真空波長
C_KMS = 299792.458
BAD_IDS = {'00024958', '00025356', '00051209', '00082961',
           '00004504', '00028139', '00045967', '10020038'}
GRATINGS = [('G140M', 0.5, 1.7), ('G235M', 1.5, 3.6), ('G395M', 3.3, 6.7)]
S2 = pn.Atom('S', 2)
R_MAX = float(S2.getEmissivity(tem=TE, den=1e-3, wave=6716) / S2.getEmissivity(tem=TE, den=1e-3, wave=6731))
R_MIN = float(S2.getEmissivity(tem=TE, den=1e8, wave=6716) / S2.getEmissivity(tem=TE, den=1e8, wave=6731))

def column(tab, name):
    """列を float 配列で返す (マスクは NaN)。"""
    return np.ma.filled(np.ma.asarray(tab[name], dtype=float), np.nan)

def in_range(x, rng):
    return np.isfinite(x) & (x >= rng[0]) & (x < rng[1])

def ne_from_ratio(r):
    """R = I(6716)/I(6731) → n_e。有効範囲 (R_MIN, R_MAX) の外は NaN。"""
    if not (np.isfinite(r) and R_MIN < r < R_MAX):
        return np.nan
    return float(S2.getTemDen(int_ratio=r, tem=TE, wave1=6716, wave2=6731, maxError=1e-6))

def ratio_draws(f1, e1, f2, e2, n_mc, rng):
    """各線の flux を独立 Gaussian で揺らし, 平均の比を返す (64 回ずつ生成)。"""
    out = np.full(n_mc, np.nan)
    for start in range(0, n_mc, 64):
        n = min(64, n_mc - start)
        num = rng.normal(f1[:, None], e1[:, None], (len(f1), n)).mean(axis=0)
        den = rng.normal(f2[:, None], e2[:, None], (len(f2), n)).mean(axis=0)
        ok = (num > 0) & (den > 0)
        out[start:start + n][ok] = num[ok] / den[ok]
    return out

def sii_model(w, f1, f2, dv, sig, c0):
    """2 本のガウシアン (中心は真空波長, 共通の dv と sigma) + 定数の連続光。"""
    m1, m2 = SII_WAVES * (1 + dv / C_KMS)
    g = lambda m, f: f * np.exp(-0.5 * ((w - m) / sig) ** 2) / (np.sqrt(2 * np.pi) * sig)
    return g(m1, f1) + g(m2, f2) + c0

def fit_sii(y, err):
    """フィット窓内で 5 パラメータのフィット。flux を正規化して解き, 元の単位に戻す。"""
    m = (WAVE_GRID >= FIT_WINDOW[0]) & (WAVE_GRID <= FIT_WINDOW[1]) \
        & np.isfinite(y) & np.isfinite(err) & (err > 0)
    if m.sum() <= 5:
        raise ValueError('JADES: フィットに使えるピクセルが足りません')
    scale = np.max(np.abs(y[m]))
    yy = y[m] / scale
    cont = np.median(yy)
    amp = max(np.max(yy) - cont, 1e-8) * np.sqrt(2 * np.pi) * 2
    p, _ = curve_fit(sii_model, WAVE_GRID[m], yy, p0=[amp, amp, 0, 2, cont],
                     bounds=([0, 0, -300, 0.3, -np.inf], [np.inf, np.inf, 300, 15, np.inf]),
                     sigma=err[m] / scale, absolute_sigma=True, maxfev=20000)
    p[[0, 1, 4]] *= scale
    return p


def log_sigma(sfr, radius_arcsec, z):
    radius = np.asarray(radius_arcsec, float) * Planck18.kpc_proper_per_arcmin(z).value / 60
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where((radius > 0) & (np.asarray(z) > 0),
                        sfr - np.log10(2 * np.pi * radius**2), np.nan)


def load_local():
    tab = Table.read(LOCAL_PATH, hdu=1)
    if 'SELECTED_V2' not in tab.colnames or not np.all(np.asarray(tab['SELECTED_V2'], bool)):
        raise ValueError('local: SELECTED_V2 の選択済みファイルが必要です')
    names = dict(mass='sm_MEDIAN', sfr='sfr_MEDIAN', f1='SII_6717_FLUX',
                 f2='SII_6731_FLUX', e1='SII_6717_FLUX_ERR', e2='SII_6731_FLUX_ERR')
    d = {k: column(tab, c) for k, c in names.items()}
    for key in ('mass', 'sfr'):
        d[key][d[key] <= -90] = np.nan
    d['ssfr'] = d['sfr'] - d['mass']
    # 選択済み Reu カタログに保存された u-band 有効半径 [kpc]。
    radius = column(tab, 'Re_kpc_u')
    with np.errstate(divide='ignore', invalid='ignore'):
        d['sigma'] = np.where(radius > 0, d['sfr'] - np.log10(2 * np.pi * radius**2), np.nan)
    d['use'] = (np.isfinite(d['f1']) & np.isfinite(d['f2']) &
                np.isfinite(d['e1']) & np.isfinite(d['e2']) & (d['e1'] > 0) & (d['e2'] > 0))
    print(f'local: {len(tab):,} selected, {d["use"].sum():,} valid flux/error')
    return d


def load_jades():
    """参照v3のCSV選択・grating順・Hα上位0.5%除外・既知bad IDを使用。

    重複IDは最初に使用可能なスペクトルが見つかった時点で登録する。
    半径のない銀河も mass/SFR/sSFR パネルでは使用できる。
    """
    df = pd.read_csv(JADES_CSV).drop_duplicates('NIRSpec_ID', keep='first')
    df = df.dropna(subset=['z_spec', 'HA_6563_flux', 'logSFR_hb'])
    items, used = [], set()
    for gr, zlo, zhi in GRATINGS:
        sub = df[(df.z_spec > zlo) & (df.z_spec < zhi)]
        ha = sub.HA_6563_flux.to_numpy()
        good = np.isfinite(ha) & (ha > 0)
        if not good.any():
            continue
        cutoff = np.percentile(ha[good], 99.5)
        for _, row in sub.iterrows():
            sid = f'{int(row.NIRSpec_ID):08d}'
            if sid in used or sid in BAD_IDS or not (0 < row.HA_6563_flux <= cutoff):
                continue
            files = sorted((SPEC_DIR / f'JADES_DR3_{gr}').glob(f'*{sid}*_x1d.fits'))
            if not files:
                continue
            try:
                with fits.open(files[0]) as h:
                    data = h['EXTRACT1D'].data
                    w = np.asarray(data['WAVELENGTH'], float) * 1e4 / (1 + row.z_spec)
                    f = np.asarray(data['FLUX'], float).copy()
                    e = np.asarray(data['FLUX_ERR'], float).copy()
                regions = [(6550, 6575, 2), (6705, 6740, 2), (6600, 6650, 3)]
                if any(np.count_nonzero((w > lo) & (w < hi) & np.isfinite(f)) < n
                       for lo, hi, n in regions):
                    continue
                bad = (f < np.nanmedian(f) - 5 * np.nanstd(f)) | ~np.isfinite(e) | (e <= 0)
                f[bad], e[bad] = np.nan, np.nan
                fi = interp1d(w, f, bounds_error=False, fill_value=np.nan)(WAVE_GRID)
                ei = interp1d(w, e, bounds_error=False, fill_value=np.nan)(WAVE_GRID)
                if not np.isfinite(fi).any():
                    continue
                mass, sfr = float(row.logM), float(row.logSFR_hb)
                mass = mass if mass > -90 else np.nan
                sfr = sfr if sfr > -90 else np.nan
                items.append(dict(id=sid, z=row.z_spec, mass=mass, sfr=sfr,
                                  ssfr=sfr-mass, sigma=float(log_sigma(sfr, row.ReffOpt, row.z_spec)),
                                  flux=fi, err=ei, spec_file=str(files[0]), grating=gr))
                used.add(sid)
            except (OSError, ValueError) as exc:
                warnings.warn(f'{sid}: {exc}')
    if not items:
        raise ValueError('JADES: 使用できるスペクトルがありません')
    print(f'JADES: {len(items)} usable spectra')
    return items


def density_summary(ratio, ratios):
    finite = np.isfinite(ratios)
    r16, r84 = np.percentile(ratios[finite], [16, 84]) if finite.any() else (np.nan, np.nan)
    ne, lo, hi = ne_from_ratio(ratio), ne_from_ratio(r84), ne_from_ratio(r16)
    ok = np.all(np.isfinite([ne, lo, hi])) and 0 < lo <= ne <= hi
    return dict(R=ratio, R_p16=r16, R_p84=r84, ne=ne, ne_lo=lo, ne_hi=hi,
                status='ok' if ok else 'unconstrained_interval',
                mc_success_fraction=finite.mean(),
                physical_ratio_fraction=np.mean((ratios > R_MIN) & (ratios < R_MAX)))


def position_summary(values):
    lo, med, hi = np.percentile(values, [16, 50, 84])
    return dict(x=med, x_p16=lo, x_p84=hi)


def stack_local(d, key, config, n_mc, rng):
    # 描画範囲を覆う0.2 dexビン。端でも幅を0.2に保ち、描画時のみxlimで切る。
    lo, hi = config['xlim']
    start = np.floor(lo / LOCAL_BIN_WIDTH + 1e-8) * LOCAL_BIN_WIDTH
    edges = np.round(start + np.arange(int(np.ceil((hi-start)/LOCAL_BIN_WIDTH)) + 1) * LOCAL_BIN_WIDTH, 10)
    rows = []
    for low, high in zip(edges[:-1], edges[1:]):
        idx = np.flatnonzero(d['use'] & in_range(d[key], (low, high)))
        row = dict(sample='local', parameter=key, bin_lo=low, bin_hi=high,
                   N_before_cut=len(idx), N=0, status='insufficient_sample', Te=TE, n_mc=n_mc)
        if len(idx) >= NMIN:
            keep = np.ones(len(idx), bool)
            for name in ('f1', 'f2'):
                a, b = np.percentile(d[name][idx], FLUX_PERCENTILES)
                keep &= (d[name][idx] >= a) & (d[name][idx] <= b)
            idx = idx[keep]
            row['N'] = len(idx)
            if len(idx) >= NMIN:
                draws = ratio_draws(d['f1'][idx], d['e1'][idx], d['f2'][idx], d['e2'][idx], n_mc, rng)
                central = np.nanmedian(draws) if np.isfinite(draws).any() else np.nan
                row.update(position_summary(d[key][idx]))
                row.update(density_summary(central, draws))
        rows.append(row)
    return rows


def stack_highz(items, key, config, n_mc, rng):
    rows, members = [], []
    for low, high in config['bins']:
        selected = [it for it in items if np.isfinite(it[key]) and low <= it[key] < high and 0.5 <= it['z'] < 8]
        row = dict(sample='high-z', parameter=key, bin_lo=low, bin_hi=high,
                   N=len(selected), Te=TE, n_mc=n_mc, status='empty_bin')
        if not selected:
            warnings.warn(f'JADES {key} [{low}, {high}): empty bin')
            rows.append(row)
            continue
        for it in selected:
            members.append(dict(parameter=key, bin_lo=low, bin_hi=high, id=it['id'],
                                x=it[key], z=it['z'], spec_file=it['spec_file'], grating=it['grating']))
        F, E = np.array([it['flux'] for it in selected]), np.array([it['err'] for it in selected])
        valid = np.isfinite(F) & np.isfinite(E) & (E > 0)
        count = valid.sum(axis=0)
        flux = np.divide(np.where(valid, F, 0).sum(axis=0), count,
                         out=np.full(WAVE_GRID.size, np.nan), where=count > 0)
        err = np.divide(np.sqrt(np.where(valid, E**2, 0).sum(axis=0)), count,
                        out=np.full(WAVE_GRID.size, np.nan), where=count > 0)
        # Gaussian MCで平均を取る分布のsigma = sqrt(sum(err_i**2))/N。
        # スタックをこのsigmaで揺らすことは個々のfluxを揺らして平均することと等価。
        tag = f'{key}_{low:.1f}_{high:.1f}'
        path = STACK_DIR / f'{tag}.fits'
        Table(dict(WAVE=WAVE_GRID, FLUX=flux, FLUX_ERR=err, N_GAL=count)).write(path, overwrite=True)
        row.update(position_summary(np.array([it[key] for it in selected])))
        row.update(z_median=np.median([it['z'] for it in selected]), stack_file=str(path))
        try:
            best = fit_sii(flux, err)
            ratios = np.full(n_mc, np.nan)
            for i in range(n_mc):
                try:
                    p = fit_sii(rng.normal(flux, err), err)
                    ratios[i] = p[0] / p[1] if p[1] > 0 else np.nan
                except (ValueError, RuntimeError):
                    pass
            row.update(density_summary(best[0]/best[1], ratios))
            row.update(dv=best[2], sigma_angstrom=best[3])
            if row['mc_success_fraction'] < 0.9:
                row['status'] = 'mc_fit_failure'
        except (ValueError, RuntimeError) as exc:
            row.update(status='fit_failure', error=str(exc))
        print(f'JADES {key} [{low}, {high}): N={len(selected)}, {row["status"]}, ne={row.get("ne", np.nan):.1f}')
        rows.append(row)
    return rows, members


def plot_results(results, show):
    plt.rcParams.update({'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix',
                         'font.size': 32, 'axes.labelsize': 32, 'xtick.labelsize': 32,
                         'ytick.labelsize': 32, 'xtick.direction': 'in', 'ytick.direction': 'in',
                         'xtick.top': True, 'ytick.right': True, 'axes.linewidth': 2})
    fig, axes = plt.subplots(1, 4, figsize=(24, 6), sharey=True, gridspec_kw={'wspace': 0})
    for ax, (key, config) in zip(axes, PANELS.items()):
        for sample, color in [('local', 'k'), ('high-z', 'crimson')]:
            sub = results[(results.parameter == key) & (results['sample'] == sample) & (results.status == 'ok')]
            if sub.empty:
                continue
            x, y = sub.x.to_numpy(), np.log10(sub['ne'].to_numpy())
            ax.errorbar(x, y, xerr=np.vstack([x-sub.x_p16, sub.x_p84-x]),
                        yerr=np.vstack([y-np.log10(sub.ne_lo), np.log10(sub.ne_hi)-y]),
                        fmt='s', color=color, ms=8 if sample == 'local' else 11,
                        capsize=3, lw=1.5, zorder=3 if sample == 'high-z' else 2)
        ax.set_xlim(config['xlim'])
        ax.set_xlabel(config['label'])
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, prune='both'))
        ax.tick_params(which='major', length=10, width=2)
        ax.tick_params(which='minor', length=5, width=1.5)
        ax.minorticks_on()
        ax.tick_params(labelleft=ax is axes[0])
    axes[0].set_ylabel(r'$\log(n_{\mathrm{e}}/\mathrm{cm}^{-3})$')
    valid = results[results.status == 'ok']
    if not valid.empty:
        axes[0].set_ylim(min(0.5, np.log10(valid.ne_lo.min())-0.15),
                         max(3.5, np.log10(valid.ne_hi.max())+0.15))
    else:
        axes[0].set_ylim(0.5, 3.5)
        warnings.warn('制約可能な電子密度がありません。CSVのstatusを確認してください')
    # tight_layout は使わず、4枚の描画領域を直接接続する。
    fig.subplots_adjust(left=0.065, right=0.99, bottom=0.20, top=0.97, wspace=0)
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=200)
    fig.savefig(FIG_PATH.with_suffix('.pdf'))
    if show:
        plt.show()
    plt.close(fig)
    print('Saved:', FIG_PATH)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local-n-mc', type=int, default=LOCAL_N_MC)
    parser.add_argument('--jades-n-mc', type=int, default=JADES_N_MC)
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()
    if min(args.local_n_mc, args.jades_n_mc) < 2:
        parser.error('MC回数は2以上にしてください')
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    STACK_DIR.mkdir(parents=True, exist_ok=True)
    local, jades = load_local(), load_jades()
    rng = np.random.default_rng(SEED)
    rows, members = [], []
    for key, config in PANELS.items():
        rows.extend(stack_local(local, key, config, args.local_n_mc, rng))
        highz_rows, used = stack_highz(jades, key, config, args.jades_n_mc, rng)
        rows.extend(highz_rows)
        members.extend(used)
    results = pd.DataFrame(rows)
    # 全ビンが空の場合も出力列と描画を保つ。
    for name in ('x', 'x_p16', 'x_p84', 'ne', 'ne_lo', 'ne_hi'):
        if name not in results:
            results[name] = np.nan
    results['x_interval_type'] = 'sample_percentiles_not_measurement_error'
    results.to_csv(CSV_DIR / 'density_results.csv', index=False)
    pd.DataFrame(members).to_csv(CSV_DIR / 'jades_members.csv', index=False)
    print(results.groupby(['sample', 'parameter', 'status']).size().to_string())
    plot_results(results, not args.no_show)


if __name__ == '__main__':
    main()
