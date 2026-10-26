#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stacked_sii_ne_vs_oh.py

M*–O/H 平面の同じ領域で local (SDSS) と high-z (JADES) の [SII] から n_e を求め,
log n_e vs 12+log(O/H) に描く。

  local : JADES と同じ M* 範囲で区切り, O/H ビンごとに各線の flux の平均の比を取る
          (ビン内で両線の flux の 2–98% を共通選別)。
          誤差は固定サンプルの独立 Gaussian flux MC (bootstrap ではない)。
  JADES : BASE_SAMPLE かつ M*・O/H の長方形領域の銀河を静止系で平均スタックし,
          [SII] ダブレット (5 パラメータ) をフィット。誤差はスタックの MC。

金属量はどちらも strong-line 法のカタログ値 (較正の違いは考慮しない)。
O/H・M* の横棒や 16–84% は銀河間の分布であり, 測定誤差ではない。

実行: python electron_density_src/analysis/SDSS/stacked_sii_ne_vs_oh.py [--n-mc N]
"""
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from scipy.optimize import curve_fit
from scipy.stats import binned_statistic_2d
import pyneb as pn

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


# =====================================================================
# 設定
# =====================================================================
ROOT = Path(__file__).resolve().parents[3]

# ---- 入力 ----
LOCAL_PATH = ROOT / 'results/fits/sdss_sample_v2_zlt0.1879_Lgt1e+39_Ka03_Reu.fits'  # 選択後
FLAGS_PATH = ROOT / 'results/fits/mpajhu_dr7_v5_2_master_v2_Ka03_Reu.fits'          # 全行+フラグ
JADES_PATH = ROOT / 'results/JADES/sample/jades_all_with_flags.fits'

# ---- 比較領域 (lo <= x < hi) ----
JADES_MASS_RANGE = (9.0, 9.5)          # JADES の log(M*/Msun)
JADES_OH_RANGE = (8.1, 8.5)            # JADES の 12+log(O/H)
LOCAL_MASS_RANGE = JADES_MASS_RANGE    # local は M* のみで区切る (O/H はビン分け)

# ---- 列名 ----
LOCAL_COLS = dict(mass='sm_MEDIAN', oh='oh_MEDIAN', oh16='oh_P16', oh84='oh_P84',
                  sfr='sfr_MEDIAN', f1='SII_6717_FLUX', f2='SII_6731_FLUX',
                  e1='SII_6717_FLUX_ERR', e2='SII_6731_FLUX_ERR')
JADES_MASS_COL, JADES_OH_COL = 'N26_logM', 'N26_logOH'   # 既存推定値 (R23 からの再推定ではない)

# ---- 出力 ----
CSV_DIR = ROOT / 'results/csv'
FIG_DIR = ROOT / 'results/figure'
OUT = dict(
    local_csv=CSV_DIR / 'stacked_sii_ne_vs_oh.csv',
    jades_csv=CSV_DIR / 'stacked_sii_ne_vs_oh_jades.csv',
    jades_stack_fits=CSV_DIR / 'stacked_sii_ne_vs_oh_jades.fits',
    jades_mass_csv=CSV_DIR / 'jades_oh_sample_mass_summary.csv',
    compl_csv=CSV_DIR / 'completeness_vs_oh.csv',
    ne_png=FIG_DIR / 'stacked_sii_ne_vs_oh.png',
    overview_png=FIG_DIR / 'stellar_mass_vs_metallicity_sdss_jades.png',
    dist_png=FIG_DIR / 'oh_comparison_mass_oh_distributions.png',
    compl_png=FIG_DIR / 'completeness_vs_oh.png',
    sfr_map_png=FIG_DIR / 'stellar_mass_vs_metallicity_sfr_sdss.png',
    jades_stack_png=FIG_DIR / 'stacked_sii_ne_vs_oh_jades_stack.png',
    jades_oh_hist_png=FIG_DIR / 'jades_metallicity_histogram.png',
    jades_mass_hist_png=FIG_DIR / 'jades_oh_sample_mass_histogram.png',
)

# ---- 解析 ----
TE = 1e4
SEED = 42
OH_BIN_WIDTH = 0.1          # local の O/H ビン幅
NMIN = 100                  # local の 1 ビンの最小銀河数 (flux カットの前後とも)
FLUX_PERCENTILES = (2, 98)  # local のビン内 flux 共通選別
N_MC = 5000                 # local の flux MC 回数
JADES_N_MC = 500            # JADES のスタック MC 回数
COMPL_NMIN = 30             # completeness 図に描く分母の最小数
WAVE_GRID = np.arange(6500., 6900., 0.5)   # 静止系 [Å]
FIT_WINDOW = (6650., 6800.)
SII_WAVES = np.array([6718.29, 6732.67])   # 真空波長
C_KMS = 299792.458

# ---- 作図 ----
DIST_BIN_WIDTH = 0.05
JADES_MASS_HIST_BIN_WIDTH = 0.2
JADES_OH_HIST_BIN_WIDTH = 0.1
MAP_MASS_BIN, MAP_OH_BIN, MAP_NMIN = 0.1, 0.05, 5

# ---- PyNeb ----
S2 = pn.Atom('S', 2)
R_MAX = float(S2.getEmissivity(tem=TE, den=1e-3, wave=6716) / S2.getEmissivity(tem=TE, den=1e-3, wave=6731))
R_MIN = float(S2.getEmissivity(tem=TE, den=1e8, wave=6716) / S2.getEmissivity(tem=TE, den=1e8, wave=6731))


# =====================================================================
# 共通の関数
# =====================================================================
def column(tab, name):
    """列を float 配列で返す (マスクは NaN)。"""
    return np.ma.filled(np.ma.asarray(tab[name], dtype=float), np.nan)


def in_range(x, rng):
    return np.isfinite(x) & (x >= rng[0]) & (x < rng[1])


def bin_edges(values, width):
    """values を覆う等間隔のビン境界。"""
    lo = np.floor(np.min(values) / width) * width
    hi = np.floor(np.max(values) / width) * width + 1.5 * width
    return np.arange(lo, hi, width)


def ne_from_ratio(r):
    """R = I(6716)/I(6731) → n_e。有効範囲 (R_MIN, R_MAX) の外は NaN。"""
    if not (np.isfinite(r) and R_MIN < r < R_MAX):
        return np.nan
    return float(S2.getTemDen(int_ratio=r, tem=TE, wave1=6716, wave2=6731, maxError=1e-6))


def finish(fig, path):
    """枠線を太くして保存・表示する。"""
    for ax in fig.axes:
        for sp in ax.spines.values():
            sp.set_linewidth(2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.show()
    print("Saved figure:", path)
    plt.close(fig)


def save_csv(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


# =====================================================================
# データの読み込みと選択
# =====================================================================
def load_local():
    """local の選択後サンプル。'use' = 有効な O/H・[SII] かつ M* 範囲内。"""
    tab = Table.read(LOCAL_PATH, hdu=1)
    assert 'SELECTED_V2' in tab.colnames, 'SELECTED_V2 列がありません（選択後のファイルではない可能性）'
    assert np.all(np.asarray(tab['SELECTED_V2'], bool)), '選択されていない行が含まれています'
    d = {k: column(tab, c) for k, c in LOCAL_COLS.items()}
    # -99.9 などの欠損値と, 順序が不正な金属量区間を除外
    valid = (np.isfinite(d['oh']) & np.isfinite(d['oh16']) & np.isfinite(d['oh84'])
             & (d['oh16'] > 0) & (d['oh16'] <= d['oh']) & (d['oh'] <= d['oh84'])
             & np.isfinite(d['f1']) & np.isfinite(d['f2'])
             & np.isfinite(d['e1']) & np.isfinite(d['e2']) & (d['e1'] > 0) & (d['e2'] > 0))
    d['use'] = valid & in_range(d['mass'], LOCAL_MASS_RANGE)
    if not d['use'].any():
        raise ValueError('local: M* 範囲内に有効な O/H・[SII] がありません')
    print(f'[local] 選択後 {len(tab):,} → O/H・[SII] 有効 {valid.sum():,} '
          f'→ logM* {LOCAL_MASS_RANGE} {d["use"].sum():,}')
    return d


def load_jades():
    """JADES。'sel' = BASE_SAMPLE かつ M*・O/H の長方形領域内。"""
    tab = Table.read(JADES_PATH)
    d = dict(tab=tab, mass=column(tab, JADES_MASS_COL), oh=column(tab, JADES_OH_COL),
             z=column(tab, 'z_Spec'))
    d['sel'] = (np.asarray(tab['BASE_SAMPLE'], bool)
                & in_range(d['mass'], JADES_MASS_RANGE) & in_range(d['oh'], JADES_OH_RANGE)
                & np.isfinite(d['z']) & (d['z'] > 0))
    if not d['sel'].any():
        raise ValueError('JADES: 領域内に BASE_SAMPLE の銀河がありません')
    print(f'[JADES] logM* {JADES_MASS_RANGE}, O/H {JADES_OH_RANGE}: N = {d["sel"].sum()}')
    return d


# =====================================================================
# local: O/H ビンごとの flux 平均の比
# =====================================================================
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


def stack_local(loc, edges, n_mc, rng):
    rows = []
    print('[local] O/H ビン: N (flux カット前 → 後), log n_e')
    for lo, hi in zip(edges[:-1], edges[1:]):
        idx = np.flatnonzero(loc['use'] & (loc['oh'] >= lo) & (loc['oh'] < hi))
        n_before = len(idx)
        if n_before < NMIN:
            continue
        keep = np.ones(n_before, bool)
        for f in (loc['f1'], loc['f2']):
            p_lo, p_hi = np.percentile(f[idx], FLUX_PERCENTILES)
            keep &= (f[idx] >= p_lo) & (f[idx] <= p_hi)
        idx = idx[keep]
        if len(idx) < NMIN:
            continue
        ratios = ratio_draws(loc['f1'][idx], loc['e1'][idx], loc['f2'][idx], loc['e2'][idx], n_mc, rng)
        r16, r50, r84 = (np.nanpercentile(ratios, [16, 50, 84])
                         if np.isfinite(ratios).any() else (np.nan,) * 3)
        ne, ne_lo, ne_hi = ne_from_ratio(r50), ne_from_ratio(r84), ne_from_ratio(r16)  # R→ne は単調減少
        status = 'ok' if np.all(np.isfinite([ne, ne_lo, ne_hi])) else 'unconstrained_interval'
        rows.append(dict(
            oh_lo=lo, oh_hi=hi, oh_median=np.median(loc['oh'][idx]),
            oh_p16_median=np.median(loc['oh16'][idx]), oh_p84_median=np.median(loc['oh84'][idx]),
            logM_median=np.median(loc['mass'][idx]),
            N_before_cut=n_before, N=len(idx), n_mc=n_mc, Te=TE,
            R_mean=r50, R_mean_p16=r16, R_mean_p84=r84,
            ne_mean=ne, ne_mean_p16=ne_lo, ne_mean_p84=ne_hi, status_mean=status,
            valid_ratio_fraction_mean=np.isfinite(ratios).mean(),
            physical_ratio_fraction_mean=np.mean((ratios > R_MIN) & (ratios < R_MAX))))
        ne_text = f'{np.log10(ne):.2f}' if status == 'ok' else '(区間が定まらない)'
        print(f'  {lo:.1f}–{hi:.1f}: {n_before:>6,} → {len(idx):>6,}   {ne_text}')
    if not rows:
        raise ValueError('local: flux カット後に NMIN を満たすビンがありません')
    res = pd.DataFrame(rows)
    save_csv(res, OUT['local_csv'])
    return res


# =====================================================================
# JADES: 静止系の平均スタックと [SII] フィット
# =====================================================================
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


def read_rest_spectrum(path, z):
    """静止系に直して WAVE_GRID に補間。5σ 以上の負の外れ値は NaN。"""
    with fits.open(path) as h:
        d = h['EXTRACT1D'].data
        w = np.asarray(d['WAVELENGTH'], float) * 1e4 / (1 + z)
        f = np.asarray(d['FLUX'], float)
        e = np.asarray(d['FLUX_ERR'], float)
    good = np.isfinite(w) & np.isfinite(f) & np.isfinite(e) & (e > 0)
    if good.sum() < 2:
        raise ValueError(f'JADES: 有効なピクセルが足りません: {path}')
    o = np.argsort(w[good])
    fr = np.interp(WAVE_GRID, w[good][o], f[good][o], left=np.nan, right=np.nan)
    er = np.interp(WAVE_GRID, w[good][o], e[good][o], left=np.nan, right=np.nan)
    bad = fr < np.nanmedian(fr) - 5 * np.nanstd(fr)
    fr[bad], er[bad] = np.nan, np.nan
    centers = [np.argmin(np.abs(WAVE_GRID - l)) for l in SII_WAVES]
    if not np.all(np.isfinite(fr[centers]) & np.isfinite(er[centers])):
        raise ValueError(f'JADES: [SII] が波長範囲にありません: {path}')
    return fr, er


def stack_jades(jd):
    rows = jd['tab'][jd['sel']]
    spectra = []
    for row in rows:
        path = Path(str(row['SPEC_FILE']).strip())
        path = path if path.is_absolute() else ROOT / path
        if not path.is_file():
            raise FileNotFoundError(f'JADES SPEC_FILE がありません: {path}')
        spectra.append(read_rest_spectrum(path, float(row['z_Spec'])))
    F = np.array([s[0] for s in spectra])
    E = np.array([s[1] for s in spectra])
    n_pix = np.sum(np.isfinite(F), axis=0)
    stack = np.nanmean(F, axis=0)
    # 独立 Gaussian の平均の誤差 sqrt(Σσ²)/N
    stack_err = np.divide(np.sqrt(np.nansum(E ** 2, axis=0)), n_pix,
                          out=np.full(WAVE_GRID.size, np.nan), where=n_pix > 0)

    best = fit_sii(stack, stack_err)
    rng = np.random.default_rng(SEED + 1)
    ratios = np.full(JADES_N_MC, np.nan)
    for k in range(JADES_N_MC):
        try:
            p = fit_sii(rng.normal(stack, stack_err), stack_err)
            if p[1] > 0:
                ratios[k] = p[0] / p[1]
        except RuntimeError:
            pass
    if np.isfinite(ratios).sum() < 0.9 * JADES_N_MC:
        raise RuntimeError('JADES: MC フィットの 10% 以上が失敗しました')

    R = best[0] / best[1]
    r16, r84 = np.nanpercentile(ratios, [16, 84])
    ne, ne_lo, ne_hi = ne_from_ratio(R), ne_from_ratio(r84), ne_from_ratio(r16)
    oh = jd['oh'][jd['sel']]
    oh16, oh50, oh84 = np.percentile(oh, [16, 50, 84])
    status = ('ok' if np.all(np.isfinite([ne, ne_lo, ne_hi])) and 0 < ne_lo <= ne <= ne_hi
              else 'unconstrained_interval')
    res = dict(N=len(rows), z_median=np.median(jd['z'][jd['sel']]), oh_column=JADES_OH_COL,
               mass_range_lo=JADES_MASS_RANGE[0], mass_range_hi=JADES_MASS_RANGE[1],
               oh_range_lo=JADES_OH_RANGE[0], oh_range_hi=JADES_OH_RANGE[1],
               oh_median=oh50, oh_p16=oh16, oh_p84=oh84,
               oh_interval_type='sample_percentiles_not_measurement_error',
               R=R, R_p16=r16, R_p84=r84, ne=ne, ne_lo=ne_lo, ne_hi=ne_hi,
               dv=best[2], sigma=best[3], Te=TE, n_mc=JADES_N_MC,
               mc_success_fraction=np.isfinite(ratios).mean(), status=status)
    if status == 'ok':
        print(f'[JADES] N = {res["N"]}, z_med = {res["z_median"]:.2f}, R = {R:.3f}, '
              f'log n_e = {np.log10(ne):.2f} (+{np.log10(ne_hi / ne):.2f}/-{np.log10(ne / ne_lo):.2f})')
    else:
        print(f'[JADES] N = {res["N"]}, R = {R:.3f}: n_e の区間が定まりません (CSV 参照)')

    # スタックの保存と確認図
    OUT['jades_stack_fits'].parent.mkdir(parents=True, exist_ok=True)
    Table(dict(WAVE=WAVE_GRID, FLUX=stack, FLUX_ERR=stack_err, N_GAL=n_pix)).write(
        OUT['jades_stack_fits'], overwrite=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.step(WAVE_GRID, stack, where='mid', color='k', label=f'JADES mean (N={res["N"]})')
    ax.fill_between(WAVE_GRID, stack - stack_err, stack + stack_err, alpha=0.3, color='gray')
    wf = np.linspace(*FIT_WINDOW, 1000)
    ax.plot(wf, sii_model(wf, *best), color='crimson', label=f'Fit R={R:.3f}')
    ax.set_xlim(*FIT_WINDOW)
    vis = (WAVE_GRID >= FIT_WINDOW[0]) & (WAVE_GRID <= FIT_WINDOW[1]) \
        & np.isfinite(stack) & np.isfinite(stack_err)
    if vis.any():   # 誤差帯の min/max に上下 5% の余白
        ymin, ymax = np.min((stack - stack_err)[vis]), np.max((stack + stack_err)[vis])
        pad = 0.05 * ((ymax - ymin) if ymax > ymin else max(abs(ymin), 1e-30))
        ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_xlabel('Rest-frame wavelength [Å]')
    ax.set_ylabel('Mean flux')
    # ax.legend(fontsize=12)
    finish(fig, OUT['jades_stack_png'])
    return res


# =====================================================================
# 作図
# =====================================================================
def plot_overview(loc, jd):
    """M*–O/H 平面の全銀河と local の M* 範囲の帯。"""
    fig, ax = plt.subplots(figsize=(6, 6))
    for mass, oh, color, label, s, a in [
            (loc['mass'], loc['oh'], 'gray', 'SDSS', 3, 0.15),
            (jd['mass'], jd['oh'], 'tab:red', 'JADES', 10, 0.8)]:
        ok = np.isfinite(mass) & np.isfinite(oh) & (mass > 0) & (oh > 0)
        ax.scatter(mass[ok], oh[ok], s=s, color=color, alpha=a, rasterized=True,
                   label=f'{label} (N={ok.sum():,})')
    ax.axvspan(*LOCAL_MASS_RANGE, color='tab:blue', alpha=0.08, label=r'Local $M_\ast$ range')
    ax.set_xlabel(r'$\log(M_\ast/M_\odot)$')
    ax.set_ylabel(r'$12+\log(\mathrm{O/H})$')
    # ax.legend(fontsize=16, markerscale=2)
    ax.tick_params(which='major', length=10)
    finish(fig, OUT['overview_png'])


def plot_distributions(loc, jd):
    """比較に使う local / JADES の M* と O/H の分布 (確率密度) と中央値。
    local は M* 範囲内の有効な銀河 (O/H ビン内の flux カット前)。"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    print('[分布] 中央値 local / JADES (差)')
    for ax, key, label in [(axes[0], 'mass', r'$\log(M_\ast/M_\odot)$'),
                           (axes[1], 'oh', r'$12+\log(\mathrm{O/H})$')]:
        lv, jv = loc[key][loc['use']], jd[key][jd['sel']]
        lo = np.floor(min(lv.min(), jv.min()) / DIST_BIN_WIDTH) * DIST_BIN_WIDTH
        hi = np.ceil(max(lv.max(), jv.max()) / DIST_BIN_WIDTH) * DIST_BIN_WIDTH
        edges = np.arange(lo, hi + DIST_BIN_WIDTH, DIST_BIN_WIDTH)
        for v, c, lab in [(lv, 'gray', f'Local (N={len(lv):,})'), (jv, 'firebrick', f'JADES (N={len(jv)})')]:
            ax.hist(v, bins=edges, density=True, histtype='step', color=c, lw=2, label=lab)
            ax.axvline(np.median(v), color=c, ls='--', lw=1.5)
        ax.set_xlabel(label)
        ax.set_ylabel('Probability density')
        print(f'  {key:4s}: {np.median(lv):.3f} / {np.median(jv):.3f} ({np.median(jv) - np.median(lv):+.3f})')
    # axes[0].legend(fontsize=16)
    finish(fig, OUT['dist_png'])


def plot_jades_histograms(jd):
    """JADES 比較サンプルの M* と O/H のヒストグラム。M* の要約を CSV に保存して返す。"""
    summary = {}
    for key, width, color, path in [('mass', JADES_MASS_HIST_BIN_WIDTH, 'tab:red', OUT['jades_mass_hist_png']),
                                    ('oh', JADES_OH_HIST_BIN_WIDTH, 'crimson', OUT['jades_oh_hist_png'])]:
        v = jd[key][jd['sel']]
        p16, med, p84 = np.percentile(v, [16, 50, 84])
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.axvspan(p16, p84, color=color, alpha=0.12, label=f'16–84%: {p16:.2f}–{p84:.2f}')
        ax.hist(v, bins=bin_edges(v, width), histtype='step', color=color, lw=2,
                label=f'JADES (N={len(v)})')
        ax.axvline(med, color='k', ls='--', lw=1.5, label=f'Median: {med:.2f}')
        ax.set_xlabel(r'$\log_{10}(M_\ast/M_\odot)$' if key == 'mass' else r'$12+\log_{10}(\mathrm{O/H})$')
        ax.set_ylabel('Number of galaxies')
        # ax.legend(fontsize=12)
        finish(fig, path)
        if key == 'mass':
            summary = dict(mass_column=JADES_MASS_COL, mass_N=len(v),
                           logM_p16=p16, logM_median=med, logM_p84=p84,
                           logM_std=np.std(v, ddof=1) if len(v) > 1 else np.nan,
                           logM_min=v.min(), logM_max=v.max(),
                           mass_interval_type='sample_percentiles_not_measurement_error')
    save_csv(pd.DataFrame([summary]), OUT['jades_mass_csv'])
    return summary


def plot_completeness(edges):
    """光度カットの completeness (O/H ビンごと, 全 M*)。
    分母 = PARENT_V2, 分子 = PARENT_V2 & LUM_OK_V2 (build_sample.py 11c と同じ定義)。"""
    flags = Table.read(FLAGS_PATH, hdu=1)
    oh, oh16, oh84 = (column(flags, c) for c in ('oh_MEDIAN', 'oh_P16', 'oh_P84'))
    oh_ok = (np.isfinite(oh) & np.isfinite(oh16) & np.isfinite(oh84)
             & (oh16 > 0) & (oh16 <= oh) & (oh <= oh84))
    parent = np.asarray(flags['PARENT_V2'], bool) & oh_ok
    lum = np.asarray(flags['LUM_OK_V2'], bool)
    n_den = np.histogram(oh[parent], edges)[0]
    n_num = np.histogram(oh[parent & lum], edges)[0]
    with np.errstate(invalid='ignore', divide='ignore'):
        frac = n_num / n_den
        ferr = np.sqrt(frac * (1 - frac) / n_den)
    centers = 0.5 * (edges[1:] + edges[:-1])
    save_csv(pd.DataFrame(dict(oh_center=centers, f=frac, f_err=ferr, N_den=n_den, N_num=n_num)),
             OUT['compl_csv'])
    v = n_den >= COMPL_NMIN
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.errorbar(centers[v], frac[v], yerr=ferr[v], fmt='o-', color='k', ms=5, lw=1.5)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel(r'$12+\log_{10}(\mathrm{O/H})$')
    ax.set_ylabel(r'$N(L_{6731} > L_{\rm min})\,/\,N$')
    finish(fig, OUT['compl_png'])


def plot_ne_vs_oh(res, jres):
    """log n_e vs 12+log(O/H): local の O/H ビンと JADES の 1 点。"""
    fig, ax = plt.subplots(figsize=(6, 6))
    ok = res['status_mean'].eq('ok').to_numpy()
    x = res['oh_median'].to_numpy()
    y = np.log10(res['ne_mean'].to_numpy())
    ax.errorbar(x[ok], y[ok],
                xerr=np.vstack([x - res['oh_p16_median'], res['oh_p84_median'] - x])[:, ok],
                yerr=np.vstack([y - np.log10(res['ne_mean_p16']), np.log10(res['ne_mean_p84']) - y])[:, ok],
                fmt='s', ms=10, color='k', capsize=3,
                label=rf'Local (${LOCAL_MASS_RANGE[0]}\leq\log M_\ast<{LOCAL_MASS_RANGE[1]}$)')
    if (~ok).any():
        print(f'[local] 区間が定まらない {(~ok).sum()} ビンは描いていません (CSV 参照)')
    xlim, ylim = [x.min() - 0.1, x.max() + 0.1], [1.25, 3]
    if jres['status'] == 'ok':
        jx, jy = jres['oh_median'], np.log10(jres['ne'])
        jlo, jhi = np.log10(jres['ne_lo']), np.log10(jres['ne_hi'])
        ax.errorbar(jx, jy, xerr=[[jx - jres['oh_p16']], [jres['oh_p84'] - jx]],
                    yerr=[[jy - jlo], [jhi - jy]], fmt='s', ms=10, color='crimson',
                    capsize=4, label=f'JADES (N={jres["N"]})')
        xlim = [min(xlim[0], jres['oh_p16'] - 0.1), max(xlim[1], jres['oh_p84'] + 0.1)]
        ylim = [min(ylim[0], jlo - 0.1), max(ylim[1], jhi + 0.1)]
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(r'$12+\log(\mathrm{O/H})$')
    ax.set_ylabel(r'$\log n_{\mathrm{e}}$')
    # ax.legend(fontsize=12)
    finish(fig, OUT['ne_png'])


def plot_sfr_map(loc):
    """local の M*–O/H 平面を, 各ビンの log SFR 中央値で色付け (全 M*, flux カットなし)。"""
    m, oh, sfr = loc['mass'], loc['oh'], loc['sfr']
    ok = np.isfinite(m) & np.isfinite(oh) & np.isfinite(sfr) & (m > 0) & (oh > 0) & (sfr > -90)
    bins = [bin_edges(m[ok], MAP_MASS_BIN), bin_edges(oh[ok], MAP_OH_BIN)]
    med, _, _, _ = binned_statistic_2d(m[ok], oh[ok], sfr[ok], statistic='median', bins=bins)
    counts = np.histogram2d(m[ok], oh[ok], bins=bins)[0]
    color = np.ma.masked_where((counts < MAP_NMIN) | ~np.isfinite(med), med)
    fig, ax = plt.subplots(figsize=(10, 10))
    pc = ax.pcolormesh(bins[0], bins[1], color.T, shading='flat', cmap='viridis', rasterized=True)
    fig.colorbar(pc, ax=ax, pad=0.02).set_label(r'Median $\log_{10}(\mathrm{SFR}\,/\,M_\odot\,\mathrm{yr}^{-1})$')
    ax.set_xlim(8, 12)
    ax.set_ylim(7.5, 9.5)
    ax.set_xlabel(r'$\log_{10}(M_\ast/M_\odot)$')
    ax.set_ylabel(r'$12+\log_{10}(\mathrm{O/H})$')
    finish(fig, OUT['sfr_map_png'])


# =====================================================================
# main
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n-mc', type=int, default=N_MC)
    args = parser.parse_args()
    if args.n_mc < 2:
        parser.error('--n-mc must be at least 2')
    rng = np.random.default_rng(SEED)

    # 1. 読み込みと比較サンプルの選択
    loc = load_local()
    jd = load_jades()

    # 2. 比較領域と分布の確認
    plot_overview(loc, jd)
    plot_distributions(loc, jd)
    jades_mass_summary = plot_jades_histograms(jd)

    # 3. local: O/H ビンごとの n_e と completeness
    edges = bin_edges(loc['oh'][loc['use']], OH_BIN_WIDTH)
    plot_completeness(edges)
    res = stack_local(loc, edges, args.n_mc, rng)

    # 4. JADES: スタックの n_e
    jres = stack_jades(jd)
    jres.update(jades_mass_summary)
    save_csv(pd.DataFrame([jres]), OUT['jades_csv'])

    # 5. 結果の図
    plot_ne_vs_oh(res, jres)
    plot_sfr_map(loc)
    print(f'出力: {CSV_DIR} , {FIG_DIR}')


if __name__ == '__main__':
    main()
