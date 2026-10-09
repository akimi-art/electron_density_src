#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ne_overlap_box.py

M*-SFR 重なり箱内の local / high-z サンプルで [SII]6716/6731 比 → n_e を求め,
log10(n_e) vs z に線形軸でプロットする。

  local  : カタログ flux の和の比 R = ΣF6716 / ΣF6731 (flux 誤差による MC)
  high-z : JADES 個別スペクトルを静止系でスタック (平均) し,
           [SII] ダブレットをフィットして R を求める (MC で誤差)

※ 手法が local と high-z で異なる (カタログ和 vs スペクトルスタック)。今回は目を瞑る。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import pyneb as pn
import os


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


# =========================================================
# 設定
# =========================================================
TAG       = "logM9.0-10.0_logSFR0.0-2.0"
LOCAL_BOX = f"results/comparison/local_box_{TAG}.fits"
HIGHZ_BOX = f"results/comparison/highz_box_{TAG}.fits"

OUT_CSV   = f"results/comparison/ne_overlap_box_{TAG}.csv"
OUT_FIG   = "results/figure/comparison/ne_vs_z_overlap_box.png"
OUT_STACK = f"results/figure/comparison/highz_stack_SII_{TAG}.png"
OUT_STACK_FITS = f"results/comparison/highz_stack_{TAG}.fits"

N_LOCAL_MC = 1000      # local flux 測定誤差の MC
LOCAL_ERROR_MAX_FACTOR = 100.0  # 各線の誤差中央値に対する上限 (極端な不良測定の除去)
N_MC   = 500           # high-z MC (元のスタックコードと同じ)
SEED   = 42

# スタック (元コードと同じ)
WAVE_GRID = np.arange(6500, 6900, 0.5)   # Å, rest
SII_1 = 6718.29        # vacuum
SII_2 = 6732.67        # vacuum

# フィット窓 (Hα+[NII]6583 を避ける)
FIT_MIN, FIT_MAX = 6650.0, 6800.0

TE = 1e4
S2 = pn.Atom("S", 2)
C_KMS = 299792.458

rng = np.random.default_rng(SEED)

for p in [OUT_CSV, OUT_FIG, OUT_STACK, OUT_STACK_FITS]:
    os.makedirs(os.path.dirname(p), exist_ok=True)


# =========================================================
# 共通関数
# =========================================================
def ne_from_ratio(r):
    """R = I(6716)/I(6731) → n_e [cm^-3]。低密度極限を超えると NaN。"""
    if not np.isfinite(r) or r <= 0:
        return np.nan
    ne = S2.getTemDen(int_ratio=r, tem=TE, wave1=6716, wave2=6731)
    return float(ne) if np.isfinite(ne) else np.nan


def summarize(r_best, r_dist):
    """R の best と分布 → R, n_e の 16/50/84 をまとめる"""
    r16, r84 = np.nanpercentile(r_dist, [16, 84])
    ne_best = ne_from_ratio(r_best)
    # R が大きいほど n_e は小さい → R の p84 が n_e の下側, p16 が上側
    ne_lo = ne_from_ratio(r84)
    ne_hi = ne_from_ratio(r16)
    return dict(R=r_best, R_p16=r16, R_p84=r84,
                ne=ne_best, ne_lo=ne_lo, ne_hi=ne_hi,
                upper_limit=bool(~np.isfinite(ne_best)))


# =========================================================
# 1. local : カタログ flux の和の比 + 測定誤差による MC
# =========================================================
loc = Table.read(LOCAL_BOX)
f16 = np.asarray(loc["SII_6717_FLUX"], float)
f31 = np.asarray(loc["SII_6731_FLUX"], float)
e16 = np.asarray(loc["SII_6717_FLUX_ERR"], float)
e31 = np.asarray(loc["SII_6731_FLUX_ERR"], float)
z_loc = np.asarray(loc["Z"], float)
# 両線で共通の有効サンプルを使う。負の flux は測定値として保持する。
valid_loc = (np.isfinite(f16) & np.isfinite(f31)
             & np.isfinite(e16) & np.isfinite(e31)
             & (e16 >= 0) & (e31 >= 0) & np.isfinite(z_loc))
print(f"[local ] invalid flux/error/z excluded: {np.sum(~valid_loc)}/{len(loc)}")
if not np.any(valid_loc):
    raise ValueError("local sample has no valid flux measurements")
# S/N では選別せず、極端な誤差を持つ測定を両線からまとめて除外する。
error_limit16 = LOCAL_ERROR_MAX_FACTOR * np.median(e16[valid_loc])
error_limit31 = LOCAL_ERROR_MAX_FACTOR * np.median(e31[valid_loc])
bad_error_loc = valid_loc & ((e16 > error_limit16) | (e31 > error_limit31))
print(f"[local ] extreme errors excluded: {np.sum(bad_error_loc)} "
      f"(6717 error > {error_limit16:.3g} or 6731 error > {error_limit31:.3g})")
# 除外対象と理由を保存し、後からカタログの測定を確認できるようにする。
excluded_loc = ~valid_loc | bad_error_loc
audit = loc[excluded_loc].to_pandas()
audit["exclusion_reason"] = np.where(bad_error_loc[excluded_loc],
                                     "extreme_flux_error", "invalid_flux_error_or_z")
audit.to_csv(f"results/comparison/local_flux_rejected_{TAG}.csv", index=False)
valid_loc &= ~bad_error_loc
f16, f31 = f16[valid_loc], f31[valid_loc]
e16, e31 = e16[valid_loc], e31[valid_loc]
z_loc = z_loc[valid_loc]
n_loc = len(f16)
if n_loc == 0 or np.sum(f31) <= 0:
    raise ValueError("local sample requires valid flux errors and a positive summed 6731 flux")

R_loc = np.sum(f16) / np.sum(f31)
# 各銀河・各線の測定誤差は独立な Gaussian と仮定 (共分散は未使用)。
# 銀河の再抽出はせず、固定したサンプルの flux のみを揺らす。
R_loc_mc = np.full(N_LOCAL_MC, np.nan)
for k in range(N_LOCAL_MC):
    sum16 = rng.normal(f16, e16).sum()
    sum31 = rng.normal(f31, e31).sum()
    if sum16 > 0 and sum31 > 0:
        R_loc_mc[k] = sum16 / sum31

res_loc = summarize(R_loc, R_loc_mc)
res_loc.update(sample="local", method="catalog_sum_flux_mc", N=n_loc,
               z_med=np.median(z_loc),
               z_p16=np.percentile(z_loc, 16), z_p84=np.percentile(z_loc, 84))
print(f"[local ] N={n_loc}  R={R_loc:.4f}  ne={res_loc['ne']:.1f}")


# =========================================================
# 2. high-z : スペクトルスタック (元コードのパターン)
# =========================================================
def read_spectrum(path):
    with fits.open(path) as h:
        d = h["EXTRACT1D"].data
        wave = np.asarray(d["WAVELENGTH"], float) * 1e4   # μm → Å
        flux = np.asarray(d["FLUX"], float)
        err  = np.asarray(d["FLUX_ERR"], float)
    return wave, flux, err


def to_rest_grid(wave, flux, err, z):
    w_rest = wave / (1 + z)
    fi = interp1d(w_rest, flux, bounds_error=False, fill_value=np.nan)
    ei = interp1d(w_rest, err,  bounds_error=False, fill_value=np.nan)
    return fi(WAVE_GRID), ei(WAVE_GRID)


def mask_artifact(flux, err):
    med = np.nanmedian(flux)
    std = np.nanstd(flux)
    bad = flux < med - 5 * std
    flux = flux.copy(); err = err.copy()
    flux[bad] = np.nan
    err[bad]  = np.nan
    return flux, err


hz = Table.read(HIGHZ_BOX)
spec_files = [str(s).strip() for s in hz["SPEC_FILE"]]
z_hz = np.asarray(hz["z_Spec"], float)
n_hz = len(hz)

F = np.full((n_hz, WAVE_GRID.size), np.nan)
E = np.full((n_hz, WAVE_GRID.size), np.nan)
for i, (path, z) in enumerate(zip(spec_files, z_hz)):
    w, f, e = read_spectrum(path)
    fr, er = to_rest_grid(w, f, e, z)
    F[i], E[i] = mask_artifact(fr, er)

# [SII] に寄与している銀河数 (確認用)
j1 = np.argmin(np.abs(WAVE_GRID - SII_1))
j2 = np.argmin(np.abs(WAVE_GRID - SII_2))
n_at_sii = int(np.sum(np.isfinite(F[:, j1]) & np.isfinite(F[:, j2])))
print(f"[high-z] N={n_hz}  [SII] 両線で有効な銀河: {n_at_sii}")

# 平均スタック
stack = np.nanmean(F, axis=0)

# MC
E_mc = np.where(np.isfinite(E), E, 0.0)
mc = rng.normal(F[:, :, None], E_mc[:, :, None], size=(n_hz, WAVE_GRID.size, N_MC))
stack_mc = np.nanmean(mc, axis=0)                       # (Npix, N_MC)
p16, p84 = np.nanpercentile(stack_mc, [16, 84], axis=1)
stack_err = (p84 - p16) / 2


# ---------------------------------------------------------
# [SII] ダブレットのフィット
#   2 Gaussian: 中心は真空波長, 共通の速度シフト dv と共通の幅 sigma
#   + 定数の連続光 (傾き c1 は 0 に固定)
#   パラメータは各線の積分 flux → R = F1/F2
# ---------------------------------------------------------
def doublet(w, F1, F2, dv, sig, c0):
    m1 = SII_1 * (1 + dv / C_KMS)
    m2 = SII_2 * (1 + dv / C_KMS)
    g1 = F1 / (np.sqrt(2 * np.pi) * sig) * np.exp(-0.5 * ((w - m1) / sig) ** 2)
    g2 = F2 / (np.sqrt(2 * np.pi) * sig) * np.exp(-0.5 * ((w - m2) / sig) ** 2)
    return g1 + g2 + c0


fit_win = (WAVE_GRID >= FIT_MIN) & (WAVE_GRID <= FIT_MAX)


def fit_doublet(y, yerr=None):
    m = fit_win & np.isfinite(y)
    if yerr is not None:
        m &= np.isfinite(yerr) & (yerr > 0)
    w, yy = WAVE_GRID[m], y[m]
    if yy.size <= 5:
        raise ValueError("[SII] fit requires more than 5 valid pixels")
    # flux と誤差を同時に正規化し、パラメータの桁の差を小さくする。
    scale = np.max(np.abs(yy))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("[SII] fit requires a finite, nonzero flux scale")
    yy = yy / scale
    ee = None if yerr is None else yerr[m] / scale
    cont = np.nanmedian(yy)
    peak = np.nanmax(yy) - cont
    amp0 = max(peak, 1e-30) * np.sqrt(2 * np.pi) * 2.0
    p0 = [amp0, amp0, 0.0, 2.0, cont]
    lo = [0.0, 0.0, -300.0, 0.3, -np.inf]
    hi = [np.inf, np.inf, 300.0, 15.0, np.inf]
    popt, _ = curve_fit(doublet, w, yy, p0=p0, bounds=(lo, hi),
                        sigma=ee,
                        absolute_sigma=yerr is not None, maxfev=20000)
    # flux 関連のパラメータだけ元の単位に戻す。
    popt[[0, 1, 4]] *= scale
    return popt



popt = fit_doublet(stack, stack_err)
R_hz = popt[0] / popt[1]

R_hz_mc = np.full(N_MC, np.nan)
for k in range(N_MC):
    try:
        pk = fit_doublet(stack_mc[:, k], stack_err)
        R_hz_mc[k] = pk[0] / pk[1]
    except RuntimeError:
        pass
print(f"[high-z] MC fit 失敗: {np.sum(~np.isfinite(R_hz_mc))}/{N_MC}")

res_hz = summarize(R_hz, R_hz_mc)
res_hz.update(sample="highz", method="spectral_stack", N=n_hz,
              z_med=np.median(z_hz),
              z_p16=np.percentile(z_hz, 16), z_p84=np.percentile(z_hz, 84))
print(f"[high-z] R={R_hz:.4f}  ne={res_hz['ne']:.1f}  "
      f"(dv={popt[2]:.1f} km/s, sigma={popt[3]:.2f} Å)")

# スタックを保存
Table(dict(WAVE=WAVE_GRID, FLUX=stack, FLUX_ERR=stack_err,
           N_GAL=np.sum(np.isfinite(F), axis=0))).write(OUT_STACK_FITS, overwrite=True)

# スタック + フィットの確認図
fig, ax = plt.subplots(figsize=(12, 6))
ax.step(WAVE_GRID, stack, where="mid", color="k", lw=1, label=f"mean stack (N={n_hz})")
ax.fill_between(WAVE_GRID, stack - stack_err, stack + stack_err,
                step="mid", color="gray", alpha=0.4)
wf = np.linspace(FIT_MIN, FIT_MAX, 1000)
ax.plot(wf, doublet(wf, *popt), color="r", lw=1.2, label=f"fit  R={R_hz:.3f}")
for l in (SII_1, SII_2):
    ax.axvline(l, color="b", ls=":", lw=0.8)
# ax.axvspan(WAVE_GRID[0], FIT_MIN, color="gray", alpha=0.1)
# ax.axvspan(FIT_MAX, WAVE_GRID[-1], color="gray", alpha=0.1)
mask_fit = (WAVE_GRID >= FIT_MIN) & (WAVE_GRID <= FIT_MAX)
ax.set_xlabel("Rest-frame wavelength [Å]")
ax.set_ylabel("Flux (mean)")
ax.set_xlim(FIT_MIN, FIT_MAX )
ax.set_ylim(-0.5 * np.nanstd(stack[mask_fit]), 1.5 * np.nanmax(stack[mask_fit]))
for sp in ax.spines.values():
    sp.set_linewidth(2)
fig.tight_layout()
fig.savefig(OUT_STACK, dpi=200)
plt.close(fig)


# =========================================================
# 3. 保存 & log10(n_e) vs z (両軸とも線形)
# =========================================================
df = pd.DataFrame([res_loc, res_hz])
df.to_csv(OUT_CSV, index=False)
print(df.to_string())

fig, ax = plt.subplots(figsize=(8, 8))
for r, col in [(res_loc, "k"), (res_hz, "r")]:
    x = r["z_med"]
    xerr = [[x - r["z_p16"]], [r["z_p84"] - x]]
    lab = f"{r['sample']} ({r['method']}, N={r['N']})"
    if not r["upper_limit"]:
        y = np.log10(r["ne"])
        ylo = y - np.log10(r["ne_lo"]) if np.isfinite(r["ne_lo"]) and r["ne_lo"] > 0 else 0.0
        yhi = np.log10(r["ne_hi"]) - y if np.isfinite(r["ne_hi"]) and r["ne_hi"] > 0 else 0.0
        ax.errorbar(x, y, xerr=xerr, yerr=[[ylo], [yhi]],
                    ms=10, mew=2, fmt="o", color=col, capsize=10, label=lab)
    else:
        # best の R が低密度極限超え → R_p16 から上限
        ne_limit = r["ne_hi"]
        if np.isfinite(ne_limit) and ne_limit > 0:
            y = np.log10(ne_limit)
            ax.errorbar(x, y, xerr=xerr, yerr=[[-np.log10(0.6)], [0]], uplims=True,
                        ms=10, mew=2, fmt="v", color=col, capsize=10, label=lab + " (upper limit)")
        else:
            print(f"[{r['sample']}] n_e 上限も定まらない (R_p16 も低密度極限超え)")

ax.set_xscale("linear"); ax.set_yscale("linear")
ax.set_xlabel(r"$z$")
ax.set_ylabel(r"$\log_{10}(n_e\,/\,\mathrm{cm}^{-3})$")
# ax.set_xlim(0, 10)
# ax.set_ylim(0, 4)
for sp in ax.spines.values():
    sp.set_linewidth(2)
fig.tight_layout()
fig.savefig(OUT_FIG, dpi=200)
plt.close(fig)
print("saved:", OUT_CSV, OUT_FIG, OUT_STACK, OUT_STACK_FITS)
