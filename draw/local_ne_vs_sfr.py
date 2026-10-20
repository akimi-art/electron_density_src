"""Local [SII] density versus log SFR at fixed stellar mass.

Run from the repository root: .venv/bin/python electron_density_src/draw/local_ne_vs_sfr.py
Ratios are ratios of summed fluxes, not averages of individual densities.
Uncertainties propagate independent Gaussian flux errors only (no bootstrap).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.table import Table
from matplotlib.patches import Rectangle
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

LOCAL_SAMPLE = "results/fits/sdss_sample_v2_zlt0.1879_Lgt1e+39_Ka03_Reu.fits"
JADES_SAMPLE = "results/JADES/sample/jades_all_with_flags.fits"
MASS_CENTER = 9.5
MASS_HALF_WIDTH = 0.1
SFR_BIN_WIDTH = 0.2  # non-overlapping bins in log10(SFR / Msun yr^-1)
MIN_GALAXIES = 10
ERROR_MAX_FACTOR = 100.0  # same error quality criterion as overlap_box_ne.py
TE = 1e4
N_MC = 10000
SEED = 42
OUT_DIR = Path("results/comparison")
FIG_DIR = Path("results/figure/comparison")


def f64(col):
    return np.ma.filled(np.ma.asarray(col).astype(float), np.nan)


def main():
    if MASS_HALF_WIDTH <= 0 or SFR_BIN_WIDTH <= 0 or MIN_GALAXIES < 1:
        raise ValueError("Bin widths must be positive and MIN_GALAXIES >= 1")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    tab = Table.read(LOCAL_SAMPLE)
    mass, sfr, f16, f31, e16, e31 = [f64(tab[c]) for c in
        ("sm_MEDIAN", "sfr_MEDIAN", "SII_6717_FLUX", "SII_6731_FLUX",
         "SII_6717_FLUX_ERR", "SII_6731_FLUX_ERR")]
    mlo, mhi = MASS_CENTER - MASS_HALF_WIDTH, MASS_CENTER + MASS_HALF_WIDTH
    mass_mask = np.isfinite(mass) & (mass >= mlo) & (mass < mhi)
    valid = (mass_mask & np.isfinite(sfr) & np.isfinite(f16) & np.isfinite(f31)
             & np.isfinite(e16) & np.isfinite(e31) & (e16 >= 0) & (e31 >= 0))
    if not valid.any():
        raise ValueError("No valid galaxies in the mass bin")
    # Determine error thresholds once across the mass-selected sample, not per SFR bin.
    lim16, lim31 = ERROR_MAX_FACTOR * np.median(e16[valid]), ERROR_MAX_FACTOR * np.median(e31[valid])
    bad_error = valid & ((e16 > lim16) | (e31 > lim31))
    valid &= ~bad_error
    if not valid.any():
        raise ValueError("No galaxies remain after flux error cuts")
    print(f"Mass [{mlo:g}, {mhi:g}): {mass_mask.sum():,}; usable={valid.sum():,}; "
          f"extreme flux errors excluded={bad_error.sum()}")
    # Integer cell indices avoid cumulative floating point drift in 0.2 dex edges.
    ids = np.zeros(len(tab), dtype=int)
    ids[valid] = np.floor(sfr[valid] / SFR_BIN_WIDTH).astype(int)
    atom = pn.Atom("S", 2)
    # Finite numerical inversion interval, also used to identify limits explicitly.
    ne_min, ne_max = 1e-2, 1e6
    ratio_max = float(atom.getEmissivity(tem=TE, den=ne_min, wave=6716)
                      / atom.getEmissivity(tem=TE, den=ne_min, wave=6731))
    ratio_min = float(atom.getEmissivity(tem=TE, den=ne_max, wave=6716)
                      / atom.getEmissivity(tem=TE, den=ne_max, wave=6731))

    def density(r):
        if not np.isfinite(r) or not ratio_min < r < ratio_max:
            return np.nan
        return float(atom.getTemDen(int_ratio=r, tem=TE, wave1=6716, wave2=6731,
                                    start_x=np.log10(ne_min), end_x=np.log10(ne_max)))

    rng = np.random.default_rng(SEED)
    rows = []
    for cell in range(ids[valid].min(), ids[valid].max() + 1):
        mask = valid & (ids == cell)
        n = int(mask.sum())
        row = dict(logSFR_lo=cell * SFR_BIN_WIDTH, logSFR_hi=(cell + 1) * SFR_BIN_WIDTH,
                   N=n, logSFR_med=np.median(sfr[mask]) if n else np.nan,
                   logM_med=np.median(mass[mask]) if n else np.nan,
                   R=np.nan, R_p16=np.nan, R_p84=np.nan,
                   ne=np.nan, ne_lo=np.nan, ne_hi=np.nan, status="too_few")
        if n >= MIN_GALAXIES:
            a, b = f16[mask].sum(), f31[mask].sum()
            if a <= 0 or b <= 0:
                row["status"] = "nonpositive_flux_sum"
            else:
                # Sum of independent Gaussian fluxes is exactly Gaussian.
                amc = rng.normal(a, np.sqrt(np.sum(e16[mask] ** 2)), N_MC)
                bmc = rng.normal(b, np.sqrt(np.sum(e31[mask] ** 2)), N_MC)
                ok = (amc > 0) & (bmc > 0)
                r = a / b
                if ok.sum() < N_MC * 0.9:
                    row.update(R=r, status="unstable_flux_ratio")
                else:
                    r16, r84 = np.percentile(amc[ok] / bmc[ok], [16, 84])
                    row.update(R=r, R_p16=r16, R_p84=r84, ne=density(r),
                               ne_lo=density(r84), ne_hi=density(r16),
                               status="measured" if ratio_min < r < ratio_max else
                               ("low_density_limit" if r >= ratio_max else "high_density_limit"))
        rows.append(row)
    df = pd.DataFrame(rows)
    tag = f"logM{mlo:.1f}-{mhi:.1f}_sfrbin{SFR_BIN_WIDTH:g}"
    csv_path = OUT_DIR / f"local_ne_vs_sfr_{tag}.csv"
    df.to_csv(csv_path, index=False)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 16})
    fig, (ax, counts) = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})
    for row in rows:
        x = row["logSFR_med"]
        if row["status"] == "measured" and np.isfinite(row["ne"]):
            y = np.log10(row["ne"])
            # Clip error bounds to the finite numerical inversion interval.
            lo = row["ne_lo"] if np.isfinite(row["ne_lo"]) else ne_min
            hi = row["ne_hi"] if np.isfinite(row["ne_hi"]) else ne_max
            ax.errorbar(x, y, xerr=[[x-row["logSFR_lo"]], [row["logSFR_hi"]-x]],
                        yerr=[[max(0, y-np.log10(lo))], [max(0, np.log10(hi)-y)]], fmt="o", color="k", capsize=3)
        elif row["status"] == "low_density_limit" and np.isfinite(row["ne_hi"]):
            ax.errorbar(x, np.log10(row["ne_hi"]), yerr=-np.log10(0.6),
                        uplims=True, fmt="v", color="tab:blue")
        elif row["status"] == "high_density_limit" and np.isfinite(row["ne_lo"]):
            ax.errorbar(x, np.log10(row["ne_lo"]), yerr=np.log10(1.4),
                        lolims=True, fmt="^", color="tab:red")

    # High-zの値（x=1.062 −0.011 / +0.188, y=2.40-0.04+0.04）
    high_z_x = np.array([1.062])
    high_z_y = np.array([2.40])
    high_z_x_errors = np.array([0.011])
    high_z_y_errors = np.array([0.04])
    ax.errorbar(high_z_x, high_z_y, xerr=high_z_x_errors, yerr=high_z_y_errors, fmt='o', capsize=4, color='tab:red')
    ax.set_yscale("linear")
    ax.set_ylabel(r"$\log(n_e / {\rm cm}^{-3})$")
    centers = (df.logSFR_lo + df.logSFR_hi) / 2
    counts.bar(centers, df.N, width=SFR_BIN_WIDTH * 0.9, color="gray")
    counts.axhline(MIN_GALAXIES, color="tab:red", ls="--", lw=1)
    counts.set_ylabel("N")
    counts.set_xlabel(r"$\log({\rm SFR}/M_\odot\,{\rm yr}^{-1})$")
    for axis in (ax, counts):
        axis.tick_params(direction="in", top=True, right=True)
    fig.tight_layout()
    fig_path = FIG_DIR / f"local_ne_vs_sfr_{tag}.png"
    fig.savefig(fig_path, dpi=200)
    print(f"[DONE] {csv_path}\n[DONE] {fig_path}")
    # Selection footprint: gray = parent catalog, blue = mass slice,
    # orange boxes = bins with enough usable galaxies for the flux-ratio analysis.
    with plt.rc_context({"font.size": 16, "axes.labelsize": 20,
                         "axes.titlesize": 18, "xtick.labelsize": 16,
                         "ytick.labelsize": 16, "xtick.major.size": 8,
                         "ytick.major.size": 8, "xtick.minor.size": 4,
                         "ytick.minor.size": 4}):
        footprint, region_ax = plt.subplots(figsize=(10, 8))
        finite = np.isfinite(mass) & np.isfinite(sfr)
        region_ax.scatter(mass[finite], sfr[finite], s=2, color="gray", alpha=0.15,
                          rasterized=True, label="All local galaxies")
        region_ax.axvspan(mlo, mhi, color="tab:blue", alpha=0.12,
                          label=rf"Mass slice: {mlo:.1f}–{mhi:.1f}")
        eligible = np.zeros(len(tab), dtype=bool)
        first = True
        for row in rows:
            if row["N"] < MIN_GALAXIES:
                continue
            eligible |= valid & (sfr >= row["logSFR_lo"]) & (sfr < row["logSFR_hi"])
            region_ax.add_patch(Rectangle(
                (mlo, row["logSFR_lo"]), mhi-mlo, SFR_BIN_WIDTH,
                facecolor="none", edgecolor="darkorange", linewidth=1.5,
                label=f"SFR bins with N ≥ {MIN_GALAXIES}" if first else None,
                zorder=4))
            first = False
        region_ax.scatter(mass[eligible], sfr[eligible], s=3, color="tab:blue",
                          alpha=0.3, rasterized=True,
                          label=f"Usable galaxies in these bins (N={eligible.sum():,})")
        highz = Table.read(JADES_SAMPLE)
        highz_mass = f64(highz["N26_logM"])
        highz_sfr = f64(highz["N26_logSFR_hb"])
        highz_mask = (np.asarray(np.ma.filled(highz["SAMPLE_SSFR"], False), bool)
                      & np.isfinite(highz_mass) & np.isfinite(highz_sfr))
        region_ax.scatter(highz_mass[highz_mask], highz_sfr[highz_mask],
                          s=20, color="tab:red", zorder=5,
                          label=f"High-z (SAMPLE_SSFR; N={highz_mask.sum():,})")
        region_ax.axvline(mlo, color="tab:blue", linewidth=1.5)
        region_ax.axvline(mhi, color="tab:blue", linewidth=1.5)
        region_ax.set_xlim(6, 12)
        region_ax.set_ylim(min(-3, df.logSFR_lo.min()-0.2),
                           max(3, df.logSFR_hi.max()+0.2))
        region_ax.set_xlabel(r"$\log(M_*/M_\odot)$")
        region_ax.set_ylabel(r"$\log({\rm SFR}/M_\odot\,{\rm yr}^{-1})$")
        footprint.tight_layout()
        selection_path = FIG_DIR / f"local_mstar_sfr_selection_{tag}.png"
        footprint.savefig(selection_path, dpi=200)
        print(f"[DONE] {selection_path}")
    plt.show()


if __name__ == "__main__":
    main()
