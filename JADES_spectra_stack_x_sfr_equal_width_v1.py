#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
スクリプトの概要:
JADESスペクトルスタックを作成します。
z, SFRを基に、スペクトルを複数のビンに分割してスタックします。
スタック方法を新たに3つ（median, median (Ha norm), weighted mean)
追加しました。

使用方法:
    JADES_spectra_stack_x_sfr_equal_width_v1.py [オプション]

著者: A. M.
作成日: 2026-06-26
最終更新日: 2026-06-26

参考文献:
    - PEP 8:                  https://peps.python.org/pep-0008/
    - PEP 257 (Docstring規約): https://peps.python.org/pep-0257/
    - Python公式ドキュメント:    https://docs.python.org/ja/3/
"""

# ============================
# z-bin stacking only version
# （機能は維持しつつ、後半の z-bin 制御部分のみ残した版）
# ============================

import glob
import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from astropy.cosmology import Planck18

# ============================
# SETTINGS
# ============================

csv_file = "results/JADES/JADES_DR3/data_from_Nishigaki/jades_info_with_HA_plus_logSFR_with_Reff.csv" # 変更

spec_dir = "results/JADES/JADES_DR3/JADES_DR3_full_spectra"

gratings = {
    "G140M": f"{spec_dir}/JADES_DR3_G140M",
    "G235M": f"{spec_dir}/JADES_DR3_G235M",
    "G395M": f"{spec_dir}/JADES_DR3_G395M"
}

wave_grid = np.arange(6500, 6900, 0.5)

# ← ここだけ変えればOK
# 試金石
# sigma_bins = [
#     (-3.0, -2.0),
#     (-2.0, -1.0),
#     (-1.0, 0.0),
#     (0.0, 1.0),
# ]
# 2ビンでも可
sfr_bins = [
    (0.0, 1.0),
    (1.0, 2.0),
]


# ============================
# CSV
# ============================

df = pd.read_csv(csv_file)

df = df[df["z_spec"].notna()]
df = df[df["HA_6563_flux"].notna()]
df = df[df["logSFR_hb"].notna()]
# df = df[df["logSFR_hb"] >= 0] # なぜ入っている?

print("usable rows after CSV filtering:", len(df))

# ============================
# FUNCTIONS
# ============================

def read_spectrum(file):

    with fits.open(file) as h:

        data = h["EXTRACT1D"].data

        wave = data["WAVELENGTH"] * 1e4
        flux = data["FLUX"] 
        err  = data["FLUX_ERR"] 

    return wave, flux, err


def restframe(wave, flux, err, z):

    wave = wave / (1 + z)

    return wave, flux, err


def resample(wave, flux, err):

    f = interp1d(wave, flux, bounds_error=False, fill_value=np.nan)
    e = interp1d(wave, err , bounds_error=False, fill_value=np.nan)

    return f(wave_grid), e(wave_grid)


def coverage_ok(wave, flux):

    ha_region  = (wave > 6550) & (wave < 6575)
    sii_region = (wave > 6705) & (wave < 6740)
    cont_region = (wave > 6600) & (wave < 6650)

    if np.sum(np.isfinite(flux[ha_region])) < 2:
        return False

    if np.sum(np.isfinite(flux[sii_region])) < 2:
        return False

    if np.sum(np.isfinite(flux[cont_region])) < 3:
        return False

    return True


def mask_artifact(flux):

    med = np.nanmedian(flux)
    std = np.nanstd(flux)

    mask = flux < med - 5 * std

    flux = flux.copy()
    flux[mask] = np.nan

    return flux


def stack(fluxes, errs):

    fluxes = np.array(fluxes)
    errs   = np.array(errs)

    floor = 0.05 * np.nanmedian(errs)

    errs = np.sqrt(errs**2 + floor**2)

    w = 1 / errs**2

    flux_stack = np.nansum(fluxes * w, axis=0) / np.nansum(w, axis=0)

    err_stack = np.sqrt(1 / np.nansum(w, axis=0))

    return flux_stack, err_stack


def weighted_mean(values, err_lo, err_hi):

    values = np.array(values)

    err_lo = np.array(err_lo)
    err_hi = np.array(err_hi)

    # 非対称誤差 → 対称化
    sigma = 0.5 * (err_lo + err_hi)

    mask = (
        np.isfinite(values)
        &
        np.isfinite(sigma)
        &
        (sigma > 0)
    )

    values = values[mask]
    sigma = sigma[mask]

    w = 1 / sigma**2

    mean = np.sum(w * values) / np.sum(w)

    err = np.sqrt(1 / np.sum(w))

    return mean, err

# 追加1（medianスタック関数）
def median_stack(fluxes):
    fluxes = np.array(fluxes)
    return np.nanmedian(fluxes, axis=0)

# 追加1（median誤差関数）
def bootstrap_median_error(fluxes, nboot=500):

    fluxes = np.array(fluxes)

    n_spec = fluxes.shape[0]
    n_wave = fluxes.shape[1]

    med_boot = np.zeros((nboot, n_wave))

    for i in range(nboot):

        idx = np.random.randint(0, n_spec, n_spec)

        sample = fluxes[idx]

        med_boot[i] = np.nanmedian(sample, axis=0)

    return np.nanstd(med_boot, axis=0)

# ============================
# MAIN
# ============================

used_items_all = []

for gr in gratings:

    print("\nGRATING:", gr)

    spec_path = gratings[gr]

    if gr == "G140M":
        df_gr = df[(df["z_spec"] > 0.5) & (df["z_spec"] < 1.7)]

    elif gr == "G235M":
        df_gr = df[(df["z_spec"] > 1.5) & (df["z_spec"] < 3.6)]

    elif gr == "G395M":
        df_gr = df[(df["z_spec"] > 3.3) & (df["z_spec"] < 6.7)]

    print("candidates:", len(df_gr))

    ha_vals = df_gr["HA_6563_flux"].values
    ha_vals = ha_vals[np.isfinite(ha_vals) & (ha_vals > 0)]

    ha_p995 = np.nanpercentile(ha_vals, 99.5)

    for _, row in df_gr.iterrows():

        nid = int(row["NIRSpec_ID"])
        z   = row["z_spec"]
        ha  = row["HA_6563_flux"]

        if (not np.isfinite(ha)) or (ha <= 0) or (ha > ha_p995):
            continue

        sid = f"{nid:08d}"

        pattern = f"{spec_path}/*{sid}*_x1d.fits"

        files = glob.glob(pattern)

        if len(files) == 0:
            continue

        try:

            wave, flux, err = read_spectrum(files[0])

            wave, flux, err = restframe(wave, flux, err, z)

            if not coverage_ok(wave, flux):
                continue

            flux = mask_artifact(flux)

            # 元のまま保存
            flux_raw = flux.copy()
            err_raw  = err.copy()

            # Hα正規化版も作る
            flux_norm = flux / ha
            err_norm  = err  / ha

            # resampleは両方やる
            flux_i_raw,  err_i_raw  = resample(wave, flux_raw,  err_raw)
            flux_i_norm, err_i_norm = resample(wave, flux_norm, err_norm)


            if (
                not np.isfinite(flux_i_raw).any()
                or
                not np.isfinite(flux_i_norm).any()
            ):
                continue


            # ↓ 追加
            logSFR_hb = row["logSFR_hb"]

            logSFR_hb_err_lo = row["err1_logSFR_hb"]
            logSFR_hb_err_hi = row["err2_logSFR_hb"]

            used_items_all.append({
            
                "id": sid,

                # --- raw（非normalize） ---
                "flux_raw": flux_i_raw,
                "err_raw": err_i_raw,

                # --- normalized ---
                "flux_norm": flux_i_norm,
                "err_norm": err_i_norm,

                "logSFR_hb":        logSFR_hb,
                "logSFR_hb_err_lo": logSFR_hb_err_lo,
                "logSFR_hb_err_hi": logSFR_hb_err_hi,
            })

        # try ブロック内で発生したほぼすべてのエラー（例外）をキャッチ, 
        # 発生したエラーの具体的な内容（メッセージなど）が変数 e に代入される
        except Exception as e:
            print("ERROR:", e)
            continue


# ============================
# SFR-bin split
# ============================
if len(used_items_all) == 0:

    print("No usable spectra.")

else:

    # =====================================
    # use SFR as binning variable
    # =====================================

    used_sfr_all = np.array([
        it["logSFR_hb"]
        for it in used_items_all
    ])

    valid_mask = np.isfinite(used_sfr_all)

    used_sfr_all = used_sfr_all[valid_mask]

    used_items_valid = [
        used_items_all[i]
        for i in range(len(used_items_all))
        if valid_mask[i]
    ]

    N = len(used_sfr_all)

    print("\nTotal usable spectra:", N)

    # =====================================
    # histogram
    # =====================================

    plt.figure(figsize=(6,4))

    plt.hist(
        used_sfr_all,
        bins=60,
        color="0.7",
        edgecolor="black"
    )

    plt.xlabel(r'$\log SFR$')
    plt.ylabel("count")

    plt.tight_layout()
    save_hist_path = "results/JADES/figure/hist_sfr_JADES.png"
    plt.savefig(f"{save_hist_path}")
    print(f"Saved as {save_hist_path}.")
    plt.show()

    print("median =", np.nanmedian(used_sfr_all))
    print("std =",       np.nanstd(used_sfr_all))

    # =====================================
    # stack each SFR bin
    # =====================================

    for b, (lo, hi) in enumerate(sfr_bins):

        # outlierを除去するために、以下のIDを除外します。
        # 確実に弾いてよいもの
        # Ha, SII領域に欠損(NaN)がある
        # 明らかなデータ落ち
        # SII波長域がスペクトル端にかかる
        # 極端な単一ピクセルスパイク
        bad_ids = [
            "00024958", 
            "00025356", 
            "00051209", 
            "00082961",

            "00004504", 
            "00028139", 

            "00045967", 

        ]

        selected = [
        
            it

            for it in used_items_valid

            if (
                (it["logSFR_hb"] >= lo)
                and
                (it["logSFR_hb"] < hi)
                and
                (it["id"] not in bad_ids)
            )
        ]

        if len(selected) == 0:

            print(
                f"logSFR_hb [{lo},{hi}) : empty"
            )

            continue


        sfr_vals = np.array([
            it["logSFR_hb"]
            for it in selected
        ])

        # raw
        flux_list_raw = [it["flux_raw"] for it in selected]
        err_list_raw  = [it["err_raw"]  for it in selected]

        # normalized
        flux_list_norm = [it["flux_norm"] for it in selected]
        err_list_norm  = [it["err_norm"]  for it in selected]



        # =====================================
        # weighted mean SFR
        # =====================================

        sfr_err_lo = [
            it["logSFR_hb_err_lo"]
            for it in selected
        ]

        sfr_err_hi = [
            it["logSFR_hb_err_hi"]
            for it in selected
        ]

        sfr_mean, sfr_err = weighted_mean(
            sfr_vals,
            sfr_err_lo,
            sfr_err_hi
        )

        print(
            f"\nlogSFR_hb [{lo},{hi})"
        )

        print(
            f"N = {len(selected)}"
        )

        print(
            f"logSFR_hb = "
            f"{sfr_mean:.3f} ± {sfr_err:.3f}"
        )

        print(
            f"range = "
            f"[{np.min(sfr_vals):.3f}, "
             f"{np.max(sfr_vals):.3f}]"
        )

        # =========================
        # weighted stack
        # =========================

        # raw（非正規化）
        flux_stack_w_raw, err_stack_w_raw = stack(
            flux_list_raw,
            err_list_raw
        )

        # normalized
        flux_stack_w_norm, err_stack_w_norm = stack(
            flux_list_norm,
            err_list_norm
        )

        # =========================
        # median stack
        # =========================

        # raw, normalized
        flux_stack_m_raw  = np.nanmedian(np.array(flux_list_raw), axis=0)
        flux_stack_m_norm = np.nanmedian(np.array(flux_list_norm), axis=0)
        err_stack_m_raw   = bootstrap_median_error(flux_list_raw)
        err_stack_m_norm  = bootstrap_median_error(flux_list_norm)


        # =========================
        # output names
        # =========================

        outname_base = (
            "results/JADES/JADES_DR3/spectra/"
            f"stack_sfr_{lo:+.1f}_{hi:+.1f}"
        )

        # =========================
        # save
        # =========================

        # --- weighted raw ---
        np.savetxt(
            outname_base + "_w_raw.txt",
            np.column_stack([wave_grid, flux_stack_w_raw, err_stack_w_raw]),
            header=f"weighted raw | logSFR_hb=[{lo},{hi}) | N={len(selected)}"
        )

        # --- weighted normalized ---
        np.savetxt(
            outname_base + "_w_norm.txt",
            np.column_stack([wave_grid, flux_stack_w_norm, err_stack_w_norm]),
            header=f"weighted normalized | logSFR_hb=[{lo},{hi}) | N={len(selected)}"
        )

        # --- median raw ---
        np.savetxt(
            outname_base + "_m_raw.txt",
            np.column_stack([wave_grid, flux_stack_m_raw, err_stack_m_raw]),
            header=f"median raw | logSFR_hb=[{lo},{hi}) | N={len(selected)}"
        )

        # --- median normalized ---
        np.savetxt(
            outname_base + "_m_norm.txt",
            np.column_stack([wave_grid, flux_stack_m_norm, err_stack_m_norm]),
            header=f"median normalized | logSFR_hb=[{lo},{hi}) | N={len(selected)}"
        )

        print("saved:", outname_base)


print("\nDone.")