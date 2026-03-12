#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
スクリプトの概要:
JADESスペクトルスタックを作成します。
SFRのビンごとにスタックを作成します。

使用方法:
    JADES_spectra_stack_v1.py [オプション]

著者: A. M.
作成日: 2026-03-12

参考文献:
    - PEP 8:                  https://peps.python.org/pep-0008/
    - PEP 257 (Docstring規約): https://peps.python.org/pep-0257/
    - Python公式ドキュメント:    https://docs.python.org/ja/3/
"""

import os
import glob
import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.interpolate import interp1d

# ============================
# 設定
# ============================

csv_file = "results/JADES/JADES_DR3/data_from_Nishigaki/jades_info_with_HA_plus_logSFR.csv"

spec_dir = "results/JADES/JADES_DR3/JADES_DR3_full_spectra"

gratings = {
    "G140M": f"{spec_dir}/JADES_DR3_G140M",
    "G235M": f"{spec_dir}/JADES_DR3_G235M",
    "G395M": f"{spec_dir}/JADES_DR3_G395M"
}

wave_grid = np.arange(6500, 6900, 0.5)

# # SFR bins
# sfr_bins = np.arange(-2, 4, 1)
# SFR >= 0 のみ使用

# ============================
# CSV読み込み
# ============================

df = pd.read_csv(csv_file)

df = df[df["z_spec"].notna()]
df = df[df["HA_6563_flux"].notna()]
df = df[df["log10_SFR_hb"].notna()]

# SFR >= 0 のみ
df = df[df["log10_SFR_hb"] >= 0]

# equal-number bin (1分割、とりあえず全体を1ビンにする)
df["SFR_bin"] = pd.qcut(df["log10_SFR_hb"], q=1)

groups = df.groupby("SFR_bin")

print("Total objects:", len(df))
print("SFR bins:")
print(df["SFR_bin"].value_counts())

# ============================
# スペクトル読み込み
# ============================

def read_spectrum(file):

    with fits.open(file) as h:

        data = h["EXTRACT1D"].data

        wave = data["WAVELENGTH"] * 1e4
        flux = data["FLUX"]
        err  = data["FLUX_ERR"]

    return wave, flux, err

# ============================
# rest-frame
# ============================

def restframe(wave, flux, err, z):

    wave = wave / (1 + z)
    flux = flux * (1 + z)
    err  = err  * (1 + z)

    return wave, flux, err

# ============================
# 補間
# ============================

def resample(wave, flux, err):

    f = interp1d(wave, flux, bounds_error=False, fill_value=np.nan)
    e = interp1d(wave, err , bounds_error=False, fill_value=np.nan)

    return f(wave_grid), e(wave_grid)

# ============================
# スタック
# ============================

def stack(fluxes, errs):

    fluxes = np.array(fluxes)
    errs   = np.array(errs)

    # error floor
    floor = 0.05 * np.nanmedian(errs)
    errs  = np.sqrt(errs**2 + floor**2)

    errs = np.clip(errs, 1e-20, None)

    w = 1 / errs**2

    flux_stack = np.nansum(fluxes*w, axis=0) / np.nansum(w, axis=0)
    err_stack  = np.sqrt(1 / np.nansum(w, axis=0))

    return flux_stack, err_stack

# ============================
# SFR bin number per grating
# ============================

sfr_bins_per_grating = {
    "G140M": 1,
    "G235M": 2,
    "G395M": 2
}

# ============================
# メイン
# ============================

for gr in gratings:

    print("\n==========")
    print("GRATING:", gr)

    spec_path = gratings[gr]

    # ===== gratingに入る銀河だけ選択 =====

    if gr == "G140M":
        df_gr = df[(df["z_spec"] > 0.5) & (df["z_spec"] < 1.7)]

    elif gr == "G235M":
        df_gr = df[(df["z_spec"] > 1.5) & (df["z_spec"] < 3.6)]

    elif gr == "G395M":
        df_gr = df[(df["z_spec"] > 3.3) & (df["z_spec"] < 6.7)]

    if len(df_gr) < 5:
        print("Too few galaxies")
        continue

    df_gr = df_gr.copy()

    # ===== equal-number SFR bins =====
    q = sfr_bins_per_grating[gr]
    # 1 bin ≈ 最低5銀河
    if len(df_gr) < q*5:
        q = max(1, len(df_gr)//5)

    df_gr["SFR_bin"] = pd.qcut(df_gr["log10_SFR_hb"], q=q, duplicates="drop")
    print("SFR bins:")
    print(df_gr["SFR_bin"].value_counts().sort_index())

    groups = df_gr.groupby("SFR_bin")

    # ============================
    # SFR bin loop
    # ============================

    for sfr_bin, group in groups:

        if len(group) < 5:
            continue

        b0 = sfr_bin.left
        b1 = sfr_bin.right

        print(f"\nSFR bin {b0:.2f} to {b1:.2f} (N={len(group)})")

        flux_list = []
        err_list  = []

        sfr_used = []
        z_used   = []

        # ============================
        # 銀河ループ
        # ============================

        for _, row in group.iterrows():

            nid = int(row["NIRSpec_ID"])
            z   = row["z_spec"]
            ha  = row["HA_6563_flux"]

            sid = f"{nid:08d}"

            pattern = f"{spec_path}/*{sid}*_x1d.fits"
            files = glob.glob(pattern)

            if len(files) == 0:
                continue

            file = files[0]

            try:

                wave, flux, err = read_spectrum(file)

                wave, flux, err = restframe(wave, flux, err, z)

                # ============================
                # Hα check
                # ============================

                if ha <= 0:
                    print("Bad Halpha:", nid)
                    continue

                if ha > 1e-15 or ha < 1e-20:
                    print("Suspicious Halpha:", nid, ha)

                flux = flux / ha
                err  = err  / ha

                flux_i, err_i = resample(wave, flux, err)

                flux_list.append(flux_i)
                err_list.append(err_i)

                # small error check
                if np.nanmin(err_i) < 1e-25:
                    print("Very small error pixel in", nid)

                sfr_used.append(row["log10_SFR_hb"])
                z_used.append(z)

            except Exception as e:
                print("read error:", nid, e)
                continue

        print("spectra used:", len(flux_list))

        if len(flux_list) < 3:
            continue

        # ============================
        # SFR statistics
        # ============================

        sfr_used = np.array(sfr_used)

        sfr_mean = np.mean(sfr_used)
        sfr_std  = np.std(sfr_used)
        sfr_err  = sfr_std / np.sqrt(len(sfr_used))

        print(f"mean logSFR = {sfr_mean:.3f} ± {sfr_err:.3f}")

        # ============================
        # redshift statistics
        # ============================

        z_used = np.array(z_used)

        z_mean = np.mean(z_used)
        z_std  = np.std(z_used)
        z_err  = z_std / np.sqrt(len(z_used))

        print(f"mean z = {z_mean:.3f} ± {z_err:.3f}")

        # ============================
        # coverage check
        # ============================

        flux_array = np.array(flux_list)

        coverage = np.sum(np.isfinite(flux_array), axis=0)

        print("coverage min:", np.min(coverage))
        print("coverage max:", np.max(coverage))

        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(wave_grid, coverage)
        plt.xlabel("Rest wavelength")
        plt.ylabel("N spectra")
        plt.title(f"{gr} coverage")
        plt.show()

        # ============================
        # individual spectra check
        # ============================

        plt.figure()

        for f in flux_list[:20]:
            plt.plot(wave_grid, f, alpha=0.2)

        plt.xlabel("Rest wavelength")
        plt.ylabel("Flux")
        plt.title(f"{gr} individual spectra")
        plt.show()

        # ============================
        # stack
        # ============================

        flux_stack, err_stack = stack(flux_list, err_list)

        outname = f"stack_{gr}_SFR_{b0:.2f}_{b1:.2f}.txt"

        np.savetxt(
            outname,
            np.column_stack([wave_grid, flux_stack, err_stack]),
            header="wave_A flux err"
        )

print("\nDone.")