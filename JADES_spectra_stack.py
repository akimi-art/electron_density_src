#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
スクリプトの概要:
JADESスペクトルスタックを作成します。
まず、単純にgratingsごとにスタックを作成します。

使用方法:
    JADES_spectra_stack.py [オプション]

著者: A. M.
作成日: 2026-03-11

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

# rest-frame grid (Å)
wave_grid = np.arange(6500, 6900, 0.5)

# ============================
# カタログ読み込み
# ============================

df = pd.read_csv(csv_file)

df = df[df["z_spec"].notna()]
df = df[df["HA_6563_flux"].notna()]

print("objects used:", len(df))

# ============================
# スペクトル読み込み関数
# ============================

def read_spectrum(file):

    with fits.open(file) as h:

        data = h["EXTRACT1D"].data

        wave = data["WAVELENGTH"] * 1e4   # μm → Å
        flux = data["FLUX"]
        err  = data["FLUX_ERR"]

    return wave, flux, err

# ============================
# rest-frame変換
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

    w = 1 / errs**2

    flux_stack = np.nansum(fluxes*w, axis=0) / np.nansum(w, axis=0)
    err_stack  = np.sqrt(1 / np.nansum(w, axis=0))

    return flux_stack, err_stack

# ============================
# メインループ
# ============================

stack_results = {}

for gr in gratings:

    print("\nProcessing", gr)

    spec_path = gratings[gr]

    flux_list = []
    err_list  = []

    for _, row in df.iterrows():

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

            flux = flux / ha
            err  = err  / ha

            flux_i, err_i = resample(wave, flux, err)

            flux_list.append(flux_i)
            err_list.append(err_i)

        except:
            continue

    print("spectra used:", len(flux_list))

    if len(flux_list) == 0:
        continue

    flux_stack, err_stack = stack(flux_list, err_list)

    stack_results[gr] = (flux_stack, err_stack)

# ============================
# 保存
# ============================

for gr in stack_results:

    flux, err = stack_results[gr]

    np.savetxt(
        f"stack_{gr}.txt",
        np.column_stack([wave_grid, flux, err]),
        header="wave_A flux err"
    )

print("\nDone.")