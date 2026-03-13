#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
スクリプトの概要:
JADESスペクトルスタックを作成します。
SFRのビンごとにスタックを作成します。

使用方法:
    JADES_spectra_stack_v2.py [オプション]

著者: A. M.
作成日: 2026-03-13

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
import matplotlib.pyplot as plt
from matplotlib import gridspec

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

# ============================
# CSV
# ============================

df = pd.read_csv(csv_file)

df = df[df["z_spec"].notna()]
df = df[df["HA_6563_flux"].notna()]
df = df[df["log10_SFR_hb"].notna()]
df = df[df["log10_SFR_hb"] >= 0]



# ===== デバッグ: CSV段階の件数確認 =====
print("CSV total:", len(df))
print("  with z_spec:", df["z_spec"].notna().sum())
print("  with HA_6563_flux:", df["HA_6563_flux"].notna().sum())
print("  with log10_SFR_hb:", df["log10_SFR_hb"].notna().sum())

df0 = pd.read_csv(csv_file)
df1 = df0[df0["z_spec"].notna()]
df2 = df1[df1["HA_6563_flux"].notna()]
df3 = df2[df2["log10_SFR_hb"].notna()]
df4 = df3[df3["log10_SFR_hb"] >= 0]
print("After all CSV filters:", len(df4))



# ============================
# functions
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


def stack(fluxes, errs):

    fluxes = np.array(fluxes)
    errs   = np.array(errs)

    floor = 0.05 * np.nanmedian(errs)
    errs  = np.sqrt(errs**2 + floor**2)

    w = 1 / errs**2

    flux_stack = np.nansum(fluxes*w, axis=0) / np.nansum(w, axis=0)
    err_stack  = np.sqrt(1 / np.nansum(w, axis=0))

    return flux_stack, err_stack


# coverage check

def coverage_ok(wave, flux):

    ha_region = (wave>6550) & (wave<6575)
    sii_region = (wave>6705) & (wave<6740)
    cont_region = (wave>6600) & (wave<6650)

    if np.sum(np.isfinite(flux[ha_region])) < 2:
        return False

    if np.sum(np.isfinite(flux[sii_region])) < 2:
        return False

    if np.sum(np.isfinite(flux[cont_region])) < 3:
        return False
    
    return True


# artifact mask

def mask_artifact(flux):

    med = np.nanmedian(flux)
    std = np.nanstd(flux)

    mask = flux < med - 5*std
    flux[mask] = np.nan

    return flux


# ============================
# main
# ============================

for gr in gratings:
    print("\nGRATING:", gr)
    spec_path = gratings[gr]
    if gr == "G140M":
        df_gr = df[(df["z_spec"] > 0.5) & (df["z_spec"] < 1.7)]
    elif gr == "G235M":
        df_gr = df[(df["z_spec"] > 1.5) & (df["z_spec"] < 3.6)]
    elif gr == "G395M":
        df_gr = df[(df["z_spec"] > 3.3) & (df["z_spec"] < 6.7)]

    print(f"  candidates after z-range filter: {len(df_gr)}")
    median_sfr = np.nanmedian(df_gr["log10_SFR_hb"])
    print(f"  median log10(SFR) = {median_sfr:.2f}")

    # ===== デバッグ: ふるい落としの理由別カウンタ =====
    cnt_flux_range_bad = 0
    cnt_no_file = 0
    cnt_coverage_ng = 0
    cnt_resample_allnan = 0
    cnt_exception = 0

    bad_flux_ids = []
    no_file_ids = []
    coverage_ng_ids = []
    resample_nan_ids = []
    exception_ids = []

    flux_list = []
    err_list  = []

    # ループ前（各gratingごと）でパーセンタイルを一度計算しておくとよい
    ha_vals_gr = df_gr["HA_6563_flux"].values
    # 正の値のみ対象（NaN/<=0 を除く）
    ha_vals_gr = ha_vals_gr[np.isfinite(ha_vals_gr) & (ha_vals_gr > 0)]
    if ha_vals_gr.size > 0:
        ha_p995 = np.nanpercentile(ha_vals_gr, 99.5)  # 99.5% 以上を外れ値候補
    else:
        ha_p995 = np.inf  # デフォルトではカットしない

    for _, row in df_gr.iterrows():
        nid = int(row["NIRSpec_ID"])
        z   = row["z_spec"]
        ha  = row["HA_6563_flux"]

        # === 修正後のHAチェック ===
        # 1) 有限 & 正
        if (not np.isfinite(ha)) or (ha <= 0):
            cnt_flux_range_bad += 1
            if len(bad_flux_ids) < 3:
                bad_flux_ids.append((nid, ha))
                print(f"    skip[HA invalid]: id={nid}, HA={ha}")
            continue

        # 2) （任意）上位外れ値カット（ユニット非依存）
        if ha > ha_p995:
            cnt_flux_range_bad += 1
            if len(bad_flux_ids) < 3:
                bad_flux_ids.append((nid, ha))
                print(f"    skip[HA outlier >P99.5]: id={nid}, HA={ha:.3e}, P99.5={ha_p995:.3e}")
            continue


        sid = f"{nid:08d}"

        pattern = f"{spec_path}/*{sid}*_x1d.fits"
        files = glob.glob(pattern)

        # ファイルが見つからない理由の可視化
        if len(files) == 0:
            cnt_no_file += 1
            if len(no_file_ids) < 3:
                no_file_ids.append(sid)
                print(f"    skip[no file]: pattern={pattern}")
            continue

        try:
            wave, flux, err = read_spectrum(files[0])
            wave, flux, err = restframe(wave, flux, err, z)

            # カバレッジNGの理由を表示（波長レンジ等）
            if not coverage_ok(wave, flux):
                cnt_coverage_ng += 1
                if len(coverage_ng_ids) < 3:
                    coverage_ng_ids.append(sid)
                    print(f"    skip[coverage NG]: id={sid}, restframe wave [{np.nanmin(wave):.1f}, {np.nanmax(wave):.1f}]")
                continue

            flux = mask_artifact(flux)
            flux = flux / ha
            err  = err  / ha

            flux_i, err_i = resample(wave, flux, err)

            # resample後が全NaNの場合
            if not np.isfinite(flux_i).any():
                cnt_resample_allnan += 1
                if len(resample_nan_ids) < 3:
                    resample_nan_ids.append(sid)
                    print(f"    skip[resample all-NaN]: id={sid}")
                continue

            flux_list.append(flux_i)
            err_list.append(err_i)

        except Exception as e:
            cnt_exception += 1
            if len(exception_ids) < 3:
                exception_ids.append((sid, str(e)))
                print(f"    skip[exception]: id={sid}, err={e}")
            continue

    print("spectra used:", len(flux_list))
    print("  dropped by HA range   :", cnt_flux_range_bad)
    print("  dropped by no file    :", cnt_no_file)
    print("  dropped by coverage   :", cnt_coverage_ng)
    print("  dropped by resampleNaN:", cnt_resample_allnan)
    print("  dropped by exception  :", cnt_exception)


    flux_array = np.array(flux_list)

    # ============================
    # individual spectra
    # ============================

    n_spec = flux_array.shape[0]
    ncol = 5
    nrow = int(np.ceil(n_spec/ncol))

    fig = plt.figure(figsize=(3*ncol,1.5*nrow))
    gs = gridspec.GridSpec(nrow,ncol)
    gs.update(hspace=0.0,wspace=0.0)

    for i in range(n_spec):

        ax = fig.add_subplot(gs[i])

        ax.plot(wave_grid,flux_array[i],lw=1)

        ax.axvline(6563,color="red")
        ax.axvline(6716,color="red")
        ax.axvline(6731,color="red")

        ax.set_xlim(6500,6900)
        ax.set_yticks([])

        if i < (nrow-1)*ncol:
            ax.set_xticklabels([])

    plt.suptitle(f"{gr} individual spectra")
    plt.show()

    # ============================
    # spectra matrix
    # ============================

    plt.figure(figsize=(6,8))

    plt.imshow(
        flux_array,
        aspect="auto",
        vmin=np.nanpercentile(flux_array,5),
        vmax=np.nanpercentile(flux_array,95)
    )

    plt.xlabel("wavelength pixel")
    plt.ylabel("galaxy index")
    plt.title(f"{gr} spectra matrix")
    plt.colorbar()
    plt.show()

    # ============================
    # stack
    # ============================

    flux_stack, err_stack = stack(flux_list, err_list)

    outname = f"stack_{gr}.txt"

    np.savetxt(
        outname,
        np.column_stack([wave_grid, flux_stack, err_stack]),
        header="wave flux err"
    )

print("\nDone")