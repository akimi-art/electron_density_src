#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
スクリプトの概要:
JADESスペクトルスタックを作成します。
z, ΣSFRを基に、スペクトルを複数のビンに分割してスタックします。
スタック方法を新たに3つ（median, median (Ha norm), weighted mean)
追加しました。

v1との変更点: 
* mean, Ha normalized meanの結果を追加
* 誤差の評価をMCで統一（重要）


使用方法:
    JADES_spectra_stack_x_sigma_sfr_equal_width_v2.py [オプション]

著者: A. M.
作成日: 2026-07-09
最終更新日: 2026-07-09

参考文献:
    - PEP 8:                  https://peps.python.org/pep-0008/
    - PEP 257 (Docstring規約): https://peps.python.org/pep-0257/
    - Python公式ドキュメント:    https://docs.python.org/ja/3/
"""


import glob
import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from astropy.cosmology import Planck18

# -----------------------
# 軸の設定
# -----------------------
# 軸の設定
plt.rcParams.update({
    # --- 図全体 ---
    "figure.figsize": (12, 6),       # 図サイズ
    "font.size": 20,                 # 全体フォントサイズ
    "axes.labelsize": 24,            # 軸ラベルのサイズ
    "axes.titlesize": 20,            # タイトルのサイズ
    "axes.grid": False,              # グリッドOFF

    # --- 目盛り設定 (ticks) ---
    "xtick.direction": "in",         # x軸目盛りの向き
    "ytick.direction": "in",         # y軸目盛りの向き
    "xtick.top": True,               # 上にも目盛り
    "ytick.right": True,             # 右にも目盛り

    # 主目盛り（major ticks）
    "xtick.major.size": 20,          # 長さ
    "ytick.major.size": 20,
    "xtick.major.width": 2,          # 太さ
    "ytick.major.width": 2,

    # 補助目盛り（minor ticks）
    "xtick.minor.visible": True,     # 補助目盛りON
    "ytick.minor.visible": True,
    "xtick.minor.size": 8,           # 長さ
    "ytick.minor.size": 8,
    "xtick.minor.width": 1.5,        # 太さ
    "ytick.minor.width": 1.5,

    # --- 目盛りラベル ---
    "xtick.labelsize": 20,           # x軸ラベルサイズ
    "ytick.labelsize": 20,           # y軸ラベルサイズ

    # --- フォント ---
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
})

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
sigma_sfr_bins = [
    # (-1.0, -0.5), 
    # (-0.5, 0.0), 
    (0.0, 0.5), 
]


z_bins = [
    # (0.5, 3.0),
    # (3.0, 8.0),
    (0.5, 8.0)
]



# ============================
# CSV
# ============================

df = pd.read_csv(csv_file)

# 重複を消す
df = df.drop_duplicates(
    subset="NIRSpec_ID",
    keep="first"
)

df = df[df["z_spec"].notna()]
df = df[df["HA_6563_flux"].notna()]
df = df[df["logSFR_hb"].notna()]

print("usable rows after CSV filtering:", len(df))

# ============================
# FUNCTIONS
# ============================
N_MC = 500
rng = np.random.default_rng()

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

def median_stack(fluxes):
    fluxes = np.array(fluxes)
    return np.nanmedian(fluxes, axis=0)

# 追加
def mean_stack(fluxes):
    fluxes = np.array(fluxes)
    return np.nanmean(fluxes, axis=0)

def compute_log_sigma_sfr(
    logSFR,
    err_lo,
    err_hi,
    Re_arcsec,
    eRe_arcsec,
    z
):

    if (
        (not np.isfinite(logSFR))
        or
        (not np.isfinite(Re_arcsec))
        or
        (not np.isfinite(eRe_arcsec))
        or
        (Re_arcsec <= 0)
    ):

        return np.nan, np.nan, np.nan

    # --------------------------------
    # arcsec -> kpc
    # --------------------------------

    kpc_per_arcsec = (
        Planck18.kpc_proper_per_arcmin(z).value
        / 60.0
    )

    Re_kpc = Re_arcsec * kpc_per_arcsec
    eRe_kpc = eRe_arcsec * kpc_per_arcsec

    # --------------------------------
    # central value
    # --------------------------------

    logSigma = (
        logSFR
        -
        np.log10(2 * np.pi * Re_kpc**2)
    )

    # --------------------------------
    # SFR uncertainty
    # --------------------------------

    sfr_sigma = 0.5 * (
        err_lo + err_hi
    )

    # --------------------------------
    # Re uncertainty
    # --------------------------------

    re_sigma = (
        2 * eRe_kpc
        /
        (Re_kpc * np.log(10))
    )

    total_sigma = np.sqrt(
        sfr_sigma**2
        +
        re_sigma**2
    )

    return (
        logSigma,
        total_sigma,
        total_sigma
    )

# ============================
# MAIN
# ============================

used_items_all = []

used_ids = set()   # ←追加(二重防御)

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

        # 同じNIRSpec_IDは1回だけ使う
        if sid in used_ids:
            continue
        
        used_ids.add(sid)

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

            # ↓ 追加
            logSigma, logSigma_err_lo, logSigma_err_hi = (
                compute_log_sigma_sfr(
                    row["logSFR_hb"],
                    row["err1_logSFR_hb"],
                    row["err2_logSFR_hb"],
                    # row["ReffOpt"], # 必要に応じて変更
                    # row["e_ReffOpt"],
                    row["ReffUV"], # 必要に応じて変更
                    row["e_ReffUV"],
                    z
                )
            )

            used_items_all.append({

                "id": sid,
                "z": z,
                # --- raw（非normalize） ---
                "flux_raw": flux_i_raw,
                "err_raw": err_i_raw,

                # --- normalized ---
                "flux_norm": flux_i_norm,
                "err_norm": err_i_norm,

                "logSigma": logSigma,
                "logSigma_err_lo": logSigma_err_lo,
                "logSigma_err_hi": logSigma_err_hi,

            })

        # try ブロック内で発生したほぼすべてのエラー（例外）をキャッチ, 
        # 発生したエラーの具体的な内容（メッセージなど）が変数 e に代入される
        except Exception as e:
            print("ERROR:", e)
            continue


# ============================
# sigma_sfr-bin split
# ============================
if len(used_items_all) == 0:

    print("No usable spectra.")

else:

    # =====================================
    # use sigma_sfr as binning variable
    # =====================================

    used_sigma_sfr_all = np.array([
        it["logSigma"]
        for it in used_items_all
    ])

    valid_mask = np.isfinite(used_sigma_sfr_all)

    used_sigma_sfr_all = used_sigma_sfr_all[valid_mask]

    used_items_valid = [
        used_items_all[i]
        for i in range(len(used_items_all))
        if valid_mask[i]
    ]

    N = len(used_sigma_sfr_all)

    print("\nTotal usable spectra:", N)

    # =====================================
    # histogram
    # =====================================

    fig, ax = plt.subplots(figsize=(6,4))

    ax.hist(
        used_sigma_sfr_all,
        bins=15,
        color="0.7",
        edgecolor="black"
    )

    ax.set_xlabel(r"$\log(\Sigma_{\rm SFR})\ [{\rm M_\odot\ yr^{-1}\ kpc^{-2}}]$")
    ax.set_ylabel("count")

    for spine in ax.spines.values():
        spine.set_linewidth(2)

    plt.tight_layout()
    save_hist_path_sigma_sfr = "results/JADES/figure/hist_sigma_sfr_JADES.png"
    plt.savefig(f"{save_hist_path_sigma_sfr}")
    print(f"Saved as {save_hist_path_sigma_sfr}.")
    plt.show()

    print("median =", np.nanmedian(used_sigma_sfr_all))
    print("std =", np.nanstd(used_sigma_sfr_all))

    used_z_all = np.array([
        it["z"]
        for it in used_items_valid
    ])

    plt.figure(figsize=(6,4))

    plt.hist(
        used_z_all,
        bins=20,
        color="0.7",
        edgecolor="black"
    )

    plt.xlabel("z")
    plt.ylabel("count")

    plt.tight_layout()

    save_hist_path_z = (
        "results/JADES/figure/hist_z_JADES.png"
    )

    plt.savefig(save_hist_path_z)
    print(f"Saved as {save_hist_path_z}.")
    plt.show()

    print(
        "z median =",
        np.nanmedian(used_z_all)
    )

    print(
        "z std =",
        np.nanstd(used_z_all)
    )

    # =====================================
    # stack each sigma_sfr bin
    # =====================================

    for b_sigma_sfr, (m_lo, m_hi) in enumerate(sigma_sfr_bins):

        for b_z, (z_lo, z_hi) in enumerate(z_bins):

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
                "10020038", # 追加, スパイク

            ]

            selected = [
            
                it

                for it in used_items_valid

                if (
                    (it["logSigma"] >= m_lo)
                    and
                    (it["logSigma"] < m_hi)

                    and

                    (it["z"] >= z_lo)
                    and
                    (it["z"] < z_hi)

                    and

                    (it["id"] not in bad_ids)
                )
            ]

            if len(selected) == 0:

                print(
                    f"\nlogSigma [{m_lo},{m_hi}) "
                    f"z [{z_lo},{z_hi})"
                )

                continue


            sigma_sfr_vals = np.array([
                it["logSigma"]
                for it in selected
            ])

            z_vals = np.array([
                it["z"]
                for it in selected
            ])

            # raw
            flux_list_raw = [it["flux_raw"] for it in selected]
            err_list_raw  = [it["err_raw"]  for it in selected]

            # =====================================
            # visualize SII region
            # =====================================

            plt.figure(figsize=(8,5))

            sii_mask = (
                (wave_grid > 6718.29-100)
                &
                (wave_grid < 6732.67+100)
            )

            # -------------------------
            # individual spectra
            # -------------------------

            for spec in flux_list_raw:
            
                plt.plot(
                    wave_grid[sii_mask],
                    spec[sii_mask],
                    color="0.75",
                    lw=0.7,
                    alpha=0.4,
                    zorder=1
                )

            # -------------------------
            # simple mean stack
            # -------------------------

            mean_spec = np.nanmean(
                np.array(flux_list_raw),
                axis=0
            )

            plt.plot(
                wave_grid[sii_mask],
                mean_spec[sii_mask],
                color="black",
                lw=2.5,
                label=f"Mean stack (N={len(selected)})",
                zorder=10
            )

            plt.axvline(
                6718.29,
                ls="--",
                color="tab:red",
                alpha=0.7
            )

            plt.axvline(
                6732.67,
                ls="-.",
                color="tab:red",
                alpha=0.7
            )

            plt.xlabel(r"Rest wavelength [$\AA$]")
            plt.ylabel("Flux")

            # plt.title(
            #     f"logSFR=[{m_lo},{m_hi}) "
            #     f" z=[{z_lo},{z_hi})"
            # )

            plt.legend()
            plt.tight_layout()

            savefig = (
                "results/JADES/figure/"
                f"SII_individual_stack_"
                f"sigma_sfr_{m_lo:.1f}_{m_hi:.1f}"
                f"_z_{z_lo:.1f}_{z_hi:.1f}.png"
            )

            plt.savefig(savefig, dpi=200)
            plt.show()

            print("saved:", savefig)


            for it in selected:
            
                flux = it["flux_raw"]

                peak = np.nanmax(
                    flux[(wave_grid>6795)&(wave_grid<6810)]
                )

                if peak > 5e-19: # 閾値次第でバイアスが生じうるので注意
                
                    print(it["id"], peak)

            for it in selected:
            
                flux = it["flux_raw"]

                m = (
                    (wave_grid > 6718.29-100)
                    &
                    (wave_grid < 6732.67+100)
                )

                if np.any(np.isfinite(flux[m])):
                
                    peak = np.nanmax(flux[m])

                    # print(
                    #     peak,
                    #     it["id"]
                    # )

            # normalized
            flux_list_norm = [it["flux_norm"] for it in selected]
            err_list_norm  = [it["err_norm"]  for it in selected]

            # 追加
            flux_raw = np.array(flux_list_raw)
            err_raw  = np.array(err_list_raw)

            flux_norm = np.array(flux_list_norm)
            err_norm  = np.array(err_list_norm)


            # 追加
            flux_raw_mc = rng.normal(
                flux_raw[:, :, None],
                err_raw[:, :, None],
                size=(
                    flux_raw.shape[0],
                    flux_raw.shape[1],
                    N_MC
                )
            )

            # 追加
            flux_norm_mc = rng.normal(
                flux_norm[:, :, None],
                err_norm[:, :, None],
                size=(
                    flux_norm.shape[0],
                    flux_norm.shape[1],
                    N_MC
                )
            )

            # 各ビンの代表値を計算する
            sigma_sfr_mean = np.mean(sigma_sfr_vals)
            sigma_sfr_std = np.std(sigma_sfr_vals)
            z_mean = np.mean(z_vals)
            z_std = np.std(z_vals)

            print(
                f"\nlogSigma [{m_lo},{m_hi}) "
            )

            print(
                f"z [{z_lo},{z_hi})"
            )

            print(
                f"N = {len(selected)}"
            )

            print(
                f"logSigma = "
                f"{sigma_sfr_mean:.3f}"
            )

            print(
                f"z = "
                f"{z_mean:.3f}"
            )


            print(
                f"logSigma_std = "
                f"{sigma_sfr_std:.3f}"
            )

            print(
                f"z_std = "
                f"{z_std:.3f}"
            )

            print(
                f"sigma_sfr_range = "
                f"[{np.min(sigma_sfr_vals):.3f}, "
                f"{np.max(sigma_sfr_vals):.3f}]"
            )

            print(
                f"z_range = "
                f"[{np.min(z_vals):.3f}, "
                f"{np.max(z_vals):.3f}]"
            )

            # =========================
            # weighted stack
            # =========================
            floor_raw = 0.05*np.nanmedian(err_raw)

            err_raw_eff = np.sqrt(
                err_raw**2 + floor_raw**2
            )

            w_raw = 1.0/err_raw_eff[:,:,None]**2

            weighted_raw_mc = (
                np.nansum(
                    w_raw * flux_raw_mc,
                    axis=0
                )
                /
                np.nansum(
                    w_raw,
                    axis=0
                )
            )

            flux_stack_w_raw = np.nanmedian(
                weighted_raw_mc,
                axis=1
            )

            err_stack_w_raw = (
                np.nanpercentile(weighted_raw_mc,84,axis=1)
                -
                np.nanpercentile(weighted_raw_mc,16,axis=1)
            )/2

            floor_norm = 0.05*np.nanmedian(err_norm)

            err_norm_eff = np.sqrt(
                err_norm**2 + floor_norm**2
            )

            w_norm = 1.0 / err_norm_eff[:, :, None]**2

            weighted_norm_mc = (
                np.nansum(
                    w_norm * flux_norm_mc,
                    axis=0
                )
                /
                np.nansum(
                    w_norm,
                    axis=0
                )
            )

            flux_stack_w_norm = np.nanmedian(
                weighted_norm_mc,
                axis=1
            )

            err_stack_w_norm = (
                np.nanpercentile(weighted_norm_mc,84,axis=1)
                -
                np.nanpercentile(weighted_norm_mc,16,axis=1)
            )/2

            # =========================
            # median stack
            # =========================

            # raw, normalized
            median_raw_mc = np.nanmedian(flux_raw_mc, axis=0)
            median_norm_mc = np.nanmedian(flux_norm_mc, axis=0)
            flux_stack_m_raw = np.nanmedian(median_raw_mc, axis=1)
            flux_stack_m_norm = np.nanmedian(median_norm_mc, axis=1)
            err_stack_m_raw = (np.nanpercentile(median_raw_mc,84,axis=1) - np.nanpercentile(median_raw_mc,16,axis=1))/2
            err_stack_m_norm = (np.nanpercentile(median_norm_mc,84,axis=1) - np.nanpercentile(median_norm_mc,16,axis=1))/2

            # =========================
            # mean stack
            # =========================
            mean_raw_mc = np.nanmean(flux_raw_mc, axis=0)
            mean_norm_mc = np.nanmean(flux_norm_mc, axis=0)
            flux_stack_mean_raw = np.nanmedian(mean_raw_mc, axis=1)
            flux_stack_mean_norm = np.nanmedian(mean_norm_mc, axis=1)
            err_stack_mean_raw = (np.nanpercentile(mean_raw_mc,84,axis=1) - np.nanpercentile(mean_raw_mc,16,axis=1))/2
            err_stack_mean_norm = (np.nanpercentile(mean_norm_mc,84,axis=1) - np.nanpercentile(mean_norm_mc,16,axis=1))/2


            # =========================
            # output names
            # =========================

            outname_base = (
                "results/JADES/JADES_DR3/spectra/"
                f"stack_sigma_sfr_{m_lo:.1f}_{m_hi:.1f}"
                f"_z_{z_lo:.1f}_{z_hi:.1f}"
            )

            # =========================
            # save
            # =========================

            # --- mean raw ---

            np.savetxt(
                outname_base + "_mean_raw.txt",
                np.column_stack([
                    wave_grid,
                    flux_stack_mean_raw,
                    err_stack_mean_raw
                ]),
                header=(
                    f"mean raw | "
                    f"logSigma=[{m_lo},{m_hi}) | "
                    f"z=[{z_lo},{z_hi}) | "
                    f"N={len(selected)}"
                )
            )

            # # --- mean normalized ---

            # np.savetxt(
            #     outname_base + "_mean_norm.txt",
            #     np.column_stack([
            #         wave_grid,
            #         flux_stack_mean_norm,
            #         err_stack_mean_norm
            #     ]),
            #     header=(
            #         f"mean normalized | "
            #         f"logSigma=[{m_lo},{m_hi}) | "
            #         f"z=[{z_lo},{z_hi}) | "
            #         f"N={len(selected)}"
            #     )
            # )
            # # --- weighted raw ---
            # np.savetxt(
            #     outname_base + "_weighted_mean_raw.txt",
            #     np.column_stack([wave_grid, flux_stack_w_raw, err_stack_w_raw]),
            #     header=(
            #         f"weighted raw  | "
            #         f"logSigma=[{m_lo},{m_hi}) | "
            #         f"z=[{z_lo},{z_hi}) | "
            #         f"N={len(selected)}"
            #     )
            # )

            # # --- weighted normalized ---
            # np.savetxt(
            #     outname_base + "_weighted_mean_norm.txt",
            #     np.column_stack([wave_grid, flux_stack_w_norm, err_stack_w_norm]),
            #     header=(
            #         f"weighted normalized  | "
            #         f"logSigma=[{m_lo},{m_hi}) | "
            #         f"z=[{z_lo},{z_hi}) | "
            #         f"N={len(selected)}"
            #     )
            # )

            # # --- median raw ---
            # np.savetxt(
            #     outname_base + "_median_raw.txt",
            #     np.column_stack([wave_grid, flux_stack_m_raw, err_stack_m_raw]),
            #     header=(
            #         f"median raw  | "
            #         f"logSigma=[{m_lo},{m_hi}) | "
            #         f"z=[{z_lo},{z_hi}) | "
            #         f"N={len(selected)}"
            #     )
            # )

            # # --- median normalized ---
            # np.savetxt(
            #     outname_base + "_median_norm.txt",
            #     np.column_stack([wave_grid, flux_stack_m_norm, err_stack_m_norm]),
            #     header=(
            #         f"median normalized  | "
            #         f"logSigma=[{m_lo},{m_hi}) | "
            #         f"z=[{z_lo},{z_hi}) | "
            #         f"N={len(selected)}"
            #     )
            # )

            print("saved:", outname_base)


print("\nDone.")

print("CSV rows =", len(df))

print(
    "unique NIRSpec_ID =",
    df["NIRSpec_ID"].nunique()
)

print(
    "duplicate rows =",
    len(df) - df["NIRSpec_ID"].nunique()
)

dup_ids = df["NIRSpec_ID"].value_counts()
dup_ids = dup_ids[dup_ids > 1].index

print(
    df[df["NIRSpec_ID"].isin(dup_ids)]
    .sort_values("NIRSpec_ID")
)

from collections import Counter

ids = [it["id"] for it in used_items_all]

c = Counter(ids)

for k,v in c.items():

    if v > 1:
        print(k,v)