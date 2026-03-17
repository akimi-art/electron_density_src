#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
スクリプトの概要:
JADESスペクトルスタックを作成します。
SFRのビンごとにスタックを作成します。

使用方法:
    JADES_spectra_stack_v3.py [オプション]

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

def median_stack(fluxes):

    fluxes = np.array(fluxes)

    flux_med = np.nanmedian(fluxes, axis=0)

    return flux_med

def bootstrap_median_error(fluxes, nboot=500):

    fluxes = np.array(fluxes)

    n_spec = fluxes.shape[0]
    n_wave = fluxes.shape[1]

    med_boot = np.zeros((nboot, n_wave))

    for i in range(nboot):

        idx = np.random.randint(0, n_spec, n_spec)
        sample = fluxes[idx]

        med_boot[i] = np.nanmedian(sample, axis=0)

    err = np.nanstd(med_boot, axis=0)

    return err

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


def _safe_sigma_arrays(el, eu, x):
    """非対称誤差 el(e-), eu(e+) から、計算に使う sigma_l, sigma_r, sigma_sym を作る。
    欠損や非正値はグローバルな代替スケール（MAD 由来）で埋める。
    """
    el = np.asarray(el, dtype=float)
    eu = np.asarray(eu, dtype=float)
    x  = np.asarray(x,  dtype=float)

    # そのままの左右誤差（不正は NaN に）
    sl = np.where(np.isfinite(el) & (el > 0), el, np.nan)
    sr = np.where(np.isfinite(eu) & (eu > 0), eu, np.nan)

    # 対称用（保守的に max, もしくは平均でも可）
    s_sym = np.nanmax(np.stack([sl, sr], axis=0), axis=0)  # max(lower, upper)

    # グローバル代替スケール：x のロバスト分散（MAD）
    mad = np.nanmedian(np.abs(x - np.nanmedian(x)))
    fallback = 1.4826 * mad if np.isfinite(mad) and mad > 0 else np.nanstd(x)

    if not np.isfinite(fallback) or fallback <= 0:
        # 極端に小標本等で分散が出せない場合のフォールバック
        fallback = np.nanmedian(np.hstack([el, eu]))
        if not np.isfinite(fallback) or fallback <= 0:
            fallback = 0.1  # 最後の砦：小さめの定数

    # 欠損は代替スケールで埋める
    sl = np.where(np.isfinite(sl), sl, fallback)
    sr = np.where(np.isfinite(sr), sr, fallback)
    s_sym = np.where(np.isfinite(s_sym) & (s_sym > 0), s_sym, fallback)

    return sl, sr, s_sym


def split_normal_bootstrap_stats(x, el, eu, B=5000, seed=42):
    """split-normal からのブートストラップで、median と weighted mean の分布を推定。
    返り値: dict(median, med_lo, med_hi, wmean, wmean_se, wmean_lo, wmean_hi)
    """
    x = np.asarray(x, dtype=float)
    el = np.asarray(el, dtype=float)
    eu = np.asarray(eu, dtype=float)
    n  = x.size

    sl, sr, s_sym = _safe_sigma_arrays(el, eu, x)

    # 重み（閉形式の平均と SE 用）
    w = 1.0 / (s_sym**2)
    wsum = np.sum(w)
    wmean = np.sum(w * x) / wsum

    # 誤差のスケール補正（任意）：chi^2_nu > 1 の場合に SE を拡張
    resid = x - wmean
    chi2 = np.sum(w * resid**2)
    dof = max(n - 1, 1)
    chi2_nu = chi2 / dof
    wmean_se = np.sqrt(1.0 / wsum) * np.sqrt(max(1.0, chi2_nu))

    # --- ブートストラップ（split-normal 近似） ---
    rng = np.random.default_rng(seed)
    # 左側確率 p_left = sl / (sl + sr)
    p_left = sl / (sl + sr)

    med_samples = np.empty(B)
    wmn_samples = np.empty(B)

    for b in range(B):
        u = rng.random(n)
        z = np.abs(rng.standard_normal(n))  # 半正規

        # 片側選択してから左右の sigma でずらす
        left = u < p_left
        s = np.empty(n)
        s[left]  = x[left]  - z[left]  * sl[left]
        s[~left] = x[~left] + z[~left] * sr[~left]

        # ブートストラップ標本での median
        med_samples[b] = np.nanmedian(s)

        # ブートストラップ標本での weighted mean
        # ここでも重みは s_sym を流用（測定誤差由来の重みなので固定が自然）
        wmn_samples[b] = np.sum(w * s) / wsum

    # 16–84 パーセンタイル（≒1σ）
    med = np.nanmedian(x)
    med_lo = med - np.nanpercentile(med_samples, 16)
    med_hi = np.nanpercentile(med_samples, 84) - med

    wmn_lo = wmean - np.nanpercentile(wmn_samples, 16)
    wmn_hi = np.nanpercentile(wmn_samples, 84) - wmean

    return {
        "median": med, "median_err_lo": med_lo, "median_err_hi": med_hi,
        "wmean": wmean, "wmean_se": wmean_se,
        "wmean_err_lo": wmn_lo, "wmean_err_hi": wmn_hi,
        "chi2_nu": chi2_nu, "N": n
    }

# ============================
# main
# ============================

n_bins = 3  # ← 等数ビンの数（ここでは3）
used_z_all = []  # ← すべての grating で「使用できた」銀河の z を集約

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

    # --- ここは従来どおり ---
    flux_list = []
    err_list  = []

    # （任意）HAの上位外れ値カット用パーセンタイル（単位に依らない健全性チェック）
    ha_vals_gr = df_gr["HA_6563_flux"].values
    ha_vals_gr = ha_vals_gr[np.isfinite(ha_vals_gr) & (ha_vals_gr > 0)]
    ha_p995 = np.nanpercentile(ha_vals_gr, 99.5) if ha_vals_gr.size > 0 else np.inf

    for _, row in df_gr.iterrows():

        nid = int(row["NIRSpec_ID"])
        z   = row["z_spec"]
        ha  = row["HA_6563_flux"]

        # --- Hαの健全性チェック（単位非依存）---
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
            flux = flux / ha
            err  = err  / ha

            flux_i, err_i = resample(wave, flux, err)
            if not np.isfinite(flux_i).any():
                continue

            # --- ここで「使用できた」と判定できるので z を収集 ---
            used_z_all.append(z)

            # （従来の格納：使うなら）
            flux_list.append(flux_i)
            err_list.append(err_i)

        except Exception:
            continue

    print("spectra used:", len(flux_list))

# =========================================
# ここから「統合ヒストグラム + 等数3分割の境界線」だけを描画
# =========================================

if len(used_z_all) == 0:
    print("No usable galaxies across all gratings. Skip histogram.")
else:
    used_z_all = np.array(used_z_all)
    N = len(used_z_all)

    # z を昇順にソート
    sort_idx = np.argsort(used_z_all)
    z_sorted = used_z_all[sort_idx]

    # N を 3 等分（余りは先頭ビンから配分）→ 等数ビンのカウント配列
    q, r = divmod(N, n_bins)
    counts = [q + 1 if i < r else q for i in range(n_bins)]  # 例: N=11 → [4,4,3]
    cum = np.cumsum([0] + counts)  # 例: [0,4,8,11]

    # プロットする縦線位置：ビン境界の「中点」
    boundaries = []
    for i in range(1, n_bins):
        left_end = z_sorted[cum[i] - 1]
        right_start = z_sorted[cum[i]]
        boundaries.append(0.5 * (left_end + right_start))

    # ヒストグラム描画（bin は自動 or 任意で調整）
    plt.figure(figsize=(7, 4))
    plt.hist(used_z_all, bins="auto", color="0.6", edgecolor="black", alpha=0.8)

    # 等数3分割の境界線を追加
    for bx in boundaries:
        plt.axvline(bx, color="tab:red", linestyle="--", lw=2, alpha=0.9)

    # 軸・注記
    plt.xlabel("redshift (z) of usable galaxies")
    plt.ylabel("count")
    plt.title(f"Usable galaxies across all gratings: N={N}\nEqual-count 3-bin split: {counts}")
    # 凡例風に境界値を注記
    for bx in boundaries:
        plt.text(bx, plt.ylim()[1]*0.95, f"{bx:.3f}", color="tab:red",
                 ha="center", va="top", fontsize=9, rotation=90, bbox=dict(fc="white", ec="none", alpha=0.7))

    plt.tight_layout()
    plt.show()




# ============================
# main
# ============================

n_bins = 3  # ← 等数ビン数
max_individual_show = 40  # ← 個別スペクトルの図で表示する最大枚数（多すぎると図が巨大になるため）

# 全グレーティング横断で「使用できた」スペクトルを集約
# 各要素: dict(z, flux, err, id, gr)
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

    print(f"  candidates after z-range filter: {len(df_gr)}")

    flux_list = []
    err_list  = []

    # （任意）HAの上位外れ値カット（単位非依存・健全性チェック）
    ha_vals_gr = df_gr["HA_6563_flux"].values
    ha_vals_gr = ha_vals_gr[np.isfinite(ha_vals_gr) & (ha_vals_gr > 0)]
    ha_p995 = np.nanpercentile(ha_vals_gr, 99.5) if ha_vals_gr.size > 0 else np.inf

    for _, row in df_gr.iterrows():

        nid = int(row["NIRSpec_ID"])
        z   = row["z_spec"]
        ha  = row["HA_6563_flux"]

        # Hα の健全性（単位に依らず有限かつ正、必要なら上位外れ値を除去）
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

            # Hα で正規化（相対化）
            flux = flux / ha
            err  = err  / ha

            flux_i, err_i = resample(wave, flux, err)
            if not np.isfinite(flux_i).any():
                continue

            # --- この時点で「使用できた」ので、両方に格納 ---
            flux_list.append(flux_i)
            err_list.append(err_i)

            used_items_all.append({
                "z": z,
                "flux": flux_i,
                "err": err_i,
                "id": sid,
                "gr": gr,
                "sfr": row["log10_SFR_hb"],                         # ★ 追加
                "sfr_err_lo": row["log10_SFR_hb_err_lower"],        # ★ 追加
                "sfr_err_hi": row["log10_SFR_hb_err_upper"],        # ★ 追加
            })

        except Exception:
            continue

    print("spectra used:", len(flux_list))

    # ======== 既存の「グレーティング別」可視化・行列 ========
    if len(flux_list) > 0:
        flux_array = np.array(flux_list)

        # 個別スペクトル
        n_spec = flux_array.shape[0]
        ncol = 5
        nrow = int(np.ceil(min(n_spec, max_individual_show) / ncol))

        fig = plt.figure(figsize=(3*ncol, 1.5*nrow))
        gs = gridspec.GridSpec(nrow, ncol)
        gs.update(hspace=0.0, wspace=0.0)

        for i in range(min(n_spec, max_individual_show)):
            ax = fig.add_subplot(gs[i])
            ax.plot(wave_grid, flux_array[i], lw=1)
            ax.axvline(6563, color="red"); ax.axvline(6716, color="red"); ax.axvline(6731, color="red")
            ax.set_xlim(6500, 6900)
            ax.set_yticks([])
            if i < (nrow-1)*ncol:
                ax.set_xticklabels([])

        ttl = f"{gr} individual spectra"
        if n_spec > max_individual_show:
            ttl += f" (showing first {max_individual_show} of {n_spec})"
        plt.suptitle(ttl)
        plt.show()

        # スペクトル行列
        plt.figure(figsize=(6, 8))
        plt.imshow(
            flux_array,
            aspect="auto",
            vmin=np.nanpercentile(flux_array, 5),
            vmax=np.nanpercentile(flux_array, 95)
        )
        plt.xlabel("wavelength pixel")
        plt.ylabel("galaxy index")
        plt.title(f"{gr} spectra matrix (N={n_spec})")
        plt.colorbar()
        plt.show()

        # 全体スタック（従来どおり）
        flux_stack_all, err_stack_all = stack(flux_list, err_list)
        outname_all = f"stack_{gr}.txt"
        np.savetxt(
            outname_all,
            np.column_stack([wave_grid, flux_stack_all, err_stack_all]),
            header=f"wave flux err | {gr} ALL, N={len(flux_list)}"
        )

# ============================
# ここから新規：全グレーティング横断・等数3分割 z-bin ごとの可視化とスタック
# ============================

if len(used_items_all) == 0:
    print("No usable galaxies across all gratings. Skip z-bin visualizations & stacks.")
else:
    used_z_all = np.array([it["z"] for it in used_items_all])
    N = len(used_z_all)

    # z を昇順にソートし、等数で 3 分割（余りは先頭から配分）
    sort_idx = np.argsort(used_z_all)
    q, r = divmod(N, n_bins)
    counts = [q + 1 if i < r else q for i in range(n_bins)]  # 例: N=11 -> [4,4,3]
    cum = np.cumsum([0] + counts)  # 例: [0,4,8,11]

    print(f"\nGlobal equal-count z-binning across all gratings: N={N}, counts={counts}")

    # 各ビンで個別プロット・行列・スタック
    for b in range(n_bins):
        s, e = cum[b], cum[b+1]
        if e - s <= 0:
            print(f"z-bin {b+1}: N=0 (skip)")
            continue

        sel_idx_sorted = sort_idx[s:e]
        selected = [used_items_all[i] for i in sel_idx_sorted]

        z_vals = np.array([it["z"] for it in selected])
        flux_list_bin = [it["flux"] for it in selected]
        err_list_bin  = [it["err"]  for it in selected]

        # ▼▼ 追加：SFR 統計（median, weighted mean） ▼▼
        sfr_vals = np.array([it["sfr"] for it in selected], dtype=float)
        sfr_el   = np.array([it["sfr_err_lo"] for it in selected], dtype=float)
        sfr_eu   = np.array([it["sfr_err_hi"] for it in selected], dtype=float)

        stats = split_normal_bootstrap_stats(sfr_vals, sfr_el, sfr_eu, B=5000, seed=42)

        print(f"    median log10(SFR) = {stats['median']:.3f} "
              f"-{stats['median_err_lo']:.3f}/+{stats['median_err_hi']:.3f}")

        print(f"    weighted mean log10(SFR) = {stats['wmean']:.3f} "
              f"±{stats['wmean_se']:.3f}  "
              f"(bootstrap CI: -{stats['wmean_err_lo']:.3f}/+{stats['wmean_err_hi']:.3f}, "
              f"chi2_nu={stats['chi2_nu']:.2f}, N={stats['N']})")
        # ▲▲ ここまで追加 ▲▲

        z_min, z_max = float(np.min(z_vals)), float(np.max(z_vals))
        print(f"  z-bin {b+1}: N={len(selected)}  z~[{z_min:.3f}, {z_max:.3f}]")

        # --- 個別スペクトル ---
        flux_array_bin = np.array(flux_list_bin)
        n_spec = flux_array_bin.shape[0]
        ncol = 5
        nrow = int(np.ceil(min(n_spec, max_individual_show) / ncol))

        fig = plt.figure(figsize=(3*ncol, 1.5*nrow))
        gs = gridspec.GridSpec(nrow, ncol)
        gs.update(hspace=0.0, wspace=0.0)

        for i in range(min(n_spec, max_individual_show)):
            ax = fig.add_subplot(gs[i])
            ax.plot(wave_grid, flux_array_bin[i], lw=1)
            ax.axvline(6563, color="red"); ax.axvline(6716, color="red"); ax.axvline(6731, color="red")
            ax.set_xlim(6500, 6900)
            ax.set_yticks([])
            if i < (nrow-1)*ncol:
                ax.set_xticklabels([])

        ttl = f"z-bin {b+1}/{n_bins} individual spectra (z~[{z_min:.3f},{z_max:.3f}])"
        if n_spec > max_individual_show:
            ttl += f" (showing first {max_individual_show} of {n_spec})"
        plt.suptitle(ttl)
        plt.show()

        # --- スペクトル行列 ---
        plt.figure(figsize=(6, 8))
        plt.imshow(
            flux_array_bin,
            aspect="auto",
            vmin=np.nanpercentile(flux_array_bin, 5),
            vmax=np.nanpercentile(flux_array_bin, 95)
        )
        plt.xlabel("wavelength pixel")
        plt.ylabel("galaxy index")
        plt.title(f"z-bin {b+1}/{n_bins} spectra matrix (N={n_spec})")
        plt.colorbar()
        plt.show()

        # --- スタック & 保存 ---
        flux_stack_bin, err_stack_bin = stack(flux_list_bin, err_list_bin)
        # median stack
        flux_stack_med = median_stack(flux_list_bin)
        err_stack_med  = bootstrap_median_error(flux_list_bin)
        outname_bin = f"stack_all_zbin{b+1}.txt"
        outname_med = f"stack_all_zbin{b+1}_median.txt"
        np.savetxt(
            outname_bin,
            np.column_stack([wave_grid, flux_stack_bin, err_stack_bin]),
            header=f"wave flux err | ALL gratings, z-bin {b+1}/{n_bins}, N={len(flux_list_bin)}, z~[{z_min:.5f},{z_max:.5f}]"
        )
        np.savetxt(
        outname_med,
        np.column_stack([wave_grid, flux_stack_med, err_stack_med]),
        header=f"wave flux err | MEDIAN stack, z-bin {b+1}/{n_bins}"
        )
