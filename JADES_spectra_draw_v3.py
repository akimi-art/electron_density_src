#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
スクリプトの概要:
このスクリプトは
JADESのカタログを使って
スペクトル(1d)を描画するものです。
同じディレクトリに入っている銀河を一挙に描画します。
フィッティングまでする、v2の強化版です。

使用方法:
    JADES_spectra_draw_v3.py [オプション]

著者: A. M.
作成日: 2026-06-20

参考文献:
    - PEP 8: https://peps.python.org/pep-0008/
    - PEP 257 (Docstring規約): https://peps.python.org/pep-0257/
    - Python公式ドキュメント: https://docs.python.org/ja/3/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import glob
import os
from scipy.optimize import curve_fit


# ========= 設定 =========
csv_file = "results/JADES/JADES_DR3/data_from_Nishigaki/jades_info_with_HA_plus_logSFR_with_Reff_sii_gn_gs.csv"

spec_base = "results/JADES/JADES_DR3/JADES_DR3_full_spectra"

gratings = ["JADES_DR3_G140M", "JADES_DR3_G235M", "JADES_DR3_G395M"]

# SII rest wavelength (Å)
SII_lines = [6716.440, 6730.820]
Ha_rest = 6564.61  # Å（あってる?）

# プロット範囲（Å）
delta_lambda = 800

# ========= CSV読み込み =========
df = pd.read_csv(csv_file)

# z_specあるもの
df_valid = df[df["z_spec"].notna()].copy()

print(f"z_specあり: {len(df_valid)} / {len(df)}")

# ========= スペクトル探索関数 =========
def find_spectrum(nirspec_id, wave_obs):
    id_str = f"{int(nirspec_id):08d}"

    for gr in gratings:
        path = os.path.join(spec_base, gr, f"*{id_str}*_x1d.fits")
        files = glob.glob(path)

        for f in files:
            try:
                with fits.open(f) as hdul:
                    data = hdul["EXTRACT1D"].data
                    wave = data["WAVELENGTH"] * 1e4
                    wave = np.array(wave, dtype=float)

                    wmin, wmax = wave.min(), wave.max()

                    # ✅ SIIが完全に入るスペクトルだけ選ぶ
                    if (wave_obs[0] > wmin) and (wave_obs[1] < wmax):
                        return f

            except:
                continue

    return None


# ========= フィッティング関数 =========
def double_gaussian(x, A1, A2, sigma, c0, c1):
    mu1 = line1_center
    mu2 = line2_center

    g1 = A1 * np.exp(-(x - mu1)**2 / (2 * sigma**2))
    g2 = A2 * np.exp(-(x - mu2)**2 / (2 * sigma**2))

    continuum = c0 + c1 * (x - (mu1 + mu2)/2)

    return g1 + g2 + continuum



# ========= 波長カバレッジ =========
# rough（μm → Å）
coverage = {
    "G140M": (0.7e4, 1.8e4),
    "G235M": (1.7e4, 3.2e4),
    "G395M": (2.9e4, 5.2e4),
}

def in_coverage(wave_obs):
    for key, (wmin, wmax) in coverage.items():
        if (wave_obs > wmin) and (wave_obs < wmax):
            return True
    return False

# ========= プロット準備 =========
batch_size = 100
panels = []

current_batch = []
count = 0

n_file = 0
n_read = 0
n_6716 = 0
n_6731 = 0
n_both = 0
n_any = 0

n_good = 0

offsets = []
offsets_6716 = []
offsets_6731 = []
z_list = []

sii_results = []

fit_panels = []
current_fit_batch = []

# ========= メインループ =========
for i, row in df_valid.iterrows():
    z = row["z_spec"]
    nid = row["NIRSpec_ID"]
    tier = row["TIER"]

    if np.isnan(z):
        continue

    # 観測波長
    wave_obs = np.array(SII_lines) * (1 + z)
    Ha_obs = Ha_rest * (1 + z)

    # カバレッジ外ならスキップ
    if not any([in_coverage(w) for w in wave_obs]):
        continue

    filepath = find_spectrum(nid, wave_obs)
    if filepath is None:
        continue
    
    if i < 5:
        print("wave_obs:", wave_obs)
        print("diff:", wave_obs[1] - wave_obs[0])

    if i < 5:
        print("overlap:", 
              (wave_obs[0] + delta_lambda) > (wave_obs[1] - delta_lambda))

    try:
        with fits.open(filepath) as hdul:
            data = hdul["EXTRACT1D"].data
            n_file += 1
            n_read += 1

            wave = data["WAVELENGTH"] * 1e4  # μm → Å
            flux = data["FLUX"] * 1e20
            flux_err = data["FLUX_ERR"] * 1e20

            wave = np.array(wave, dtype=np.float64)
            flux = np.array(flux, dtype=np.float64)
            flux_err = np.array(flux_err, dtype=np.float64)

            # プロット領域
            mask1 = (wave > wave_obs[0] - delta_lambda) & (wave < wave_obs[0] + delta_lambda)
            mask2 = (wave > wave_obs[1] - delta_lambda) & (wave < wave_obs[1] + delta_lambda)
            
            # ✅ カウント（ここ重要）
            if np.sum(mask1) > 0:
                n_6716 += 1
            
            if np.sum(mask2) > 0:
                n_6731 += 1
            
            if np.sum(mask1) > 0 or np.sum(mask2) > 0:
                n_any += 1
            
            if (np.sum(mask1) > 0) and (np.sum(mask2) > 0):
                n_both += 1
            
            # ✅ プロット用mask
            mask = mask1 | mask2   # ← ここ超重要（AND→OR）

            wave_plot = wave[mask]
            flux_plot = flux[mask]
            flux_err_plot = flux_err[mask]

            if len(wave_plot) > 0:
                peak_wave = wave_plot[np.argmax(flux_plot)]

                offset = peak_wave - np.mean(wave_obs)
                offset1 = peak_wave - wave_obs[0]
                offset2 = peak_wave - wave_obs[1]

                offsets.append(offset)
                offsets_6716.append(offset1)
                offsets_6731.append(offset2)
                z_list.append(z)


            # ===== 品質マスク =====

            # ① 点数条件
            if len(wave_plot) < 10:
                continue
            
            # ② NaN除去
            valid = ~np.isnan(flux_plot)
            wave_plot = wave_plot[valid]
            flux_plot = flux_plot[valid]
            flux_err_plot = flux_err_plot[valid]

            if len(wave_plot) < 10:
                continue

            # ✅ NEW：wave_plotベースで再マスク
            mask1_plot = (wave_plot > wave_obs[0] - delta_lambda) & (wave_plot < wave_obs[0] + delta_lambda)
            mask2_plot = (wave_plot > wave_obs[1] - delta_lambda) & (wave_plot < wave_obs[1] + delta_lambda)

            
            # ③ ギャップ検出（重要）
            dw = np.diff(wave_plot)

            # 平均ステップ
            median_dw = np.median(dw)

            # 大きすぎるギャップを検出
            if np.max(dw) > 5 * median_dw:
                continue

            width = 50 * (1 + z)  # Å（好きに調整）

            # ④ NEW：SII window 完備条件（修正版）
            if not (
                (wave_plot.min() < wave_obs[0] - width) and
                (wave_plot.max() > wave_obs[0] + width) and
                (wave_plot.min() < wave_obs[1] - width) and
                (wave_plot.max() > wave_obs[1] + width)
            ):
                continue

            # ⑤ SII window内に十分なデータ点があるか    
            if np.sum(mask1_plot) < 5 or np.sum(mask2_plot) < 5:
                continue

            # 念のため
            if len(wave_plot) == 0:
                continue


            # =========================
            # ===== フィッティング =====
            # =========================

            # SII周辺だけ切り出し（±delta_lambdaでOK）
            fit_mask = (
                (wave_plot > wave_obs[0] - delta_lambda) &
                (wave_plot < wave_obs[1] + delta_lambda)
            )

            x = wave_plot[fit_mask]
            y = flux_plot[fit_mask]
            yerr = flux_err_plot[fit_mask]

            if len(x) < 10:
                continue
            
            # ヤバいケース避ける
            if np.any(~np.isfinite(y)):
                continue
            
            # グローバル変数として渡す（簡単のため）
            line1_center = wave_obs[0]
            line2_center = wave_obs[1]

            # 初期値
            A1_init = np.max(y)
            A2_init = np.max(y) * 0.8
            sigma_init = 3.0
            c0_init = np.median(y)
            c1_init = 0

            p0 = [A1_init, A2_init, sigma_init, c0_init, c1_init]

            try:
                popt, pcov = curve_fit(
                    double_gaussian,
                    x,
                    y,
                    p0=p0,
                    sigma=yerr,
                    absolute_sigma=True,
                    maxfev=10000
                )
            
            except Exception as e:
                print(f"error: {filepath}")
                continue
            
            A1, A2, sigma, c0, c1 = popt

            # flux
            flux1 = A1 * sigma * np.sqrt(2*np.pi)
            flux2 = A2 * sigma * np.sqrt(2*np.pi)

            # error
            perr = np.sqrt(np.diag(pcov))
            A1_err, A2_err, sigma_err, _, _ = perr

            flux1_err = flux1 * np.sqrt(
                (A1_err/A1)**2 + (sigma_err/sigma)**2
            )

            flux2_err = flux2 * np.sqrt(
                (A2_err/A2)**2 + (sigma_err/sigma)**2
            )

            ratio = flux1 / flux2

            ratio_err = ratio * np.sqrt(
                (flux1_err/flux1)**2 +
                (flux2_err/flux2)**2
            )

            sii_results.append({
                "ID": nid,
                "z": z,
                "flux6716": flux1,
                "flux6731": flux2,
                "flux6716_err": flux1_err,
                "flux6731_err": flux2_err,
                "ratio": ratio,
                "ratio_err": ratio_err
            })

            # ===== フィット用データ作成 =====

            fit_width = 50 * (1 + z)  # Å（SII専用表示）

            fit_mask = (
                (wave_plot > np.mean(wave_obs) - fit_width) &
                (wave_plot < np.mean(wave_obs) + fit_width)
            )

            wave_fitplot = wave_plot[fit_mask]
            flux_fitplot = flux_plot[fit_mask]
            flux_err_fitplot = flux_err_plot[fit_mask]

            # ✅ フィットパネルに追加
            current_fit_batch.append((
                wave_fitplot,
                flux_fitplot,
                flux_err_fitplot,
                wave_obs,
                popt
            ))


            # ✅ ここで初めて追加
            current_batch.append((wave_plot, flux_plot, flux_err_plot, wave_obs, Ha_obs))

            count += 1
            n_good += 1
    
    except Exception as e:
        print(f"error: {filepath}")
        continue


    # ===== 100個たまったら描画（スペクトル） =====
    if len(current_batch) == batch_size:
        panels.append(current_batch)
        current_batch = []

    # ===== 100個たまったら描画（フィッティングカーブ） =====
    if len(current_fit_batch) == batch_size:
        fit_panels.append(current_fit_batch)
        current_fit_batch = []

# ===== メインループ終わり =====

# 残り（スペクトル）
if len(current_batch) > 0:
    panels.append(current_batch)

# ✅ 残り（フィット）
if len(current_fit_batch) > 0:
    fit_panels.append(current_fit_batch)
    

print(f"プロット対象スペクトル数: {count}")
print(f"パネル数: {len(panels)}")
print("========== DIAGNOSTIC ==========")
print("file exists:", n_file)
print("read success:", n_read)
print("6716 exists:", n_6716)
print("6731 exists:", n_6731)
print("any exists:", n_any)
print("both exists:", n_both)
print("good spectra:", n_good)


# # ① offset分布
# plt.figure(figsize=(6,4))
# plt.hist(offsets, bins=50)
# plt.axvline(0, color='r', linestyle='--')
# plt.xlabel("Offset [Å]")
# plt.ylabel("Number")
# plt.title("SII peak offset distribution")
# plt.show()

# # ② offset vs redshift
# plt.figure(figsize=(6,4))
# plt.scatter(z_list, offsets, s=5)
# plt.axhline(0, color='r', linestyle='--')
# plt.xlabel("z_spec")
# plt.ylabel("Offset [Å]")
# plt.title("Offset vs redshift")
# plt.show()

# # ③ 6716 vs 6731
# plt.figure(figsize=(6,4))
# plt.hist(offsets_6716, bins=50, alpha=0.5, label="6716")
# plt.hist(offsets_6731, bins=50, alpha=0.5, label="6731")
# plt.axvline(0, color='k', linestyle='--')
# plt.xlabel("Offset [Å]")
# plt.ylabel("Number")
# plt.legend()
# plt.title("Line identification")
# plt.show()


# ========= 描画（スペクトル） =========
output_dir = "results/JADES/figure"
os.makedirs(output_dir, exist_ok=True)
for p, batch in enumerate(panels):
    fig, axes = plt.subplots(10, 10, figsize=(20, 20))
    axes = axes.flatten()
    fig.subplots_adjust(
        left=0.05,
        right=0.95,
        bottom=0.05,
        top=0.95,
        wspace=0.0,
        hspace=0.0
    )

    for i, (wave, flux, flux_err, wave_lines, Ha_obs) in enumerate(batch):
        ax = axes[i]

        ax.step(wave, flux, where="mid", color="black", lw=1.0)
        ax.fill_between(wave, flux - flux_err, flux + flux_err, step="mid", color="gray", alpha=0.4)

        # SIIライン
        for w in wave_lines:
            ax.axvline(w, color='r', linestyle='--', alpha=0.5)
        
        # Hαライン
        ax.axvline(Ha_obs, color='r', linestyle=':', alpha=0.7)

        ax.set_xticks([])
        ax.set_yticks([])

    # 余り消す
    for j in range(len(batch), 100):
        axes[j].axis("off")

    fig.suptitle(f"SII panels {p}")
    

    # ✅ 保存
    filename = f"{output_dir}/sii_panel_{p:03d}.png"
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"Saved as {filename}")
    plt.show()


# ========= 描画（フィッティングカーブ） =========
output_dir_fit = "results/JADES/fit_figure"
os.makedirs(output_dir_fit, exist_ok=True)

for p, batch in enumerate(fit_panels):
    fig, axes = plt.subplots(10, 10, figsize=(20, 20))
    axes = axes.flatten()

    fig.subplots_adjust(
        left=0.05,
        right=0.95,
        bottom=0.05,
        top=0.95,
        wspace=0.0,
        hspace=0.0
    )

    for i, (wave, flux, flux_err, wave_lines, popt) in enumerate(batch):
        ax = axes[i]

        # ===== データ =====
        ax.step(wave, flux, where="mid", color="black", lw=1.0)
        ax.fill_between(
            wave,
            flux - flux_err,
            flux + flux_err,
            step="mid",
            color="gray",
            alpha=0.4
        )

        # ===== フィット =====
        try:
            A1, A2, sigma, c0, c1 = popt

            x_fit = np.linspace(wave.min(), wave.max(), 200)

            mu1 = wave_lines[0]
            mu2 = wave_lines[1]
            center = np.mean(wave_lines)

            y_fit = (
                A1 * np.exp(-(x_fit - mu1)**2 / (2 * sigma**2)) +
                A2 * np.exp(-(x_fit - mu2)**2 / (2 * sigma**2)) +
                (c0 + c1 * (x_fit - center))
            )


            ax.plot(
                x_fit,
                y_fit,
                color="red",
                lw=2
            )

            # individual lines（見やすい）
            g1 = A1 * np.exp(-(x_fit - mu1)**2 / (2 * sigma**2)) + (c0 + c1 * (x_fit - center))
            g2 = A2 * np.exp(-(x_fit - mu2)**2 / (2 * sigma**2)) + (c0 + c1 * (x_fit - center))

            ax.plot(x_fit, g1, color='red', lw=2, linestyle='--', alpha=0.7)
            ax.plot(x_fit, g2, color='red', lw=2, linestyle='-.', alpha=0.7)

        except:
            pass

        # ===== SII位置 =====
        for w in wave_lines:
            ax.axvline(w, color='red', linestyle='--', alpha=0.5)

        ax.set_xticks([])
        ax.set_yticks([])

    # 空白消す
    for j in range(len(batch), 100):
        axes[j].axis("off")

    fig.suptitle(f"SII fit panels {p}")

    filename = f"{output_dir_fit}/sii_fit_panel_{p:03d}.png"
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"Saved as {filename}")

    plt.show()