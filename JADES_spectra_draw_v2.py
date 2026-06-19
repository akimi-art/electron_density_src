#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
スクリプトの概要:
このスクリプトは
JADESのカタログを使って
スペクトル(1d)を描画するものです。
同じディレクトリに入っている銀河を一挙に描画します。

使用方法:
    JADES_spectra_draw_v2.py [オプション]

著者: A. M.
作成日: 2026-02-08

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

# ========= 設定 =========
csv_file = "results/JADES/JADES_DR3/data_from_Nishigaki/jades_info_with_HA_plus_logSFR_with_Reff_sii_gn_gs.csv"

spec_base = "results/JADES/JADES_DR3/JADES_DR3_full_spectra"

gratings = ["JADES_DR3_G140M", "JADES_DR3_G235M", "JADES_DR3_G395M"]

# SII rest wavelength (Å)
SII_lines = [6716.440, 6730.820]

# プロット範囲（Å）
delta_lambda = 100

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

# ========= メインループ =========
for i, row in df_valid.iterrows():
    z = row["z_spec"]
    nid = row["NIRSpec_ID"]
    tier = row["TIER"]

    if np.isnan(z):
        continue

    # 観測波長
    wave_obs = np.array(SII_lines) * (1 + z)

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

            if len(wave_plot) == 0:
                continue

            current_batch.append((wave_plot, flux_plot, flux_err_plot, wave_obs))

            count += 1

    except Exception as e:
        print(f"error: {filepath}")
        continue

    # ===== 100個たまったら描画 =====
    if len(current_batch) == batch_size:
        panels.append(current_batch)
        current_batch = []

# 残り
if len(current_batch) > 0:
    panels.append(current_batch)

print(f"プロット対象スペクトル数: {count}")
print(f"パネル数: {len(panels)}")
print("========== DIAGNOSTIC ==========")
print("file exists:", n_file)
print("read success:", n_read)
print("6716 exists:", n_6716)
print("6731 exists:", n_6731)
print("any exists:", n_any)
print("both exists:", n_both)

# ========= 描画 =========
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

    for i, (wave, flux, flux_err, wave_lines) in enumerate(batch):
        ax = axes[i]

        ax.step(wave, flux, where="mid", color="black", lw=1.0)
        ax.fill_between(wave, flux - flux_err, flux + flux_err, step="mid", color="gray", alpha=0.4)

        # SIIライン
        for w in wave_lines:
            ax.axvline(w, color='r', linestyle='--', alpha=0.5)

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