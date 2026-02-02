#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
スクリプトの概要:
SIIのLumiosityとzの関係を描画します。


使用方法:
    sii_luminoisity_vs_z.py [オプション]

著者: A. M.
作成日: 2026-02-01

参考文献:
    - PEP 8:                  https://peps.python.org/pep-0008/
    - PEP 257 (Docstring規約): https://peps.python.org/pep-0257/
    - Python公式ドキュメント:    https://docs.python.org/ja/3/
    - Curti+17
"""

"""
luminosity_vs_z_v1.pyを参考に後でflux一定の線を入れよう
"""



# === 必要なモジュールのインポート ===
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u


# 軸の設定
plt.rcParams.update({
    # --- 図全体 ---
    "figure.figsize": (12, 6),       # 図サイズ
    "font.size": 16,                 # 全体フォントサイズ
    "axes.labelsize": 18,            # 軸ラベルのサイズ
    "axes.titlesize": 18,            # タイトルのサイズ
    "axes.grid": False,              # グリッドOFF

    # --- 目盛り設定 (ticks) ---
    "xtick.direction": "in",         # x軸目盛りの向き
    "ytick.direction": "in",         # y軸目盛りの向き
    "xtick.top": False,               # 上にも目盛り
    "ytick.right": False,             # 右にも目盛り

    # 主目盛り（major ticks）
    "xtick.major.size": 16,          # 長さ
    "ytick.major.size": 16,
    "xtick.major.width": 2,          # 太さ
    "ytick.major.width": 2,

    # 補助目盛り（minor ticks）
    "xtick.minor.visible": False,     # 補助目盛りON
    "ytick.minor.visible": False,
    "xtick.minor.size": 8,           # 長さ
    "ytick.minor.size": 8,
    "xtick.minor.width": 1.5,        # 太さ
    "ytick.minor.width": 1.5,

    # --- 目盛りラベル ---
    "xtick.labelsize": 14,           # x軸ラベルサイズ
    "ytick.labelsize": 14,           # y軸ラベルサイズ
})


# === ファイル設定 ===
current_dir = os.getcwd()

file_sii  = os.path.join(current_dir, "results/txt/sdss_ne_vs_sm_standard_re_integer.txt")
file_msfr = os.path.join(current_dir, "results/Samir16/Samir16in_standard_v3_ms_only_v3_re_no_agn.txt")


# === 1. SIIファイルの読み込み ===
sii_data = {}
with open(file_sii, "r") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) < 8:
            continue
        target_id = parts[0]
        z = float(parts[3])
        try:
            flux_6716 = float(parts[5]) * 1e-17
            flux_6731 = float(parts[7]) * 1e-17
        except ValueError:
            continue
        if flux_6716 > 0 and flux_6731 > 0:
            # 光度を計算
            d_L = cosmo.luminosity_distance(z).to(u.cm).value
            L_6716 = 4 * np.pi * d_L**2 * flux_6716
            L_6731 = 4 * np.pi * d_L**2 * flux_6731
            sii_data[target_id] = {
                "z": z,
                "L_6716": L_6716,
                "L_6731": L_6731
            }

print(f"SIIデータ読み込み完了: {len(sii_data)} objects")

# === 2. M*, SFRファイルの読み込み ===
pattern = re.compile(r"^([+-]?\d+(?:\.\d+)?)(?:\+(\d+(?:\.\d+)?))?(?:-(\d+(?:\.\d+)?))?$")
msfr_data = {}

Mstar_col = 7  # 8列目
SFR_col = 9    # 10列目

with open(file_msfr, "r") as f:
    for line in f:
        if not line.strip():
            continue
        parts = re.split(r"\s+", line.strip())
        if len(parts) <= max(Mstar_col, SFR_col):
            continue
        target_id = parts[0]

        m_m = pattern.match(parts[Mstar_col])
        m_s = pattern.match(parts[SFR_col])
        if not m_m or not m_s:
            continue
        Mstar = float(m_m.group(1))
        SFR = float(m_s.group(1))
        msfr_data[target_id] = {"Mstar": Mstar, "SFR": SFR}

print(f"M*, SFRデータ読み込み完了: {len(msfr_data)} objects")

# === 3. IDマッチング ===
common_ids = list(set(sii_data.keys()) & set(msfr_data.keys()))
print(f"共通ID数: {len(common_ids)}")

L6716, L6731, z = [], [], []
for tid in common_ids:
    L6716.append(sii_data[tid]["L_6716"])
    L6731.append(sii_data[tid]["L_6731"])
    z.append(sii_data[tid]["z"])

L6716 = np.array(L6716)
L6731 = np.array(L6731)
z = np.array(z)


# === 4. 図を描く（余白ゼロ & 右列 y軸非表示） ===
fig, axes = plt.subplots(
    2, 1, figsize=(6, 6),
    sharex='col',  # 同じ列で x 軸を共有
    sharey='row'   # 同じ行で y 軸を共有
)

ax1, ax2 = axes[0], axes[1]

# 散布図
ax1.scatter(z, L6716, s=8, color="C0", alpha=0.6)
ax2.scatter(z, L6731, s=8, color="C1", alpha=0.6)

# y 軸：対数
for ax in (ax1, ax2):
    ax.set_yscale("log")

# 左列だけ y ラベル・目盛を表示、右列は消す
ax1.set_ylabel("L(SII 6716) [erg/s]")
ax2.set_ylabel("L(SII 6731) [erg/s]")  # 下段左の y ラベル（必要に応じて 6731 に変更）
# for ax in (ax2, ax4):
#     ax.tick_params(labelleft=False)  # 目盛ラベル非表示
#     ax.yaxis.set_ticks_position('none')  # 目盛線も消す（必要なら）

# x ラベルは下段のみ（共有のため上段は不要）
ax1.set_xlabel(r"z")
ax2.set_xlabel(r"z")

# y 範囲
ax1.set_ylim(1e35, 1e39)
ax2.set_ylim(1e37, 1e41)

# サブプロット間の余白を完全にゼロ
plt.subplots_adjust(left=0.07, right=0.98, bottom=0.08, top=0.98, wspace=0, hspace=0)
# ↑ left/right/bottom/top は図枠と軸ラベルが被らないように微調整（必要に応じて調整）

savepath = os.path.join(current_dir, "results/figure/sii_luminosity_vs_z.png")
plt.savefig(savepath, dpi=200)
plt.show()