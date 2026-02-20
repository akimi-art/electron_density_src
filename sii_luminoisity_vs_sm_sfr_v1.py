#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
スクリプトの概要:
SIIのLuminosityとStellar Mass, SFRの
関係性をプロットします。
Completeなサンプルだと思われるもののみを使用しています。
SM, SFRのビンごとに分けてプロットします。

使用方法:
    sii_luminoisity_vs_sm_sfr_v1.py [オプション]

著者: A. M.
作成日: 2026-02-17

参考文献:
    - PEP 8:                  https://peps.python.org/pep-0008/
    - PEP 257 (Docstring規約): https://peps.python.org/pep-0257/
    - Python公式ドキュメント:    https://docs.python.org/ja/3/
    - Curti+17
"""


# === 必要なモジュール ===
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u


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
    "xtick.top": False,               # 上にも目盛り
    "ytick.right": False,             # 右にも目盛り

    # 主目盛り（major ticks）
    "xtick.major.size": 20,          # 長さ
    "ytick.major.size": 20,
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
    "xtick.labelsize": 20,           # x軸ラベルサイズ
    "ytick.labelsize": 20,           # y軸ラベルサイズ

    # --- フォント ---
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
})

# plt.rcParams.update({
#     "figure.figsize": (18, 9),
#     "font.size": 16,
#     "axes.labelsize": 18,
#     "xtick.direction": "in",
#     "ytick.direction": "in",
#     "xtick.top": False,
#     "ytick.right": False,
#     "xtick.major.size": 12,
#     "ytick.major.size": 12,
#     "xtick.major.width": 2,
#     "ytick.major.width": 2,
#     "xtick.labelsize": 14,
#     "ytick.labelsize": 14,
# })

# === 入力ファイル（サブセット FITS） ===
current_dir = os.getcwd()
# fits_path = os.path.join(
#     current_dir,
#     "results/fits/mpajhu_dr7_v5_2_merged_L6717_ge_4pi_dL2_1e-17_L6731_ge_4pi_dL2_1e-17_z0.00-0.40.fits"
# )
csv_path = "./results/Samir16/Samir16in_standard_re_v1.csv"

# # === 読み込み（列構造はそのまま） ===
# t = Table.read(fits_path, format="fits")

df = pd.read_csv(csv_path)

# === 必要列を取り出し ===
# MPA-JHU: sm_MEDIAN, sfr_MEDIAN は log(M*/Msun), log(SFR/Msun/yr)
# z = np.array(t["Z"], dtype=float)
# F6717_raw = np.array(t["SII_6717_FLUX"], dtype=float)
# F6731_raw = np.array(t["SII_6731_FLUX"], dtype=float)
# logM = np.array(t["sm_MEDIAN"], dtype=float)
# logSFR = np.array(t["sfr_MEDIAN"], dtype=float)
z = np.array(df["z"], dtype=float)
F6717_raw = np.array(df["SII_6717_FLUX"], dtype=float)
F6731_raw = np.array(df["SII_6731_FLUX"], dtype=float)
logM = np.array(df["logSM_median"], dtype=float)
logSFR = np.array(df["logSFR_SED_median"], dtype=float)

# ←← ここで logSFR の欠損処理を入れる
# 欠損値（-1.0）を NaN に置換
missing_sfr_value = -1.0
logSFR = np.where(logSFR == missing_sfr_value, np.nan, logSFR)

# === 単位スケール（MPA-JHUは 1e-17 cgs）→ cgs へ ===
UNIT_FLUX = 1e-17
F6717 = F6717_raw * UNIT_FLUX
F6731 = F6731_raw * UNIT_FLUX

# === 光度計算（L = 4π d_L^2 F） ===
m_all = np.isfinite(z) & np.isfinite(F6717) & np.isfinite(F6731) & np.isfinite(logM) & np.isfinite(logSFR)
dL = np.full_like(z, np.nan, dtype=float)
dL[m_all] = cosmo.luminosity_distance(z[m_all]).to(u.cm).value

L6716 = 4 * np.pi * dL**2 * F6717
L6731 = 4 * np.pi * dL**2 * F6731

# === 作図用マスク（log 軸用に L>0 のみ） ===
m6716 = m_all & (L6716 > 0)
m6731 = m_all & (L6731 > 0)

# === 図作成（2×2：上段6716, 下段6731 / 左列=logM*, 右列=logSFR） ===
fig, axes = plt.subplots(
    2, 2, figsize=(18, 9),
    sharex='col',  # 列ごとに x 共有
    sharey='row'   # 行ごとに y 共有
)
(ax11, ax12), (ax21, ax22) = axes

# --- 上段：L(6716) vs logM*, logSFR ---
ax11.scatter(logM[m6716],  L6716[m6716], s=1, color="C0", alpha=0.6)
ax12.scatter(logSFR[m6716], L6716[m6716], s=1, color="C0", alpha=0.6)

# --- 下段：L(6731) vs logM*, logSFR ---
ax21.scatter(logM[m6731],  L6731[m6731], s=1, color="C0", alpha=0.6)
ax22.scatter(logSFR[m6731], L6731[m6731], s=1, color="C0", alpha=0.6)

# --- 軸スケール ---
for ax in (ax11, ax12, ax21, ax22):
    ax.set_yscale("log")

# --- ラベル ---
ax11.set_ylabel("L([S II] 6716) [erg/s]")
ax21.set_ylabel("L([S II] 6731) [erg/s]")

ax21.set_xlabel(r"$\log(M_\ast/M_\odot)$")
ax22.set_xlabel(r"$\log(SFR) [M_{\odot}\mathrm{yr^{-1}}]$") 

# （上段の x ラベルは共有のため付けない）
ax11.tick_params(labelbottom=False)
ax12.tick_params(labelbottom=False)

# --- y-レンジ（これまでの設定に合わせる） ---
# x 範囲
ax11.set_xlim(6, 14)
ax12.set_xlim(-4, 4)
ax21.set_xlim(6, 14)
ax22.set_xlim(-4, 4)

# y 範囲
ax11.set_ylim(1e35, 1e42)
ax12.set_ylim(1e35, 1e42)
ax21.set_ylim(1e35, 1e42)
ax22.set_ylim(1e35, 1e42)

# --- 右列の y 目盛ラベルを消す（必要なら） ---
for ax in (ax12, ax22):
    ax.tick_params(labelleft=False)

# --- 枠線（spines） ---
for ax in (ax11, ax12, ax21, ax22):
    for spine in ax.spines.values():
        spine.set_linewidth(2)
        spine.set_color("black")

# --- 余白調整（ほぼ隙間なし） ---
plt.subplots_adjust(left=0.07, right=0.98, bottom=0.08, top=0.98, wspace=0.00, hspace=0.00)

# --- 保存 ---
savepath = os.path.join(current_dir, "results/figure/sii_luminosity_vs_sm_sfr_data.png")
plt.savefig(savepath, dpi=200)
print(f"Saved: {savepath}")
plt.show()