#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
スクリプトの概要:
SIIのLumiosityとzの関係を描画します。
単にデータを描画するだけではなく、
フラックスのSN比の情報も含まれます。
これはJADES用です。


使用方法:
    sii_luminoisity_vs_z_JADES.py [オプション]

著者: A. M.
作成日: 2026-02-16

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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import astropy.units as u
from matplotlib.colors import TwoSlopeNorm
from astropy.table import Table, Column
from astropy.cosmology import Planck18 as cosmo

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


# === FITS 読み込み（拡張HDUにテーブルがある場合はこれが簡単） ===
# 例: "mpajhu_dr7_v5_2_merged.fits"
t = Table.read("./results/JADES/JADES_DR3/catalog/jades_dr3_medium_gratings_public_gs_v1.1.fits", format="fits")
# pandasに変換（後続のコードをそのまま使うため）
df = t.to_pandas()
# df = pd.read_csv("./results/csv/JADES_DR3_GOODS-N_SII_ratio_only.csv")


# # 以降は元コードと同じ
# # --- 単位スケールの補正（MPA-JHU: 1e-17 erg s^-1 cm^-2 想定）---
# # UNIT_FLUX = 1e-17  # 必要に応じてヘッダで確認、カタログ
# # ＊JADEの場合
# # フラックス密度はすでに erg s^-1 cm^-2 Å^-1
# # 「単位換え」は不要。数値の見やすさのためにスケーリングしたいだけなら、
# # ラベルも合わせて「×10^-20（per Å）」スケールに“表示目的”で揃える
# # README↓
# # 「Measured emission line flux from the Prism/Clear spectrum in units of x10^-20 erg s-1 cm-2」
# UNIT_FLUX = 1e-20  # 必要に応じてヘッダで確認

# z = df["z_Spec"].values
# # F6716 = df["S2_6718_flux"].values * UNIT_FLUX
# # F6731 = df["S2_6733_flux"].values * UNIT_FLUX

# # err6716 = (df["S2_6718_err"]) * UNIT_FLUX
# # err6731 = (df["S2_6733_err"]) * UNIT_FLUX

# F6716 = df["S2_6716_flux"].values * UNIT_FLUX
# F6731 = df["S2_6730_flux"].values * UNIT_FLUX

# err6716 = 0.5 * (df["S2_6716_err_plus"] + df["S2_6716_err_minus"]) * UNIT_FLUX
# err6731 = 0.5 * (df["S2_6730_err_plus"] + df["S2_6730_err_minus"]) * UNIT_FLUX

# sn6716 = F6716 / err6716
# sn6731 = F6731 / err6731

# # Luminosity
# d_L = cosmo.luminosity_distance(z).to(u.cm).value
# L6716 = 4 * np.pi * d_L**2 * F6716
# L6731 = 4 * np.pi * d_L**2 * F6731

# # fig, axes = plt.subplots(2,1, figsize=(8,8), sharex=True)
# fig = plt.figure(figsize=(18, 10))
# gs = fig.add_gridspec(2, 1, 
#                       left=0.10, right=0.90, bottom=0.06, top=0.98,  # マージンを詰める
#                       wspace=0.00, hspace=0.00)

# axes = []
# for i in range(2):
#     for j in range(1):
#         axes.append(fig.add_subplot(gs[i, j]))

# ax1, ax2 = axes[0], axes[1]

# norm = TwoSlopeNorm(vcenter=0.0, vmin=-3, vmax=5)

# sc1 = ax1.scatter(
#     z, L6716,
#     c=sn6716,
#     cmap="coolwarm",
#     s=15,
#     norm=norm,
#     alpha=0.8
# )

# sc2 = ax2.scatter(
#     z, L6731,
#     c=sn6731,
#     cmap="coolwarm",
#     s=15,
#     norm=norm,
#     alpha=0.8
# )


# # 描画のためのzのグリッド
# z_grid = np.linspace(0.0, 7.0, 1000)  # 適当なz範囲をグリッド化
# d_L_grid = cosmo.luminosity_distance(z_grid).to(u.cm).value
# L_6717_1 = 4 * 3.141592653589793 * d_L_grid**2 * 1e-19
# L_6717_2 = 4 * 3.141592653589793 * d_L_grid**2 * 1e-18
# L_6717_3 = 4 * 3.141592653589793 * d_L_grid**2 * 1e-17
# L_6731_1 = 4 * 3.141592653589793 * d_L_grid**2 * 1e-19
# L_6731_2 = 4 * 3.141592653589793 * d_L_grid**2 * 1e-18
# L_6731_3 = 4 * 3.141592653589793 * d_L_grid**2 * 1e-17
# lum_6717_flux_const_1 = L_6717_1
# lum_6717_flux_const_2 = L_6717_2
# lum_6717_flux_const_3 = L_6717_3
# lum_6731_flux_const_1 = L_6731_1
# lum_6731_flux_const_2 = L_6731_2
# lum_6731_flux_const_3 = L_6731_3

# ax1.plot(z_grid, lum_6717_flux_const_1, color='black', linestyle='--', linewidth=2) # '-', '--', '-.', ':'が使える
# ax1.plot(z_grid, lum_6717_flux_const_2, color='black', linestyle='-.', linewidth=2)
# ax1.plot(z_grid, lum_6717_flux_const_3, color='black', linestyle=':' , linewidth=2)
# ax2.plot(z_grid, lum_6731_flux_const_1, color='black', linestyle='--', linewidth=2) # '-', '--', '-.', ':'が使える
# ax2.plot(z_grid, lum_6731_flux_const_2, color='black', linestyle='-.', linewidth=2)
# ax2.plot(z_grid, lum_6731_flux_const_3, color='black', linestyle=':' , linewidth=2)

# for ax in axes:
#     ax.set_yscale("log")
#     ax.set_xlim(0, 7.0) 
#     ax.set_ylim(1e20,1e50)
# for ax in axes[:1]:
#     ax.tick_params(labelbottom=False)

# ax1.set_ylabel("L(SII 6716)")
# ax2.set_ylabel("L(SII 6731)")
# ax2.set_xlabel("z")

# # カラーバー
# cbar = fig.colorbar(sc1, ax=axes, label="S/N")
# # save_path = "./results/figure/LSII_vs_z_JADES_DR3_GOODS-S_fit.png"
# save_path = "./results/figure/LSII_vs_z_JADES_DR3_GOODS-S_fit.png"

# plt.savefig(save_path)
# print(f"Saved as {save_path}.")
# plt.show()


# あるフラックス一定の線より上側のみのデータを抽出

# ============================
# パラメータ
# ============================
F_CONST_6717_CGS = 1e-20
F_CONST_6731_CGS = 1e-20
Z_RANGE = None   # None にすれば全z
REQUIRE_FINITE = True

# 必要列（例）
# df に以下がある前提：
# z, L6716, L6731

# ============================
# L_lim(z) 計算
# ============================
dL_each = cosmo.luminosity_distance(z).to(u.cm).value
Llim6717_each = 4 * np.pi * dL_each**2 * F_CONST_6717_CGS
Llim6731_each = 4 * np.pi * dL_each**2 * F_CONST_6731_CGS

# ============================
# マスク作成
# ============================
mask_L_6717 = (L6716 >= Llim6717_each)
mask_L_6731 = (L6731 >= Llim6731_each)
mask_L_both = mask_L_6717 & mask_L_6731

# z 範囲
if Z_RANGE is not None:
    zmin, zmax = Z_RANGE
    mask_z = np.isfinite(z) & (z >= zmin) & (z <= zmax)
else:
    mask_z = np.ones_like(z, dtype=bool)

# 数値健全性
if REQUIRE_FINITE:
    mask_finite = (
        np.isfinite(z) &
        np.isfinite(L6716) & np.isfinite(L6731) &
        np.isfinite(Llim6717_each) & np.isfinite(Llim6731_each)
    )
else:
    mask_finite = np.ones_like(z, dtype=bool)

# ============================
# 最終マスク
# ============================
select_mask = mask_L_both & mask_z & mask_finite

print(f"[INFO] 抽出件数（両線同時）: {select_mask.sum()} / {len(df)}")

# ============================
# DataFrame 行抽出（列構造はそのまま）
# ============================
df_sel = df.loc[select_mask].copy()

# ============================
# 保存
# ============================
def _sci_notation(x):
    return f"{x:.0e}".replace("+","")

suffix_parts = [
    f"L6717_ge_4pi_dL2_{_sci_notation(F_CONST_6717_CGS)}",
    f"L6731_ge_4pi_dL2_{_sci_notation(F_CONST_6731_CGS)}",
]
if Z_RANGE is not None:
    suffix_parts.append(f"z{zmin:.2f}-{zmax:.2f}")
suffix = "_".join(suffix_parts)

out_dir = "./results/csv"
os.makedirs(out_dir, exist_ok=True)

out_path = os.path.join(out_dir, f"JADES_DR3_GOODS-N_SII_ratio_only_{suffix}.csv")
df_sel.to_csv(out_path, index=False)

print(f"[DONE] 書き出し完了: {out_path}")
