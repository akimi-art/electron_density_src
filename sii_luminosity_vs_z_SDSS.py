#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
スクリプトの概要:
SIIのLumiosityとzの関係を描画します。
単にデータを描画するだけではなく、
フラックスのSN比の情報も含まれます。
これはSDSS用です。


使用方法:
    sii_luminoisity_vs_z_SDSS.py [オプション]

著者: A. M.
作成日: 2026-02-16

参考文献:
    - PEP 8:                  https://peps.python.org/pep-0008/
    - PEP 257 (Docstring規約): https://peps.python.org/pep-0257/
    - Python公式ドキュメント:    https://docs.python.org/ja/3/
    - Curti+17
    - Matplotlib colormap:    https://jp.matplotlib.net/stable/tutorials/colors/colormaps.html#google_vignette
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
from astropy.cosmology import Planck18 as cosmo
from astropy.table import Table, Column
from matplotlib.colors import TwoSlopeNorm
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize



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

# # === FITS 読み込み（拡張HDUにテーブルがある場合はこれが簡単） ===
# # 例: "mpajhu_dr7_v5_2_merged.fits"
# # CSV読み込み
# current_dir = os.getcwd()
# fits_path = os.path.join(current_dir,"results/fits/mpajhu_dr7_v5_2_merged.fits")
# t = Table.read(fits_path, format="fits")
# # csv_path = "./results/Samir16/Samir16in_standard_re_v1.csv"

# # pandasに変換（後続のコードをそのまま使うため）
# df = t.to_pandas()
# # df = pd.read_csv(csv_path)

# # 以降は元コードと同じ
# # --- 単位スケールの補正（MPA-JHU: 1e-17 erg s^-1 cm^-2 想定）---
# UNIT_FLUX = 1e-17  # 必要に応じてヘッダで確認

# z = df["Z"].values
# F6716 = df["SII_6717_FLUX"].values * UNIT_FLUX
# F6731 = df["SII_6731_FLUX"].values * UNIT_FLUX

# err6716 = (df["SII_6717_FLUX_ERR"]) * UNIT_FLUX
# err6731 = (df["SII_6731_FLUX_ERR"]) * UNIT_FLUX

# sn6716 = F6716 / err6716
# sn6731 = F6731 / err6731

# # Luminosity
# d_L = cosmo.luminosity_distance(z).to(u.cm).value
# L6716 = 4 * np.pi * d_L**2 * F6716
# L6731 = 4 * np.pi * d_L**2 * F6731

# fig, axes = plt.subplots(2,1, figsize=(12,8), sharex=True)
# fig.subplots_adjust(wspace=0, hspace=0)

# ax1, ax2 = axes

# norm = TwoSlopeNorm(vcenter=0.0, vmin=-3, vmax=5)
# sc1 = ax1.scatter(
#     z, L6716,
#     c=sn6716,
#     # Perceptually Uniform Sequential（知覚的に均一・連続）がデフォルト
#     # viridis最も推奨。色覚差別にも強い。
#     # plasma暖色より、明るく見える。
#     # inferno暖色ベース、暗い背景に相性良い。
#     # magma黒→赤→黄色、天文画像でも人気。
#     # cividis色覚バリアフリーに最適。
#     # 人間は色相差より 明度差 に敏感なため、Sequentialはお勧めしない
#     # Sequential（連続色）、Greys, Purples, Blues, Greens, Oranges, Redsなど
#     cmap="coolwarm",
#     norm=norm,
#     s=15,
#     alpha=0.8
# )

# sc2 = ax2.scatter(
#     z, L6731,
#     c=sn6731,
#     cmap="coolwarm",
#     s=15,
#     alpha=0.8,
#     norm=norm
# )

# for ax in axes:
#     ax.set_yscale("log")
#     ax.set_xlim(0,0.4)
#     ax.set_ylim(1e30,1e42) 
#     # ax.set_ylim(1e30,1e50) # 一時的に超広くとる

# # =======================================
# # 追加: Plot L(SII 6717, flux一定のライン)
# # =======================================

# # 描画のためのzのグリッド
# z_grid = np.linspace(0.0, 0.4, 200)  # 適当なz範囲をグリッド化
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

# # # Completeなサンプルを確かめるための線
# # ax1.axhline(y=1e38, color='black', linestyle='--', linewidth=0.5)
# # ax1.axhline(y=1e39, color='black', linestyle='--', linewidth=0.5)
# # ax2.axhline(y=1e38, color='black', linestyle='--', linewidth=0.5)
# # ax2.axhline(y=1e39, color='black', linestyle='--', linewidth=0.5)
# # ax1.axvline(x=0.15, color='black', linestyle='--', linewidth=0.5)
# # ax2.axvline(x=0.15, color='black', linestyle='--', linewidth=0.5)

# # === 枠線 (spines) の設定 ===
# # 線の太さ・色・表示非表示などを個別に制御
# for spine in ax1.spines.values():
#     spine.set_linewidth(2)       # 枠線の太さ
#     spine.set_color("black")     # 枠線の色
# for spine in ax2.spines.values():
#     spine.set_linewidth(2)       # 枠線の太さ
#     spine.set_color("black")     # 枠線の色

# ax1.set_ylabel("L(SII 6716)")
# ax2.set_ylabel("L(SII 6731)")
# ax2.set_xlabel("z")

# # カラーバー
# # 参考: https://jp.matplotlib.net/stable/tutorials/colors/colormaps.html#google_vignette
# cbar = fig.colorbar(sc1, ax=axes, label="S/N")

# save_path = os.path.join(current_dir, "results/figure/sii_luminosity_vs_z_SDSS_data.png")
# plt.savefig(save_path)
# print(f"Saved as {save_path}.")
# plt.show()

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from astropy.table import Table
import astropy.units as u
from astropy.cosmology import Planck18 as cosmo  # 必要に応じて変更

# === FITS 読み込み（拡張HDUにテーブルがある場合はこれが簡単） ===
# 例: "mpajhu_dr7_v5_2_merged.fits"
current_dir = os.getcwd()
fits_path = os.path.join(current_dir, "results/fits/mpajhu_dr7_v5_2_merged.fits")
t = Table.read(fits_path, format="fits")

# pandasに変換（後続のコードをそのまま使うため）
df = t.to_pandas()

# --- 単位スケールの補正（MPA-JHU: 1e-17 erg s^-1 cm^-2 想定）---
UNIT_FLUX = 1e-17  # 必要に応じてヘッダで確認

# データ抽出
z = df["Z"].values
F6716 = df["SII_6717_FLUX"].values * UNIT_FLUX
err6716 = df["SII_6717_FLUX_ERR"].values * UNIT_FLUX
sn6716 = F6716 / err6716

# === Luminosity 計算 ===
d_L = cosmo.luminosity_distance(z).to(u.cm).value  # [cm]
L6716 = 4 * np.pi * d_L**2 * F6716                # [erg s^-1]

# === 描画 ===
fig, ax = plt.subplots(1, 1, figsize=(12, 6))

# S/N のカラースケール（中央0, vmin=-5, vmax=5は元コードに準拠）
norm = TwoSlopeNorm(vcenter=0, vmin=-3, vmax=5)
sc = ax.scatter(
    z, L6716,
    c=sn6716,
    cmap="coolwarm",
    norm=norm,
    s=15,
    alpha=0.8
)


# 軸スケール・範囲（元コード準拠）
ax.set_yscale("log")
ax.set_xlim(0, 0.4)
ax.set_ylim(1e30, 1e42)  # 必要なら 1e50 まで暫定拡大可
# plt.axhline(y=1e42, color='blue', linestyle='-', linewidth=5)
# plt.axvline(x=0.0, color='blue', linestyle='-', linewidth=5)
# plt.axvline(x=0.4, color='blue', linestyle='-', linewidth=5)

# =======================================
# 追加: L([S II] 6716) の一定フラックス線
# =======================================
z_grid = np.linspace(0.0, 0.4, 200)
d_L_grid = cosmo.luminosity_distance(z_grid).to(u.cm).value
# 観測フラックス一定（erg s^-1 cm^-2）
for flux, ls, lw, color in zip(
    [1e-19, 1e-18, 1e-17],      # フラックス
    ['--', '-.', '-'],          # ラインの種類
    [2.0, 2.0, 5],             # ★ linewidth を各線で別々に指定
    ["black", "black", "blue"]   # ★ 色も別々に指定（最後の線を赤にして目立たせる）
):
    L_const = 4 * np.pi * d_L_grid**2 * flux
    ax.plot(z_grid, L_const, color=color, linestyle=ls, linewidth=lw)






# （前略）あなたのループ
for flux, ls, lw, color in zip(
    [1e-19, 1e-18, 1e-17],
    ['--',    '-.',   '-'],
    [2.0,     2.0,    5.0],
    ["black", "black","blue"]
):
    L_const = 4 * np.pi * d_L_grid**2 * flux
    ax.plot(z_grid, L_const, color=color, linestyle=ls, linewidth=lw)

# ===============================
# 1e-17 のラインより上側を塗る
# ===============================
# 1e-17 の L(z) を再計算（または上のループ内で保存しておいてもOK）
flux_thr = 1e-17
L_const_thr = 4 * np.pi * d_L_grid**2 * flux_thr

# # 上端（塗りつぶしの上側境界）を決める
# # すでに set_ylim を使っているなら、その上限を利用
# # まだの場合は、想定レンジ（例：1e42）などの正値を明示
# y_upper = 1e42  # 例：既に ax.set_ylim(1e35, 1e42) 済みなら 1e42 が入る

# # log 軸でもOK：正の値であれば fill_between は機能します
# ax.fill_between(
#     z_grid,
#     L_const_thr,
#     y_upper,
#     where=np.isfinite(L_const_thr),  # 念のため NaN/inf を除外
#     color="blue",
#     alpha=0.10,           # 塗りの濃さ（お好みで）
#     interpolate=True,
#     zorder=0,             # 点群や線の下に塗る（上に重ねたいなら大きく）
# )









# === 枠線 (spines) の設定 ===
for spine in ax.spines.values():
    spine.set_linewidth(2)
    spine.set_color("black")

# ラベル・カラーバー
ax.set_ylabel(r"L([S II] 6716 [erg s$^{-1}$]")
ax.set_xlabel("z")
cbar = fig.colorbar(sc, ax=ax, label="S/N")

# 保存
save_path = os.path.join(current_dir, "results/figure/sii6716_luminosity_vs_z_SDSS_data.png")
plt.savefig(save_path, bbox_inches="tight", dpi=200)
print(f"Saved as {save_path}.")
plt.show()

# # =============================================================
# # 密度マップを作成する
# # =============================================================
# # =============================================================
# # ヘルパー：一定フラックス線の光度
# # =============================================================
# def L_from_F_const(z, F_const_cgs):
#     dL = cosmo.luminosity_distance(z).to(u.cm).value
#     return 4*np.pi * dL**2 * F_const_cgs

# # =============================================================
# # 2段レイアウト：6716 と 6731 の密度マップを上下に隙間なく
# # =============================================================
# def plot_density_maps_stacked(
#     z, L6716, L6731, *,
#     zlim=(0.0, 0.40),
#     Llim=(1e30, 1e42),
#     gridsize=140,
#     flux_lines=(1e-19, 1e-18, 1e-17),
#     cmap='magma',
#     figscale=(12, 8),
#     linewidth_spine=2,
#     spine_color="black",
#     add_legend=False
# ):
#     """
#     [S II] 6716 と 6731 の L–z 密度マップを上下に隙間なく配置して描画する。

#     Parameters
#     ----------
#     z : array
#     L6716, L6731 : array
#         光度 [erg/s]。log軸にするため L>0 のみ採用。
#     zlim, Llim : tuple
#         軸範囲。
#     gridsize : int
#         hexbin の解像度。
#     flux_lines : tuple of float
#         一定フラックス [erg s^-1 cm^-2] を重ね描き。
#     cmap : str
#         密度カラーマップ。
#     figscale : (w, h)
#         図サイズ（インチ）。
#     linewidth_spine : float
#         枠線の太さ。
#     spine_color : str
#         枠線の色。
#     add_legend : bool
#         一定フラックス線の凡例を表示するか。
#     """
#     # ===== マスク（有限＆正） =====
#     m1 = np.isfinite(z) & np.isfinite(L6716) & (L6716 > 0)
#     m2 = np.isfinite(z) & np.isfinite(L6731) & (L6731 > 0)
#     z1, L1 = z[m1], L6716[m1]
#     z2, L2 = z[m2], L6731[m2]

#     # ===== Figure / GridSpec =====
#     fig = plt.figure(figsize=figscale)
#     gs = gridspec.GridSpec(nrows=2, ncols=1, figure=fig, hspace=0.0)  # ← 隙間ゼロ
#     ax_top = fig.add_subplot(gs[0, 0])
#     ax_bot = fig.add_subplot(gs[1, 0], sharex=ax_top)  # x共有

#     # ====== 上段：6716 ======
#     hb1 = ax_top.hexbin(
#         z1, L1, gridsize=gridsize,
#         xscale='linear', yscale='log',
#         bins='log', cmap=cmap, mincnt=1
#     )
#     cbar1 = fig.colorbar(hb1, ax=ax_top)
#     cbar1.set_label('Count')

#     zgrid = np.linspace(zlim[0], zlim[1], 400)
#     for i, F0 in enumerate(flux_lines):
#         ax_top.plot(
#             zgrid, L_from_F_const(zgrid, F0),
#             color='k', lw=1.6, ls=['--','-.',':'][i % 3], alpha=0.9,
#             label=f'F={F0:.0e} cgs'
#         )

#     ax_top.set_xlim(*zlim)
#     ax_top.set_ylim(*Llim)
#     ax_top.set_yscale('log')
#     ax_top.set_ylabel(r'L([S II] 6716) [erg s$^{-1}$]')
#     ax_top.tick_params(axis='x', labelbottom=False)  # 上段のx目盛ラベルを隠す

#     # ====== 下段：6731 ======
#     hb2 = ax_bot.hexbin(
#         z2, L2, gridsize=gridsize,
#         xscale='linear', yscale='log',
#         bins='log', cmap=cmap, mincnt=1
#     )
#     cbar2 = fig.colorbar(hb2, ax=ax_bot)
#     cbar2.set_label('Count')

#     for i, F0 in enumerate(flux_lines):
#         ax_bot.plot(
#             zgrid, L_from_F_const(zgrid, F0),
#             color='k', lw=1.6, ls=['--','-.',':'][i % 3], alpha=0.9
#         )

#     ax_bot.set_xlim(*zlim)
#     ax_bot.set_ylim(*Llim)
#     ax_bot.set_yscale('log')
#     ax_bot.set_xlabel('z')
#     ax_bot.set_ylabel(r'L([S II] 6731) [erg s$^{-1}$]')

#     # ===== 枠線（spines） =====
#     for ax in (ax_top, ax_bot):
#         for spine in ax.spines.values():
#             spine.set_linewidth(linewidth_spine)
#             spine.set_color(spine_color)

#     # 凡例（必要なら）
#     if add_legend:
#         ax_top.legend(loc='lower right', fontsize=10, frameon=True)

#     plt.tight_layout()
#     return fig, (ax_top, ax_bot)

# fig, (ax3, ax4) = plot_density_maps_stacked(
#     z=z,
#     L6716=L6716,
#     L6731=L6731,
#     zlim=(0.0, 0.40),
#     Llim=(1e30, 1e42),
#     gridsize=140,
#     flux_lines=(1e-19, 1e-18, 1e-17),
#     cmap='magma',
#     figscale=(12, 8),        # 縦を少し大きめに
#     linewidth_spine=2,
#     spine_color="black",
#     add_legend=False
# )

# # 保存
# save_path = os.path.join(current_dir, "results/figure/sii_luminosity_vs_z_SDSS_v1_density_stacked.png")
# plt.savefig(save_path, dpi=200, bbox_inches='tight')
# print(f"Saved as {save_path}.")
# plt.show()

# # ============================
# #  基準線より上側の SII6717 を抽出し、新しい FITS を保存
# # ============================
# # ============================
# #  Lベースの完全サンプル抽出（構造そのまま・行のみ削除）— 両線同時版
# # ============================
# # --- パラメータ ---
# # 一定フラックス [erg s^-1 cm^-2]（図の基準線に対応）
# F_CONST_6717_CGS = 1e-17
# F_CONST_6731_CGS = 1e-17     # 6717と同じにしてよければ同値のままでOK（別々に設定可能）
# Z_RANGE = (0.0, 0.40)        # 図と揃える場合。全 z を許容するなら None
# REQUIRE_FINITE = True        # 数値の健全性チェック（NaN/inf除外）

# # --- L_lim(z) を計算（一定フラックス線） ---
# # すでに z が配列としてあり、cosmo は Planck18 想定
# dL_each = cosmo.luminosity_distance(z).to(u.cm).value
# Llim6717_each = 4 * np.pi * dL_each**2 * F_CONST_6717_CGS
# Llim6731_each = 4 * np.pi * dL_each**2 * F_CONST_6731_CGS

# # --- マスク作成（両線の L >= L_lim(z) を同時に満たす） ---
# mask_L_6717 = (L6716 >= Llim6717_each)
# mask_L_6731 = (L6731 >= Llim6731_each)
# mask_L_both = mask_L_6717 & mask_L_6731

# # z 範囲の適用（必要に応じて）
# if Z_RANGE is not None:
#     zmin, zmax = Z_RANGE
#     mask_z = np.isfinite(z) & (z >= zmin) & (z <= zmax)
# else:
#     mask_z = np.ones_like(z, dtype=bool)

# # 数値の健全性（NaN/inf の排除）
# if REQUIRE_FINITE:
#     mask_finite = (
#         np.isfinite(z) &
#         np.isfinite(L6716) & np.isfinite(L6731) &
#         np.isfinite(Llim6717_each) & np.isfinite(Llim6731_each)
#     )
# else:
#     mask_finite = np.ones_like(z, dtype=bool)

# # --- 最終マスク（列構造は触らない） ---
# select_mask = mask_L_both & mask_z & mask_finite

# print(f"[INFO] 抽出件数（両線同時）: {select_mask.sum()} / {len(select_mask)}")
# if select_mask.sum() == 0:
#     print("[WARN] 0 件です。F_CONST_* や Z_RANGE を見直してください。")

# # --- Table を行スライスのみで抽出（列・メタデータ保持） ---
# t_sel = t[select_mask]  # ← 列構造は一切変更しない

# # --- 書き出し（ファイル名に条件を明記） ---
# def _sci_notation(x):
#     return f"{x:.0e}".replace("+","")

# suffix_parts = [
#     f"L6717_ge_4pi_dL2_{_sci_notation(F_CONST_6717_CGS)}",
#     f"L6731_ge_4pi_dL2_{_sci_notation(F_CONST_6731_CGS)}",
# ]
# if Z_RANGE is not None:
#     suffix_parts.append(f"z{zmin:.2f}-{zmax:.2f}")
# suffix = "_".join(suffix_parts)

# out_dir = os.path.join(current_dir, "results", "fits")
# os.makedirs(out_dir, exist_ok=True)
# out_path = os.path.join(out_dir, f"mpajhu_dr7_v5_2_merged_{suffix}.fits")

# t_sel.write(out_path, format="fits", overwrite=True)
# print(f"[DONE] 書き出し完了: {out_path}")