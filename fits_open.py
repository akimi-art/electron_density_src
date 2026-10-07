#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
スクリプトの概要:
FITSファイルをオープンします。

使用方法:
    fits_open.py [オプション]

著者: A. M.
作成日: 2026-01-07

参考文献:
    - PEP 8:                  https://peps.python.org/pep-0008/
    - PEP 257 (Docstring規約): https://peps.python.org/pep-0257/
    - Python公式ドキュメント:    https://docs.python.org/ja/3/
"""

# === 必要なパッケージのインストール === #
import os
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table


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



# === ファイルパスを取得する === #file_path = os.path.join(current_dir, "results/JADES/JADES_NIRSpec_Gratings_Line_Fluxes_GOODS_S_DeepHST_v1.0/hlsp_jades_jwst_nirspec_goods-s-deephst_gratings_line-fluxes_v1.0_catalog.fits")
current_dir = os.getcwd()
file_galex =  "results/JADES/sample/jades_all_with_flags.fits" 

# === FITSファイルを開く === #
# 重要な情報はhdul[1]の方にのっている
with fits.open(file_galex) as hdul:
    # HDUの構造を表示
    hdul.info()

    # 0番目のHDU（通常はプライマリHDU）を取得
    primary_hdu = hdul[0]
    # 拡張HDU（通常 index 1）を取得
    ext_hdu = hdul[1]

   # プライマリHDUの情報を表示
    print("\n=== プライマリHDU ===")
    print("データの形状:", primary_hdu.data.shape if primary_hdu.data is not None else "None")
    # # 列名とデータ型を表示
    print("列名:", ext_hdu.columns.names)
    print("データ型:", ext_hdu.columns.formats)

    # データの最初の5行を表示
    print("最初の5行のデータ:")
    for row in ext_hdu.data[:1]:
        print(row)

# fits_path = file_galex  # ← ここをあなたのファイル名に

# with fits.open(fits_path) as hdul:
#     # テーブルが入っていそうな拡張HDUを探す
#     table_hdu = None
#     for hdu in hdul:
#         if hasattr(hdu, "data") and hdu.data is not None and hdu.header.get("XTENSION", "").upper() in ("BINTABLE", "TABLE"):
#             table_hdu = hdu
#             break

#     if table_hdu is None:
#         raise ValueError("テーブルHDUが見つかりません。'z_Spec'がヘッダーキーワードなら、ヒストグラムは作れません（単一値のため）。")

#     # z_Spec列を取得
#     cols = table_hdu.columns.names
#     if "z_Spec" not in cols:
#         raise KeyError(f"'z_Spec' 列が見つかりません。見つかった列名: {cols}")

#     z = np.array(table_hdu.data["z_Spec"], dtype=float)
#     z = z[np.isfinite(z)]  # NaNやinfを除去

# # ヒストグラムの描画
# fig, ax = plt.subplots(figsize=(6, 6))
# ax.hist(z, bins=30, color="#4e79a7", alpha=0.85, edgecolor="white")
# ax.set_xlabel("z_Spec", fontsize=20)
# ax.set_ylabel("Numbers", fontsize=20)
# ax.set_xlim(0, 10)
# ax.set_ylim(0, 300)
# plt.title("z_Spec jades dr3 gs", fontsize=20)
# # === 枠線 (spines) の設定 ===
# # 線の太さ・色・表示非表示などを個別に制御
# for spine in ax.spines.values():
#     spine.set_linewidth(2)       # 枠線の太さ
#     spine.set_color("black")     # 枠線の色
# plt.tight_layout()
# plt.show()


# from astropy.io import fits
# import numpy as np

# with fits.open(file_galex) as hdul:
#     h0 = hdul[0].header
#     data0 = hdul[0].data  # shape (3852, 5) の2D配列
#     print("Header of HDU 0:")
#     print(h0)

# # === オプション: 'PLATEID', 'FIBERID', の部分だけ抜き出す ===
# with fits.open(file_galex) as hdul:
#     # hdul[1] のデータを取得
#     data = hdul[1].data
    
#     # 'Z'列の最初の5行を抜き出す
#     PLATEID_values = data['tExp_G140M'][:1000]
#     # FIBERID_values = data['SII_6731_FLUX_ERR'][:100]
    
#     # 結果を表示
#     print(PLATEID_values)
#     # print(FIBERID_values)

# 結果（SDSS GALEX: Z）
# [0.0718 0.0217 0.171  0.052  0.0963 0.1718 0.0671 0.0839 0.2054 0.204
#  0.1282 0.2073 0.1383 0.2074 0.0378 0.0282 0.0218 0.2297 0.135  0.2216]



# from astropy.io import fits
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# # FITSファイル
# fits_path = file_galex

# # 読み込み
# with fits.open(fits_path) as hdul:
#     data = hdul[1].data

# # DataFrame化
# df = pd.DataFrame({
#     "flux_6731": np.array(data["SII_6731_FLUX"], dtype=float),
#     "fluxerr_6731": np.array(data["SII_6731_FLUX_ERR"], dtype=float),
# })

# # 有効な値のみ
# mask = (
#     np.isfinite(df["flux_6731"]) &
#     np.isfinite(df["fluxerr_6731"]) &
#     (df["fluxerr_6731"] > 0)
# )

# df = df[mask].copy()

# # S/N
# df["sn_6731"] = df["flux_6731"] / df["fluxerr_6731"]

# # 基本情報
# print(f"Number of valid measurements: {len(df)}")
# print()
# print(df[["flux_6731", "fluxerr_6731", "sn_6731"]].describe())

# # 何σ以上が何個あるか
# for threshold in [1, 2, 3, 5, 10]:
#     n = (df["sn_6731"] >= threshold).sum()
#     print(f"S/N >= {threshold}: {n} ({n / len(df) * 100:.1f}%)")

# # --------------------------------------------------
# # S/N の分布
# # --------------------------------------------------

# plt.figure(figsize=(7, 5))

# plt.hist(
#     df["sn_6731"],
#     bins=100,
#     range=(-5, 20)
# )

# plt.axvline(3, linestyle="--", label="3σ")
# plt.axvline(5, linestyle="--", label="5σ")

# plt.xlabel(r"[S II] $\lambda6731$ S/N")
# plt.ylabel("Number of galaxies")
# plt.legend()
# plt.tight_layout()
# plt.show()

# # --------------------------------------------------
# # flux と S/N の関係
# # --------------------------------------------------

# plt.figure(figsize=(7, 5))

# plt.scatter(
#     df["flux_6731"],
#     df["sn_6731"],
#     s=5,
#     alpha=0.3
# )

# plt.axhline(3, linestyle="--", label="3σ")
# plt.axhline(5, linestyle="--", label="5σ")

# plt.xscale("log")
# plt.xlabel(r"[S II] $\lambda6731$ flux [erg s$^{-1}$ cm$^{-2}$]")
# plt.ylabel(r"S/N")
# plt.ylim(0, 10)
# plt.legend()
# plt.tight_layout()
# plt.show()



# # =====================================
# # JADES z_Spec histogram
# # =====================================

# t = Table.read(file_galex, format="fits")
# df = t.to_pandas()
# z_spec = df["z_Spec"].values

# mask = np.isfinite(z_spec) & (z_spec > 0)

# z_spec = z_spec[mask]

# fig, ax = plt.subplots(figsize=(10, 6))

# ax.hist(
#     z_spec,
#     bins=30,
#     color="firebrick",
#     edgecolor="black",
#     alpha=0.8,
# )

# z_median = np.median(z_spec)

# ax.axvline(
#     z_median,
#     color="k",
#     linestyle="--",
#     linewidth=2,
#     label=fr"Median = {z_median:.2f}"
# )

# ax.set_xlabel(r"$z_{\rm spec}$")
# ax.set_ylabel("Number of galaxies")
# ax.set_xlim(0, np.max(z_spec))

# ax.legend()

# for spine in ax.spines.values():
#     spine.set_linewidth(2)

# plt.tight_layout()
# plt.show()


# # =====================================
# # Statistics
# # =====================================

# print("\n===== JADES z_Spec Statistics =====")

# print(f"N = {len(z_spec):,}")

# print(f"Mean     = {np.mean(z_spec):.3f}")
# print(f"Median   = {np.median(z_spec):.3f}")
# print(f"Std      = {np.std(z_spec):.3f}")

# print(f"Min      = {np.min(z_spec):.3f}")
# print(f"Max      = {np.max(z_spec):.3f}")

# p16, p50, p84 = np.percentile(
#     z_spec,
#     [16, 50, 84]
# )

# print(
#     f"16/50/84 percentile = "
#     f"{p16:.3f}, {p50:.3f}, {p84:.3f}"
# )

# print(
#     f"Median -16/+84 = "
#     f"{p50-p16:.3f} / +{p84-p50:.3f}"
# )

# print("===============================\n")


# import numpy as np
# from astropy.table import Table

# merged_path = "results/fits/mpajhu_dr7_v5_2_merged_radius.fits"
# line_path   = "data/data_SDSS/DR7/fits_files/gal_line_dr7_v5_2.fit"
# out_path    = "results/fits/mpajhu_dr7_v5_2_merged_radius.fits"

# tm = Table.read(merged_path)
# tl = Table.read(line_path)
# N = len(tm)

# add_cols = ["H_ALPHA_EQW", "H_ALPHA_EQW_ERR", "NII_6584_EQW", "NII_6584_EQW_ERR"]

# # ------------------------------------------------------------
# # 1. 行数
# # ------------------------------------------------------------
# assert len(tl) == N, f"行数が違います: merged {N}, gal_line {len(tl)}"

# # ------------------------------------------------------------
# # 2. plate と fiber が全行で一致するか
# # ------------------------------------------------------------
# for k in ["PLATEID", "FIBERID"]:
#     same = np.array_equal(np.asarray(tm[k]), np.asarray(tl[k]))
#     print(f"{k:8s} が全行で一致: {same}")
#     assert same, f"{k} が一致しません"

# # ------------------------------------------------------------
# # 3. 共通する値の列が全行で一致するか（MJD の代わりの確認）
# # ------------------------------------------------------------
# check_cols = [c for c in ["H_ALPHA_FLUX", "NII_6584_FLUX", "SII_6731_FLUX"]
#               if c in tm.colnames and c in tl.colnames]
# print("値で確かめる列:", check_cols)
# assert check_cols, "共通する値の列がありません"
# for c in check_cols:
#     a = np.asarray(tm[c], dtype=float)
#     b = np.asarray(tl[c], dtype=float)
#     same = np.array_equal(a, b, equal_nan=True)
#     print(f"{c:15s} が全行で一致: {same}")
#     assert same, f"{c} が一致しません"

# # ------------------------------------------------------------
# # 4. 行の対応が確かめられたので、列を追加する
# # ------------------------------------------------------------
# for c in add_cols:
#     tm[c] = tl[c]

# # ------------------------------------------------------------
# # 5. 確認と保存
# # ------------------------------------------------------------
# assert len(tm) == N, "行数が変わっています"
# assert np.array_equal(np.asarray(tm["ROW_ID"]), np.arange(N)), "行の順番が変わっています"

# eqw = np.asarray(tm["H_ALPHA_EQW"], dtype=float)
# print("H_ALPHA_EQW の 5/50/95 %点:", np.nanpercentile(eqw, [5, 50, 95]))

# tm.write(out_path, overwrite=True)
# print(f"[DONE] {out_path}（{N:,} 行）")


# # =====================================================================
# # check_replace_mass.py
# #   MPA/JHU の質量ファイル v5_2b を確認し、妥当なら merged の sm_* 列を差し替える
# #   1 回目：REPLACE = False で確認だけ → 出力と図を見る
# #   2 回目：REPLACE = True で差し替えたファイルを書き出す
# #   方針：行は消さない。旧版の値は *_v5_2 という列名で残す
# # =====================================================================
# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from astropy.table import Table

# REPLACE = False                       # 確認後に True にする

# current_dir = os.getcwd()
# mpa_path = os.path.join(current_dir, "results/fits/mpajhu_dr7_v5_2_merged_radius.fits")
# new_path = os.path.join(current_dir, "data/data_SDSS/DR7/fits_files/totlgm_dr7_v5_2b.fit")
# out_path = os.path.join(current_dir, "results/fits/mpajhu_dr7_v5_2_merged_radius_lgm2b.fits")
# fig_dir  = os.path.join(current_dir, "results/figure/sample")
# os.makedirs(fig_dir, exist_ok=True)
# N_EXPECTED = 927552

# def f64(a):
#     return np.ma.filled(np.ma.asarray(a).astype(float), np.nan)

# # =====================================
# # 1. 読み込み：行数と列の対応
# #    totlgm には plate/mjd/fiber が無いので、gal_info と同じ行順であることを前提にする
# #    → 下の 3. で「旧版で値がある行は新旧がほぼ一致する」ことで行順を検証する
# # =====================================
# t   = Table.read(mpa_path)
# new = Table.read(new_path, hdu=1)
# N = len(t)
# print("merged の行数:", N, "  v5_2b の行数:", len(new))
# assert N == N_EXPECTED and len(new) == N, "行数が一致しない → 行順の対応が取れないので差し替え不可"
# assert np.array_equal(np.asarray(t["ROW_ID"]), np.arange(N))

# print("v5_2b の列:", new.colnames)
# pairs = [(c, f"sm_{c}") for c in new.colnames if f"sm_{c}" in t.colnames]
# print("対応する merged の列:", [p[1] for p in pairs])
# assert ("MEDIAN", "sm_MEDIAN") in pairs, "MEDIAN 列の対応が見つからない → 列名を確認"

# old = f64(t["sm_MEDIAN"])
# nw  = f64(new["MEDIAN"])

# # =====================================
# # 2. 欠損（-1 など）の数：旧版と新版
# # =====================================
# def count(v, name):
#     print(f"  {name}: -1 = {np.sum(v == -1):,}   NaN = {np.sum(np.isnan(v)):,}   "
#           f"< 6（-1 以外）= {np.sum((v < 6) & (v != -1)):,}   > 13 = {np.sum(v > 13):,}   "
#           f"6–13 = {np.sum((v >= 6) & (v <= 13)):,}")

# print("\n===== 欠損の数 =====")
# count(old, "旧 v5_2 ")
# count(nw,  "新 v5_2b")
# ok_old = np.isfinite(old) & (old >= 6) & (old <= 13)
# ok_new = np.isfinite(nw)  & (nw  >= 6) & (nw  <= 13)
# print(f"  旧で欠損 → 新で妥当（救われた行）: {np.sum(~ok_old & ok_new):,}")
# print(f"  旧で妥当 → 新で欠損（失われた行）: {np.sum(ok_old & ~ok_new):,}")

# # =====================================
# # 3. 行順の検証：旧版で値がある行は新旧がほぼ一致するはず
# # =====================================
# both = ok_old & ok_new
# d = nw[both] - old[both]
# print("\n===== 両方で妥当な行の差（新 - 旧）=====")
# print(f"  行数: {both.sum():,}")
# print(f"  完全一致: {np.mean(d == 0):.4f}   |差| < 0.01: {np.mean(np.abs(d) < 0.01):.4f}   "
#       f"|差| > 0.1: {np.sum(np.abs(d) > 0.1):,}")
# print(f"  差の 1/50/99%: {np.percentile(d, [1, 50, 99])}")

# # =====================================
# # 4. 救われた行の性質：質量がある行と同じ M*–SFR 関係に乗るか
# # =====================================
# z      = f64(t["Z"])
# logSFR = f64(t["sfr_MEDIAN"])
# sfr_ok = np.isfinite(logSFR) & (logSFR > -10) & (logSFR < 3)
# rescued = ~ok_old & ok_new
# ref     = ok_old & ok_new

# print("\n===== 救われた行の性質 =====")
# for name, m in [("救われた行", rescued), ("元から妥当な行", ref)]:
#     print(f"  {name:10s}: log M* 1/50/99% = {np.nanpercentile(nw[m], [1, 50, 99]).round(2)}   "
#           f"z 1/50/99% = {np.nanpercentile(z[m & (z > 0)], [1, 50, 99]).round(3)}")

# bins = np.arange(-2.0, 2.01, 0.5)
# print("  log SFR のビンごとの log M* 中央値（救われた行 / 元から妥当な行）")
# for lo, hi in zip(bins[:-1], bins[1:]):
#     a = rescued & sfr_ok & (logSFR >= lo) & (logSFR < hi)
#     b = ref     & sfr_ok & (logSFR >= lo) & (logSFR < hi)
#     if a.sum() > 20:
#         print(f"    {lo:+.1f} -- {hi:+.1f}: {np.median(nw[a]):.2f} (N={a.sum():,})  /  "
#               f"{np.median(nw[b]):.2f} (N={b.sum():,})")

# # =====================================
# # 5. 図
# # =====================================
# fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# ax = axes[0]   # 新旧の比較（行順が合っていれば対角線上）
# ax.hexbin(old[both], nw[both], gridsize=150, bins="log", cmap="Greys", extent=(7, 12.5, 7, 12.5))
# ax.plot([7, 12.5], [7, 12.5], "r-", lw=1)
# ax.set_xlabel(r"$\log M_\ast$ (v5_2)"); ax.set_ylabel(r"$\log M_\ast$ (v5_2b)")

# ax = axes[1]   # 質量の分布
# hb = np.arange(6, 13, 0.1)
# ax.hist(nw[ref], bins=hb, histtype="step", color="k", lw=2, density=True, label="original")
# ax.hist(nw[rescued], bins=hb, histtype="step", color="firebrick", lw=2, density=True, label="rescued in v5_2b")
# ax.set_xlabel(r"$\log M_\ast$ (v5_2b)"); ax.set_ylabel("normalized"); ax.legend()

# ax = axes[2]   # M*–SFR（同じ関係に乗るか）
# m1, m2 = ref & sfr_ok, rescued & sfr_ok
# ax.scatter(nw[m1], logSFR[m1], s=0.1, alpha=0.05, color="gray", rasterized=True)
# ax.scatter(nw[m2], logSFR[m2], s=0.3, alpha=0.2, color="firebrick", rasterized=True)
# ax.set_xlim(7, 12.5); ax.set_ylim(-3, 2.5)
# ax.set_xlabel(r"$\log M_\ast$ (v5_2b)"); ax.set_ylabel(r"$\log$ SFR")

# plt.tight_layout()
# path = os.path.join(fig_dir, "mass_v52_vs_v52b.png")
# plt.savefig(path, dpi=150, bbox_inches="tight"); plt.show()
# print(f"[DONE] {path}")

# # =====================================
# # 6. 差し替え（REPLACE = True のときだけ）
# # =====================================
# if REPLACE:
#     for c_new, c_old in pairs:
#         t.rename_column(c_old, f"{c_old}_v5_2")              # 旧版の値は残す
#         t[c_old] = f64(new[c_new])
#     t["MASS_FROM_v52b_RESCUED"] = rescued                    # 救われた行の印
#     assert len(t) == N and np.array_equal(np.asarray(t["ROW_ID"]), np.arange(N))
#     t.write(out_path, overwrite=True)
#     print(f"\n[DONE] 差し替え済み: {out_path}")
#     print("  置き換えた列:", [p[1] for p in pairs], " 旧版は *_v5_2 として保存")
# else:
#     print("\n[INFO] 確認のみ（REPLACE = False）。結果が妥当なら True にして再実行")


# sfr = f64(t["sfr_MEDIAN"])
# m1 = sfr == -1.0
# zz = z > 0
# print(f"log SFR = -1 ちょうど: {m1.sum():,}")
# print(f"  z 1/50/99%: {np.percentile(z[m1 & zz], [1, 50, 99]).round(3)}")
# print(f"  z < 0.1879 の中: {np.sum(m1 & zz & (z < 0.1879)):,}")
# print(f"  うち v5_2b で質量が埋まった行: {np.sum(m1 & rescued):,}")
# for c in [c for c in t.colnames if c.startswith("sfr_")]:
#     print(f"  {c}: -1 ちょうど = {np.sum(f64(t[c]) == -1.0):,}")



# # =====================================================================
# # fig_mass_v52b.py
# #   質量カタログ v5_2 → v5_2b の差し替えが妥当であることを示す図
# #   (a) 既存の値は変わっていない
# #   (b) 新しく値が入った行の大半は z > Z_MAX（解析の体積の外）
# #   (c) 体積内で値が入った銀河は、もとからある銀河と同じ M*–SFR 関係に乗る
# # =====================================================================
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from astropy.table import Table
# from astropy.cosmology import FlatLambdaCDM
# import astropy.units as u
# from scipy.optimize import brentq

# cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
# L_MIN, FLUX_LIMIT = 1e39, 1e-17

# current_dir = os.getcwd()
# path    = os.path.join(current_dir, "results/fits/mpajhu_dr7_v5_2_merged_lgm2b.fits")
# fig_dir = os.path.join(current_dir, "results/figure/sample")
# os.makedirs(fig_dir, exist_ok=True)

# def f64(a):
#     return np.ma.filled(np.ma.asarray(a).astype(float), np.nan)

# t = Table.read(path)
# old    = f64(t["sm_MEDIAN_v5_2"])
# new    = f64(t["sm_MEDIAN"])
# z      = f64(t["Z"])
# logSFR = f64(t["sfr_MEDIAN"])

# Z_MAX = brentq(lambda zz: 4*np.pi*cosmo.luminosity_distance(zz).to(u.cm).value**2*FLUX_LIMIT - L_MIN,
#                1e-4, 1.0)

# valid_old = np.isfinite(old) & (old > 6) & (old < 13)
# valid_new = np.isfinite(new) & (new > 6) & (new < 13)
# both      = valid_old & valid_new
# rescued   = ~valid_old & valid_new
# original  = valid_old
# zok       = np.isfinite(z) & (z > 0)
# sfr_ok    = np.isfinite(logSFR) & (logSFR > -10) & (logSFR < 3) & (logSFR != -1.0)

# fig, axes = plt.subplots(1, 3, figsize=(30, 8.5))
# TXT = dict(fontsize=22, va="top", transform=None)

# # ---------------- (a) 既存の値は変わらない ----------------
# ax = axes[0]
# d = new[both] - old[both]
# ax.hist(d, bins=np.linspace(-0.02, 0.02, 81), color="gray", edgecolor="k", log=True)
# ax.set_xlabel(r"$\log M_\ast$(v5_2b) $-$ $\log M_\ast$(v5_2)")
# ax.set_ylabel("Number of spectra")
# ax.set_title("(a) Existing values are unchanged", fontsize=26, loc="left")
# ax.text(0.04, 0.95,
#         f"N = {both.sum():,}\nidentical: {np.mean(d == 0)*100:.2f}%\n"
#         f"$|\\Delta| > 0.01$: {np.sum(np.abs(d) > 0.01)}",
#         transform=ax.transAxes, fontsize=22, va="top")

# # ---------------- (b) 埋まった行の大半は体積の外 ----------------
# ax = axes[1]
# zb = np.linspace(0, 0.7, 71)
# ax.hist(z[original & zok], bins=zb, histtype="step", color="k", lw=2.5, label="mass in v5_2")
# ax.hist(z[rescued & zok],  bins=zb, histtype="step", color="firebrick", lw=2.5, label="new in v5_2b")
# ax.axvline(Z_MAX, color="k", ls="--", lw=2)
# ax.axvspan(0, Z_MAX, color="tab:blue", alpha=0.08)
# ax.text(Z_MAX + 0.01, 0.97, r"$z_{\rm max}$", transform=ax.get_xaxis_transform(), fontsize=24, va="top")
# n_in  = np.sum(rescued & zok & (z < Z_MAX))
# n_out = np.sum(rescued & zok & (z >= Z_MAX))
# ax.text(0.45, 0.80, f"new in v5_2b:\n  $z < z_{{\\rm max}}$: {n_in:,}\n  $z \\geq z_{{\\rm max}}$: {n_out:,}",
#         transform=ax.transAxes, fontsize=22, va="top", color="firebrick")
# ax.set_xlim(0, 0.7)
# ax.set_xlabel(r"$z$"); ax.set_ylabel("Number of spectra")
# ax.set_title("(b) Most new values are outside our volume", fontsize=26, loc="left")
# ax.legend(fontsize=20, loc="upper right")

# # ---------------- (c) 体積内の埋まった銀河は普通の M*–SFR 関係 ----------------
# ax = axes[2]
# vol = zok & (z < Z_MAX) & sfr_ok
# m_o = original & vol
# m_r = rescued & vol
# ax.hexbin(new[m_o], logSFR[m_o], gridsize=120, bins="log", cmap="Greys",
#           extent=(7, 12.5, -3, 2.5), mincnt=1)
# ax.scatter(new[m_r], logSFR[m_r], s=1.5, alpha=0.3, color="firebrick", rasterized=True)
# ax.scatter([], [], s=40, color="firebrick", label=f"new in v5_2b (N = {m_r.sum():,})")
# ax.scatter([], [], s=40, marker="h", color="gray", label=f"mass in v5_2 (N = {m_o.sum():,})")
# n_sfr_m1 = np.sum(rescued & zok & (z < Z_MAX) & (logSFR == -1.0))
# ax.text(0.04, 0.95, f"$z < z_{{\\rm max}}$ only\nSFR = $-1$ (no estimate) excluded: {n_sfr_m1:,}",
#         transform=ax.transAxes, fontsize=20, va="top")
# ax.set_xlim(7, 12.5); ax.set_ylim(-3, 2.5)
# ax.set_xlabel(r"$\log(M_\ast/M_\odot)$ (v5_2b)"); ax.set_ylabel(r"$\log({\rm SFR}/M_\odot\,{\rm yr}^{-1})$")
# ax.set_title("(c) New galaxies in our volume follow the same relation", fontsize=26, loc="left")
# ax.legend(fontsize=18, loc="lower right")

# for ax in axes:
#     for s in ax.spines.values():
#         s.set_linewidth(2)

# plt.tight_layout()
# out = os.path.join(fig_dir, "mass_v52b_validation.png")
# plt.savefig(out, dpi=200, bbox_inches="tight"); plt.show()
# print(f"[DONE] {out}")


# === JADES BPT diagram ===
from pathlib import Path

root = Path(__file__).resolve().parent.parent
t = Table.read(root / "results/JADES/sample/jades_all_with_flags.fits")

def values(name):
    return np.ma.filled(np.ma.asarray(t[name]).astype(float), np.nan)

mask = np.asarray(t["BASE_SAMPLE"], dtype=bool).copy()
flux = {}

for line in ["N2_6584", "HA_6563", "O3_5007", "HB_4861"]:
    f = values(f"{line}_flux")
    flux[line] = f
    mask &= f > 0

print(f"[BPT] BASE_SAMPLE のうち4輝線すべてのフラックス > 0: {mask.sum()} 天体")

x = np.log10(flux["N2_6584"][mask] / flux["HA_6563"][mask])
y = np.log10(flux["O3_5007"][mask] / flux["HB_4861"][mask])

fig, ax = plt.subplots(figsize=(10, 9))
if mask.any():
    points = ax.scatter(x, y, c=values("z_Spec")[mask],
                        cmap="viridis", edgecolors="black", s=50)
    fig.colorbar(points, ax=ax, label=r"$z_{\rm spec}$")

# 比較用の局所銀河の境界線
xx = np.linspace(-2.5, 0.35, 400)
ax.plot(xx, 0.61 / (xx - 0.47) + 1.19, "k-", label="Kewley+2001")
xx = np.linspace(-2.5, -0.05, 400)
ax.plot(xx, 0.61 / (xx - 0.05) + 1.30, "k--", label="Kauffmann+2003")

ax.set_xlabel(r"$\log_{10}([\mathrm{N\,II}]6584/\mathrm{H}\alpha)$")
ax.set_ylabel(r"$\log_{10}([\mathrm{O\,III}]5007/\mathrm{H}\beta)$")
ax.set_title(f"JADES BPT (N = {mask.sum()}, all four fluxes > 0)")
ax.legend()
fig.tight_layout()

out = root / "results/JADES/figure/sample/bpt_nii_JADES.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"[DONE] {out}")
plt.show()