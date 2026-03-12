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
file_galex =  "./results/JADES/JADES_DR3/catalog/jades_dr3_medium_gratings_public_gs_v1.1.fits"

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

    # データの最初の1行を表示
    # print("最初の1行のデータ:")
    # for row in ext_hdu.data[:5]:
    #     print(row)

fits_path = file_galex  # ← ここをあなたのファイル名に

with fits.open(fits_path) as hdul:
    # テーブルが入っていそうな拡張HDUを探す
    table_hdu = None
    for hdu in hdul:
        if hasattr(hdu, "data") and hdu.data is not None and hdu.header.get("XTENSION", "").upper() in ("BINTABLE", "TABLE"):
            table_hdu = hdu
            break

    if table_hdu is None:
        raise ValueError("テーブルHDUが見つかりません。'z_Spec'がヘッダーキーワードなら、ヒストグラムは作れません（単一値のため）。")

    # z_Spec列を取得
    cols = table_hdu.columns.names
    if "z_Spec" not in cols:
        raise KeyError(f"'z_Spec' 列が見つかりません。見つかった列名: {cols}")

    z = np.array(table_hdu.data["z_Spec"], dtype=float)
    z = z[np.isfinite(z)]  # NaNやinfを除去

# ヒストグラムの描画
fig, ax = plt.subplots(figsize=(6, 6))
ax.hist(z, bins=30, color="#4e79a7", alpha=0.85, edgecolor="white")
ax.set_xlabel("z_Spec", fontsize=20)
ax.set_ylabel("Numbers", fontsize=20)
ax.set_xlim(0, 10)
ax.set_ylim(0, 300)
plt.title("z_Spec jades dr3 gs", fontsize=20)
# === 枠線 (spines) の設定 ===
# 線の太さ・色・表示非表示などを個別に制御
for spine in ax.spines.values():
    spine.set_linewidth(2)       # 枠線の太さ
    spine.set_color("black")     # 枠線の色
plt.tight_layout()
plt.show()


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