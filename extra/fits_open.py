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
fits_path =  "results/JADES/JADES_DR3/master/jades_dr3_mr_step3_shibuya.fits" 

# === FITSファイルを開く === #
# 重要な情報はhdul[1]の方にのっている
with fits.open(fits_path) as hdul:
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

    # # データの最初の5行を表示
    # print("最初の5行のデータ:")
    # for row in ext_hdu.data[:1]:
    #     print(row)