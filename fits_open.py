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

# test


# === 必要なパッケージのインストール === #
import os
from astropy.io import fits

# === ファイルパスを取得する === #file_path = os.path.join(current_dir, "results/JADES/JADES_NIRSpec_Gratings_Line_Fluxes_GOODS_S_DeepHST_v1.0/hlsp_jades_jwst_nirspec_goods-s-deephst_gratings_line-fluxes_v1.0_catalog.fits")
current_dir = os.getcwd()
file_galex =  "./results/fits/mpajhu_dr7_v5_2_merged.fits"

# === FITSファイルを開く === #
# 重要な情報はhdul[1]の方にのっている
with fits.open(file_galex) as hdul:
    # HDUの構造を表示
    hdul.info()

    # 拡張HDU（通常 index 1）を取得
    ext_hdu = hdul[1]

    # # 列名とデータ型を表示
    print("列名:", ext_hdu.columns.names)
    print("データ型:", ext_hdu.columns.formats)

    # データの最初の5行を表示
    print("最初の5行のデータ:")
    for row in ext_hdu.data[:1]:
        print(row)


# # === オプション: 'PLATEID', 'FIBERID', の部分だけ抜き出す ===
# with fits.open(file_galex) as hdul:
#     # hdul[1] のデータを取得
#     data = hdul[1].data
    
#     # 'Z'列の最初の5行を抜き出す
#     PLATEID_values = data['SII_6717_FLUX_ERR'][:1000]
#     # FIBERID_values = data['SII_6731_FLUX_ERR'][:100]
    
#     # 結果を表示
#     print(PLATEID_values)
#     # print(FIBERID_values)

# 結果（SDSS GALEX: Z）
# [0.0718 0.0217 0.171  0.052  0.0963 0.1718 0.0671 0.0839 0.2054 0.204
#  0.1282 0.2073 0.1383 0.2074 0.0378 0.0282 0.0218 0.2297 0.135  0.2216]