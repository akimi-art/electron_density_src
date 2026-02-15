#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
スクリプトの概要:
このスクリプトは
FITSファイルの必要な要素のみを抽出して
txtファイルを作るものです。
他のコードとの兼ね合いがなければ
csvファイルを作った方が便利かもしれません。

使用方法:
    python fits_extraction.py [オプション]

著者: A. M.
作成日: 2026-01-29

参考文献:
    - PEP 8: https://peps.python.org/pep-0008/
    - PEP 257 (Docstring規約): https://peps.python.org/pep-0257/
    - Python公式ドキュメント: https://docs.python.org/ja/3/
    - Elbaz et al. (2007)
    - Chen et al. (2016)
"""


# == 必要なパッケージのインストール == #
import re
import os
from astropy.io import fits
import pandas as pd

current_dir = os.getcwd()

# ファイルの読み込み
file_path = os.path.join(current_dir, "results/JADES/JADES_NIRSpec_Gratings_Line_Fluxes_GOODS_S_DeepHST_v1.0/hlsp_jades_jwst_nirspec_goods-s-deephst_gratings_line-fluxes_v1.0_catalog.fits")
output_path = os.path.join(current_dir, 'results/txt/hlsp_jades_jwst_nirspec_goods-s-deephst_gratings_line-fluxes_v1.0_catalog.txt')

# FITSファイルを開く
with fits.open(file_path) as hdul:
    data = hdul[1].data  # BinTableHDUを取得
    col1 = 'NIRSpec_ID'
    col2 = 'z_Spec'
    col3 = 'S2_6718_flux'
    col4 = 'S2_6718_err'
    col5 = 'S2_6733_flux'
    col6 = 'S2_6733_err'
    with open(output_path, 'w', encoding='utf-8') as f:
        for row in data:
            val1 = row[col1]
            val2 = row[col2]
            val3 = row[col3]
            val4 = row[col4]
            val5 = row[col5]
            val6 = row[col6]
            f.write(f"{val1}\t{val2}\t{val3}\t{val4}\t{val6}\t{val6}\n")