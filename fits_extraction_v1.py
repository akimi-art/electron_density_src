#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
スクリプトの概要:
このスクリプトは
FITSファイルの必要な要素のみを抽出して
csvファイルを作るものです。

使用方法:
    python fits_extraction_v1.py [オプション]

著者: A. M.
作成日: 2026-02-15

参考文献:
    - PEP 8: https://peps.python.org/pep-0008/
    - PEP 257 (Docstring規約): https://peps.python.org/pep-0257/
    - Python公式ドキュメント: https://docs.python.org/ja/3/
    - Elbaz et al. (2007)
    - Chen et al. (2016)
"""


# == 必要なパッケージのインストール == #
import os
from astropy.io import fits
import pandas as pd
import numpy as np

current_dir = os.getcwd()

file_path = os.path.join(
    current_dir,
    "results/JADES/JADES_DR3/catalog/jades_dr3_medium_gratings_public_gs_v1.1.fits"
)

output_path = os.path.join(
    current_dir,
    "results/csv/jades_dr3_medium_gratings_public_gs_v1.1_sii.csv"
)

# FITS読み込み
with fits.open(file_path) as hdul:
    data = hdul[1].data

    df = pd.DataFrame({
        "NIRSpec_ID": data["NIRSpec_ID"],
        "z_Spec": data["z_Spec"],
        "S2_6718_flux": data["S2_6718_flux"],
        "S2_6718_err": data["S2_6718_err"],
        "S2_6733_flux": data["S2_6733_flux"],
        "S2_6733_err": data["S2_6733_err"],
    })

# S/N 追加
df["S2_6718_SN"] = df["S2_6718_flux"] / df["S2_6718_err"]
df["S2_6733_SN"] = df["S2_6733_flux"] / df["S2_6733_err"]

# 比も追加（超重要）
df["S2_ratio"] = df["S2_6718_flux"] / df["S2_6733_flux"]

# 無限・NaN除去
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# 保存
df.to_csv(output_path, index=False)

print("Saved to:", output_path)
