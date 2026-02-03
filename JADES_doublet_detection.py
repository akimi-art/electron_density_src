#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
スクリプトの概要:
このスクリプトは
JADESのカタログを使って
SII（補助的にOII）のスペクトル解析ができそうな
銀河を絞り込むものです。

使用方法:
    JADES_doublet_detection.py [オプション]

著者: A. M.
作成日: 2026-02-02

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
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u


# ==============================
# 1. FITS カタログを読む
# ==============================
current_dir = os.getcwd()
fits_file = os.path.join(current_dir, "results/JADES/JADES_NIRSpec_Gratings_Line_Fluxes_GOODS_S_DeepHST_v1.0/hlsp_jades_jwst_nirspec_goods-s-deephst_gratings_line-fluxes_v1.0_catalog.fits")

with fits.open(fits_file) as hdul:
    cat = hdul[1].data

df = pd.DataFrame(cat)


# ==============================
# 2. 検出条件を定義
# ==============================
snr_min = 3.0

detected = (
    (df["O2_3727_flux"] > 0) &
    (df["S2_6718_flux"] > 0) &
    (df["S2_6733_flux"] > 0) &
    (df["O2_3727_flux"] / df["O2_3727_err"] > snr_min) &
    (df["S2_6718_flux"] / df["S2_6718_err"] > snr_min) &
    (df["S2_6733_flux"] / df["S2_6733_err"] > snr_min)
)

df_ne = df[detected].copy()

print(f"ne解析候補天体数: {len(df_ne)}")

# ==============================
# 3. SII 比と誤差を計算
# ==============================
df_ne["SII_ratio"] = df_ne["S2_6718_flux"] / df_ne["S2_6733_flux"]

df_ne["SII_ratio_err"] = df_ne["SII_ratio"] * np.sqrt(
    (df_ne["S2_6718_err"] / df_ne["S2_6718_flux"])**2 +
    (df_ne["S2_6733_err"] / df_ne["S2_6733_flux"])**2
)

# ==============================
# 4. 明らかに非物理な値を除外（任意）
# ==============================
physical = (df_ne["SII_ratio"] > 0.4) & (df_ne["SII_ratio"] < 1.5)
df_ne = df_ne[physical]

print(f"物理的に妥当な天体数: {len(df_ne)}")


# ==============================
# 5.0 RA, Dec を sexagesimal 形式に変換
# ==============================
coord = SkyCoord(
    ra=df_ne["RA_NIRCam"].values * u.deg,
    dec=df_ne["Dec_NIRCam"].values * u.deg,
    frame="icrs"
)

df_ne["RA_hms"]  = coord.ra.to_string(unit=u.hour, sep=":", precision=2, pad=True)
df_ne["Dec_dms"] = coord.dec.to_string(unit=u.deg,  sep=":", precision=2, alwayssign=True, pad=True)


# ==============================
# 5. 必要な列だけ残す
# ==============================
cols = [
    "NIRSpec_ID",
    "z_Spec",
    'RA_NIRCam',
    'Dec_NIRCam',
    "RA_hms",      # ← 追加
    "Dec_dms",     # ← 追加
    "O2_3727_flux",
    "S2_6718_flux",
    "S2_6733_flux",
    "SII_ratio",
    "SII_ratio_err",
]

df_ne = df_ne[cols]


# ==============================
# 6. 保存（後でスペクトルと照合）
# ==============================
df_ne.to_csv("results/csv/JADES_ne_candidates.csv", index=False)
