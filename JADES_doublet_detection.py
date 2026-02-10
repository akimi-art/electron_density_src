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
fits_file = os.path.join(current_dir, "results/JADES/JADES_DR4/catalog/Combined_DR4_external_v1.2.1.fits")

with fits.open(fits_file) as hdul:
    cat = hdul[1].data

df = pd.DataFrame(cat)


# ==============================
# 2. 検出条件を定義
# ==============================
# snr_min = 3.0
# DR4のカタログにはerrの情報がない
# zの情報で選定してみよう（6.7~12.8）
# あとG395Hで観測されている（T）という情報も添えて

detected = (
    (df["[OII]3727"] > 0) &
    (df['z_Spec'] > 6.7) & 
    (df['z_Spec'] < 12.8) &
    (df['assigned_G395H'] == "T")
    # (df["O2_3727_flux"] > 0) &
    # (df["S2_6718_flux"] > 0) &
    # (df["S2_6733_flux"] > 0) &
    # (df["O2_3727_flux"] / df["O2_3727_err"] > snr_min) 
    # (df["S2_6718_flux"] / df["S2_6718_err"] > snr_min) &
    # (df["S2_6733_flux"] / df["S2_6733_err"] > snr_min)
)

df_ne = df[detected].copy()

print(f"ne解析候補天体数: {len(df_ne)}")

# ==============================
# 3. SII 比と誤差を計算
# ==============================
# df_ne["SII_ratio"] = df_ne["S2_6718_flux"] / df_ne["S2_6733_flux"]

# df_ne["SII_ratio_err"] = df_ne["SII_ratio"] * np.sqrt(
#     (df_ne["S2_6718_err"] / df_ne["S2_6718_flux"])**2 +
#     (df_ne["S2_6733_err"] / df_ne["S2_6733_flux"])**2
# )

# ==============================
# 4. 明らかに非物理な値を除外（任意）
# ==============================
# physical = (df_ne["SII_ratio"] > 0.4) & (df_ne["SII_ratio"] < 1.5)
# df_ne = df_ne[physical]

# print(f"物理的に妥当な天体数: {len(df_ne)}")


# ==============================
# 5.0 RA, Dec を sexagesimal 形式に変換
# ==============================
coord = SkyCoord(
    # ra=df_ne["RA_NIRCam"].values * u.deg, # DR4カタログにはない
    # dec=df_ne["Dec_NIRCam"].values * u.deg,
    ra=df_ne["RA_TARG"].values * u.deg,
    dec=df_ne["Dec_TARG"].values * u.deg,
    frame="icrs"
)

df_ne["RA_hms"]  = coord.ra.to_string(unit=u.hour, sep=":", precision=2, pad=True)
df_ne["Dec_dms"] = coord.dec.to_string(unit=u.deg,  sep=":", precision=2, alwayssign=True, pad=True)


# ==============================
# 5. 必要な列だけ残す
# ==============================
# Deepest(Keywards)
# ['NIRSpec_ID', 'NIRCam_ID', 'RA_TARG', 'Dec_TARG', 'RA_NIRCam',
#  'Dec_NIRCam', 'Priority', 'z_Spec', 'z_Spec_flag', 'x_offset', 
#  'y_offset', 'nexp_Prism', 'nexp_R1000', 'pre_JWST_priority', 
#  'z_R1000', 'LyA_1216_flux', 'LyA_1216_err', 'C4_1549_flux', 
#  'C4_1549_err', 'C3_1907_flux', 'C3_1907_err', 'He2_1640_flux', 
#  'He2_1640_err', 'O3_1666_flux', 'O3_1666_err', 'O2_3727_flux', 
#  'O2_3727_err', 'Ne3_3869_flux', 'Ne3_3869_err', 'Ne3_3968_flux', 
#  'Ne3_3968_err', 'HD_4102_flux', 'HD_4102_err', 'HG_4340_flux', 
#  'HG_4340_err', 'O3_4363_flux', 'O3_4363_err', 'HB_4861_flux', 
#  'HB_4861_err', 'O3_4959_flux', 'O3_4959_err', 'O3_5007_flux', 
#  'O3_5007_err', 'He1_5875_flux', 'He1_5875_err', 'O1_6300_flux', 
#  'O1_6300_err', 'HA_6563_flux', 'HA_6563_err', 'N2_6584_flux', 
#  'N2_6584_err', 'S2_6718_flux', 'S2_6718_err', 'S2_6733_flux', 
#  'S2_6733_err', 'S3_9069_flux', 'S3_9069_err', 'S3_9532_flux', 
#  'S3_9532_err', 'PaD_10049_flux', 'PaD_10049_err', 'He1_10829_flux', 
#  'He1_10829_err', 'PaG_10938_flux', 'PaG_10938_err', 'PaB_12818_flux', 
#  'PaB_12818_err', 'PaA_18751_flux', 'PaA_18751_err']
cols = [
    "NIRSpec_ID",
    "z_Spec",

    # 座標'RA_TARG', 'Dec_TARG',
    "RA_TARG",
    "Dec_TARG",
    "RA_hms",
    "Dec_dms",

    # 観測情報（超重要）
    "assigned_G140M",
    "assigned_G235M",
    "assigned_G395M",
    "assigned_G395H",

    # line flux
    "[OII]3727",
]
# medium
# cols = [
#     "NIRSpec_ID",
#     "z_Spec",

#     # 座標
#     "RA_NIRCam",
#     "Dec_NIRCam",
#     "RA_hms",
#     "Dec_dms",

#     # 観測情報（超重要）
#     "assigned_G140M",
#     "assigned_G235M",
#     "assigned_G395M",
#     "assigned_G395H",
#     "tExp_G140M",
#     "tExp_G235M",

#     # line flux
#     "O2_3727_flux",
#     "S2_6718_flux",
#     "S2_6733_flux",

#     # ratio
#     "SII_ratio",
#     "SII_ratio_err",
# ]


df_ne = df_ne[cols]


# ==============================
# 6. 保存（後でスペクトルと照合）
# ==============================
df_ne.to_csv("results/csv/JADES_ne_candidates_dr4_oii.csv", index=False)
