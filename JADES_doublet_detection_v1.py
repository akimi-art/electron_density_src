#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
スクリプトの概要:
このスクリプトは
JADES DR4のカタログを使って
SII（補助的にOII）のスペクトル解析ができそうな
銀河を絞り込むものです。

使用方法:
    JADES_doublet_detection_v1.py [オプション]

著者: A. M.
作成日: 2026-02-15

参考文献:
    - PEP 8: https://peps.python.org/pep-0008/
    - PEP 257 (Docstring規約): https://peps.python.org/pep-0257/
    - Python公式ドキュメント: https://docs.python.org/ja/3/
    - Elbaz et al. (2007)
    - Chen et al. (2016)
    - Kiyota et al. (2026)
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
# 0. ID を指定（ここを自分で変える）
# ==============================
# Kiyota et al. (2026)のTable1参照
target_ids = [14279, 209357, 209108, 50039680, 10007444,
              209777, 208134, 208267, 10008071]

# ==============================
# 1. 指定IDだけ抽出
# ==============================
df_sel = df[df["NIRSpec_ID"].isin(target_ids)].copy()

print("Selected objects:", len(df_sel))

# ==============================
# 2. [S II] 観測波長を計算
# ==============================
lambda_rest = 6723.0  # Å

df_sel["SII_obs_A"] = lambda_rest * (1 + df_sel["z_Spec"])

# ==============================
# 3. グレーティング判定
# ==============================
def judge_grating(wave_A):

    if 10000 <= wave_A <= 18000:
        return "G140M"
    elif 17000 <= wave_A <= 31000:
        return "G235M"
    elif 29000 <= wave_A <= 51000:
        return "G395M/H"
    else:
        return "Out of range"

df_sel["SII_grating_from_z"] = df_sel["SII_obs_A"].apply(judge_grating)

# ==============================
# 4. 必要カラムだけ残す
# ==============================
cols = [
    "NIRSpec_ID",
    "z_Spec",
    "z_Spec_flag",

    "RA_TARG",
    "Dec_TARG",

    "assigned_G140M",
    "assigned_G235M",
    "assigned_G395M",
    "assigned_G395H",

    "tExp_G140M",
    "tExp_G235M",
    "tExp_G395M",
    "tExp_G395H",

    "[OII]3727",

    "SII_obs_A",
    "SII_grating_from_z"
]

df_sel = df_sel[cols]

# ==============================
# 5. 保存
# ==============================
df_sel.to_csv("results/csv/JADES_DR4_kiyota_2026_selected_IDs.csv", index=False)

print("Saved: JADES_DR4_kiyota_2026_selected_IDs.csv")

