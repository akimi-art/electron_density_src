#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
スクリプトの概要:
このスクリプトは
JADESの銀河カタログとShibuya+2022のHST画像解析カタログをクロスマッチし、
JADESの銀河に対してShibuya+2022のサイズ（ReffUV, ReffOpt）を付与するものです。

使用方法:
    JADES_HST_crossmatch.py [オプション]

著者: A. M.
作成日: 2026-05-25

参考文献:
    - PEP 8: https://peps.python.org/pep-0008/
    - PEP 257 (Docstring規約): https://peps.python.org/pep-0257/
    - Python公式ドキュメント: https://docs.python.org/ja/3/
"""

import pandas as pd
import numpy as np
from io import StringIO

from astropy.coordinates import SkyCoord
import astropy.units as u

# =====================================================
# 1. JADES catalog 読み込み
# =====================================================

jades = pd.read_csv("results/JADES/JADES_DR3/data_from_Nishigaki/jades_info_with_HA_plus_logSFR.csv")

# =====================================================
# 2. Shibuya+15 catalog 読み込み
#    （VOTable風 TSV）
# =====================================================

fname = "data/data_HST/Shibuya_et_al_2015/tsv/Shibuya_et_al_2015_table4.tsv"

with open(fname, "r") as f:
    lines = f.readlines()

# -----------------------------------------------------
# <![CDATA[ の位置を探す
# -----------------------------------------------------

start = None

for i, line in enumerate(lines):
    if "<![CDATA[" in line:
        start = i + 1
        break

if start is None:
    raise ValueError("CDATA start not found")

# -----------------------------------------------------
# CDATA 部分のみ抽出
# -----------------------------------------------------

data_lines = []

for line in lines[start:]:

    if "]]>" in line:
        break

    data_lines.append(line)

csv_text = "".join(data_lines)

# -----------------------------------------------------
# pandas で読む
# -----------------------------------------------------

shibuya = pd.read_csv(
    StringIO(csv_text),
    sep=";"
)

# -----------------------------------------------------
# 単位行・区切り行を削除
# -----------------------------------------------------

shibuya = shibuya.iloc[3:].reset_index(drop=True)

# -----------------------------------------------------
# 列名の空白除去
# -----------------------------------------------------

shibuya.columns = shibuya.columns.str.strip()

# -----------------------------------------------------
# 数値化
# -----------------------------------------------------

cols = [
    "ReffUV", "e_ReffUV",
    "ReffOpt", "e_ReffOpt",
    "_RA", "_DE"
]

for c in cols:
    shibuya[c] = pd.to_numeric(
        shibuya[c],
        errors="coerce"
    )

# -----------------------------------------------------
# RA/DEC 欠損除去
# -----------------------------------------------------

shibuya = shibuya.dropna(subset=["_RA", "_DE"])

print("Shibuya catalog loaded")
print(shibuya.head())

# =====================================================
# 3. SkyCoord 作成
# =====================================================

coord_jades = SkyCoord(
    ra=jades["RA_TARG"].values * u.deg,
    dec=jades["Dec_TARG"].values * u.deg
)

coord_shibuya = SkyCoord(
    ra=shibuya["_RA"].values * u.deg,
    dec=shibuya["_DE"].values * u.deg
)

# =====================================================
# 4. クロスマッチ
# =====================================================

idx, d2d, _ = coord_jades.match_to_catalog_sky(
    coord_shibuya
)

# -----------------------------------------------------
# マッチ半径
# -----------------------------------------------------

max_sep = 0.5 * u.arcsec

matched = d2d < max_sep

print(f"Matched: {matched.sum()} / {len(jades)}")

# =====================================================
# 5. Reff 列追加
# =====================================================

jades["ReffUV"]    = np.nan
jades["e_ReffUV"]  = np.nan
jades["ReffOpt"]   = np.nan
jades["e_ReffOpt"] = np.nan

# -----------------------------------------------------
# マッチしたものだけ代入
# -----------------------------------------------------

jades.loc[matched, "ReffUV"] = (
    shibuya.iloc[idx[matched]]["ReffUV"].values
)

jades.loc[matched, "e_ReffUV"] = (
    shibuya.iloc[idx[matched]]["e_ReffUV"].values
)

jades.loc[matched, "ReffOpt"] = (
    shibuya.iloc[idx[matched]]["ReffOpt"].values
)

jades.loc[matched, "e_ReffOpt"] = (
    shibuya.iloc[idx[matched]]["e_ReffOpt"].values
)

# =====================================================
# 6. separation も保存（便利）
# =====================================================

jades["match_sep_arcsec"] = np.nan

jades.loc[matched, "match_sep_arcsec"] = (
    d2d[matched].arcsec
)

# =====================================================
# 7. 保存
# =====================================================

outname = "results/JADES/JADES_DR3/data_from_Nishigaki/jades_info_with_HA_plus_logSFR_with_Reff.csv"

jades.to_csv(
    outname,
    index=False
)

print(f"Saved: {outname}")