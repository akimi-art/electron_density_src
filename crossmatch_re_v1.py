#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
スクリプトの概要:
FITSファイル同士をクロスマッチします。
RADECでのマッチに切り替えました。

使用方法:
    crossmatch_re_v1.py [オプション]

著者: A. M.
作成日: 2026-05-25

参考文献:
    - PEP 8:                  https://peps.python.org/pep-0008/
    - PEP 257 (Docstring規約): https://peps.python.org/pep-0257/
    - Python公式ドキュメント:    https://docs.python.org/ja/3/
"""

from astropy.table import Table
import pandas as pd
import numpy as np

from astropy.coordinates import SkyCoord
import astropy.units as u

# ======================
# 1. データ読み込み
# ======================

# ✅ data1 → FITSのまま
data1 = Table.read("./results/fits/mpajhu_dr7_v5_2_merged_zlt0.2_Lgt1e+39.fits")

# ✅ data2 → CSV
data2 = pd.read_csv("./data/data_SDSS/DR7/csv_files/SDSS_galaxy_radius.csv")

# ======================
# 2. numpy配列に変換（ここが核心）
# ======================

ra1 = np.array(data1["RA"])
dec1 = np.array(data1["DEC"])

ra2 = data2["ra"].values
dec2 = data2["dec"].values

# ======================
# 3. 念のためクリーニング（重要）
# ======================

mask1 = (
    np.isfinite(ra1) &
    np.isfinite(dec1) &
    (dec1 >= -90) & (dec1 <= 90)
)

mask2 = (
    np.isfinite(ra2) &
    np.isfinite(dec2) &
    (dec2 >= -90) & (dec2 <= 90)
)

ra1, dec1 = ra1[mask1], dec1[mask1]
ra2, dec2 = ra2[mask2], dec2[mask2]

data1 = data1[mask1]
data2 = data2.iloc[mask2].reset_index(drop=True)

# ======================
# 4. SkyCoord
# ======================

c1 = SkyCoord(ra=ra1*u.deg, dec=dec1*u.deg)
c2 = SkyCoord(ra=ra2*u.deg, dec=dec2*u.deg)

# ======================
# 5. 最近傍マッチ
# ======================

idx, d2d, _ = c1.match_to_catalog_sky(c2)

# ======================
# 6. マッチ条件
# ======================

max_sep = 1.0 * u.arcsec
mask = d2d < max_sep

print("マッチ数:", np.sum(mask))
print("マッチ率:", np.mean(mask))

# ======================
# 7. 値をdata1に追加（ここ重要）
# ======================

# 新しい列を作る（Astropy Table）
data1["deVRad_r"] = np.full(len(data1), np.nan)
data1["expRad_r"] = np.full(len(data1), np.nan)
data1["fracDeV_r"] = np.full(len(data1), np.nan)

# 対応する値を代入
data1["deVRad_r"][mask] = data2.iloc[idx[mask]]["deVRad_r"].values
data1["expRad_r"][mask] = data2.iloc[idx[mask]]["expRad_r"].values
data1["fracDeV_r"][mask] = data2.iloc[idx[mask]]["fracDeV_r"].values

# ======================
# 8. Re計算
# ======================

Re = np.where(
    data1["fracDeV_r"] > 0.5,
    data1["deVRad_r"],
    1.678 * data1["expRad_r"]
)

data1["Re"] = Re

# ======================
# 9. 保存（FITSのまま）
# ======================
data1.write("./results/fits/mpajhu_dr7_v5_2_merged_zlt0.2_Lgt1e+39_radius.fits", overwrite=True)

print("✅ 完了")
print("中央値距離:", np.median(d2d.arcsec))