#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
スクリプトの概要:
FITSファイル同士をクロスマッチします。

使用方法:
    crossmatch_re.py [オプション]

著者: A. M.
作成日: 2026-05-25

参考文献:
    - PEP 8:                  https://peps.python.org/pep-0008/
    - PEP 257 (Docstring規約): https://peps.python.org/pep-0257/
    - Python公式ドキュメント:    https://docs.python.org/ja/3/
"""

from astropy.io import fits
import pandas as pd
import numpy as np

# =====================
# 1. FITSを読み込む
# =====================

# あなたのデータ
with fits.open("./results/fits/mpajhu_dr7_v5_2_merged_zlt0.2_Lgt1e+39.fits") as hdul:
    data1 = pd.DataFrame(hdul[1].data)

# SDSSデータ
with fits.open("./data/data_SDSS/DR7/fits_files/SDSS_galaxy_radius.fits") as hdul:
    data2 = pd.DataFrame(hdul[1].data)


# ========= endian修正 =========
def fix_endian(df):
    return df.apply(
        lambda x: x.values.byteswap().view(x.dtype.newbyteorder('='))
        if x.dtype.kind in 'fi' else x
    )

data1 = fix_endian(data1)
data2 = fix_endian(data2)


# =====================
# 2. カラム名を統一
# =====================

# 念のため小文字に揃える
data1 = data1.rename(columns={
    "PLATEID": "plate",
    "MJD": "mjd",
    "FIBERID": "fiberid"
})

# data2はすでにOK（確認だけ）
# ['plate', 'mjd', 'fiberid', 'deVRad_r', 'expRad_r', 'fracDeV_r']

# =====================
# 3. 必要な列だけ抽出（軽量化）
# =====================

data2_sub = data2[[
    "plate", "mjd", "fiberid",
    "deVRad_r", "expRad_r", "fracDeV_r"
]]

# =====================
# 4. クロスマッチ（JOIN）
# =====================

merged = pd.merge(
    data1,
    data2_sub,
    on=["plate", "mjd", "fiberid"],
    how="left"   # 元データを保持
)

# =====================
# 5. Re を計算（重要）
# =====================

merged["Re"] = np.where(
    merged["fracDeV_r"] > 0.5,
    merged["deVRad_r"],
    1.678 * merged["expRad_r"]
)

# =====================
# 6. 保存
# =====================

# pandas → FITSに戻す
from astropy.table import Table

table = Table.from_pandas(merged)
table.write("./results/fits/mpajhu_dr7_v5_2_merged_zlt0.2_Lgt1e+39_radius.fits", overwrite=True)

print("完了: ./results/fits/mpajhu_dr7_v5_2_merged_zlt0.2_Lgt1e+39_radius.fits に保存しました")


print("data1 size:", len(data1))
print("data2 size:", len(data2))

# 内部結合でマッチ数確認
matched = pd.merge(
    data1,
    data2,
    on=["plate", "mjd", "fiberid"],
    how="inner"
)

print("matched size:", len(matched))
print("match rate:", len(matched) / len(data1))
# data1のplate分布
print("data1 plate sample:", data1["plate"].unique()[:10])

# data2のplate分布
print("data2 plate sample:", data2["plate"].unique()[:10])