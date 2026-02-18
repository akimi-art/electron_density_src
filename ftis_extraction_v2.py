#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
スクリプトの概要:
このスクリプトは
FITSファイルの必要な要素のみを抽出して
fitsファイルを作るものです。

使用方法:
    python fits_extraction_v2.py [オプション]

著者: A. M.
作成日: 2026-02-18

参考文献:
    - PEP 8: https://peps.python.org/pep-0008/
    - PEP 257 (Docstring規約): https://peps.python.org/pep-0257/
    - Python公式ドキュメント: https://docs.python.org/ja/3/
    - Elbaz et al. (2007)
    - Chen et al. (2016)
"""


# == 必要なパッケージのインポート == #
import os
import numpy as np
from astropy.io import fits

# 入出力パス
merged_path  = os.path.join("results", "fits", "mpajhu_dr7_v5_2_merged.fits")
galline_path = os.path.join("data", "data_SDSS", "DR7", "fits_files", "gal_line_dr7_v5_2.fit")
out_path     = os.path.join("results", "fits", "mpajhu_dr7_v5_2_merged_with_OII.fits")

# 読み込み（拡張 #1 はバイナリテーブル前提）
with fits.open(merged_path, mode="readonly") as hdul_m, fits.open(galline_path, mode="readonly") as hdul_l:
    prim_hdu_m = hdul_m[0].copy()         # 主ヘッダはそのまま継承
    data_m     = hdul_m[1].data           # 既存テーブル（行数 N）
    cols_m     = hdul_m[1].columns

    data_l     = hdul_l[1].data           # gal_line テーブル（行数 N）

# 行数チェック（安全確認）
n_m, n_l = len(data_m), len(data_l)
if n_m != n_l:
    raise ValueError(f"行数が一致しません: merged={n_m}, gal_line={n_l}")

# 必要列の存在チェック（gal_line 側）
need_cols = [
    "OII_3726_FLUX", "OII_3726_FLUX_ERR",
    "OII_3729_FLUX", "OII_3729_FLUX_ERR",
]
for c in need_cols:
    if c not in data_l.dtype.names:
        raise KeyError(f"[gal_line] 必要列が見つかりません: {c}")

# 3726/3729 の配列を抽出
O2_26    = np.asarray(data_l["OII_3726_FLUX"],     dtype=float)
O2_26_E  = np.asarray(data_l["OII_3726_FLUX_ERR"], dtype=float)
O2_29    = np.asarray(data_l["OII_3729_FLUX"],     dtype=float)
O2_29_E  = np.asarray(data_l["OII_3729_FLUX_ERR"], dtype=float)

# ブレンド（3727 = 3726 + 3729）、誤差は共分散ゼロ仮定の二乗和平方根
O2_27    = O2_26 + O2_29
O2_27_E  = np.sqrt(O2_26_E**2 + O2_29_E**2)

# 追加カラム作成（倍精度 'D'）
new_cols = fits.ColDefs([
    fits.Column(name="OII_3726_FLUX",     format="D", array=O2_26),
    fits.Column(name="OII_3726_FLUX_ERR", format="D", array=O2_26_E),
    fits.Column(name="OII_3729_FLUX",     format="D", array=O2_29),
    fits.Column(name="OII_3729_FLUX_ERR", format="D", array=O2_29_E),
    fits.Column(name="OII_3727_FLUX",     format="D", array=O2_27),
    fits.Column(name="OII_3727_FLUX_ERR", format="D", array=O2_27_E),
])

# 既存テーブルに追記
out_hdu = fits.BinTableHDU.from_columns(cols_m + new_cols)

# 書き出し
fits.HDUList([prim_hdu_m, out_hdu]).writeto(out_path, overwrite=True)

print(f"[OK] 書き出し完了: {out_path}")
print("追加列: OII_3726_FLUX, OII_3726_FLUX_ERR, OII_3729_FLUX, OII_3729_FLUX_ERR, OII_3727_FLUX, OII_3727_FLUX_ERR")