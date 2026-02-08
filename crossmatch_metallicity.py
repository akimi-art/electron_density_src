#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
スクリプトの概要:
fitsファイルとtxtファイルの
クロスマッチしてtxtファイルに
金属量の情報を追加します。

使用方法:
    crossmatch_metallicity.py [オプション]

著者: A. M.
作成日: 2026-02-08

参考文献:
    - PEP 8:                  https://peps.python.org/pep-0008/
    - PEP 257 (Docstring規約): https://peps.python.org/pep-0257/
    - Python公式ドキュメント:    https://docs.python.org/ja/3/
"""


import os
import numpy as np
from astropy.io import fits

# =========================
# ファイル名
# =========================
current_dir = os.getcwd()
txt_in     = os.path.join(current_dir,"results/Samir16/Samir16in_standard_v3.txt")
txt_out    = os.path.join(current_dir,"results/Samir16/Samir16in_standard_v4.txt")
fiboh_fits = os.path.join(current_dir, "data/data_SDSS/DR7/fits_files/gal_fiboh_dr7_v5_2.fits")

# =========================
# 金属量 FITS 読み込み 
# =========================
with fits.open(fiboh_fits) as hdul:
    data = hdul[1].data
    median = np.array(data['MEDIAN'], dtype=np.float64)
    p16    = np.array(data['P16'], dtype=np.float64)
    p84    = np.array(data['P84'], dtype=np.float64)

n_fits = len(median)

# =========================
# txt 処理
# =========================
out_lines = []
bad_index = 0
good_rows = 0  # ★追加：m,l,u が全部欠損(-99.9)ではない行数

MISSING = -99.9000  # ★欠損値

with open(txt_in, "r") as fin:
    for line in fin:
        line = line.rstrip("\n")
        cols = line.split()

        # 最後の列：DR7 fits の行番号
        row = int(cols[-1])

        if 0 <= row < n_fits:
            m  = median[row]
            l  = p16[row]
            u  = p84[row]

            # ★追加：欠損(-99.9)かどうか判定（3つとも欠損でなければカウント）
            is_missing = np.isclose([m, l, u], MISSING).any()
            if not is_missing:
                good_rows += 1

            new_line = f"{line} {m:.4f} {l:.4f} {u:.4f}"
        else:
            # 念のため
            new_line = f"{line} NaN NaN NaN"
            bad_index += 1

        out_lines.append(new_line)

# =========================
# 保存
# =========================
with open(txt_out, "w") as fout:
    fout.write("\n".join(out_lines))

print("Saved:", txt_out)
print("Out-of-range rows:", bad_index)
print("Rows with valid metallicity (not -99.9 in m/l/u):", good_rows)
