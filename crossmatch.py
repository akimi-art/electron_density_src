#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
スクリプトの概要:
fitsファイルとtxtファイルの
クロスマッチをします。


使用方法:
    crossmatch.py [オプション]

著者: A. M.
作成日: 2026-01-08

参考文献:
    - PEP 8:                  https://peps.python.org/pep-0008/
    - PEP 257 (Docstring規約): https://peps.python.org/pep-0257/
    - Python公式ドキュメント:    https://docs.python.org/ja/3/
"""


# === 必要なパッケージのインストール === #
import os
import numpy as np
import pandas as pd
from astropy.io import fits

# =====================
# 入力ファイル
# =====================
current_dir = os.getcwd()
txt_in  = os.path.join(current_dir, "results/Samir16/Samir16in_standard_v3_ms_only.txt")
txt_out = out_file  = os.path.join(current_dir, "results/Samir16/Samir16in_standard_v3_ms_only_v3.txt")
# fits_file = os.path.join(current_dir, "data/data_SDSS/DR7/fits_files/gal_info_dr7_v5_2.fit")
in_csv  = os.path.join(current_dir, "results/csv/Samir16in_standard_v3_ms_only_v3.csv")

# =========================
# ファイル 読み込み
# =========================
df = pd.read_csv(in_csv)
plate_csv = []
fiber_csv = []
df["PLATEID"] = plate_csv
df["FIBERID"] = fiber_csv   

# with fits.open(fits_file) as hdul:
#     data = hdul[1].data

#     plate_fits = np.array(data['PLATEID'], dtype=np.int64)
#     fiber_fits = np.array(data['FIBERID'], dtype=np.int64)

# (PLATEID, FIBERID) → fits行番号
# ※ 同じ組が複数あっても「最初の行」を採用
row_dict = {}
for i, (p, f) in enumerate(zip(plate_csv, fiber_csv)):
    if (p, f) not in row_dict:
        row_dict[(p, f)] = i

# =========================
# txt 処理
# =========================
out_lines = []
missing = 0

with open(txt_in, "r") as fin:
    for line in fin:
        line = line.rstrip("\n")
        cols = line.split()

        # txt の最後の2列
        plate = int(cols[-2])
        fiber = int(cols[-1])

        key = (plate, fiber)

        if key in row_dict:
            row = row_dict[key]
            new_line = f"{line} {row}"
        else:
            # 本来は起きない想定
            new_line = f"{line} -1"
            missing += 1

        out_lines.append(new_line)

# =========================
# 保存
# =========================
with open(txt_out, "w") as fout:
    fout.write("\n".join(out_lines))

print("Saved:", txt_out)
print("Missing matches:", missing)

