#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
スクリプトの概要:
fitsファイルとtxtファイルの
クロスマッチしてtxtファイルに
金属量の情報を追加します。

使用方法:
    crossmatch_metallicity_v1.py [オプション]

著者: A. M.
作成日: 2026-01-08

参考文献:
    - PEP 8:                  https://peps.python.org/pep-0008/
    - PEP 257 (Docstring規約): https://peps.python.org/pep-0257/
    - Python公式ドキュメント:    https://docs.python.org/ja/3/
"""


# === 必要なパッケージのインストール === #
import pandas as pd
import numpy as np
import os

# ===============================
# Input files
# ===============================
current_dir = os.getcwd()
txt_in   = os.path.join(current_dir, "results/txt/Samir16in_standard_v3_MS_only_v1.txt")
txt_out  = os.path.join(current_dir, "results/Samir16/Samir16in_standard_v3_ms_only_v3.txt")
csv_file = os.path.join(current_dir, "results/csv/sdss_dr7_curti17_N2_O3N2_R23.csv")

# ===============================
# Read metallicity CSV
# ===============================
df = pd.read_csv(csv_file)

# クロスマッチ用辞書
# key = (PLATEID, FIBERID)
Zdict = {}

for _, r in df.iterrows():
    key = (int(r["PLATEID"]), int(r["FIBERID"]))
    Zdict[key] = {
        "N2":   (r["12LOGOH_CURTI17_N2"],   r["12LOGOH_CURTI17_N2_ERR"]),
        "O3N2": (r["12LOGOH_CURTI17_O3N2"], r["12LOGOH_CURTI17_O3N2_ERR"]),
        "R23":  (r["12LOGOH_CURTI17_R23"],  r["12LOGOH_CURTI17_R23_ERR"]),
    }

# ===============================
# Helper: metallicity string
# ===============================
def fmt_metal(Z, Zerr):
    if not np.isfinite(Z) or not np.isfinite(Zerr):
        return "NaN+NaN-NaN"
    return f"{Z:.6f}+{Zerr:.6f}-{Zerr:.6f}"

# ===============================
# Process txt
# ===============================
out_lines = []

with open(txt_in, "r") as f:
    for line in f:
        line = line.rstrip()

        # 空行・コメント行はそのまま
        if line.strip() == "" or line.lstrip().startswith("#"):
            out_lines.append(line)
            continue

        cols = line.split()

        # 末尾2列が PLATEID FIBERID
        plateid = int(cols[-2])
        fiberid = int(cols[-1])

        key = (plateid, fiberid)

        if key in Zdict:
            z = Zdict[key]

            z_n2   = fmt_metal(*z["N2"])
            z_o3n2 = fmt_metal(*z["O3N2"])
            z_r23  = fmt_metal(*z["R23"])
        else:
            z_n2 = z_o3n2 = z_r23 = "NaN+NaN-NaN"

        new_line = (
            line
            + "\t" + z_n2
            + "\t" + z_o3n2
            + "\t" + z_r23
        )

        out_lines.append(new_line)

# ===============================
# Write output
# ===============================
with open(txt_out, "w") as f:
    for l in out_lines:
        f.write(l + "\n")

print(f"Saved: {txt_out}")
