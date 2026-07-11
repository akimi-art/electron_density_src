#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
スクリプトの概要:
SDSSのfitファイルを、
resultファイルの物理量を元にdownloadするコードです。

使用方法:
    fit_download.py [オプション]

著者: A. M.
作成日: 2026-07-10

参考文献:
    - PEP 8:                  https://peps.python.org/pep-0008/
    - PEP 257 (Docstring規約): https://peps.python.org/pep-0257/
    - Python公式ドキュメント:    https://docs.python.org/ja/3/
"""

from astropy.io import fits
import numpy as np
import os
import requests
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import matplotlib.pyplot as plt
import glob


# カタログ読込
data = fits.getdata("results/fits/mpajhu_dr7_v5_2_merged_zlt0.2_Lgt1e+39.fits")

# stellar mass選択
mask = (
    (data["sm_MEDIAN"] >= 9.5)
    & (data["sm_MEDIAN"] < 10.0)
)

sample = data[mask]

# ランダム500個
rng = np.random.default_rng(42)

N = min(500, len(sample))

targets = sample[
    rng.choice(len(sample), N, replace=False)
]


outdir = "data/data_SDSS/DR7/spectra/fit"
os.makedirs(outdir, exist_ok=True)

def download(row):

    plate = int(row["PLATEID"])
    mjd = int(row["MJD"])
    fiber = int(row["FIBERID"])

    filename = f"spSpec-{mjd}-{plate:04d}-{fiber:03d}.fit"

    url = (
        f"https://das.sdss.org/spectro/1d_23/"
        f"{plate:04d}/1d/{filename}"
    )

    path = os.path.join(outdir, filename)

    if os.path.exists(path):
        return

    try:
        r = requests.get(url, timeout=30)

        if r.status_code == 200:
            with open(path, "wb") as f:
                f.write(r.content)

            print("OK", filename)

        else:
            print("FAIL", filename)

    except Exception as e:
        print("ERROR", filename, e)


with ThreadPoolExecutor(max_workers=16) as pool:
    pool.map(download, targets)


rows = []

for row in targets:

    plate = int(row["PLATEID"])
    mjd = int(row["MJD"])
    fiber = int(row["FIBERID"])

    url = (
        f"https://das.sdss.org/spectro/1d_23/"
        f"{plate:04d}/1d/"
        f"spSpec-{mjd}-{plate:04d}-{fiber:03d}.fit"
    )

    rows.append(
        {
            "PLATEID": plate,
            "MJD": mjd,
            "FIBERID": fiber,
            "sm_MEDIAN": row["sm_MEDIAN"],
            "URL": url,
        }
    )

pd.DataFrame(rows).to_csv(
    "data/data_SDSS/DR7/spectra/csv/sample_sm95_100.csv",
    index=False
)