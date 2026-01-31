#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
スクリプトの概要:
Strong line methodsに基づいて
ガス相の金属量を推定します。
calibrationはCurti+17の結果を
使用しています。
係数が正確かは確認が必要です。

使用方法:
    strong_line_callibration.py [オプション]

著者: A. M.
作成日: 2026-01-22

参考文献:
    - PEP 8:                  https://peps.python.org/pep-0008/
    - PEP 257 (Docstring規約): https://peps.python.org/pep-0257/
    - Python公式ドキュメント:    https://docs.python.org/ja/3/
    - Curti+17
"""

# === 必要なパッケージのインストール === #
import os
import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.optimize import brentq

# ===============================
# FITS file
# ===============================
current_dir = os.getcwd()
fits_file = os.path.join(current_dir, "data/data_SDSS/DR7/fits_files/gal_line_dr7_v5_2.fit") 

with fits.open(fits_file) as hdul:
    d = hdul[1].data

# ===============================
# Basic IDs
# ===============================
plateid = d['PLATEID']
fiberid = d['FIBERID']

# ===============================
# Fluxes & errors
# ===============================
Ha, Hae = d['H_ALPHA_FLUX'], d['H_ALPHA_FLUX_ERR']
Hb, Hbe = d['H_BETA_FLUX'], d['H_BETA_FLUX_ERR']

NII, NIIe = d['NII_6584_FLUX'], d['NII_6584_FLUX_ERR']

OIII5, OIII5e = d['OIII_5007_FLUX'], d['OIII_5007_FLUX_ERR']
OIII4, OIII4e = d['OIII_4959_FLUX'], d['OIII_4959_FLUX_ERR']

OII6, OII6e = d['OII_3726_FLUX'], d['OII_3726_FLUX_ERR']
OII9, OII9e = d['OII_3729_FLUX'], d['OII_3729_FLUX_ERR']

# ===============================
# Valid mask
# ===============================
mask = (
    (Ha > 0) & (Hb > 0) & (NII > 0) &
    (OIII5 > 0) & (OIII4 > 0) &
    (OII6 > 0) & (OII9 > 0)
)

# apply mask
plateid, fiberid = plateid[mask], fiberid[mask]
Ha, Hae = Ha[mask], Hae[mask]
Hb, Hbe = Hb[mask], Hbe[mask]
NII, NIIe = NII[mask], NIIe[mask]
OIII5, OIII5e = OIII5[mask], OIII5e[mask]
OIII4, OIII4e = OIII4[mask], OIII4e[mask]
OII6, OII6e = OII6[mask], OII6e[mask]
OII9, OII9e = OII9[mask], OII9e[mask]

# ===============================
# N2
# ===============================
N2 = np.log10(NII / Ha)
N2_err = (1/np.log(10)) * np.sqrt((NIIe/NII)**2 + (Hae/Ha)**2)

Z_N2 = 8.743 + 0.462 * N2
Z_N2_err = 0.462 * N2_err

# ===============================
# O3N2
# ===============================
O3N2 = np.log10((OIII5/Hb) / (NII/Ha))
O3N2_err = (1/np.log(10)) * np.sqrt(
    (OIII5e/OIII5)**2 + (Hbe/Hb)**2 +
    (NIIe/NII)**2 + (Hae/Ha)**2
)

Z_O3N2 = 8.533 - 0.214 * O3N2
Z_O3N2_err = 0.214 * O3N2_err

# ===============================
# R23
# ===============================
OII = OII6 + OII9
OIIe = np.sqrt(OII6e**2 + OII9e**2)

OIII = OIII4 + OIII5
OIIIe = np.sqrt(OIII4e**2 + OIII5e**2)

R23 = (OII + OIII) / Hb
R23_err = R23 * np.sqrt(
    (OIIe/(OII+OIII))**2 +
    (OIIIe/(OII+OIII))**2 +
    (Hbe/Hb)**2
)

logR23 = np.log10(R23)

# Curti+17 coefficients (Table 2)
c0, c1, c2, c3 = 0.7462, -0.7149, -0.9401, -0.6154

def logR23_model(Z):
    x = Z - 8.69
    return c0 + c1*x + c2*x**2 + c3*x**3

Z_R23 = np.full_like(logR23, np.nan)

for i, lr in enumerate(logR23):
    try:
        Z_R23[i] = brentq(
            lambda Z: logR23_model(Z) - lr,
            7.0, 9.3
        )
    except ValueError:
        pass

# 誤差（数値微分）
dlogR23_dZ = (
    c1 +
    2*c2*(Z_R23-8.69) +
    3*c3*(Z_R23-8.69)**2
)
Z_R23_err = np.abs((R23_err / (R23*np.log(10))) / dlogR23_dZ)

# ===============================
# Output DataFrame
# ===============================
df = pd.DataFrame({
    "PLATEID": plateid,
    "FIBERID": fiberid,

    "H_ALPHA_FLUX": Ha,
    "H_ALPHA_FLUX_ERR": Hae,
    "H_BETA_FLUX": Hb,
    "H_BETA_FLUX_ERR": Hbe,

    "NII_6584_FLUX": NII,
    "NII_6584_FLUX_ERR": NIIe,
    "OIII_5007_FLUX": OIII5,
    "OIII_5007_FLUX_ERR": OIII5e,

    "N2": N2,
    "N2_ERR": N2_err,
    "12LOGOH_CURTI17_N2": Z_N2,
    "12LOGOH_CURTI17_N2_ERR": Z_N2_err,

    "O3N2": O3N2,
    "O3N2_ERR": O3N2_err,
    "12LOGOH_CURTI17_O3N2": Z_O3N2,
    "12LOGOH_CURTI17_O3N2_ERR": Z_O3N2_err,

    "R23": R23,
    "R23_ERR": R23_err,
    "12LOGOH_CURTI17_R23": Z_R23,
    "12LOGOH_CURTI17_R23_ERR": Z_R23_err
})

# ===============================
# Save CSV
# ===============================
out_csv = "results/csv/sdss_dr7_curti17_N2_O3N2_R23.csv"
df.to_csv(out_csv, index=False)

print(f"Saved: {out_csv}")
