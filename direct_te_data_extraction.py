#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
スクリプトの概要:
Direct-Te methodに基づいて
ガス相の金属量を推定するための
データを抽出します。


使用方法:
    direct-te_data_extraction.py [オプション]

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

OIII43, OIII43e = d['OIII_4363_FLUX'], d['OIII_4363_FLUX_ERR']

# ===============================
# Valid mask
# ===============================
mask = (
    (Ha > 0) & (Hb > 0) & (NII > 0) &
    (OIII5 > 0) & (OIII4 > 0) &
    (OII6 > 0) & (OII9 > 0) &
    (OIII43 > 0)
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
OIII43, OIII43e = OIII43[mask], OIII43e[mask]

# ===============================
# N2
# ===============================
N2 = np.log10(NII / Ha)
N2_err = (1/np.log(10)) * np.sqrt((NIIe/NII)**2 + (Hae/Ha)**2)

# ===============================
# R3
# ===============================
R3 = np.log10(OIII5 / Hb)
R3_err = (1/np.log(10)) * np.sqrt((OIII5e/OIII5)**2 + (Hbe/Hb)**2)

# ===============================
# Direct-Te methodによるTe, Z（保留）
# ===============================





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

    "OIII_4959_FLUX": OIII4,
    "OIII_4959_FLUX_ERR": OIII4e,
    "OII_3726_FLUX": OII6,
    "OII_3726_FLUX_ERR": OII6e,
    "OII_3729_FLUX": OII9,
    "OII_3729_FLUX_ERR": OII9e,
    "OIII_4363_FLUX": OIII43,
    "OIII_4363_FLUX_ERR": OIII43e,

    # "Direct-Te": Te,
    # "Direct-Te_ERR": Tee,
    # "Direct-Z": Z,
    # "Direct-Z_ERR": Ze,
})

# ===============================
# Save CSV
# ===============================
out_csv = os.path.join(current_dir, "results/csv/sdss_dr7_curti17_direct_Te_N2_O3_data.csv")
df.to_csv(out_csv, index=False)

print(f"Saved: {out_csv}")