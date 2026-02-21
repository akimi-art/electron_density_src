#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
スクリプトの概要:
このスクリプトは
JADESのカタログを使って
BAGPIPESによりSEDをフィッティングするものです。
csvファイルに存在する全てのIDのスペクトルに対して
同時にフィッティングを行います。

使用方法:
    BAGPIPES_SED_fitting_JADES.py [オプション]

著者: A. M.
作成日: 2026-02-21

参考文献:
    - PEP 8: https://peps.python.org/pep-0008/
    - PEP 257 (Docstring規約): https://peps.python.org/pep-0257/
    - Python公式ドキュメント: https://docs.python.org/ja/3/
"""

import numpy as np
import pandas as pd
from astropy.io import fits
import bagpipes as pipes
import os
import time
from datetime import datetime

############################################################
# 1. Catalog Load
############################################################

cat = fits.open("jades_dr3_medium_gratings_public_gn_v1.1.fits")
data = cat[1].data
df = pd.DataFrame(data)

df_spec = df[~np.isnan(df["Z_SPEC"])].reset_index(drop=True)

print(f"Total objects with spec-z: {len(df_spec)}")

############################################################
# 2. Spectrum Loader
############################################################

def load_spectrum(object_id, spec_dir):

    file = os.path.join(spec_dir, f"{object_id}_prism.fits")

    if not os.path.exists(file):
        raise FileNotFoundError(f"{file} not found")

    hdu = fits.open(file)

    wave = hdu[1].data["WAVELENGTH"]
    flux = hdu[1].data["FLUX"]
    err  = hdu[1].data["ERROR"]

    mask = (err > 0) & np.isfinite(flux)

    return wave[mask], flux[mask], err[mask]

############################################################
# 3. BAGPIPES Model
############################################################

def build_model(z):

    model = {}

    model["redshift"] = z

    model["delayed"] = {
        "massformed": (6, 13),
        "tau": (0.1, 10),
        "age": (0.01, 3),
        "metallicity": (0.01, 2.5)
    }

    model["dust"] = {
        "type": "Calzetti",
        "Av": (0.0, 3.0)
    }

    model["nebular"] = {
        "logU": (-4, -1)
    }

    return model

############################################################
# 4. Run Fit
############################################################

def run_fit(object_id, z, spec_dir):

    wave, flux, err = load_spectrum(object_id, spec_dir)

    spec = np.column_stack([wave, flux, err])

    galaxy = pipes.galaxy(object_id, spectrum=spec)

    model = build_model(z)

    fit = pipes.fit(galaxy, model)

    fit.fit(verbose=False)

    return fit

############################################################
# 5. Extract Results
############################################################

def extract_results(fit):

    posterior = fit.posterior.samples

    logM = np.median(posterior["stellar_mass"])
    sfr  = np.median(posterior["sfr"])
    Z    = np.median(posterior["metallicity"])

    return logM, sfr, Z

############################################################
# 6. Main Loop with Progress Display
############################################################

spec_dir = "./prism_spectra/"
output_file = "jades_prism_results_live.csv"
log_file = "fit_log.txt"

results = []

start_time = time.time()

for i, row in df_spec.iterrows():

    obj_id = row["ID"]
    z_spec = row["Z_SPEC"]

    print("-------------------------------------------------")
    print(f"[{i+1}/{len(df_spec)}] Fitting object {obj_id}")
    print(f"z_spec = {z_spec}")
    print(f"Start time: {datetime.now()}")
    
    t0 = time.time()

    try:
        fit = run_fit(obj_id, z_spec, spec_dir)
        logM, sfr, Z = extract_results(fit)

        elapsed = time.time() - t0

        print(f"✓ SUCCESS")
        print(f"logM = {logM:.3f}")
        print(f"SFR  = {sfr:.3f}")
        print(f"Z    = {Z:.3f}")
        print(f"Time = {elapsed/60:.2f} min")

        results.append([obj_id, z_spec, logM, sfr, Z])

        # 即時保存（途中で止まってもOK）
        df_out = pd.DataFrame(results,
                              columns=["ID","z_spec","logM","SFR","Metallicity"])
        df_out.to_csv(output_file, index=False)

        # ログ保存
        with open(log_file, "a") as f:
            f.write(f"{obj_id}, SUCCESS, {elapsed:.1f} sec\n")

    except Exception as e:

        print(f"✗ FAILED: {e}")

        with open(log_file, "a") as f:
            f.write(f"{obj_id}, FAILED, {str(e)}\n")

        continue

total_elapsed = time.time() - start_time
print("===============================================")
print(f"All done. Total time: {total_elapsed/3600:.2f} hr")