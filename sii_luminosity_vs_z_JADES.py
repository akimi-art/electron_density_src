#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
スクリプトの概要:
SIIのLumiosityとzの関係を描画します。
単にデータを描画するだけではなく、
フラックスのSN比の情報も含まれます。
これはJADES用です。


使用方法:
    sii_luminoisity_vs_z_JADES.py [オプション]

著者: A. M.
作成日: 2026-02-16

参考文献:
    - PEP 8:                  https://peps.python.org/pep-0008/
    - PEP 257 (Docstring規約): https://peps.python.org/pep-0257/
    - Python公式ドキュメント:    https://docs.python.org/ja/3/
    - Curti+17
"""

"""
luminosity_vs_z_v1.pyを参考に後でflux一定の線を入れよう
"""

# === 必要なモジュールのインポート ===
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import astropy.units as u
from astropy.cosmology import Planck18 as cosmo

# CSV読み込み
df = pd.read_csv("results/csv/JADES_DR3_GOODS-S_SII_ratio_only.csv")

z = df["z_Spec"].values
F6716 = df["S2_6716_flux"].values
F6731 = df["S2_6730_flux"].values

err6716 = 0.5*(df["S2_6716_err_minus"] + df["S2_6716_err_plus"])
err6731 = 0.5*(df["S2_6730_err_minus"] + df["S2_6730_err_plus"])

sn6716 = F6716 / err6716
sn6731 = F6731 / err6731

# Luminosity
d_L = cosmo.luminosity_distance(z).to(u.cm).value
L6716 = 4*np.pi*d_L**2 * F6716
L6731 = 4*np.pi*d_L**2 * F6731

fig, axes = plt.subplots(2,1, figsize=(8,8), sharex=True)

ax1, ax2 = axes

sc1 = ax1.scatter(
    z, L6716,
    c=sn6716,
    cmap="viridis",
    s=15,
    alpha=0.8
)

sc2 = ax2.scatter(
    z, L6731,
    c=sn6731,
    cmap="viridis",
    s=15,
    alpha=0.8
)

for ax in axes:
    ax.set_yscale("log")
    ax.set_xlim(0,0.25)
    ax.set_ylim(1e36,1e42)

ax1.set_ylabel("L(SII 6716)")
ax2.set_ylabel("L(SII 6731)")
ax2.set_xlabel("z")

# カラーバー
cbar = fig.colorbar(sc1, ax=axes, label="S/N")

plt.show()
