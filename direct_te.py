#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
スクリプトの概要:
Direct-Te methodに基づいて
ガス相の金属量を推定します。


使用方法:
    direct-te.py [オプション]

著者: A. M.
作成日: 2026-01-22

参考文献:
    - PEP 8:                  https://peps.python.org/pep-0008/
    - PEP 257 (Docstring規約): https://peps.python.org/pep-0257/
    - Python公式ドキュメント:    https://docs.python.org/ja/3/
    - Curti+17
"""

import os
import numpy as np
import pandas as pd
from scipy.optimize import brentq
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# ============================================================
# 入出力設定
# ============================================================
current_dir = os.getcwd()

in_csv  = os.path.join(current_dir, "results/csv/sdss_dr7_curti17_direct_Te_N2_O3_data_with_ne.csv")
out_csv = os.path.join(current_dir, "results/csv/direct_Te.csv")

SN_CUT = 4.0
N_MC = 300
RNG_SEED = 0

# Case B intrinsic Halpha/Hbeta（代表値）
HaHb_int = 2.86  # [1](https://usm.uni-muenchen.de/people/saglia/praktikum/galspectra/node11.html)[5](https://ned.ipac.caltech.edu/level5/Sept01/Rosa/Rosa_appendix.html)

# ============================================================
# (A) 消光曲線 k(λ)=A(λ)/E(B-V)：CCM89+O'Donnell94（R_V=3.1）
#     ※MPA-JHUは前景消光にO'Donnell(1994)を使っている [3](https://wwwmpa.mpa-garching.mpg.de/SDSS/DR7/raw_data.html)[4](http://www.sdss3.org/dr9/algorithms/galaxy_mpa_jhu.php)
# ============================================================
def k_odonnell94(wave_ang, Rv=3.1):
    """
    O'Donnell(1994) optical update of Cardelli+89: return k(λ)=A(λ)/E(B-V)
    wave_ang: wavelength in Angstrom
    valid in optical/NIR roughly 0.3–1.0 micron
    """
    wave_um = wave_ang / 1e4
    x = 1.0 / wave_um  # inverse micron

    # optical: 1.1 <= x <= 3.3
    if np.any((x < 1.1) | (x > 3.3)):
        # ここでは対象線が全部opticalなので簡略にチェックだけ
        pass

    y = x - 1.82
    # O'Donnell 1994 coefficients (a(x), b(x))
    a = (1
         + 0.17699*y
         - 0.50447*y**2
         - 0.02427*y**3
         + 0.72085*y**4
         + 0.01979*y**5
         - 0.77530*y**6
         + 0.32999*y**7)
    b = (1.41338*y
         + 2.28305*y**2
         + 1.07233*y**3
         - 5.38434*y**4
         - 0.62251*y**5
         + 5.30260*y**6
         - 2.09002*y**7)

    A_over_AV = a + b / Rv
    k = A_over_AV * Rv  # A(λ)/E(B-V)
    return k

# 必要波長の k(λ)
WAVE = {
    "Ha": 6563.0,
    "Hb": 4861.0,
    "OII": 3727.0,   # 3726+3729 を代表値で扱う
    "OIII43": 4363.0,
    "OIII4": 4959.0,
    "OIII5": 5007.0,
}
kHa = float(k_odonnell94(WAVE["Ha"]))
kHb = float(k_odonnell94(WAVE["Hb"]))
kOII = float(k_odonnell94(WAVE["OII"]))
kOIII43 = float(k_odonnell94(WAVE["OIII43"]))
kOIII4  = float(k_odonnell94(WAVE["OIII4"]))
kOIII5  = float(k_odonnell94(WAVE["OIII5"]))

def ebv_from_balmer(Ha, Hb, HaHb_int=HaHb_int, kHa=kHa, kHb=kHb):
    """
    E(B-V)_gas = 2.5/(k(Hb)-k(Ha)) * log10( (Ha/Hb) / (Ha/Hb)_int )
    観測比が intrinsic より小さい場合は 0 に丸める（負の消光を避ける）
    [1](https://usm.uni-muenchen.de/people/saglia/praktikum/galspectra/node11.html)[5](https://ned.ipac.caltech.edu/level5/Sept01/Rosa/Rosa_appendix.html)
    """
    if (Ha <= 0) or (Hb <= 0):
        return np.nan
    ratio = Ha / Hb
    val = 2.5 / (kHb - kHa) * np.log10(ratio / HaHb_int)
    return max(0.0, val)  # negative -> 0

def deredden_flux(F, ebv, k):
    """F_corr = F * 10^(0.4 * k(λ) * E(B-V))"""
    if (F <= 0) or (not np.isfinite(ebv)):
        return np.nan
    return F * (10**(0.4 * k * ebv))

# ============================================================
# (B) direct-Te method（Shi+2014に明示された Izotov+2006 近似式）[2](blob:https://m365.cloud.microsoft/e103decc-f29d-42a8-91bd-9d514182557f)
# ============================================================
def _CT(t3, ne):
    x3 = 1e-4 * ne * (t3 ** -0.5)
    return (8.44 - 1.09*t3 + 0.5*(t3**2) - 0.08*(t3**3)) * (1.0 + 0.0004*x3) / (1.0 + 0.044*x3)

def solve_t3_from_R(R, ne):
    if (R <= 0) or (not np.isfinite(R)) or (ne <= 0) or (not np.isfinite(ne)):
        return np.nan
    logR = np.log10(R)

    def f(t3):
        CT = _CT(t3, ne)
        return t3 - (1.432 / (logR - np.log10(CT)))

    try:
        return brentq(f, 0.6, 2.5, maxiter=200)
    except ValueError:
        return np.nan

def direct_te_Z_from_fluxes(Hb, OIII4, OIII5, OIII43, OII6, OII9, ne):
    """
    入力は「内部消光補正済み」の flux を想定
    返り値: (Te[K], Z=12+log(O/H))
    """
    I3 = OIII4 + OIII5
    I2 = OII6 + OII9

    if (Hb <= 0) or (I3 <= 0) or (I2 <= 0) or (OIII43 <= 0) or (ne <= 0):
        return np.nan, np.nan

    R = I3 / OIII43
    t3 = solve_t3_from_R(R, ne)
    if not np.isfinite(t3):
        return np.nan, np.nan

    # t2 = 0.7 t3 + 0.3 [2](blob:https://m365.cloud.microsoft/e103decc-f29d-42a8-91bd-9d514182557f)
    t2 = 0.7*t3 + 0.3

    # O++/H+ (Eq.7) [2](blob:https://m365.cloud.microsoft/e103decc-f29d-42a8-91bd-9d514182557f)
    opp_12 = (np.log10(I3 / Hb) +
              6.200 + 1.251/t3 - 0.55*np.log10(t3) - 0.014*t3)

    # O+/H+ (Eq.8) [2](blob:https://m365.cloud.microsoft/e103decc-f29d-42a8-91bd-9d514182557f)
    x2 = 1e-4 * ne * (t2 ** -0.5)
    op_12 = (np.log10(I2 / Hb) +
             5.961 + 1.676/t2 - 0.40*np.log10(t2) - 0.034*t2 +
             np.log10(1.0 + 1.35*x2))

    op  = 10**(op_12  - 12.0)
    opp = 10**(opp_12 - 12.0)
    OH  = op + opp
    Z   = 12.0 + np.log10(OH)

    TeK = t3 * 1e4
    return TeK, Z

# ============================================================
# (C) CSV読み込み & S/N>=4 フィルタ
# ============================================================
df = pd.read_csv(in_csv)

df["PLATEID"] = df["PLATEID"].astype(int)
df["FIBERID"] = df["FIBERID"].astype(int)

df["OIII4363_SN"] = df["OIII_4363_FLUX"] / df["OIII_4363_FLUX_ERR"]
df = df[df["OIII4363_SN"] >= SN_CUT].copy()
print(f"After S/N cut (>= {SN_CUT}): {len(df)} rows")

# ============================================================
# (D) Te, Z（中心値）＋MCで誤差 Tee, Ze（Balmer decrementも揺らす）
# ============================================================
rng = np.random.default_rng(RNG_SEED)

Te_list, Z_list, Tee_list, Ze_list, EBV_list = [], [], [], [], []

for _, row in df.iterrows():

    # --- 観測 flux と error ---
    Ha, Hae = row["H_ALPHA_FLUX"], row["H_ALPHA_FLUX_ERR"]
    Hb, Hbe = row["H_BETA_FLUX"],  row["H_BETA_FLUX_ERR"]

    OIII5, OIII5e   = row["OIII_5007_FLUX"], row["OIII_5007_FLUX_ERR"]
    OIII4, OIII4e   = row["OIII_4959_FLUX"], row["OIII_4959_FLUX_ERR"]
    OIII43, OIII43e = row["OIII_4363_FLUX"], row["OIII_4363_FLUX_ERR"]

    OII6, OII6e = row["OII_3726_FLUX"], row["OII_3726_FLUX_ERR"]
    OII9, OII9e = row["OII_3729_FLUX"], row["OII_3729_FLUX_ERR"]

    ne = row["ne_cm3"]

    # --- Balmer decrement から E(B-V)_gas ---
    ebv0 = ebv_from_balmer(Ha, Hb)
    EBV_list.append(ebv0)

    # --- 内部消光補正（必要線だけ） ---
    Hb_c    = deredden_flux(Hb,    ebv0, kHb)
    OIII5_c = deredden_flux(OIII5, ebv0, kOIII5)
    OIII4_c = deredden_flux(OIII4, ebv0, kOIII4)
    OIII43_c= deredden_flux(OIII43,ebv0, kOIII43)
    OII6_c  = deredden_flux(OII6,  ebv0, kOII)
    OII9_c  = deredden_flux(OII9,  ebv0, kOII)

    te0, z0 = direct_te_Z_from_fluxes(Hb_c, OIII4_c, OIII5_c, OIII43_c, OII6_c, OII9_c, ne)
    Te_list.append(te0)
    Z_list.append(z0)

    # --- MC ---
    te_samp, z_samp = [], []
    for k in range(N_MC):
        Ha_k    = rng.normal(Ha, Hae)
        Hb_k    = rng.normal(Hb, Hbe)
        OIII5_k = rng.normal(OIII5, OIII5e)
        OIII4_k = rng.normal(OIII4, OIII4e)
        OIII43_k= rng.normal(OIII43, OIII43e)
        OII6_k  = rng.normal(OII6, OII6e)
        OII9_k  = rng.normal(OII9, OII9e)

        # 負値棄却
        if (Ha_k<=0) or (Hb_k<=0) or (OIII5_k<=0) or (OIII4_k<=0) or (OIII43_k<=0) or (OII6_k<=0) or (OII9_k<=0):
            continue

        ebv_k = ebv_from_balmer(Ha_k, Hb_k)
        Hb_ck    = deredden_flux(Hb_k,    ebv_k, kHb)
        OIII5_ck = deredden_flux(OIII5_k, ebv_k, kOIII5)
        OIII4_ck = deredden_flux(OIII4_k, ebv_k, kOIII4)
        OIII43_ck= deredden_flux(OIII43_k,ebv_k, kOIII43)
        OII6_ck  = deredden_flux(OII6_k,  ebv_k, kOII)
        OII9_ck  = deredden_flux(OII9_k,  ebv_k, kOII)

        te_k, z_k = direct_te_Z_from_fluxes(Hb_ck, OIII4_ck, OIII5_ck, OIII43_ck, OII6_ck, OII9_ck, ne)
        if np.isfinite(te_k) and np.isfinite(z_k):
            te_samp.append(te_k)
            z_samp.append(z_k)

    if len(te_samp) >= max(20, int(0.2*N_MC)):
        Tee_list.append(np.std(te_samp, ddof=1))
        Ze_list.append(np.std(z_samp,  ddof=1))
    else:
        Tee_list.append(np.nan)
        Ze_list.append(np.nan)

df["E_BV_gas"] = np.array(EBV_list)
df["Te_K"]  = np.array(Te_list)
df["Tee_K"] = np.array(Tee_list)
df["Z_OH"]  = np.array(Z_list)   # 12+log(O/H)
df["Ze_OH"] = np.array(Ze_list)

# 保存
os.makedirs(os.path.dirname(out_csv), exist_ok=True)
df.to_csv(out_csv, index=False)

print("Saved:", out_csv)
print(df[["PLATEID","FIBERID","OIII4363_SN","E_BV_gas","Te_K","Tee_K","Z_OH","Ze_OH"]].head())


"""
N2, R3と比較して描画
"""

# =========================
# 入力CSV
# =========================
current_dir = os.getcwd()
in_csv = out_csv

# =========================
# Curti+17 coefficients
# diag = log10(line ratio)
# x = Z - 8.69
# =========================
curti_coef = {
    "R3": [-0.277, -3.549, -3.593, -0.981,  0.000],
    "N2": [-0.489,  1.513, -2.554, -5.293, -2.867],
}

def poly4(x, c):
    c0, c1, c2, c3, c4 = c
    return c0 + c1*x + c2*x**2 + c3*x**3 + c4*x**4

# =========================
# Read data
# =========================
df = pd.read_csv(in_csv)

# =========================
# Balmer decrement
# =========================
Ha = df["H_ALPHA_FLUX"]
Hb = df["H_BETA_FLUX"]

intrinsic = 2.86

# extinction curve (Cardelli+89)
k_Ha   = 2.535
k_Hb   = 3.609
k_NII  = 2.528   # ~ Hα
k_OIII = 3.472   # ~ Hβ

# E(B-V)
EBV = (2.5 / (k_Hb - k_Ha)) * np.log10((Ha / Hb) / intrinsic)
EBV = EBV.clip(lower=0)  # マイナスは物理的に0

df["EBV"] = EBV

# =========================
# deredden
# =========================
df["H_ALPHA_corr"] = Ha * 10**(0.4 * EBV * k_Ha)
df["H_BETA_corr"]  = Hb * 10**(0.4 * EBV * k_Hb)
df["NII_6584_corr"] = df["NII_6584_FLUX"] * 10**(0.4 * EBV * k_NII)
df["OIII_5007_corr"] = df["OIII_5007_FLUX"] * 10**(0.4 * EBV * k_OIII)



# --- define diagnostics if missing ---
if "N2" not in df.columns:
    df["N2"] = np.log10(df["NII_6584_corr"] / df["H_ALPHA_corr"])

if "R3" not in df.columns:
    df["R3"] = np.log10(df["OIII_5007_corr"] / df["H_BETA_corr"])

# --- clean ---
dfp = (
    df.replace([np.inf, -np.inf], np.nan)
      .dropna(subset=["Z_OH", "N2", "R3"])
      .copy()
)

# =========================
# Z grid
# =========================
Z_grid = np.linspace(7.6, 8.85, 500)
x_grid = Z_grid - 8.69

# --- Curti curves (log scale) ---
N2_curve = poly4(x_grid, curti_coef["N2"])
R3_curve = poly4(x_grid, curti_coef["R3"])

# --- validity masks ---
mask_N2 = (Z_grid >= 7.6) & (Z_grid <= 8.85)
mask_R3 = (Z_grid >= 7.6) & (Z_grid <= 8.85)

# =========================
# Plot
# =========================
fig = plt.figure(figsize=(12, 6))
gs = GridSpec(1, 2, wspace=0.25)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)

# ---- N2 ----
ax1.scatter(dfp["Z_OH"], dfp["N2"],
            s=10, alpha=0.3, label="SDSS (direct-Te)")
ax1.plot(Z_grid[mask_N2], N2_curve[mask_N2],
         color="black", lw=2, label="Curti+17 (valid)")

ax1.set_xlabel("12 + log(O/H)")
ax1.set_ylabel("N2 = log10([NII]6584 / Hα)")
ax1.grid(True, alpha=0.25)
ax1.legend(frameon=False)

# ---- R3 ----
ax2.scatter(dfp["Z_OH"], dfp["R3"],
            s=10, alpha=0.3, label="SDSS (direct-Te)")
ax2.plot(Z_grid[mask_R3], R3_curve[mask_R3],
         color="black", lw=2, label="Curti+17 (valid)")

ax2.set_xlabel("12 + log(O/H)")
ax2.set_ylabel("R3 = log10([OIII]5007 / Hβ)")
ax2.grid(True, alpha=0.25)
ax2.legend(frameon=False)

plt.tight_layout()

out_png = os.path.join(
    current_dir, "results/figure/N2_R3_vs_ZOH_with_Curti17.png"
)
os.makedirs(os.path.dirname(out_png), exist_ok=True)
plt.savefig(out_png, dpi=200)
plt.show()


O3 = np.log10(df["OIII_5007_corr"] / df["H_BETA_corr"])
N2 = np.log10(df["NII_6584_corr"] / df["H_ALPHA_corr"])

plt.scatter(N2, O3, s=5, alpha=0.3)
plt.show()


print("Saved:", out_png)

