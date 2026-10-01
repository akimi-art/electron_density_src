#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
スクリプトの概要:
SIIのLumiosityとzの関係を描画します。
これはSDSS用です。

v0との変更点
・SIIの暗い側のみをサンプル選択に含める
・フラックス一定の曲線とSDSSの観測装置の特性をつなげる
・フラックス一定の曲線とLuminosity一定の曲線の交点をサンプル選択に使用
・使用するファイル(merged)にReの情報をあらかじめ入れておく
・BPT分類を行い、SF領域のみに絞る

使用方法:
    sii_luminoisity_vs_z_SDSS_v3.py [オプション]

著者: A. M.
作成日: 2026-09-30

参考文献:
    - フラックス一定の曲線とSDSSの観測装置の特性
"""


# === 必要なモジュールのインポート ===
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from scipy.optimize import brentq

# 軸の設定
plt.rcParams.update({
    # --- 図全体 ---
    "figure.figsize": (12, 6),       # 図サイズ
    "font.size": 32,                 # 全体フォントサイズ
    "axes.labelsize": 32,            # 軸ラベルのサイズ
    "axes.titlesize": 32,            # タイトルのサイズ
    "axes.grid": False,              # グリッドOFF

    # --- 目盛り設定 (ticks) ---
    "xtick.direction": "in",         # x軸目盛りの向き
    "ytick.direction": "in",         # y軸目盛りの向き
    "xtick.top": True,               # 上にも目盛り
    "ytick.right": True,             # 右にも目盛り

    # 主目盛り（major ticks）
    "xtick.major.size": 32,          # 長さ
    "ytick.major.size": 32,
    "xtick.major.width": 2,          # 太さ
    "ytick.major.width": 2,

    # 補助目盛り（minor ticks）
    "xtick.minor.visible": True,     # 補助目盛りON
    "ytick.minor.visible": True,
    "xtick.minor.size": 8,           # 長さ
    "ytick.minor.size": 8,
    "xtick.minor.width": 1.5,        # 太さ
    "ytick.minor.width": 1.5,

    # --- 目盛りラベル ---
    "xtick.labelsize": 28,           # x軸ラベルサイズ
    "ytick.labelsize": 28,           # y軸ラベルサイズ

    # --- フォント ---
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
})


# =====================================
# 0. 設定（02_completeness.py / selection.py と同じ値にする）
# =====================================
cosmo      = FlatLambdaCDM(H0=70, Om0=0.3)
UNIT_FLUX  = 1e-17          # MPA/JHU のフラックスの単位 [erg s^-1 cm^-2]
FLUX_LIMIT = 1e-17          # [SII]6731 のフラックスの限界 [erg s^-1 cm^-2]
L_MIN      = 1e39           # [SII]6731 の光度の下限 [erg s^-1]
SN_MIN     = 3.0
METHOD     = "Ke01"         # AGN の除去：Kewley+01

current_dir = os.getcwd()
fits_path = os.path.join(current_dir, "results/fits/mpajhu_dr7_v5_2_merged_radius.fits")
fig_dir   = os.path.join(current_dir, "results/figure/sample")
out_dir   = os.path.join(current_dir, "results/fits")
os.makedirs(fig_dir, exist_ok=True)
os.makedirs(out_dir, exist_ok=True)

# =====================================
# 1. 読み込みと確認（行が消えていないこと）
# =====================================
t = Table.read(fits_path, format="fits")
df = t.to_pandas()
N_ALL = len(t)
assert N_ALL == 927552, f"行数が想定と違います: {N_ALL}"
assert np.array_equal(np.asarray(t["ROW_ID"]), np.arange(N_ALL)), "ROW_ID が連番ではありません"

# =====================================
# 2. Z_MAX：光度の下限とフラックスの限界の交点
# =====================================
def z_max_from_flux_limit(l_min=L_MIN, f_lim=FLUX_LIMIT):
    def f(z):
        dL = cosmo.luminosity_distance(z).to(u.cm).value
        return 4 * np.pi * dL**2 * f_lim - l_min
    return brentq(f, 1e-4, 1.0)

Z_MAX = z_max_from_flux_limit()
print(f"[INFO] Z_MAX = {Z_MAX:.4f}")

# =====================================
# 3. 必要な量
# =====================================
z      = df["Z"].values
logM   = df["sm_MEDIAN"].values
logSFR = df["sfr_MEDIAN"].values

def flux(name):
    return (df[f"{name}_FLUX"].values * UNIT_FLUX,
            df[f"{name}_FLUX_ERR"].values * UNIT_FLUX)

F_S16, E_S16 = flux("SII_6717")
F_S31, E_S31 = flux("SII_6731")
F_Hb,  E_Hb  = flux("H_BETA")
F_O3,  E_O3  = flux("OIII_5007")
F_Ha,  E_Ha  = flux("H_ALPHA")
F_N2,  E_N2  = flux("NII_6584")

dL = cosmo.luminosity_distance(np.clip(z, 1e-4, None)).to(u.cm).value
with np.errstate(invalid="ignore", divide="ignore"):
    L6716 = 4 * np.pi * dL**2 * F_S16
    L6731 = 4 * np.pi * dL**2 * F_S31

    # S/N（フラックス ÷ 誤差）
    sn_Hb, sn_O3 = F_Hb / E_Hb, F_O3 / E_O3
    sn_Ha, sn_N2 = F_Ha / E_Ha, F_N2 / E_N2

    # BPT の輝線比
    log_N2Ha = np.log10(F_N2 / F_Ha)
    log_O3Hb = np.log10(F_O3 / F_Hb)

# =====================================
# 4. 選択の条件
# =====================================
# 4a. 親サンプル
base = (np.isfinite(z) & (z > 0) & (z < Z_MAX) &
        np.isfinite(logM) & np.isfinite(logSFR))

# 4b. 光度の下限（体積限定）
lum = np.isfinite(L6731) & (L6731 > L_MIN)

# 4c. 分類に必要な S/N（4本）
sn4 = (sn_Hb >= SN_MIN) & (sn_O3 >= SN_MIN) & (sn_Ha >= SN_MIN) & (sn_N2 >= SN_MIN)

# 4d. 星形成の判定（Kewley+01。参考に Kauffmann+03 も計算）
with np.errstate(invalid="ignore", divide="ignore"):
    sf_ke01 = (log_N2Ha < 0.47) & (log_O3Hb < 0.61 / (log_N2Ha - 0.47) + 1.19)
    sf_ka03 = (log_N2Ha < 0.05) & (log_O3Hb < 0.61 / (log_N2Ha - 0.05) + 1.3)

selected = base & lum & sn4 & sf_ke01

# =====================================
# 5. 選択の流れ（2章の表に使う）
# =====================================
flow = [
    ("SDSS DR7 MPA/JHU（全行）",          np.ones(N_ALL, bool)),
    (f"0 < z < {Z_MAX:.4f}",              np.isfinite(z) & (z > 0) & (z < Z_MAX)),
    ("+ M*, SFR が有限",                  base),
    (f"+ L([SII]6731) > {L_MIN:.0e}",     base & lum),
    (f"+ S/N >= {SN_MIN:g}（4本）",        base & lum & sn4),
    (f"+ 星形成（{METHOD}）",              selected),
]
print("\n===== Selection flow =====")
rows = []
for name, m in flow:
    print(f"  {name:32s}: {m.sum():>8,}")
    rows.append({"step": name, "N": int(m.sum())})
pd.DataFrame(rows).to_csv(os.path.join(out_dir, f"selection_flow_{METHOD}.csv"), index=False)

# 参考：Kauffmann+03 を使った場合の数
print(f"  （参考）Kauffmann+03 の場合           : {(base & lum & sn4 & sf_ka03).sum():>8,}")

# =====================================
# 6. 最終サンプルの性質（範囲で示す）
# =====================================
def rng(x):
    return f"{np.nanmin(x):.3f} -- {np.nanmax(x):.3f}"

print("\n===== Sample statistics =====")
print(f"  N(selected)   = {selected.sum():,}")
print(f"  z range       = {rng(z[selected])}")
print(f"  logM range    = {rng(logM[selected])}")
print(f"  logSFR range  = {rng(logSFR[selected])}")
print(f"  logL6731 range= {rng(np.log10(L6731[selected]))}")

# =====================================
# 7. 列を追加して保存（ROW_ID で元の行に戻れる）
# =====================================
t["L_SII6716"]  = L6716
t["L_SII6731"]  = L6731
t["log_N2Ha"]   = log_N2Ha
t["log_O3Hb"]   = log_O3Hb
t["SN4_OK"]     = sn4
t["SF_Ke01"]    = sf_ke01
t["SF_Ka03"]    = sf_ka03
t["SELECTED"]   = selected

out_path = os.path.join(out_dir, f"sdss_sample_zlt{Z_MAX:.4f}_Lgt{L_MIN:.0e}_{METHOD}.fits")
t[selected].write(out_path, format="fits", overwrite=True)
print(f"\n[DONE] {out_path}（{selected.sum():,} 行）")

# =====================================
# 8. 図
# =====================================
def finish(ax, path):
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"[DONE] {path}")

# 8a. 体積限定の図：縦線と横線がフラックス限界の曲線上で交わる
fig, ax = plt.subplots(figsize=(12, 6))
ok = np.isfinite(L6731) & (L6731 > 0)
ax.scatter(z[ok], L6731[ok], s=0.2, alpha=0.2, color="gray", rasterized=True)
vol = base & lum
ax.scatter(z[vol], L6731[vol], s=0.2, alpha=0.5, color="firebrick", rasterized=True)
zg = np.linspace(1e-4, 0.4, 400)
ax.plot(zg, 4 * np.pi * cosmo.luminosity_distance(zg).to(u.cm).value**2 * FLUX_LIMIT,
        color="k", lw=2)
ax.axvline(Z_MAX, color="k", lw=2)
ax.axhline(L_MIN, color="k", lw=2)
ax.set_yscale("log")
ax.set_xlim(0, 0.4); ax.set_ylim(1e36, 1e42)
ax.set_xlabel(r"$z$")
ax.set_ylabel(r"$L([{\rm S\,II}]\lambda6731)$ [erg s$^{-1}$]")
finish(ax, os.path.join(fig_dir, "sii6731_luminosity_vs_z.png"))

# 8b. BPT 図（分類の前の銀河 = 体積限定 + S/N）
pre = base & lum & sn4
fig, ax = plt.subplots(figsize=(10, 9))
ax.scatter(log_N2Ha[pre], log_O3Hb[pre], s=0.5, alpha=0.1, color="gray", rasterized=True)
ax.scatter(log_N2Ha[selected], log_O3Hb[selected], s=0.5, alpha=0.2,
           color="firebrick", rasterized=True)
xx = np.linspace(-2.0, 0.46, 500)
ax.plot(xx, 0.61 / (xx - 0.47) + 1.19, color="k", lw=2, label="Kewley+01")
xx = np.linspace(-2.0, 0.04, 500)
ax.plot(xx, 0.61 / (xx - 0.05) + 1.3, color="k", lw=2, ls="--", label="Kauffmann+03")
ax.set_xlim(-2.0, 0.5); ax.set_ylim(-1.5, 1.5)
ax.set_xlabel(r"$\log([{\rm N\,II}]\lambda6584/{\rm H}\alpha)$")
ax.set_ylabel(r"$\log([{\rm O\,III}]\lambda5007/{\rm H}\beta)$")
ax.legend(loc="lower left")
finish(ax, os.path.join(fig_dir, "bpt_diagram.png"))

# 8c. 最終サンプルの z の分布
fig, ax = plt.subplots(figsize=(12, 6))
ax.hist(z[selected], bins=50, color="gray", edgecolor="black", alpha=0.8)
ax.set_xlabel(r"$z$"); ax.set_ylabel("Number of galaxies")
ax.set_xlim(0, Z_MAX)
finish(ax, os.path.join(fig_dir, "selected_redshift_histogram.png"))