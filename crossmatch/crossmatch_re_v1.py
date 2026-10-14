#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
スクリプトの概要:
FITSファイル同士をクロスマッチします。
RADECでのマッチに切り替えました。

使用方法:
    crossmatch_re_v1.py [オプション]

著者: A. M.
作成日: 2026-05-25

参考文献:
    - PEP 8:                  https://peps.python.org/pep-0008/
    - PEP 257 (Docstring規約): https://peps.python.org/pep-0257/
    - Python公式ドキュメント:    https://docs.python.org/ja/3/
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u

# ======================
# 0. 設定
# ======================
in_path  = "./results/fits/mpajhu_dr7_v5_2_merged.fits"
csv_path = "./data/data_SDSS/DR7/csv_files/SDSS_galaxy_radius.csv"
out_path = "./results/fits/mpajhu_dr7_v5_2_merged_radius.fits"
fig_dir  = "./results/figure/crossmatch"
os.makedirs(fig_dir, exist_ok=True)

MAX_SEP = 1.0     # arcsec
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

# ======================
# 1. データ読み込み
# ======================
data1 = Table.read(in_path)
data2 = pd.read_csv(csv_path)
N1 = len(data1)

# 行番号（最初に一度だけ付け、以後変えない）
if "ROW_ID" not in data1.colnames:
    data1["ROW_ID"] = np.arange(N1)

ra1  = np.asarray(data1["RA"],  dtype=float)
dec1 = np.asarray(data1["DEC"], dtype=float)
ra2  = data2["ra"].values.astype(float)
dec2 = data2["dec"].values.astype(float)

# ======================
# 2. 座標が有効か（行は消さず、印を付けるだけ）
# ======================
valid1 = np.isfinite(ra1) & np.isfinite(dec1) & (np.abs(dec1) <= 90)
valid2 = np.isfinite(ra2) & np.isfinite(dec2) & (np.abs(dec2) <= 90)
print(f"座標が不正な行（data1）: {(~valid1).sum()}")
print(f"座標が不正な行（data2）: {(~valid2).sum()}")

rows1_valid = np.where(valid1)[0]     # data1 の有効な行の元の行番号
rows2_valid = np.where(valid2)[0]     # data2 の有効な行の元の行番号

# ======================
# 3. 有効な行だけでマッチする
# ======================
c1 = SkyCoord(ra=ra1[valid1] * u.deg, dec=dec1[valid1] * u.deg)
c2 = SkyCoord(ra=ra2[valid2] * u.deg, dec=dec2[valid2] * u.deg)
idx_v, d2d_v, _ = c1.match_to_catalog_sky(c2)

# ======================
# 4. 結果を全行の配列に戻す
# ======================
sep = np.full(N1, np.nan)                 # マッチ距離 [arcsec]
sep[valid1] = d2d_v.arcsec

match_row = np.full(N1, -1, dtype=int)    # 対応する data2 の元の行番号（なしは -1）
match_row[valid1] = rows2_valid[idx_v]

matched = np.isfinite(sep) & (sep < MAX_SEP)
print(f"マッチ数: {matched.sum():,} / {N1:,}（{matched.mean():.4f}）")

# ======================
# 5. 値を全行の列として追加（マッチしない行は NaN）
# ======================
for col in ["deVRad_r", "expRad_r", "fracDeV_r", "petroR50_r"]:
    if col not in data2.columns:
        print(f"[SKIP] {col} は data2 にありません")
        continue
    arr = np.full(N1, np.nan)
    arr[matched] = data2[col].values[match_row[matched]]
    data1[col] = arr

# Rₑ（1.678 倍は不要。ソース：https://www.sdss4.org/dr12/algorithms/magnitudes/）
# selection.py でも同じ式で計算し直す
frac_dev = np.asarray(data1["fracDeV_r"], dtype=float)
Re_arcsec = np.where(frac_dev > 0.5,
                     np.asarray(data1["deVRad_r"], dtype=float),
                     np.asarray(data1["expRad_r"], dtype=float))
data1["Re_arcsec"] = Re_arcsec

if "Re" in data1.colnames:          # 1.678 倍を含む古い列は削除
    data1.remove_column("Re")

data1["COORD_VALID"]       = valid1
data1["RADIUS_SEP_ARCSEC"] = sep
data1["RADIUS_MATCHED"]    = matched

# ======================
# 6. 行が消えていないことを確かめて保存
# ======================
assert len(data1) == N1, "行数が変わっています"
assert np.array_equal(np.asarray(data1["ROW_ID"]), np.arange(N1)), "行の順番が変わっています"

data1.write(out_path, overwrite=True)
print(f"[DONE] {out_path}（{N1:,} 行）")
print(f"マッチ距離の中央値: {np.nanmedian(sep[matched]):.3f} arcsec")


# ============================================================
# 確認1：マッチ距離の分布と、偶然の一致の割合
# ============================================================
c1_shift = SkyCoord(ra=(ra1[valid1] + 1 / 60) * u.deg, dec=dec1[valid1] * u.deg)
_, d2d_shift, _ = c1_shift.match_to_catalog_sky(c2)

bins = np.logspace(-3, 1.5, 100)
fig, ax = plt.subplots(figsize=(7, 5))
ax.hist(d2d_v.arcsec, bins=bins, histtype="step", lw=2, label="real")
ax.hist(d2d_shift.arcsec, bins=bins, histtype="step", lw=2, ls="--",
        label="shifted by 1 arcmin (random)")
ax.axvline(MAX_SEP, color="gray", ls=":")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Separation [arcsec]"); ax.set_ylabel("Number")
ax.legend()
plt.savefig(os.path.join(fig_dir, "separation.png"), bbox_inches="tight")
plt.show()

n_real = matched.sum()
n_rand = np.sum(d2d_shift.arcsec < MAX_SEP)
print(f"マッチ数（< {MAX_SEP} arcsec）  : {n_real:,}")
print(f"偶然の一致の推定数         : {n_rand:,}")
print(f"偶然の一致の割合（推定）   : {n_rand / n_real:.2e}")
print(f"マッチ距離の 50/90/99 %点  : "
      f"{np.percentile(sep[matched], [50, 90, 99]).round(3)} arcsec")

# ============================================================
# 確認2：一対一か
# ============================================================
uniq, counts = np.unique(match_row[matched], return_counts=True)
print(f"data2 の同一天体に複数対応した数: {np.sum(counts > 1):,} 天体"
      f"（{np.sum(counts[counts > 1]):,} 行）")

# 相互最近傍（有効な行の中で計算し、全行に戻す）
idx_back_v, _, _ = c2.match_to_catalog_sky(c1)
mutual_v = idx_back_v[idx_v] == np.arange(len(c1))
mutual = np.zeros(N1, dtype=bool)
mutual[valid1] = mutual_v
print(f"相互最近傍の割合（マッチした中で）: {np.mean(mutual[matched]):.4f}")

# ============================================================
# 確認3：マッチできなかった銀河に偏りはないか
# ============================================================
z_all = np.asarray(data1["Z"], dtype=float)
logM_all = np.asarray(data1["sm_MEDIAN"], dtype=float)

def match_rate(x, bins):
    ok = np.isfinite(x)
    n_all, _ = np.histogram(x[ok], bins=bins)
    n_m, _ = np.histogram(x[ok & matched], bins=bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        return 0.5 * (bins[1:] + bins[:-1]), n_m / n_all

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
for ax, x, b, lab in zip(axes,
                         [z_all, logM_all],
                         [np.linspace(0, 0.3, 31), np.arange(7, 12.01, 0.2)],
                         [r"$z$", r"$\log(M_*/M_\odot)$"]):
    xc, r = match_rate(x, b)
    ax.plot(xc, r, "o-", color="k")
    ax.set_xlabel(lab)
axes[0].set_ylabel("Match rate")
axes[0].set_ylim(0, 1.05)
plt.savefig(os.path.join(fig_dir, "match_rate.png"), bbox_inches="tight")
plt.show()

# ============================================================
# 確認4：取り込んだ値は正しいか
# ============================================================
Re_arr = np.asarray(data1["Re_arcsec"], dtype=float)
print(f"Re が有限かつ正       : {np.sum(np.isfinite(Re_arr) & (Re_arr > 0)):,}")
print(f"Re <= 0 または異常値 : {np.sum(matched & ~(Re_arr > 0)):,}")
print(f"fracDeV が [0,1] の外 : {np.sum(matched & ((frac_dev < 0) | (frac_dev > 1))):,}")
print(f"Re の 1/50/99 %点 [arcsec]: "
      f"{np.nanpercentile(Re_arr[matched], [1, 50, 99]).round(2)}")

# 4a. 角度の Re の分布
fig, ax = plt.subplots(figsize=(7, 5))
ax.hist(Re_arr[matched & (Re_arr > 0)], bins=np.logspace(-1, 2, 100),
        histtype="step", lw=2)
ax.set_xscale("log")
ax.set_xlabel(r"$R_{\rm e}$ [arcsec]"); ax.set_ylabel("Number")
plt.savefig(os.path.join(fig_dir, "Re_arcsec_hist.png"), bbox_inches="tight")
plt.show()

# 4b. サイズと質量の関係
ok = (matched & (Re_arr > 0) & np.isfinite(z_all) & (z_all > 0) &
      np.isfinite(logM_all))
kpc_per_arcsec = cosmo.kpc_proper_per_arcmin(z_all[ok]).value / 60
Re_kpc = Re_arr[ok] * kpc_per_arcsec

fig, ax = plt.subplots(figsize=(7, 5))
h = ax.hist2d(logM_all[ok], np.log10(Re_kpc),
              bins=[np.arange(8, 12, 0.05), np.arange(-0.5, 1.5, 0.02)],
              cmin=1, norm=LogNorm())
ax.set_xlabel(r"$\log(M_*/M_\odot)$")
ax.set_ylabel(r"$\log(R_{\rm e}/{\rm kpc})$")
plt.colorbar(h[3], ax=ax, label="Number")
plt.savefig(os.path.join(fig_dir, "size_mass.png"), bbox_inches="tight")
plt.show()

# 4c. （petroR50_r がある場合）Rₑ の定義の確認：円盤優勢で比が ≈ 1 になるはず
if "petroR50_r" in data1.colnames:
    petro = np.asarray(data1["petroR50_r"], dtype=float)
    disk = matched & (frac_dev < 0.5) & (Re_arr > 0) & (petro > 0)
    print(f"Re / petroR50（円盤優勢）の中央値: {np.nanmedian(Re_arr[disk] / petro[disk]):.3f}")