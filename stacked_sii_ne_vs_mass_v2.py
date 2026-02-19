#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
スクリプトの概要:
logM* ビンごとに
  [SII]6717,6731 フラックスをスタック
→ MCで ratio 分布
→ PyNebで ne 分布
→ P16, P50, P84 を保存・描画
→ 完全なサンプルのみを対象とする


使用方法:
    stacked_sii_ne_vs_mass_v2.py [オプション]

著者: A. M.
作成日: 2026-02-19

参考文献:
    - PEP 8:                  https://peps.python.org/pep-0008/
    - PEP 257 (Docstring規約): https://peps.python.org/pep-0257/
    - Python公式ドキュメント:    https://docs.python.org/ja/3/
    - Curti+17
"""


# === 必要なモジュールのインポート ===
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.table import Table
import re
import matplotlib.gridspec as gridspec
from pathlib import Path
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u



# -----------------------
# 軸の設定
# -----------------------
# 軸の設定
plt.rcParams.update({
    # --- 図全体 ---
    "figure.figsize": (12, 6),       # 図サイズ
    "font.size": 20,                 # 全体フォントサイズ
    "axes.labelsize": 24,            # 軸ラベルのサイズ
    "axes.titlesize": 20,            # タイトルのサイズ
    "axes.grid": False,              # グリッドOFF

    # --- 目盛り設定 (ticks) ---
    "xtick.direction": "in",         # x軸目盛りの向き
    "ytick.direction": "in",         # y軸目盛りの向き
    "xtick.top": True,               # 上にも目盛り
    "ytick.right": True,             # 右にも目盛り

    # 主目盛り（major ticks）
    "xtick.major.size": 20,          # 長さ
    "ytick.major.size": 20,
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
    "xtick.labelsize": 20,           # x軸ラベルサイズ
    "ytick.labelsize": 20,           # y軸ラベルサイズ

    # --- フォント ---
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
})

# ==========================================
# Imports
# ==========================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u

# ==========================================
# 入出力
# ==========================================
current_dir = os.getcwd()
# fits_path = os.path.join(current_dir, "results/JADES/JADES_DR3/data_from_Nishigaki/jades_info_crossmatch.csv")
csv_path = "./results/JADES/JADES_DR3/data_from_Nishigaki/jades_info_crossmatch_fit_L6717_ge_4pi_dL2_1e-17_L6731_ge_4pi_dL2_1e-17.csv"

out_csv = os.path.join(current_dir, "results/table/stacked_sii_ratio_vs_mass_COMPLETE_JADES_fit_COMPLETE.csv")
out_png = os.path.join(current_dir, "results/figure/stacked_sii_ratio_vs_mass_COMPLETE_JADES_fit_COMPLETE.png")

os.makedirs(os.path.dirname(out_csv), exist_ok=True)
os.makedirs(os.path.dirname(out_png), exist_ok=True)

# ==========================================
# パラメータ
# ==========================================
BIN_WIDTH = 0.1 # 変更
NMIN = 3          # スタックに含める最小データ数
N_MC = 5000
# Lcut = 1e39     # 完全サンプル条件（使うなら下で有効化）

# 単位スケール（カタログの単位に合わせて調整）
# UNIT_FLUX = 1e-17  # 例: MPA-JHU などの慣例
UNIT_FLUX = 1e-19    # いまの設定を踏襲

# z-bin の定義（左開右閉ではなく、ここでは「(lo, hi]」を採用しないよう注意）
# 今回は (1,4), (4,7), (>7) を色分け
Z_BINS = [
    dict(name="1<z<4", color="tab:blue",  lo=1.0, hi=4.0,  inclusive="(,)"),
    dict(name="4<z<7", color="tab:green", lo=4.0, hi=7.0,  inclusive="(,)"),
    dict(name="z>7",   color="tab:red",   lo=7.0, hi=np.inf, inclusive="(,]"),  # hi=inf
]
# 参考：inclusive の意味
# "(,)"   :   lo <  z <  hi
# "[,)"   :  lo <= z <  hi
# "(,]"   :   lo <  z <= hi
# ここは慣例的に開区間 "(,)" を使用（境界の二重カウント回避）

# 質量ビンの作り方：
#   'per_z'  : z-bin ごとに最小～最大から edges を作る（ユーザの意図）
#   'global' : 全体の最小～最大から共通 edges を作る（比較しやすい）
MASS_BIN_MODE = 'per_z'  # 'per_z' または 'global'

# ==========================================
# 読み込み
# ==========================================
# tab = Table.read(fits_path, hdu=1)
# df = tab.to_pandas()
df = pd.read_csv(csv_path)

# ==========================================
# 基本量の計算
# ==========================================
z = df["z_spec"].values

# SII line fluxes（列名はユーザの現状に合わせる）
# F6716 = df["S2_6718_flux"].values * UNIT_FLUX
# F6731 = df["S2_6733_flux"].values * UNIT_FLUX
# err6716 = df["S2_6718_err"].values * UNIT_FLUX
# err6731 = df["S2_6733_err"].values * UNIT_FLUX
F6716 = df["S2_6716_flux"].values * UNIT_FLUX
F6731 = df["S2_6730_flux"].values * UNIT_FLUX
err6716 = 0.5 * (df["S2_6716_err_plus"].values + df["S2_6716_err_minus"].values) * UNIT_FLUX
err6731 = 0.5 * (df["S2_6730_err_plus"].values + df["S2_6730_err_minus"].values) * UNIT_FLUX
sn6716 = F6716 / err6716
sn6731 = F6731 / err6731

# luminosity
d_L = cosmo.luminosity_distance(z).to(u.cm).value
L6716 = 4 * np.pi * d_L**2 * F6716
L6731 = 4 * np.pi * d_L**2 * F6731

# [SII] ratio
df["S2_ratio"] = F6716 / F6731

# ==========================================
# マスク定義
# ==========================================
def valid_mass(x):
    x = np.asarray(x, float)
    m = np.isfinite(x)
    m &= (x > 0) & (x < 13)
    return m

m_sii = (
    np.isfinite(F6716) & np.isfinite(F6731) &
    np.isfinite(err6716) & np.isfinite(err6731) &
    (err6716 > 0) & (err6731 > 0)
)
m_sm = valid_mass(df["logM"])
m_ratio = np.isfinite(df["S2_ratio"])
mask_all = m_sii & m_sm & m_ratio

# 完全サンプル（有効化する場合は Lcut を使う）
# m_complete = mask_all & (L6716 >= Lcut) & (L6731 >= Lcut)
m_complete = mask_all
# m_incomplete = mask_all & (~m_complete)

# ==========================================
# ビン作成ユーティリティ
# ==========================================
def make_mass_edges(logM_values, bin_width):
    lo = np.floor(logM_values.min()/bin_width)*bin_width
    hi = np.ceil(logM_values.max()/bin_width)*bin_width + bin_width
    edges = np.arange(lo, hi, bin_width)
    return edges

def weighted_mean(flux, err):
    w = 1.0 / err**2
    mu = np.sum(w * flux) / np.sum(w)
    sigma = np.sqrt(1.0 / np.sum(w))
    return mu, sigma

rng = np.random.default_rng()

def in_interval(vals, lo, hi, inclusive="(,)"):
    """
    inclusive:
      "(,)":  lo <  vals <  hi
      "[,)":  lo <= vals <  hi
      "(,]":  lo <  vals <= hi
      "[,]":  lo <= vals <= hi
    """
    if inclusive == "(,)":
        return (vals > lo) & (vals < hi)
    elif inclusive == "[,)":
        return (vals >= lo) & (vals < hi)
    elif inclusive == "(,]":
        return (vals > lo) & (vals <= hi)
    elif inclusive == "[,]":
        return (vals >= lo) & (vals <= hi)
    else:
        raise ValueError("inclusive must be one of '(,)', '[,)', '(,]', '[,]'")

# 先に "global" 用の質量ビンを用意しておく（必要な場合）
if MASS_BIN_MODE == 'global':
    logM_all = df.loc[m_complete, "logM"].values
    mass_edges_global = make_mass_edges(logM_all, BIN_WIDTH)
else:
    mass_edges_global = None

# ==========================================
# z-bin ごとにスタック
# ==========================================
all_rows = []           # 全 z-bin を連結して out_csv に保存
res_by_z = {}           # 可視化用に保持

for zb in Z_BINS:
    name = zb["name"]
    color = zb["color"]
    lo, hi = zb["lo"], zb["hi"]
    inclusive = zb.get("inclusive", "(,)")

    # この z-bin のマスク
    m_z = m_complete & in_interval(z, lo, hi, inclusive=inclusive)

    # サンプルが空ならスキップ
    if not np.any(m_z):
        print(f"[{name}] no data")
        continue

    # この z-bin で使う logM と質量ビン
    logM_sub = df.loc[m_z, "logM"].values

    if MASS_BIN_MODE == 'global':
        edges = mass_edges_global
    else:
        edges = make_mass_edges(logM_sub, BIN_WIDTH)

    rows = []
    # 各 logM ビンでスタック
    for loM, hiM in zip(edges[:-1], edges[1:]):
        m_bin = (
            m_z &
            (df["logM"].values >= loM) &
            (df["logM"].values <  hiM)     # 左閉右開 [low, high)
        )
        N = int(np.sum(m_bin))
        if N < NMIN:
            continue

        # 取り出し
        f1 = F6716[m_bin]
        e1 = err6716[m_bin]
        f2 = F6731[m_bin]
        e2 = err6731[m_bin]

        # 重み付き平均の“スタックフラックス”
        F1, e1_stack = weighted_mean(f1, e1)
        F2, e2_stack = weighted_mean(f2, e2)

        # Monte Carlo で比の分布を推定
        f1_mc = rng.normal(F1, e1_stack, N_MC)
        f2_mc = rng.normal(F2, e2_stack, N_MC)
        valid = (f2_mc > 0)

        if not np.any(valid):
            continue

        R_mc = f1_mc[valid] / f2_mc[valid]
        R50 = np.nanmedian(R_mc)
        R16 = np.nanpercentile(R_mc, 16)
        R84 = np.nanpercentile(R_mc, 84)

        rows.append(dict(
            z_bin=name,
            z_lo=lo, z_hi=hi,
            logM_lo=loM, logM_hi=hiM,
            logM_cen=0.5*(loM+hiM),
            N=N,
            R_med=R50,
            R_err_lo=R50 - R16,
            R_err_hi=R84 - R50,
            N_MC_valid=int(valid.sum())
        ))

    res_z = pd.DataFrame(rows)
    res_by_z[name] = dict(df=res_z, color=color)
    all_rows.extend(rows)

# まとめて CSV 出力
res_all = pd.DataFrame(all_rows)
res_all.to_csv(out_csv, index=False)
print("Saved stacked table:", out_csv)

# ==========================================
# 描画
# ==========================================
fig, ax = plt.subplots(figsize=(12,6))

# スキャッタ（個々の点）を z-bin 色分けで描画
for zb in Z_BINS:
    name = zb["name"]; color = zb["color"]
    lo, hi = zb["lo"], zb["hi"]; inclusive = zb.get("inclusive", "(,)")

    m_z = m_complete & in_interval(z, lo, hi, inclusive=inclusive)
    if not np.any(m_z):
        continue

    ax.scatter(
        df.loc[m_z, "logM"],
        df.loc[m_z, "S2_ratio"],
        s=8.0, marker='.', alpha=1.0, color=color, 
    )

# スタック結果を上に重ねる
for name, pack in res_by_z.items():
    res_z = pack["df"]
    color = pack["color"]
    if len(res_z) == 0:
        continue
    x = res_z["logM_cen"].values
    y = res_z["R_med"].values
    yerr = np.vstack([res_z["R_err_lo"].values, res_z["R_err_hi"].values])

    ax.errorbar(
        x, y, yerr=yerr,
        fmt="s", mec=color, mfc=color,
        ecolor=color, color=color,
        capsize=3, ms=5, lw=1.2, 
    )

ax.set_xlabel(r"log $M_\star$ [M$_\odot$]")
ax.set_ylabel(r"[SII] 6717 / 6731")
ax.set_xlim(6, 12)
ax.set_ylim(0.0, 2.0)

# 体裁
for spine in ax.spines.values():
    spine.set_linewidth(2)

# ax.legend(ncol=2, fontsize=10, frameon=False)
plt.tight_layout()
plt.savefig(out_png, dpi=200)
plt.show()
print("Saved figure:", out_png)