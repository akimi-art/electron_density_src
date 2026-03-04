#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
スクリプトの概要:
logSFR ビンごとに
  [SII]6717,6731 フラックスをスタック
→ MCで ratio 分布
→ PyNebで ne 分布
→ P16, P50, P84 を保存・描画


使用方法:
    stacked_sii_ne_vs_ssfr_v1.py [オプション]

著者: A. M.
作成日: 2026-02-27

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


# ===============================
# 入出力
# ===============================
current_dir = os.getcwd()

csv_path = "./results/JADES/JADES_DR3/data_from_Nishigaki/jades_info_crossmatch_with_logSFR.csv"
out_csv = "./results/csv/stacked_sii_ratio_vs_ssfr_JADES_DR3.csv"
out_png = "./results/figure/stacked_sii_ratio_vs_ssfr_JADES_DR3.png"

sdss_csv = "./results/csv/stacked_sii_ne_vs_ssfr_from_ratio_COMPLETE.csv"

os.makedirs(os.path.dirname(out_csv), exist_ok=True)
os.makedirs(os.path.dirname(out_png), exist_ok=True)


# ===============================
# パラメータ
# ===============================
BIN_WIDTH = 0.1
NMIN = 1
N_MC = 5000
UNIT_FLUX = 1e-20
Z_BINS = [
    dict(name="0.5<z<2.0", color="tab:blue",  lo=0.5, hi=2.0,  inclusive="(,)", n_ssfr_bin=2),
    dict(name="2.0<z<3.0", color="tab:green", lo=2.0, hi=3.0,  inclusive="(,)", n_ssfr_bin=2),
    dict(name="3.0<z<6.0", color="tab:red",   lo=3.0, hi=6.0, inclusive="(,]", n_ssfr_bin=1),
]

# ===============================
# ユーティリティ
# ===============================
def in_interval(x, lo, hi, inclusive="(,)"):
    if inclusive == "(,)": return (x > lo) & (x < hi)
    if inclusive == "[,)": return (x >= lo) & (x < hi)
    if inclusive == "(,]": return (x > lo) & (x <= hi)
    if inclusive == "[,]": return (x >= lo) & (x <= hi)
    raise ValueError("invalid inclusive flag")
def weighted_mean(f, e):
    w = 1.0 / e**2
    return np.sum(w*f)/np.sum(w), np.sqrt(1.0/np.sum(w))


# ===============================
# データ読み込み & 基本量
# ===============================
df = pd.read_csv(csv_path)

z = df["z_spec"].values

F6716 = df["S2_6718_flux"].values * UNIT_FLUX
F6731 = df["S2_6733_flux"].values * UNIT_FLUX
e6716 = df["S2_6718_err"].values * UNIT_FLUX
e6731 = df["S2_6733_err"].values * UNIT_FLUX

df["S2_ratio"] = F6716 / F6731


# ===============================
# sSFR 定義
# ===============================

logSFR = df["logSFR_hb"].values
logSM  = df["logM"].values

log_sSFR = logSFR - logSM
df["log_sSFR"] = log_sSFR

# ===============================
# マスク
# ===============================
def valid_mass(x):
    x = np.asarray(x, float)
    m = np.isfinite(x)
    m &= (x > 0) & (x < 13)
    return m

def valid_sfr(x):
    return np.isfinite(x) & (x > -5) & (x < 3) & (x != -1)

m_sm   = valid_mass(logSM)
m_sfr  = valid_sfr(logSFR)

m_valid = (
    np.isfinite(F6716) & np.isfinite(F6731) &
    (e6716 > 0) & (e6731 > 0) &
    m_sm & m_sfr &
    np.isfinite(log_sSFR)
)


# ===============================
# z-bin × SFR-bin スタック（z情報付き）
# ===============================
rng = np.random.default_rng()

all_rows = []     # ← CSV 保存用
res_by_z = {}     # ← 可視化用


# ===============================
# z-bin × SFR-bin スタック（z情報付き）
# ===============================
rng = np.random.default_rng()

all_rows = []     # ← CSV 保存用
res_by_z = {}     # ← 可視化用

for zb in Z_BINS:
    name  = zb["name"]
    color = zb["color"]
    z_lo, z_hi = zb["lo"], zb["hi"]
    inclusive = zb.get("inclusive", "(,)")

    m_z = m_valid & in_interval(z, z_lo, z_hi, inclusive)

    if not np.any(m_z):
        print(f"[{name}] no data")
        continue

    logsSFR_sub = df.loc[m_z, "log_sSFR"].values

    # ★ ここに入れる
    n_bin = zb["n_ssfr_bin"]
    losSFR = logsSFR_sub.min()
    hisSFR = logsSFR_sub.max()
    edges = np.linspace(losSFR, hisSFR, n_bin + 1)

    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m_bin = (
            m_z &
            (df["log_sSFR"].values >= lo) &
            (df["log_sSFR"].values <  hi)
        )

        N = int(np.sum(m_bin))
        if N < NMIN:
            continue

        # --- stack ---
        F1, e1 = weighted_mean(F6716[m_bin], e6716[m_bin])
        F2, e2 = weighted_mean(F6731[m_bin], e6731[m_bin])

        f1_mc = rng.normal(F1, e1, N_MC)
        f2_mc = rng.normal(F2, e2, N_MC)
        valid = f2_mc > 0

        if not np.any(valid):
            continue

        R_mc = f1_mc[valid] / f2_mc[valid]
        R50 = np.nanmedian(R_mc)
        R16 = np.nanpercentile(R_mc, 16)
        R84 = np.nanpercentile(R_mc, 84)

        row = dict(
            # --- z-bin 情報（← 追加点） ---
            z_bin=name,
            z_lo=z_lo,
            z_hi=z_hi,

            # --- SFR bin ---
            logsSFR_lo=lo,
            logsSFR_hi=hi,
            logsSFR_cen=0.5*(lo+hi),

            # --- stack 結果 ---
            N=N,
            R_med=R50,
            R_err_lo=R50 - R16,
            R_err_hi=R84 - R50,
            N_MC_valid=int(valid.sum())
        )

        rows.append(row)
        all_rows.append(row)

    res_by_z[name] = dict(
        df=pd.DataFrame(rows),
        color=color
    )

# ===== ここから下に書く =====
res_all = pd.DataFrame(all_rows)
res_all.to_csv(out_csv, index=False)
print("Saved:", out_csv)

# ===============================
# 描画
# ===============================
fig, ax = plt.subplots(figsize=(6,6))

# ← ここが余白調整
fig.subplots_adjust(
    left=0.15,
    right=0.95,
    bottom=0.15,
    top=0.95,
    wspace=0.0,
    hspace=0.0
)

for zb in Z_BINS:
    m = m_valid & in_interval(z, zb["lo"], zb["hi"], zb["inclusive"])
    ax.scatter(
        df.loc[m, "log_sSFR"],
        df.loc[m, "S2_ratio"],
        s=8, alpha=0.6,
        color=zb["color"], label=zb["name"]
    )

for name, pack in res_by_z.items():
    res = pack["df"]
    if len(res) == 0:
        continue
    ax.errorbar(
        res["logsSFR_cen"], res["R_med"],
        yerr=[res["R_err_lo"], res["R_err_hi"]],
        fmt="s", ms=8,
        mec=pack["color"], mfc=pack["color"],
        ecolor=pack["color"], capsize=3
    )


# =================================================
# SDSS
# =================================================
res = pd.read_csv(sdss_csv)

x   = res["logsSFR_cen"].to_numpy(float)
y   = res["log_ne_med"].to_numpy(float)
elo = res["log_ne_err_lo"].to_numpy(float)
ehi = res["log_ne_err_hi"].to_numpy(float)

m_twoside = (
    np.isfinite(x) &
    np.isfinite(y) &
    np.isfinite(elo) &
    np.isfinite(ehi)
)

thr = -11
mask_lt = (x < thr) & m_twoside
mask_ge = (x >= thr) & m_twoside

# Ensure error values are non-negative
elo_sdss_mask_lt = np.maximum(0, elo[mask_lt])
ehi_sdss_mask_lt = np.maximum(0, ehi[mask_lt])
elo_sdss_mask_ge = np.maximum(0, elo[mask_ge])
ehi_sdss_mask_ge = np.maximum(0, ehi[mask_ge])

# x < thr（白四角）
ax.errorbar(
    x[mask_lt], y[mask_lt],
    yerr=np.vstack([elo_sdss_mask_lt, ehi_sdss_mask_lt]),
    fmt="s",
    mfc="white", mec="black",
    ecolor="black", color="black",
    capsize=3,
    label=f"SDSS logSFR < {thr}"
)

# x >= thr（黒四角）
ax.errorbar(
    x[mask_ge], y[mask_ge],
    yerr=np.vstack([elo_sdss_mask_ge, ehi_sdss_mask_ge]),
    fmt="s",
    mfc="black", mec="black",
    ecolor="black", color="black",
    capsize=3,
    label=f"SDSS logSFR ≥ {thr}"
)


for s in ax.spines.values():
    s.set_linewidth(2)
ax.set_xlim(-11, -7)
ax.set_xlabel("log sSFR [yr$^{-1}$]")
ax.set_ylabel("[SII] 6716 / 6731")
plt.savefig(out_png, dpi=300)
print("Saved:", out_png)
plt.show()