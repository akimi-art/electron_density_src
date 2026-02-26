#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
スクリプトの概要:
logSFR ビンごとに
  [SII]6717,6731 フラックスをスタック
→ MCで ratio 分布
→ PyNebで ne 分布
→ P16, P50, P84 を保存・描画
→ 完全なサンプルのみを対象とする


使用方法:
    stacked_sii_ne_vs_sfr_v3.py [オプション]

著者: A. M.
作成日: 2026-02-26

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


# ===============================
# 入出力
# ===============================
current_dir = os.getcwd()

csv_path = "./results/JADES/JADES_DR3/data_from_Nishigaki/jades_info_crossmatch_with_logSFR.csv"
out_csv = "./results/table/stacked_sii_ratio_vs_sfr_JADES_DR3.csv"
out_png = "./results/figure/stacked_sii_ratio_vs_sfr_JADES_DR3.png"

sdss_csv = "./results/csv/stacked_sii_ne_vs_sfr_from_ratio_data.csv"

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
    dict(name="1<z<4", color="tab:blue",  lo=1.0, hi=4.0,  inclusive="(,)"),
    dict(name="4<z<7", color="tab:green", lo=4.0, hi=7.0,  inclusive="(,)"),
    dict(name="z>7",   color="tab:red",   lo=7.0, hi=np.inf, inclusive="(,]"),
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
# マスク
# ===============================
def valid_logSFR(x):
    return np.isfinite(x) & (x > -5) & (x < 3) & (x != -1)

m_valid = (
    np.isfinite(F6716) & np.isfinite(F6731) &
    (e6716 > 0) & (e6731 > 0) &
    valid_logSFR(df["logSFR_hb"].values)
)


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

    logSFR_sub = df.loc[m_z, "logSFR_hb"].values

    edges = np.arange(
        np.floor(logSFR_sub.min()/BIN_WIDTH)*BIN_WIDTH,
        np.ceil(logSFR_sub.max()/BIN_WIDTH)*BIN_WIDTH + BIN_WIDTH,
        BIN_WIDTH
    )

    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m_bin = (
            m_z &
            (df["logSFR_hb"].values >= lo) &
            (df["logSFR_hb"].values <  hi)
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
            logSFR_lo=lo,
            logSFR_hi=hi,
            logSFR_cen=0.5*(lo+hi),

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

for zb in Z_BINS:
    m = m_valid & in_interval(z, zb["lo"], zb["hi"], zb["inclusive"])
    ax.scatter(
        df.loc[m, "logSFR_hb"],
        df.loc[m, "S2_ratio"],
        s=8, alpha=0.6, color=zb["color"], label=zb["name"]
    )

for name, pack in res_by_z.items():
    res = pack["df"]
    if len(res) == 0: continue
    ax.errorbar(
        res["logSFR_cen"], res["R_med"],
        yerr=[res["R_err_lo"], res["R_err_hi"]],
        fmt="s", ms=8, mec=pack["color"], mfc=pack["color"],
        ecolor=pack["color"], capsize=3
    )

ax.set_xlabel(r"$\log(SFR)\ [M_\odot\,\mathrm{yr^{-1}}]$")
ax.set_ylabel(r"[SII] 6717 / 6731")
ax.set_xlim(0,2)
ax.set_ylim(0.5,2.0)

for s in ax.spines.values():
    s.set_linewidth(2)

plt.tight_layout()
plt.savefig(out_png, dpi=200)
plt.show()



# sdssのデータをプロットしたい場合は描画の前に以下をアンコメントして実行
# sdss = pd.read_csv(sdss_csv)

# x = sdss["logSFR_cen"].values
# y = sdss["R_med"].values
# yerr = [sdss["log_ne_err_lo"], sdss["log_ne_err_hi"]]

# thr = 0.0
# hi = x < thr

# ax.errorbar(x[hi], y[hi], yerr=[yerr[0][hi], yerr[1][hi]],
#             fmt="s", mfc="black", mec="black", ecolor="black", capsize=3, zorder=0)
# ax.errorbar(x[~hi], y[~hi], yerr=[yerr[0][~hi], yerr[1][~hi]],
#             fmt="s", mfc="white", mec="black", ecolor="black", capsize=3)