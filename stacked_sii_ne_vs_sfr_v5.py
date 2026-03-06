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
# 入出力
# ==========================================
current_dir = os.getcwd()

fits_path = os.path.join(
    current_dir,
    "results/fits/mpajhu_dr7_v5_2_merged_zlt0.2_Lgt1e+39.fits"
)

out_csv = os.path.join(
    current_dir,
    "results/csv/stacked_sii_ratio_vs_sfr_highz.csv"
)

out_png = os.path.join(
    current_dir,
    "results/figure/stacked_sii_ratio_vs_sfr_highz.png"
)

os.makedirs(os.path.dirname(out_csv), exist_ok=True)
os.makedirs(os.path.dirname(out_png), exist_ok=True)


# ==========================================
# パラメータ
# ==========================================
BIN_WIDTH = 0.1
NMIN = 100
N_MC = 5000

UNIT_FLUX = 1e-17


# ==========================================
# 読み込み
# ==========================================
tab = Table.read(fits_path, hdu=1)
df = tab.to_pandas()


# ==========================================
# 基本量
# ==========================================
z = df["Z"].values

F6716 = df["SII_6717_FLUX"].values * UNIT_FLUX
F6731 = df["SII_6731_FLUX"].values * UNIT_FLUX

err6716 = df["SII_6717_FLUX_ERR"].values * UNIT_FLUX
err6731 = df["SII_6731_FLUX_ERR"].values * UNIT_FLUX

df["R_SII"] = F6716 / F6731


# ==========================================
# マスク
# ==========================================
def valid_sfr(x):

    x = np.asarray(x, float)

    m = np.isfinite(x)
    m &= (x > -5) & (x < 3)
    m &= (x != -1.0)

    return m


m_sii = (
    np.isfinite(F6716) &
    np.isfinite(F6731) &
    np.isfinite(err6716) &
    np.isfinite(err6731) &
    (err6716 > 0) &
    (err6731 > 0)
)

m_sfr = valid_sfr(df["sfr_MEDIAN"])

m_ratio = np.isfinite(df["R_SII"])

mask_all = m_sii & m_sfr & m_ratio


# ==========================================
# high-z sample
# ==========================================
m_z_high = (z >= 0.15) & (z <= 0.20)

m_complete_highz = mask_all & m_z_high


# ==========================================
# bin作成（high-zのみ）
# ==========================================
logSFR = df.loc[m_complete_highz, "sfr_MEDIAN"].values

edges = np.arange(
    np.floor(logSFR.min()/BIN_WIDTH)*BIN_WIDTH,
    np.ceil(logSFR.max()/BIN_WIDTH)*BIN_WIDTH + BIN_WIDTH,
    BIN_WIDTH
)


# ==========================================
# weighted mean
# ==========================================
def weighted_mean(flux, err):

    w = 1.0 / err**2

    mu = np.sum(w * flux) / np.sum(w)

    sigma = np.sqrt(1.0 / np.sum(w))

    return mu, sigma


rng = np.random.default_rng()


# ==========================================
# stack関数
# ==========================================
def run_stack(mask):

    rows = []

    for lo, hi in zip(edges[:-1], edges[1:]):

        m_bin = (
            mask &
            (df["sfr_MEDIAN"] >= lo) &
            (df["sfr_MEDIAN"] < hi)
        )

        N = np.sum(m_bin)

        if N < NMIN:
            continue

        f1 = F6716[m_bin]
        e1 = err6716[m_bin]

        f2 = F6731[m_bin]
        e2 = err6731[m_bin]

        F1, e1_stack = weighted_mean(f1, e1)
        F2, e2_stack = weighted_mean(f2, e2)

        f1_mc = rng.normal(F1, e1_stack, N_MC)
        f2_mc = rng.normal(F2, e2_stack, N_MC)

        valid = (f2_mc > 0)

        R_mc = f1_mc[valid] / f2_mc[valid]

        R50 = np.nanmedian(R_mc)
        R16 = np.nanpercentile(R_mc, 16)
        R84 = np.nanpercentile(R_mc, 84)

        rows.append(dict(

            logSFR_lo=lo,
            logSFR_hi=hi,
            logSFR_cen=0.5*(lo+hi),

            N=N,

            R_med=R50,
            R_err_lo=R50-R16,
            R_err_hi=R84-R50

        ))

    return pd.DataFrame(rows)


# ==========================================
# stack実行（high-z）
# ==========================================
res_highz = run_stack(m_complete_highz)


# ==========================================
# CSV保存
# ==========================================
res_highz.to_csv(out_csv, index=False)

print("Saved:", out_csv)


# ==========================================
# 描画
# ==========================================
fig, ax = plt.subplots(figsize=(6,6))


# scatter
ax.scatter(

    df.loc[m_complete_highz, "sfr_MEDIAN"],
    df.loc[m_complete_highz, "R_SII"],

    s=0.01,
    marker='.',
    alpha=0.8,
    color="C0"

)


# stack結果
ax.errorbar(

    res_highz["logSFR_cen"],
    res_highz["R_med"],

    yerr=[
        res_highz["R_err_lo"],
        res_highz["R_err_hi"]
    ],

    fmt="s",
    color="k",
    mec="k",
    mfc="k",
    capsize=3,

    label="0.15 < z < 0.20"

)


ax.set_xlabel(r"$\log(SFR)\ [M_{\odot}\mathrm{yr^{-1}}]$")
ax.set_ylabel(r"[SII] 6717 / 6731")

ax.set_xlim(0, 2.0)
ax.set_ylim(0.5, 2.0)

for spine in ax.spines.values():
    spine.set_linewidth(2)

plt.tight_layout()

plt.savefig(out_png, dpi=200)

plt.show()

print("Saved:", out_png)