# =====================================================================
# plot_mstar_sfr_local_noagncut_highz.py
#   局所（AGN の除去の前。データの整理・z < z_max・光度カットは通ったもの）を黒、
#   高 z（JADES）を赤で M*–SFR 平面に描く
# =====================================================================
import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table

plt.rcParams.update({
    "figure.figsize": (12, 6), "font.size": 32, "axes.labelsize": 32, "axes.titlesize": 32,
    "axes.grid": False, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "xtick.major.size": 32, "ytick.major.size": 32, "xtick.major.width": 2, "ytick.major.width": 2,
    "xtick.minor.visible": True, "ytick.minor.visible": True,
    "xtick.minor.size": 8, "ytick.minor.size": 8, "xtick.minor.width": 1.5, "ytick.minor.width": 1.5,
    "xtick.labelsize": 28, "ytick.labelsize": 28,
    "font.family": "STIXGeneral", "mathtext.fontset": "stix",
})

LOCAL_FLAGS = "results/fits/mpajhu_dr7_v5_2_master_v2_Ka03_Reu.fits"   # 局所：全行 + 選択のフラグ
JADES_FLAGS = "results/JADES/sample/jades_all_with_flags.fits"
FIG_DIR     = "results/figure/comparison"
os.makedirs(FIG_DIR, exist_ok=True)

def f64(col):
    return np.ma.filled(np.ma.asarray(col).astype(float), np.nan)

# 局所：AGN の除去の前
loc = Table.read(LOCAL_FLAGS)
base    = np.asarray(loc["BASE_V2"], bool)       # データの整理 + z < z_max
lum     = np.asarray(loc["LUM_OK_V2"], bool)     # L([SII]6731) > 1e39
not_agn = np.asarray(loc["NOT_AGN_V2"], bool)
parent  = np.asarray(loc["PARENT_V2"], bool)     # base かつ AGN でない かつ フラックスが妥当
flux_bad = base & not_agn & ~parent              # フラックスの失敗（L >= 1e43）
pre_agn = base & lum & ~flux_bad
logM_l, logSFR_l = f64(loc["sm_MEDIAN"])[pre_agn], f64(loc["sfr_MEDIAN"])[pre_agn]
print(f"局所（AGN の除去の前）: {pre_agn.sum():,}   最終サンプル: {np.sum(loc['SELECTED_V2']):,}")

# 高 z：M* と SFR の両方が妥当な銀河
jad = Table.read(JADES_FLAGS)
sel_j = np.asarray(jad["SAMPLE_SSFR"], bool)
logM_j, logSFR_j = f64(jad["N26_logM"])[sel_j], f64(jad["N26_logSFR_hb"])[sel_j]

fig, ax = plt.subplots(figsize=(12, 10))
ax.scatter(logM_l, logSFR_l, s=0.5, alpha=0.05, color="k", rasterized=True)
ax.scatter(logM_j, logSFR_j, s=30, color="firebrick", zorder=5)
ax.scatter([], [], s=60, color="k",         label=f"Local, before AGN removal ({len(logM_l):,})")
ax.scatter([], [], s=60, color="firebrick", label=f"High-$z$ ({len(logM_j)})")
ax.set_xlim(6, 12); ax.set_ylim(-3, 3)
ax.set_xlabel(r"$\log(M_\ast/M_\odot)$")
ax.set_ylabel(r"$\log({\rm SFR}/M_\odot\,{\rm yr}^{-1})$")
ax.legend(loc="upper left", fontsize=22)
for sp in ax.spines.values():
    sp.set_linewidth(2)

out = f"{FIG_DIR}/mstar_sfr_local_noagncut_highz.png"
plt.tight_layout()
plt.savefig(out, dpi=200)
plt.show()
print(f"[DONE] {out}")