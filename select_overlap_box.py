# =====================================================================
# select_overlap_box.py
#   M*–SFR 平面で、局所と高 z が重なる箱を決め、両方のサブサンプルを作る
#   1. 候補の箱ごとに、銀河の数と中央値を数える
#   2. 選んだ箱を図に描き、サブサンプルの一覧を保存する
# =====================================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
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

LOCAL_SAMPLE = "results/fits/sdss_sample_v2_zlt0.1879_Lgt1e+39_Ka03_Reu.fits"
JADES_FLAGS  = "results/JADES/sample/jades_all_with_flags.fits"
OUT_DIR      = "results/comparison"
FIG_DIR      = "results/figure/comparison"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# 候補の箱：(log M* の下限, 上限, log SFR の下限, 上限)
BOXES = [
    (9.0, 10.0, 0.0, 2.0),
]
CHOSEN = 0          # 数を見て選んだ箱の番号

def f64(col):
    return np.ma.filled(np.ma.asarray(col).astype(float), np.nan)

loc = Table.read(LOCAL_SAMPLE)
logM_l, logSFR_l = f64(loc["sm_MEDIAN"]), f64(loc["sfr_MEDIAN"])
z_l = f64(loc["Z"])

jad = Table.read(JADES_FLAGS)
sel_j = np.asarray(jad["SAMPLE_SSFR"], bool)
logM_j, logSFR_j = f64(jad["N26_logM"]), f64(jad["N26_logSFR_hb"])
z_j = f64(jad["z_Spec"])

def in_box(m, s, b):
    return (m >= b[0]) & (m < b[1]) & (s >= b[2]) & (s < b[3])

# ---- 1. 候補の箱ごとの数と中央値 ----
rows = []
for k, b in enumerate(BOXES):
    ml = in_box(logM_l, logSFR_l, b)
    mj = sel_j & in_box(logM_j, logSFR_j, b)
    rows.append({
        "box": k, "logM": f"{b[0]}–{b[1]}", "logSFR": f"{b[2]}–{b[3]}",
        "N_local": int(ml.sum()), "N_highz": int(mj.sum()),
        "med_logM_local": np.median(logM_l[ml]) if ml.any() else np.nan,
        "med_logM_highz": np.median(logM_j[mj]) if mj.any() else np.nan,
        "med_logSFR_local": np.median(logSFR_l[ml]) if ml.any() else np.nan,
        "med_logSFR_highz": np.median(logSFR_j[mj]) if mj.any() else np.nan,
        "med_z_highz": np.median(z_j[mj]) if mj.any() else np.nan,
    })
tab = pd.DataFrame(rows)
print(tab.round(2).to_string(index=False))

# ---- 2. 選んだ箱を描き、サブサンプルを保存する ----
b = BOXES[CHOSEN]
ml = in_box(logM_l, logSFR_l, b)
mj = sel_j & in_box(logM_j, logSFR_j, b)

fig, ax = plt.subplots(figsize=(12, 10))
ax.scatter(logM_l, logSFR_l, s=5, alpha=0.5, color="gray", rasterized=True)
ax.scatter(logM_j[sel_j], logSFR_j[sel_j], s=30, color="firebrick", zorder=5)
ax.add_patch(Rectangle((b[0], b[2]), b[1] - b[0], b[3] - b[2],
                       fill=False, edgecolor="tab:blue", lw=3, zorder=6))
ax.scatter([], [], s=60, color="k",         label=f"Local ({len(logM_l):,}; in box {ml.sum():,})")
ax.scatter([], [], s=60, color="firebrick", label=f"High-$z$ ({sel_j.sum()}; in box {mj.sum()})")
ax.set_xlim(6, 12); ax.set_ylim(-3, 3)
ax.set_xlabel(r"$\log(M_\ast/M_\odot)$")
ax.set_ylabel(r"$\log({\rm SFR}/M_\odot\,{\rm yr}^{-1})$")
ax.legend(loc="upper left", fontsize=20)
for sp in ax.spines.values():
    sp.set_linewidth(2)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/mstar_sfr_overlap_box{CHOSEN}.png", dpi=200)
plt.show()

tag = f"logM{b[0]}-{b[1]}_logSFR{b[2]}-{b[3]}"
loc[ml].write(f"{OUT_DIR}/local_box_{tag}.fits", overwrite=True)
jad[mj].write(f"{OUT_DIR}/highz_box_{tag}.fits", overwrite=True)
print(f"\n[DONE] 局所 {ml.sum():,} 銀河 → {OUT_DIR}/local_box_{tag}.fits")
print(f"[DONE] 高 z {mj.sum()} 銀河 → {OUT_DIR}/highz_box_{tag}.fits")