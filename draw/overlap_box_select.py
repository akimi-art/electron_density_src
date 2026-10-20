# =====================================================================
# overlap_box_select.py
#   M*–SFR 平面で、局所と高 z が重なる箱を決め、両方のサブサンプルを作る
#   1. 候補の箱ごとに、銀河の数と中央値を数える
#   2. 選んだ箱を図に描き、サブサンプルの一覧を保存する
#   3. 選択内の local / high-z の log M*・log SFR 分布を比較する
# =====================================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
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

# 候補の箱：(log M* 下限, 上限, 質量下限での log SFR 下限, 上限, 傾き)
# log SFR の境界は slope * (log M* - 質量下限) だけ上がる。傾き 0 なら長方形。
BOXES = [
    (8.5, 10.0, -0.5, 0.0, 1.0),
]
# rectangle: 左右辺を上下辺に直角にする / parallelogram: 従来の垂直な左右辺
# rectangle では質量の2値は「中心線上の両端の質量」。頂点の質量は少し外へ広がる。
BOX_SHAPE = "rectangle"
CHOSEN = 0          # 数を見て選んだ箱の番号
HIST_BINS = 12      # 両サンプルで共通の等幅ビン数

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
    m_min, m_max, sfr_min, sfr_max, slope = b
    offset_sfr = s - slope * (m - m_min)
    if BOX_SHAPE == "rectangle":
        # 中心線への射影。端面は中心線に垂直で、中心線の両端を通る。
        mid_sfr = (sfr_min + sfr_max) / 2
        along = (m - m_min) + slope * (s - mid_sfr)
        ends = (along >= 0) & (along < (1 + slope ** 2) * (m_max - m_min))
    elif BOX_SHAPE == "parallelogram":
        ends = (m >= m_min) & (m < m_max)
    else:
        raise ValueError("BOX_SHAPE must be rectangle or parallelogram")
    return ends & (offset_sfr >= sfr_min) & (offset_sfr < sfr_max)


def box_vertices(b):
    m_min, m_max, sfr_min, sfr_max, slope = b
    shift = slope * (m_max - m_min)
    if BOX_SHAPE == "parallelogram":
        return [(m_min, sfr_min), (m_max, sfr_min + shift),
                (m_max, sfr_max + shift), (m_min, sfr_max)]
    mid = (sfr_min + sfr_max) / 2
    # 法線方向へ移動し、従来の上下境界線と同じ直線上に頂点を置く。
    normal = np.array([-slope, 1.0]) * (sfr_max - sfr_min) / (2 * (1 + slope ** 2))
    start, end = np.array([m_min, mid]), np.array([m_max, mid + shift])
    return [start - normal, end - normal, end + normal, start + normal]

# ---- 1. 候補の箱ごとの数と中央値 ----
rows = []
for k, b in enumerate(BOXES):
    ml = in_box(logM_l, logSFR_l, b)
    mj = sel_j & in_box(logM_j, logSFR_j, b)
    rows.append({
        "box": k, "logM": f"{b[0]}–{b[1]}",
        "logSFR_at_Mmin": f"{b[2]}–{b[3]}", "slope": b[4],
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

fig, ax = plt.subplots(figsize=(12, 12))
ax.scatter(logM_l, logSFR_l, s=5, alpha=0.5, color="gray", rasterized=True)
ax.scatter(logM_j[sel_j], logSFR_j[sel_j], s=30, color="firebrick", zorder=5)
vertices = box_vertices(b)
ax.add_patch(Polygon(vertices, closed=True,
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
plt.savefig(f"{FIG_DIR}/mstar_sfr_overlap_box{CHOSEN}_slope{b[4]}_{BOX_SHAPE}.png", dpi=200)
print(f"[DONE] {FIG_DIR}/mstar_sfr_overlap_box{CHOSEN}_slope{b[4]}_{BOX_SHAPE}.png")

# ---- 3. 選択領域内の質量・SFR 分布（面積を 1 に正規化） ----
# SFR の表示範囲は傾いた箱の全頂点から決める（負の傾きにも対応）。
mass_limits = (min(v[0] for v in vertices), max(v[0] for v in vertices))
sfr_limits = (min(v[1] for v in vertices), max(v[1] for v in vertices))
fig_hist, axes = plt.subplots(1, 2, figsize=(16, 8))
panels = [
    (logM_l[ml], logM_j[mj], mass_limits,
     r"$\log(M_\ast/M_\odot)$"),
    (logSFR_l[ml], logSFR_j[mj], sfr_limits,
     r"$\log({\rm SFR}/M_\odot\,{\rm yr}^{-1})$"),
]
for ax_hist, (local_values, highz_values, limits, xlabel) in zip(axes, panels):
    edges = np.linspace(*limits, HIST_BINS + 1)
    for values, color, label in [
        (local_values, "gray", "Local"),
        (highz_values, "firebrick", "High-$z$"),
    ]:
        values = values[np.isfinite(values)]
        if values.size:
            median = np.median(values)
            ax_hist.hist(values, bins=edges, density=True, histtype="step",
                         linewidth=2.5, color=color,
                         label=f"{label}: N={values.size:,}, median={median:.2f}")
            ax_hist.axvline(median, color=color, linestyle="--", linewidth=2)
        else:
            # 空サンプルでは density=True によるゼロ除算を避ける。
            ax_hist.plot([], [], color=color, label=f"{label}: N=0")
    ax_hist.set_xlim(*limits)
    ax_hist.set_xlabel(xlabel)
    ax_hist.set_ylabel("Probability density")
    ax_hist.tick_params(which="major", length=12)
    for sp in ax_hist.spines.values():
        sp.set_linewidth(2)
fig_hist.tight_layout()
hist_path = f"{FIG_DIR}/mstar_sfr_hist_overlap_box{CHOSEN}_slope{b[4]}_{BOX_SHAPE}.png"
fig_hist.savefig(hist_path, dpi=200)
print(f"[DONE] {hist_path}")

tag = f"logM{b[0]}-{b[1]}_logSFR{b[2]}-{b[3]}"
if b[4] != 0:
    tag += f"_slope{b[4]}"
if BOX_SHAPE == "rectangle":
    tag += "_rectangle"
loc[ml].write(f"{OUT_DIR}/local_box_{tag}.fits", overwrite=True)
jad[mj].write(f"{OUT_DIR}/highz_box_{tag}.fits", overwrite=True)
print(f"\n[DONE] 局所 {ml.sum():,} 銀河 → {OUT_DIR}/local_box_{tag}.fits")
print(f"[DONE] 高 z {mj.sum()} 銀河 → {OUT_DIR}/highz_box_{tag}.fits")
print(f"[INFO] overlap_box_ne.py の TAG に指定する値: {tag}")
plt.show()
