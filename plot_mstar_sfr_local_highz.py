# =====================================================================
# plot_mstar_sfr_local_highz.py
#   局所（SDSS、最終サンプル）を黒、高 z（JADES）を赤で M*–SFR 平面に描く
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

LOCAL_SAMPLE = "results/fits/sdss_sample_v2_zlt0.1879_Lgt1e+39_Ka03_Reu.fits"   # 局所の最終サンプル
JADES_FLAGS  = "results/JADES/sample/jades_all_with_flags.fits"
FIG_DIR      = "results/figure/comparison"
os.makedirs(FIG_DIR, exist_ok=True)

def f64(col):
    return np.ma.filled(np.ma.asarray(col).astype(float), np.nan)

# 局所：最終サンプル
loc = Table.read(LOCAL_SAMPLE)
logM_l, logSFR_l = f64(loc["sm_MEDIAN"]), f64(loc["sfr_MEDIAN"])

# 高 z：M* と SFR の両方が妥当な銀河
jad = Table.read(JADES_FLAGS)
sel_j = np.asarray(jad["SAMPLE_SSFR"], bool)
logM_j, logSFR_j = f64(jad["N26_logM"])[sel_j], f64(jad["N26_logSFR_hb"])[sel_j]

fig, ax = plt.subplots(figsize=(12, 10))
ax.scatter(logM_l, logSFR_l, s=5, alpha=0.5, color="gray", rasterized=True)
ax.scatter(logM_j, logSFR_j, s=30, color="firebrick", zorder=5)
ax.scatter([], [], s=60, color="k",         label=f"Local ({len(logM_l):,})")
ax.scatter([], [], s=60, color="firebrick", label=f"High-$z$ ({len(logM_j)})")
ax.set_xlim(6, 12); ax.set_ylim(-3, 3)
ax.set_xlabel(r"$\log(M_\ast/M_\odot)$")
ax.set_ylabel(r"$\log({\rm SFR}/M_\odot\,{\rm yr}^{-1})$")
ax.legend(loc="upper left", fontsize=24)
for sp in ax.spines.values():
    sp.set_linewidth(2)

out = f"{FIG_DIR}/mstar_sfr_local_highz.png"
plt.tight_layout()
plt.savefig(out, dpi=200)
plt.show()
print(f"[DONE] {out}")




# import numpy as np
# import pandas as pd
# from astropy.table import Table

# t = Table.read("results/JADES/JADES_DR3/master/jades_dr3_mr_step3_shibuya.fits")
# f64 = lambda c: np.ma.filled(np.ma.asarray(c).astype(float), np.nan)

# df = pd.DataFrame({
#     "in_n26":  np.asarray(t["IN_N26"], bool),
#     "zflag":   np.char.strip(np.asarray(t["z_Spec_flag"]).astype(str)),
#     "dr_flag": np.asarray(t["DR_flag"]).astype(bool),
#     "z":       f64(t["z_Spec"]),
#     "field":   np.asarray(t["FILE_FIELD"]).astype(str),
# })
# hb, hbe = f64(t["HB_4861_flux"]), f64(t["HB_4861_err"])
# with np.errstate(invalid="ignore", divide="ignore"):
#     df["hb_sn"] = hb / hbe

# print("z > 0 の行:", np.sum(df["z"] > 0), "/", len(df))

# print("\n[1] z の信頼度のフラグ × Nishigaki+26 に入ったか")
# print(pd.crosstab(df["zflag"], df["in_n26"], margins=True))

# print("\n[2] データ処理のフラグ × Nishigaki+26 に入ったか（z > 0 の行）")
# m = df["z"] > 0
# print(pd.crosstab(df.loc[m, "dr_flag"], df.loc[m, "in_n26"], margins=True))

# print("\n[3] z の信頼度 × データ処理のフラグ ごとの、Nishigaki+26 に入った割合（z > 0 の行）")
# print(df[m].groupby(["zflag", "dr_flag"])["in_n26"].agg(["sum", "count", "mean"]).round(2))

# print("\n[4] Hβ の S/N（z > 0 の行）")
# for flag in [True, False]:
#     s = df.loc[m & (df["in_n26"] == flag), "hb_sn"]
#     print(f"  Nishigaki+26 に {'入った' if flag else '入らなかった'}: N = {len(s)}"
#           f"   Hβ あり {np.isfinite(s).sum()}   S/N>3 {np.sum(s > 3)}   中央値 {np.nanmedian(s):.1f}")



# a = (df["zflag"] == "A") & (df["z"] > 0)

# print("[5] z のビンごと（フラグ A）")
# zb = pd.cut(df.loc[a, "z"], [0, 1, 2, 3, 4, 5, 6, 7, 10])
# print(df[a].groupby(zb, observed=True)["in_n26"].agg(["sum", "count", "mean"]).round(2))

# print("\n[6] 観測の区分ごと（フラグ A）")
# df["tier"] = np.char.strip(np.asarray(t["TIER"]).astype(str))
# print(df[a].groupby("tier")["in_n26"].agg(["sum", "count", "mean"]).round(2))

# print("\n[7] NIRCam の対応天体があるか（フラグ A）")
# df["has_nircam"] = np.asarray(t["NIRCam_ID"]).astype(np.int64) > 0
# print(pd.crosstab(df.loc[a, "has_nircam"], df.loc[a, "in_n26"], margins=True))

# print("\n[8] 観測したグレーティングの組み合わせ（フラグ A）")
# for g in ["assigned_Prism", "assigned_G140M", "assigned_G235M", "assigned_G395M"]:
#     df[g] = np.char.strip(np.asarray(t[g]).astype(str)) == "T"
# combo = (df["assigned_Prism"].map({True: "P", False: "-"}) + df["assigned_G140M"].map({True: "1", False: "-"})
#          + df["assigned_G235M"].map({True: "2", False: "-"}) + df["assigned_G395M"].map({True: "3", False: "-"}))
# df["combo"] = combo
# print(df[a].groupby("combo")["in_n26"].agg(["sum", "count", "mean"]).round(2))


# n26 = pd.read_csv("results/JADES/JADES_DR3/data_from_Nishigaki/jades_info.csv")
# print(n26.notna().sum().to_string())