import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.table import Table
import astropy.units as u

import selection as S

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

current_dir = os.getcwd()
fits_path = os.path.join(current_dir, "results/fits/mpajhu_dr7_v5_2_merged_radius.fits")
fig_dir = os.path.join(current_dir, "results/figure/samples")
out_dir = os.path.join(current_dir, "results/fits/samples")
os.makedirs(fig_dir, exist_ok=True)
os.makedirs(out_dir, exist_ok=True)

t = Table.read(fits_path, format="fits")
df = t.to_pandas()
q = S.derived_quantities(df)

# ============================================================
# 0. 確認：z_max と等価幅の符号
# ============================================================
Z_MAX = S.z_max_from_flux_limit()
print(f"[INFO] Z_MAX = {Z_MAX:.4f}（L_MIN = {S.L_MIN:.0e}, FLUX_LIMIT = {S.FLUX_LIMIT:.0e}）")

pct = np.nanpercentile(q["EQW_raw"], [5, 50, 95])
print(f"[CHECK] H_ALPHA_EQW の 5/50/95 %点（生の値）: {pct}")
print("        星形成銀河の多いカタログで中央値が負なら EQW_EMISSION_NEGATIVE = True")

# ============================================================
# 1. 派生量と分類の結果を列として追加する（すべての FITS に共通）
# ============================================================
t["L_SII6731"]   = q["L6731"]
t["logsSFR"]     = q["logsSFR"]
t["Re_kpc"]      = q["Re_kpc"]
t["logSigmaSFR"] = q["logSigmaSFR"]
t["log_N2Ha"]    = q["log_N2Ha"]
t["log_O3Hb"]    = q["log_O3Hb"]
t["W_Ha"]        = q["W_Ha"]
for method in S.METHODS:
    sn, sf = S.classification(q, method)
    t[f"SF_{method}"] = sn & sf

# ============================================================
# 2. 3つの方法でサンプルを作り、保存する
# ============================================================
flow_rows = []
masks_all = {}
for method in S.METHODS:
    m = S.selection_masks(df, q, Z_MAX, method)
    masks_all[method] = m

    print(f"\n===== Selection flow ({method}) =====")
    for name, mk in m["flow"]:
        print(f"  {name:36s}: {mk.sum():,}")
        flow_rows.append({"method": method, "step": name, "N": int(mk.sum())})

    sel = m["selected"]
    print(f"  z range    : {np.nanmin(q['z'][sel]):.4f} -- {np.nanmax(q['z'][sel]):.4f}")
    print(f"  logM range : {np.nanmin(q['logM'][sel]):.2f} -- {np.nanmax(q['logM'][sel]):.2f}")

    out_path = os.path.join(
        out_dir,
        f"mpajhu_dr7_zlt{Z_MAX:.4f}_Lgt{S.L_MIN:.0e}_{method}.fits"
    )
    t[sel].write(out_path, format="fits", overwrite=True)
    print(f"[DONE] {out_path}")

flow_df = pd.DataFrame(flow_rows)
flow_path = os.path.join(out_dir, "selection_flow.csv")
flow_df.to_csv(flow_path, index=False)
print(f"\n[DONE] 選択の流れ: {flow_path}")

# ============================================================
# 3. 体積限定の図（L–z）
#    縦線（Z_MAX）と横線（L_MIN）が、フラックス限界の曲線上で交わる
# ============================================================
m0 = masks_all[S.METHODS[0]]
vol = m0["base"] & m0["lum"]

fig, ax = plt.subplots(figsize=(12, 6))
ok = np.isfinite(q["L6731"]) & (q["L6731"] > 0)
ax.scatter(q["z"][ok], q["L6731"][ok], s=0.2, alpha=0.2, color="gray", rasterized=True)
ax.scatter(q["z"][vol], q["L6731"][vol], s=0.2, alpha=0.5, color="firebrick", rasterized=True)

zg = np.linspace(1e-4, 0.4, 400)
Lg = 4 * np.pi * S.COSMO.luminosity_distance(zg).to(u.cm).value**2 * S.FLUX_LIMIT
ax.plot(zg, Lg, color="k", lw=2)
ax.axvline(Z_MAX, color="k", lw=2)
ax.axhline(S.L_MIN, color="k", lw=2)

ax.set_yscale("log")
ax.set_xlim(0, 0.4); ax.set_ylim(1e36, 1e42)
ax.set_xlabel(r"$z$")
ax.set_ylabel(r"$L([{\rm S\,II}]\lambda6731)$ [erg s$^{-1}$]")
save = os.path.join(fig_dir, "sii6731_luminosity_vs_z.png")
plt.savefig(save, dpi=200, bbox_inches="tight"); plt.show()
print(f"[DONE] {save}")

# ============================================================
# 4. 分類図（BPT 図と WHAN 図）
# ============================================================
x, y, W = q["log_N2Ha"], q["log_O3Hb"], q["W_Ha"]

# 4a. BPT 図（Ka03 と Ke01 の線を両方描く）
pre = m0["base"] & m0["lum"] & masks_all["Ka03"]["sn"]
fig, ax = plt.subplots(figsize=(10, 9))
ax.scatter(x[pre], y[pre], s=0.5, alpha=0.1, color="gray", rasterized=True)
ax.scatter(x[masks_all["Ka03"]["selected"]], y[masks_all["Ka03"]["selected"]],
           s=0.5, alpha=0.2, color="firebrick", rasterized=True)
xx = np.linspace(-2.0, 0.04, 500)
ax.plot(xx, 0.61 / (xx - 0.05) + 1.3, color="k", lw=2, label="Kauffmann+03")
xx = np.linspace(-2.0, 0.46, 500)
ax.plot(xx, 0.61 / (xx - 0.47) + 1.19, color="k", lw=2, ls="--", label="Kewley+01")
ax.set_xlim(-2.0, 0.5); ax.set_ylim(-1.5, 1.5)
ax.set_xlabel(r"$\log([{\rm N\,II}]\lambda6584/{\rm H}\alpha)$")
ax.set_ylabel(r"$\log([{\rm O\,III}]\lambda5007/{\rm H}\beta)$")
ax.legend(loc="lower left")
save = os.path.join(fig_dir, "bpt_diagram.png")
plt.savefig(save, dpi=200, bbox_inches="tight"); plt.show()
print(f"[DONE] {save}")

# 4b. WHAN 図
pre = m0["base"] & m0["lum"] & masks_all["WHAN"]["sn"]
with np.errstate(divide="ignore", invalid="ignore"):
    logW = np.log10(W)
fig, ax = plt.subplots(figsize=(10, 9))
ax.scatter(x[pre], logW[pre], s=0.5, alpha=0.1, color="gray", rasterized=True)
ax.scatter(x[masks_all["WHAN"]["selected"]], logW[masks_all["WHAN"]["selected"]],
           s=0.5, alpha=0.2, color="firebrick", rasterized=True)
ax.axvline(-0.4, color="k", lw=2)
ax.axhline(np.log10(3), color="k", lw=2)
ax.plot([-0.4, 1.0], [np.log10(6)] * 2, color="k", lw=2, ls="--")
ax.set_xlim(-2.0, 1.0); ax.set_ylim(-1.0, 3.0)
ax.set_xlabel(r"$\log([{\rm N\,II}]\lambda6584/{\rm H}\alpha)$")
ax.set_ylabel(r"$\log\,W({\rm H}\alpha)$ [$\rm \AA$]")
save = os.path.join(fig_dir, "whan_diagram.png")
plt.savefig(save, dpi=200, bbox_inches="tight"); plt.show()
print(f"[DONE] {save}")