# =====================================================================
# build_sample.py
#   サンプル構築（2026-10 版の方針）
#   1. BPT で明らかな AGN を除く（S/N カットなし）
#   2. 光度カット（体積限定）
#   3. 完全性：光度カットの影響だけを評価する
#      - 対象は M*, SFR, sSFR, ΣSFR。それぞれ独立に評価し、
#        M* の完全な範囲を他の量の評価には課さない
#      - 主系列に限らず、1 を通った銀河全体を分母にする
#   欠損値・重複・半径の扱いは build_master.py の master をそのまま使う
#   全行を残したファイル（フラグ付き）と、最終サンプルの両方を保存する
# =====================================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from scipy.optimize import brentq

# 軸の設定
plt.rcParams.update({
    "figure.figsize": (12, 6),
    "font.size": 32,
    "axes.labelsize": 32,
    "axes.titlesize": 32,
    "axes.grid": False,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.size": 32,
    "ytick.major.size": 32,
    "xtick.major.width": 2,
    "ytick.major.width": 2,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.minor.size": 8,
    "ytick.minor.size": 8,
    "xtick.minor.width": 1.5,
    "ytick.minor.width": 1.5,
    "xtick.labelsize": 28,
    "ytick.labelsize": 28,
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
})

# =====================================
# 0. 設定
# =====================================
cosmo      = FlatLambdaCDM(H0=70, Om0=0.3)
UNIT_FLUX  = 1e-17          # MPA/JHU のフラックスの単位 [erg s^-1 cm^-2]
FLUX_LIMIT = 1e-17          # [SII]6731 のフラックスの限界 [erg s^-1 cm^-2]
L_MIN      = 1e39           # [SII]6731 の光度の下限 [erg s^-1]
L6731_MAX  = 1e43           # これより明るいのはフラックスの失敗（欠損値の扱い）
METHOD     = "Ka03"         # AGN の分類線："Ka03" または "Ke01"
REQUIRE_RE = True           # ΣSFR のため、有効半径がある銀河に限る
RE_BAND    = "u"            # ΣSFR に使う有効半径のバンド："u" または "r"

# 値の妥当性（build_local_sample.py と同じ。-1 は「値なし」）
LOGM_RANGE   = (6.0, 13.0)
LOGSFR_RANGE = (-10.0, 3.0)

# 完全性
COMPL_THRESH = 0.9          # この割合以上を「完全」とする
NMIN_BIN     = 30           # 分母がこれ未満のビンは判定しない

# BPT の分類線：y = a / (x - b) + c、x >= b は常に分類線の外側
BPT_LINES = {
    "Ka03": dict(a=0.61, b=0.05, c=1.30),   # Kauffmann+03
    "Ke01": dict(a=0.61, b=0.47, c=1.19),   # Kewley+01
}

current_dir = os.getcwd()
master_path = os.path.join(current_dir, "results/fits/mpajhu_dr7_v5_2_master.fits")
fig_dir     = os.path.join(current_dir, "results/figure/sample_v2")
out_dir     = os.path.join(current_dir, "results/fits")
os.makedirs(fig_dir, exist_ok=True)
os.makedirs(out_dir, exist_ok=True)

def col(t, name):
    """列を float の配列で返す（マスクは NaN に）"""
    return np.ma.filled(np.ma.asarray(t[name]).astype(float), np.nan)

def in_range(x, lo, hi):
    return np.isfinite(x) & (x > lo) & (x < hi)

# =====================================
# 1. 読み込みと確認
# =====================================
t = Table.read(master_path, format="fits")
N_ALL = len(t)
assert N_ALL == 927552, f"行数が想定と違います: {N_ALL}"
assert np.array_equal(np.asarray(t["ROW_ID"]), np.arange(N_ALL)), "ROW_ID が連番ではありません"

LINES = ["SII_6731", "H_BETA", "OIII_5007", "H_ALPHA", "NII_6584"]
sfx = "_u" if RE_BAND == "u" else ""
need = (["Z", "sm_MEDIAN", "sfr_MEDIAN", "COORD_Z_VALID", "IS_PRIMARY",
         f"Re_kpc{sfx}", f"RE_SANE{sfx}"]
        + [f"{l}_FLUX" for l in LINES])
missing = [c for c in need if c not in t.colnames]
assert not missing, f"master に列がありません: {missing}"

# =====================================
# 2. 欠損値の扱い（master と同じ）
# =====================================
z          = col(t, "Z")
logM       = col(t, "sm_MEDIAN")
logSFR     = col(t, "sfr_MEDIAN")
Re_kpc     = col(t, f"Re_kpc{sfx}")
coord_ok   = np.asarray(t["COORD_Z_VALID"], bool)
is_primary = np.asarray(t["IS_PRIMARY"], bool)
re_sane    = np.asarray(t[f"RE_SANE{sfx}"], bool)
mass_ok    = np.asarray(t["MASS_OK"], bool)
sfr_ok     = np.asarray(t["SFR_OK"], bool)

# =====================================
# 3. Z_MAX：光度の下限とフラックスの限界の交点
# =====================================
def z_max_from_flux_limit(l_min=L_MIN, f_lim=FLUX_LIMIT):
    def f(zz):
        return 4 * np.pi * cosmo.luminosity_distance(zz).to(u.cm).value**2 * f_lim - l_min
    return brentq(f, 1e-4, 1.0)

Z_MAX = z_max_from_flux_limit()
print(f"[INFO] Z_MAX = {Z_MAX:.4f}")

# =====================================
# 4. 必要な量
# =====================================
F = {l: col(t, f"{l}_FLUX") * UNIT_FLUX for l in LINES}

dL = np.full(N_ALL, np.nan)
dL[coord_ok] = cosmo.luminosity_distance(z[coord_ok]).to(u.cm).value
with np.errstate(invalid="ignore", divide="ignore"):
    L6731       = 4 * np.pi * dL**2 * F["SII_6731"]
    log_N2Ha    = np.log10(F["NII_6584"] / F["H_ALPHA"])
    log_O3Hb    = np.log10(F["OIII_5007"] / F["H_BETA"])
    logsSFR     = logSFR - logM
    logSigmaSFR = logSFR - np.log10(2 * np.pi * Re_kpc**2)   # ΣSFR = SFR / (2π Re²)

# =====================================
# 5. 明らかな AGN の判定（S/N カットなし）
#    0 = BPT で星形成側
#    1 = BPT で AGN 側（除く）
#    3 = 判定できない（明らかな AGN ではないので残す）
# =====================================
pos4 = ((F["NII_6584"] > 0) & (F["H_ALPHA"] > 0) & (F["OIII_5007"] > 0) & (F["H_BETA"] > 0)
        & np.isfinite(log_N2Ha) & np.isfinite(log_O3Hb))

# BPT の分類線の式を使って、AGN 側かどうかを判定する関数
def classify(method):
    p = BPT_LINES[method]
    with np.errstate(invalid="ignore", divide="ignore"):
        above = (log_N2Ha >= p["b"]) | (log_O3Hb >= p["a"] / (log_N2Ha - p["b"]) + p["c"])
    cls = np.full(N_ALL, 3, np.int8)
    cls[pos4 & ~above] = 0
    cls[pos4 & above]  = 1
    return cls

bpt_class = classify(METHOD)
not_agn   = (bpt_class == 0) | (bpt_class == 3)

# =====================================
# 6. 選択の条件
# =====================================
in_z = coord_ok & (z < Z_MAX)
base = is_primary & in_z & mass_ok & sfr_ok
if REQUIRE_RE:
    base &= re_sane

flux_bad = np.isfinite(L6731) & (L6731 >= L6731_MAX)
parent   = base & not_agn & ~flux_bad
lum      = np.isfinite(L6731) & (L6731 > L_MIN)
selected = parent & lum                                     # 最終サンプル = 完全性の分子

# =====================================
# 7. 選択の流れ（2 章の表に使う）
# =====================================
flow = [
    ("SDSS DR7 MPA/JHU（全スペクトル）",    np.ones(N_ALL, bool)),
    ("座標・z が有効",                      coord_ok),
    ("+ 重複を除く（1 銀河 1 スペクトル）",  coord_ok & is_primary),
    (f"+ z < {Z_MAX:.4f}",                  is_primary & in_z),
    ("+ M*, SFR が妥当",                    is_primary & in_z & mass_ok & sfr_ok),
]
if REQUIRE_RE:
    flow.append((f"+ 有効半径が妥当（{RE_BAND}）", base))
flow += [
    (f"+ 明らかな AGN を除く（{METHOD}）",          base & not_agn),
    (f"+ フラックスの失敗を除く（L ≥ {L6731_MAX:.0e}）", parent),
    (f"+ L([SII]6731) > {L_MIN:.0e}",               selected),
]
print("\n===== Selection flow =====")
rows = []
for name, m in flow:
    print(f"  {name:34s}: {m.sum():>9,}")
    rows.append({"step": name, "N": int(m.sum())})
pd.DataFrame(rows).to_csv(os.path.join(fig_dir, f"selection_flow_v2_{METHOD}.csv"), index=False)

print(f"\n===== AGN 判定の内訳（base {base.sum():,} の中）=====")
labels = {0: "BPT で星形成側（残す）", 1: "BPT で AGN 側（除く）",
          3: "判定できない（残す）"}
for k, lab in labels.items():
    n_base = np.sum(base & (bpt_class == k))
    n_sel  = np.sum(selected & (bpt_class == k))
    print(f"  {lab:28s}: base {n_base:>9,}   最終サンプル {n_sel:>9,}")

other = "Ke01" if METHOD == "Ka03" else "Ka03"
cls_o = classify(other)
sel_o = base & ((cls_o == 0) | (cls_o == 3)) & ~flux_bad & lum
print(f"\n  （参考）{other} の場合の最終サンプル: {sel_o.sum():,}")

# =====================================
# 8. 完全性（光度カットの影響のみ）
#    分母 = parent、分子 = parent かつ光度カットを通過
#    各量を独立に評価（他の量の範囲は課さない）
# =====================================
QUANT = {
    "logM":        (logM,        np.arange(7.0, 12.01, 0.1),  r"$\log(M_\ast/M_\odot)$"),
    "logSFR":      (logSFR,      np.arange(-3.0, 2.51, 0.1),  r"$\log({\rm SFR}/M_\odot\,{\rm yr}^{-1})$"),
    "logsSFR":     (logsSFR,     np.arange(-13.0, -7.99, 0.1), r"$\log({\rm sSFR}/{\rm yr}^{-1})$"),
    "logSigmaSFR": (logSigmaSFR, np.arange(-5.0, 1.01, 0.1),  r"$\log(\Sigma_{\rm SFR}/M_\odot\,{\rm yr}^{-1}\,{\rm kpc}^{-2})$"),
}

def completeness(xv, den, num, bins):
    nb = len(bins) - 1
    idx = np.digitize(xv, bins) - 1
    ok = np.isfinite(xv) & (idx >= 0) & (idx < nb)
    n_den = np.bincount(idx[ok & den], minlength=nb)
    n_num = np.bincount(idx[ok & num], minlength=nb)
    with np.errstate(invalid="ignore", divide="ignore"):
        f = n_num / n_den
        err = np.sqrt(f * (1 - f) / n_den)
    return 0.5 * (bins[1:] + bins[:-1]), f, err, n_den, n_num

def complete_interval(bins, f, n_den, thr=COMPL_THRESH, nmin=NMIN_BIN):
    """f が最大のビンを含み、f >= thr が続く区間。端がデータの端かどうかも返す"""
    valid = n_den >= nmin
    if not valid.any():
        return None
    i0 = np.nanargmax(np.where(valid, f, -1))
    good = valid & (f >= thr)
    if not good[i0]:
        return None
    lo, hi = i0, i0
    while lo - 1 >= 0 and good[lo - 1]:
        lo -= 1
    while hi + 1 < len(good) and good[hi + 1]:
        hi += 1
    lo_data = np.argmax(valid)
    hi_data = len(valid) - 1 - np.argmax(valid[::-1])
    return bins[lo], bins[hi + 1], lo == lo_data, hi == hi_data

print(f"\n===== 完全性（光度カットのみ、閾値 {COMPL_THRESH}）=====")
curves, summary = {}, []
for key, (xv, bins, label) in QUANT.items():
    c, f, err, n_den, n_num = completeness(xv, parent, selected, bins)
    curves[key] = (c, f, err, n_den, n_num)
    iv = complete_interval(bins, f, n_den)
    if iv is None:
        print(f"  {key:12s}: f >= {COMPL_THRESH} の区間なし（最大 f = {np.nanmax(f[n_den >= NMIN_BIN]):.2f}）")
        summary.append({"quantity": key, "lo": np.nan, "hi": np.nan,
                        "lo_is_data_edge": False, "hi_is_data_edge": False, "N_selected_in": 0})
        continue
    lo, hi, lo_edge, hi_edge = iv
    n_in = np.sum(selected & (xv >= lo) & (xv < hi))
    note = []
    if lo_edge: note.append("下端はデータの端")
    if hi_edge: note.append("上端はデータの端")
    print(f"  {key:12s}: {lo:6.2f} <= x < {hi:6.2f}   最終サンプル中 N = {n_in:,}  {' / '.join(note)}")
    summary.append({"quantity": key, "lo": lo, "hi": hi,
                    "lo_is_data_edge": lo_edge, "hi_is_data_edge": hi_edge, "N_selected_in": int(n_in)})
    pd.DataFrame({"x_center": c, "f": f, "f_err": err, "N_den": n_den, "N_num": n_num}).to_csv(
        os.path.join(fig_dir, f"completeness_{key}_{METHOD}.csv"), index=False)
pd.DataFrame(summary).to_csv(os.path.join(fig_dir, f"completeness_summary_v2_{METHOD}.csv"), index=False)

# =====================================
# 9. 最終サンプルの性質（範囲と 1%–99%）
# =====================================
def rng(x):
    x = x[np.isfinite(x)]
    p1, p99 = np.percentile(x, [1, 99])
    return f"{x.min():8.4f} -- {x.max():8.4f}   (1–99%: {p1:7.3f} -- {p99:7.3f})"

s = selected
print("\n===== Sample statistics =====")
print(f"  N(selected)  = {s.sum():,}")
print(f"  z            = {rng(z[s])}")
print(f"  log M*       = {rng(logM[s])}")
print(f"  log SFR      = {rng(logSFR[s])}")
print(f"  log sSFR     = {rng(logsSFR[s])}")
print(f"  log ΣSFR     = {rng(logSigmaSFR[s])}")
print(f"  log L(6731)  = {rng(np.log10(L6731[s]))}")

# =====================================
# 10. 保存（全行 + フラグ、最終サンプル）
# =====================================
t[f"BPT_CLASS_{METHOD}"] = bpt_class
t["NOT_AGN_V2"]  = not_agn
t["BASE_V2"]     = base
t["PARENT_V2"]   = parent
t["LUM_OK_V2"]   = lum
t["SELECTED_V2"] = selected
assert len(t) == N_ALL and np.array_equal(np.asarray(t["ROW_ID"]), np.arange(N_ALL))

flags_path = os.path.join(out_dir, f"mpajhu_dr7_v5_2_master_v2_{METHOD}_Re{RE_BAND}.fits")
t.write(flags_path, format="fits", overwrite=True)
print(f"\n[DONE] {flags_path}（全 {N_ALL:,} 行）")

sample_path = os.path.join(out_dir, f"sdss_sample_v2_zlt{Z_MAX:.4f}_Lgt{L_MIN:.0e}_{METHOD}_Re{RE_BAND}.fits")
t[selected].write(sample_path, format="fits", overwrite=True)
print(f"[DONE] {sample_path}（{selected.sum():,} 行）")

# =====================================
# 11. 図
# =====================================
def finish(fig, axes, path):
    for ax in np.atleast_1d(axes).ravel():
        for spine in ax.spines.values():
            spine.set_linewidth(2)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"[DONE] {path}")

# 11a. 体積限定（灰：AGN を除いた銀河、赤：光度カットを通過）
fig, ax = plt.subplots(figsize=(12, 6))
ok = is_primary & coord_ok & not_agn & np.isfinite(L6731) & (L6731 > 0)
ax.scatter(z[ok], L6731[ok], s=0.5, alpha=1, color="gray", rasterized=True)
ax.scatter(z[selected], L6731[selected], s=0.5, alpha=1, color="firebrick", rasterized=True)
zg = np.linspace(1e-4, 0.4, 400)
ax.plot(zg, 4 * np.pi * cosmo.luminosity_distance(zg).to(u.cm).value**2 * FLUX_LIMIT, color="k", lw=2)
ax.axvline(Z_MAX, color="k", lw=2)
ax.axhline(L_MIN, color="k", lw=2)
ax.set_yscale("log")
ax.set_xlim(0, 0.4); ax.set_ylim(1e36, 1e42)
ax.set_xlabel(r"$z$")
ax.set_ylabel(r"$L([{\rm S\,II}]\lambda6731)$ [erg s$^{-1}$]")
finish(fig, ax, os.path.join(fig_dir, f"sii6731_luminosity_vs_z_{METHOD}.png"))

# 11b. BPT 図（base のうち 4 本とも正のフラックス。青：残す、赤：除く）
m0 = base & (bpt_class == 0)
m1 = base & (bpt_class == 1)
fig, ax = plt.subplots(figsize=(10, 9))
ax.scatter(log_N2Ha[m0], log_O3Hb[m0], s=0.5, alpha=0.1, color="firebrick", rasterized=True)
ax.scatter(log_N2Ha[m1], log_O3Hb[m1], s=0.5, alpha=0.1, color="gray", rasterized=True)
xx = np.linspace(-2.0, 0.04, 500)
ax.plot(xx, 0.61 / (xx - 0.05) + 1.3, color="k", lw=2, ls="-" if METHOD == "Ka03" else "--",
        label="Kauffmann+03")
# xx = np.linspace(-2.0, 0.46, 500)
# ax.plot(xx, 0.61 / (xx - 0.47) + 1.19, color="k", lw=2, ls="-" if METHOD == "Ke01" else "--",
#         label="Kewley+01")
ax.set_xlim(-2.0, 0.5); ax.set_ylim(-1.5, 1.5)
ax.set_xlabel(r"$\log([{\rm N\,II}]\lambda6584/{\rm H}\alpha)$")
ax.set_ylabel(r"$\log([{\rm O\,III}]\lambda5007/{\rm H}\beta)$")
finish(fig, ax, os.path.join(fig_dir, f"bpt_diagram_{METHOD}.png"))

# 11c. 完全性の曲線（2×2）
fig, axes = plt.subplots(2, 2, figsize=(24, 16))
for ax, (key, (xv, bins, label)) in zip(axes.ravel(), QUANT.items()):
    c, f, err, n_den, n_num = curves[key]
    v = n_den >= NMIN_BIN
    ax.errorbar(c[v], f[v], yerr=err[v], fmt="o-", color="k", ms=5, lw=1.5)
    # ax.axhline(COMPL_THRESH, color="firebrick", ls="--", lw=2)
    # row_s = [r for r in summary if r["quantity"] == key][0]
    # if np.isfinite(row_s["lo"]):
    #     ax.axvspan(row_s["lo"], row_s["hi"], color="tab:blue", alpha=0.12)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel(label); ax.set_ylabel(r"$N(L_{6731} > L_{\rm min})\,/\,N$")
plt.tight_layout()
finish(fig, axes, os.path.join(fig_dir, f"completeness_luminosity_cut_{METHOD}.png"))

# 11d. 分母と分子の分布（2×2）
fig, axes = plt.subplots(2, 2, figsize=(24, 16))
for ax, (key, (xv, bins, label)) in zip(axes.ravel(), QUANT.items()):
    ax.hist(xv[parent],   bins=bins, histtype="step", color="gray", lw=2.5, label="AGN removed")
    ax.hist(xv[selected], bins=bins, histtype="step", color="firebrick", lw=2.5, label="+ luminosity cut")
    ax.set_yscale("log")
    ax.set_xlabel(label); ax.set_ylabel("Number of galaxies")
axes[0, 0].legend(fontsize=22)
plt.tight_layout()
finish(fig, axes, os.path.join(fig_dir, f"completeness_den_num_{METHOD}.png"))

# 11e. 最終サンプルの z 分布
fig, ax = plt.subplots(figsize=(12, 6))
ax.hist(z[selected], bins=50, color="gray", edgecolor="black", alpha=0.8)
ax.set_xlabel(r"$z$"); ax.set_ylabel("Number of galaxies")
ax.set_xlim(0, Z_MAX)
finish(fig, ax, os.path.join(fig_dir, f"selected_redshift_histogram_{METHOD}.png"))

# =====================================
# 12. 診断：M*–SFR 平面で、光度カットがどの銀河を落としているか
#     分母 = parent、通過 = parent & lum（完全性の評価と同じ）
#     主系列の線（Renzini & Peng 2015）は読むための目安。選択には使わない
# =====================================
from matplotlib.colors import LogNorm

MS_SLOPE, MS_ZP = 0.76, -7.64
NMIN_2D = 10                                  # 2 次元のビンで割合を出す最小の数
mb = np.arange(7.0, 12.01, 0.1)
sb = np.arange(-3.0, 2.51, 0.1)

okp  = parent & np.isfinite(logM) & np.isfinite(logSFR)
lost = okp & ~lum                             # 光度カットで落ちた銀河

H_den, _, _ = np.histogram2d(logM[okp],       logSFR[okp],       bins=[mb, sb])
H_num, _, _ = np.histogram2d(logM[okp & lum], logSFR[okp & lum], bins=[mb, sb])
with np.errstate(invalid="ignore", divide="ignore"):
    frac = H_num / H_den
frac_m = np.ma.masked_where(H_den < NMIN_2D, frac)
mc, sc = 0.5 * (mb[1:] + mb[:-1]), 0.5 * (sb[1:] + sb[:-1])
xx_ms = np.linspace(7, 12, 100)

fig, axes = plt.subplots(1, 3, figsize=(33, 10), sharey=True)

# (a) 分母の分布
ax = axes[0]
pc = ax.pcolormesh(mb, sb, np.ma.masked_where(H_den == 0, H_den).T,
                   norm=LogNorm(), cmap="Greys", shading="flat")
fig.colorbar(pc, ax=ax, label="Number of galaxies")
ax.set_title("(a) AGN removed, $z < z_{\\rm max}$", fontsize=28, loc="left")

# (b) 光度カットを通る割合
ax = axes[1]
pc = ax.pcolormesh(mb, sb, frac_m.T, vmin=0, vmax=1, cmap="viridis", shading="flat")
fig.colorbar(pc, ax=ax, label=r"$N(L_{6731} > L_{\rm min})\,/\,N$")
cs = ax.contour(mc, sc, np.ma.filled(frac_m, np.nan).T, levels=[0.5, 0.9],
                colors=["white", "red"], linewidths=2.5)
ax.clabel(cs, fmt="%.1f", fontsize=20)
ax.set_title("(b) Fraction above $L_{\\rm min}$", fontsize=28, loc="left")

# (c) 落ちた銀河：BPT で星形成側（赤）と BPT 図に置けない（青）
ax = axes[2]
m3 = lost & (bpt_class == 3)
m0 = lost & (bpt_class == 0)
ax.scatter(logM[m3], logSFR[m3], s=0.3, alpha=0.05, color="tab:blue", rasterized=True)
ax.scatter(logM[m0], logSFR[m0], s=0.3, alpha=0.05, color="firebrick", rasterized=True)
ax.scatter([], [], s=60, color="tab:blue",  label=f"not on BPT ({m3.sum():,})")
ax.scatter([], [], s=60, color="firebrick", label=f"BPT star-forming ({m0.sum():,})")
ax.legend(loc="upper left", fontsize=20)
ax.set_title("(c) Removed by the luminosity cut", fontsize=28, loc="left")

for ax in axes:
    ax.set_xlim(7, 12); ax.set_ylim(-3, 2.5)
    ax.set_xlabel(r"$\log(M_\ast/M_\odot)$")
axes[0].set_ylabel(r"$\log({\rm SFR}/M_\odot\,{\rm yr}^{-1})$")
plt.tight_layout()
finish(fig, axes, os.path.join(fig_dir, f"mstar_sfr_luminosity_cut_{METHOD}.png"))

# ---- M* のビンごとの表：何が落ちているか ----
print("\n===== M* のビンごと：光度カットで落ちた銀河の性質 =====")
print("  log M*        N(分母)   通過割合   通過の SFR 中央値   落ちた SFR 中央値   落ちたうち BPT 不可の割合")
for lo in np.arange(8.0, 12.0, 0.25):
    b = okp & (logM >= lo) & (logM < lo + 0.25)
    if b.sum() < 30:
        continue
    k_ = b & lum
    l_ = b & ~lum
    med_k = np.median(logSFR[k_]) if k_.any() else np.nan
    med_l = np.median(logSFR[l_]) if l_.any() else np.nan
    f3 = np.mean(bpt_class[l_] == 3) if l_.any() else np.nan
    print(f"  {lo:5.2f}-{lo+0.25:5.2f}  {b.sum():>9,}   {np.mean(lum[b]):6.2f}"
          f"        {med_k:+6.2f}             {med_l:+6.2f}               {f3:5.2f}")