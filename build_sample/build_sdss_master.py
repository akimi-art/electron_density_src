# =====================================================================
# build_master.py
#   MPA/JHU（DR7）と DR17 測光のクロスマッチ、有効半径（r, u）、データの整理だけを行う
#   - 番兵値、値の妥当性、重複のフラグを付ける（行は消さない）
#   - BPT・光度カットなどのサンプル選択はここではしない（build_sample_v2.py で行う）
# =====================================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


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
RE_DEF     = "circ"         # "circ"（円形化）か "maj"（長軸）
DZ_TOL     = 0.001          # 同じ銀河とみなす z の差
DUP_SEP    = 1.0            # 同じ銀河とみなす距離 [arcsec]
SENTINELS  = (-9999., -999., -99.)
N_EXPECTED = 927552

# 測定の失敗を除くための緩い範囲（科学的なカットではない）
LOGM_RANGE    = (6.0, 13.0)
LOGSFR_RANGE  = (-10.0, 3.0)
RE_ARCSEC_MIN = 0.1         # これより小さい有効半径は測光モデルの失敗とみなす
RE_KPC_MAX    = 50.0

current_dir = os.getcwd()
mpa_path  = os.path.join(current_dir, "results/fits/mpajhu_dr7_v5_2_merged_lgm2b.fits")
dr17_path = os.path.join(current_dir, "data/data_SDSS/DR17/sdss_legacy_spec_radius.csv")
fig_dir   = os.path.join(current_dir, "results/figure/sample")
out_dir   = os.path.join(current_dir, "results/fits")
os.makedirs(fig_dir, exist_ok=True)
os.makedirs(out_dir, exist_ok=True)

def col(t, name):
    """列を float の配列で返す（マスクは NaN に）"""
    return np.ma.filled(np.ma.asarray(t[name]).astype(float), np.nan)

def key3(plate, mjd, fiber):
    return ((np.asarray(plate, np.int64) * 100000 + np.asarray(mjd, np.int64)) * 1000
            + np.asarray(fiber, np.int64))

def in_range(x, lo, hi):
    return np.isfinite(x) & (x > lo) & (x < hi)

# =====================================
# 1. 読み込みと確認
# =====================================
t = Table.read(mpa_path, format="fits")
N_ALL = len(t)
row = np.arange(N_ALL)
assert N_ALL == N_EXPECTED, f"行数が想定と違います: {N_ALL}"
assert np.array_equal(np.asarray(t["ROW_ID"]), row), "ROW_ID が連番ではありません"

OLD_COLS = ["COORD_VALID", "RADIUS_SEP_ARCSEC", "RADIUS_MATCHED", "deVRad_r", "expRad_r",
            "fracDeV_r", "deVAB_r", "expAB_r", "petroR50_r", "petroR90_r", "Re_arcsec"]
removed = [c for c in OLD_COLS if c in t.colnames]
t.remove_columns(removed)
print("[INFO] 旧半径列を削除:", removed)

# =====================================
# 2. 番兵値（-9999 など）を NaN にする
# =====================================
print("\n===== 番兵値の確認 =====")
for c in ["RA", "DEC", "Z", "sm_MEDIAN", "sfr_MEDIAN", "SN_MEDIAN"]:
    v = col(t, c)
    bad = np.isin(v, SENTINELS) | ~np.isfinite(v)
    print(f"  {c:11s}: 番兵/非有限 {bad.sum():7,}   範囲 {np.nanmin(v[~bad]):.4g} -- {np.nanmax(v[~bad]):.4g}")
    v[bad] = np.nan
    t.replace_column(c, v)

ra, dec, z = col(t, "RA"), col(t, "DEC"), col(t, "Z")
coord_ok = (np.isfinite(ra) & np.isfinite(dec) & np.isfinite(z) &
            (ra >= 0) & (ra <= 360) & (np.abs(dec) <= 90) & (z > 0))
print(f"  座標・z が不正: {np.sum(~coord_ok):,}"
      f"（うち z <= 0: {np.sum(np.isfinite(z) & (z <= 0)):,}）")

# =====================================
# 3. DR17 と plate / mjd / fiber で結合
# =====================================
d = pd.read_csv(
    dr17_path, na_values=["null", "NULL", ""], keep_default_na=True,
    dtype={"specObjID": "UInt64", "bestObjID": "Int64",
           "plate": "int64", "mjd": "int64", "fiberID": "int64",
           "run2d": "string", "survey": "string"},
)
print("\n===== DR17 ファイル =====")
print(f"  行数: {len(d):,}")
print("  欠損の数:", d.isna().sum()[d.isna().sum() > 0].to_dict())

kd = key3(d["plate"].to_numpy(), d["mjd"].to_numpy(), d["fiberID"].to_numpy())
assert len(np.unique(kd)) == len(kd), "DR17 側で plate/mjd/fiber が重複"
km = key3(t["PLATEID"], t["MJD"], t["FIBERID"])
srt = np.argsort(kd)
pos = np.clip(np.searchsorted(kd[srt], km), 0, len(kd) - 1)
matched = kd[srt][pos] == km
jd = np.where(matched, srt[pos], -1)

def take(name, fill):
    s = d[name]
    if isinstance(fill, float):
        src = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
        out = np.full(N_ALL, np.nan)
    else:
        src = s.fillna(fill).to_numpy(dtype=np.int64)
        out = np.full(N_ALL, fill, dtype=np.int64)
    out[matched] = src[jd[matched]]
    return out

print(f"  一致: {matched.sum():,} / {N_ALL:,}（不一致 {np.sum(~matched):,}）")
um = ~matched
pm = pd.Series(list(zip(np.asarray(t["PLATEID"])[um], np.asarray(t["MJD"])[um]))).value_counts()
print(f"  不一致のプレート（plate, mjd）: {len(pm)} 枚")
print(pm.head(10).to_string())

bestobjid = take("bestObjID", 0)
z_dr17    = take("spec_z", np.nan)
ra_dr17   = take("spec_ra", np.nan)
dec_dr17  = take("spec_dec", np.nan)
rad = {}
for band in ["r", "u"]:
    for c in ["deVRad", "expRad", "fracDeV", "deVAB", "expAB", "petroR50", "petroR90"]:
        name = f"{c}_{band}"
        v = take(name, np.nan)
        v[v < 0] = np.nan
        rad[name] = v

print("  一致した行の中で半径が無い数（r / u）:")
for c in ["deVRad", "expRad", "fracDeV", "deVAB", "expAB"]:
    print(f"    {c:8s}: r {np.sum(matched & ~np.isfinite(rad[c + '_r'])):>9,}   "
          f"u {np.sum(matched & ~np.isfinite(rad[c + '_u'])):>9,}")

ok = matched & coord_ok & np.isfinite(z_dr17)
dz = np.abs(z - z_dr17)
sep = np.full(N_ALL, np.nan)
sep[ok] = SkyCoord(ra[ok] * u.deg, dec[ok] * u.deg).separation(
          SkyCoord(ra_dr17[ok] * u.deg, dec_dr17[ok] * u.deg)).arcsec
print(f"  |Δz|   中央値 {np.median(dz[ok]):.1e}  99% {np.percentile(dz[ok], 99):.1e}  "
      f"> {DZ_TOL}: {np.sum(dz[ok] > DZ_TOL):,}")
print(f"  位置ずれ 中央値 {np.nanmedian(sep):.3f}″  99% {np.nanpercentile(sep, 99):.3f}″")
far = np.where(sep > 1.0)[0]
print(f"  位置ずれ > 1″: {len(far)} 行")
for r in far[:10]:
    print(f"    {t['PLATEID'][r]} {t['MJD'][r]} {t['FIBERID'][r]}  "
          f"MPA ({ra[r]:.4f}, {dec[r]:.4f})  DR17 ({ra_dr17[r]:.4f}, {dec_dr17[r]:.4f})  z={z[r]:.4f}")

fix = matched & (sep > 1.0)
ra[fix], dec[fix] = ra_dr17[fix], dec_dr17[fix]
print(f"  座標を DR17 で置き換え: {fix.sum()} 行")

# =====================================
# 4. 有効半径（deVRad/expRad は既に有効半径。1.678 は不要）
#    r と u を同じ定義で計算する
# =====================================
kpc_per_arcsec = np.full(N_ALL, np.nan)
kpc_per_arcsec[coord_ok] = (cosmo.angular_diameter_distance(z[coord_ok]).to(u.kpc).value
                            * np.pi / 180 / 3600)

def effective_radius(band):
    f = rad[f"fracDeV_{band}"]
    r_maj  = f * rad[f"deVRad_{band}"] + (1 - f) * rad[f"expRad_{band}"]
    r_circ = r_maj * np.sqrt(f * rad[f"deVAB_{band}"] + (1 - f) * rad[f"expAB_{band}"])
    r_arc  = r_circ if RE_DEF == "circ" else r_maj
    return r_maj, r_circ, r_arc, r_arc * kpc_per_arcsec

Re_maj,   Re_circ,   Re_arcsec,   Re_kpc   = effective_radius("r")
Re_maj_u, Re_circ_u, Re_arcsec_u, Re_kpc_u = effective_radius("u")

re_ok   = matched & (bestobjid != 0) & np.isfinite(Re_kpc)   & (Re_kpc > 0)
re_ok_u = matched & (bestobjid != 0) & np.isfinite(Re_kpc_u) & (Re_kpc_u > 0)

# =====================================
# 5. 値の妥当性（測定の失敗を除く。科学的なカットではない）
# =====================================
logM   = col(t, "sm_MEDIAN")
logSFR = col(t, "sfr_MEDIAN")
mass_ok = in_range(logM, *LOGM_RANGE) & (logM != -1.0)
sfr_ok  = in_range(logSFR, *LOGSFR_RANGE) & (logSFR != -1.0)
re_sane   = re_ok   & (Re_arcsec   > RE_ARCSEC_MIN) & (Re_kpc   < RE_KPC_MAX)
re_sane_u = re_ok_u & (Re_arcsec_u > RE_ARCSEC_MIN) & (Re_kpc_u < RE_KPC_MAX)

print("\n===== 値の妥当性（座標・z が有効な行の中で）=====")
for name, m in [("log M* が範囲外", ~mass_ok), ("log SFR が範囲外", ~sfr_ok),
                ("Re が無い（r）", ~re_ok),
                (f"Re < {RE_ARCSEC_MIN}″ または > {RE_KPC_MAX} kpc（r）", re_ok & ~re_sane),
                ("Re が無い（u）", ~re_ok_u),
                (f"Re < {RE_ARCSEC_MIN}″ または > {RE_KPC_MAX} kpc（u）", re_ok_u & ~re_sane_u)]:
    print(f"  {name:34s}: {np.sum(coord_ok & m):7,}")

# =====================================
# 6. 重複（同じ銀河の別スペクトル）のフラグ
#    同じ bestObjID、または 1″ 以内かつ |Δz| < 0.001 → 同じ銀河（両方の和集合）
#    代表：値が妥当なものを優先し、その中で SN_MEDIAN が最大（nₑ に依存しない）
# =====================================
idv = np.where(coord_ok & (bestobjid != 0))[0]
_, first_i, inv_id = np.unique(bestobjid[idv], return_index=True, return_inverse=True)
eA_i, eA_j = idv, idv[first_i][inv_id]

idx = np.where(coord_ok)[0]
cc = SkyCoord(ra[idx] * u.deg, dec[idx] * u.deg)
i, j, _, _ = cc.search_around_sky(cc, DUP_SEP * u.arcsec)
k = (i < j) & (np.abs(z[idx][i] - z[idx][j]) < DZ_TOL)
eB_i, eB_j = idx[i[k]], idx[j[k]]
same_id = (bestobjid[eB_i] == bestobjid[eB_j]) & (bestobjid[eB_i] != 0)

G = coo_matrix((np.ones(len(eA_i) + len(eB_i)),
                (np.r_[eA_i, eB_i], np.r_[eA_j, eB_j])), shape=(N_ALL, N_ALL))
_, gal = connected_components(G, directed=False)
cnt = np.bincount(gal)
n_spec = cnt[gal]
multi = n_spec > 1

good = coord_ok & mass_ok & sfr_ok & re_sane_u
rank = col(t, "SN_MEDIAN")
rank = np.where(np.isfinite(rank), rank, -np.inf)
order = np.lexsort((row, -rank, ~good, gal))
first = np.r_[True, gal[order][1:] != gal[order][:-1]]
is_primary = np.zeros(N_ALL, bool)
is_primary[order[first]] = True
is_primary &= coord_ok

sizes = cnt[cnt > 1]
zspread = pd.Series(z[multi]).groupby(gal[multi]).agg(np.ptp)
print("\n===== 重複 =====")
print(f"  位置で見つけたペア: {k.sum():,}（同じ bestObjID {same_id.mean():.4f}、"
      f"違う bestObjID {np.sum(~same_id):,} → 位置でまとめた）")
print(f"  複数スペクトルを持つ銀河: {len(sizes):,}（関係する行 {multi.sum():,}）")
print(f"  1 銀河あたりのスペクトル数: {dict(zip(*np.unique(sizes, return_counts=True)))}")
print(f"  同じ銀河の中で z の差 > {DZ_TOL}: {np.sum(zspread > DZ_TOL):,} 銀河")
print(f"  重複のない有効な銀河（IS_PRIMARY）: {is_primary.sum():,}")

# =====================================
# 7. 列を追加して保存（選択の列は入れない）
# =====================================
new_cols = {
    "COORD_Z_VALID": coord_ok, "SPEC_MATCHED": matched, "BESTOBJID": bestobjid,
    "Z_DR17": z_dr17, "RA_DR17": ra_dr17, "DEC_DR17": dec_dr17, "SEP_DR17_ARCSEC": sep,
    **rad,
    "Re_maj_arcsec": Re_maj, "Re_circ_arcsec": Re_circ, "Re_kpc": Re_kpc,
    "RE_VALID": re_ok, "RE_SANE": re_sane,
    "Re_maj_arcsec_u": Re_maj_u, "Re_circ_arcsec_u": Re_circ_u, "Re_kpc_u": Re_kpc_u,
    "RE_VALID_u": re_ok_u, "RE_SANE_u": re_sane_u,
    "MASS_OK": mass_ok, "SFR_OK": sfr_ok,
    "GALAXY_ID": gal, "N_SPEC": n_spec, "IS_PRIMARY": is_primary,
}
for name, v in new_cols.items():
    t[name] = v
assert len(t) == N_ALL and np.array_equal(np.asarray(t["ROW_ID"]), row)

master_path = os.path.join(out_dir, "mpajhu_dr7_v5_2_master.fits")
t.write(master_path, format="fits", overwrite=True)
print(f"\n[DONE] {master_path}（全 {N_ALL:,} 行）")

# =====================================
# 8. 図：半径の確認
#    対象：重複を除き、M* が妥当で、r の Re が妥当な銀河（z の範囲は限らない）
# =====================================
def finish(ax, path):
    for a in np.atleast_1d(ax).ravel():
        for spine in a.spines.values():
            spine.set_linewidth(2)
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"[DONE] {path}")

valid_r = is_primary & mass_ok & re_sane

# 8a. Re（r）– M*
fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(logM[valid_r], np.log10(Re_kpc[valid_r]), s=0.5, alpha=0.05, color="gray", rasterized=True)
ax.set_xlim(7, 12); ax.set_ylim(-1, 1.5)
ax.set_xlabel(r"$\log(M_\ast/M_\odot)$")
ax.set_ylabel(r"$\log(R_{{\rm e},r}/{\rm kpc})$")
finish(ax, os.path.join(fig_dir, "size_mass_r.png"))

# 8b. r と u の Re の比較
#   (a) u の Re と r の Re（1 対 1 の線）
#   (b) Re の比 log(Re_u/Re_r) と M*
#   (c) Re の比と z
#   線：各ビンの中央値、帯：16–84 パーセンタイル
cmp = valid_r & re_sane_u
log_re_r = np.log10(Re_kpc[cmp])
log_re_u = np.log10(Re_kpc_u[cmp])
dlog = log_re_u - log_re_r

p16, p50, p84 = np.percentile(dlog, [16, 50, 84])
print("\n===== r と u の Re の比較 =====")
print(f"  対象: {cmp.sum():,} 銀河（r の Re が妥当な {valid_r.sum():,} のうち u の Re も妥当）")
print(f"  log(Re_u / Re_r): 中央値 {p50:+.3f}   16–84%: {p16:+.3f} -- {p84:+.3f}")
print(f"  → ΣSFR に換算すると中央値で {-2 * p50:+.3f} dex（ΣSFR ∝ Re^-2）")

def running_median(x, y, bins, nmin=30):
    c, m, lo, hi = [], [], [], []
    for a, b in zip(bins[:-1], bins[1:]):
        s_ = (x >= a) & (x < b)
        if s_.sum() < nmin:
            continue
        c.append(0.5 * (a + b))
        q16, q50, q84 = np.percentile(y[s_], [16, 50, 84])
        m.append(q50); lo.append(q16); hi.append(q84)
    return np.array(c), np.array(m), np.array(lo), np.array(hi)

fig, axes = plt.subplots(1, 3, figsize=(30, 9))

ax = axes[0]
ax.hexbin(log_re_r, log_re_u, gridsize=120, bins="log", cmap="Greys",
          extent=(-1, 1.5, -1, 1.5), mincnt=1)
ax.plot([-1, 1.5], [-1, 1.5], color="firebrick", lw=2)
ax.set_xlim(-1, 1.5); ax.set_ylim(-1, 1.5)
ax.set_xlabel(r"$\log(R_{{\rm e},r}/{\rm kpc})$")
ax.set_ylabel(r"$\log(R_{{\rm e},u}/{\rm kpc})$")

for ax, xv, bins, xlabel, ext in [
        (axes[1], logM[cmp], np.arange(8.0, 12.01, 0.2), r"$\log(M_\ast/M_\odot)$", (8, 12)),
        (axes[2], z[cmp],    np.arange(0.0, 0.41, 0.01),  r"$z$",                    (0, 0.4))]:
    ax.hexbin(xv, dlog, gridsize=100, bins="log", cmap="Greys",
              extent=(ext[0], ext[1], -1, 1), mincnt=1)
    c, m, lo, hi = running_median(xv, dlog, bins)
    ax.plot(c, m, color="firebrick", lw=3)
    ax.fill_between(c, lo, hi, color="firebrick", alpha=0.2)
    ax.axhline(0, color="k", ls="--", lw=2)
    ax.set_xlim(*ext); ax.set_ylim(-1, 1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$\log(R_{{\rm e},u}/R_{{\rm e},r})$")

plt.tight_layout()
finish(axes, os.path.join(fig_dir, "re_u_vs_r.png"))