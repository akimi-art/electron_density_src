# =====================================================================
# build_local_sample.py
#   SDSS MPA/JHU から局所銀河サンプルを作る（1 本で完結）
#   読み込み → 番兵値 → DR17 と結合（半径）→ 値の妥当性 → 重複フラグ
#   → Z_MAX → 光度・S/N・BPT → 選択の流れ → 統計量 → 保存 → 図
#   master ファイルには全 927,552 行を残す（カットはフラグ列で記録）
# =====================================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.optimize import brentq
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


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



# =====================================
# 0. 設定
# =====================================
cosmo      = FlatLambdaCDM(H0=70, Om0=0.3)
UNIT_FLUX  = 1e-17          # MPA/JHU のフラックスの単位 [erg s^-1 cm^-2]
FLUX_LIMIT = 1e-17          # [SII]6731 のフラックスの限界 [erg s^-1 cm^-2]
L_MIN      = 1e39           # [SII]6731 の光度の下限 [erg s^-1]
SN_MIN     = 3.0
METHOD     = "Ke01"
REQUIRE_RE = True
RE_DEF     = "circ"         # "circ"（円形化）か "maj"（長軸）
DZ_TOL     = 0.001          # 同じ銀河とみなす z の差
DUP_SEP    = 1.0            # 同じ銀河とみなす距離 [arcsec]
SENTINELS  = (-9999., -999., -99.)
N_EXPECTED = 927552

# 測定の失敗を除くための緩い範囲（科学的なカットではない）
LOGM_RANGE   = (6.0, 13.0)
LOGSFR_RANGE = (-10.0, 3.0)
RE_ARCSEC_MIN = 0.1         # これより小さい有効半径は測光モデルの失敗とみなす
RE_KPC_MAX    = 50.0
L6731_MAX     = 1e43        # [erg s^-1]

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
for c in ["deVRad_r", "expRad_r", "fracDeV_r", "deVAB_r", "expAB_r", "petroR50_r", "petroR90_r"]:
    v = take(c, np.nan)
    v[v < 0] = np.nan
    rad[c] = v

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
# =====================================
f = rad["fracDeV_r"]
Re_maj  = f * rad["deVRad_r"] + (1 - f) * rad["expRad_r"]
Re_circ = Re_maj * np.sqrt(f * rad["deVAB_r"] + (1 - f) * rad["expAB_r"])
Re_arcsec = Re_circ if RE_DEF == "circ" else Re_maj
kpc_per_arcsec = np.full(N_ALL, np.nan)
kpc_per_arcsec[coord_ok] = (cosmo.angular_diameter_distance(z[coord_ok]).to(u.kpc).value
                            * np.pi / 180 / 3600)
Re_kpc = Re_arcsec * kpc_per_arcsec
re_ok = matched & (bestobjid != 0) & np.isfinite(Re_kpc) & (Re_kpc > 0)

# =====================================
# 5. 値の妥当性（測定の失敗を除く。科学的なカットではない）
# =====================================
logM   = col(t, "sm_MEDIAN")
logSFR = col(t, "sfr_MEDIAN")
mass_ok = in_range(logM, *LOGM_RANGE) & (logM != -1.0)
sfr_ok  = in_range(logSFR, *LOGSFR_RANGE) & (logSFR != -1.0)
re_sane = re_ok & (Re_arcsec > RE_ARCSEC_MIN) & (Re_kpc < RE_KPC_MAX)

print("\n===== 値の妥当性（座標・z が有効な行の中で）=====")
for name, m in [("log M* が範囲外", ~mass_ok), ("log SFR が範囲外", ~sfr_ok),
                ("Re が無い", ~re_ok),
                (f"Re < {RE_ARCSEC_MIN}″ または > {RE_KPC_MAX} kpc", re_ok & ~re_sane)]:
    print(f"  {name:30s}: {np.sum(coord_ok & m):7,}")
print("  log M* の範囲外の値（例）:", np.unique(np.round(logM[coord_ok & ~mass_ok], 3))[:10])
print("  log SFR の範囲外の値（例）:", np.unique(np.round(logSFR[coord_ok & ~sfr_ok], 3))[:10])

# =====================================
# 6. 重複（同じ銀河の別スペクトル）のフラグ
#    同じ bestObjID、または 1″ 以内かつ |Δz| < 0.001 → 同じ銀河（両方の和集合）
#    代表：値が妥当なものを優先し、その中で SN_MEDIAN が最大（nₑ に依存しない）
# =====================================
# 6a. bestObjID が同じ行どうしをつなぐ
idv = np.where(coord_ok & (bestobjid != 0))[0]
_, first_i, inv_id = np.unique(bestobjid[idv], return_index=True, return_inverse=True)
eA_i, eA_j = idv, idv[first_i][inv_id]

# 6b. 位置と z が一致する行どうしをつなぐ
idx = np.where(coord_ok)[0]
cc = SkyCoord(ra[idx] * u.deg, dec[idx] * u.deg)
i, j, _, _ = cc.search_around_sky(cc, DUP_SEP * u.arcsec)
k = (i < j) & (np.abs(z[idx][i] - z[idx][j]) < DZ_TOL)
eB_i, eB_j = idx[i[k]], idx[j[k]]
same_id = (bestobjid[eB_i] == bestobjid[eB_j]) & (bestobjid[eB_i] != 0)

# 6c. つながった行を 1 つの銀河にまとめる（連結成分）
G = coo_matrix((np.ones(len(eA_i) + len(eB_i)),
                (np.r_[eA_i, eB_i], np.r_[eA_j, eB_j])), shape=(N_ALL, N_ALL))
_, gal = connected_components(G, directed=False)
cnt = np.bincount(gal)
n_spec = cnt[gal]
multi = n_spec > 1

# 6d. 代表を 1 本選ぶ
good = coord_ok & mass_ok & sfr_ok & re_sane
rank = col(t, "SN_MEDIAN")
rank = np.where(np.isfinite(rank), rank, -np.inf)
order = np.lexsort((row, -rank, ~good, gal))    # 銀河ごと → 妥当なもの優先 → SN_MEDIAN 降順
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
# 7. Z_MAX：光度の下限とフラックスの限界の交点
# =====================================
def z_max_from_flux_limit(l_min=L_MIN, f_lim=FLUX_LIMIT):
    def fz(zz):
        return 4 * np.pi * cosmo.luminosity_distance(zz).to(u.cm).value**2 * f_lim - l_min
    return brentq(fz, 1e-4, 1.0)

Z_MAX = z_max_from_flux_limit()
print(f"\n[INFO] Z_MAX = {Z_MAX:.4f}")

# =====================================
# 8. 必要な量
# =====================================
def flux(name):
    return col(t, f"{name}_FLUX") * UNIT_FLUX, col(t, f"{name}_FLUX_ERR") * UNIT_FLUX

def sn(F, E):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(E > 0, F / E, np.nan)

F_S16, E_S16 = flux("SII_6717")
F_S31, E_S31 = flux("SII_6731")
F_Hb,  E_Hb  = flux("H_BETA")
F_O3,  E_O3  = flux("OIII_5007")
F_Ha,  E_Ha  = flux("H_ALPHA")
F_N2,  E_N2  = flux("NII_6584")

dL = np.full(N_ALL, np.nan)
dL[coord_ok] = cosmo.luminosity_distance(z[coord_ok]).to(u.cm).value
with np.errstate(invalid="ignore", divide="ignore"):
    L6716 = 4 * np.pi * dL**2 * F_S16
    L6731 = 4 * np.pi * dL**2 * F_S31
    log_N2Ha = np.log10(F_N2 / F_Ha)
    log_O3Hb = np.log10(F_O3 / F_Hb)
    logsSFR     = logSFR - logM
    logSigmaSFR = logSFR - np.log10(2 * np.pi * Re_kpc**2)   # ΣSFR = SFR / (2π Re²)

sn_Hb, sn_O3, sn_Ha, sn_N2 = sn(F_Hb, E_Hb), sn(F_O3, E_O3), sn(F_Ha, E_Ha), sn(F_N2, E_N2)

# =====================================
# 9. 選択の条件
# =====================================
in_z  = coord_ok & (z < Z_MAX)
base  = is_primary & in_z & mass_ok & sfr_ok
if REQUIRE_RE:
    base &= re_sane

lum_sane = np.isfinite(L6731) & (L6731 < L6731_MAX)
lum = lum_sane & (L6731 > L_MIN)
print(f"[INFO] base の中で L(6731) > {L6731_MAX:.0e}（フラックスの失敗）: {np.sum(base & ~lum_sane & np.isfinite(L6731)):,}")

sn4 = (sn_Hb >= SN_MIN) & (sn_O3 >= SN_MIN) & (sn_Ha >= SN_MIN) & (sn_N2 >= SN_MIN)

with np.errstate(invalid="ignore"):
    sf_ke01 = (log_N2Ha < 0.47) & (log_O3Hb < 0.61 / (log_N2Ha - 0.47) + 1.19)
    sf_ka03 = (log_N2Ha < 0.05) & (log_O3Hb < 0.61 / (log_N2Ha - 0.05) + 1.3)

selected = base & lum & sn4 & sf_ke01

# =====================================
# 10. 選択の流れ（2 章の表に使う）
# =====================================
s1 = coord_ok
s2 = s1 & is_primary
s3 = s2 & in_z
s4 = s3 & mass_ok & sfr_ok
s5 = base
flow = [
    ("SDSS DR7 MPA/JHU（全スペクトル）",    np.ones(N_ALL, bool)),
    ("座標・z が有効",                      s1),
    ("+ 重複を除く（1 銀河 1 スペクトル）",  s2),
    (f"+ z < {Z_MAX:.4f}",                  s3),
    ("+ M*, SFR が妥当",                    s4),
]
if REQUIRE_RE:
    flow.append(("+ 有効半径が妥当", s5))
flow += [
    (f"+ L([SII]6731) > {L_MIN:.0e}",       base & lum),
    (f"+ S/N >= {SN_MIN:g}（4 本）",         base & lum & sn4),
    (f"+ 星形成（{METHOD}）",                selected),
]
print("\n===== Selection flow =====")
rows = []
for name, m in flow:
    print(f"  {name:34s}: {m.sum():>9,}")
    rows.append({"step": name, "N": int(m.sum())})
pd.DataFrame(rows).to_csv(os.path.join(out_dir, f"selection_flow_{METHOD}.csv"), index=False)
print(f"  （参考）Kauffmann+03 の場合            : {(base & lum & sn4 & sf_ka03).sum():>9,}")

# =====================================
# 11. 最終サンプルの性質（範囲と 1%–99%）
# =====================================
def rng(x):
    x = x[np.isfinite(x)]
    p1, p99 = np.percentile(x, [1, 99])
    return f"{x.min():8.3f} -- {x.max():8.3f}   (1–99%: {p1:7.3f} -- {p99:7.3f})"

s = selected
print("\n===== Sample statistics =====")
print(f"  N(selected)      = {s.sum():,}")
print(f"  z                = {rng(z[s])}")
print(f"  log M*           = {rng(logM[s])}")
print(f"  log SFR          = {rng(logSFR[s])}")
print(f"  log sSFR         = {rng(logsSFR[s])}")
print(f"  log ΣSFR         = {rng(logSigmaSFR[s])}")
print(f"  Re [kpc]         = {rng(Re_kpc[s])}")
print(f"  Re [arcsec]      = {rng(Re_arcsec[s])}")
print(f"  log L(6731)      = {rng(np.log10(L6731[s]))}")
print(f"  元は重複だった銀河 = {np.sum(s & multi):,}")

# =====================================
# 12. 列を追加して保存
# =====================================
new_cols = {
    "COORD_Z_VALID": coord_ok, "SPEC_MATCHED": matched, "BESTOBJID": bestobjid,
    "Z_DR17": z_dr17, "RA_DR17": ra_dr17, "DEC_DR17": dec_dr17, "SEP_DR17_ARCSEC": sep,
    **rad,
    "Re_maj_arcsec": Re_maj, "Re_circ_arcsec": Re_circ, "Re_kpc": Re_kpc,
    "RE_VALID": re_ok, "RE_SANE": re_sane, "MASS_OK": mass_ok, "SFR_OK": sfr_ok,
    "GALAXY_ID": gal, "N_SPEC": n_spec, "IS_PRIMARY": is_primary,
    "L_SII6716": L6716, "L_SII6731": L6731,
    "logsSFR": logsSFR, "logSigmaSFR": logSigmaSFR,
    "log_N2Ha": log_N2Ha, "log_O3Hb": log_O3Hb,
    "BASE": base, "LUM_OK": lum, "SN4_OK": sn4,
    "SF_Ke01": sf_ke01, "SF_Ka03": sf_ka03, "SELECTED": selected,
}
for name, v in new_cols.items():
    t[name] = v
assert len(t) == N_ALL and np.array_equal(np.asarray(t["ROW_ID"]), row)

master_path = os.path.join(out_dir, "mpajhu_dr7_v5_2_master.fits")
t.write(master_path, format="fits", overwrite=True)
print(f"\n[DONE] {master_path}（全 {N_ALL:,} 行）")

sample_path = os.path.join(out_dir, f"sdss_sample_zlt{Z_MAX:.4f}_Lgt{L_MIN:.0e}_{METHOD}.fits")
t[selected].write(sample_path, format="fits", overwrite=True)
print(f"[DONE] {sample_path}（{selected.sum():,} 行）")

# =====================================
# 13. 図
# =====================================
def finish(ax, path):
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"[DONE] {path}")

# 13a. 体積限定
fig, ax = plt.subplots(figsize=(12, 6))
ok = is_primary & np.isfinite(L6731) & (L6731 > 0)
ax.scatter(z[ok], L6731[ok], s=0.2, alpha=0.2, color="gray", rasterized=True)
vol = base & lum
ax.scatter(z[vol], L6731[vol], s=0.2, alpha=0.5, color="firebrick", rasterized=True)
zg = np.linspace(1e-4, 0.4, 400)
ax.plot(zg, 4 * np.pi * cosmo.luminosity_distance(zg).to(u.cm).value**2 * FLUX_LIMIT, color="k", lw=2)
ax.axvline(Z_MAX, color="k", lw=2)
ax.axhline(L_MIN, color="k", lw=2)
ax.set_yscale("log")
ax.set_xlim(0, 0.4); ax.set_ylim(1e36, 1e42)
ax.set_xlabel(r"$z$")
ax.set_ylabel(r"$L([{\rm S\,II}]\lambda6731)$ [erg s$^{-1}$]")
finish(ax, os.path.join(fig_dir, "sii6731_luminosity_vs_z.png"))

# 13b. BPT 図
pre = base & lum & sn4
fig, ax = plt.subplots(figsize=(10, 9))
ax.scatter(log_N2Ha[pre], log_O3Hb[pre], s=0.5, alpha=0.1, color="gray", rasterized=True)
ax.scatter(log_N2Ha[selected], log_O3Hb[selected], s=0.5, alpha=0.2, color="firebrick", rasterized=True)
xx = np.linspace(-2.0, 0.46, 500)
ax.plot(xx, 0.61 / (xx - 0.47) + 1.19, color="k", lw=2, label="Kewley+01")
xx = np.linspace(-2.0, 0.04, 500)
ax.plot(xx, 0.61 / (xx - 0.05) + 1.3, color="k", lw=2, ls="--", label="Kauffmann+03")
ax.set_xlim(-2.0, 0.5); ax.set_ylim(-1.5, 1.5)
ax.set_xlabel(r"$\log([{\rm N\,II}]\lambda6584/{\rm H}\alpha)$")
ax.set_ylabel(r"$\log([{\rm O\,III}]\lambda5007/{\rm H}\beta)$")
ax.legend(loc="lower left")
finish(ax, os.path.join(fig_dir, "bpt_diagram.png"))

# 13c. z 分布
fig, ax = plt.subplots(figsize=(12, 6))
ax.hist(z[selected], bins=50, color="gray", edgecolor="black", alpha=0.8)
ax.set_xlabel(r"$z$"); ax.set_ylabel("Number of galaxies")
ax.set_xlim(0, Z_MAX)
finish(ax, os.path.join(fig_dir, "selected_redshift_histogram.png"))

# 13d. Re–M*
fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(logM[selected], np.log10(Re_kpc[selected]), s=0.5, alpha=0.1, color="gray", rasterized=True)
ax.set_xlim(7, 12); ax.set_ylim(-1, 1.5)
ax.set_xlabel(r"$\log(M_\ast/M_\odot)$")
ax.set_ylabel(r"$\log(R_{\rm e}/{\rm kpc})$")
finish(ax, os.path.join(fig_dir, "selected_size_mass.png"))




# # ---- (1) log M* が範囲外の行の中身 ----
# m = coord_ok & ~mass_ok
# v = logM[m]
# print("log M* 範囲外:", m.sum())
# print("  NaN:", np.isnan(v).sum(), "  = -1:", np.sum(v == -1),
#       "  -1 以外で < 6:", np.sum((v < 6) & (v != -1)), "  > 13:", np.sum(v > 13))
# print("  多い値 top10:\n", pd.Series(np.round(v[np.isfinite(v)], 3)).value_counts().head(10).to_string())

# mm = is_primary & in_z & ~mass_ok
# print("\n重複除去後・z<Z_MAX で範囲外:", mm.sum())
# print("  そのうち L>1e39:", (mm & lum).sum(),
#       "  さらに S/N>=3:", (mm & lum & sn4).sum(),
#       "  さらに Ke01 で星形成:", (mm & lum & sn4 & sf_ke01).sum())

# # 偏りの確認：プレートと MJD に集中しているか
# pl = pd.Series(np.asarray(t["PLATEID"])[mm]).value_counts()
# print(f"\n  範囲外が出るプレート数: {len(pl)}  上位:\n{pl.head(10).to_string()}")
# mjd = np.asarray(t["MJD"])
# print("  MJD の範囲（範囲外）:", np.percentile(mjd[mm], [1, 50, 99]))
# print("  MJD の範囲（全体）  :", np.percentile(mjd[is_primary & in_z], [1, 50, 99]))

# # SFR は普通の値か（質量だけ失敗しているのか）
# print("\n  範囲外の行の log SFR 1–99%:", np.nanpercentile(logSFR[mm], [1, 50, 99]))
# print("  範囲外の行の z 1–99%     :", np.nanpercentile(z[mm], [1, 50, 99]))

# # ---- (2) 不一致プレートが最終サンプルにどれだけ効くか ----
# um_sel = ~matched & is_primary & in_z & mass_ok & sfr_ok & lum & sn4 & sf_ke01
# print("\nDR17 不一致で、Re 以外の条件を満たす星形成銀河:", um_sel.sum())