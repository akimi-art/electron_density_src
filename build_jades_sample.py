# =====================================================================
# build_jades_sample.py
#   step3 の FITS（JADES DR3 + Nishigaki+26 + Shibuya+15）から高 z サンプルを作る。スタックはしない
#   - 全行を残し、条件はフラグ列で記録する
#   - スペクトルの条件までを全パネル共通の「基本サンプル」とし、
#     M*, SFR, sSFR, ΣSFR のパネルごとに、必要な量がある銀河でサンプルを作る
# =====================================================================
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
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

# =====================================
# 0. 設定
# =====================================
STEP3    = "results/JADES/JADES_DR3/master/jades_dr3_mr_step3_shibuya.fits"
SPEC_DIR = "results/JADES/JADES_DR3/JADES_DR3_full_spectra"
OUT_DIR  = "results/JADES/sample"
FIG_DIR  = "results/JADES/figure/sample"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# グレーティング、使う z の範囲（上から順に試す）、露出時間の列
GRATINGS = [
    ("G140M", f"{SPEC_DIR}/JADES_DR3_G140M", 0.5, 1.7, "tExp_G140M"),
    ("G235M", f"{SPEC_DIR}/JADES_DR3_G235M", 1.5, 3.6, "tExp_G235M"),
    ("G395M", f"{SPEC_DIR}/JADES_DR3_G395M", 3.3, 6.7, "tExp_G395M"),
]
FALLBACK = True                 # ファイルがない・範囲が足りないときに次のグレーティングを試す

# 静止系の波長窓 [Å] と必要な有限ピクセル数（Hα は条件にしない）
WIN = {"sii": (6705, 6740, 2), "cont": (6600, 6650, 3)}

# 測定の失敗を除くための緩い範囲（局所側と同じ。科学的なカットではない）
LOGM_RANGE   = (6.0, 13.0)
LOGSFR_RANGE = (-10.0, 3.0)

RE_COL = "Re_UV_circ_kpc"       # Shibuya+15 の長軸の Re を √qUV で円形化した値

# 目視で除いた ID。理由を ID ごとに書く
BAD_IDS = {
    "00024958": "要記入", "00025356": "要記入", "00051209": "要記入", "00082961": "要記入",
    "00004504": "要記入", "00028139": "要記入", "00045967": "要記入",
    "10020038": "単一ピクセルのスパイク",
}

def f64(col):
    return np.ma.filled(np.ma.asarray(col).astype(float), np.nan)

def sstr(col):
    return np.char.strip(np.asarray(col).astype(str))

def in_range(x, lo, hi):
    return np.isfinite(x) & (x > lo) & (x < hi)

# =====================================
# 1. 読み込み
# =====================================
t = Table.read(STEP3)
N_ALL = len(t)
print(f"[LOADED] {STEP3}（{N_ALL} 行）")

ids    = np.asarray(t["NIRSpec_ID"]).astype(np.int64)
sid    = np.array([f"{i:08d}" for i in ids])
tier   = sstr(t["TIER"])
field  = sstr(t["FILE_FIELD"])
z      = f64(t["z_Spec"])
in_n26 = np.asarray(t["IN_N26"], bool)

# =====================================
# 2. z が有効で、いずれかのグレーティングの範囲にある（[SII] が観測範囲に入りうる）
# =====================================
z_ok = np.isfinite(z) & (z > 0)
z_in_grating = np.zeros(N_ALL, bool)
for _, _, zlo, zhi, _ in GRATINGS:
    z_in_grating |= z_ok & (z > zlo) & (z < zhi)

# =====================================
# 3. 同じ銀河の重複：同じフィールドで同じ NIRSpec_ID（TIER だけ違う）の行を 1 行にする
#    決め方：(1) [SII] を観測するグレーティングの露出時間が長い方
#            (2) 同じなら、中分散グレーティング（G140M, G235M, G395M）の露出時間の合計が長い方
#            (3) それも同じなら ROW_ID の小さい方
#    z がグレーティングの範囲にある行だけを対象にする（範囲外の行は先に落ちる）
# =====================================
def texp_col(name):
    return np.nan_to_num(f64(t[name]), nan=0.0)

texp_sii = np.zeros(N_ALL)
for i in range(N_ALL):
    for _, _, zlo, zhi, tcol in GRATINGS:
        if z_ok[i] and zlo < z[i] < zhi:
            texp_sii[i] = texp_col(tcol)[i]
            break
texp_total = texp_col("tExp_G140M") + texp_col("tExp_G235M") + texp_col("tExp_G395M")
key = np.char.add(np.char.add(field, "|"), sid)

cand = np.where(in_n26 & z_in_grating)[0]
df_k = pd.DataFrame({"row": cand, "key": key[cand],
                     "texp_sii": texp_sii[cand], "texp_total": texp_total[cand]})
df_k = df_k.sort_values(["key", "texp_sii", "texp_total", "row"],
                        ascending=[True, False, False, True])
one_per_gal = np.zeros(N_ALL, bool)
one_per_gal[df_k.drop_duplicates("key", keep="first")["row"].to_numpy()] = True

dup_keys = df_k["key"][df_k["key"].duplicated(keep=False)].unique()
print(f"\n同じ銀河が複数行ある（Nishigaki+26 の中、z がグレーティングの範囲）: {len(dup_keys)} 銀河")
for k in dup_keys:
    rows = df_k[df_k["key"] == k]
    print(f"  {k}: " + ", ".join(
        f"{tier[r]}(tExp[SII]={texp_sii[r]:.0f}, 合計={texp_total[r]:.0f}{', 採用' if one_per_gal[r] else ''})"
        for r in rows["row"]))

# =====================================
# 4. スペクトル：TIER と ID が一致するファイルを探し、[SII] と連続光の波長範囲を確かめる
# =====================================
def n_finite(wave, flux, lo, hi):
    m = (wave > lo) & (wave < hi)
    return int(np.sum(np.isfinite(flux[m])))

def coverage(file, zz):
    with fits.open(file) as h:
        d = h["EXTRACT1D"].data
        wave = d["WAVELENGTH"] * 1e4 / (1 + zz)
        flux = d["FLUX"]
    return {k: n_finite(wave, flux, lo, hi) >= n for k, (lo, hi, n) in WIN.items()}

grating   = np.full(N_ALL, "", dtype="U8")
spec_file = np.full(N_ALL, "", dtype="U300")
found = np.zeros(N_ALL, bool); cov_sii = np.zeros(N_ALL, bool); cov_cont = np.zeros(N_ALL, bool)
n_multi = 0

for i in np.where(one_per_gal)[0]:
    for gname, gdir, zlo, zhi, _ in GRATINGS:
        if not (zlo < z[i] < zhi):
            continue
        files = glob.glob(f"{gdir}/*{tier[i]}-{sid[i]}*_x1d.fits")
        if len(files) > 1:
            n_multi += 1
        if not files:
            if FALLBACK:
                continue
            break
        cov = coverage(files[0], z[i])
        ok = cov["sii"] and cov["cont"]
        if ok or not FALLBACK:
            grating[i], spec_file[i], found[i] = gname, files[0], True
            cov_sii[i], cov_cont[i] = cov["sii"], cov["cont"]
            break
print(f"\n一つの行に複数のファイルが当たった数: {n_multi}")

# =====================================
# 5. 基本サンプル（全パネル共通）
# =====================================
not_bad = ~np.isin(sid, list(BAD_IDS.keys()))

steps = [
    ("DR3 中分散カタログ（全行）",                np.ones(N_ALL, bool)),
    ("+ Nishigaki+26 にある",                     in_n26),
    ("+ z がグレーティングの範囲（0.5–6.7）",     z_in_grating),
    ("+ 同じ銀河は 1 行",                         one_per_gal),
    ("+ スペクトルがある",                        found),
    ("+ [SII] と連続光が観測範囲にある",          cov_sii & cov_cont),
    ("+ 目視で除いた ID ではない",                not_bad),
]
cum = np.ones(N_ALL, bool)
rows = []
print("\n===== 選択の流れ（基本サンプルまで）=====")
for name, m in steps:
    cum = cum & m
    print(f"  {name:34s}: {cum.sum():>5}")
    rows.append({"step": name, "N": int(cum.sum())})
base = cum
print("  グレーティングごと（基本サンプル）:", pd.Series(grating[base]).value_counts().to_dict())

# =====================================
# 6. パネルごとのサンプル（値があること → 局所側と同じ妥当な範囲）
# =====================================
logM   = f64(t["N26_logM"])
logSFR = f64(t["N26_logSFR_hb"])
re_kpc = f64(t[RE_COL])

mass_has  = np.isfinite(logM)
mass_ok   = in_range(logM, *LOGM_RANGE)
sfr_has   = np.asarray(t["N26_SFR_POSITIVE"], bool) & np.isfinite(logSFR)
sfr_ok    = sfr_has & in_range(logSFR, *LOGSFR_RANGE)
re_ok     = np.asarray(t["RE_UV_VALID"], bool) & np.isfinite(re_kpc) & (re_kpc > 0)

panel_steps = {
    "MASS":  [("M* あり", mass_has), (f"+ {LOGM_RANGE[0]:g} < log M* < {LOGM_RANGE[1]:g}", mass_ok)],
    "SFR":   [("SFR > 0", sfr_has),  (f"+ {LOGSFR_RANGE[0]:g} < log SFR < {LOGSFR_RANGE[1]:g}", sfr_ok)],
    "SSFR":  [("M*, SFR とも妥当", mass_ok & sfr_ok)],
    "SIGMA": [("SFR が妥当", sfr_ok), ("+ UV の Re あり", re_ok)],
}
panels = {}
print("\n===== パネルごとのサンプル（基本サンプルから）=====")
for pname, plist in panel_steps.items():
    cum_p = base.copy()
    for name, m in plist:
        cum_p = cum_p & m
        print(f"  [{pname:5s}] {name:28s}: {cum_p.sum():>5}")
        rows.append({"step": f"[{pname}] {name}", "N": int(cum_p.sum())})
    panels[pname] = cum_p
pd.DataFrame(rows).to_csv(f"{OUT_DIR}/selection_flow_jades.csv", index=False)

# =====================================
# 7. 各パネルのサンプルの性質
# =====================================
with np.errstate(invalid="ignore", divide="ignore"):
    logsSFR     = logSFR - logM
    logSigmaSFR = logSFR - np.log10(2 * np.pi * re_kpc**2)

def show(name, x):
    x = x[np.isfinite(x)]
    if len(x):
        print(f"  {name:10s}: {x.min():7.3f} -- {x.max():7.3f}   中央値 {np.median(x):7.3f}   N = {len(x)}")

print("\n===== パネルごとの範囲 =====")
show("z (基本)", z[base])
show("log M*",   logM[panels["MASS"]])
show("log SFR",  logSFR[panels["SFR"]])
show("log sSFR", logsSFR[panels["SSFR"]])
show("log ΣSFR", logSigmaSFR[panels["SIGMA"]])

# =====================================
# 8. 保存（全行 + フラグ、基本サンプル）
# =====================================
t["Z_IN_GRATING"]   = z_in_grating
t["ONE_PER_GALAXY"] = one_per_gal
t["TEXP_SII"]       = texp_sii
t["TEXP_TOTAL_MR"]  = texp_total
t["GRATING_SII"]    = grating
t["SPEC_FILE"]      = spec_file
t["SPEC_FOUND"]     = found
t["COV_SII"], t["COV_CONT"] = cov_sii, cov_cont
t["NOT_BAD_ID"]     = not_bad
t["MASS_OK"], t["SFR_OK"], t["RE_OK"] = mass_ok, sfr_ok, re_ok
t["BASE_SAMPLE"]    = base
for k, v in panels.items():
    t[f"SAMPLE_{k}"] = v
t["logsSFR"], t["logSigmaSFR"] = logsSFR, logSigmaSFR

t.write(f"{OUT_DIR}/jades_all_with_flags.fits", overwrite=True)
t[base].write(f"{OUT_DIR}/jades_base_sample.fits", overwrite=True)
print(f"\n[DONE] {OUT_DIR}/jades_all_with_flags.fits（全 {N_ALL} 行）")
print(f"[DONE] {OUT_DIR}/jades_base_sample.fits（{base.sum()} 行。パネルは SAMPLE_* の列で選ぶ）")

# =====================================
# 9. 図：z（基本サンプル）と、各パネルの量の分布
# =====================================
def finish(ax, path):
    for sp in ax.spines.values():
        sp.set_linewidth(2)
    plt.tight_layout(); plt.savefig(path, dpi=200); plt.show()
    print(f"[DONE] {path}")

fig, ax = plt.subplots(figsize=(12, 6))
ax.hist(z[base], bins=15, color="0.7", edgecolor="black")
ax.set_xlabel(r"$z$"); ax.set_ylabel("Number of galaxies")
finish(ax, f"{FIG_DIR}/hist_z_JADES.png")

for x, pname, xlabel, fname in [
        (logM,        "MASS",  r"$\log(M_\ast/M_\odot)$",                                   "hist_mass_JADES.png"),
        (logSFR,      "SFR",   r"$\log({\rm SFR}/M_\odot\,{\rm yr}^{-1})$",                  "hist_sfr_JADES.png"),
        (logsSFR,     "SSFR",  r"$\log({\rm sSFR}/{\rm yr}^{-1})$",                          "hist_ssfr_JADES.png"),
        (logSigmaSFR, "SIGMA", r"$\log(\Sigma_{\rm SFR}/M_\odot\,{\rm yr}^{-1}\,{\rm kpc}^{-2})$", "hist_sigmasfr_JADES.png")]:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(x[panels[pname]], bins=20, color="0.7", edgecolor="black")
    ax.set_xlabel(xlabel); ax.set_ylabel("Number of galaxies")
    finish(ax, f"{FIG_DIR}/{fname}")