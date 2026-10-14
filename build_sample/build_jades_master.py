#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
スクリプトの概要:
このスクリプトは
JADESの銀河カタログとShibuya+2022のHST画像解析カタログをクロスマッチし、
JADESの銀河に対してShibuya+2022のサイズ（ReffUV, ReffOpt）を付与するものです。

使用方法:
    JADES_HST_crossmatch.py [オプション]

著者: A. M.
作成日: 2026-10-27

参考文献:
    - PEP 8: https://peps.python.org/pep-0008/
    - PEP 257 (Docstring規約): https://peps.python.org/pep-0257/
    - Python公式ドキュメント: https://docs.python.org/ja/3/
"""

# import pandas as pd
# import numpy as np
# from io import StringIO

# from astropy.coordinates import SkyCoord
# import astropy.units as u

# # =====================================================
# # 1. JADES catalog 読み込み
# # =====================================================

# jades = pd.read_csv("results/JADES/JADES_DR3/data_from_Nishigaki/jades_info_with_HA_plus_logSFR.csv")

# # =====================================================
# # 2. Shibuya+15 catalog 読み込み
# #    （VOTable風 TSV）
# # =====================================================

# fname = "data/data_HST/Shibuya_et_al_2015/tsv/Shibuya_et_al_2015_table4.tsv"

# with open(fname, "r") as f:
#     lines = f.readlines()

# # -----------------------------------------------------
# # <![CDATA[ の位置を探す
# # -----------------------------------------------------

# start = None

# for i, line in enumerate(lines):
#     if "<![CDATA[" in line:
#         start = i + 1
#         break

# if start is None:
#     raise ValueError("CDATA start not found")

# # -----------------------------------------------------
# # CDATA 部分のみ抽出
# # -----------------------------------------------------

# data_lines = []

# for line in lines[start:]:

#     if "]]>" in line:
#         break

#     data_lines.append(line)

# csv_text = "".join(data_lines)

# # -----------------------------------------------------
# # pandas で読む
# # -----------------------------------------------------

# shibuya = pd.read_csv(
#     StringIO(csv_text),
#     sep=";"
# )

# # -----------------------------------------------------
# # 単位行・区切り行を削除
# # -----------------------------------------------------

# shibuya = shibuya.iloc[3:].reset_index(drop=True)

# # -----------------------------------------------------
# # 列名の空白除去
# # -----------------------------------------------------

# shibuya.columns = shibuya.columns.str.strip()

# # -----------------------------------------------------
# # 数値化
# # -----------------------------------------------------

# cols = [
#     "ReffUV", "e_ReffUV",
#     "ReffOpt", "e_ReffOpt",
#     "_RA", "_DE"
# ]

# for c in cols:
#     shibuya[c] = pd.to_numeric(
#         shibuya[c],
#         errors="coerce"
#     )

# # -----------------------------------------------------
# # RA/DEC 欠損除去
# # -----------------------------------------------------

# shibuya = shibuya.dropna(subset=["_RA", "_DE"])

# print("Shibuya catalog loaded")
# print(shibuya.head())

# # =====================================================
# # 3. SkyCoord 作成
# # =====================================================

# coord_jades = SkyCoord(
#     ra=jades["RA_TARG"].values * u.deg,
#     dec=jades["Dec_TARG"].values * u.deg
# )

# coord_shibuya = SkyCoord(
#     ra=shibuya["_RA"].values * u.deg,
#     dec=shibuya["_DE"].values * u.deg
# )

# # =====================================================
# # 4. クロスマッチ
# # =====================================================

# idx, d2d, _ = coord_jades.match_to_catalog_sky(
#     coord_shibuya
# )

# # -----------------------------------------------------
# # マッチ半径
# # -----------------------------------------------------

# max_sep = 0.5 * u.arcsec

# matched = d2d < max_sep

# print(f"Matched: {matched.sum()} / {len(jades)}")

# # =====================================================
# # 5. Reff 列追加
# # =====================================================

# jades["ReffUV"]    = np.nan
# jades["e_ReffUV"]  = np.nan
# jades["ReffOpt"]   = np.nan
# jades["e_ReffOpt"] = np.nan

# # -----------------------------------------------------
# # マッチしたものだけ代入
# # -----------------------------------------------------

# jades.loc[matched, "ReffUV"] = (
#     shibuya.iloc[idx[matched]]["ReffUV"].values
# )

# jades.loc[matched, "e_ReffUV"] = (
#     shibuya.iloc[idx[matched]]["e_ReffUV"].values
# )

# jades.loc[matched, "ReffOpt"] = (
#     shibuya.iloc[idx[matched]]["ReffOpt"].values
# )

# jades.loc[matched, "e_ReffOpt"] = (
#     shibuya.iloc[idx[matched]]["e_ReffOpt"].values
# )

# # =====================================================
# # 6. separation も保存（便利）
# # =====================================================

# jades["match_sep_arcsec"] = np.nan

# jades.loc[matched, "match_sep_arcsec"] = (
#     d2d[matched].arcsec
# )

# # =====================================================
# # 7. 保存
# # =====================================================

# outname = "results/JADES/JADES_DR3/data_from_Nishigaki/jades_info_with_HA_plus_logSFR_with_Reff.csv"

# jades.to_csv(
#     outname,
#     index=False
# )

# print(f"Saved: {outname}")





# =====================================================================
# build_jades_master.py
#   1. JADES DR3（中分散グレーティング）の GN と GS をまとめる       → step1
#   2. Nishigaki+26 の値を NIRSpec_ID（＋必要なら TIER）で入れる     → step2
#   3. Shibuya+15 と位置で照合し、Re を入れる（確認の数字と図つき）   → step3
#   全行を残し、対応の有無はフラグ列で記録する。段階ごとに保存し、途中から再開できる
# =====================================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u

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
CAT_DIR     = "results/JADES/JADES_DR3/catalog"
GN_FITS     = f"{CAT_DIR}/jades_dr3_medium_gratings_public_gn_v1.1.fits"
GS_FITS     = f"{CAT_DIR}/jades_dr3_medium_gratings_public_gs_v1.1.fits"
N26_CSV     = "results/JADES/JADES_DR3/data_from_Nishigaki/jades_info.csv"
SHIBUYA_TSV = "data/data_HST/Shibuya_et_al_2015/tsv/Shibuya_et_al_2015_table4.tsv"

OUT_DIR = "results/JADES/JADES_DR3/master"
FIG_DIR = "results/JADES/figure/master"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
STEP1 = f"{OUT_DIR}/jades_dr3_mr_step1_gn_gs.fits"
STEP2 = f"{OUT_DIR}/jades_dr3_mr_step2_nishigaki.fits"
STEP3 = f"{OUT_DIR}/jades_dr3_mr_step3_shibuya.fits"

START_FROM   = 3                    # 1, 2, 3 のどこから始めるか（前の段階のファイルを読む）
MATCH_RADIUS = 0.2                  # Shibuya+15 との照合の半径 [arcsec]
SHIFT_TEST   = 30.0                 # 偶然の一致率を調べるために座標をずらす量 [arcsec]
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)   # 局所側と同じ宇宙論

def sstr(col):
    """文字列の列を、前後の空白を除いた str の配列にする"""
    return np.char.strip(np.asarray(col).astype(str))

def f64(col):
    return np.ma.filled(np.ma.asarray(col).astype(float), np.nan)

# =====================================
# 1. GN と GS をまとめる
# =====================================
if START_FROM <= 1:
    gn, gs = Table.read(GN_FITS), Table.read(GS_FITS)
    print(f"GN: {len(gn)} 行   GS: {len(gs)} 行")
    assert gn.colnames == gs.colnames, "GN と GS で列が違う"
    t = vstack([gn, gs], join_type="exact")
    t["ROW_ID"] = np.arange(len(t))
    t["FILE_FIELD"] = np.array(["GN"] * len(gn) + ["GS"] * len(gs))
    assert len(t) == len(gn) + len(gs)

    ids, tiers = np.asarray(t["NIRSpec_ID"]).astype(np.int64), sstr(t["TIER"])
    n_id = len(np.unique(ids))
    n_idt = len(set(zip(ids, tiers)))
    print(f"まとめた行数: {len(t)}   ユニークな NIRSpec_ID: {n_id}   ユニークな (NIRSpec_ID, TIER): {n_idt}")
    if n_id < len(t):
        u_, c_ = np.unique(ids, return_counts=True)
        ex = u_[c_ > 1][:5]
        print("  同じ NIRSpec_ID が複数行ある例:")
        for e in ex:
            m = ids == e
            print("   ", e, list(zip(tiers[m], sstr(t["Field"])[m])))

    t.write(STEP1, overwrite=True)
    print(f"[SAVED] {STEP1}")
else:
    t = Table.read(STEP1)
    print(f"[LOADED] {STEP1}（{len(t)} 行）")

N_ALL = len(t)

# =====================================
# 2. Nishigaki+26 の値を入れる
# =====================================
if START_FROM <= 2:
    n26 = pd.read_csv(N26_CSV)
    n26["TIER"] = n26["TIER"].astype(str).str.strip()
    n26_id = n26["NIRSpec_ID"].astype(np.int64).to_numpy()
    print(f"\nNishigaki+26: {len(n26)} 行   ユニークな NIRSpec_ID: {len(np.unique(n26_id))}")

    dr_id, dr_tier = np.asarray(t["NIRSpec_ID"]).astype(np.int64), sstr(t["TIER"])
    # NIRSpec_ID だけで一意でなければ (NIRSpec_ID, TIER) で対応づける
    use_tier = (len(np.unique(dr_id)) < N_ALL) or (len(np.unique(n26_id)) < len(n26))
    if use_tier:
        print("  NIRSpec_ID だけでは一意でないので (NIRSpec_ID, TIER) で対応づける")
        dr_key  = [f"{i}|{tr}" for i, tr in zip(dr_id, dr_tier)]
        n26_key = [f"{i}|{tr}" for i, tr in zip(n26_id, n26["TIER"])]
    else:
        dr_key, n26_key = [str(i) for i in dr_id], [str(i) for i in n26_id]

    dup = pd.Series(n26_key)[pd.Series(n26_key).duplicated(keep=False)]
    assert len(dup) == 0, f"Nishigaki+26 側でキーが重複: {sorted(set(dup))[:10]}"

    lookup = {k: j for j, k in enumerate(n26_key)}
    jj = np.array([lookup.get(k, -1) for k in dr_key])
    in_n26 = jj >= 0
    not_found = set(n26_key) - set(dr_key)
    print(f"  DR3 の中で Nishigaki+26 にある行: {in_n26.sum()} / {len(n26)}"
          f"（DR3 に見つからない Nishigaki+26 の行: {len(not_found)}）")
    if not_found:
        print("   例:", sorted(not_found)[:10])

    def take(name):
        out = np.full(N_ALL, np.nan)
        out[in_n26] = pd.to_numeric(n26[name], errors="coerce").to_numpy()[jj[in_n26]]
        return out

    for c in ["z_spec", "logM", "err1_logM", "err2_logM", "E_BV",
              "SFR_hb", "SFR_hb_lower", "SFR_hb_upper",
              "R3", "R2", "R23", "O32", "logOH_direct", "logOH_r23", "logOH",
              "RA_TARG", "Dec_TARG"]:
        t[f"N26_{c}"] = take(c)
    t["IN_N26"] = in_n26

    # SFR_hb は線形の値として扱い、正のものだけ log を取る（要確認）
    sfr = np.asarray(t["N26_SFR_hb"])
    with np.errstate(invalid="ignore", divide="ignore"):
        t["N26_logSFR_hb"] = np.where(sfr > 0, np.log10(sfr), np.nan)
    t["N26_SFR_POSITIVE"] = np.isfinite(sfr) & (sfr > 0)

    # 確認：z と座標が DR3 と一致しているか
    dz = np.abs(np.asarray(t["N26_z_spec"]) - f64(t["z_Spec"]))
    sep = SkyCoord(np.asarray(t["N26_RA_TARG"])[in_n26] * u.deg, np.asarray(t["N26_Dec_TARG"])[in_n26] * u.deg) \
        .separation(SkyCoord(f64(t["RA_TARG"])[in_n26] * u.deg, f64(t["Dec_TARG"])[in_n26] * u.deg)).arcsec
    print(f"  |Δz|（Nishigaki+26 − DR3）: 中央値 {np.nanmedian(dz[in_n26]):.1e}  最大 {np.nanmax(dz[in_n26]):.1e}")
    print(f"  座標のずれ: 中央値 {np.median(sep):.3f}″  最大 {np.max(sep):.3f}″")
    print(f"  Nishigaki+26 の行のうち  M* あり: {np.sum(in_n26 & np.isfinite(t['N26_logM']))}"
          f"   SFR > 0: {np.sum(t['N26_SFR_POSITIVE'])}"
          f"   SFR <= 0: {np.sum(in_n26 & np.isfinite(sfr) & (sfr <= 0))}")

    t.write(STEP2, overwrite=True)
    print(f"[SAVED] {STEP2}")
elif START_FROM == 3:
    t = Table.read(STEP2)
    print(f"[LOADED] {STEP2}（{len(t)} 行）")

# =====================================
# 3. Shibuya+15 と位置で照合する
#   3a 読み込み → 3b 仮の照合（補正前）→ 3c ずれの確認 → 3d ずれの位置依存の確認
#   → 3e 補正（1 次式）→ 3f 補正後の確認 → 3g 本番の照合 → 3h 列を入れて保存
# =====================================
PRE_RADIUS = 1.0     # 仮の照合で、ずれを調べる範囲 [arcsec]
# MATCH_RADIUS（0. の設定）は 3f の出力を見て決める

def save(fig, axes, name):
    for a in np.atleast_1d(axes).ravel():
        for sp in a.spines.values():
            sp.set_linewidth(2)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/{name}", dpi=150)
    plt.show()
    print(f"[DONE] {FIG_DIR}/{name}")

# ---- 3a. VizieR の TSV（CDATA 部分）を読む ----
with open(SHIBUYA_TSV) as fh:
    lines = fh.readlines()
start = next(i for i, l in enumerate(lines) if "<![CDATA[" in l) + 1
end   = next(i for i, l in enumerate(lines[start:], start) if "]]>" in l)
sh = pd.read_csv(StringIO("".join(lines[start:end])), sep=";")
sh = sh.iloc[2:].reset_index(drop=True)          # 単位の行と区切りの行を除く
sh.columns = sh.columns.str.strip()
sh["Field"] = sh["Field"].astype(str).str.strip()
for c in ["ID", "ReffUV", "e_ReffUV", "qUV", "ReffOpt", "e_ReffOpt", "qOpt", "Flag", "_RA", "_DE"]:
    sh[c] = pd.to_numeric(sh[c], errors="coerce")
sh = sh.dropna(subset=["_RA", "_DE"]).reset_index(drop=True)
sh_ra, sh_de = sh["_RA"].to_numpy(), sh["_DE"].to_numpy()
print(f"\nShibuya+15: {len(sh)} 天体   フィールド: {sh['Field'].value_counts().to_dict()}")

ra, dec = f64(t["RA_TARG"]), f64(t["Dec_TARG"])
okc   = np.isfinite(ra) & np.isfinite(dec)
field = np.asarray(t["FILE_FIELD"]).astype(str)
n26m  = np.asarray(t["IN_N26"], bool)
cs    = SkyCoord(sh_ra * u.deg, sh_de * u.deg)

# フィールド × TIER の種類（確認の図の色分け用）
tier = sstr(t["TIER"])
kind = np.where(np.char.find(np.char.lower(tier), "jwst") >= 0, "jwst",
       np.where(np.char.find(np.char.lower(tier), "hst") >= 0, "hst", "other"))
grp  = np.char.add(np.char.add(field, "-"), kind)

def nearest(ra_, dec_):
    """各行について、最も近い Shibuya の天体の番号と距離 [arcsec]"""
    c = SkyCoord(ra_[okc] * u.deg, dec_[okc] * u.deg)
    i_, d_, _ = c.match_to_catalog_sky(cs)
    sep_ = np.full(N_ALL, np.nan); sep_[okc] = d_.arcsec
    idx_ = np.full(N_ALL, -1);     idx_[okc] = i_
    return sep_, idx_

def offsets(ra_, dec_, idx_, m):
    """ずれ（Shibuya − JADES）[arcsec]：Δα cosδ と Δδ"""
    j = idx_[m]
    dra_  = (sh_ra[j] - ra_[m]) * np.cos(np.deg2rad(dec_[m])) * 3600
    ddec_ = (sh_de[j] - dec_[m]) * 3600
    return dra_, ddec_

def plot_offsets(dra_, ddec_, g_, lim, radius, name):
    """左：ずれのベクトル、右：距離の分布（どちらもフィールド × TIER で色分け）"""
    groups = list(np.unique(g_))
    colors = dict(zip(groups, ["firebrick", "tab:blue", "tab:green", "tab:orange", "tab:purple", "k"]))
    fig, axes = plt.subplots(1, 2, figsize=(24, 11))
    ax = axes[0]
    for x in groups:
        m = g_ == x
        ax.scatter(dra_[m], ddec_[m], s=25, color=colors[x], label=f"{x} ({m.sum()})")
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(radius * np.cos(th), radius * np.sin(th), "k--", lw=2)
    ax.axhline(0, color="0.6", lw=1); ax.axvline(0, color="0.6", lw=1)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ax.set_xlabel(r"$\Delta\alpha\cos\delta$ [arcsec]"); ax.set_ylabel(r"$\Delta\delta$ [arcsec]")
    ax.legend(fontsize=20, loc="upper left")
    ax = axes[1]
    bins = np.logspace(-2.5, np.log10(lim), 40)
    for x in groups:
        m = g_ == x
        ax.hist(np.hypot(dra_[m], ddec_[m]), bins=bins, histtype="step", lw=2.5, color=colors[x], label=x)
    ax.axvline(radius, color="k", ls="--", lw=2)
    ax.set_xscale("log"); ax.set_xlabel("Separation [arcsec]"); ax.set_ylabel("Number")
    ax.legend(fontsize=20)
    save(fig, axes, name)

# ---- 3b. 仮の照合（補正前の座標）----
sep_raw, idx_raw = nearest(ra, dec)
near_raw = np.isfinite(sep_raw) & (sep_raw < PRE_RADIUS)
dra_raw, ddec_raw = offsets(ra, dec, idx_raw, near_raw)
g_raw = grp[near_raw]

# ---- 3c. ずれの確認（補正前）----
print(f"\n===== 3c. グループごとのずれ（補正前、{PRE_RADIUS}″ 以内、Shibuya − JADES）=====")
for x in np.unique(g_raw):
    m = g_raw == x
    print(f"  {x:10s}: N = {m.sum():4d}   ΔRA cosδ 中央値 {np.median(dra_raw[m]):+.3f}″"
          f"   ΔDec 中央値 {np.median(ddec_raw[m]):+.3f}″")
plot_offsets(dra_raw, ddec_raw, g_raw, lim=1.0, radius=0.5, name="shibuya_offset_check_raw.png")

# フィールドごとのかたまりの中心と、0.5″ 以内でかたまりから離れた点の数（補正前）
centers = {}
for fld in ["GN", "GS"]:
    m = np.char.startswith(g_raw, fld)
    c0 = np.array([np.median(dra_raw[m]), np.median(ddec_raw[m])])
    nc = np.hypot(dra_raw[m] - c0[0], ddec_raw[m] - c0[1]) < 0.15
    centers[fld] = np.array([np.median(dra_raw[m][nc]), np.median(ddec_raw[m][nc])])
    dist_c = np.hypot(dra_raw[m] - centers[fld][0], ddec_raw[m] - centers[fld][1])
    inside = np.hypot(dra_raw[m], ddec_raw[m]) < 0.5
    print(f"  {fld}: かたまりの中心 ({centers[fld][0]:+.3f}, {centers[fld][1]:+.3f})″"
          f"   0.5″ 以内 {inside.sum()}   そのうち中心から 0.1″ 以上離れたもの {np.sum(inside & (dist_c > 0.1))}")

# ---- 3d. ずれが位置によって変わるか（フィールドごと、かたまりの点）----
for fld in ["GN", "GS"]:
    m = np.char.startswith(g_raw, fld)
    inc = m & (np.hypot(dra_raw - centers[fld][0], ddec_raw - centers[fld][1]) < 0.1)
    rr, dd = ra[near_raw][inc], dec[near_raw][inc]
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    for i, (x, xl) in enumerate([(rr, "RA [deg]"), (dd, "Dec [deg]")]):
        for k, (y, yl) in enumerate([(dra_raw[inc], r"$\Delta\alpha\cos\delta$ [arcsec]"),
                                     (ddec_raw[inc], r"$\Delta\delta$ [arcsec]")]):
            ax = axes[k, i]
            ax.scatter(x, y, s=15, color="k")
            ax.axhline(centers[fld][k], color="firebrick", ls="--", lw=2)
            ax.set_xlabel(xl); ax.set_ylabel(yl)
    axes[0, 0].set_title(fld, fontsize=28, loc="left")
    save(fig, axes, f"shibuya_offset_vs_position_{fld}.png")

# ---- 3e. 補正：ずれ = a + b × (座標 − 中心)（赤経方向は赤経、赤緯方向は赤緯の 1 次式）----
print("\n===== 3e. 補正の式（Shibuya − JADES）=====")
ra_corr, dec_corr = ra.copy(), dec.copy()
for fld in ["GN", "GS"]:
    m = np.char.startswith(g_raw, fld)
    x_ra, x_de = ra[near_raw][m], dec[near_raw][m]
    y_ra, y_de = dra_raw[m], ddec_raw[m]
    ra0, de0 = np.median(x_ra), np.median(x_de)
    use = np.hypot(y_ra - centers[fld][0], y_de - centers[fld][1]) < 0.1   # かたまりの点から始める
    for _ in range(5):                                                     # 外れた点を除きながらあてはめる
        pa  = np.polyfit(x_ra[use] - ra0, y_ra[use], 1)
        pd_ = np.polyfit(x_de[use] - de0, y_de[use], 1)
        res = np.hypot(y_ra - np.polyval(pa, x_ra - ra0), y_de - np.polyval(pd_, x_de - de0))
        use = res < max(3 * np.median(res[use]), 0.03)
    print(f"  {fld}: ΔRA cosδ = {pa[1]:+.3f}″ + {pa[0]:+.3f}″/deg × (RA − {ra0:.4f})"
          f"   ΔDec = {pd_[1]:+.3f}″ + {pd_[0]:+.3f}″/deg × (Dec − {de0:.4f})"
          f"   使った数 {use.sum()}   残りのばらつき（中央値）{np.median(res[use]):.3f}″")
    f_ = field == fld
    ra_corr[f_]  = ra[f_] + np.polyval(pa, ra[f_] - ra0) / 3600 / np.cos(np.deg2rad(dec[f_]))
    dec_corr[f_] = dec[f_] + np.polyval(pd_, dec[f_] - de0) / 3600

# ---- 3f. 補正後の確認 ----
sep_cor, idx_cor = nearest(ra_corr, dec_corr)
near_cor = np.isfinite(sep_cor) & (sep_cor < PRE_RADIUS)
dra_cor, ddec_cor = offsets(ra_corr, dec_corr, idx_cor, near_cor)
plot_offsets(dra_cor, ddec_cor, grp[near_cor], lim=1.0, radius=MATCH_RADIUS,
             name="shibuya_offset_check_corrected.png")

# 偶然の一致：補正後の座標を赤緯方向に SHIFT_TEST だけずらして同じ照合をする
sep_shift, _ = nearest(ra_corr, dec_corr + SHIFT_TEST / 3600)

fig, ax = plt.subplots(figsize=(14, 9))
bins = np.logspace(-2.5, 1.5, 60)
for s_, col, lab in [(sep_raw, "0.6", "before correction"), (sep_cor, "k", "after correction"),
                     (sep_shift, "firebrick", f"shifted by {SHIFT_TEST:g}\"")]:
    ax.hist(s_[n26m & np.isfinite(s_)], bins=bins, histtype="step", lw=2.5, color=col, label=lab)
ax.axvline(MATCH_RADIUS, color="k", ls="--", lw=2)
ax.set_xscale("log"); ax.set_xlabel("Separation to the nearest source [arcsec]"); ax.set_ylabel("Number")
ax.legend(fontsize=22)
save(fig, ax, "shibuya_separation_before_after.png")

print("\n===== 3f. 半径を決めるための表（Nishigaki+26 の行）=====")
print("  半径     補正前の一致   補正後の一致   偶然の一致（期待数）")
for r_ in [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]:
    print(f"  {r_:4.2f}″   {np.sum(n26m & (sep_raw < r_)):8d}       {np.sum(n26m & (sep_cor < r_)):8d}"
          f"        {np.sum(n26m & (sep_shift < r_)):6d}")

# ---- 3g. 本番の照合（補正後の座標）----
nn_sep, nn_idx = sep_cor, idx_cor
matched = np.isfinite(nn_sep) & (nn_sep < MATCH_RADIUS)
false_n26 = np.sum(n26m & (sep_shift < MATCH_RADIUS))
mi = nn_idx[matched]
u_, c_ = np.unique(mi, return_counts=True)
many = u_[c_ > 1]

# ---- 3h. 列を入れて保存 ----
def take_sh(name):
    out = np.full(N_ALL, np.nan)
    out[matched] = sh[name].to_numpy()[nn_idx[matched]]
    return out

t["RA_TARG_CORR"], t["Dec_TARG_CORR"] = ra_corr, dec_corr   # 照合に使った補正後の座標
t["SH_NN_SEP_RAW"]    = sep_raw                            # 補正前の最も近い天体までの距離
t["SH_NN_SEP_ARCSEC"] = nn_sep                             # 補正後の最も近い天体までの距離
t["SH_MATCHED"] = matched
t["SH_ID"] = np.where(matched, sh["ID"].to_numpy()[np.clip(nn_idx, 0, None)], -1).astype(np.int64)
sh_fields = sh["Field"].to_numpy(dtype=str)
t["SH_FIELD"] = np.where(matched, sh_fields[np.clip(nn_idx, 0, None)], "")
for c in ["ReffUV", "e_ReffUV", "qUV", "ReffOpt", "e_ReffOpt", "qOpt", "Flag"]:
    t[f"SH_{c}"] = take_sh(c)

zz = f64(t["z_Spec"])
kpc = np.full(N_ALL, np.nan)
okz = np.isfinite(zz) & (zz > 0)
kpc[okz] = cosmo.angular_diameter_distance(zz[okz]).to(u.kpc).value * np.pi / 180 / 3600
t["Re_UV_kpc"]      = np.asarray(t["SH_ReffUV"]) * kpc
t["Re_UV_circ_kpc"] = np.asarray(t["SH_ReffUV"]) * np.sqrt(np.asarray(t["SH_qUV"])) * kpc
t["RE_UV_VALID"] = matched & np.isfinite(t["Re_UV_kpc"]) & (t["Re_UV_kpc"] > 0)

print(f"\n===== 3g. 本番の照合（補正後、半径 {MATCH_RADIUS}″）=====")
print(f"  全行: {matched.sum()} / {N_ALL}   Nishigaki+26 の行: {np.sum(matched & n26m)} / {n26m.sum()}")
print(f"  Nishigaki+26 の行で UV の Re あり: {np.sum(t['RE_UV_VALID'] & n26m)}")
print(f"  偶然の一致の期待数（Nishigaki+26 の中）: {false_n26}")
print(f"  一つの Shibuya の天体に複数の JADES 天体: {len(many)} 件")

def frac_table(x, bins, label):
    print(f"  {label} ごとの一致率（Nishigaki+26 の行、UV の Re あり）:")
    for lo, hi in zip(bins[:-1], bins[1:]):
        b = n26m & np.isfinite(x) & (x >= lo) & (x < hi)
        if b.sum():
            print(f"    {lo:5.1f}-{hi:5.1f}: {np.sum(b & t['RE_UV_VALID']):4d} / {b.sum():4d}"
                  f" = {np.mean(np.asarray(t['RE_UV_VALID'])[b]):.2f}")

for fld in ["GN", "GS"]:
    b = n26m & (field == fld)
    print(f"  {fld}: {np.sum(b & t['RE_UV_VALID'])} / {b.sum()}")
frac_table(zz, np.array([0, 1, 2, 3, 4, 5, 6, 8]), "z")
frac_table(np.asarray(t["N26_logM"]), np.arange(7.0, 11.6, 0.5), "log M*")

t.write(STEP3, overwrite=True)
print(f"[SAVED] {STEP3}（全 {N_ALL} 行）")

# =====================================
# 4. 照合の確認の図（補正後）
#   (a) 最も近い天体までの距離（補正後の座標と、ずらした座標）
#   (b) GOODS-S、(c) GOODS-N：灰 = Shibuya+15、赤 = 照合した、青 = 照合しなかった（Nishigaki+26 の行）
#   (d) z ごとの UV の Re がある割合（Nishigaki+26 の行）
# =====================================
fig, axes = plt.subplots(2, 2, figsize=(24, 20))

ax = axes[0, 0]
bins = np.logspace(-2.5, 1.5, 60)
ax.hist(sep_cor[n26m & np.isfinite(sep_cor)], bins=bins, histtype="step", color="k", lw=2.5, label="observed (corrected)")
ax.hist(sep_shift[n26m & np.isfinite(sep_shift)], bins=bins, histtype="step", color="firebrick", lw=2.5,
        label=f"shifted by {SHIFT_TEST:g}\"")
ax.axvline(MATCH_RADIUS, color="k", ls="--", lw=2)
ax.set_xscale("log"); ax.set_xlabel("Separation to the nearest source [arcsec]"); ax.set_ylabel("Number")
ax.legend(fontsize=22)

for ax, fld in [(axes[0, 1], "GS"), (axes[1, 0], "GN")]:
    m_f = field == fld
    rr, dd = ra[m_f & okc], dec[m_f & okc]
    box = (sh_ra > rr.min() - 0.05) & (sh_ra < rr.max() + 0.05) & (sh_de > dd.min() - 0.05) & (sh_de < dd.max() + 0.05)
    ax.scatter(sh_ra[box], sh_de[box], s=0.3, color="0.8", rasterized=True)
    mm, nm = m_f & n26m & matched, m_f & n26m & ~matched
    ax.scatter(ra[mm], dec[mm], s=12, color="firebrick", label="matched")
    ax.scatter(ra[nm], dec[nm], s=12, color="tab:blue", label="not matched")
    ax.invert_xaxis()
    ax.set_xlabel("RA [deg]"); ax.set_ylabel("Dec [deg]"); ax.set_title(fld, fontsize=28, loc="left")
    ax.legend(fontsize=20, loc="upper right")

ax = axes[1, 1]
zb = np.arange(0, 8.5, 0.5)
fr, zc = [], []
for lo, hi in zip(zb[:-1], zb[1:]):
    b = n26m & (zz >= lo) & (zz < hi)
    if b.sum() >= 5:
        zc.append(0.5 * (lo + hi)); fr.append(np.mean(np.asarray(t["RE_UV_VALID"])[b]))
ax.plot(zc, fr, "o-", color="k", lw=2)
ax.set_ylim(0, 1.05); ax.set_xlabel(r"$z$"); ax.set_ylabel(r"Fraction with UV $R_{\rm e}$")

save(fig, axes, "shibuya_match_check.png")