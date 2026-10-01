"""
サンプル選択の条件を定義する唯一の場所。
01_build_samples.py と 02_completeness.py は、すべてここを呼び出す。
"""
import numpy as np
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from scipy.optimize import brentq

# ============================================================
# 設定（結果を見る前に決めて固定する）
# ============================================================
COSMO      = FlatLambdaCDM(H0=70, Om0=0.3)
UNIT_FLUX  = 1e-17          # MPA/JHU のフラックスの単位 [erg s^-1 cm^-2]
FLUX_LIMIT = 1e-17          # [SII]6731 のフラックスの限界 [erg s^-1 cm^-2]（根拠を2章に書く）
L_MIN      = 1e39           # [SII]6731 の光度の下限 [erg s^-1]
SN_MIN     = 3.0
METHODS    = ("Ka03", "Ke01", "WHAN")

# MPA/JHU の等価幅は、輝線を負の値で記録している可能性がある。
# 01_build_samples.py が出力する分布を見て確認すること。
EQW_EMISSION_NEGATIVE = True

# MPA/JHU DR7 は、輝線フラックスの誤差の補正係数を推奨している可能性がある（要確認）。
# 確認できたら {"H_ALPHA": xx, "NII_6584": xx, ...} の形で指定する。
ERR_SCALE = None

# 半径のカタログを取り直した後は True にする（全パネルで同じサンプルを使うため）
REQUIRE_RE = False

# 重複を除く列ができたら、その列名を指定する（True の行を残す）
PRIMARY_COL = None

# main sequence の帯（完全性を数える領域。最終サンプルの選択には使わない）
MS_SLOPE, MS_ZP, MS_WIDTH = 0.76, -7.64, 0.6


# ============================================================
# z_max：光度の下限が、フラックスの限界と交わる赤方偏移
# ============================================================
def z_max_from_flux_limit(l_min=L_MIN, f_lim=FLUX_LIMIT, cosmo=COSMO):
    def f(z):
        dL = cosmo.luminosity_distance(z).to(u.cm).value
        return 4 * np.pi * dL**2 * f_lim - l_min
    return brentq(f, 1e-4, 1.0)


# ============================================================
# 輝線と派生量
# ============================================================
def line_flux(df, name):
    F = df[f"{name}_FLUX"].values * UNIT_FLUX
    E = df[f"{name}_FLUX_ERR"].values * UNIT_FLUX
    if ERR_SCALE and name in ERR_SCALE:
        E = E * ERR_SCALE[name]
    return F, E


def derived_quantities(df, cosmo=COSMO):
    """選択と完全性に必要な量をまとめて計算する（1回だけ呼ぶ）"""
    q = {}
    z = df["Z"].values
    zc = np.clip(z, 1e-4, None)
    q["z"] = z
    q["logM"] = df["sm_MEDIAN"].values
    q["logSFR"] = df["sfr_MEDIAN"].values
    q["logsSFR"] = q["logSFR"] - q["logM"]

    # [SII]6731 の光度
    dL = cosmo.luminosity_distance(zc).to(u.cm).value
    F31, _ = line_flux(df, "SII_6731")
    with np.errstate(invalid="ignore"):
        q["L6731"] = 4 * np.pi * dL**2 * F31

    # Rₑ と ΣSFR（SDSS の deVRad と expRad は有効半径なので 1.678 倍しない）
    cols = ["deVRad_r", "expRad_r", "fracDeV_r"]
    if all(c in df.columns for c in cols):
        deV, expR, fdev = (df[c].values for c in cols)
        Re_arcsec = np.where(fdev > 0.5, deV, expR)
        good = (np.isfinite(Re_arcsec) & (Re_arcsec > 0) &
                np.isfinite(fdev) & (fdev >= 0) & (fdev <= 1))
        Re_arcsec = np.where(good, Re_arcsec, np.nan)
        kpc_per_arcsec = cosmo.kpc_proper_per_arcmin(zc).value / 60.0
        q["Re_kpc"] = Re_arcsec * kpc_per_arcsec
        with np.errstate(divide="ignore", invalid="ignore"):
            q["logSigmaSFR"] = q["logSFR"] - np.log10(2 * np.pi * q["Re_kpc"]**2)
    else:
        q["Re_kpc"] = np.full(len(df), np.nan)
        q["logSigmaSFR"] = np.full(len(df), np.nan)

    # 分類に使う輝線
    sn = {}
    flux = {}
    for name in ["H_BETA", "OIII_5007", "H_ALPHA", "NII_6584"]:
        F, E = line_flux(df, name)
        flux[name] = F
        with np.errstate(divide="ignore", invalid="ignore"):
            sn[name] = F / E
    q["sn"] = sn

    with np.errstate(divide="ignore", invalid="ignore"):
        q["log_N2Ha"] = np.log10(flux["NII_6584"] / flux["H_ALPHA"])
        q["log_O3Hb"] = np.log10(flux["OIII_5007"] / flux["H_BETA"])

    eqw = df["H_ALPHA_EQW"].values
    q["W_Ha"] = -eqw if EQW_EMISSION_NEGATIVE else eqw
    q["EQW_raw"] = eqw

    # main sequence の帯
    q["on_ms"] = np.abs(q["logSFR"] - (MS_SLOPE * q["logM"] + MS_ZP)) < MS_WIDTH
    return q


# ============================================================
# 選択のマスク
# ============================================================
def classification(q, method):
    """(S/N の条件, 星形成の条件) を返す"""
    sn = q["sn"]
    sn2 = (sn["H_ALPHA"] >= SN_MIN) & (sn["NII_6584"] >= SN_MIN)
    sn4 = sn2 & (sn["H_BETA"] >= SN_MIN) & (sn["OIII_5007"] >= SN_MIN)
    x, y, W = q["log_N2Ha"], q["log_O3Hb"], q["W_Ha"]

    with np.errstate(invalid="ignore", divide="ignore"):
        if method == "Ka03":
            return sn4, (x < 0.05) & (y < 0.61 / (x - 0.05) + 1.3)
        if method == "Ke01":
            return sn4, (x < 0.47) & (y < 0.61 / (x - 0.47) + 1.19)
        if method == "WHAN":
            return sn2, (x < -0.4) & (W > 3)
    raise ValueError(f"unknown method: {method}")


def selection_masks(df, q, z_max, method, l_min=L_MIN):
    """
    選択の各段階のマスクを返す。
    base    : 親サンプル（輝線の条件なし）
    lum     : 光度の下限
    sn, sf  : 分類の S/N と星形成の判定
    selected: base & lum & sn & sf
    """
    z = q["z"]
    base = (np.isfinite(z) & (z > 0) & (z < z_max) &
            np.isfinite(q["logM"]) & np.isfinite(q["logSFR"]))
    if REQUIRE_RE:
        base &= np.isfinite(q["Re_kpc"])
    if PRIMARY_COL is not None:
        base &= df[PRIMARY_COL].values.astype(bool)

    lum = np.isfinite(q["L6731"]) & (q["L6731"] > l_min)
    sn, sf = classification(q, method)
    selected = base & lum & sn & sf

    # 論文の選択の流れ（体積限定 → 分類の S/N → 分類）
    flow = [
        ("parent (z, M*, SFR"
         + (", Re" if REQUIRE_RE else "")
         + (", primary" if PRIMARY_COL else "") + ")", base),
        ("+ L([SII]6731) > L_MIN",       base & lum),
        (f"+ S/N >= {SN_MIN:g} ({method})", base & lum & sn),
        (f"+ star-forming ({method})",   selected),
    ]
    return dict(base=base, lum=lum, sn=sn, sf=sf, selected=selected, flow=flow)