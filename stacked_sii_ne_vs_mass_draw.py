#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
スクリプトの概要:
ne vs stellar mass の図を作成する
使用方法:
    stacked_sii_ne_vs_mass_draw.py [オプション]

著者: A. M.
作成日: 2026-09-29

参考文献:
    - PEP 8:                  https://peps.python.org/pep-0008/
    - PEP 257 (Docstring規約): https://peps.python.org/pep-0257/
    - Python公式ドキュメント:    https://docs.python.org/ja/3/
    - Curti+17
"""

import numpy as np
import pandas as pd
import pyneb as pn
import matplotlib.pyplot as plt


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



# ===== 設定 =====
in_csv  = "results/csv/stacked_sii_ratio_vs_mass_COMPLETE_v4.csv"
out_csv = "results/csv/stacked_sii_ne_vs_mass_from_ratio_COMPLETE_v4.csv"
Te = 1.0e4  # K

# ===== 読み込み =====
res = pd.read_csv(in_csv)

# 1. mean
R_mean = res["R_mean"].to_numpy(float)
R_mean_lo = (res["R_mean"] - res["R_mean_err_lo"]).to_numpy(float)
R_mean_hi  = (res["R_mean"] + res["R_mean_err_hi"]).to_numpy(float)

# 2. median
R_med = res["R_med"].to_numpy(float)
R_med_lo  = (res["R_med"] - res["R_med_err_lo"]).to_numpy(float)
R_med_hi  = (res["R_med"] + res["R_med_err_hi"]).to_numpy(float)

# 3. weighted mean
R_w = res["R_w"].to_numpy(float)
R_w_lo = (res["R_w"] - res["R_w_err_lo"]).to_numpy(float)
R_w_hi = (res["R_w"] + res["R_w_err_hi"]).to_numpy(float)

# 4. mean (ha norm)
R_ha_mean = res["R_Ha_mean"].to_numpy(float)
R_ha_mean_lo   = (res["R_Ha_mean"]   - res["R_Ha_mean_err_lo"]).to_numpy(float)
R_ha_mean_hi   = (res["R_Ha_mean"]   + res["R_Ha_mean_err_hi"]).to_numpy(float)

# 5. median (ha norm)
R_ha_med = res["R_Ha_med"].to_numpy(float)
R_ha_med_lo   = (res["R_Ha_med"]   - res["R_Ha_med_err_lo"]).to_numpy(float)
R_ha_med_hi   = (res["R_Ha_med"]   + res["R_Ha_med_err_hi"]).to_numpy(float)

# 6. weighted mean (ha norm)
R_ha_w = res["R_Ha_w"].to_numpy(float)
R_ha_w_lo   = (res["R_Ha_w"]   - res["R_Ha_w_err_lo"]).to_numpy(float)
R_ha_w_hi   = (res["R_Ha_w"]   + res["R_Ha_w_err_hi"]).to_numpy(float)

# ===== PyNeb =====
S2 = pn.Atom("S", 2)

R_lowlim  = S2.getLowDensRatio(wave1=6716, wave2=6731)
R_highlim = S2.getHighDensRatio(wave1=6716, wave2=6731)

def safe_getTemDen(R):
    R = np.asarray(R, float)
    ne = np.full(R.shape, np.nan)
    m = np.isfinite(R)
    if np.any(m):
        ne[m] = S2.getTemDen(R[m], tem=Te, wave1=6716, wave2=6731)
    return ne

def safe_log10(x):
    x = np.asarray(x, float)
    y = np.full(x.shape, np.nan)
    m = np.isfinite(x) & (x > 0)
    y[m] = np.log10(x[m])
    return y

# ===== ne 計算 =====
ne_mean = safe_getTemDen(R_mean)
ne_mean_hi = safe_getTemDen(R_mean_hi)
ne_mean_lo = safe_getTemDen(R_mean_lo)
ne_med = safe_getTemDen(R_med)
ne_med_hi  = safe_getTemDen(R_med_lo)  # 高密度側
ne_med_lo  = safe_getTemDen(R_med_hi)  # 低密度側
ne_w = safe_getTemDen(R_w)
ne_w_hi = safe_getTemDen(R_w_hi)
ne_w_lo = safe_getTemDen(R_w_lo)
ne_ha_mean = safe_getTemDen(R_ha_mean)
ne_ha_mean_hi = safe_getTemDen(R_ha_mean_hi)
ne_ha_mean_lo = safe_getTemDen(R_ha_mean_lo)
ne_ha_med = safe_getTemDen(R_ha_med)
ne_ha_med_hi = safe_getTemDen(R_ha_med_hi)
ne_ha_med_lo = safe_getTemDen(R_ha_med_lo)
ne_ha_w = safe_getTemDen(R_ha_w)
ne_ha_w_hi = safe_getTemDen(R_ha_w_hi)
ne_ha_w_lo = safe_getTemDen(R_ha_w_lo)

# 線形誤差
ne_mean_err_lo = ne_mean - ne_mean_lo
ne_mean_err_hi = ne_mean_hi  - ne_mean
ne_med_err_lo = ne_med - ne_med_lo
ne_med_err_hi = ne_med_hi  - ne_med
ne_w_err_lo = ne_w - ne_w_lo
ne_w_err_hi = ne_w_hi  - ne_w
ne_ha_mean_err_lo = ne_ha_mean - ne_ha_mean_lo
ne_ha_mean_err_hi = ne_ha_mean_hi  - ne_ha_mean
ne_ha_med_err_lo = ne_ha_med - ne_ha_med_lo
ne_ha_med_err_hi = ne_ha_med_hi  - ne_ha_med
ne_ha_w_err_lo = ne_ha_w - ne_ha_w_lo
ne_ha_w_err_hi = ne_ha_w_hi  - ne_ha_w


# log
log_ne_mean = safe_log10(ne_mean)
log_ne_mean_lo = safe_log10(ne_mean_lo)
log_ne_mean_hi = safe_log10(ne_mean_hi)
log_ne_med = safe_log10(ne_med)
log_ne_med_lo = safe_log10(ne_med_lo)
log_ne_med_hi = safe_log10(ne_med_hi)
log_ne_w = safe_log10(ne_w)
log_ne_w_lo = safe_log10(ne_w_lo)
log_ne_w_hi = safe_log10(ne_w_hi)
log_ne_ha_mean = safe_log10(ne_ha_mean)
log_ne_ha_mean_lo = safe_log10(ne_ha_mean_lo)
log_ne_ha_mean_hi = safe_log10(ne_ha_mean_hi)
log_ne_ha_med = safe_log10(ne_ha_med)
log_ne_ha_med_lo = safe_log10(ne_ha_med_lo)
log_ne_ha_med_hi = safe_log10(ne_ha_med_hi)
log_ne_ha_w = safe_log10(ne_ha_w)
log_ne_ha_w_lo = safe_log10(ne_ha_w_lo)
log_ne_ha_w_hi = safe_log10(ne_ha_w_hi)

log_ne_mean_err_lo = log_ne_mean - log_ne_mean_lo
log_ne_mean_err_hi = log_ne_mean_hi  - log_ne_mean
log_ne_med_err_lo = log_ne_med - log_ne_med_lo
log_ne_med_err_hi = log_ne_med_hi  - log_ne_med
log_ne_w_err_lo = log_ne_w - log_ne_w_lo
log_ne_w_err_hi = log_ne_w_hi  - log_ne_w
log_ne_ha_mean_err_lo = log_ne_ha_mean - log_ne_ha_mean_lo
log_ne_ha_mean_err_hi = log_ne_ha_mean_hi  - log_ne_ha_mean
log_ne_ha_med_err_lo = log_ne_ha_med - log_ne_ha_med_lo
log_ne_ha_med_err_hi = log_ne_ha_med_hi  - log_ne_ha_med
log_ne_ha_w_err_lo = log_ne_ha_w - log_ne_ha_w_lo
log_ne_ha_w_err_hi = log_ne_ha_w_hi  - log_ne_ha_w

# ===== 外れ値判定 =====

# 範囲外フラグ
R_outside_mean = (R_mean > R_lowlim) | (R_mean < R_highlim)
R_outside_med = (R_med > R_lowlim) | (R_med < R_highlim)
R_outside_wmed = (R_w > R_lowlim) | (R_w < R_highlim)
R_outside_ha_mean = (R_ha_mean > R_lowlim) | (R_ha_mean < R_highlim)
R_outside_ha_med = (R_ha_med > R_lowlim) | (R_ha_med < R_highlim)
R_outside_ha_wmed = (R_ha_w > R_lowlim) | (R_ha_w < R_highlim)

# ===== 保存 =====
res["Te_assumed"] = Te
res["R_outside_mean"] = R_outside_mean
res["R_outside_med"] = R_outside_med
res["R_outside_wmed"] = R_outside_wmed
res["R_outside_ha_mean"] = R_outside_ha_mean
res["R_outside_ha_med"] = R_outside_ha_med
res["R_outside_ha_wmed"] = R_outside_ha_wmed

res["ne_mean"] = ne_mean
res["ne_err_lo"] = ne_mean_err_lo
res["ne_err_hi"] = ne_mean_err_hi
res["ne_med"]    = ne_med
res["ne_med_err_lo"] = ne_med_err_lo
res["ne_med_err_hi"] = ne_med_err_hi
res["ne_w"]   = ne_w
res["ne_w_err_lo"] = ne_w_err_lo
res["ne_w_err_hi"] = ne_w_err_hi
res["ne_ha"]   = ne_ha_mean
res["ne_ha_err_lo"] = ne_ha_mean_err_lo
res["ne_ha_err_hi"] = ne_ha_mean_err_hi
res["ne_ha_med"]   = ne_ha_med
res["ne_ha_med_err_lo"] = ne_ha_med_err_lo
res["ne_ha_med_err_hi"] = ne_ha_med_err_hi
res["ne_ha_wmed"]   = ne_ha_w
res["ne_ha_wmed_err_lo"] = ne_ha_w_err_lo
res["ne_ha_wmed_err_hi"] = ne_ha_w_err_hi

res["log_ne_mean"] = log_ne_mean
res["log_ne_mean_err_lo"] = log_ne_mean_err_lo
res["log_ne_mean_err_hi"] = log_ne_mean_err_hi
res["log_ne_med"]    = log_ne_med
res["log_ne_med_err_lo"] = log_ne_med_err_lo
res["log_ne_med_err_hi"] = log_ne_med_err_hi
res["log_ne_w"]   = log_ne_w
res["log_ne_w_err_lo"] = log_ne_w_err_lo
res["log_ne_w_err_hi"] = log_ne_w_err_hi
res["log_ne_ha"]   = log_ne_ha_mean
res["log_ne_ha_err_lo"] = log_ne_ha_mean_err_lo
res["log_ne_ha_err_hi"] = log_ne_ha_mean_err_hi
res["log_ne_ha_med"]   = log_ne_ha_med
res["log_ne_ha_med_err_lo"] = log_ne_ha_med_err_lo
res["log_ne_ha_med_err_hi"] = log_ne_ha_med_err_hi
res["log_ne_ha_w"]   = log_ne_ha_w
res["log_ne_ha_wmed_err_lo"] = log_ne_ha_w_err_lo
res["log_ne_ha_wmed_err_hi"] = log_ne_ha_w_err_hi

res.to_csv(out_csv, index=False)
print("Saved:", out_csv)






# =========================
# log10変換（非対称誤差対応）
# =========================
def log10_with_errors(x, err_plus, err_minus):
    """
    線形値 x と非対称誤差 (+err_plus, -err_minus) を
    log10スケールに変換

    Parameters
    ----------
    x : array-like
        中心値
    err_plus : array-like
        上側誤差
    err_minus : array-like
        下側誤差

    Returns
    -------
    log_x, err_plus_log, err_minus_log
    """
    x = np.asarray(x)
    err_plus = np.asarray(err_plus)
    err_minus = np.asarray(err_minus)

    log_central = np.log10(x)
    log_upper = np.log10(x + err_plus)
    log_lower = np.log10(x - err_minus)

    err_plus_log = log_upper - log_central
    err_minus_log = log_central - log_lower

    return log_central, err_plus_log, err_minus_log


# =========================
# SII ratio → ne 変換関数
# =========================
def compute_ne_from_ratio(ratios, err_plus, err_minus, Te=15000):
    """
    [S II] 6716/6731 比から電子密度 ne を計算（非対称誤差付き）

    Parameters
    ----------
    ratios : array-like
        観測された line ratio
    err_plus : array-like
        上側誤差
    err_minus : array-like
        下側誤差
    Te : float
        電子温度 (K)

    Returns
    -------
    ne_median : ndarray
    err_plus_ne : ndarray
    err_minus_ne : ndarray
    log_ne : ndarray
    log_err_plus : ndarray
    log_err_minus : ndarray
    """

    ratios = np.asarray(ratios)
    err_plus = np.asarray(err_plus)
    err_minus = np.asarray(err_minus)

    S2 = pn.Atom('S', 2)

    ne_median = []
    ne_upper = []
    ne_lower = []

    # --- 各データに対して計算 ---
    for r, ep, em in zip(ratios, err_plus, err_minus):

        ne_c = S2.getTemDen(int_ratio=r, tem=Te, wave1=6716, wave2=6731)

        # 注意：ratioが大きいほど ne は小さくなる
        ne_u = S2.getTemDen(int_ratio=r - em, tem=Te, wave1=6716, wave2=6731)
        ne_l = S2.getTemDen(int_ratio=r + ep, tem=Te, wave1=6716, wave2=6731)

        ne_median.append(ne_c)
        ne_upper.append(ne_u)
        ne_lower.append(ne_l)

    ne_median = np.array(ne_median)
    ne_upper = np.array(ne_upper)
    ne_lower = np.array(ne_lower)

    # --- 非対称誤差 ---
    err_plus_ne = ne_upper - ne_median
    err_minus_ne = ne_median - ne_lower

    # --- log変換 ---
    log_ne, log_err_plus, log_err_minus = log10_with_errors(
        ne_median, err_plus_ne, err_minus_ne
    )

    return ne_median, err_plus_ne, err_minus_ne, log_ne, log_err_plus, log_err_minus


# =========================
# ✅ 使用例
# =========================
if __name__ == "__main__":

    # =========================
    # 手法ごとの入力
    # =========================
    data = {
        "mean": {
            "ratios": [1.2047486012100572, 1.242853285618657],
            "err_plus":  [0.029777448201631973, 0.016637027571879903],
            "err_minus": [0.027074742389230355, 0.015893902820539818],
        },
        "mean_ha_norm": {
            "ratios": [1.369827021245738, 1.2334550072719908],
            "err_plus":  [0.05517335754065078, 0.024566121276547337],
            "err_minus": [0.05154786373784481, 0.023255793241125255],
        },
        "weighted_mean": {
            "ratios": [1.2481768781513405, 1.1387462060318507],
            "err_plus":  [0.03469522850057505, 0.017462509520587588],
            "err_minus": [0.03271390909922589, 0.017087700646649218],
        },
        "weighted_mean_ha_norm": {
            "ratios": [1.2301896977178766, 1.168989344472414],
            "err_plus":  [0.01969201318406899, 0.013253344350796281],
            "err_minus": [0.020089312536189396, 0.012290367561316184],
        },
        "median": {
            "ratios": [1.4635854795598513, 0.9791818098387947],
            "err_plus":  [0.09489266822991294, 0.043013019143494424],
            "err_minus": [0.08601481607568662, 0.04312219080644908],
        },
        "median_ha_norm": {
            "ratios": [1.3097424667915973, 1.3304660338264849],
            "err_plus":  [0.061927184729066775, 0.06183165301297833],
            "err_minus": [0.06034094877354712, 0.05781705415472782],
        },
    }

    # =========================
    # ファイル初期化
    # =========================
    with open("results/txt/ne_jades_results_mass.txt", "w") as f:
        f.write("# method index log_ne err_plus err_minus\n")

    # =========================
    # 計算ループ
    # =========================
    for method, d in data.items():

        ne, ep, em, logne, lep, lem = compute_ne_from_ratio(
            d["ratios"],
            d["err_plus"],
            d["err_minus"],
            Te=15000
        )

        print(f"\n===== {method} =====")

        # ✅ txt保存
        with open("results/txt/ne_jades_results_mass.txt", "a") as f:
            for i in range(len(d["ratios"])):
                f.write(
                    f"{method.replace(' ', '_')} "
                    f"{i} "
                    f"{logne[i]:.5f} "
                    f"{lep[i]:.5f} "
                    f"{lem[i]:.5f}\n"
                )

        # ✅ 既存のprint
        for i in range(len(d["ratios"])):
            print(f"--- Object {i+1} ---")
            print(f"[S II] ratio = {d['ratios'][i]:.3f} "
                  f"(+{d['err_plus'][i]:.3f} -{d['err_minus'][i]:.3f})")
            print(f"ne = {ne[i]:.3f} (+{ep[i]:.3f} -{em[i]:.3f})")
            print(f"log10(ne) = {logne[i]:.3f} (+{lep[i]:.3f} -{lem[i]:.3f})")




# =================================================
# 入出力
# =================================================
in_csv_sdss  = out_csv
out_png      = "results/figure/stacked_sii_logne_vs_mass_JADES_DR3_only_twosided_v4.png"

# =================================================
# Figure
# =================================================
fig, ax = plt.subplots(figsize=(6, 6))


# =================================================
# SDSS
# =================================================
res = pd.read_csv(in_csv_sdss)

x   = res["logM_cen"].to_numpy(float)
y_mean = res["log_ne_mean"].to_numpy(float)
elo_mean = res["log_ne_mean_err_lo"].to_numpy(float)
ehi_mean = res["log_ne_mean_err_hi"].to_numpy(float)
y_med   = res["log_ne_med"].to_numpy(float)
elo_med = res["log_ne_med_err_lo"].to_numpy(float)
ehi_med = res["log_ne_med_err_hi"].to_numpy(float)
y_w   = res["log_ne_w"].to_numpy(float)
elo_w = res["log_ne_w_err_lo"].to_numpy(float)
ehi_w = res["log_ne_w_err_hi"].to_numpy(float)
y_ha   = res["log_ne_ha"].to_numpy(float)
elo_ha = res["log_ne_ha_err_lo"].to_numpy(float)
ehi_ha = res["log_ne_ha_err_hi"].to_numpy(float)
y_ha_med = res["log_ne_ha_med"].to_numpy(float)
elo_ha_med = res["log_ne_ha_med_err_lo"].to_numpy(float)
ehi_ha_med = res["log_ne_ha_med_err_hi"].to_numpy(float)
y_ha_w = res["log_ne_ha_w"].to_numpy(float)
elo_ha_w = res["log_ne_ha_wmed_err_lo"].to_numpy(float)
ehi_ha_w = res["log_ne_ha_wmed_err_hi"].to_numpy(float)

# =================================================
# SDSS
# =================================================

# エラーを非負に
elo_mean_safe = np.maximum(0, elo_mean)
ehi_mean_safe = np.maximum(0, ehi_mean)
elo_med_safe = np.maximum(0, elo_med)
ehi_med_safe = np.maximum(0, ehi_med)
elo_w_safe = np.maximum(0, elo_w)
ehi_w_safe = np.maximum(0, ehi_w)
elo_ha_safe = np.maximum(0, elo_ha)
ehi_ha_safe = np.maximum(0, ehi_ha)
elo_ha_med_safe = np.maximum(0, elo_ha_med)
ehi_ha_med_safe = np.maximum(0, ehi_ha_med)
elo_ha_w_safe = np.maximum(0, elo_ha_w)
ehi_ha_w_safe = np.maximum(0, ehi_ha_w)

# =================================================

# SDSS
# 1.mean
ax.errorbar(
    x, y_mean,
    yerr=np.vstack([elo_mean_safe, ehi_mean_safe]),
    fmt="s",
    mfc="black", mec="black",
    ecolor="black", color="black",
    ms=10,
    capsize=10,
)

# # 2. median
# ax.errorbar(
#     x, y_med,
#     yerr=np.vstack([elo_med_safe, ehi_med_safe]),
#     fmt="^",
#     mfc="black", mec="black",
#     ecolor="black", color="black",
#     ms=10,
#     capsize=10,
# )

# # 3. weighted mean
# ax.errorbar(
#     x, y_w,
#     yerr=np.vstack([elo_w_safe, ehi_w_safe]),
#     fmt="D",
#     mfc="black", mec="black",
#     ecolor="black", color="black",
#     ms=10,
#     capsize=10,
# )

# # 4. mean (ha norm)
# ax.errorbar(
#     x, y_ha,
#     yerr=np.vstack([elo_ha_safe, ehi_ha_safe]),
#     fmt="s",
#     mfc="white", mec="black",
#     ecolor="black", color="black",
#     ms=10,
#     capsize=10,
# )

# # 5. median (ha norm)
# ax.errorbar(
#     x, y_ha_med,
#     yerr=np.vstack([elo_ha_med_safe, ehi_ha_med_safe]),
#     fmt="^",
#     mfc="white", mec="black",
#     ecolor="black", color="black",
#     ms=10,
# )

# # 6. weighted mean (ha norm)
# ax.errorbar(
#     x, y_ha_w,
#     yerr=np.vstack([elo_ha_w_safe, ehi_ha_w_safe]),
#     fmt="D",
#     mfc="white", mec="black",
#     ecolor="black", color="black",
#     ms=10,
# )

# =================================================
# JADES
# =================================================
def load_errorbar_from_txt(filename="ne_jades_results.txt", x_map=None):
    """
    txtファイルから errorbar 用データを読み取り（index→x対応）

    Parameters
    ----------
    filename : str
        入力ファイル
    x_map : dict or list
        index → x の対応
        例：
            {0: 9.18, 1: 9.35, ...}
        または
            [9.18, 9.35, ...]

    Returns
    -------
    results : list of dict
        [
            {
                "method": str,
                "x": float,
                "y": float,
                "yerr": [[err_minus], [err_plus]],
                "fmt": str
            },
            ...
        ]
    """

    style_map = {
        "mean": {
            "fmt": "s",
            "color":  "tab:red",
            "mfc":    "tab:red",
            "mec":    "tab:red",
            "ecolor": "tab:red",
            "ms": 10,
            "capsize": 10,
        },
        # "mean_ha_norm": {
        #     "fmt": "s",
        #     "color":  "tab:red",
        #     "mfc":    "white",
        #     "mec":    "tab:red",
        #     "ecolor": "tab:red",
        #     "ms": 10,
        #     "capsize": 10,
        # },
        # "weighted_mean": {
        #     "fmt": "D",
        #     "color":  "tab:red",
        #     "mfc":    "tab:red",
        #     "mec":    "tab:red",
        #     "ecolor": "tab:red",
        #     "ms": 10,
        #     "capsize": 10,
        # },
        # "weighted_mean_ha_norm": {
        #     "fmt": "D",
        #     "color":  "tab:red",
        #     "mfc":    "white",
        #     "mec":    "tab:red",
        #     "ecolor": "tab:red",
        #     "ms": 10,
        #     "capsize": 10,
        # },
        # "median": {
        #     "fmt": "^",
        #     "color":  "tab:red",
        #     "mfc":    "tab:red",
        #     "mec":    "tab:red",
        #     "ecolor": "tab:red",
        #     "ms": 10,
        #     "capsize": 10,
        # },
        # "median_ha_norm": {
        #     "fmt": "^",
        #     "color":  "tab:red",
        #     "mfc":    "white",
        #     "mec":    "tab:red",
        #     "ecolor": "tab:red",
        #     "ms": 10,
        #     "capsize": 10,
        # },
        #   "mean": {
        #     "fmt": "s",
        #     "color":  "None",
        #     "mfc":    "None",
        #     "mec":    "None",
        #     "ecolor": "None",
        #     "ms": 10,
        #     "capsize": 10,
        # },
        # "mean_ha_norm": {
        #     "fmt": "s",
        #     "color":  "None",
        #     "mfc":    "None",
        #     "mec":    "None",
        #     "ecolor": "None",
        #     "ms": 10,
        #     "capsize": 10,
        # },
        # "weighted_mean": {
        #     "fmt": "D",
        #     "color":  "None",
        #     "mfc":    "None",
        #     "mec":    "None",
        #     "ecolor": "None",
        #     "ms": 10,
        #     "capsize": 10,
        # },
        # "weighted_mean_ha_norm": {
        #     "fmt": "D",
        #     "color":  "None",
        #     "mfc":    "None",
        #     "mec":    "None",
        #     "ecolor": "None",
        #     "ms": 10,
        #     "capsize": 10,
        # },
        # "median": {
        #     "fmt": "^",
        #     "color":  "None",
        #     "mfc":    "None",
        #     "mec":    "None",
        #     "ecolor": "None",
        #     "ms": 10,
        #     "capsize": 10,
        # },
        # "median_ha_norm": {
        #     "fmt": "^",
        #     "color":  "None",
        #     "mfc":    "None",
        #     "mec":    "None",
        #     "ecolor": "None",
        #     "ms": 10,
        #     "capsize": 10,
        # }
    }

    data = np.loadtxt(filename, dtype=str, comments="#")

    results = []

    for row in data:
        method = row[0]
        idx = int(row[1])
        log_ne = float(row[2])
        err_p = float(row[3])
        err_m = float(row[4])

        # =========================
        # index → x変換
        # =========================
        if x_map is None:
            raise ValueError("x_map を指定してください")

        if isinstance(x_map, dict):
            x_val = x_map[idx]
        else:
            x_val = x_map[idx]

        # =========================
        # style決定
        # =========================
        method_key = method.replace("(", "").replace(")", "")
        style = style_map.get(method_key, {"fmt": "o"})

        results.append({
            "method": method,
            "x": x_val,
            "y": log_ne,
            "yerr": [[err_m], [err_p]],
            **style   # ← これが核心
        })

    return results

# x_map = [9.116, 9.523] # 計算方法は?
x_map = [8.977, 9.546] # 平均
# x_map = [8.9, 9.5] # 近い値, 無理に合わせなくて良い

results = load_errorbar_from_txt("ne_jades_results_mass.txt", x_map=x_map)

for r in results:
    ax.errorbar(
        r["x"], r["y"],
        yerr=r["yerr"],
        fmt=r.get("fmt"),
        color=r.get("color"),
        mfc=r.get("mfc"),
        mec=r.get("mec"),
        ecolor=r.get("ecolor"),
        ms=r.get("ms"),
        capsize=r.get("capsize"),
    )



# # ΣSFRと同じサンプル
# ax.errorbar(
#     9.926, 2.427,
#     yerr=[[0.056], [0.060]], # ← これがポイント（形状 (2, 1)）
#     fmt="D",
#     mfc="white", mec="tab:red",
#     ecolor="tab:red", color="tab:red",
#     ms=10,
#     capsize=10,
# )

# =================================================
# 装飾
# =================================================
ax.set_xlabel(r"$\log(M_\ast/M_\odot)$")
ax.set_ylabel(r"$\log\,n_e\ [\mathrm{cm}^{-3}]$")

# ax.set_xlim(9.0, 11.0)
ax.set_xlim(8.6, 11.0)
ax.set_ylim(1.25, 3.0)

for spine in ax.spines.values():
    spine.set_linewidth(2)
    spine.set_color("black")

# ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(out_png)
plt.show()
print("Saved:", out_png)
