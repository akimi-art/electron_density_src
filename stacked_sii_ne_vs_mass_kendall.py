#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
スクリプトの概要:
このスクリプトはne_vs_smのスタックデータの
統計検定を行います。

使用方法:
    stacked_sii_ne_vs_mass_kendall.py [オプション]

著者: A. M.
作成日: 2026-02-02

参考文献:
    - PEP 8:                  https://peps.python.org/pep-0008/
    - PEP 257 (Docstring規約): https://peps.python.org/pep-0257/
    - Python公式ドキュメント:    https://docs.python.org/ja/3/
"""

# === 必要なパッケージのインストール === #
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kendalltau
import emcee

# 軸の設定
plt.rcParams.update({
    # --- 図全体 ---
    "figure.figsize": (12, 6),       # 図サイズ
    "font.size": 16,                 # 全体フォントサイズ
    "axes.labelsize": 18,            # 軸ラベルのサイズ
    "axes.titlesize": 18,            # タイトルのサイズ
    "axes.grid": False,              # グリッドOFF

    # --- 目盛り設定 (ticks) ---
    "xtick.direction": "in",         # x軸目盛りの向き
    "ytick.direction": "in",         # y軸目盛りの向き
    "xtick.top": True,               # 上にも目盛り
    "ytick.right": True,             # 右にも目盛り

    # 主目盛り（major ticks）
    "xtick.major.size": 16,          # 長さ
    "ytick.major.size": 16,
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
    "xtick.labelsize": 18,           # x軸ラベルサイズ
    "ytick.labelsize": 18,           # y軸ラベルサイズ
})

# ============================================================
# 設定
# ============================================================
# in_csv  = "results/table/stacked_sii_ne_vs_mass_from_ratio_COMPLETE.csv"
in_csv  = "results/csv/stacked_sii_ne_vs_mass_from_ratio_COMPLETE.csv"
band_csv = "results/csv/stacked_ne_vs_sm_regression_band_COMPLETE.csv"
out_png = "results/figure/stacked_ne_vs_sm_COMPLETE.png"

# ============================================================
# 読み込み
# ============================================================
df = pd.read_csv(in_csv)

# x,y（log空間）
x_data = df["logM_cen"].to_numpy(float)
y_data = df["log_ne_med"].to_numpy(float)

# xerr：ビン幅/2（なければ固定値でもOK）
xerr_data = 0.5 * (df["logM_hi"].to_numpy(float) - df["logM_lo"].to_numpy(float))

# yerr：非対称（すでに log 空間）
yerr_lo = df["log_ne_err_lo"].to_numpy(float)
yerr_hi = df["log_ne_err_hi"].to_numpy(float)

# ratioが理論範囲外の点は落としたいなら（推奨）
if "R_outside" in df.columns:
    inside = ~df["R_outside"].to_numpy(bool)
else:
    inside = np.ones_like(x_data, dtype=bool)

# ============================================================
# データ健全性マスク
# ============================================================
mask = (
    inside &
    np.isfinite(x_data) & np.isfinite(y_data) &
    np.isfinite(xerr_data) & (xerr_data >= 0) &
    np.isfinite(yerr_lo) & (yerr_lo > 0) &
    np.isfinite(yerr_hi) & (yerr_hi > 0)
)

x_m = x_data[mask]
y_m = y_data[mask]
xerr_m = xerr_data[mask]
yerr_lo_m = yerr_lo[mask]
yerr_hi_m = yerr_hi[mask]

print(f"[DEBUG] used points: {mask.sum()} / {len(x_data)}")

# ============================================================
# 1) Kendall's tau（誤差なし）
# ============================================================
if len(x_m) >= 2:
    tau, p_value = kendalltau(x_m, y_m)
    print(f"Kendall's tau = {tau:.3f}")
    print(f"p-value      = {p_value:.3g}")
else:
    print("[WARN] Kendall: 有効データ不足 (n < 2)")

# ============================================================
# 数値安定化：誤差フロア（必要なら）
# ============================================================
xerr_floor = 1e-4  # dex（ビン幅があるならほぼ不要）
yerr_floor = 1e-4  # dex
xerr_m = np.clip(xerr_m, xerr_floor, None)
yerr_lo_m = np.clip(yerr_lo_m, yerr_floor, None)
yerr_hi_m = np.clip(yerr_hi_m, yerr_floor, None)

# ============================================================
# モデル：y = a*x + b
# 追加パラメータ：log_s (intrinsic scatter in dex)
# ============================================================
def model(theta, x):
    a, b, log_s = theta
    return a * x + b

def log_prior(theta):
    a, b, log_s = theta
    # ざっくり広い事前（必要なら調整）
    if (-10 < a < 10) and (-50 < b < 50) and (-10 < log_s < 1):
        return 0.0
    return -np.inf

def log_likelihood(theta, x, y, xerr, yerr_lo, yerr_hi):
    a, b, log_s = theta
    s_int = 10**log_s

    y_model = a * x + b
    resid = y - y_model

    # 非対称誤差：残差の符号で使う sigma_y を切り替え
    sigma_y = np.where(resid >= 0, yerr_hi, yerr_lo)

    # effective variance：yerr^2 + (a*xerr)^2 + s_int^2
    sigma2 = sigma_y**2 + (a * xerr)**2 + s_int**2

    return -0.5 * np.sum(resid**2 / sigma2 + np.log(sigma2))

def log_posterior(theta, x, y, xerr, yerr_lo, yerr_hi):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y, xerr, yerr_lo, yerr_hi)

# ============================================================
# MCMC 実行
# ============================================================
ndim = 3
nwalkers = 60
nsteps = 4000

# 初期値（適当に：傾き0、切片平均、散乱0.1dex）
initial_guess = np.array([0.0, np.nanmedian(y_m), np.log10(0.1)])
pos = initial_guess + 1e-3 * np.random.randn(nwalkers, ndim)

if len(x_m) < 2:
    print("[WARN] MCMC: 有効データ不足 (n < 2)。回帰をスキップします。")
else:
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_posterior,
        args=(x_m, y_m, xerr_m, yerr_lo_m, yerr_hi_m)
    )
    sampler.run_mcmc(pos, nsteps, progress=True)

    # burn-in & thinning
    flat = sampler.get_chain(discard=800, thin=10, flat=True)

    a_med, b_med, log_s_med = np.percentile(flat, [50], axis=0)[0]
    a16, b16, log_s16 = np.percentile(flat, [16], axis=0)[0]
    a84, b84, log_s84 = np.percentile(flat, [84], axis=0)[0]

    print(f"MCMC slope a = {a_med:.3f} (+{a84-a_med:.3f} -{a_med-a16:.3f})")
    print(f"MCMC intercept b = {b_med:.3f} (+{b84-b_med:.3f} -{b_med-b16:.3f})")
    print(f"intrinsic scatter (dex) = {10**log_s_med:.3f}")
    print(f"results: {tau:.3f}, {p_value:.3f}, {a_med:.3f}, {(a84-a16)/2:.3f}, {b_med:.3f}, {(b84-b16)/2:.3f}")

    # ========================================================
    # 回帰帯（posteriorから y(x) の分布）
    # ========================================================
    x_band = np.linspace(6, 12, 500)
    y_band_samples = np.array([a * x_band + b for a, b, ls in flat])

    y_med  = np.percentile(y_band_samples, 50, axis=0)
    y_low  = np.percentile(y_band_samples, 16, axis=0)
    y_high = np.percentile(y_band_samples, 84, axis=0)

    pd.DataFrame({"x": x_band, "y_med": y_med, "y_low": y_low, "y_high": y_high}).to_csv(band_csv, index=False)
    print("[INFO] Regression band saved:", band_csv)

    # ========================================================
    # 可視化
    # ========================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(
        x_m, y_m,
        yerr=[yerr_lo_m, yerr_hi_m],
        xerr=xerr_m,
        fmt="o", capsize=2, ms=5, lw=1, alpha=0.9,
        label="binned points"
    )
    ax.plot(x_band, y_med, color="k", label="median")
    ax.fill_between(x_band, y_low, y_high, color="k", alpha=0.2, label="16-84%")

    ax.set_xlabel(r"log $M_\star$ [M$_\odot$]")
    ax.set_ylabel(r"log $n_e$ [cm$^{-3}$]")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    # === 枠線 (spines) の設定 ===
    # 線の太さ・色・表示非表示などを個別に制御
    for spine in ax.spines.values():
        spine.set_linewidth(2)       # 枠線の太さ
        spine.set_color("black")     # 枠線の色
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.show()
    print("[INFO] Figure saved:", out_png)