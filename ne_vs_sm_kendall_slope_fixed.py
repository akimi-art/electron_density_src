#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
スクリプトの概要:
このスクリプトはne_vs_smのプロットデータの
統計検定を行います。
線形回帰分析に関しては、
傾きを固定して切片のみをfree parameterとします。

使用方法:
    ne_vs_sm_kendall.py [オプション]

著者: A. M.
作成日: 2026-01-19

参考文献:
    - PEP 8:                  https://peps.python.org/pep-0008/
    - PEP 257 (Docstring規約): https://peps.python.org/pep-0257/
    - Python公式ドキュメント:    https://docs.python.org/ja/3/
"""

# === 必要なパッケージのインストール === #
import re
import os
import matplotlib.pyplot as plt
import importlib.util
import sys
import psutil
import numpy as np
import matplotlib.pyplot as plt
import emcee
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.stats import kendalltau
import seaborn as sns

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


# === ファイルパスを変数として格納 ===
current_dir = os.getcwd()
Samir16out     = os.path.join(current_dir, "results/Samir16/Samir16out_standard_v3_ms_only_v3_re_no_agn.py")
Mingozzi22out  = os.path.join(current_dir, "results/Mingozzi22/Mingozzi22out.py")
Harikane25out  = os.path.join(current_dir, "results/Harikane25/Harikane25out.py")
Isobe23out     = os.path.join(current_dir, "results/Isobe23/Isobe23out.py")
# SED fittingの結果よりMetallicisyを算出↓
Bunker23out    = os.path.join(current_dir, "results/Bunker23/Bunker23out.py")
# SED fittingの結果よりM*, SFR, metallicityを算出 (Monna+14)
Topping24out   = os.path.join(current_dir, "results/Topping24/Topping24out.py")
Mainali23out   = os.path.join(current_dir, "results/Mainali23/Mainali23out.py")
# 12+log(O/H)を計測できず↓
Berg18out      = os.path.join(current_dir, "results/Berg18/Berg18out.py")
Sanders25out   = os.path.join(current_dir, "results/Sanders25/Sanders25out.py")

# Curti+25のデータ (z~2)
Rigby21out    = os.path.join(current_dir, "results/Rigby21/Rigby21out.py")
Steidel16out  = os.path.join(current_dir, "results/Steidel16/Steidel16out.py")
Bayliss14out  = os.path.join(current_dir, "results/Bayliss14/Bayliss14out.py")

# 20260118 Harikane+25のcitation
Berg25out     = os.path.join(current_dir, "results/Berg25/Berg25out.py")

# === 既存のモジュール読み込み ===
spec1   = importlib.util.spec_from_file_location("Samir16out", Samir16out)
module1 = importlib.util.module_from_spec(spec1)
sys.modules["Samir16out"] = module1
spec1.loader.exec_module(module1)
all_data = module1.all_data 

# --- 追加のモジュール読み込み ---
spec2   = importlib.util.spec_from_file_location("Mingozzi22out", Mingozzi22out)
module2 = importlib.util.module_from_spec(spec2)
sys.modules["Mingozzi22out"] = module2
spec2.loader.exec_module(module2)
all_data.update(module2.all_data)

spec3   = importlib.util.spec_from_file_location("Harikane25out", Harikane25out)
module3 = importlib.util.module_from_spec(spec3)
sys.modules["Harikane25out"] = module3
spec3.loader.exec_module(module3)
all_data.update(module3.all_data)

spec4   = importlib.util.spec_from_file_location("Isobe23out", Isobe23out)
module4 = importlib.util.module_from_spec(spec4)
sys.modules["Isobe23out"] = module4
spec4.loader.exec_module(module4)
all_data.update(module4.all_data)

spec5   = importlib.util.spec_from_file_location("Bunker23out", Bunker23out)
module5 = importlib.util.module_from_spec(spec5)
sys.modules["Bunker23out"] = module5
spec5.loader.exec_module(module5)
all_data.update(module5.all_data)

spec6   = importlib.util.spec_from_file_location("Topping24out", Topping24out)
module6 = importlib.util.module_from_spec(spec6)
sys.modules["Topping24out"] = module6
spec6.loader.exec_module(module6)
all_data.update(module6.all_data)

spec7   = importlib.util.spec_from_file_location("Mainali23out", Mainali23out)
module7 = importlib.util.module_from_spec(spec7)
sys.modules["Mainali23out"] = module7
spec7.loader.exec_module(module7)
all_data.update(module7.all_data)

spec8   = importlib.util.spec_from_file_location("Berg18out", Berg18out)
module8 = importlib.util.module_from_spec(spec8)
sys.modules["Berg18out"] = module8
spec8.loader.exec_module(module8)
all_data.update(module8.all_data)

spec9   = importlib.util.spec_from_file_location("Sanders25out", Sanders25out)
module9 = importlib.util.module_from_spec(spec9)
sys.modules["Sanders25out"] = module9
spec9.loader.exec_module(module9)
all_data.update(module9.all_data)

spec10   = importlib.util.spec_from_file_location("Rigby21out", Rigby21out)
module10 = importlib.util.module_from_spec(spec10)
sys.modules["Rigby21out"] = module10
spec10.loader.exec_module(module10)
all_data.update(module10.all_data)

spec11   = importlib.util.spec_from_file_location("Steidel16out", Steidel16out)
module11 = importlib.util.module_from_spec(spec11)
sys.modules["Steidel16out"] = module11
spec11.loader.exec_module(module11)
all_data.update(module11.all_data)

spec12   = importlib.util.spec_from_file_location("Bayliss14out", Bayliss14out)
module12 = importlib.util.module_from_spec(spec12)
sys.modules["Bayliss14out"] = module12
spec12.loader.exec_module(module12)
all_data.update(module12.all_data)

spec13   = importlib.util.spec_from_file_location("Berg25out", Berg25out)
module13 = importlib.util.module_from_spec(spec13)
sys.modules["Berg25out"] = module13
spec13.loader.exec_module(module13)
all_data.update(module13.all_data)

# === inputファイルから情報を抜き出す ===
Samir16in     = os.path.join(current_dir, "results/Samir16/Samir16in_standard_v3_ms_only_v3_re_no_agn.txt")
Mingozzi22in  = os.path.join(current_dir, "results/Mingozzi22/Mingozzi22in.txt")
Harikane25in  = os.path.join(current_dir, "results/Harikane25/Harikane25in.txt")
Isobe23in     = os.path.join(current_dir, "results/Isobe23/Isobe23in.txt")
Bunker23in    = os.path.join(current_dir, "results/Bunker23/Bunker23in.txt")
Topping24in   = os.path.join(current_dir, "results/Topping24/Topping24in.txt")
Mainali23in   = os.path.join(current_dir, "results/Mainali23/Mainali23in.txt")
Berg18in      = os.path.join(current_dir, "results/Berg18/Berg18in.txt")
Sanders25in   = os.path.join(current_dir, "results/Sanders25/Sanders25in.txt")
Rigby21in     = os.path.join(current_dir, "results/Rigby21/Rigby21in.txt")
Steidel16in   = os.path.join(current_dir, "results/Steidel16/Steidel16in.txt")
Bayliss14in   = os.path.join(current_dir, "results/Bayliss14/Bayliss14in.txt")
Berg25in      = os.path.join(current_dir, "results/Berg25/Berg25in.txt")

# IDの追加
galaxy_ids_Samir16     = []
galaxy_ids_Mingozzi22  = []
galaxy_ids_Harikane25  = []
galaxy_ids_Isobe23     = []
galaxy_ids_Bunker23    = []
galaxy_ids_Topping24   = []
galaxy_ids_Mainali23   = []
galaxy_ids_Berg18      = []
galaxy_ids_Sanders25   = []
galaxy_ids_Rigby21     = []
galaxy_ids_Steidel16   = []
galaxy_ids_Bayliss14   = []
galaxy_ids_Berg25      = []

with open(Samir16in, "r") as f:
    for i, line in enumerate(f):
        parts = line.strip().split()
        if parts:
            galaxy_ids_Samir16.append(parts[0])

with open(Mingozzi22in, "r") as f:
    for i, line in enumerate(f):
        parts = line.strip().split()
        if parts:
            galaxy_ids_Mingozzi22.append(parts[0])

with open(Harikane25in, "r") as f:
    for i, line in enumerate(f):
        parts = line.strip().split()
        if parts:
            galaxy_ids_Harikane25.append(parts[0])

with open(Isobe23in, "r") as f:
    for i, line in enumerate(f):
        parts = line.strip().split()
        if parts:
            galaxy_ids_Isobe23.append(parts[0])

with open(Bunker23in, "r") as f:
    for i, line in enumerate(f):
        parts = line.strip().split()
        if parts:
            galaxy_ids_Bunker23.append(parts[0])

with open(Topping24in, "r") as f:
    for i, line in enumerate(f):
        parts = line.strip().split()
        if parts:
            galaxy_ids_Topping24.append(parts[0])

with open(Mainali23in, "r") as f:
    for i, line in enumerate(f):
        parts = line.strip().split()
        if parts:
            galaxy_ids_Mainali23.append(parts[0])

with open(Berg18in, "r") as f:
    for i, line in enumerate(f):
        parts = line.strip().split()
        if parts:
            galaxy_ids_Berg18.append(parts[0])

with open(Sanders25in, "r") as f:
    for i, line in enumerate(f):
        parts = line.strip().split()
        if parts:
            galaxy_ids_Sanders25.append(parts[0])

with open(Rigby21in, "r") as f:
    for i, line in enumerate(f):
        parts = line.strip().split()
        if parts:
            galaxy_ids_Rigby21.append(parts[0])

with open(Steidel16in, "r") as f:
    for i, line in enumerate(f):
        parts = line.strip().split()
        if parts:
            galaxy_ids_Steidel16.append(parts[0])

with open(Bayliss14in, "r") as f:
    for i, line in enumerate(f):
        parts = line.strip().split()
        if parts:
            galaxy_ids_Bayliss14.append(parts[0])

with open(Berg25in, "r") as f:
    for i, line in enumerate(f):
        parts = line.strip().split()
        if parts:
            galaxy_ids_Berg25.append(parts[0])

data_groups = {
    "data_Samir16"    : galaxy_ids_Samir16,
    "data_Mingozzi22" : galaxy_ids_Mingozzi22,
    "data_Harikane25" : galaxy_ids_Harikane25,
    "data_Isobe23"    : galaxy_ids_Isobe23,
    "data_Bunker23"   : galaxy_ids_Bunker23,
    "data_Topping24"  : galaxy_ids_Topping24,
    "data_Mainali23"  : galaxy_ids_Mainali23,
    "data_Berg18"     : galaxy_ids_Berg18,
    "data_Sanders25"  : galaxy_ids_Sanders25,
    "data_Rigby21"    : galaxy_ids_Rigby21,
    "data_Steidel16"  : galaxy_ids_Steidel16,
    "data_Bayliss14"  : galaxy_ids_Bayliss14,
    "data_Berg25"     : galaxy_ids_Berg25,
}

# マーカーの対応
ne_types = {
    "low": "v",           # ▼
    "intermediate": "s",  # □
    "high": "p",          # ⬟
    "very_high": "o"      # ○
}

# エラーバーの太さ
elinewidths = [0.1, 0.1, 0.1, 0.1]
ecolors = ['gray', 'gray', 'gray', 'gray']

# 色の対応
def get_color(z):
    if z < 1:
        return "gray"
    elif 1 <= z < 4:
        return "blue"
    elif 4 <= z <= 7:
        return "green"
    else:
        return "red"

def get_zorder(z):
    if z < 1:
        return 1
    elif z < 4:
        return 2
    elif z < 7:
        return 3
    else:
        return 4

# 塗りの有無
def get_fill(AGN):
    return True if AGN == 1 else False

# プロット
fig, ax = plt.subplots(figsize=(10, 6))

# x, y, yerr の値を格納するリスト
sm_list = []
ne_list = []
sm_xerr_list = []
ne_yerr_list = []  


# use_only = {"data_Samir16"}

for ref_name, galaxy_list in data_groups.items():
    # if ref_name not in use_only:
    #     continue

    for g_name in galaxy_list:
        g = all_data[g_name]  # ← ここで辞書から取り出す

        color = get_color(g["z"])
        fill = get_fill(g["AGN"])
        edgecolor = color
        facecolor = color if fill else "white"
        z = g["z"]

        # zのフィルタリング
        if z <= 7:
            continue

        # --- x = SM (log10 M*) ---
        x = g["SM"].get("value", np.nan)
        xerr_minus = g["SM"].get("err_minus", np.nan)
        xerr_plus  = g["SM"].get("err_plus", np.nan)

        if x is None or np.isnan(x) or x == 0:
            x = np.nan
            xerr = np.nan
        else:
            if np.isnan(xerr_minus) or np.isnan(xerr_plus):
                xerr = np.nan
            else:
                xerr = 0.5 * (xerr_minus + xerr_plus)  # 対称誤差

        # # x (critical density)のフィルタリング
        # if x < 10.5:
        #     continue

        # --- y = ne (log10 n_e) ---
        for ne_type, ne_info in g["ne_values"].items():
            marker = ne_types[ne_type]

            # OII, SII以外のデータを入れないための一時的な対策
            if ne_type != "low":
                continue   # low 以外は無視


            y = ne_info.get("value", np.nan)
            yerr_minus = ne_info.get("err_minus", np.nan)
            yerr_plus  = ne_info.get("err_plus", np.nan)

            if y is None or np.isnan(y) or y == 0:
                y = np.nan
                yerr = np.nan
            else:
                if np.isnan(yerr_minus) or np.isnan(yerr_plus):
                    yerr = np.nan
                else:
                    yerr = 0.5 * (yerr_minus + yerr_plus)        

            ms = 8     # マーカーサイズ
            mew = 0.8  # マーカーの枠線の太さ

            # プロット
            ax.errorbar(
                x, y,
                xerr=xerr, yerr=yerr,
                fmt=marker,
                markersize=ms,
                markeredgewidth=mew,
                markerfacecolor=facecolor,
                markeredgecolor=edgecolor,
                ecolor=ecolors,
                capsize=0,
                elinewidth=elinewidths,
                linestyle='none',
                zorder=get_zorder(g["z"]),
                label=f"{ref_name}" if g_name == galaxy_list[0] else None
            )

            # ---- 相関計算用にリストに追加 ----
            if not np.isnan(x) and not np.isnan(y):
                sm_list.append(x)
                ne_list.append(y)
                sm_xerr_list.append(xerr)
                ne_yerr_list.append(yerr)

# ---- Marcov Chainにより最適化直線のパラメータの誤差を求める (追加) ----
# 有効分散（effective variance）法を用いてxerrも考慮する

# ============================================================
# --- 収集済みの配列（すべて log10 空間） ---
# ============================================================
x_data = np.array(sm_list)            # log10(M*)
y_data = np.array(ne_list)            # log10(ne)
xerr_data = np.array(sm_xerr_list)    # dex
yerr_data = np.array(ne_yerr_list)    # dex

# ============================================================
# --- データ健全性チェック ---
# ============================================================
print("NaN in xerr_data?", np.any(~np.isfinite(xerr_data)))
print("NaN in yerr_data?", np.any(~np.isfinite(yerr_data)))
print("Zero in xerr_data?", np.any(xerr_data == 0))
print("Zero in yerr_data?", np.any(yerr_data == 0))
print("Inf in xerr_data?", np.any(np.isinf(xerr_data)))
print("Inf in yerr_data?", np.any(np.isinf(yerr_data)))

# ============================================================
# 1) Kendall's tau（誤差は使わない）
# ============================================================
mask_tau = np.isfinite(x_data) & np.isfinite(y_data)

if mask_tau.sum() >= 2:
    tau, p_value = kendalltau(x_data[mask_tau], y_data[mask_tau])
    print(f"Kendall's tau = {tau:.3f}")
    print(f"p-value      = {p_value:.3f}")
else:
    print("[WARN] Kendall: 有効データ不足 (n < 2)")

# ============================================================
# 2) MCMC 用マスク（xerr, yerr 必須）
# ============================================================
mask_mcmc = (
    mask_tau &
    np.isfinite(xerr_data) & (xerr_data > 0) &
    np.isfinite(yerr_data) & (yerr_data > 0)
)

print(f"[DEBUG] Kendall points: {mask_tau.sum()} / {len(x_data)}")
print(f"[DEBUG] MCMC points   : {mask_mcmc.sum()} / {len(x_data)}")

x_m = x_data[mask_mcmc]
y_m = y_data[mask_mcmc]
xerr_m = xerr_data[mask_mcmc]
yerr_m = yerr_data[mask_mcmc]

# ============================================================
# --- 数値安定化のための誤差フロア ---
# ============================================================
xerr_floor = 1e-3
yerr_floor = 1e-3
xerr_m = np.clip(xerr_m, xerr_floor, None)
yerr_m = np.clip(yerr_m, yerr_floor, None)

# ============================================================
# --- 固定する傾き ---
# ============================================================
a_fixed = 0.222   # ← 任意に設定（例：z~0 サンプル）

print(f"[INFO] Fixed slope a = {a_fixed:.3f}")

# ============================================================
# --- 有効分散を用いた対数尤度（b のみ） ---
#     sigma_eff^2 = yerr^2 + (a_fixed * xerr)^2
# ============================================================
def log_likelihood(b, x, y, xerr, yerr, a_fixed):
    y_model = a_fixed * x + b
    sigma2 = yerr**2 + (a_fixed * xerr)**2
    return -0.5 * np.sum(
        (y - y_model)**2 / sigma2 + np.log(sigma2)
    )

# ============================================================
# --- 事前分布（b のみ） ---
# ============================================================
def log_prior(b):
    if -20.0 < b < 20.0:
        return 0.0
    return -np.inf

# ============================================================
# --- 対数事後分布 ---
# ============================================================
def log_posterior(b, x, y, xerr, yerr, a_fixed):
    lp = log_prior(b)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(b, x, y, xerr, yerr, a_fixed)

# ============================================================
# --- MCMC 実行（1次元） ---
# ============================================================
ndim = 1
nwalkers = 50
nsteps = 3000

initial_guess = np.array([0.0])
pos = initial_guess + 1e-4 * np.random.randn(nwalkers, ndim)

if len(x_m) < 2:
    print("[WARN] MCMC: 有効データ不足 (n < 2)")
else:
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_posterior,
        args=(x_m, y_m, xerr_m, yerr_m, a_fixed)
    )

    sampler.run_mcmc(pos, nsteps, progress=True)

    flat_samples = sampler.get_chain(
        discard=500, thin=10, flat=True
    )

    b_mcmc = np.mean(flat_samples)
    b_std  = np.std(flat_samples)

    print(f"MCMC intercept b = {b_mcmc:.3f} ± {b_std:.3f}")
    print(f"{tau:.3f}, {p_value:.3f}, {a_fixed:.3f}, 0.000, {b_mcmc:.3f}, {b_std:.3f}")

# ============================================================
# --- フィットの可視化 ---
# ============================================================
if len(x_m) >= 2:
    x_fit = np.linspace(np.min(x_m), np.max(x_m), 200)
    y_fit = a_fixed * x_fit + b_mcmc

    nsample = min(300, len(flat_samples))
    idx = np.random.randint(len(flat_samples), size=nsample)

    y_fit_samples = np.array([
        a_fixed * x_fit + flat_samples[i]
        for i in idx
    ])

    y_fit_lower = np.percentile(y_fit_samples, 16, axis=0)
    y_fit_upper = np.percentile(y_fit_samples, 84, axis=0)

    plt.plot(x_fit, y_fit, color='black', label='Fixed-slope fit')
    plt.fill_between(
        x_fit, y_fit_lower, y_fit_upper,
        color='black', alpha=0.2, label='1σ interval'
    )



# 2種類以上の輝線でneが求められている天体は、縦線でつなぐ
# Mingozzi+22
plt.axvline(x=np.log10(44668359),  color='gray', linestyle=':', alpha=0.3)
plt.axvline(x=np.log10(575439937), color='gray', linestyle=':', alpha=0.3)
plt.axvline(x=np.log10(204173790), color='gray', linestyle=':', alpha=0.3)
plt.axvline(x=np.log10(630957344), color='gray', linestyle=':', alpha=0.3)
plt.axvline(x=np.log10(33113112),  color='gray', linestyle=':', alpha=0.3)
# Bunker+23
plt.axvline(x=np.log10(537031796), color='gray', linestyle=':', alpha=0.3)
# Topping+24
plt.axvline(x=np.log10(112201845), color='gray', linestyle=':', alpha=0.3)
# Mainali+23
plt.axvline(x=np.log10(98107171), color='gray', linestyle=':', alpha=0.3)
# SIIのcritical densityを表示する (6716, 6731)
plt.axhline(y=np.log10(1917.5607046610592),  color='tab:red', linestyle='-', alpha=0.7)
plt.axhline(y=np.log10(5067.434274587508),  color='tab:red', linestyle='-', alpha=0.7)
nc_6716 = np.log10(1917.5607046610592)
nc_6731 = np.log10(5067.434274587508)
plt.text(x=6+0.1, y=nc_6716+0.1, s=r"$n_{\mathrm{crit}}$([SII]6716)", fontsize=12,)
plt.text(x=6+0.1, y=nc_6731+0.1, s=r"$n_{\mathrm{crit}}$([SII]6731)", fontsize=12,)
plt.xlim(5, 12)
plt.ylim(0, 6)
ax.set_xlabel(r"$\log(M_\ast/M_\odot)$", fontsize=16)
ax.set_ylabel(r"$log(n_e) [\mathrm{cm^{-3}}]$", fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=14, direction='in')
plt.tick_params(axis='both', which='minor', labelsize=14, direction='in')
plt.tight_layout()
plt.show()

# # SDSSのデータのmass histgramを描画する
# plt.figure(figsize=(12, 6))
# sns.histplot(sm_list, bins=30, kde=True, color="C0")
# plt.xlabel(r"$\log(M_\ast/M_\odot)$")
# plt.text(x=10+0.1, y=250, s="critical density", fontsize=12)
# plt.axvline(x=10, color="black")
# plt.title("SDSS data")
# plt.savefig(os.path.join(current_dir, "results/figure/sm_histogram_sdss_ms.png"))
# plt.show()

# # SDSSのデータに関して、yがnp.log10(1917.5607046610592)（SII6716のncより大きい）ものの割合を調べる
# # →8.35 %
# def percent_ge_numpy(values, threshold, decimals=2):
#     arr = np.asarray(values, dtype=float)

#     # NaN を除外
#     valid = ~np.isnan(arr)
#     if valid.sum() == 0:
#         return 0.0

#     pct = 100.0 * (arr[valid] >= threshold).mean()
#     return round(pct, decimals)

# print(percent_ge_numpy(ne_list, nc_6716))  # しきい値以上の割合（%）

# Monitor memory usage
process = psutil.Process()
mem_info_before = process.memory_info().rss / 1024**3 # in GB
print(f"Memory usage before processing: {mem_info_before:.2f} GB")
print(np.min(ne_list), np.max(ne_list))
print(np.min(sm_list), np.max(sm_list))
# print(np.min(x_data), np.max(x_data))