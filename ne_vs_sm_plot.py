#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
スクリプトの概要:
このスクリプトはne_vs_smの図を描画します。

使用方法:
    ne_vs_sm_plot.py [オプション]

著者: A. M.
作成日: 2026-02-07

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
import pandas as pd
import sys
import psutil
import numpy as np
import seaborn as sns

# 軸の設定
plt.rcParams.update({
    # --- 図全体 ---
    "figure.figsize": (12, 6),       # 図サイズ
    "font.size": 20,                 # 全体フォントサイズ
    "axes.labelsize": 20,            # 軸ラベルのサイズ
    "axes.titlesize": 20,            # タイトルのサイズ
    "axes.grid": False,              # グリッドOFF

    # --- 目盛り設定 (ticks) ---
    "xtick.direction": "in",         # x軸目盛りの向き
    "ytick.direction": "in",         # y軸目盛りの向き
    "xtick.top": True,               # 上にも目盛り
    "ytick.right": True,             # 右にも目盛り

    # 主目盛り（major ticks）
    "xtick.major.size": 20,          # 長さ
    "ytick.major.size": 20,
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
    "xtick.labelsize": 20,           # x軸ラベルサイズ
    "ytick.labelsize": 20,           # y軸ラベルサイズ

    # --- フォント ---
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
})

# === ファイルパスを変数として格納 ===
current_dir = os.getcwd()
Samir16out     = os.path.join(current_dir, "results/Samir16/Samir16out_standard_v6.py")
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
Samir16in     = os.path.join(current_dir, "results/Samir16/Samir16in_standard_v6.txt")
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
        if i >= 1000: # 現時点でまだSDSSのmetallicityの情報は載せていない
            break
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
# ne_types = {
#     "low": "v",           # ▼
#     "intermediate": "s",  # □
#     "high": "p",          # ⬟
#     "very_high": "o"      # ○
# }

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
fig, ax = plt.subplots(figsize=(6, 6))
fig.subplots_adjust(left=0.20, right=0.95, top=0.95, bottom=0.15)

# x, y, yerr の値を格納するリスト
sm_list = []
ne_list = []
ne_yerr_list = []  


# z>6のデータを格納するリスト
z6_x_vals = []
z6_y_vals = []
z6_x_errs = []
z6_y_errs = []


sdss = {"data_Samir16"}
for ref_name, galaxy_list in data_groups.items():

    for g_name in galaxy_list:
        g = all_data[g_name]
        z = g["z"]
        AGN = g["AGN"]
        if "main_sequence" in g:
            main_sequence = g["main_sequence"]
        else:
            main_sequence = None  # or np.nan

        # high-zのデータのみをプロット
        if z < 6:
            continue
        # z > 1 のデータはプロットしない（現時点では）
        # if z > 1:
        #     continue  
        # # SDSSのデータ以外はプロットしない
        # if ref_name not in sdss:
        #     continue

        # 色の対応
        def get_color(z):
            if z < 1:
                if ref_name in sdss:
                    return "gray"
                else:
                    return "black"
                # if main_sequence == 1:
                #     return "black"
                # else:
                #     return "gray"
                # return "gray" # ひとまずは色を変えない
            elif 1 <= z < 4:
                return "tab:blue"
            elif 4 <= z <= 7:
                return "tab:green"
            else:
                return "tab:red"
            
        # 薄さの対応
        def get_alpha(z):
            if z < 1:
                if ref_name in sdss:
                    return 0.5
                else:
                    return 1
            elif 1 <= z < 4:
                return 1
            elif 4 <= z <= 7:
                return 1
            else:
                return 1

        # エラーバーの色
        def get_ecolors(z):
            # if z < 1:
            #     if ref_name in sdss:
            #         return "gray"
            #     else:
            #         return "black"
            # elif 1 <= z < 4:
            #     return 'tab:blue'
            # elif 4 <= z <= 7:
            #     return 'tab:green'
            # else:
            #     return 'tab:red'
            return "tab:purple"

        # エラーバーの太さ
        def get_elinewidth(z):
            if z < 1:
                if ref_name in sdss:
                    return 0.0
                else:
                    # return 0.25
                    return 0.25
            elif 1 <= z < 4:
                return 0.25
            elif 4 <= z <= 7:
                return 0.25
            else:
                return 0.25
        
        # markerをlow-zかどうかで変える
        if z <= 1:
            if ref_name in sdss:
                ne_types = {
                    "low": ".",           # 点
                    "intermediate": "s",  # □
                    "high": "p",          # ⬟
                    "very_high": "o"      # ○
                }
            else:
                ne_types = {
                    "low": "x",           # 点
                    "intermediate": "s",  # □
                    "high": "p",          # ⬟
                    "very_high": "o"      # ○
                }
        else:
            ne_types = {
                "low": "o",           # 点
                "intermediate": "s",  # □
                "high": "p",          # ⬟
                "very_high": "o"      # ○
            }

        color = get_color(z)
        fill = get_fill(AGN)
        edgecolor = color
        facecolor = color if (fill or z < 1) else "white"

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

            if z <= 1:
                if ref_name in sdss:
                    ms = 0.5
                else:
                    ms = 4
            else: 
                ms = 4
            # ax.scatter(x, y, s=0.01, alpha=0.5, rasterized=True, color='gray', marker='.')
            mew = 0.8  # マーカーの枠線の太さ

            # プロット
            ax.errorbar(
                x, y,
                xerr=xerr, yerr=yerr,
                fmt=marker,
                markersize=ms,
                markeredgewidth=mew,
                # markerfacecolor=facecolor,
                # markeredgecolor=edgecolor,
                markerfacecolor="tab:purple",
                markeredgecolor="tab:purple",
                # markerfacecolor="gray",
                # markeredgecolor="gray",
                ecolor=get_ecolors(z),
                capsize=0,
                lw=get_elinewidth(z),
                linestyle='none',
                zorder=get_zorder(z),
                label=f"{ref_name}" if g_name == galaxy_list[0] else None,
                alpha=get_alpha(z),
            )



        # ---- 相関計算用にリストに追加 ----
        if not np.isnan(x) and not np.isnan(y):
            sm_list.append(x)
            ne_list.append(y)
            ne_yerr_list.append(yerr)


        # ===== z>6 データを保存 =====
        if np.isfinite(x) and np.isfinite(y) and np.isfinite(yerr) and yerr > 0:
            z6_x_vals.append(x)
            z6_y_vals.append(y)
            z6_x_errs.append(xerr if np.isfinite(xerr) else np.nan)
            z6_y_errs.append(yerr)


# =====================================
# z>6 平均値（誤差考慮）を計算
# =====================================

z6_x_vals = np.array(z6_x_vals)
z6_y_vals = np.array(z6_y_vals)
z6_y_errs = np.array(z6_y_errs)

if len(z6_y_vals) > 0:

    # 重み付き平均（1/σ²）
    weights = 1.0 / z6_y_errs**2
    y_mean = np.sum(weights * z6_y_vals) / np.sum(weights)
    y_mean_err = np.sqrt(1.0 / np.sum(weights))

    # xは単純平均（必要なら重み付きにもできる）
    x_mean = np.nanmean(z6_x_vals)

    # =========================
    # 大きくプロット
    # =========================
    ax.errorbar(
        x_mean,
        y_mean,
        yerr=y_mean_err,
        fmt="s",               # ダイヤマーカー
        markersize=10,         # 大きく
        markeredgewidth=1.5,
        markerfacecolor="tab:purple",
        markeredgecolor="tab:purple",
        ecolor="tab:purple",
        capsize=4,
        zorder=100,
        label="z>6 weighted mean"
    )


# # 2種類以上の輝線でneが求められている天体は、縦線でつなぐ
# # Mingozzi+22
# plt.axvline(x=np.log10(44668359),  color='gray', linestyle=':', alpha=0.3)
# plt.axvline(x=np.log10(575439937), color='gray', linestyle=':', alpha=0.3)
# plt.axvline(x=np.log10(204173790), color='gray', linestyle=':', alpha=0.3)
# plt.axvline(x=np.log10(630957344), color='gray', linestyle=':', alpha=0.3)
# plt.axvline(x=np.log10(33113112),  color='gray', linestyle=':', alpha=0.3)
# # Bunker+23
# plt.axvline(x=np.log10(537031796), color='gray', linestyle=':', alpha=0.3)
# # Topping+24
# plt.axvline(x=np.log10(112201845), color='gray', linestyle=':', alpha=0.3)
# # Mainali+23
# plt.axvline(x=np.log10(98107171),  color='gray', linestyle=':', alpha=0.3)

# # SIIのcritical densityを表示する (6716, 6731)
# # 値の定義
# nc_6716 = np.log10(1917.5607046610592)
# nc_6731 = np.log10(5067.434274587508)  # 使わないなら消してOK

# x = np.linspace(6, 7, 400)
# y = np.full_like(x, nc_6716)  # ★ 定数をxと同じ長さの配列にする
# T = 6  # 閾値（しきい値）
# # 曲線（水平線）
# ax.plot(x, y, color="gray", linestyle="-.", linewidth=1)
# # しきい値の水平線
# ax.axhline(T, color="gray", linestyle="-.", linewidth=1)
# plt.text(x=6+0.1, y=np.log10(1917.5607046610592)+0.1, s=r"$n_{\mathrm{crit}}$([SII]6716)", fontsize=16,)

 
# # CLASSYのデータのみで線形回帰した時の直線を表示する
# slope_classy = 0.013
# intercept_classy = 2.187
# x_range = np.linspace(6, 12, 1000)
# y_range = slope_classy * x_range + intercept_classy
# plt.plot(x_range, y_range, color='black', linestyle='-.')

# # === 推定結果（あなたの値に置き換え） ===
# # slope（固定）
# m_hat_z0   = 0.230

# # intercept
# b_hat_z0   = 0.006
# b_hat_z3   = 0.367
# b_hat_z6   = 0.803
# b_hat_z9   = 1.043

# # slope の標準誤差
# sigma_m_z0 = 0.004
# sigma_m_z3 = 0.000  
# sigma_m_z6 = 0.000  
# sigma_m_z9 = 0.000 

# # intercept の標準誤差 
# sigma_b_z0 = 0.044 
# sigma_b_z3 = 0.019 
# sigma_b_z6 = 0.050 
# sigma_b_z9 = 0.128 

# # slope と intercept の相関（例：不明なら 0）
# rho_mb_z0  = 0    
# rho_mb_z3  = 0    
# rho_mb_z6  = 0    
# rho_mb_z9  = 0    

# # x 範囲
# x = np.linspace(6, 12, 1000)
# # 推定直線
# y_hat_z0 = m_hat_z0 * x + b_hat_z0
# y_hat_z3 = m_hat_z0 * x + b_hat_z3
# y_hat_z6 = m_hat_z0 * x + b_hat_z6
# y_hat_z9 = m_hat_z0 * x + b_hat_z9
# # パラメータ不確かさ由来の y の標準偏差
# cov_mb_z0 = rho_mb_z0 * sigma_m_z0 * sigma_b_z0
# cov_mb_z3 = rho_mb_z3 * sigma_m_z3 * sigma_b_z3
# cov_mb_z6 = rho_mb_z6 * sigma_m_z6 * sigma_b_z6
# cov_mb_z9 = rho_mb_z9 * sigma_m_z9 * sigma_b_z9
# sigma_y_z0 = np.sqrt((x * sigma_m_z0)**2 + sigma_b_z0**2 + 2 * x * cov_mb_z0)
# sigma_y_z3 = np.sqrt((x * sigma_m_z3)**2 + sigma_b_z3**2 + 2 * x * cov_mb_z3)
# sigma_y_z6 = np.sqrt((x * sigma_m_z6)**2 + sigma_b_z6**2 + 2 * x * cov_mb_z6)
# sigma_y_z9 = np.sqrt((x * sigma_m_z9)**2 + sigma_b_z9**2 + 2 * x * cov_mb_z9)
# # 信頼水準（k=1 なら約68%, k=1.96 なら約95%）
# k = 1
# lower_z0 = y_hat_z0 - k * sigma_y_z0
# lower_z3 = y_hat_z3 - k * sigma_y_z3
# lower_z6 = y_hat_z6 - k * sigma_y_z6
# lower_z9 = y_hat_z9 - k * sigma_y_z9
# upper_z0 = y_hat_z0 + k * sigma_y_z0
# upper_z3 = y_hat_z3 + k * sigma_y_z3
# upper_z6 = y_hat_z6 + k * sigma_y_z6
# upper_z9 = y_hat_z9 + k * sigma_y_z9
# # ax.plot(x, y_hat_z0, color='black')
# ax.plot(x, y_hat_z3, color='tab:blue')
# ax.plot(x, y_hat_z6, color='tab:green')
# ax.plot(x, y_hat_z9, color='tab:red')
# # ax.fill_between(x, lower_z0, upper_z0, color='gray' , alpha=0.15)
# ax.fill_between(x, lower_z3, upper_z3, color='tab:blue' , alpha=0.05)
# ax.fill_between(x, lower_z6, upper_z6, color='tab:green', alpha=0.05)
# ax.fill_between(x, lower_z9, upper_z9, color='tab:red'  , alpha=0.05)

# # 変わりに回帰分析をした時に得るを使う
# band = pd.read_csv(os.path.join(current_dir, "results/csv/ne_vs_sm_regression_band_.csv"))

# plt.plot(
#     band["x"],
#     band["y_med"],
#     color="black",
#     lw=2,
#     label="MCMC best-fit"
# )

# plt.fill_between(
#     band["x"],
#     band["y_low"],
#     band["y_high"],
#     color="black",
#     alpha=0.15,
# )













# =============================================
# SDSSのstackデータ（Massビンごと、データ点）をプロットする 
# =============================================

# ===== 入出力 =====
in_csv_data  = os.path.join(current_dir, "results/csv/stacked_sii_ne_vs_mass_from_ratio_COMPLETE_v1.csv")

# ===== 読み込み =====
res_data = pd.read_csv(in_csv_data)

# ===== 必要列 =====
x_data = res_data["logM_cen"].to_numpy(float)
y_data = res_data["log_ne_med"].to_numpy(float)
yerr_lo_data = res_data["log_ne_err_lo"].to_numpy(float)
yerr_hi_data = res_data["log_ne_err_hi"].to_numpy(float)

# outsideフラグ（なければ全部False）
if "R_outside" in res_data.columns:
    outside_data = res_data["R_outside"].to_numpy(bool)
else:
    outside_data = np.zeros_like(x_data, dtype=bool)

# ===== 有効値マスク =====
m_ok_data = (
    np.isfinite(x_data) &
    np.isfinite(y_data) &
    np.isfinite(yerr_lo_data) &
    np.isfinite(yerr_hi_data) &
    ((yerr_lo_data > 0) | (yerr_hi_data > 0))  # ← 修正
)

# 完全サンプル閾値
thr_data = 10.0
mask_mass = x_data > thr_data

# 有効かつmass条件
base_mask = m_ok_data & mask_mass

# =============================================
# 誤差タイプ分類
# =============================================
is_lower_limit = base_mask & (yerr_lo_data == 0) & (yerr_hi_data > 0)
is_upper_limit = base_mask & (yerr_hi_data == 0) & (yerr_lo_data > 0)
is_normal = base_mask & ~(is_lower_limit | is_upper_limit)

# =============================================
# 描画
# =============================================

arrow_length = 0.15  # dex

# --- 通常点（両側誤差） ---
ax.errorbar(
    x_data[is_normal],
    y_data[is_normal],
    yerr=[yerr_lo_data[is_normal], yerr_hi_data[is_normal]],
    fmt="s",
    mec="k",
    mfc="k",
    ecolor="k",
    color="k",
    capsize=3
)

# # --- lower limit（↑）---
# ax.errorbar(
#     x_data[is_lower_limit],
#     y_data[is_lower_limit],
#     yerr=arrow_length,
#     fmt="s",
#     color="k",
#     lolims=True,
#     capsize=4
# )
# # --- upper limit（↓）---
# ax.errorbar(
#     x_data[is_upper_limit],
#     y_data[is_upper_limit],
#     yerr=arrow_length,
#     fmt="s",
#     color="k",
#     uplims=True,
#     capsize=4
# )








# # =============================================
# # SDSS stackデータ（massビンごと）プロット
# # 両側に誤差があるものだけ描画
# # =============================================

# # ===== 入出力 =====
# in_csv_data = os.path.join(
#     current_dir,
#     "results/csv/stacked_sii_ne_vs_ssfr_from_ratio_COMPLETE_v1.csv"
# )

# # ===== 読み込み =====
# res_data = pd.read_csv(in_csv_data)

# # ===== 必要列 =====
# x_data = res_data["logsSFR_cen"].to_numpy(float)
# y_data = res_data["log_ne_med"].to_numpy(float)
# yerr_lo_data = res_data["log_ne_err_lo"].to_numpy(float)
# yerr_hi_data = res_data["log_ne_err_hi"].to_numpy(float)

# # ===== 完全サンプル閾値 =====
# thr_data = -11.0
# mask_mass = x_data > thr_data

# # ===== 両側誤差があるものだけ残す =====
# eps = 1e-8

# mask_valid = (
#     np.isfinite(x_data) &
#     np.isfinite(y_data) &
#     np.isfinite(yerr_lo_data) &
#     np.isfinite(yerr_hi_data) &
#     (yerr_lo_data > eps) &
#     (yerr_hi_data > eps) &
#     mask_mass
# )

# # =============================================
# # 描画（通常の両側エラーバーのみ）
# # =============================================

# ax.errorbar(
#     x_data[mask_valid],
#     y_data[mask_valid],
#     yerr=[yerr_lo_data[mask_valid], yerr_hi_data[mask_valid]],
#     fmt="s",
#     mec="k",
#     mfc="k",
#     ecolor="k",
#     color="k",
#     capsize=3
# )

# # x < 10（白四角・黒縁）
# ax.errorbar(
#     x[mask_ge], y[mask_ge],
#     yerr=yerr[:, mask_ge],
#     fmt="s", mec="gray", mfc="white",
#     ecolor="gray", color="gray",  # 誤差線色/線色（同時指定）
#     capsize=3
# )

# # stackの回帰分析結果もプロットする
# band_stacked = pd.read_csv(os.path.join(current_dir, "results/csv/stacked_ne_vs_sm_regression_band_COMPLETE_v1.csv"))

# plt.plot(
#     band_stacked["x"],
#     band_stacked["y_med"],
#     color="black",
#     lw=2,
# )

# plt.fill_between(
#     band_stacked["x"],
#     band_stacked["y_low"],
#     band_stacked["y_high"],
#     color="black",
#     alpha=0.15,
# )


# =============================================
# SDSSのstackデータ（Massビンごと、データ点）をプロットする 
# =============================================
# ===== 入出力 =====
in_csv_data  = os.path.join(current_dir, "results/csv/stacked_sii_ne_vs_mass_from_ratio_data.csv")

# ===== 読み込み =====
res_data = pd.read_csv(in_csv_data)

# ===== 必要列を取り出し =====
x_data = res_data["logM_cen"].to_numpy(float)

y_data = res_data["log_ne_med"].to_numpy(float)
yerr_lo_data = res_data["log_ne_err_lo"].to_numpy(float)
yerr_hi_data = res_data["log_ne_err_hi"].to_numpy(float)

# outsideフラグ（なければ全部False）
if "R_outside" in res_data.columns:
    outside_data = res_data["R_outside"].to_numpy(bool)
else:
    outside_data = np.zeros_like(x_data, dtype=bool)

# ===== 有効値マスク（NaN/inf除外）=====
m_ok_data = (
    np.isfinite(x_data) &
    np.isfinite(y_data) &
    np.isfinite(yerr_lo_data) &
    np.isfinite(yerr_hi_data) &
    ((yerr_lo_data > 0) | (yerr_hi_data > 0))  # ← 修正
)

# stack結果（完全なものとそうでないものの色を分ける）
thr_data = 10.0

yerr_data = np.vstack([res_data["log_ne_err_lo"].values, res_data["log_ne_err_hi"].values])

mask_lt_data = x_data > thr_data
mask_ge_data = ~mask_lt_data

# # x >= 10（黒四角）
# ax.errorbar(
#     x_data[mask_lt_data], y_data[mask_lt_data],
#     yerr=yerr_data[:, mask_lt_data],
#     fmt="s", mec="gray", mfc="gray",
#     ecolor="gray", color="gray",  # 誤差線色/線色（同時指定）
#     capsize=3
# )



# # x < 10（白四角・黒縁）
# ax.errorbar(
#     x_data[mask_ge_data], y_data[mask_ge_data],
#     yerr=yerr_data[:, mask_ge_data],
#     fmt="s", mec="black", mfc="white",
#     ecolor="k", color="k",
#     capsize=3
# )












# stackの回帰分析結果もプロットする
band_stacked = pd.read_csv(os.path.join(current_dir, "results/csv/stacked_ne_vs_sm_regression_band_COMPLETE_v1.csv"))

plt.plot(
    band_stacked["x"],
    band_stacked["y_med"],
    color="black",
    lw=2,
)

plt.fill_between(
    band_stacked["x"],
    band_stacked["y_low"],
    band_stacked["y_high"],
    color="black",
    alpha=0.15,
)













# =================================================
# JADES
# =================================================
in_csv_jades = "./results/csv/stacked_sii_ne_vs_mass_from_ratio_JADES_DR3.csv"
res = pd.read_csv(in_csv_jades)

x   = res["logM_cen"].to_numpy(float)
y   = res["log_ne_med"].to_numpy(float)
elo = res["log_ne_err_lo"].to_numpy(float)
ehi = res["log_ne_err_hi"].to_numpy(float)
zbin = res["z_bin"].values

# ---- 両側誤差がある点のみ ----
m_twoside = (
    np.isfinite(x) &
    np.isfinite(y) &
    np.isfinite(elo) &
    np.isfinite(ehi)
)

# z-bin
# mask_2 = (zbin == "0.5<z<2.0") & m_twoside
mask_2 = (zbin == "0.5<z<2.0") # 2はoneside
mask_3 = (zbin == "2.0<z<3.0") & m_twoside
mask_6 = (zbin == "3.0<z<6.0") & m_twoside

# =================================================
# 片側 limit 描画（1σ,2σ,3σ）（追加）
# =================================================

arrow_len = 0.15  # 見た目用（dex）

# limit列読み込み
log_ne_1_lo = res["log_ne_1sig_lo_limit"].to_numpy(float)
log_ne_1_hi = res["log_ne_1sig_hi_limit"].to_numpy(float)

log_ne_2_lo = res["log_ne_2sig_lo_limit"].to_numpy(float)
log_ne_2_hi = res["log_ne_2sig_hi_limit"].to_numpy(float)

log_ne_3_lo = res["log_ne_3sig_lo_limit"].to_numpy(float)
log_ne_3_hi = res["log_ne_3sig_hi_limit"].to_numpy(float)

def plot_limits(mask, xarr, ylimit, color, sigma_label, uplim=False):
    """
    uplim=True  → 上向き矢印（lower limit）
    uplim=False → 下向き矢印（upper limit）
    """
    m = mask & np.isfinite(ylimit)

    if np.any(m):
        ax.errorbar(
            xarr[m],
            ylimit[m],
            yerr=arrow_len,
            fmt="s",
            color=color,
            uplims=uplim,
            lolims=not uplim,
            ms=10,
            alpha=0.8,
            capsize=4,
        )

# =============================
# 各z binごとに描画
# =============================

# 全て描きたい場合
# for zb, mask_z, color in [
#     ("0.5<z<2.0", mask_2, "tab:blue"),
#     ("2.0<z<3.0", mask_3, "tab:green"),
#     ("3.0<z<6.0", mask_6, "tab:red"),
# ]:

#     # 1σ
#     plot_limits(mask_z, x, log_ne_1_lo, color, "1σ", uplim=False)  # upper limit
#     plot_limits(mask_z, x, log_ne_1_hi, color, "1σ", uplim=True)   # lower limi
#     # 2σ
#     plot_limits(mask_z, x, log_ne_2_lo, color, "2σ", uplim=False)
#     plot_limits(mask_z, x, log_ne_2_hi, color, "2σ", uplim=True)
#     # 3σ
#     plot_limits(mask_z, x, log_ne_3_lo, color, "3σ", uplim=False)
#     plot_limits(mask_z, x, log_ne_3_hi, color, "3σ", uplim=True)

# 3σ upper limit のみ
mask_2 = (zbin == "0.5<z<2.0") 
mask_2_2 = (zbin == "0.5<z<2.0") & (m_twoside==False) # 変更
plot_limits(
    mask_2_2, 
    x,
    log_ne_3_hi,
    color="tab:blue",
    sigma_label="3σ",
    uplim=True  # upper limit（下向き矢印）
)

# Ensure error values are non-negative
elo_jades_mask_2 = np.maximum(0, elo[mask_2])
ehi_jades_mask_2 = np.maximum(0, ehi[mask_2])
elo_jades_mask_3 = np.maximum(0, elo[mask_3])
ehi_jades_mask_3 = np.maximum(0, ehi[mask_3])
elo_jades_mask_6 = np.maximum(0, elo[mask_6])
ehi_jades_mask_6 = np.maximum(0, ehi[mask_6])


# 0.5<z<2.0
ax.errorbar(
    x[mask_2], y[mask_2],
    yerr=np.vstack([elo_jades_mask_2, ehi_jades_mask_2]),
    fmt="s",
    mfc="tab:blue", mec="tab:blue",
    ecolor="tab:blue", color="tab:blue",
    ms=10,
    capsize=3,
)

# 2.0<z<3.0
ax.errorbar(
    x[mask_3], y[mask_3],
    yerr=np.vstack([elo_jades_mask_3, ehi_jades_mask_3]),
    fmt="s",
    mfc="tab:green", mec="tab:green",
    ecolor="tab:green", color="tab:green",
    ms=10,
    capsize=3,
)

# 3.0<z<6.0
ax.errorbar(
    x[mask_6], y[mask_6],
    yerr=np.vstack([elo_jades_mask_6, ehi_jades_mask_6]),
    fmt="s",
    mfc="tab:red", mec="tab:red",
    ecolor="tab:red", color="tab:red",
    ms=10,
    capsize=3,
)








# =============================================
# 全体のプロットの設定
# =============================================
plt.xlim(10, 12)
plt.ylim(1.80, 3.3)
ax.set_xlabel(r"$\log(M_\ast/M_\odot)$")
ax.set_ylabel(r"$\log(n_e) [\mathrm{cm^{-3}}]$")
# === 枠線 (spines) の設定 ===
# 線の太さ・色・表示非表示などを個別に制御
for spine in ax.spines.values():
    spine.set_linewidth(2)       # 枠線の太さ
    spine.set_color("black")     # 枠線の色
plt.tight_layout()
plt.savefig(os.path.join(current_dir, "results/figure/ne_vs_sm_plot_v6_highz_slide.png"))
plt.show()

# Monitor memory usage
process = psutil.Process()
mem_info_before = process.memory_info().rss / 1024**3 # in GB
print(f"Memory usage before processing: {mem_info_before:.2f} GB")