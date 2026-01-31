#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
スクリプトの概要:
このスクリプトはne_vs_12+log(O/H)の図を描画します。

使用方法:
    ne_vs_metallicity_plot_sdss_ms.oy [オプション]

著者: A. M.
作成日: 2026-01-22

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
Samir16out    = os.path.join(current_dir, "results/Samir16/Samir16out_standard_v5.py")
Mingozzi22out = os.path.join(current_dir, "results/Mingozzi22/Mingozzi22out.py")
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

# # Curti+25のデータ (z~2)
Rigby21out    = os.path.join(current_dir, "results/Rigby21/Rigby21out.py")
Steidel16out  = os.path.join(current_dir, "results/Steidel16/Steidel16out.py")
Bayliss14out  = os.path.join(current_dir, "results/Bayliss14/Bayliss14out.py")

# === 既存のモジュール読み込み ===
spec1 = importlib.util.spec_from_file_location("Samir16out", Samir16out)
module1 = importlib.util.module_from_spec(spec1)
sys.modules["Samir16out"] = module1
spec1.loader.exec_module(module1)
all_data = module1.all_data 

# --- 追加のモジュール読み込み ---
spec2 = importlib.util.spec_from_file_location("Mingozzi22out", Mingozzi22out)
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

spec11   = importlib.util.spec_from_file_location("Steidelout", Steidel16out)
module11 = importlib.util.module_from_spec(spec11)
sys.modules["Steidel16out"] = module11
spec11.loader.exec_module(module11)
all_data.update(module11.all_data)

spec12   = importlib.util.spec_from_file_location("Baylissout", Bayliss14out)
module12 = importlib.util.module_from_spec(spec12)
sys.modules["Bayliss14out"] = module12
spec12.loader.exec_module(module12)
all_data.update(module12.all_data)

# === inputファイルから情報を抜き出す ===
Samir16in     = os.path.join(current_dir, "results/Samir16/Samir16in_standard_v5.txt")
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

with open(Samir16in, "r") as f:
    for i, line in enumerate(f):
        if i >= 10000: # まずは10000個
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
ne_yerr_list = []  


for ref_name, galaxy_list in data_groups.items():
    for g_name in galaxy_list:
        g = all_data[g_name]

        color = get_color(g["z"])
        fill = get_fill(g["AGN"])
        edgecolor = color
        facecolor = color if fill else "white"

        # --- x = 12+log(O/H)  ---
        x = g["metal"].get("value", np.nan)
        xerr_minus = g["metal"].get("err_minus", np.nan)
        xerr_plus  = g["metal"].get("err_plus", np.nan)

        if x is None or np.isnan(x) or x == 0 or x < -90: # 極端な値の除去
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

            y = ne_info.get("value", np.nan)
            yerr_minus = ne_info.get("err_minus", np.nan)
            yerr_plus  = ne_info.get("err_plus", np.nan)

            if y is None or np.isnan(y) or y == 0 : # 極端な値の除去
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
            ne_yerr_list.append(yerr)


# 2種類以上の輝線でneが求められている天体は、縦線でつなぐ
# Mingozzi+22
plt.axvline(x=6.98, color='gray', linestyle=':', alpha=0.3)
plt.axvline(x=7.58, color='gray', linestyle=':', alpha=0.3)
plt.axvline(x=7.75, color='gray', linestyle=':', alpha=0.3)
plt.axvline(x=7.76, color='gray', linestyle=':', alpha=0.3)
plt.axvline(x=8.10, color='gray', linestyle=':', alpha=0.3)
# # Bunker+23
# plt.axvline(x=np.log10(537031796), color='gray', linestyle=':', alpha=0.3)
# # Topping+24
# plt.axvline(x=np.log10(112201845), color='gray', linestyle=':', alpha=0.3)
# # Mainali+23
# plt.axvline(x=np.log10(98107171),  color='gray', linestyle=':', alpha=0.3)

# SIIのcritical densityを表示する (6716, 6731)
plt.axhline(y=np.log10(1917.5607046610592), color='tab:red', linestyle='-', alpha=0.7)
plt.axhline(y=np.log10(5067.434274587508),  color='tab:red', linestyle='-', alpha=0.7)
plt.text(x=7+0.1, y=np.log10(1917.5607046610592)+0.1, s=r"$n_{\mathrm{crit}}$([SII]6716)")
plt.text(x=7+0.1, y=np.log10(5067.434274587508)+0.1,  s=r"$n_{\mathrm{crit}}$([SII]6731)")

# CLASSYのデータのみで線形回帰した時の直線を表示する
# MCMC slope (a): 0.050 ± 0.011
# MCMC intercept (b): 1.864 ± 0.090
# slope_classy = 0.050
# intercept_classy = 1.864
# x_range = np.linspace(6, 12, 1000)
# y_range = slope_classy * x_range + intercept_classy
# plt.plot(x_range, y_range, color='black')
# plt.text(x=6+0.1, y=intercept_classy + 0.1,  s="best fit (CLASSY)")

# plt.xlim(6, 12)
plt.ylim(0, 6)
ax.set_xlabel(r"$12+\log(O/H)$")
ax.set_ylabel(r"$\log(n_e) [\mathrm{cm^{-3}}]$")
# === 枠線 (spines) の設定 ===
# 線の太さ・色・表示非表示などを個別に制御
for spine in ax.spines.values():
    spine.set_linewidth(2)       # 枠線の太さ
    spine.set_color("black")     # 枠線の色
plt.tight_layout()
plt.savefig(os.path.join(current_dir, "results/figure/ne_vs_metallicity_plot.png"))
plt.show()

# Monitor memory usage
process = psutil.Process()
mem_info_before = process.memory_info().rss / 1024**3 # in GB
print(f"Memory usage before processing: {mem_info_before:.2f} GB")