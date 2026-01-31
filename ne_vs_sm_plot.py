#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
スクリプトの概要:
このスクリプトはne_vs_smの図を描画します。

使用方法:
    ne_vs_sm_plot_sdss_ms.py [オプション]

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
Samir16out     = os.path.join(current_dir, "results/Samir16/Samir16out_standard_v3_ms_only.py")
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
Samir16in     = os.path.join(current_dir, "results/Samir16/Samir16in_standard_v3_ms_only_v1.txt")
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
ne_yerr_list = []  


for ref_name, galaxy_list in data_groups.items():
    for g_name in galaxy_list:
        g = all_data[g_name]

        color = get_color(g["z"])
        fill = get_fill(g["AGN"])
        edgecolor = color
        facecolor = color if fill else "white"

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
            ne_yerr_list.append(yerr)


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
plt.axvline(x=np.log10(98107171),  color='gray', linestyle=':', alpha=0.3)

# SIIのcritical densityを表示する (6716, 6731)
# 値の定義
nc_6716 = np.log10(1917.5607046610592)
nc_6731 = np.log10(5067.434274587508)  # 使わないなら消してOK

x = np.linspace(6, 12, 400)
y = np.full_like(x, nc_6716)  # ★ 定数をxと同じ長さの配列にする
T = 6  # 閾値（しきい値）
# 曲線（水平線）
ax.plot(x, y, color="gray")
# y >= T の部分を塗る
ax.fill_between(x, y, T, color="gray", alpha=0.3, interpolate=True)
# しきい値の水平線
ax.axhline(T, color="gray", linestyle="--", linewidth=1)
# plt.axhline(y=np.log10(1917.5607046610592),  color='tab:red', linestyle='-.', alpha=0.7)
# plt.axhline(y=np.log10(5067.434274587508),  color='tab:red', linestyle='-', alpha=0.7)
plt.text(x=6+0.1, y=np.log10(1917.5607046610592)+0.1, s=r"$n_{\mathrm{crit}}$([SII]6716)", fontsize=12,)
# plt.text(x=6+0.1, y=np.log10(5067.434274587508)+0.1,  s=r"$n_{\mathrm{crit}}$([SII]6731)", fontsize=12,)

# SDSSのデータのみで線形回帰した時の直線を表示する
slope_sdss     = 0.026
intercept_sdss = 2.513
x_range = np.linspace(6, 12, 1000)
y_range = slope_sdss * x_range + intercept_sdss
plt.plot(x_range, y_range, color='black', linestyle='--')
plt.text(x=11, y=intercept_sdss - 0.1,  s="best fit (SDSS)", fontsize=12)

# CLASSYのデータのみで線形回帰した時の直線を表示する
slope_classy = 0.013
intercept_classy = 2.187
x_range = np.linspace(6, 12, 1000)
y_range = slope_classy * x_range + intercept_classy
plt.plot(x_range, y_range, color='black', linestyle='-.')
plt.text(x=11, y=intercept_classy - 0.1,  s="best fit (CLASSY)", fontsize=12)


# === 推定結果（あなたの値に置き換え） ===
# slope（固定）
m_hat_z0   = 0.174 

# intercept
b_hat_z0   = 0.882
b_hat_z3   = 0.922
b_hat_z6   = 1.313
b_hat_z9   = 1.535

# slope の標準誤差
sigma_m_z0 = 0.006  
sigma_m_z3 = 0.000  
sigma_m_z6 = 0.000  
sigma_m_z9 = 0.000 

# intercept の標準誤差 
sigma_b_z0 = 0.064 
sigma_b_z3 = 0.018 
sigma_b_z6 = 0.050 
sigma_b_z9 = 0.129 

# slope と intercept の相関（例：不明なら 0）
rho_mb_z0  = 0    
rho_mb_z3  = 0    
rho_mb_z6  = 0    
rho_mb_z9  = 0    

# x 範囲
x = np.linspace(6, 12, 1000)
# 推定直線
y_hat_z0 = m_hat_z0 * x + b_hat_z0
y_hat_z3 = m_hat_z0 * x + b_hat_z3
y_hat_z6 = m_hat_z0 * x + b_hat_z6
y_hat_z9 = m_hat_z0 * x + b_hat_z9
# パラメータ不確かさ由来の y の標準偏差
cov_mb_z0 = rho_mb_z0 * sigma_m_z0 * sigma_b_z0
cov_mb_z3 = rho_mb_z3 * sigma_m_z3 * sigma_b_z3
cov_mb_z6 = rho_mb_z6 * sigma_m_z6 * sigma_b_z6
cov_mb_z9 = rho_mb_z9 * sigma_m_z9 * sigma_b_z9
sigma_y_z0 = np.sqrt((x * sigma_m_z0)**2 + sigma_b_z0**2 + 2 * x * cov_mb_z0)
sigma_y_z3 = np.sqrt((x * sigma_m_z3)**2 + sigma_b_z3**2 + 2 * x * cov_mb_z3)
sigma_y_z6 = np.sqrt((x * sigma_m_z6)**2 + sigma_b_z6**2 + 2 * x * cov_mb_z6)
sigma_y_z9 = np.sqrt((x * sigma_m_z9)**2 + sigma_b_z9**2 + 2 * x * cov_mb_z9)
# 信頼水準（k=1 なら約68%, k=1.96 なら約95%）
k = 1
lower_z0 = y_hat_z0 - k * sigma_y_z0
lower_z3 = y_hat_z3 - k * sigma_y_z3
lower_z6 = y_hat_z6 - k * sigma_y_z6
lower_z9 = y_hat_z9 - k * sigma_y_z9
upper_z0 = y_hat_z0 + k * sigma_y_z0
upper_z3 = y_hat_z3 + k * sigma_y_z3
upper_z6 = y_hat_z6 + k * sigma_y_z6
upper_z9 = y_hat_z9 + k * sigma_y_z9
ax.plot(x, y_hat_z0, color='black')
ax.plot(x, y_hat_z3, color='blue')
ax.plot(x, y_hat_z6, color='green')
ax.plot(x, y_hat_z9, color='red')
ax.fill_between(x, lower_z0, upper_z0, color='gray' , alpha=0.15)
ax.fill_between(x, lower_z3, upper_z3, color='blue' , alpha=0.15)
ax.fill_between(x, lower_z6, upper_z6, color='green', alpha=0.15)
ax.fill_between(x, lower_z9, upper_z9, color='red'  , alpha=0.15)


# 全体のプロットの設定
plt.xlim(6, 12)
plt.ylim(0, 6)
ax.set_xlabel(r"$\log(M_\ast/M_\odot)$")
ax.set_ylabel(r"$log(n_e) [\mathrm{cm^{-3}}]$")
# === 枠線 (spines) の設定 ===
# 線の太さ・色・表示非表示などを個別に制御
for spine in ax.spines.values():
    spine.set_linewidth(2)       # 枠線の太さ
    spine.set_color("black")     # 枠線の色
plt.tight_layout()
plt.savefig(os.path.join(current_dir, "results/figure/ne_vs_sm_plot_sdss_ms.png"))
plt.show()

# Monitor memory usage
process = psutil.Process()
mem_info_before = process.memory_info().rss / 1024**3 # in GB
print(f"Memory usage before processing: {mem_info_before:.2f} GB")