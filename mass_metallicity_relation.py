#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
スクリプトの概要:
このスクリプトは星質量と金属量の
プロットをします。

使用方法:
    mass_metallicity_relation.py [オプション]

著者: A. M.
作成日: 2026-01-27

参考文献:
    - PEP 8:                  https://peps.python.org/pep-0008/
    - PEP 257 (Docstring規約): https://peps.python.org/pep-0257/
    - Python公式ドキュメント:    https://docs.python.org/ja/3/
"""



# == 必要なパッケージのインストール == #
import re
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


# 軸の設定
plt.rcParams.update({
    # --- 図全体 ---
    "figure.figsize": (12, 6),       # 図サイズ
    "font.size": 16,                 # 全体フォントサイズ
    "axes.labelsize": 18,            # 軸ラベルのサイズ
    "axes.titlesize": 18,            # タイトルのサイズ
    "axes.grid": True,               # グリッドOFF

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


# === 入力ファイル ===
current_dir = os.getcwd()
input_file = os.path.join(current_dir, "results/Samir16/Samir16in_standard_v3_ms_only_v3_merged_direct_Te.txt")

# === フィールド抽出設定 ===
# 8列目 → log(M*), 10列目 → log(SFR)
Mstar_col = 7   
metal_col = 41     

# 正規表現パターン: 中央値 + 上誤差 - 下誤差
pattern = re.compile(r'^([+-]?\d+(?:\.\d+)?)(?:\+(\d+(?:\.\d+)?))?(?:-(\d+(?:\.\d+)?))?$')

Mstar, Mstar_up, Mstar_down = [], [], []
metal, metal_up, metal_down = [], [], []

with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip() == "":
            continue
        parts = re.split(r'\s+', line.strip())
        # 列数チェック
        if len(parts) <= max(Mstar_col, metal_col):
            continue

        # --- Mstar ---
        m_m = pattern.match(parts[Mstar_col])
        # --- SFR ---
        m_me = pattern.match(parts[metal_col])

        if not m_m or not m_me:
            continue

        mstar_med = float(m_m.group(1))
        mstar_up = float(m_m.group(2)) if m_m.group(2) else 0.0
        mstar_down = float(m_m.group(3)) if m_m.group(3) else 0.0

        metal_med = float(m_me.group(1))
        metal_up = float(m_me.group(2)) if m_me.group(2) else 0.0
        metal_down = float(m_me.group(3)) if m_me.group(3) else 0.0

        Mstar.append(mstar_med)
        Mstar_up.append(mstar_up)
        Mstar_down.append(mstar_down)
        metal.append(metal_med)
        # metal_up.append(metal_up)
        # metal_down.append(metal_down)

# === numpy配列に変換 ===
Mstar = np.array(Mstar)
metal = np.array(metal)
# Mstar, SFRの誤差を定義する → shape: (2, N)の形
Mstar_err = np.array([Mstar_down, Mstar_up])  
metal_err   = np.array([metal_down,  metal_up])     

# === プロット ===
fig, ax = plt.subplots()
# 帯の中に入っているかの判定
ax.scatter(Mstar, metal, color='tab:blue') 
# ax.legend()
# ax.set_xlim(9, 11.5)
# ax.set_ylim(8.4, 9.2)
ax.set_xlabel(r'$\log(M_\ast/M_\odot)$')
ax.set_ylabel(r'$12+\log(\mathrm{O/H})$')
# === 枠線 (spines) の設定 ===
# 線の太さ・色・表示非表示などを個別に制御
for spine in ax.spines.values():
    spine.set_linewidth(2)       # 枠線の太さ
    spine.set_color("black")     # 枠線の色
plt.tight_layout()
plt.savefig(os.path.join(current_dir, "results/figure/mass_metallicity_relation_direct_Te.png"))
plt.show()