
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
スクリプトの概要:
このスクリプトはSDSSの銀河に関して
main sequence上にあるものの選定を行います。

使用方法:
    python main_sequence.py [オプション]

著者: A. M.
作成日: 2026-01-06

参考文献:
    - PEP 8: https://peps.python.org/pep-0008/
    - PEP 257 (Docstring規約): https://peps.python.org/pep-0257/
    - Python公式ドキュメント: https://docs.python.org/ja/3/
    - Elbaz et al. (2007)
    - Chen et al. (2016)
"""


# === 必要なパッケージのインストール === #
import re
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path


# === 軸の設定 === #
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
input_file = os.path.join(current_dir, "results/Samir16/Samir16in_standard_v3_ms_only_v3.txt")

# === フィールド抽出設定 ===
# 8列目 → log(M*), 10列目 → log(SFR)
Mstar_col = 7   
SFR_col = 9     

# 正規表現パターン: 中央値 + 上誤差 - 下誤差
pattern = re.compile(r'^([+-]?\d+(?:\.\d+)?)(?:\+(\d+(?:\.\d+)?))?(?:-(\d+(?:\.\d+)?))?$')

Mstar, Mstar_up, Mstar_down = [], [], []
SFR, SFR_up, SFR_down = [], [], []

with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip() == "":
            continue
        parts = re.split(r'\s+', line.strip())
        # 列数チェック
        if len(parts) <= max(Mstar_col, SFR_col):
            continue

        # --- Mstar ---
        m_m = pattern.match(parts[Mstar_col])
        # --- SFR ---
        m_s = pattern.match(parts[SFR_col])

        if not m_m or not m_s:
            continue

        mstar_med = float(m_m.group(1))
        mstar_up = float(m_m.group(2)) if m_m.group(2) else 0.0
        mstar_down = float(m_m.group(3)) if m_m.group(3) else 0.0

        sfr_med = float(m_s.group(1))
        sfr_up = float(m_s.group(2)) if m_s.group(2) else 0.0
        sfr_down = float(m_s.group(3)) if m_s.group(3) else 0.0

        Mstar.append(mstar_med)
        Mstar_up.append(mstar_up)
        Mstar_down.append(mstar_down)
        SFR.append(sfr_med)
        SFR_up.append(sfr_up)
        SFR_down.append(sfr_down)

# === numpy配列に変換 ===
Mstar = np.array(Mstar)
SFR = np.array(SFR)
# Mstar, SFRの誤差を定義する → shape: (2, N)の形
Mstar_err = np.array([Mstar_down, Mstar_up])  
SFR_err   = np.array([SFR_down,  SFR_up])     

# === プロット ===
fig, ax = plt.subplots(figsize=(10, 6))
# --- Main Sequence (Elbaz+07近似線) ---
SFR_upper = 0.77 * Mstar - 7.26317412397
SFR_lower = 0.77 * Mstar - 7.77102999566

# 描画用
M_line = np.linspace(7, 13, 100)
SFR_line_median = 0.77 * M_line - 7.53048074738
SFR_line_upper  = 0.77 * M_line - 7.26317412397
SFR_line_lower  = 0.77 * M_line - 7.77102999566
# 帯の中に入っているかの判定
in_ms_band = (SFR >= SFR_lower) & (SFR <= SFR_upper)
ax.plot(M_line, SFR_line_median, color='tab:blue', linestyle='--', label='Main Sequence (Elbaz+07)', zorder=2) # k:黒
ax.fill_between(M_line, SFR_line_lower, SFR_line_upper, color='tab:blue', alpha=0.2)
ax.legend()
# 帯内のみを強調表示（条件でフィルタ）
ax.errorbar(Mstar[in_ms_band], SFR[in_ms_band],
            xerr=Mstar_err[:, in_ms_band], yerr=SFR_err[:, in_ms_band], # (2, N) → (2, M) に絞る
            fmt='o', ecolor='tab:blue', elinewidth=0.5, capsize=0,
            markersize=5, color='tab:blue', alpha=0.4, zorder=1)
# 帯外は薄く表示（参考）
ax.errorbar(Mstar[~in_ms_band], SFR[~in_ms_band],
            xerr=Mstar_err[:, ~in_ms_band], yerr=SFR_err[:, ~in_ms_band],
            fmt='o', ecolor='gray', elinewidth=0.5, capsize=0,
            markersize=5, color='gray', alpha=0.25, zorder=1)
ax.set_xlim(7, 13)
ax.set_ylim(-4, 3)
ax.set_xlabel(r'$\log(M_\ast/M_\odot)$')
ax.set_ylabel(r'$\log(\mathrm{SFR}/M_\odot~\mathrm{yr}^{-1})$')
# === 枠線 (spines) の設定 ===
# 線の太さ・色・表示非表示などを個別に制御
for spine in ax.spines.values():
    spine.set_linewidth(2)     # 枠線の太さ
    spine.set_color("black")     # 枠線の色
plt.tight_layout()
# plt.savefig(os.path.join(current_dir, "results/figure/main_sequence_original.png"))
plt.show()


# === ここから追加: Main Sequence 銀河のみの行を抽出して新しい txt を作成 ===
# マスク in_ms_band は上で確定済み。
# 元ファイルをもう一度走査し、パースに成功した行のみインデックスを進め、
# そのインデックスが in_ms_band で True のものだけ行を収集して書き出します。
output_dir = os.path.join(current_dir, "results/txt")
os.makedirs(output_dir, exist_ok=True)

# 出力ファイル名は元ファイル名に接尾辞を付与
base = Path(input_file).name         # 例: Samir16in_standard_v3.txt
stem = Path(base).stem               # 例: Samir16in_standard_v3
# out_path = os.path.join(output_dir, f"{stem}_MS_only_v1.txt")

selected_lines = []
idx = -1  # Mstar/SFRに格納済みのレコードのインデックス（パース成功時のみ増える）

with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip() == "":
            continue
        parts = re.split(r"\s+", line.strip())
        if len(parts) <= max(Mstar_col, SFR_col):
            continue
        # 既存のパターン（中央値+上誤差-下誤差）で両方の列をチェック
        m_m = pattern.match(parts[Mstar_col])
        m_s = pattern.match(parts[SFR_col])
        if not m_m or not m_s:
            continue

        idx += 1  # ← パース成功したときだけ進めるので Mstar/SFR 配列と対応
        if in_ms_band[idx]:
            selected_lines.append(line.rstrip("\n"))

# 書き出し
with open(out_path, 'w', encoding='utf-8') as g:
    g.write("\n".join(selected_lines))

print(f"MS帯の銀河のみを抽出: {len(selected_lines)} 行を書き出しました → {out_path}")
# MS帯の銀河のみを抽出: 2402 行を書き出しました → /Users/matsuuraakimi/electron_density/results/Samir16in_standard_v3_MS_only.txt