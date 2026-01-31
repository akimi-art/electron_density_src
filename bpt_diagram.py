#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
スクリプトの概要:
BPT-Diagramを描画します。


使用方法:
    direct-te.py [オプション]

著者: A. M.
作成日: 2026-01-22

参考文献:
    - PEP 8:                  https://peps.python.org/pep-0008/
    - PEP 257 (Docstring規約): https://peps.python.org/pep-0257/
    - Python公式ドキュメント:    https://docs.python.org/ja/3/
    - Curti+17
"""

"""
2026-01-27
今度はDirect-Teの結果を使ってtxtファイルを作ろう（その前にcsvファイルからtxtファイルを
作る（merged）
"""



# === 必要なモジュールのインポート ===
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path


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
# 入出力設定
# ============================================================
current_dir = os.getcwd()

in_csv  = os.path.join(current_dir, "results/csv/Samir16in_standard_v3_ms_only_v3.csv")
df = pd.read_csv(in_csv)

# -------------------------
# O'Donnell (1994) extinction curve
# -------------------------
def k_odonnell94(wave_ang, Rv=3.1):
    wave_um = wave_ang / 1e4
    x = 1.0 / wave_um
    y = x - 1.82

    a = (1
         + 0.17699*y
         - 0.50447*y**2
         - 0.02427*y**3
         + 0.72085*y**4
         + 0.01979*y**5
         - 0.77530*y**6
         + 0.32999*y**7)

    b = (1.41338*y
         + 2.28305*y**2
         + 1.07233*y**3
         - 5.38434*y**4
         - 0.62251*y**5
         + 5.30260*y**6
         - 2.09002*y**7)

    return (a + b / Rv) * Rv  # A(λ)/E(B-V)


# -------------------------
# constants
# -------------------------
HaHb_int = 2.86   # Case B
kHa = k_odonnell94(6563.0)
kHb = k_odonnell94(4861.0)
kNII = k_odonnell94(6584.0)
kOIII = k_odonnell94(5007.0)


def ebv_from_balmer(Ha, Hb):
    if (Ha <= 0) or (Hb <= 0):
        return np.nan
    val = 2.5 / (kHb - kHa) * np.log10((Ha / Hb) / HaHb_int)
    return max(0.0, val)


def deredden(F, ebv, k):
    if (F <= 0) or (not np.isfinite(ebv)):
        return np.nan
    return F * 10**(0.4 * k * ebv)


# df は SDSS の CSV を読み込んだものとする
# 必要列:
# H_ALPHA_FLUX, H_BETA_FLUX, NII_6584_FLUX, OIII_5007_FLUX

EBV = []
N2  = []
R3  = []

for _, r in df.iterrows():
    ebv = ebv_from_balmer(r["H_ALPHA_FLUX"], r["H_BETA_FLUX"])
    EBV.append(ebv)

    Ha_c   = deredden(r["H_ALPHA_FLUX"], ebv, kHa)
    Hb_c   = deredden(r["H_BETA_FLUX"],  ebv, kHb)
    NII_c  = deredden(r["NII_6584_FLUX"], ebv, kNII)
    OIII_c = deredden(r["OIII_5007_FLUX"], ebv, kOIII)

    if (Ha_c > 0) and (Hb_c > 0) and (NII_c > 0) and (OIII_c > 0):
        N2.append(np.log10(NII_c / Ha_c))
        R3.append(np.log10(OIII_c / Hb_c))
    else:
        N2.append(np.nan)
        R3.append(np.nan)

df["E_BV_gas"] = EBV
df["N2"] = N2   # log([NII]/Ha)
df["R3"] = R3   # log([OIII]/Hb)

dfp = df.dropna(subset=["N2", "R3"])



# === GridSpecを使って図を左右に並べる === 

fig = plt.figure(figsize=(12, 6))
gs = gridspec.GridSpec(1, 2)   # 1行2列のグリッドを作成
ax1 = fig.add_subplot(gs[0, 0])  # 左
ax2 = fig.add_subplot(gs[0, 1])  # 右
# ax3 = fig.add_subplot(gs[0, 2])  # 右

# -------------------------
# Kewley+2013 BPT line (z=0)
# -------------------------
x = np.linspace(-2.0, 0.0, 500)  # log([NII]/Ha)
y_kewley = 0.61 / (x - 0.02) + 1.2
ax1.scatter(dfp["N2"], dfp["R3"],
            s=10, alpha=0.3, label="SDSS")
ax1.plot(x, y_kewley,
         color="black", lw=2, label="Kewley+13 (z=0)")
ax1.set_xlabel(r"log([NII]6584 / H$\alpha$)")
ax1.set_ylabel(r"log([OIII]5007 / H$\beta$)")
ax1.set_xlim(-2.0, 0.3)
ax1.set_ylim(-1.5, 1.5)
# === 枠線 (spines) の設定 ===
# 線の太さ・色・表示非表示などを個別に制御
for spine in ax1.spines.values():
    spine.set_linewidth(2)     # 枠線の太さ
    spine.set_color("black")     # 枠線の色
ax1.legend(frameon=False)




# === 入力ファイル ===
current_dir = os.getcwd()
input_file = os.path.join(current_dir, "results/Samir16/Samir16in_standard_v3_ms_only_v3_re.txt")

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
ax2.plot(M_line, SFR_line_median, color='tab:blue', linestyle='--', label='Main Sequence (Elbaz+07)', zorder=2) # k:黒
ax2.fill_between(M_line, SFR_line_lower, SFR_line_upper, color='tab:blue', alpha=0.2)
ax2.legend()
# 帯内のみを強調表示（条件でフィルタ）
ax2.errorbar(Mstar[in_ms_band], SFR[in_ms_band],
            xerr=Mstar_err[:, in_ms_band], yerr=SFR_err[:, in_ms_band], # (2, N) → (2, M) に絞る
            fmt='o', ecolor='tab:blue', elinewidth=0.5, capsize=0,
            markersize=5, color='tab:blue', alpha=0.4, zorder=1)
# 帯外は薄く表示（参考）
ax2.errorbar(Mstar[~in_ms_band], SFR[~in_ms_band],
            xerr=Mstar_err[:, ~in_ms_band], yerr=SFR_err[:, ~in_ms_band],
            fmt='o', ecolor='gray', elinewidth=0.5, capsize=0,
            markersize=5, color='gray', alpha=0.25, zorder=1)
ax2.set_xlim(7, 13)
ax2.set_ylim(-4, 3)
ax2.set_xlabel(r'$\log(M_\ast/M_\odot)$')
ax2.set_ylabel(r'$\log(\mathrm{SFR}/M_\odot~\mathrm{yr}^{-1})$')
# === 枠線 (spines) の設定 ===
# 線の太さ・色・表示非表示などを個別に制御
for spine in ax2.spines.values():
    spine.set_linewidth(2)     # 枠線の太さ
    spine.set_color("black")     # 枠線の色


# # === metallicityのヒストグラムを描画する === 
# # === フィールド抽出設定 ===
# metal_col = 12

# # 空白/タブ区切りを想定して読み込み
# data = np.loadtxt(input_file)

# x = data[:, metal_col]  # 指定列を取り出し

# ax3.set_hist(x, bins=30, edgecolor="black")
# ax3.set_xlabel(f"12+log(O/H)")
# ax3.set_ylabel("Count")


plt.tight_layout()
plt.savefig(os.path.join(current_dir, "results/figure/BPT_Diagram_Samir16in_standard_v3_ms_only_v3_re_no_agn.png"))
plt.show()


# Kewley+13の線の下側にあるもののみを抽出する
# 参考↓
# === ここから追加: Main Sequence 銀河のみの行を抽出して新しい txt を作成 ===
# マスク in_ms_band は上で確定済み。
# 元ファイルをもう一度走査し、パースに成功した行のみインデックスを進め、
# そのインデックスが in_ms_band で True のものだけ行を収集して書き出します。
output_dir = os.path.join(current_dir, "results/Samir16")
os.makedirs(output_dir, exist_ok=True)

# 出力ファイル名は元ファイル名に接尾辞を付与
base = Path(input_file).name         # 例: Samir16in_standard_v3.txt
stem = Path(base).stem               # 例: Samir16in_standard_v3
# out_path = os.path.join(output_dir, f"{stem}_MS_only_v1.txt")

selected_lines = []
idx = -1  # Mstar/SFRに格納済みのレコードのインデックス（パース成功時のみ増える）

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

R3 = np.array(R3)
N2 = np.array(N2)
y_kewley_mask = 0.61 / (N2) + 1.2
in_sfg_band = (R3 < y_kewley_mask) 

with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip() == "":
            continue
        # parts = re.split(r"\s+", line.strip())
        # if len(parts) <= max(Mstar_col, SFR_col):
        #     continue
        # # 既存のパターン（中央値+上誤差-下誤差）で両方の列をチェック
        # m_m = pattern.match(parts[Mstar_col])
        # m_s = pattern.match(parts[SFR_col])
        # if not m_m or not m_s:
        #     continue

        idx += 1  # ← パース成功したときだけ進めるので Mstar/SFR 配列と対応
        if in_sfg_band[idx]:
            selected_lines.append(line.rstrip("\n"))

        
# 書き出し
output_path = os.path.join(output_dir, "Samir16/Samir16in_standard_v3_ms_only_v3_re_no_agn.txt")
with open(output_path, 'w', encoding='utf-8') as g:
    g.write("\n".join(selected_lines))

print(f"星形成銀河のみを抽出: {len(selected_lines)} 行を書き出しました → {output_path}")