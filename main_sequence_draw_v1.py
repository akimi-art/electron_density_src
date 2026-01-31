
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
スクリプトの概要:
このスクリプトはSDSSの銀河に関して
main sequence上にあるもののみを描画します。
CLASSYのデータも追加しました。

使用方法:
    python main_sequence_draw_v1.py [オプション]

著者: A. M.
作成日: 2026-01-22

参考文献:
    - PEP 8: https://peps.python.org/pep-0008/
    - PEP 257 (Docstring規約): https://peps.python.org/pep-0257/
    - Python公式ドキュメント: https://docs.python.org/ja/3/
    - Elbaz et al. (2007)
    - Chen et al. (2016)
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
    "axes.grid": False,               # グリッドOFF

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



import os, re
import numpy as np
import matplotlib.pyplot as plt

# --- 正規表現: 中央値 + 上誤差 - 下誤差  (例: 11.229000+0.023000-0.023000) ---
pattern = re.compile(r'^([+-]?\d+(?:\.\d+)?)(?:\+(\d+(?:\.\d+)?))?(?:-(\d+(?:\.\d+)?))?$')

def load_catalog(path, Mstar_col, SFR_col):
    """
    テキストファイルから
      Mstar_med, Mstar_err(2,N), SFR_med, SFR_err(2,N)
    を返す。
    """
    Mstar, Mstar_up, Mstar_down = [], [], []
    SFR, SFR_up, SFR_down = [], [], []

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            parts = re.split(r'\s+', line.strip())
            if len(parts) <= max(Mstar_col, SFR_col):
                continue

            m_m = pattern.match(parts[Mstar_col])
            m_s = pattern.match(parts[SFR_col])
            if not m_m or not m_s:
                continue

            mstar_med  = float(m_m.group(1))
            mstar_up   = float(m_m.group(2)) if m_m.group(2) else 0.0
            mstar_down = float(m_m.group(3)) if m_m.group(3) else 0.0

            sfr_med  = float(m_s.group(1))
            sfr_up   = float(m_s.group(2)) if m_s.group(2) else 0.0
            sfr_down = float(m_s.group(3)) if m_s.group(3) else 0.0

            Mstar.append(mstar_med)
            Mstar_up.append(mstar_up)
            Mstar_down.append(mstar_down)

            SFR.append(sfr_med)
            SFR_up.append(sfr_up)
            SFR_down.append(sfr_down)

    Mstar = np.array(Mstar)
    SFR   = np.array(SFR)
    Mstar_err = np.array([Mstar_down, Mstar_up])   # shape (2,N)
    SFR_err   = np.array([SFR_down,  SFR_up])      # shape (2,N)
    return Mstar, Mstar_err, SFR, SFR_err


def ms_band_mask(Mstar, SFR):
    """Elbaz+07 のMS帯（あなたの定義）"""
    # --- Main Sequence (Elbaz+07近似線) ---
    SFR_upper = 0.77 * Mstar - 7.26317412397 
    SFR_lower = 0.77 * Mstar - 7.77102999566
    return (SFR >= SFR_lower) & (SFR <= SFR_upper)


# === 入力ファイル ===
current_dir = os.getcwd()

datasets = [
    # SDSS: 8列目→logM*, 10列目→logSFR  (0-indexで 7,9)
    dict(
        name="SDSS",
        path=os.path.join(current_dir, "results/Samir16/Samir16in_standard_v5.txt"),
        Mstar_col=7,
        SFR_col=9,
        color_in="tab:blue",
        color_out="gray",
        plot_ms_split=True,   # MS内外を分けて表示する
        marker="."
    ),

    # CLASSY: 例（ファイル名・列番号は適宜変更）
    dict(
        name="CLASSY",
        path=os.path.join(current_dir, "results/Mingozzi22/Mingozzi22in.txt"),  # ←ここを実ファイルに
        Mstar_col=7,   # ←CLASSYの列に合わせて変更
        SFR_col=8,     # ←CLASSYの列に合わせて変更
        color_in="black",
        color_out="black",
        plot_ms_split=False,  # まずは全点を同色で表示（確認不足の間はこれが安全）
        marker="."
    ),
]

# === プロット ===
fig, ax = plt.subplots(figsize=(10, 6))

# --- Main Sequence (Elbaz+07近似線) ---
M_line = np.linspace(7, 13, 200)
SFR_line_median = 0.77 * M_line - 7.53048074738
SFR_line_upper  = 0.77 * M_line - 7.26317412397 
SFR_line_lower  = 0.77 * M_line - 7.77102999566

ax.plot(M_line, SFR_line_median, color='tab:blue', linestyle='--',
        label='MS (Elbaz+07)', zorder=3)
ax.fill_between(M_line, SFR_line_lower, SFR_line_upper, color='tab:blue', alpha=0.2, zorder=0)

# --- datasets を順に重ね描き ---
for ds in datasets:
    Mstar, Mstar_err, SFR, SFR_err = load_catalog(ds["path"], ds["Mstar_col"], ds["SFR_col"])

    if ds["plot_ms_split"]:
        in_ms = ms_band_mask(Mstar, SFR)

        # MS内
        ax.errorbar(Mstar[in_ms], SFR[in_ms],
                    xerr=Mstar_err[:, in_ms], yerr=SFR_err[:, in_ms],
                    fmt=ds["marker"], markersize=6,
                    color=ds["color_in"], ecolor=ds["color_in"],
                    elinewidth=0.0, capsize=0, alpha=1,
                    label=f'{ds["name"]}: MS', zorder=2)

        # MS外
        ax.errorbar(Mstar[~in_ms], SFR[~in_ms],
                    xerr=Mstar_err[:, ~in_ms], yerr=SFR_err[:, ~in_ms],
                    fmt=ds["marker"], markersize=6,
                    color=ds["color_out"], ecolor=ds["color_out"],
                    elinewidth=0.0, capsize=0, alpha=0.50,
                    label=f'{ds["name"]}', zorder=1)

    else:
        # 全点を同色で（まずはCLASSYはこれが安全）
        ax.errorbar(Mstar, SFR,
                    xerr=Mstar_err, yerr=SFR_err,
                    fmt=ds["marker"], markersize=12,
                    color=ds["color_in"], ecolor=ds["color_in"],
                    elinewidth=0.0, capsize=0, alpha=1,
                    label=ds["name"], zorder=2)

# --- 体裁 ---
ax.set_xlim(7, 13)
ax.set_ylim(-4, 3)
ax.set_xlabel(r'$\log(M_\ast/M_\odot)$')
ax.set_ylabel(r'$\log(\mathrm{SFR}/M_\odot~\mathrm{yr}^{-1})$')

for spine in ax.spines.values():
    spine.set_linewidth(2)
    spine.set_color("black")

ax.legend(frameon=False)
plt.tight_layout()

out_png = os.path.join(current_dir, "results/figure/main_sequence_SDSS_CLASSY_v1.png")
plt.savefig(out_png, dpi=200)
plt.show()

print("Saved:", out_png)
