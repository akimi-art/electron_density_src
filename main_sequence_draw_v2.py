
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
スクリプトの概要:
このスクリプトはSDSSの銀河に関して
main sequence上にあるもののみを描画します。
CLASSYのデータも追加しました。
また、SDSSの欠損値を取り除くようにしています。

使用方法:
    python main_sequence_draw_v2.py [オプション]

著者: A. M.
作成日: 2026-01-29

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


# =========================================================
# 1) CLASSY向け： "中央値+上誤差-下誤差" をパースする正規表現
#    例: 11.229000+0.023000-0.023000
# =========================================================
pattern = re.compile(r'^([+-]?\d+(?:\.\d+)?)(?:\+(\d+(?:\.\d+)?))?(?:-(\d+(?:\.\d+)?))?$')

def load_catalog_pattern(path, Mstar_col, SFR_col):
    """
    1セル形式: "median+up-down" が入っているカタログ用
    返り値：
      Mstar_med, Mstar_err(2,N), SFR_med, SFR_err(2,N)
    ※ err は [down, up] の形（中央値からの差）
    """
    Mstar, Mstar_up, Mstar_down = [], [], []
    SFR,   SFR_up,   SFR_down   = [], [], []

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


# =========================================================
# 2) SDSS向け：中央値・-1σ・+1σ が「別カラム」で並んでいるカタログ用
#    欠損パターン (median=-1, err=0,0) をピンポイントに除外
# =========================================================
def load_catalog_sdss_errwidth(path,
                               m_med_col, m_errm_col, m_errp_col,
                               sfr_med_col, sfr_errm_col, sfr_errp_col,
                               missing_med=-1.0, missing_sig=0.0,
                               delimiter=r'\s+',
                               tol=1e-12):
    """
    6列が (Mmed, Merr-, Merr+, SFRmed, SFRerr-, SFRerr+) の形式（=誤差幅）を読む。
    欠損 (-1,0,0) を除外し、errorbar用に err幅をそのまま返す。
    """
    Mmed, Merrm, Merrp = [], [], []
    Smed, Serrm, Serrp = [], [], []

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            parts = re.split(delimiter, line.strip())

            need = max(m_med_col, m_errm_col, m_errp_col,
                       sfr_med_col, sfr_errm_col, sfr_errp_col)
            if len(parts) <= need:
                continue

            m0  = float(parts[m_med_col])
            m_em = float(parts[m_errm_col])  # err-
            m_ep = float(parts[m_errp_col])  # err+

            s0  = float(parts[sfr_med_col])
            s_em = float(parts[sfr_errm_col])
            s_ep = float(parts[sfr_errp_col])

            # NaN/inf除外
            if not np.isfinite(m0+m_em+m_ep+s0+s_em+s_ep):
                continue

            # 欠損 (-1,0,0) 除外（isclose で安全に）
            miss_m = np.isclose(m0, missing_med, atol=tol) and np.isclose(m_em, missing_sig, atol=tol) and np.isclose(m_ep, missing_sig, atol=tol)
            miss_s = np.isclose(s0, missing_med, atol=tol) and np.isclose(s_em, missing_sig, atol=tol) and np.isclose(s_ep, missing_sig, atol=tol)
            if miss_m or miss_s:
                continue

            # errorbar は負のerr不可
            if (m_em < 0) or (m_ep < 0) or (s_em < 0) or (s_ep < 0):
                continue

            Mmed.append(m0); Merrm.append(m_em); Merrp.append(m_ep)
            Smed.append(s0); Serrm.append(s_em); Serrp.append(s_ep)

    Mmed = np.array(Mmed)
    Smed = np.array(Smed)
    Merr = np.array([np.array(Merrm), np.array(Merrp)])  # (2,N)
    Serr = np.array([np.array(Serrm), np.array(Serrp)])  # (2,N)

    return Mmed, Merr, Smed, Serr


# =========================================================
# 3) MS帯（Elbaz+07近似、あなたの定義）
# =========================================================
def ms_band_mask(Mstar, SFR):
    SFR_upper = 0.77 * Mstar - 7.26317412397
    SFR_lower = 0.77 * Mstar - 7.77102999566
    return (SFR >= SFR_lower) & (SFR <= SFR_upper)


# =========================================================
# 4) 入力ファイル設定
# =========================================================
current_dir = os.getcwd()

datasets = [
    # --- SDSS：列分割形式（中央値, -1σ値, +1σ値 が別カラム）---
    dict(
        name="SDSS",
        path=os.path.join(current_dir, "results/txt/totlgm_dr7_v5_2_gal_fibsfr_dr7_v5_2.txt"),

        loader="sdss_cols",

        # ★ここをあなたのSDSSファイルに合わせる（0-index）
        # 例： 8列目→logM* median, 9列目→logM* -1σ値, 10列目→logM* +1σ値
        #      11列目→logSFR median, 12列目→logSFR -1σ値, 13列目→logSFR +1σ値
        #
        # ※あなたが「8列目がlogM*, 10列目がlogSFR」と言っていた部分は
        #   “中央値だけ”の話に見えるので、-1σ,+1σ列も含めて指定してください。
        #
        # ひとまず「1〜6列目が(Mmed,M-1σ,M+1σ,Smed,S-1σ,S+1σ)」のファイルなら
        #   m_med_col=0, m_minus1s_col=1, m_plus1s_col=2, sfr_med_col=3, ...
        #
        m_med_col=0,
        m_minus1s_col=1,
        m_plus1s_col=2,
        sfr_med_col=3,
        sfr_minus1s_col=4,
        sfr_plus1s_col=5,

        color_in="tab:blue",
        color_out="gray",
        plot_ms_split=True,
        marker="."
    ),

    # --- CLASSY：1セル "median+up-down" 形式（いまのローダー）---
    dict(
        name="CLASSY",
        path=os.path.join(current_dir, "results/Mingozzi22/Mingozzi22in.txt"),
        loader="pattern",
        Mstar_col=7,  # ←CLASSYの列に合わせて
        SFR_col=8,    # ←CLASSYの列に合わせて
        color_in="crimson",
        color_out="crimson",
        plot_ms_split=False,
        marker="s"
    ),
]


# =========================================================
# 5) プロット
# =========================================================
fig, ax = plt.subplots(figsize=(10, 10))

# --- Main Sequence (Elbaz+07近似線) ---
M_line = np.linspace(7, 13, 200)
SFR_line_median = 0.77 * M_line - 7.53048074738
SFR_line_upper  = 0.77 * M_line - 7.26317412397
SFR_line_lower  = 0.77 * M_line - 7.77102999566

ax.plot(M_line, SFR_line_median, color='tab:blue', linestyle='--',
        label='MS (Elbaz+07)', zorder=3)
ax.fill_between(M_line, SFR_line_lower, SFR_line_upper,
                color='tab:blue', alpha=0.2, zorder=0)

# --- datasets を順に重ね描き ---
for ds in datasets:

    if ds["loader"] == "sdss_cols":
        Mstar, Mstar_err, SFR, SFR_err = load_catalog_sdss_errwidth(
            ds["path"],
            ds["m_med_col"], ds["m_minus1s_col"], ds["m_plus1s_col"],
            ds["sfr_med_col"], ds["sfr_minus1s_col"], ds["sfr_plus1s_col"],
            missing_med=-1.0, missing_sig=0.0
        )
    elif ds["loader"] == "pattern":
        Mstar, Mstar_err, SFR, SFR_err = load_catalog_pattern(
            ds["path"], ds["Mstar_col"], ds["SFR_col"]
        )
    else:
        raise ValueError(f"Unknown loader type: {ds['loader']}")
    print(ds["name"], "N=", len(Mstar))
    if ds["plot_ms_split"]:
        in_ms = ms_band_mask(Mstar, SFR)

        # MS内
        ax.errorbar(Mstar[in_ms], SFR[in_ms],
                    xerr=Mstar_err[:, in_ms], yerr=SFR_err[:, in_ms],
                    fmt=ds["marker"], markersize=4,
                    color=ds["color_in"], ecolor=ds["color_in"],
                    elinewidth=0.0, capsize=0, alpha=0.45,
                    label=f'{ds["name"]}: MS', zorder=2)

        # MS外
        ax.errorbar(Mstar[~in_ms], SFR[~in_ms],
                    xerr=Mstar_err[:, ~in_ms], yerr=SFR_err[:, ~in_ms],
                    fmt=ds["marker"], markersize=4,
                    color=ds["color_out"], ecolor=ds["color_out"],
                    elinewidth=0.0, capsize=0, alpha=0.20,
                    label=f'{ds["name"]}', zorder=1)

    else:
        ax.errorbar(Mstar, SFR,
                    xerr=Mstar_err, yerr=SFR_err,
                    fmt=ds["marker"], markersize=8,
                    color=ds["color_in"], ecolor=ds["color_in"],
                    elinewidth=0.0, capsize=0, alpha=0.55,
                    label=ds["name"], zorder=2)

# --- 体裁 ---
ax.set_xlim(8.5, 12.0)
ax.set_ylim(-5, 2)
ax.set_xlabel(r'$\log(M_\ast/M_\odot)$')
ax.set_ylabel(r'$\log(\mathrm{SFR}/M_\odot~\mathrm{yr}^{-1})$')

for spine in ax.spines.values():
    spine.set_linewidth(2)
    spine.set_color("black")

ax.legend(frameon=False)
plt.tight_layout()

out_png = os.path.join(current_dir, "results/figure/main_sequence_SDSS_CLASSY_v2.png")
plt.savefig(out_png, dpi=200)
plt.show()

print("Saved:", out_png)