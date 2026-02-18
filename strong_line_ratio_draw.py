#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FITS ファイルに保存された強輝線指標（R2, R3, O32, R23, N2, O3N2）の
ヒストグラムを 2行×3列で隙間なく並べて一枚の図に描画します。

- タイトル：なし
- 凡例：各パネルで指標名のみ表示
- 横軸：初期値は全パネルで [-3, 3] に統一
- 後で各パネルの xlim を個別調整できるよう xlim_overrides を用意
"""

# 必要なモジュールのインポート
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from astropy.io import fits

# 軸の設定
plt.rcParams.update({
    # --- 図全体 ---
    "figure.figsize": (12, 6),       # 図サイズ
    "font.size": 16,                 # 全体フォントサイズ
    "axes.labelsize": 16,            # 軸ラベルのサイズ
    "axes.titlesize": 16,            # タイトルのサイズ
    "axes.grid": False,              # グリッドOFF

    # --- 目盛り設定 (ticks) ---
    "xtick.direction": "in",         # x軸目盛りの向き
    "ytick.direction": "in",         # y軸目盛りの向き
    "xtick.top": False,               # 上にも目盛り
    "ytick.right": False,             # 右にも目盛り

    # 主目盛り（major ticks）
    "xtick.major.size": 16,          # 長さ
    "ytick.major.size": 16,
    "xtick.major.width": 2,          # 太さ
    "ytick.major.width": 2,

    # 補助目盛り（minor ticks）
    "xtick.minor.visible": False,     # 補助目盛りON
    "ytick.minor.visible": False,
    "xtick.minor.size": 8,           # 長さ
    "ytick.minor.size": 8,
    "xtick.minor.width": 1.5,        # 太さ
    "ytick.minor.width": 1.5,

    # --- 目盛りラベル ---
    "xtick.labelsize": 16,           # x軸ラベルサイズ
    "ytick.labelsize": 16,           # y軸ラベルサイズ

    # --- フォント ---
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
})

# ==========================
#  設 定
# ==========================

# 入力FITSのパス（必要に応じて書き換えてください）
current_dir = os.getcwd()
fits_path = os.path.join(
    current_dir,
    "results", "fits", "mpajhu_dr7_v5_2_merged_with_indices_Curti+17.fits"
)

# プロットする列とラベル（FITSの列名に合わせています）
ratios = [
    ("R2",   "R2"),
    ("R3",   "R3"),
    ("O32",  "O32"),
    ("R23",  "R23"),
    ("N2",   "N2"),
    ("O3N2", "O3N2"), 
]

# ヒストグラムの設定
# すべてのパネルの初期 x 範囲を統一（-3～3）
xlim_default = (-3.0, 3.0)

# 後で個別に x 範囲を変えたい場合は、ここに上書きを追加してください。
# 例） xlim_overrides = {"R2": (-1, 2), "O3N2": (-0.5, 1.5)}
xlim_overrides = {}

# ビン設定：横軸範囲に合わせて等間隔ビンを作成
bins = np.linspace(xlim_default[0], xlim_default[1], 61)  # 60 bins

# 塗りつぶし色やスタイル（必要なら調整）
hist_kwargs = dict(bins=bins, histtype="stepfilled", alpha=0.85, color="#4C78A8", edgecolor="none")

# ==========================
#  データ読み込み
# ==========================

with fits.open(fits_path) as hdul:
    data = hdul[1].data  # バイナリテーブル想定

# NaN/inf を除去するユーティリティ
def clean(arr):
    arr = np.asarray(arr, dtype=float)
    m = np.isfinite(arr)
    return arr[m]

# 各指標データを辞書に格納（存在しない列があれば KeyError）
series = {}
for colname, _label in ratios:
    if colname not in data.names:
        raise KeyError(f"FITS に列 '{colname}' が見つかりません。: {fits_path}")
    series[colname] = clean(data[colname])

# ==========================
#  図の生成（2 行 × 3 列, 隙間なし）
# ==========================

fig = plt.figure(figsize=(12, 6), dpi=120)  # サイズはお好みで
gs = GridSpec(
    nrows=2, ncols=3, figure=fig,
    left=0.01, right=0.99, bottom=0.06, top=0.98,  # マージンを詰める
    wspace=0.0, hspace=0.0                         # パネル間の隙間ゼロ
)

axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]

# 共有Y軸（自動スケール）で OK。X軸はあとで揃える
for ax, (colname, label) in zip(axes, ratios):
    arr = series[colname]
    # 横軸の範囲は初期値 [-3, 3]。後で個別上書きがあれば適用。
    xl = xlim_overrides.get(colname, xlim_default)
    # ビンは初期範囲ベースの bins を使う（必要なら個別 bins も作れるように変更可）
    ax.hist(arr, **hist_kwargs)
    ax.set_xlim(*xl)
    # 目盛りを控えめに（必要なら調整）
    ax.tick_params(axis="both", labelsize=10, length=3)
    # 凡例（タイトルの代わりにどの比かを示す）
    ax.legend([label], fontsize=10, frameon=False, loc="upper left")

# 共有 x 軸の範囲を統一しておきたい場合は、下で強制的に適用（個別上書きを残したいのでここは未適用）
# for ax in axes:
#     ax.set_xlim(*xlim_default)

# 目盛りが重ならないように、上段の x 軸ラベルは消す（必要に応じて）
for ax in axes[:3]:
    ax.set_xticklabels([])

# y 軸ラベルも省略可（完全に隙間のない “並べただけ” の絵に）
for ax in axes:
    ax.set_ylabel("")
    ax.set_xlabel("")

# 保存（任意）
out_png = os.path.join(current_dir, "results/figure/strongline_hist_2x3.png")
fig.savefig(out_png, dpi=150)

plt.show()