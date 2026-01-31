#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
スクリプトの概要:
このスクリプトは表全般の作成のテンプレートです。
csv形式でデータを保存する機能と、
図にまとめる機能があります。

使用方法:
    ne_vs_metallicity_table_making.py [オプション]

著者: A. M.
作成日: 2026-01-19

参考文献:
    - PEP 8:                  https://peps.python.org/pep-0008/
    - PEP 257 (Docstring規約): https://peps.python.org/pep-0257/
    - Python公式ドキュメント:    https://docs.python.org/ja/3/
"""

# === 必要なパッケージのインストール === #
import os
import pandas as pd
import matplotlib.pyplot as plt

# === ファイルパスを変数として格納 ===
current_dir   = os.getcwd()
output_dir_1  = os.path.join(current_dir, "results/csv")
output_dir_2  = os.path.join(current_dir, "results/figure")
file_name     = "ne_vs_metallicity_statistical_results"

data = [[0.008, 0.565, 0.016, 0.022, 2.190, 0.190],
        [0.203,  0.054, -0.508, 0.039,  6.375, 0.315],
        [0.000,  0.000,  0.000, 0.000,  0.000, 0.000],
        [0.000,  0.000,  0.000, 0.000,  0.000, 0.000],
        [0.207, 0.000, 0.682, 0.019, -3.419, 0.165],
        [0.113,  0.339,  0.900, 0.142, -4.955, 1.193],
        [-0.076, 0.731,  0.183, 0.377,  1.390, 3.104],
        [0.400,  0.483,  0.582, 0.563, -1.728, 4.632],
        [0.113, 0.339, 0.016, 0.000, 2.491, 0.017],
        [-0.076, 0.731, 0.016, 0.000, 2.762, 0.050],
        [0.400, 0.483, 0.016, 0.000, 2.931, 0.127],
        [-0.004, 0.958, -0.162, 0.152, 3.407, 1.238],
        [0.113, 0.339, -0.162, 0.000, 3.985, 0.018],
        [-0.076, 0.731, -0.162, 0.000, 4.226, 0.050],
        [0.400, 0.483, -0.162, 0.000, 4.402, 0.128],
        ]

columns = ["tau", "p-value", "slope", "slope_err", "intercept", "intercept_err"]
index = ["SDSS_only", "CLASSY_only", "SDSS_x<10", "SDSS_x>10.5",
         "z~0", "z~3", "z~6", "z~9",
         "z~3 (a=0.016)", "z~6 (a=0.016)", "z~9 (a=0.016)",
         "SDSS_only (direct)", "z~3 (a=-0.162)", "z~6 (a=-0.162)", "z~9 (a=-0.162)"]

df = pd.DataFrame(data, columns=columns, index=index)
# ※ インデックスを CSV に出すなら index=True（デフォルト）
df.to_csv(os.path.join(output_dir_1, f"{file_name}.csv"))
print(df)


# ===== 描画 =====
fig, ax = plt.subplots(figsize=(12, 6))  # 画像サイズ(インチ)
ax.axis("off")  # 軸は非表示

# セルの幅は colWidths で割合指定（列数に合わせる）
table = ax.table(
    cellText=data,
    colLabels=columns,
    rowLabels=index,
    loc="center",
    cellLoc="center",
    colLoc="center",
)

# レイアウト調整
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.0, 1.3)  # (x方向, y方向) 拡大率

plt.tight_layout()
plt.savefig(os.path.join(output_dir_2, f"{file_name}.png"), dpi=200, bbox_inches="tight")
plt.show()

