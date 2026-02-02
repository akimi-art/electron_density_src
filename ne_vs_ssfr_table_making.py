#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
スクリプトの概要:
このスクリプトは表全般の作成のテンプレートです。
csv形式でデータを保存する機能と、
図にまとめる機能があります。

使用方法:
    ne_vs_ssfr_table_making.py [オプション]

著者: A. M.
作成日: 2026-01-22

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
file_name     = "ne_vs_ssfr_statistical_results"

data = [[-0.274, 0.000, -0.627, 0.014, -3.846, 0.138],
        [ 0.331, 0.002, -0.203, 0.719,  0.565, 5.661],
        [0.196, 0.087, -0.627, 0.000, -2.557, 0.029],
        [-0.121, 0.638, -0.627, 0.000, -1.801, 0.057],
        [0.400, 0.483, -0.627, 0.000, -1.623, 0.130],
        [-0.361, 0.000, -0.297, 0.002, -0.537, 0.024],
        [0.196, 0.087, -0.297, 0.000, 0.142, 0.023],
        [-0.121, 0.638, -0.297, 0.000, 0.670, 0.051],
        [0.400, 0.483, -0.297, 0.000, 0.848, 0.127]]


columns = ["tau", "p-value", "slope", "slope_err", "intercept", "intercept_err"]
index = ["SDSS_only", "CLASSY_only", 
         "z~3 (a=0.206)", "z~6 (a=0.206)", "z~9 (a=0.206)",
         "SDSS_only (all)", "z~3 (a=-0.297)", "z~6 (a=-0.297)", "z~9 (a=-0.297)"]

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

plt.savefig(os.path.join(output_dir_2, f"{file_name}.png"), dpi=200, bbox_inches="tight")
plt.show()