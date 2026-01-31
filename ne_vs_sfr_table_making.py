#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
スクリプトの概要:
このスクリプトは表全般の作成のテンプレートです。
csv形式でデータを保存する機能と、
図にまとめる機能があります。

使用方法:
    ne_vs_sfr_table_making.py [オプション]

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
file_name     = "ne_vs_sfr_statistical_results"

data = [[0.204, 0.000, 0.206, 0.006, 2.271, 0.003],
        [0.351, 0.001, 0.159, 0.253, 2.196, 0.098],
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
        [0.215, 0.000, 0.223, 0.004, 2.316, 0.002],
        [0.109, 0.339, -0.012, 0.111, 2.633, 0.179],
        [0.121, 0.638, 0.134, 0.238, 2.674, 0.392],
        [0.400, 0.483, 0.527, 0.573, 2.366, 0.770],
        [0.141, 0.223, 0.206, 0.000, 2.262, 0.020],
        [0.121, 0.638, 0.206, 0.000, 2.555, 0.050],
        [0.400, 0.483, 0.206, 0.000, 2.793, 0.127],]


columns = ["tau", "p-value", "slope", "slope_err", "intercept", "intercept_err"]
index = ["SDSS_only", "CLASSY_only", "SDSS_x<10", "SDSS_x>10.5",
         "z~0", "z~3", "z~6", "z~9",
         "z~3 (a=0.206)", "z~6 (a=0.206)", "z~9 (a=0.206)"]

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