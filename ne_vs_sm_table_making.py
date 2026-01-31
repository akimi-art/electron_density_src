#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
スクリプトの概要:
このスクリプトは表全般の作成のテンプレートです。
csv形式でデータを保存する機能と、
図にまとめる機能があります。

使用方法:
    ne_vs_sm_table_making.py [オプション]

著者: A. M.
作成日: 2026-01-18

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
file_name     = "ne_vs_sm_statistical_results"

data = [[0.275, 0.000, 0.222, 0.005, 0.075, 0.053],
        [0.173, 0.099, 0.013, 0.007, 2.187, 0.054],
        [0.120, 0.381, 0.229, 0.436, 0.499, 4.274],
        [0.007, 0.628, -0.015,  0.033, 2.965, 0.364],
        [0.280, 0.000, 0.209, 0.003, 0.252, 0.034],
        [0.017, 0.880, 0.142, 0.034, 1.243, 0.339],
        [0.212, 0.381, 0.098, 0.112, 2.002, 1.024],
        [0.000, 1.000, 0.069, 0.458, 2.457, 4.030],
        [0.017, 0.880, 0.222, 0.000, 0.447, 0.018],
        [0.212, 0.381, 0.222, 0.000, 0.876, 0.051],
        [0.000, 1.000, 0.222, 0.000, 1.111, 0.129],]

columns = ["tau", "p-value", "slope", "slope_err", "intercept", "intercept_err"]
index = ["SDSS_only", "CLASSY_only", "SDSS_x<10", "SDSS_x>10.5",
         "z~0", "z~3", "z~6", "z~9",
         "z~3 (a=0.222)", "z~6 (a=0.222)", "z~9 (a=0.222)",]

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

