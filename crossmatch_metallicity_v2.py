#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
スクリプトの概要:
csvファイルとcsvファイルの
クロスマッチしてcsvファイルに
金属量を導出するためのの情報を追加します。

使用方法:
    crossmatch_metallicity_v2.py [オプション]

著者: A. M.
作成日: 2026-01-23

参考文献:
    - PEP 8:                  https://peps.python.org/pep-0008/
    - PEP 257 (Docstring規約): https://peps.python.org/pep-0257/
    - Python公式ドキュメント:    https://docs.python.org/ja/3/
"""


# === 必要なパッケージのインストール === #
import os
import numpy as np
import pandas as pd

# ===============================
# 入力ファイル（パスはあなたの環境に合わせて変更）
# ===============================
current_dir = os.getcwd()

# A) FITSから抽出した emission line のCSV
flux_csv = os.path.join(current_dir, "results/csv/sdss_dr7_curti17_direct_Te_N2_O3_data.csv")

# B) ne + plate/fiber のCSV（ユーザー指定）
ne_csv = os.path.join(current_dir, "results/csv/Samir16out_standard_v3_ms_only_v3_ne_plate_fiber.csv")

# 出力
out_csv = os.path.join(current_dir, "results/csv/sdss_dr7_curti17_direct_Te_N2_O3_data_with_ne.csv")

# ===============================
# 読み込み
# ===============================
df_flux = pd.read_csv(flux_csv)
df_ne   = pd.read_csv(ne_csv)

# ===============================
# 型の統一（超重要：merge失敗の主因）
# ===============================
df_flux["PLATEID"] = df_flux["PLATEID"].astype(int)
df_flux["FIBERID"] = df_flux["FIBERID"].astype(int)

df_ne["PLATEID"] = df_ne["PLATEID"].astype(int)
df_ne["FIBERID"] = df_ne["FIBERID"].astype(int)

# ===============================
# ne列の名前に対応（logne_med か ne_cm3 のどちらかがある想定）
# ===============================
# よくある候補名に対応しておく（必要なら追加してください）
possible_log_cols = ["logne_med", "log_ne", "logne", "log_ne_med"]
possible_ne_cols  = ["ne_cm3", "ne", "ne_med", "ne_linear"]

log_col = next((c for c in possible_log_cols if c in df_ne.columns), None)
ne_col  = next((c for c in possible_ne_cols  if c in df_ne.columns), None)

if ne_col is None:
    # ne（線形）が無いなら、logneから作る
    if log_col is None:
        raise ValueError(
            f"ne CSV に ne も log(ne) も見つかりません。列名を確認してください。columns={list(df_ne.columns)}"
        )
    df_ne["ne_cm3"] = 10 ** df_ne[log_col]
    ne_col = "ne_cm3"

# logne も残したい場合（任意）
if log_col is None:
    # logne が無ければ線形から作る（任意：いらなければ消してOK）
    df_ne["logne_med"] = np.log10(df_ne[ne_col])
    log_col = "logne_med"

# ===============================
# 重複がある場合の処理（plate,fiberがユニークでないとmergeで行が増える）
# ===============================
df_ne = df_ne.drop_duplicates(subset=["PLATEID", "FIBERID"], keep="first")

# ===============================
# inner join（一致した行だけ残す）
# ===============================
df_merged = pd.merge(
    df_flux,
    df_ne[["PLATEID", "FIBERID", log_col, ne_col]],
    on=["PLATEID", "FIBERID"],
    how="inner"
)

# ===============================
# 保存
# ===============================
os.makedirs(os.path.dirname(out_csv), exist_ok=True)
df_merged.to_csv(out_csv, index=False)

print("Saved:", out_csv)
print("flux rows:", len(df_flux), "ne rows:", len(df_ne), "merged rows:", len(df_merged))
print(df_merged.head())
