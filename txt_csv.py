import pandas as pd
import numpy as np
import re

# ============================
# 入力ファイル
# ============================
input_txt = "./results/Samir16/Samir16in_standard_v7.txt"
output_csv = "./results/Samir16/Samir16in_standard.csv"

# ============================
# 読み込み（ヘッダーなし）
# ============================
df = pd.read_csv(
    input_txt,
    delim_whitespace=True,
    header=None,
    dtype=str  # まずは全部文字列で読む
)

# 列名を col1, col2, ...
df.columns = [f"col{i+1}" for i in range(df.shape[1])]

# ============================
# 1. col13~16を削除
# ============================
cols_to_drop = ["col13", "col14", "col15", "col16"]
df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

# ============================
# 2. median+err_plus-err_minus を分解
# 対象: col4~8, col10
# ============================
cols_to_split = ["col4", "col5", "col6", "col7", "col8", "col10"]

def split_value(val):
    """
    '2.839641+0.418761-0.381423'
    → median, err_plus, err_minus
    """
    if pd.isna(val):
        return np.nan, np.nan, np.nan

    # -99.9000+-99.9000--99.9000 対策
    pattern = r"([+-]?\d+\.?\d*)\+([+-]?\d+\.?\d*)-([+-]?\d+\.?\d*)"
    match = re.match(pattern, val)

    if match:
        return float(match.group(1)), float(match.group(2)), float(match.group(3))
    else:
        return np.nan, np.nan, np.nan


for col in cols_to_split:
    if col in df.columns:
        med_list = []
        plus_list = []
        minus_list = []

        for v in df[col]:
            m, p, mi = split_value(v)
            med_list.append(m)
            plus_list.append(p)
            minus_list.append(mi)

        df[f"{col}_median"] = med_list
        df[f"{col}_err_plus"] = plus_list
        df[f"{col}_err_minus"] = minus_list

        df = df.drop(columns=[col])

# ============================
# 3. 型変換
# ============================

# int列
int_cols = ["col1", "col2", "col9", "col11", "col12"]

for col in int_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

# 残りは float
for col in df.columns:
    if col not in int_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# ============================
# 保存
# ============================
df.to_csv(output_csv, index=False)

print("変換完了 →", output_csv)
