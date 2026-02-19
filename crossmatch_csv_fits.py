import pandas as pd
import numpy as np
from astropy.io import fits
from astropy.table import Table

# ============================
# ファイルパス
# ============================
fits_path = "./data/data_SDSS/GALEX/fits_files/hlsp_gswlc_galex-sdss-wise_multi_x1_multi_v1_cat.fits"
csv_path  = "./results/Samir16/Samir16in_standard_re.csv"
output_csv = "./results/Samir16/Samir16in_standard_re_v1.csv"

# ============================
# 1. CSV読み込み
# ============================
df_csv = pd.read_csv(csv_path)
# 型を明示（超重要）
df_csv["PLATE"] = df_csv["PLATE"].astype(int)
df_csv["MJD"] = df_csv["MJD"].astype(int)
df_csv["FIBER_ID"] = df_csv["FIBER_ID"].astype(int)


# ============================
# 2. FITS読み込み
# ============================
t = Table.read(fits_path)

df_fits = t[[
    "PLATE",
    "MJD",
    "FIBER_ID",
    "GLXID",
    "RA",
    "DECL"
]].to_pandas()

df_fits["PLATE"] = df_fits["PLATE"].astype(int)
df_fits["FIBER_ID"] = df_fits["FIBER_ID"].astype(int)
df_fits["MJD"] = df_fits["MJD"].astype(int)



# ============================
# 3. 重複チェック（重要）
# ============================
df_fits = df_fits.sort_values("MJD")
df_fits = df_fits.drop_duplicates(
    subset=["PLATE", "FIBER_ID"],
    keep="first"
)


# ============================
# 4. クロスマッチ（左結合）
# ============================
df_merged = df_csv.merge(
    df_fits,
    on=["PLATE", "FIBER_ID"],
    how="left",
    validate="many_to_one"
)

# ============================
# 5. マッチ数確認
# ============================
n_match = df_merged["GLXID"].notna().sum()
print(f"Matched objects: {n_match} / {len(df_csv)}")


# ============================
# 6. 保存
# ============================
df_merged.to_csv(output_csv, index=False)
print("保存完了:", output_csv)
