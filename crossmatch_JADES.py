import pandas as pd

# =========================
# 読み込み（最初から型指定が安全）
# =========================
df_sii = pd.read_csv("./results/csv/JADES_DR3_SII_ratio_only_merged.csv")
df_target = pd.read_csv("./results/JADES/JADES_DR3/data_from_Nishigaki/jades_info_crossmatch.csv")
df_sii = df_sii[df_sii["NIRSpec_ID"] != "NIRSpec_ID"]
df_target = df_target[df_target["NIRSpec_ID"] != "NIRSpec_ID"]
df_sii["NIRSpec_ID"] = df_sii["NIRSpec_ID"].astype(int)
df_target["NIRSpec_ID"] = df_target["NIRSpec_ID"].astype(int)


# =========================
# 必要な列だけ抽出
# =========================
cols_to_add = [
    "NIRSpec_ID",
    "S2_6716_flux",
    "S2_6716_err_minus",
    "S2_6716_err_plus",
    "S2_6730_flux",
    "S2_6730_err_minus",
    "S2_6730_err_plus",
    "ratio_median",
    "ratio_minus",
    "ratio_plus"
]

df_sii_small = df_sii[cols_to_add].copy()

# =========================
# クロスマッチ（左結合）
# =========================
df_merged = df_target.merge(
    df_sii_small,
    on="NIRSpec_ID",
    how="left",
)

# =========================
# マッチ数確認
# =========================
n_matched = df_merged["ratio_median"].notna().sum()
print(f"Matched objects: {n_matched} / {len(df_target)}")

# =========================
# 保存
# =========================
df_merged.to_csv("./results/JADES/JADES_DR3/data_from_Nishigaki/jades_info_crossmatch_fit.csv", index=False)

print("Crossmatch completed.")
