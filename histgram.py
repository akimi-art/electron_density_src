# 確認用（好きに変えてOKのコード）
# === 必要なパッケージのインストール === #
import os
import numpy as np
import pandas as pd
import seaborn as sns
from astropy.io import fits
import matplotlib.pyplot as plt

# # CSV読み込み
# current_dir = os.getcwd()
# df = pd.read_csv(os.path.join(current_dir, "results/csv/sdss_dr7_curti17_direct_Te_N2_O3_data_with_ne.csv"))

# # SN比を計算（ゼロ割り回避のため、ERR<=0 は NaN にする）
# sn = df["OIII_4363_FLUX"] / df["OIII_4363_FLUX_ERR"].where(df["OIII_4363_FLUX_ERR"] > 0)

# # NaN/inf を除去
# sn = sn.replace([np.inf, -np.inf], np.nan).dropna()

# # ヒストグラム
# sns.histplot(sn, bins=1000)
# plt.xlim(0, 10)
# plt.xlabel("S/N = OIII_4363_FLUX / OIII_4363_FLUX_ERR")
# plt.ylabel("Count")
# plt.title("[OIII] 4363 S/N histogram")
# plt.tight_layout()
# plt.show()


# # 4363 SN
# sn4363 = df["OIII_4363_FLUX"] / df["OIII_4363_FLUX_ERR"].where(df["OIII_4363_FLUX_ERR"] > 0)
# sn4363 = sn4363.replace([np.inf, -np.inf], np.nan)

# print("N total:", len(df))
# print("N valid:", sn4363.notna().sum())
# print("frac(SN>3):", (sn4363 > 3).mean())
# print("frac(SN>5):", (sn4363 > 5).mean())

# # 分布の典型値をざっくり 
# print("\nFlux stats (4363):")
# print(df["OIII_4363_FLUX"].describe(percentiles=[.16,.5,.84]))
# print("\nErr stats (4363):")
# print(df["OIII_4363_FLUX_ERR"].describe(percentiles=[.16,.5,.84]))


# sn5007 = df["OIII_5007_FLUX"] / df["OIII_5007_FLUX_ERR"].where(df["OIII_5007_FLUX_ERR"] > 0)
# sn5007 = sn5007.replace([np.inf, -np.inf], np.nan)

# print("median SN(5007):", np.nanmedian(sn5007))
# print("frac SN(5007)>10:", (sn5007 > 10).mean())

# plt.scatter(sn5007, sn4363, s=3, alpha=0.3)
# plt.xscale("log")
# plt.yscale("log")
# plt.xlabel("SN(5007)")
# plt.ylabel("SN(4363)")
# plt.show()


# === ファイルパスを取得する === #
current_dir = os.getcwd()
file_fits = os.path.join(current_dir, "data/data_SDSS/DR7/fits_files/gal_fiboh_dr7_v5_2.fits")

# === FITSファイルを開く === #
# 重要な情報はhdul[1]の方にのっている
with fits.open(file_fits) as hdul:
    # HDUの構造を表示
    hdul.info()

    # 拡張HDU（通常 index 1）を取得
    ext_hdu = hdul[1]
    # data = hdul[1].data

    # 列名とデータ型を表示
    print("列名:", ext_hdu.columns.names)
    print("データ型:", ext_hdu.columns.formats)
        
    # 金属量のデータをfloatとして書き出す
    METALLICITY_MEDIAN = ext_hdu.data['MEDIAN'].astype(float)

# ヒストグラム
sns.histplot(METALLICITY_MEDIAN, bins=3000)
plt.xlim(6, 10)
plt.ylim(0, 20000)
plt.xlabel("12+log(O/H)")
plt.ylabel("Count")
plt.title("SDSS DR7 Metallicity histogram")
plt.tight_layout()
savepath = os.path.join(current_dir, "results/figure/SDSS_DR7_full_metallicity_histogram")
plt.savefig(savepath)
print(f"Saved as {savepath}")
plt.show()