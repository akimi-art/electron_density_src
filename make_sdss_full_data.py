#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
スクリプトの概要:
今回の研究で必要なすべてのデータを一つのファイルにまとめます。


使用方法:
    make_sdss_full_data.py [オプション]

著者: A. M.
作成日: 2026-02-01

参考文献:
    - PEP 8:                  https://peps.python.org/pep-0008/
    - PEP 257 (Docstring規約): https://peps.python.org/pep-0257/
    - Python公式ドキュメント:    https://docs.python.org/ja/3/
    - Curti+17
"""

"""
luminosity_vs_z_v1.pyを参考に後でflux一定の線を入れよう
"""



# === 必要なモジュールのインポート ===
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u
from astropy.io import fits
from astropy.table import Table

# ----------------------------
# 入力ファイル（ユーザー指定）
# ----------------------------
current_dir = os.getcwd()
data_base_dir = os.path.join(current_dir, "data/data_SDSS/DR7/fits_files")
info_path = os.path.join(data_base_dir, "gal_info_dr7_v5_2.fit")
line_path = os.path.join(data_base_dir, "gal_line_dr7_v5_2.fit")
sfr_path  = os.path.join(data_base_dir, "gal_fibsfr_dr7_v5_2.fits")
oh_path   = os.path.join(data_base_dir, "gal_fiboh_dr7_v5_2.fits")
sm_path   = os.path.join(data_base_dir, "totlgm_dr7_v5_2.fit")

def show_hdus(path):
    with fits.open(path) as hdul:
        hdul.info()

show_hdus(info_path)
show_hdus(line_path)
show_hdus(sfr_path)
show_hdus(oh_path)
show_hdus(sm_path)

# ----------------------------
# 出力ファイル
# ----------------------------
out_fits = "results/fits/mpajhu_dr7_v5_2_merged.fits"
# out_csv  = "results/csv/mpajhu_dr7_v5_2_merged.csv"  # 補助（必要なら）

# ----------------------------
# FITS -> DataFrame（必要列のみ）
# ----------------------------
# def to_native_endian(a):
#     """NumPy配列を native-endian に変換（big-endian対策）"""
#     a = np.asarray(a)
#     if a.dtype.kind in ("S", "a", "U", "O"):  # 文字列/オブジェクトは対象外
#         return a
#     # '=' は native endian
#     if a.dtype.byteorder not in ("=", "|"):  # '|' はエンディアン無関係(1byte型)
#         a = a.byteswap().newbyteorder("=")
#     return a

def to_native_endian(a):
    """NumPy配列を native-endian (=) に揃える。ndarray.newbyteorder()不要の安全版。"""
    a = np.asarray(a)

    # 文字列/オブジェクトはそのまま
    if a.dtype.kind in ("S", "a", "U", "O"):
        return a

    # 1バイト型(|)や既にnative(=)やlittle(<)ならそのまま
    if a.dtype.byteorder in ("=", "<", "|"):
        return a

    # big-endian(>)などを native endian に変換
    return a.astype(a.dtype.newbyteorder("="), copy=False)

def fits_to_df(path, cols=None, hdu=1):
    with fits.open(path, memmap=True) as hdul:
        data = hdul[hdu].data
        if cols is None:
            cols = data.columns.names
        out = {}
        for c in cols:
            arr = data[c]
            # bytes->str の列は従来通り
            if arr.dtype.kind in ("S", "a"):
                out[c] = np.array([x.decode("utf-8", "ignore").strip() for x in arr])
            else:
                # ★ここで native endian 化
                out[c] = to_native_endian(arr)
        return pd.DataFrame(out)

def add_prefix_keep_case(df, prefix):
    """
    列名衝突を避けるため prefix を付ける。
    例: MEDIAN -> sfr_MEDIAN
    """
    return df.rename(columns={c: f"{prefix}{c}" for c in df.columns})

# ----------------------------
# 1) gal_info: 主キー + 基本情報
# 主キーは (PLATEID, MJD, FIBERID)
# ----------------------------
info_cols = [
    "PLATEID", "MJD", "FIBERID",
    "RA", "DEC",
    "Z", "Z_ERR", "Z_WARNING",
    "SN_MEDIAN",
    "E_BV_SFD",
    # 必要なら以下も追加可能
    # "V_DISP", "V_DISP_ERR",
    # "SPECTROTYPE", "SUBCLASS",
]
info_df = fits_to_df(info_path, cols=info_cols)

# ----------------------------
# 2) gal_line: 目的の線 + エラー
# 追加要望: [OIII] 4363 も含める
# ※ gal_line には MJD がないので主キーは info 側を採用
# ----------------------------
line_cols = [
    "PLATEID", "FIBERID",  # 行順結合前提なら最終的に落としてOK（衝突回避）
    "H_ALPHA_FLUX", "H_ALPHA_FLUX_ERR",
    "H_BETA_FLUX",  "H_BETA_FLUX_ERR",
    "OIII_5007_FLUX", "OIII_5007_FLUX_ERR",
    "OIII_4363_FLUX", "OIII_4363_FLUX_ERR",  # ★追加
    "NII_6584_FLUX", "NII_6584_FLUX_ERR",
    "SII_6717_FLUX", "SII_6717_FLUX_ERR",
    "SII_6731_FLUX", "SII_6731_FLUX_ERR",
]
line_df_raw = fits_to_df(line_path, cols=line_cols)

# line側のキー列は、info側の主キーを使うので落とす（列重複回避）
line_df = line_df_raw.drop(columns=["PLATEID", "FIBERID"])

# ----------------------------
# 3) SFR / O/H / Stellar Mass
# いずれもID列がない分布要約のため行順で結合
# 「MEDIANを基本」: MEDIAN + (P16,P84) を残す（不確かさ用）
# ----------------------------
# SFR (fiber)
sfr_keep = ["MEDIAN", "P16", "P84"]
sfr_df = fits_to_df(sfr_path, cols=sfr_keep)
sfr_df = add_prefix_keep_case(sfr_df, "sfr_")

# metallicity O/H (fiber)
oh_keep = ["MEDIAN", "P16", "P84"]
oh_df = fits_to_df(oh_path, cols=oh_keep)
oh_df = add_prefix_keep_case(oh_df, "oh_")

# stellar mass logM* (total)
sm_keep = ["MEDIAN", "P16", "P84"]
sm_df = fits_to_df(sm_path, cols=sm_keep)
sm_df = add_prefix_keep_case(sm_df, "sm_")

# ----------------------------
# 4) 行順で横結合（マッチング・検証は省略）
# ----------------------------
for d in (info_df, line_df, sfr_df, oh_df, sm_df):
    d.reset_index(drop=True, inplace=True)

merged = pd.concat([info_df, line_df, sfr_df, oh_df, sm_df], axis=1)


# ----------------------------
# 5) 便利な派生列（任意）
# S/N列などをここで作ると後が楽
# ----------------------------
merged["SN_HA"] = merged["H_ALPHA_FLUX"] / merged["H_ALPHA_FLUX_ERR"]
merged["SN_HB"] = merged["H_BETA_FLUX"] / merged["H_BETA_FLUX_ERR"]
merged["SN_OIII5007"] = merged["OIII_5007_FLUX"] / merged["OIII_5007_FLUX_ERR"]
merged["SN_OIII4363"] = merged["OIII_4363_FLUX"] / merged["OIII_4363_FLUX_ERR"]
merged["SN_NII6584"] = merged["NII_6584_FLUX"] / merged["NII_6584_FLUX_ERR"]
merged["SN_SII6717"] = merged["SII_6717_FLUX"] / merged["SII_6717_FLUX_ERR"]
merged["SN_SII6731"] = merged["SII_6731_FLUX"] / merged["SII_6731_FLUX_ERR"]

# SII比（neの入口として便利）
merged["R_SII_6717_6731"] = merged["SII_6717_FLUX"] / merged["SII_6731_FLUX"]

# BPT用の線比（log）
# 注意：負やゼロはlog不可なので NaN にする
def safe_log10_ratio(num, den):
    num = np.array(num, dtype=float)
    den = np.array(den, dtype=float)
    out = np.full_like(num, np.nan, dtype=float)
    m = (num > 0) & (den > 0)
    out[m] = np.log10(num[m] / den[m])
    return out

merged["log_NII_HA"] = safe_log10_ratio(merged["NII_6584_FLUX"], merged["H_ALPHA_FLUX"])
merged["log_OIII_HB"] = safe_log10_ratio(merged["OIII_5007_FLUX"], merged["H_BETA_FLUX"])

# ----------------------------
# 6) FITSとして保存
# Astropy Table経由が文字列・NaN等で安定
# ----------------------------
tab = Table.from_pandas(merged, index=False)

# 既存ファイルがある場合は上書き
Path(out_fits).unlink(missing_ok=True)
tab.write(out_fits, format="fits", overwrite=True)

print("✅ Saved merged FITS:", out_fits)
print("Shape:", merged.shape)
print("Columns (head):", merged.columns[:20].tolist())

# ----------------------------
# 7) CSV出力（補助：巨大になるので注意）
# ----------------------------
# merged.to_csv(out_csv, index=False)
# print("✅ Saved CSV:", out_csv)