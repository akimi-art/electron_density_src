#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
スクリプトの概要:
このスクリプトはJWST NIRCamで撮影した画像のカットアウトをします。

使用方法:
    jwst_mircam_cutout.py [オプション]

著者: A. M.
作成日: 2026-02-26

参考文献:
    - PEP 8:                  https://peps.python.org/pep-0008/
    - PEP 257 (Docstring規約): https://peps.python.org/pep-0257/
    - Python公式ドキュメント:    https://docs.python.org/ja/3/
"""

from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.visualization import ZScaleInterval
from astropy import units as u
import numpy as np
import matplotlib.pyplot as plt


# 軸の設定
plt.rcParams.update({
    # --- 図全体 ---
    "figure.figsize": (12, 6),       # 図サイズ
    "font.size": 16,                 # 全体フォントサイズ
    "axes.labelsize": 16,            # 軸ラベルのサイズ
    "axes.titlesize": 16,            # タイトルのサイズ
    "axes.grid": False,              # グリッドOFF

    # --- 目盛り設定 (ticks) ---
    "xtick.direction": "in",         # x軸目盛りの向き
    "ytick.direction": "in",         # y軸目盛りの向き
    "xtick.top": False,               # 上にも目盛り
    "ytick.right": False,             # 右にも目盛り

    # 主目盛り（major ticks）
    "xtick.major.size": 16,          # 長さ
    "ytick.major.size": 16,
    "xtick.major.width": 2,          # 太さ
    "ytick.major.width": 2,

    # 補助目盛り（minor ticks）
    "xtick.minor.visible": False,     # 補助目盛りON
    "ytick.minor.visible": False,
    "xtick.minor.size": 8,           # 長さ
    "ytick.minor.size": 8,
    "xtick.minor.width": 1.5,        # 太さ
    "ytick.minor.width": 1.5,

    # --- 目盛りラベル ---
    "xtick.labelsize": 16,           # x軸ラベルサイズ
    "ytick.labelsize": 16,           # y軸ラベルサイズ

    # --- フォント ---
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
})



# 例：
# NIRSpec_ID,z_Spec,RA_NIRCam,Dec_NIRCam
# 3892,2.807246426407744,53.16255540472612,-27.816609445453008
# -----------------------------------------
# 入力パラメータ
# -----------------------------------------
fits_file = "results/JADES/NIRCam/fits/hlsp_jades_jwst_nircam_goods-s-deep_f277w_v2.0/hlsp_jades_jwst_nircam_goods-s-deep_f277w_v2.0_drz.fits"    # ダウンロードした FITS
ra_center = 53.16255540472612                # 中心RA (deg)
dec_center = -27.816609445453008             # 中心Dec (deg)
cutout_size_arcsec = 5.0               # 切り出しサイズ（arcsec）
png_output = "results/JADES/NIRCam/figure/JADES_DR3_galaxy_ID00003892.png"               # PNG 出力先
# -----------------------------------------

# FITS 読み込み
hdu = fits.open(fits_file)
data = hdu[1].data
header = hdu[1].header

# WCS
wcs = WCS(header)

# RA/Dec → pixel
center = SkyCoord(ra_center*u.deg, dec_center*u.deg)
x_center, y_center = wcs.world_to_pixel(center)

# pixel scale (arcsec/pixel)
pixscale_x = np.abs(header["CDELT1"]) * 3600
pixscale_y = np.abs(header["CDELT2"]) * 3600

# 切り出しピクセルサイズ
nx = int(cutout_size_arcsec / pixscale_x)
ny = int(cutout_size_arcsec / pixscale_y)

# 切り出し範囲
x_min = int(x_center - nx/2)
x_max = int(x_center + nx/2)
y_min = int(y_center - ny/2)
y_max = int(y_center + ny/2)

cutout = data[y_min:y_max, x_min:x_max]

# -----------------------------------------
# ZScale
# -----------------------------------------
z = ZScaleInterval()
vmin, vmax = z.get_limits(cutout)

# -----------------------------------------
# PNG + スケールバー
# -----------------------------------------
fig, ax = plt.subplots(figsize=(3,3))
ax.imshow(cutout, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
ax.axis("off")

# ---- 1 arcsec のスケールバー ----
one_arcsec_pix_x = 1.0 / pixscale_x   # 横方向 1"
one_arcsec_pix_y = 1.0 / pixscale_y   # 縦方向 1"

h, w = cutout.shape  # 高さ（y）, 幅（x）

# ------------------------------------------------------------
# ◇ 位置・長さ・太さを、画像サイズの「割合」で指定する
# ------------------------------------------------------------
pos_x_frac = 0.05    # 画像幅の 50% の位置にバーを置く（左から）
pos_y_frac = 0.05    # 画像高さの 5% の位置にバーを置く（下から）
linewidth_frac = 0.03  # 画像サイズに対する線の太さ（0.3%）

# ピクセル値に変換
bar_x_start = w * pos_x_frac
bar_y_start = h * pos_y_frac
lw = max(1, int(min(w, h) * linewidth_frac))

# 横バー（RA方向）
bar_x_end = bar_x_start + one_arcsec_pix_x
ax.plot([bar_x_start, bar_x_end],
        [bar_y_start, bar_y_start],
        color="white", lw=lw)

ax.text((bar_x_start +bar_x_end) / 2,
        bar_y_start + (0.01*h),
        '1″', color="white", fontsize=24,
        ha='center', va='bottom')

# # ------------------------------------------------------------
# # ◇ オプション：縦バーも割合で指定
# # ------------------------------------------------------------
# bar_x_v = bar_x_start
# bar_y_v_start = bar_y_start + (0.02 * h)  # 横バーより少し上（画像高さの2%）
# bar_y_v_end = bar_y_v_start + one_arcsec_pix_y

# ax.plot([bar_x_v, bar_x_v],
#         [bar_y_v_start, bar_y_v_end],
#         color="white", lw=lw)

# ax.text(bar_x_v + (0.01*w),
#         bar_y_v_end + (0.01*h),
#         '1″', color="white", fontsize=24,
#         ha='left', va='bottom')


# フィルターの名前を画像の左上にテキストで表示
ax.text(bar_x_start,
        h * (1 - pos_y_frac),
        'F277W', color="white", fontsize=24,
        ha='left', va='top')

plt.savefig(png_output, dpi=300, bbox_inches="tight", pad_inches=0)
plt.show()

print("Saved:", png_output)