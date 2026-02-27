# 必要パッケージ:
# pip install astropy numpy matplotlib

from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.visualization import ZScaleInterval
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy import units as u
import numpy as np
import matplotlib.pyplot as plt
import os
import re

# 軸の設定
plt.rcParams.update({
    # --- 図全体 ---
    "figure.figsize": (12, 6),       # 図サイズ
    "font.size": 20,                 # 全体フォントサイズ
    "axes.labelsize": 24,            # 軸ラベルのサイズ
    "axes.titlesize": 20,            # タイトルのサイズ
    "axes.grid": False,              # グリッドOFF

    # --- 目盛り設定 (ticks) ---
    "xtick.direction": "in",         # x軸目盛りの向き
    "ytick.direction": "in",         # y軸目盛りの向き
    "xtick.top": True,               # 上にも目盛り
    "ytick.right": True,             # 右にも目盛り

    # 主目盛り（major ticks）
    "xtick.major.size": 20,          # 長さ
    "ytick.major.size": 20,
    "xtick.major.width": 2,          # 太さ
    "ytick.major.width": 2,

    # 補助目盛り（minor ticks）
    "xtick.minor.visible": True,     # 補助目盛りON
    "ytick.minor.visible": True,
    "xtick.minor.size": 8,           # 長さ
    "ytick.minor.size": 8,
    "xtick.minor.width": 1.5,        # 太さ
    "ytick.minor.width": 1.5,

    # --- 目盛りラベル ---
    "xtick.labelsize": 20,           # x軸ラベルサイズ
    "ytick.labelsize": 20,           # y軸ラベルサイズ

    # --- フォント ---
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
})

# -----------------------------------------
# 入力パラメータ（例）
# -----------------------------------------
fits_file = "results/SDSS/image/sdss_image_000756-2-0309/frame-i-000756-2-0309.fits"   # 取得した SDSS フレーム
ra_center = 162.16704                           # 中心RA (deg)
dec_center = -0.52525                           # 中心Dec (deg)
cutout_size_arcsec = 20.0                       # 切り出しサイズ（arcsec）
png_output = "results/SDSS/figure/frame-i-000756-2-0309.png"  # 出力 PNG
# -----------------------------------------

# ---- バンド名をファイル名から抽出（ラベル用） ----
m = re.search(r"frame-([ugriz])-", os.path.basename(fits_file))
# band_label = m.group(1).upper() if m else "SDSS" # 大文字にする場合
band_label = m.group(1) if m else "SDSS"

# ---- FITS を読み込み（SDSS frame は通常 Primary HDU に画像/WCS）----
with fits.open(fits_file) as hdul:
    # 画像データは HDU0（Primary）にある場合が多い。保険で 1 も見る
    if hdul[0].data is not None:
        data = hdul[0].data
        header = hdul[0].header
    else:
        data = hdul[1].data
        header = hdul[1].header

# ---- WCS の構築 ----
wcs = WCS(header)

# ---- RA/Dec → pixel ----
center = SkyCoord(ra_center*u.deg, dec_center*u.deg, frame="icrs")
x_center, y_center = wcs.world_to_pixel(center)

# ---- ピクセルスケール（arcsec/pix）を WCS から安全に取得 ----
# SDSS は CDELT ではなく CD/PC 行列のことが多いので proj_plane_pixel_scales を使う
# （単位は度/ピクセル → arcsec に換算）
pix_scales_deg = proj_plane_pixel_scales(wcs)  # [deg/pix] for (x, y)
pixscale_x = np.abs(pix_scales_deg[0]) * 3600.0  # arcsec/pix
pixscale_y = np.abs(pix_scales_deg[1]) * 3600.0

# ---- 切り出しサイズ（ピクセル） ----
nx = int(np.round(cutout_size_arcsec / pixscale_x))
ny = int(np.round(cutout_size_arcsec / pixscale_y))

# ---- 切り出し範囲（境界に注意してクリップ）----
ny_tot, nx_tot = data.shape
x_min = max(0, int(np.floor(x_center - nx/2)))
x_max = min(nx_tot, int(np.ceil (x_center + nx/2)))
y_min = max(0, int(np.floor(y_center - ny/2)))
y_max = min(ny_tot, int(np.ceil (y_center + ny/2)))

cutout = data[y_min:y_max, x_min:x_max]

# ---- NaN や極端値の扱い（表示用）----
cutout = np.asarray(cutout, dtype=float)
# 無限大を NaN に
cutout[~np.isfinite(cutout)] = np.nan

print("data.shape =", data.shape)
print("cutout.shape =", cutout.shape)
print("x_min, x_max, y_min, y_max =", x_min, x_max, y_min, y_max)

n_finite = np.isfinite(cutout).sum()
print("finite pixels in cutout =", int(n_finite))
print("center pixel (x,y) =", x_center, y_center)


# -----------------------------------------
# ZScale（DS9 と同様の見栄え）
# -----------------------------------------
z = ZScaleInterval()
# ZScale は NaN を無視して決めてくれる
vmin, vmax = z.get_limits(cutout)

# -----------------------------------------
# PNG + スケールバー（割合指定）
# -----------------------------------------
fig, ax = plt.subplots(figsize=(3, 3), dpi=150)
ax.imshow(cutout, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
ax.axis("off")

# 1 arcsec の長さ（ピクセル）
one_arcsec_pix_x = 1.0 / pixscale_x   # 横方向 1"
one_arcsec_pix_y = 1.0 / pixscale_y   # 縦方向 1"

h, w = cutout.shape  # 高さ（y）, 幅（x）

# 位置・太さを画像サイズの割合で指定
pos_x_frac = 0.05     # 左から 5 %
pos_y_frac = 0.05     # 下から 5 %
linewidth_frac = 0.05 # 線の太さ（短辺の 1 %）

bar_x_start = w * pos_x_frac
bar_y_start = h * pos_y_frac
lw = max(1, int(min(w, h) * linewidth_frac))

# 横バー（RA方向）
bar_x_end = bar_x_start + one_arcsec_pix_x
ax.plot([bar_x_start, bar_x_end], [bar_y_start, bar_y_start],
        color="white", lw=lw)

# ラベル（1″）
ax.text((bar_x_start + bar_x_end) / 2.0,
        bar_y_start + (0.02 * h),
        '1″', color="white", fontsize=20, ha='center', va='bottom')

# 画像の左上にバンド名を表示
ax.text(w * pos_x_frac,
        h * (1 - pos_y_frac),
        f'{band_label}',
        color="white", fontsize=20, ha='left', va='top')

plt.savefig(png_output, dpi=300, bbox_inches="tight", pad_inches=0)
plt.show()
print(f"Saved: {png_output}")