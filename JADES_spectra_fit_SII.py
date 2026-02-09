#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
スクリプトの概要:
このスクリプトは
JADESのカタログを使って
スペクトル (SII) をフィッティングするものです。

使用方法:
    JADES_spectra_fit_SII.py [オプション]

著者: A. M.
作成日: 2026-02-03

参考文献:
    - PEP 8: https://peps.python.org/pep-0008/
    - PEP 257 (Docstring規約): https://peps.python.org/pep-0257/
    - Python公式ドキュメント: https://docs.python.org/ja/3/
"""

# == 必要なパッケージのインストール == #
import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import emcee
import corner
import pyneb as pn
from astropy.io import fits
from astropy.visualization import ZScaleInterval
from scipy.optimize import curve_fit
from matplotlib.gridspec import GridSpec

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


# =========================
# 1. CSV を読む
# =========================
current_dir = os.getcwd()
csv_path = os.path.join(current_dir, "results/csv/JADES_ne_candidates_GOODS_S_v1.1.csv")
df = pd.read_csv(csv_path)

# df.iloc[0]: 「1行目（最初の1天体）」を取り出す
nir_id = df.iloc[16]["NIRSpec_ID"] 
z_spec = df.iloc[16]["z_Spec"]
nir_id_str = f"{int(nir_id):08d}"
z_spec_str = f"{float(z_spec):.3f}"
z_fix = z_spec
print(z_spec)
print(nir_id)
print(nir_id_str)

# =========================
# 基本設定
# =========================
wave_length_6716 = 6716.440  # Å
wave_length_6730 = 6730.820  # Å
wave_center_s2 = ((wave_length_6716 + wave_length_6730) / 2) * (1 + z_spec) # Å
def nirspec_sigma(wavelength_A, R=1000.0):
    """
    Compute Gaussian sigma [Å] for JWST/NIRSpec given wavelength [Å] and resolving power R.
    Assumes FWHM = λ/R and sigma = FWHM/2.355.
    """
    fwhm_A = wavelength_A / R
    sigma_A = fwhm_A / 2.355
    return sigma_A, fwhm_A

sigma, fwhm = nirspec_sigma(wave_center_s2, R=1000.0)
print(f"λ = {wave_center_s2:.1f} Å, R = 1000 -> FWHM = {fwhm:.3f} Å, σ = {sigma:.3f} Å")
delta_lambda = 300.0           # fit 幅（Å）
sigma_instr = sigma            # 固定（後で grating 依存にしてOK）
if 7000.0 < wave_center_s2 < 18893.643160000556:
    filter_grating = "f070lp-g140m" # ここにフィルターグレーティング情報を追加
elif 16600.0 < wave_center_s2 < 31693.3834:
    filter_grating = "f170lp-g235m" # ここにフィルターグレーティング情報を追加
elif 28700.0 < wave_center_s2 < 52687.212:
    filter_grating = "f290lp-g395m" # ここにフィルターグレーティング情報を追加
print(filter_grating)

# =========================
# 2. スペクトル取得
# =========================
base = f"results/JADES/individual/JADES_{nir_id_str}"
# x1d = glob.glob(f"{base}/**/*_x1d.fits", recursive=True)[1] # ここでフィルターグレーディングを調整する
# s2d = glob.glob(f"{base}/**/*_s2d.fits", recursive=True)[1] # ここでフィルターグレーディングを調整する

x1d_files = glob.glob(
    os.path.join(base, "**", f"*{filter_grating}*_x1d.fits"),
    recursive=True
)

s2d_files = glob.glob(
    os.path.join(base, "**", f"*{filter_grating}*_s2d.fits"),
    recursive=True
)

x1d_files.sort()
s2d_files.sort()
x1d = x1d_files[0]
s2d = s2d_files[0]
print(f"{filter_grating}: {len(x1d)} spectra found")

# =========================
# Gaussian
# =========================
def gaussian(x, amp, mu, sigma):
    return amp * np.exp(-(x-mu)**2/(2*sigma**2)) / (np.sqrt(2*np.pi)*sigma)

# =========================
# SII model（z 固定）
# =========================
def s2_doublet_model(x, amp_6716, amp_6730,  z, sigma_int, bg):
    mu_6716 = wave_length_6716 * (1 + z)
    mu_6730 = wave_length_6730 * (1 + z)
    sigma_total = np.sqrt(sigma_int**2 + sigma_instr**2)
    f6716 = gaussian(x, amp_6716, mu_6716, sigma_total)
    f6730 = gaussian(x, amp_6730, mu_6730, sigma_total)

    return f6716 + f6730 + bg

def s2_doublet_model_6716(x, amp_6716, amp_6730, z, sigma_int, bg):
    mu_6716 = wave_length_6716 * (1 + z)
    mu_6730 = wave_length_6730 * (1 + z)
    sigma_total = np.sqrt(sigma_int**2 + sigma_instr**2)
    f6716 = gaussian(x, amp_6716, mu_6716, sigma_total)
    f6730 = gaussian(x, amp_6730, mu_6730, sigma_total)

    return f6716 + bg

def s2_doublet_model_6730(x, amp_6716, amp_6730, z, sigma_int, bg):
    mu_6716 = wave_length_6716 * (1 + z)
    mu_6730 = wave_length_6730 * (1 + z)
    sigma_total = np.sqrt(sigma_int**2 + sigma_instr**2)
    f6716 = gaussian(x, amp_6716, mu_6716, sigma_total)
    f6730 = gaussian(x, amp_6730, mu_6730, sigma_total)

    return f6730 + bg

# =========================
# 3. 1D スペクトル
# =========================
with fits.open(x1d) as hdul:
    tab = hdul["EXTRACT1D"].data
    wave_1d = tab["WAVELENGTH"] * 1e4
    flux_1d = tab["FLUX"] * 1e19
    err_1d  = tab["FLUX_ERR"] * 1e19

# =========================
# 4. 2D スペクトル
# =========================
with fits.open(s2d) as hdul:
    flux_2d = hdul["FLUX"].data
    wave_2d = hdul["WAVELENGTH"].data * 1e4

# =========================
# 5. mask 定義（z 使用）
# =========================
wave_center_s2 = 0.5 * (wave_length_6716 + wave_length_6730) * (1 + z_fix)

mask_1d = (
    (wave_1d > wave_center_s2 - delta_lambda) &
    (wave_1d < wave_center_s2 + delta_lambda)
)

print("SII center =", wave_center_s2)
print("wave_2d range =", wave_2d.min(), wave_2d.max())

x_fit = wave_1d[mask_1d]
y_fit = flux_1d[mask_1d]
yerr_fit = err_1d[mask_1d]

# =========================
# 6. フィッティング
# =========================
# === 最適化パラメータの初期値を設定する ===

amplitude_6716_init = 20
amplitude_6730_init = 20
z_init = z_spec # 変更
sigma_int_init = 10 # 適当, 目安がわからないのでLSFに合わせた
bgd_s2_mask_init = 0
p0 = [amplitude_6716_init, amplitude_6730_init, z_init, sigma_int_init, bgd_s2_mask_init]

popt, pcov = curve_fit(
    s2_doublet_model,
    x_fit, y_fit,
    p0=p0,
    sigma=yerr_fit,
    absolute_sigma=True
)

amp_6716, amp_6730, z, sigma_int, bg = popt

ratio = amp_6716 / amp_6730
print(f"[S II] 6716/6730 = {ratio:.3f}")
print(popt)

# =========================
# 7. プロット
# =========================
fig = plt.figure(figsize=(12,6))
gs = GridSpec(2,1,height_ratios=[1,5],hspace=0)

ax2d = fig.add_subplot(gs[0])
ax1d = fig.add_subplot(gs[1], sharex=ax2d)

# ---- 2D ----
flux_2d_cut = flux_2d[:, mask_1d]
wave_2d_cut = wave_2d[mask_1d]

zscale = ZScaleInterval()
vmin, vmax = zscale.get_limits(flux_2d_cut[np.isfinite(flux_2d_cut)])

ax2d.imshow(
    flux_2d_cut,
    origin="lower",
    aspect="auto",
    cmap="plasma",
    vmin=vmin, vmax=vmax,
    extent=[wave_2d_cut.min(), wave_2d_cut.max(), 0, flux_2d_cut.shape[0]]
)

# ---- 1D ----
ax1d.step(x_fit, y_fit, where="mid", color="black")
ax1d.fill_between(
    x_fit,
    y_fit-yerr_fit,
    y_fit+yerr_fit,
    step="mid",
    color="gray", alpha=0.4
)

x_model = np.linspace(x_fit.min(), x_fit.max(), 1000)
ax1d.plot(x_model, s2_doublet_model(x_model, *popt), color="red", lw=2)
ax1d.plot(x_model, s2_doublet_model_6716(x_model, *popt), color="red", lw=2, ls="--", label="SII 6716")
ax1d.plot(x_model, s2_doublet_model_6730(x_model, *popt), color="red", lw=2, ls="-.", label="SII 6730")
ax1d.legend(fontsize=16)
mu_6716 = wave_length_6716 * (1 + z)
mu_6730 = wave_length_6730 * (1 + z)
ax1d.axvline(mu_6716, color="red", ls="--")
ax1d.axvline(mu_6730, color="red", ls="-.")
ax1d.set_xlabel(r'$\lambda (Å)$')
ax1d.set_ylabel(r'F$_{\lambda}$ ($10^{-19}$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$)')
# === 枠線 (spines) の設定 ===
# 線の太さ・色・表示非表示などを個別に制御
for spine in ax1d.spines.values():
    spine.set_linewidth(2)       # 枠線の太さ
    spine.set_color("black")     # 枠線の色

ax2d.set_title(f"JADES NIRSpec {filter_grating} |  ID {nir_id_str} | z_spec = {z_spec_str}")
ax2d.tick_params(axis='both', which='both',
               bottom=False, top=False, left=False, right=False,
               labelbottom=False, labeltop=False,
               labelleft=False, labelright=False)
# === 枠線 (spines) の設定 ===
# 線の太さ・色・表示非表示などを個別に制御
for spine in ax2d.spines.values():
    spine.set_linewidth(2)       # 枠線の太さ
    spine.set_color("black")     # 枠線の色

save_path = os.path.join(current_dir, f"results/figure/JADES/JADES_NIRSpec_{filter_grating}_ID{nir_id_str}_fit.png")
plt.savefig(save_path)
print(f"Saved as {save_path}")
plt.show()


# =========================
# 8. MCMC
# =========================

# --- log prior ---
def log_prior(theta):
    amp_6716, amp_6730, z, sigma_int, bg = theta

    # 物理的制限
    if amp_6716 <= 0: return -np.inf
    if amp_6730 <= 0: return -np.inf
    if sigma_int <= 0: return -np.inf
    if not (z_fix-0.01 < z < z_fix+0.01): return -np.inf

    # 弱い一様事前
    return 0.0


# --- log likelihood ---
def log_likelihood(theta, x, y, yerr):
    model = s2_doublet_model(x, *theta)
    return -0.5 * np.sum(((y - model)/yerr)**2)


# --- posterior ---
def log_probability(theta, x, y, yerr):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y, yerr)


# =========================
# 初期値（curve_fit 結果の近傍）
# =========================
ndim = 5
nwalkers = 32

pos = popt + 1e-3 * np.random.randn(nwalkers, ndim)

sampler = emcee.EnsembleSampler(
    nwalkers,
    ndim,
    log_probability,
    args=(x_fit, y_fit, yerr_fit)
)

print("Running MCMC...")
sampler.run_mcmc(pos, 4000, progress=True)

# =========================
# バーンイン除去
# =========================
burnin = 1000
flat_samples = sampler.get_chain(discard=burnin, thin=10, flat=True)

print("MCMC done.")


amp_6716_samples = flat_samples[:,0]
amp_6730_samples = flat_samples[:,1]

ratio_samples = amp_6716_samples / amp_6730_samples

ratio_median = np.percentile(ratio_samples, 50)
ratio_low = ratio_median - np.percentile(ratio_samples, 16)
ratio_high = np.percentile(ratio_samples, 84) - ratio_median

print(f"[S II] 6716/6730 = {ratio_median:.3f} +{ratio_high:.3f} -{ratio_low:.3f}")

fig = corner.corner(
    flat_samples,
    labels=["amp6716","amp6730","z","sigma_int","bg"],
    truths=popt
)
plt.show()

save_dir = os.path.join(current_dir, f"results/JADES/parameters/{nir_id_str}")
os.makedirs(save_dir, exist_ok=True)

# =========================
# パラメータの保存
# =========================
# 推定パラメータ（中央値＋誤差）を保存
labels = ["amp_6716", "amp_6730", "z", "sigma_int", "bg"]

results = {}

for i, label in enumerate(labels):
    q16, q50, q84 = np.percentile(flat_samples[:, i], [16, 50, 84])
    results[label] = {
        "median": q50,
        "err_minus": q50 - q16,
        "err_plus": q84 - q50
    }

# ratio も追加
ratio_samples = flat_samples[:,0] / flat_samples[:,1]
q16, q50, q84 = np.percentile(ratio_samples, [16, 50, 84])
results["SII_ratio"] = {
    "median": q50,
    "err_minus": q50 - q16,
    "err_plus": q84 - q50
}

df_results = pd.DataFrame(results).T
df_results.to_csv(
    os.path.join(save_dir, f"SII_MCMC_results_ID{nir_id_str}.csv")
)

# コーナープロットの保存
fig = corner.corner(
    flat_samples,
    labels=["amp6716","amp6730","z","sigma_int","bg"],
    truths=popt,
    show_titles=True
)

corner_path = os.path.join(
    save_dir,
    f"SII_corner_ID{nir_id_str}.png"
)

plt.savefig(corner_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print("Saved:", corner_path)

fig_ratio = corner.corner(
    np.column_stack([ratio_samples]),
    labels=["SII 6716/6730"]
)

# ratio だけの corner
ratio_corner_path = os.path.join(
    save_dir,
    f"SII_ratio_corner_ID{nir_id_str}.png"
)

plt.savefig(ratio_corner_path, dpi=300, bbox_inches="tight")
plt.close(fig_ratio)

print("Saved:", ratio_corner_path)


# =========================
# 9. neの計算
# =========================
S2 = pn.Atom("S", 2)  # S II
Te = 15000.0          # 仮定電子温度（あとで拡張可能）

# === [S II] の比データ（適宜変更） ===
median = results["SII_ratio"]["median"]
err_minus = results["SII_ratio"]["err_minus"]
err_plus  = results["SII_ratio"]["err_plus"]

# === PyNeb オブジェクト作成 ===
S2 = pn.Atom('S', 2)

# === Te設定 ===
Te = 15000 # K （適宜変更）

# === 比を使って電子密度を推定 ===
ne_median_s2 = S2.getTemDen(int_ratio=median, tem=Te, wave1=6716, wave2=6731)
ne_upper_s2 = S2.getTemDen(int_ratio=median-err_minus, tem=Te, wave1=6716, wave2=6731)
ne_lower_s2 = S2.getTemDen(int_ratio=median+err_plus, tem=Te, wave1=6716, wave2=6731)

# === 結果表示 ===
print(f"[S II] 6716/6731 = {median:.3f}+{err_plus:.3f}-{err_minus:.3f}")
print(f"Estimated ne_s2 = {ne_median_s2:.3f}+{ne_upper_s2-ne_median_s2:.3f}-{ne_median_s2-ne_lower_s2:.3f}")

kv = pd.DataFrame({
    "name": ["ne(SII) median", "ne(SII) upper", "ne (SII) lower"],
    "value": [ne_median_s2, ne_upper_s2-ne_median_s2, ne_median_s2-ne_lower_s2]   # 配列は list にして格納
})

kv.to_csv(os.path.join(
    save_dir,
    f"ne_SII_ID{nir_id_str}.csv"
), index=False)