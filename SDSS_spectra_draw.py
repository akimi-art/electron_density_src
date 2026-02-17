#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
スクリプトの概要:
このスクリプトは
SDSSのカタログを使って
スペクトル（1d）を描画するものです。

使用方法:
    SDSS_spectra_draw.py [オプション]

著者: A. M.
作成日: 2026-02-17

参考文献:
    - PEP 8: https://peps.python.org/pep-0008/
    - PEP 257 (Docstring規約): https://peps.python.org/pep-0257/
    - Python公式ドキュメント: https://docs.python.org/ja/3/
"""


# == 必要なパッケージのインストール == #
import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

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

current_dir = os.getcwd()
fits_dir = os.path.join(current_dir, "results/SDSS/spectra/sdss_spectro_0329-52056-0141/spec-0329-52056-0141.fits")

def load_sdss_spectrum(fits_path):
    with fits.open(fits_path, memmap=True) as hdul:
        hdr0 = hdul[0].header
        # --- COADD table ---
        # index/name 異なる FITS が紛れても動くように名前優先で取得
        hdu = None
        for key in ('COADD', 1):
            try:
                hdu = hdul[key]
                if hasattr(hdu, 'data') and hdu.data is not None:
                    break
            except Exception:
                pass
        if hdu is None or hdu.data is None:
            raise RuntimeError("COADD table not found.")

        coadd = hdu.data
        cols = [c.lower() for c in coadd.names]
        # 必須列チェック
        need = ['loglam','flux','ivar']
        for c in need:
            if c not in cols:
                raise RuntimeError(f"Required column '{c}' not found. Found: {coadd.names}")

        loglam = np.array(coadd['loglam'], dtype=float)
        wave   = np.power(10.0, loglam)  # [Å]
        flux   = np.array(coadd['flux'],  dtype=float)
        ivar   = np.array(coadd['ivar'],  dtype=float)

        # optional
        and_mask = np.array(coadd['and_mask']) if 'and_mask' in cols else None
        or_mask  = np.array(coadd['or_mask'])  if 'or_mask'  in cols else None
        sky      = np.array(coadd['sky'],   dtype=float) if 'sky'   in cols else None
        model    = np.array(coadd['model'], dtype=float) if 'model' in cols else None

        # --- redshift from SPECOBJ (if exists) ---
        z = np.nan
        if 'SPECOBJ' in hdul:
            specobj = hdul['SPECOBJ'].data
            if specobj is not None:
                for key in ('Z', 'z'):
                    if key in specobj.names:
                        try:
                            z = float(specobj[key][0])
                            break
                        except Exception:
                            pass

    return dict(wave=wave, flux=flux, ivar=ivar,
                and_mask=and_mask, or_mask=or_mask,
                sky=sky, model=model, z=z, header0=hdr0)


def summarize_mask(name, mask):
    n = mask.size
    bad = np.count_nonzero(mask)
    print(f"[{name}] masked {bad}/{n} ({bad/n*100:.1f} %)")

def build_mask(data, mask_mode='ivar'):
    """Return boolean mask (True = BAD = drop)."""
    wave, flux, ivar = data['wave'], data['flux'], data['ivar']
    and_mask, or_mask = data['and_mask'], data['or_mask']

    bad = ~np.isfinite(wave) | ~np.isfinite(flux) | ~np.isfinite(ivar)
    if mask_mode in (None, 'none'):
        return bad

    # まず ivar<=0 は不使用に（最小限のマスク）
    if mask_mode in ('ivar', 'ivar_and', 'ivar_or'):
        bad |= (ivar <= 0)

    # 追加で and_mask または or_mask を使う（必要に応じて）
    if mask_mode == 'ivar_and' and and_mask is not None:
        # and_mask != 0 を全部捨てると“全欠損”になりやすいので注意
        bad |= (and_mask != 0)
    if mask_mode == 'ivar_or' and or_mask is not None:
        bad |= (or_mask  != 0)

    return bad


def plot_sdss_spectrum(fits_path, mask_mode='ivar', show_rest=True, ylim=None, save_png=None):
    data = load_sdss_spectrum(fits_path)
    wave, flux, ivar = data['wave'], data['flux'], data['ivar']
    sky, model, z = data['sky'], data['model'], data['z']

    # --- マスク作成 & サマリ表示 ---
    bad = build_mask(data, mask_mode=mask_mode)
    summarize_mask(f"mask_mode={mask_mode}", bad)

    # マスク適用（プロット用に NaN を入れる）
    f = np.where(bad, np.nan, flux)
    m = np.where(bad, np.nan, model) if model is not None else None
    s = np.where(bad, np.nan, sky)   if sky   is not None else None

    # 有効点が 0 なら、まず“無加工表示”も試す
    if np.all(~np.isfinite(f)):
        print("All points masked/NaN under current mask. Retrying with mask_mode='none'…")
        bad = build_mask(data, mask_mode='none')
        f = np.where(bad, np.nan, flux)
        m = model  # 無加工でそのまま表示
        s = sky

    plt.figure(figsize=(12, 6))
    plt.plot(wave, f, color='k', lw=0.8, label='Flux (observed)')
    # if m is not None:
    #     plt.plot(wave, m, color='tab:orange', lw=0.6, alpha=0.8, label='Pipeline model')
    # if s is not None:
    #     plt.plot(wave, s, color='tab:blue', lw=0.5, alpha=0.5, label='Sky model')

    # # 静止系を重ねる（任意）
    # if show_rest and np.isfinite(z) and z > 0:
    #     plt.plot(wave/(1.0+z), f, color='crimson', lw=0.7, alpha=0.6,
    #              label=f'Flux (rest frame, z={z:.4f})')
    
    plt.xlim(7400, 8000)
    plt.xlabel(r'$\lambda (Å)$')
    plt.ylabel(r'F$_{\lambda}$ ($10^{-17}$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$)')
    plt.title(os.path.basename(fits_path) + (f"  (z={z:.4f})" if np.isfinite(z) else ""))

    # y 軸範囲がすべて NaN の時に落ちないように
    if ylim is not None:
        plt.ylim(*ylim)
    else:
        # 有効な y を探して自動決定
        y = f[np.isfinite(f)]
        if y.size >= 10:
            ymed = np.nanmedian(y)
            ymad = np.nanmedian(np.abs(y - ymed))
            ymin, ymax = ymed - 10*ymad, ymed + 30*ymad
            if np.isfinite(ymin) and np.isfinite(ymax) and ymin < ymax:
                plt.ylim(ymin, ymax)

    # plt.legend(loc='best', fontsize=9)
    plt.tight_layout()
    if save_png:
        plt.savefig(save_png, dpi=150)
        print(f"Saved figure -> {save_png}")
    else:
        plt.show()


if __name__ == "__main__":
    # 例
    plot_sdss_spectrum(fits_dir,
                       mask_mode='ivar',   # まずは最小限のマスクだけに
                       show_rest=True,
                       ylim=None)