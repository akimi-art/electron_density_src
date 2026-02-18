#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SDSSのFITS（テーブル形式）に Curti+17 で用いられる強輝線指標
R2, R3, O32, R23, N2, O3N2 を（1σ誤差つきで）新規列として追記するスクリプト。

⚠️ デフォルトでは Curti+17（2017, MNRAS 465, 1384）で明示されている定義に合わせ、
    R3=[OIII]5007/Hβ, R2=[OII]3727/Hβ, O32=[OIII]5007/[OII]3727,
    R23=([OIII]5007+[OII]3727)/Hβ, N2=[NII]6584/Hα, O3N2=log10(( [OIII]5007/Hβ ) / ( [NII]6584/Hα ))
    を用います。
    文献や実務では R23 に [OIII]4959 を含める流儀（=5007に×1.33を掛ける近似）があるため、
    オプション --use-oiii-total で R23 計算時のみ [OIII]合計(5007+4959=1.33×5007) を採用できます。

使用方法:
    python add_curti17_indices_to_fits.py input.fits output_with_indices.fits \
           [--overwrite] [--use-oiii-total]

前提:
    - FITSの拡張 #1 がバイナリテーブルで、少なくとも以下の列名が存在すること:
        'H_BETA_FLUX', 'H_BETA_FLUX_ERR',
        'H_ALPHA_FLUX', 'H_ALPHA_FLUX_ERR',
        'OIII_5007_FLUX', 'OIII_5007_FLUX_ERR',
        'NII_6584_FLUX', 'NII_6584_FLUX_ERR'
    - [OII] は次のいずれかの形で存在していると最適:
        (A) 'OII_3727_FLUX' と 'OII_3727_FLUX_ERR'（=3726+3729のブレンド）
        (B) 'OII_3726_FLUX' & 'OII_3729_FLUX'（誤差はそれぞれ *_ERR）
      ※[OII]が無い場合、R2/O32/R23 は NaN になります（N2,O3N2,R3 は計算可能）。

出力:
    下記12列を拡張 #1 のテーブルに追記して新しいFITSを作成します。
        'R2','R2_ERR','R3','R3_ERR','O32','O32_ERR',
        'R23','R23_ERR','N2','N2_ERR','O3N2','O3N2_ERR'

注意:
    - ここでの比は減光補正前の観測フラックス比です。減光補正が必要なら、
      本スクリプトの前処理として Balmer 減光補正などを適用した列を別途用意してください。

著者: A. M.
作成日: 2026-02-18

参考文献:
    - PEP 8:                  https://peps.python.org/pep-0008/
    - PEP 257 (Docstring規約): https://peps.python.org/pep-0257/
    - Python公式ドキュメント:    https://docs.python.org/ja/3/
"""


# === 必要なパッケージのインストール === #
import argparse
import numpy as np
from astropy.io import fits


def _safe_ratio(num, num_err, den, den_err):
    """比 x = num/den と 1σ誤差を（共分散ゼロ仮定で）計算。
    いずれかが NaN/非正/ゼロのときは (np.nan, np.nan) を返す。
    """
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    num_err = np.asarray(num_err, dtype=float)
    den_err = np.asarray(den_err, dtype=float)

    x = np.full_like(num, np.nan, dtype=float)
    xerr = np.full_like(num, np.nan, dtype=float)

    with np.errstate(invalid='ignore', divide='ignore'): 
        good = (num > 0) & (den > 0) & (num_err >= 0) & (den_err >= 0)
        x[good] = num[good] / den[good]
        # σ_x = x * sqrt[(σn/n)^2 + (σd/d)^2]
        frac = np.sqrt((num_err[good]/num[good])**2 + (den_err[good]/den[good])**2)
        xerr[good] = x[good] * frac
    return x, xerr


def _quadrature_sum(a, aerr, b, berr):
    """和 s=a+b と誤差。 共分散ゼロ。
    非正/NaNに厳密な制約は課さない（値そのものの和）。"""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    aerr = np.asarray(aerr, dtype=float)
    berr = np.asarray(berr, dtype=float)
    s = a + b
    serr = np.sqrt(aerr**2 + berr**2)
    return s, serr


def _safe_log10(x, xerr):
    """y=log10(x) と σ_y。x<=0 は NaN。
    σ_y = (1/ln10) * σ_x/x を使用。"""
    x = np.asarray(x, dtype=float)
    xerr = np.asarray(xerr, dtype=float)
    y = np.full_like(x, np.nan, dtype=float)
    yerr = np.full_like(x, np.nan, dtype=float)
    good = (x > 0) & (xerr >= 0)
    ln10 = np.log(10.0)
    y[good] = np.log10(x[good])
    yerr[good] = (1.0/ln10) * (xerr[good] / x[good])
    return y, yerr


def add_indices(in_fits, out_fits, overwrite=False, use_oiii_total=False):
    with fits.open(in_fits, mode='readonly') as hdul:
        # 拡張 #1 を前提（SDSSの一般的テーブル）。必要に応じて変更可。
        hdu = hdul[1]
        data = hdu.data
        cols = hdu.columns

        # 必須列の取り出し（存在チェックも）
        def need(col):
            if col not in data.names:
                raise KeyError(f"必要な列が見つかりません: {col}")
            return data[col]

        HB = need('H_BETA_FLUX')
        HB_E = need('H_BETA_FLUX_ERR')
        HA = need('H_ALPHA_FLUX')
        HA_E = need('H_ALPHA_FLUX_ERR')
        O3 = need('OIII_5007_FLUX')
        O3_E = need('OIII_5007_FLUX_ERR')
        N2l = need('NII_6584_FLUX')
        N2l_E = need('NII_6584_FLUX_ERR')


        # [O II] は既存の分岐で O2, O2_E を作っている（この直後に補正を入れる）
        # [O II] は複数の可能性に対応
        if 'OII_3727_FLUX' in data.names and 'OII_3727_FLUX_ERR' in data.names:
            O2 = data['OII_3727_FLUX']
            O2_E = data['OII_3727_FLUX_ERR']
        elif all(n in data.names for n in ['OII_3726_FLUX','OII_3729_FLUX','OII_3726_FLUX_ERR','OII_3729_FLUX_ERR']):
            O2, O2_E = _quadrature_sum(data['OII_3726_FLUX'], data['OII_3726_FLUX_ERR'],
                                       data['OII_3729_FLUX'], data['OII_3729_FLUX_ERR'])
        else:
            O2 = np.full_like(HB, np.nan, dtype=float)
            O2_E = np.full_like(HB, np.nan, dtype=float)
        # ... O2, O2_E がここまでに定義済み ...

        # =========================
        # 減光補正（ここを追加）
        # =========================
        # 例: Cardelli+89 (R_V=3.1) の簡易係数 [k(λ) = A_λ / E(B-V)] を定数で与える
        # 厳密にやるならパッケージで波長から計算してください
        _k = {
            'H_ALPHA': 2.53,    # 6563 Å
            'H_BETA' : 3.61,    # 4861 Å
            'OIII5007': 3.47,   # 5007 Å
            'OII3727' : 4.74,   # 3727 Å （3726/3729 もほぼ同等として簡略）
            'NII6584' : 2.52,   # 6584 Å
        }

        # 観測 Balmer 減光から E(B-V) を推定（Case B 2.86 を仮定）
        #   E(B-V) = [2.5 / (k_Hβ - k_Hα)] * log10( (Hα/Hβ)_obs / 2.86 )
        with np.errstate(invalid='ignore', divide='ignore'):
            balmer_obs = HA / HB
            balmer_int = 2.86
            EBV = (2.5 / (_k['H_BETA'] - _k['H_ALPHA'])) * np.log10(balmer_obs / balmer_int)

        def _deredden(F, key):
            A = _k[key] * EBV              # A_λ = k(λ) * E(B-V)
            return F * np.power(10.0, 0.4 * A)

        # 線ごとの減光補正フラックス（比の計算はこの補正後 F を使う）
        HB_c  = _deredden(HB,  'H_BETA')
        HA_c  = _deredden(HA,  'H_ALPHA')
        O3_c  = _deredden(O3,  'OIII5007')
        N2l_c = _deredden(N2l, 'NII6584')
        O2_c  = _deredden(O2,  'OII3727')   # 3726/3729 の合算列（3727）に対して補正


        # ---- 指標の計算 ----
        # R3 = [OIII]5007 / Hβ
        R3, R3_E = _safe_ratio(O3, O3_E, HB, HB_E)

        # R2 = [OII]3727 / Hβ （OIIが無ければ NaN）
        R2, R2_E = _safe_ratio(O2, O2_E, HB, HB_E)

        # O32 = [OIII]5007 / [OII]3727
        O32, O32_E = _safe_ratio(O3, O3_E, O2, O2_E)

        # R23 = ([OIII] + [OII]) / Hβ
        if use_oiii_total:
            # 4959を含める近似として 1.33×5007
            O3_total = 1.33 * O3
            O3_total_E = 1.33 * O3_E
            O3plusO2, O3plusO2_E = _quadrature_sum(O3_total, O3_total_E, O2, O2_E)
        else:
            O3plusO2, O3plusO2_E = _quadrature_sum(O3, O3_E, O2, O2_E)
        R23, R23_E = _safe_ratio(O3plusO2, O3plusO2_E, HB, HB_E)

        # N2 = [NII]6584 / Hα （線形比）
        N2, N2_E = _safe_ratio(N2l, N2l_E, HA, HA_E)

        # O3N2 = log10( ( [OIII]5007/Hβ ) / ( [NII]6584/Hα ) )
        # 既存に log 列があれば再利用も可能だが、ここでは一貫してフラックスから計算
        # log_O3HB とその誤差
        O3HB, O3HB_E = _safe_ratio(O3, O3_E, HB, HB_E)
        log_O3HB, log_O3HB_E = _safe_log10(O3HB, O3HB_E)
        # log_N2HA とその誤差
        N2HA, N2HA_E = _safe_ratio(N2l, N2l_E, HA, HA_E)
        log_N2HA, log_N2HA_E = _safe_log10(N2HA, N2HA_E)
        O3N2 = log_O3HB - log_N2HA
        O3N2_E = np.sqrt(log_O3HB_E**2 + log_N2HA_E**2)

        # 追加カラムを作成
        new_cols = []
        def col(name, arr):
            # 単精度(E)だと十分だが、元のテーブルがD(倍精度)なら揃えたい場合はここを調整
            return fits.Column(name=name, array=arr, format='D')

        # new_cols.extend([
        #     col('R2', R2), col('R2_ERR', R2_E),
        #     col('R3', R3), col('R3_ERR', R3_E),
        #     col('O32', O32), col('O32_ERR', O32_E),
        #     col('R23', R23), col('R23_ERR', R23_E),
        #     col('N2', N2), col('N2_ERR', N2_E),
        #     col('O3N2', O3N2), col('O3N2_ERR', O3N2_E),
        # ])

        # ---- ここは既存の計算結果（R2, R3, O32, R23, N2, O3N2）が出た後に追記 ----
        # 線形 → log10 の変換（O3N2 はすでに log なので除外）
        log_R2,   log_R2_E   = _safe_log10(R2,   R2_E)
        log_R3,   log_R3_E   = _safe_log10(R3,   R3_E)
        log_O32,  log_O32_E  = _safe_log10(O32,  O32_E)
        log_R23,  log_R23_E  = _safe_log10(R23,  R23_E)
        log_N2,   log_N2_E   = _safe_log10(N2,   N2_E)

        # 追加カラムに log 版も加える（O3N2 は既に log なのでそのまま）
        new_cols.extend([
            col('R2',  log_R2),   col('R2_ERR',  log_R2_E),
            col('R3',  log_R3),   col('R3_ERR',  log_R3_E),
            col('O32', log_O32),  col('O32_ERR', log_O32_E),
            col('R23', log_R23),  col('R23_ERR', log_R23_E),
            col('N2',  log_N2),   col('N2_ERR',  log_N2_E),
            col('O3N2', O3N2),    col('O3N2_ERR', O3N2_E),
        ])

        new_hdu = fits.BinTableHDU.from_columns(cols + fits.ColDefs(new_cols))
        # 主ヘッダなどはそのまま引き継ぐ
        prim = hdul[0].copy()
        hdul_out = fits.HDUList([prim, new_hdu])
        hdul_out.writeto(out_fits, overwrite=overwrite)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Add Curti+17 strong-line indices to an SDSS-like FITS table.')
    p.add_argument('input', help='入力FITS（バイナリテーブル）')
    p.add_argument('output', help='出力FITSファイル名')
    p.add_argument('--overwrite', action='store_true', help='既存の出力FITSを上書き')
    p.add_argument('--use-oiii-total', action='store_true', help='R23計算時に [OIII]4959 を含める（=1.33×5007）')
    args = p.parse_args()

    add_indices(args.input, args.output, overwrite=args.overwrite, use_oiii_total=args.use_oiii_total)


# 使用例（ターミナルで "そのまま実行" するだけ）:
    # 引数を一切与えずに実行すると、カレントディレクトリを起点に
    #  下記の埋め込みパスで add_indices(...) が走ります。
    #  - input : ./results/fits/mpajhu_dr7_v5_2_merged.fits
    #  - output: ./results/fits/mpajhu_dr7_v5_2_merged_with_indices_Curti+17.fits
    #  - overwrite=True, use_oiii_total=False
    # python3 add_curti17_indices_to_fits.py
# 方法2：このまま書く
# python3 electron_density_src/add_curti17_indices_to_fits.py ./results/fits/mpajhu_dr7_v5_2_merged.fits ./results/fits/mpajhu_dr7_v5_2_merged_with_indices_Curti+17.fits --overwrite