
# -*- coding: utf-8 -*-
"""
スクリプトの概要:
log10(Zneb/Z⊙) → 12+log(O/H) へ変換するユーティリティです。
既定の太陽基準値はAsplund et al. (2009) の8.69を使用しています。
必要に応じて solar_oh = 8.73（例：Steffen+）などに変更可能です。

変換式：
  12 + log10(O/H) = (12 + log10(O/H)_⊙) + log10(Zneb/Z⊙)

使用方法:
    metallicity_conversion.py [オプション]

著者: A. M.
作成日: 2026-01-13

参考文献:
    - PEP 8:                  https://peps.python.org/pep-0008/
    - PEP 257 (Docstring規約): https://peps.python.org/pep-0257/
    - Python公式ドキュメント:    https://docs.python.org/ja/3/
    - Asplund et al. (2009), "The Chemical Composition of the Sun"
"""

import numpy as np
from typing import Tuple, Optional, Sequence, Union

Number = Union[float, int]

def zneb_to_oh(
    log10_z_over_zsun: Number,
    err_plus: Optional[Number] = None,
    err_minus: Optional[Number] = None,
    solar_oh: Number = 8.69
) -> Tuple[float, Optional[float], Optional[float]]:
    """
    パラメータ
    ----------
    log10_z_over_zsun : float
        文献で与えられる log10(Zneb/Z⊙) の中心値（例：-0.92）
    err_plus : float or None
        上側の誤差（+方向）。非対称誤差に対応（例：+0.06）。Noneなら誤差計算を省略。
    err_minus : float or None
        下側の誤差（-方向）。非対称誤差に対応（例：-0.05）。Noneなら誤差計算を省略。
    solar_oh : float
        太陽の 12+log(O/H) の参照値（既定は 8.69）

    戻り値
    -------
    oh_center, oh_err_plus, oh_err_minus : tuple
        12+log(O/H) の中心値と非対称誤差（いずれも float）。
        err_plus/err_minus が None の場合は、誤差部分も None を返す。

    備考
    ----
    この変換は「金属量 Z を酸素 O/H のスケールに線形換算できる」という
    暗黙の近似に基づきます。実際には元素ごとの増減や校正差があり得るため、
    研究目的に応じて系統誤差の考慮が必要です。
    """
    # 中心値の変換：12 + log10(O/H) = solar_oh + log10(Zneb/Z⊙)
    oh_center = solar_oh + float(log10_z_over_zsun)

    # 非対称誤差の伝播：足し算なのでそのまま同じ量が 12+log(O/H) に加算される
    if err_plus is not None and err_minus is not None:
        oh_err_plus = float(err_plus)
        oh_err_minus = float(err_minus)
        return oh_center, oh_err_plus, oh_err_minus
    else:
        return oh_center, None, None


def batch_zneb_to_oh(
    values: Sequence[Number],
    err_pluses: Optional[Sequence[Number]] = None,
    err_minuses: Optional[Sequence[Number]] = None,
    solar_oh: Number = 8.69
):
    """
    複数サンプルを一括処理したい場合のヘルパー関数。
    """
    results = []
    for i, v in enumerate(values):
        ep = err_pluses[i] if err_pluses is not None else None
        em = err_minuses[i] if err_minuses is not None else None
        results.append(zneb_to_oh(v, ep, em, solar_oh))
    return results


if __name__ == "__main__":
    # --- 使用例 1: 単一値（文献の例） ---
    # log10(Zneb/Z⊙) = -0.92^{+0.06}_{-0.05}
    oh_center, oh_err_plus, oh_err_minus = zneb_to_oh(
        log10_z_over_zsun= np.log10(0.5),
        err_plus=0,
        err_minus=0,
        solar_oh=8.69  # 必要なら 8.73 などに変更
    )
    print(f"12+log(O/H) = {oh_center:.2f} +{oh_err_plus:.2f} -{oh_err_minus:.2f}")
    # 期待出力: 12+log(O/H) ≈ 7.77 +0.06 -0.05

    # --- 使用例 2: 一括処理 ---
    vals = [-0.92, -0.50, -1.10]
    eps = [0.06, 0.10, 0.08]
    ems = [0.05, 0.09, 0.07]
    batch_results = batch_zneb_to_oh(vals, eps, ems, solar_oh=8.69)
    for j, (c, up, dn) in enumerate(batch_results):
        print(f"[{j}] 12+log(O/H) = {c:.3f} +{up:.3f} -{dn:.3f}")
