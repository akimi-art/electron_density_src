
# -*- coding: utf-8 -*-
"""
SFR計算ユーティリティ
- 平均SFR: SFR_avg ≈ M*/t_age
- τモデルの瞬間SFR（現在時刻 t=age）:
   * 減少型 (τ>0):  M* = (1-R) * SFR0 * τ * (1 - exp(-t/τ)), SFR(t) = SFR0 * exp(-t/τ)
   * 増加型 (τ<0):  近似的に |τ| を用いて、線形 or 指数増加のモデルで現在SFRを推定

単位:
- M*: 10^9 M_sun で与えられていると仮定 → 必要に応じて1e9倍
- age: Myr → 年へ変換（1 Myr = 1e6 yr）
- τ: Gyr → 年へ変換（1 Gyr = 1e9 yr）
"""

import numpy as np
from typing import Tuple, Optional, Dict

def sfr_average(mstar_1e9Msun: float, age_Myr: float) -> float:
    """
    期間平均SFR ≈ M*/t_age を返す
    """
    mstar = mstar_1e9Msun * 1e9  # M_sun
    tage = age_Myr * 1e6         # yr
    return mstar / tage          # M_sun/yr

def sfr_tau_decreasing(mstar_1e9Msun: float, age_Myr: float, tau_Gyr: float, R: float = 0.3) -> float:
    """
    減少型SFH (τ>0) の現在SFR (t=age) を厳密式から計算
    式の導出:
      M* = (1-R) * SFR0 * τ * (1 - e^{-t/τ})
      SFR(t) = SFR0 * e^{-t/τ}
      → SFR(t) = M* / [(1-R) * τ * (e^{t/τ} - 1)]
    """
    assert tau_Gyr > 0, "tau must be >0 for decreasing model"
    mstar = mstar_1e9Msun * 1e9
    t = age_Myr * 1e6
    tau = tau_Gyr * 1e9
    return mstar / ((1 - R) * tau * (np.exp(t / tau) - 1.0))

def sfr_tau_increasing_linear(mstar_1e9Msun: float, age_Myr: float, tau_abs_Gyr: float, R: float = 0.3) -> float:
    """
    増加型SFHの簡易近似（線形増加モデル）
    仮定: SFR(t) = SFR_max * (t / t_age), 0<=t<=t_age
      → M* = (1-R) * ∫ SFR(t) dt = (1-R) * SFR_max * t_age / 2
      → 現在SFR = SFR_max = 2 M* / [(1-R) * t_age]
    ※ tau_abs_Gyr は使わない（将来の拡張用、API整合のため）
    """
    mstar = mstar_1e9Msun * 1e9
    t_age = age_Myr * 1e6
    return 2.0 * mstar / ((1.0 - R) * t_age)

def sfr_tau_increasing_exp(mstar_1e9Msun: float, age_Myr: float, tau_abs_Gyr: float, R: float = 0.3) -> float:
    """
    増加型SFHの簡易近似（指数増加モデル）
    仮定: SFR(t) = SFR0 * exp(+t/τ), 0<=t<=t_age
      → M* = (1-R) * SFR0 * τ * (exp(t_age/τ) - 1)
      → 現在SFR = SFR0 * exp(t_age/τ) = M* / [(1-R)*τ] * [1 / (1 - exp(-t_age/τ))]
    """
    mstar = mstar_1e9Msun * 1e9
    t_age = age_Myr * 1e6
    tau = tau_abs_Gyr * 1e9
    return mstar / ((1.0 - R) * tau) * (1.0 / (1.0 - np.exp(-t_age / tau)))

def mc_sfr(
    mstar_center_1e9: float,
    mstar_min_1e9: float,
    mstar_max_1e9: float,
    age_center_Myr: float,
    age_min_Myr: float,
    age_max_Myr: float,
    tau_Gyr: Optional[float] = None,
    model: str = "avg",
    R: float = 0.3,
    n_draw: int = 20000,
    random_state: Optional[int] = 42
) -> Dict[str, float]:
    """
    M*とageの区間からモンテカルロでSFR分布を作り、中央値と68%区間を返す。
    model:
      - "avg" : 平均SFR (M*/t_age)
      - "tau_dec" : τ>0 減少型
      - "tau_inc_lin" : τ<0 線形増加
      - "tau_inc_exp" : τ<0 指数増加
    """
    rng = np.random.default_rng(random_state)
    # 区間からの一様サンプル（必要なら三角分布などへ拡張可能）
    m_samples = rng.uniform(mstar_min_1e9, mstar_max_1e9, n_draw)
    # 年齢は極端な幅があるため、対数一様の方が安定することが多い
    age_log_min, age_log_max = np.log10(age_min_Myr), np.log10(age_max_Myr)
    age_samples = 10 ** rng.uniform(age_log_min, age_log_max, n_draw)

    sfr_vals = np.zeros(n_draw)
    if model == "avg":
        sfr_vals = np.array([sfr_average(m, a) for m, a in zip(m_samples, age_samples)])
    elif model == "tau_dec":
        assert tau_Gyr is not None and tau_Gyr > 0
        sfr_vals = np.array([sfr_tau_decreasing(m, a, tau_Gyr, R=R) for m, a in zip(m_samples, age_samples)])
    elif model == "tau_inc_lin":
        assert tau_Gyr is not None
        sfr_vals = np.array([sfr_tau_increasing_linear(m, a, abs(tau_Gyr), R=R) for m, a in zip(m_samples, age_samples)])
    elif model == "tau_inc_exp":
        assert tau_Gyr is not None
        sfr_vals = np.array([sfr_tau_increasing_exp(m, a, abs(tau_Gyr), R=R) for m, a in zip(m_samples, age_samples)])
    else:
        raise ValueError("unknown model")

    # 分布要約統計
    med = np.nanmedian(sfr_vals)
    p16 = np.nanpercentile(sfr_vals, 16)
    p84 = np.nanpercentile(sfr_vals, 84)
    return {
        "median": med,
        "p16": p16,
        "p84": p84,
        "err_minus": med - p16,
        "err_plus": p84 - med,
    }

# -------------------------
# 使用例：あなたが挙げた値 (ID2, ID3→RXCJ2248-ID)
# -------------------------
if __name__ == "__main__":
    # ID2: τ = -1.5 Gyr（増加型）, Z/Z⊙=0.2, age=0.5 [0.1,270] Myr, M* = 0.093 [0.033,0.26] ×10^9 M⊙
    id2_avg = mc_sfr(
        mstar_center_1e9=0.093, mstar_min_1e9=0.033, mstar_max_1e9=0.26,
        age_center_Myr=0.5, age_min_Myr=0.1, age_max_Myr=270.0,
        model="avg"
    )
    print("[ID2] 平均SFR:", id2_avg)

    # 増加型の近似（線形）
    id2_inc_lin = mc_sfr(
        mstar_center_1e9=0.093, mstar_min_1e9=0.033, mstar_max_1e9=0.26,
        age_center_Myr=0.5, age_min_Myr=0.1, age_max_Myr=270.0,
        tau_Gyr=-1.5, model="tau_inc_lin"
    )
    print("[ID2] τ<0 線形増加（現在SFR）:", id2_inc_lin)

    # ID3: SSP, Z/Z⊙=0.005, age=1.5 [0.1,330] Myr, M* = 0.21 [0.058,0.75] ×10^9 M⊙
    # SSPは「瞬間形成」の仮定のため、通常は現在SFR≈0と解釈するが、
    # 期間平均SFR（=M*/t_age）は比較指標として有用。
    id3_avg = mc_sfr(
        mstar_center_1e9=0.21, mstar_min_1e9=0.058, mstar_max_1e9=0.75,
        age_center_Myr=1.5, age_min_Myr=0.1, age_max_Myr=330.0,
        model="avg"
    )
    print("[ID3] 平均SFR:", id3_avg)

# [ID2] 平均SFR: {'median': np.float64(24.007746825829337), 'p16': np.float64(1.6824755709945873), 'p84': np.float64(364.3340437626386), 'err_minus': np.float64(22.32527125483475), 'err_plus': np.float64(340.3262969368093)}
# [ID2] τ<0 線形増加（現在SFR）: {'median': np.float64(68.59356235951239), 'p16': np.float64(4.807073059984535), 'p84': np.float64(1040.9544107503962), 'err_minus': np.float64(63.78648929952786), 'err_plus': np.float64(972.3608483908838)}
# [ID3] 平均SFR: {'median': np.float64(57.44712017686136), 'p16': np.float64(3.759099215108841), 'p84': np.float64(939.1663435538071), 'err_minus': np.float64(53.68802096175252), 'err_plus': np.float64(881.7192233769457)}