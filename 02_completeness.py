import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy.table import Table

import selection as S

# 軸の設定
plt.rcParams.update({
    # --- 図全体 ---
    "figure.figsize": (12, 6),       # 図サイズ
    "font.size": 32,                 # 全体フォントサイズ
    "axes.labelsize": 32,            # 軸ラベルのサイズ
    "axes.titlesize": 32,            # タイトルのサイズ
    "axes.grid": False,              # グリッドOFF

    # --- 目盛り設定 (ticks) ---
    "xtick.direction": "in",         # x軸目盛りの向き
    "ytick.direction": "in",         # y軸目盛りの向き
    "xtick.top": True,               # 上にも目盛り
    "ytick.right": True,             # 右にも目盛り

    # 主目盛り（major ticks）
    "xtick.major.size": 32,          # 長さ
    "ytick.major.size": 32,
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
    "xtick.labelsize": 28,           # x軸ラベルサイズ
    "ytick.labelsize": 28,           # y軸ラベルサイズ

    # --- フォント ---
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
})


# ============================================================
# 0. 設定（結果を見る前に固定する）
# ============================================================
COMP_THRESH = 0.9
NMIN_CELL = 10
MS_WIDTH = S.MS_WIDTH               # 0.6 dex（局所の main sequence のばらつき約 0.3 dex の2倍）

# high-z の銀河の値（local の銀河がその近くに何個いるかを数える）
HIGHZ_VALUES = {"logSFR": 1.4, "logsSFR": -8.6, "logSigmaSFR": 0.1}
HIGHZ_TOL = 0.1

current_dir = os.getcwd()
fits_path = os.path.join(current_dir, "results/fits/mpajhu_dr7_v5_2_merged_radius.fits")
fig_dir = os.path.join(current_dir, "results/figure/completeness")
os.makedirs(fig_dir, exist_ok=True)

df = Table.read(fits_path, format="fits").to_pandas()
q = S.derived_quantities(df)
Z_MAX = S.z_max_from_flux_limit()
logM, logSFR = q["logM"], q["logSFR"]
print(f"[INFO] Z_MAX = {Z_MAX:.4f}")


# ============================================================
# 1. main sequence：Renzini & Peng (2015) の線 ± MS_WIDTH dex
# ============================================================
def ridge_at(x):
    return S.MS_SLOPE * np.asarray(x) + S.MS_ZP

dMS = logSFR - ridge_at(logM)
on_ms    = np.abs(dMS) < MS_WIDTH          # M* の完全性の分母
above_ms = dMS > -MS_WIDTH                 # SFR などの完全性の分母（starburst を含む）

# 補強：文献の線より上側のばらつき（約 0.3 dex になるか）
par_ms = S.selection_masks(df, q, Z_MAX, S.METHODS[0])["base"]
ms_bins = np.arange(8.6, 11.61, 0.2)
print("\n logM     N   sigma(upper)")
for lo, hi in zip(ms_bins[:-1], ms_bins[1:]):
    inb = par_ms & (logM >= lo) & (logM < hi) & np.isfinite(logSFR)
    up = dMS[inb][dMS[inb] > 0]
    s = 1.4826 * np.median(up) if up.size >= 50 else np.nan
    print(f" {0.5*(lo+hi):4.1f} {inb.sum():7d}   {s:.2f}")

# 確認の図：線、±0.3 dex（点線）、±0.6 dex（破線）
xb, yb = np.arange(7.0, 12.01, 0.1), np.arange(-3.0, 3.01, 0.1)
N, _, _ = np.histogram2d(logM[par_ms], logSFR[par_ms], bins=[xb, yb])
mm = np.linspace(7, 12, 200)
fig, ax = plt.subplots(figsize=(8, 6))
ax.pcolormesh(xb, yb, np.where(N > 0, N, np.nan).T, norm=LogNorm(),
              cmap="viridis", shading="auto")
ax.plot(mm, ridge_at(mm), color="w", lw=2, label="Renzini & Peng 2015")
for k, ls, lab in [(0.5, ":", r"$\pm0.3$ dex"), (1.0, "--", r"$\pm0.6$ dex")]:
    ax.plot(mm, ridge_at(mm) + k * MS_WIDTH, color="w", lw=1.2, ls=ls, label=lab)
    ax.plot(mm, ridge_at(mm) - k * MS_WIDTH, color="w", lw=1.2, ls=ls)
ax.set_xlabel(r"$\log(M_*/M_\odot)$")
ax.set_ylabel(r"$\log({\rm SFR}/M_\odot\,{\rm yr^{-1}})$")
ax.set_xlim(7, 12); ax.set_ylim(-3, 3)
ax.legend(fontsize=12, loc="upper left")
plt.savefig(os.path.join(fig_dir, "ms_renzini_peng.png"), bbox_inches="tight")
plt.show()


# ============================================================
# 2. 共通の関数
# ============================================================
def frac_in_bins(x, num_mask, den_mask, bins):
    n_den, _ = np.histogram(x[den_mask], bins=bins)
    n_num, _ = np.histogram(x[num_mask], bins=bins)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = (n_num / n_den).astype(float)
    f[n_den < NMIN_CELL] = np.nan
    return f, n_den


def lowest_complete_edge(bins, f, thresh):
    """そのビンより上のすべての有効なビンが基準を満たす、最も低いビンの下端（M_MIN 用）"""
    for i in range(len(f)):
        if not np.isfinite(f[i]):
            continue
        rest = f[i:][np.isfinite(f[i:])]
        if np.all(rest >= thresh):
            return bins[i]
    return np.nan


def upper_complete_edge(bins, f, thresh, lower):
    """lower から上に、基準を連続して満たすビンが続く範囲の上端（M_MAX 用）"""
    centers = 0.5 * (bins[1:] + bins[:-1])
    upper = np.nan
    for i, c in enumerate(centers):
        if c < lower:
            continue
        if np.isfinite(f[i]) and f[i] >= thresh:
            upper = bins[i + 1]
        else:
            break
    return upper


def complete_interval(bins, f, thresh):
    """基準を満たすビンが連続する、最も長い範囲 (下端, 上端)"""
    ok = np.isfinite(f) & (f >= thresh)
    best, cur_start, best_len = (np.nan, np.nan), None, 0
    for i, v in enumerate(np.append(ok, False)):
        if v and cur_start is None:
            cur_start = i
        elif not v and cur_start is not None:
            if i - cur_start > best_len:
                best_len, best = i - cur_start, (bins[cur_start], bins[i])
            cur_start = None
    return best


def step_curves(x, den, m, bins):
    """累積の3本と、光度のカット｜星形成 の1本"""
    sn, sf, lum = m["sn"], m["sf"], m["lum"]
    f_sn,  n = frac_in_bins(x, den & sn,            den, bins)
    f_sf,  _ = frac_in_bins(x, den & sn & sf,       den, bins)
    f_all, _ = frac_in_bins(x, den & sn & sf & lum, den, bins)
    f_L,   _ = frac_in_bins(x, den & sn & sf & lum, den & sn & sf, bins)
    return f_sn, f_sf, f_all, f_L, n


def plot_steps(xc, curves, method, xlabel, ylabel, save, span=None, extra=None):
    """span=(下端, 上端) を塗る。extra=(x, y, label) を灰色で重ねる"""
    f_sn, f_sf, f_all, f_L, _ = curves
    fig, ax = plt.subplots(figsize=(8, 5.5))
    if span is not None and np.all(np.isfinite(span)):
        ax.axvspan(span[0], span[1], color="c", alpha=0.12, lw=0)
    if extra is not None:
        ax.plot(extra[0], extra[1], "-", color="0.6", lw=2, label=extra[2])
    ax.plot(xc, f_sn,  "o-", label=r"S/N $\geq$ 3")
    ax.plot(xc, f_sf,  "o-", label=f"+ SF ({method})")
    ax.plot(xc, f_all, "o-", label=r"+ $L$([S II]) > $10^{39}$")
    ax.plot(xc, f_L,   "s--", color="k", label=r"$L$ cut | SF")
    ax.axhline(COMP_THRESH, ls=":", color="gray")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=12, loc="lower left")
    plt.savefig(save, bbox_inches="tight"); plt.show()


def mass_range(m, mbins, width):
    """幅 ±width dex の main sequence で M_MIN と M_MAX を求める（感度の確認用）"""
    den = m["base"] & (np.abs(dMS) < width)
    f_sn, f_sf, f_all, f_L, _ = step_curves(logM, den, m, mbins)
    lo = lowest_complete_edge(mbins, f_L, COMP_THRESH)
    return lo, upper_complete_edge(mbins, f_all, COMP_THRESH, lo)


Y_QUANT = {
    "SFR": dict(key="logSFR", bins2d=np.arange(-3.0, 3.01, 0.1),
                bins1d=np.arange(-1.5, 2.51, 0.2),
                label=r"$\log({\rm SFR}/M_\odot\,{\rm yr^{-1}})$"),
    "sSFR": dict(key="logsSFR", bins2d=np.arange(-13.0, -7.99, 0.1),
                 bins1d=np.arange(-12.0, -7.99, 0.2),
                 label=r"$\log({\rm sSFR}/{\rm yr^{-1}})$"),
    "SigmaSFR": dict(key="logSigmaSFR", bins2d=np.arange(-4.0, 1.51, 0.1),
                     bins1d=np.arange(-3.5, 1.01, 0.2),
                     label=r"$\log(\Sigma_{\rm SFR}/M_\odot\,{\rm yr^{-1}\,kpc^{-2}})$"),
}

mbins = np.arange(8.0, 11.51, 0.2)
mcen = 0.5 * (mbins[1:] + mbins[:-1])
summary = []
curves_by_method = {}
ranges_by_method = {}


# ============================================================
# 3. 方法ごとの完全性
# ============================================================
for method in S.METHODS:
    m = S.selection_masks(df, q, Z_MAX, method)
    base = m["base"]

    # --------------------------------------------------------
    # 3-1. M* の関数としての完全性 → M_MIN と M_MAX
    #      分母：main sequence（線 ± MS_WIDTH）
    # --------------------------------------------------------
    den = base & on_ms
    curves = step_curves(logM, den, m, mbins)
    f_sn, f_sf, f_all, f_L, n_den = curves
    curves_by_method[method] = curves

    M_MIN = lowest_complete_edge(mbins, f_L, COMP_THRESH)
    M_MAX = upper_complete_edge(mbins, f_all, COMP_THRESH, M_MIN)
    M_MAX_sn = upper_complete_edge(mbins, f_sn, COMP_THRESH, M_MIN)
    ranges_by_method[method] = (M_MIN, M_MAX)
    print(f"\n===== {method}: {M_MIN:.1f} <= log M* < {M_MAX:.1f}"
          f"（S/N だけで決めると上限 {M_MAX_sn:.1f}）=====")

    sfm = den & m["sn"] & m["sf"]
    print("  logM   N(MS)   S/N   +SF   +L    L|SF   ΔmedSFR")
    for lo, hi, a, b, c, d, n in zip(mbins[:-1], mbins[1:], f_sn, f_sf, f_all, f_L, n_den):
        inb = sfm & (logM >= lo) & (logM < hi)
        if inb.sum() >= NMIN_CELL and (inb & m["lum"]).sum() >= NMIN_CELL:
            dmed = np.median(logSFR[inb & m["lum"]]) - np.median(logSFR[inb])
        else:
            dmed = np.nan
        c_ = 0.5 * (lo + hi)
        flag = "*" if (c_ >= M_MIN and c_ < M_MAX) else " "
        print(f" {flag}{c_:4.1f} {n:7d}  {a:.2f}  {b:.2f}  {c:.2f}  {d:.2f}  {dmed:+.3f}")

    plot_steps(mcen, curves, method, r"$\log(M_*/M_\odot)$",
               "Fraction of MS galaxies retained",
               os.path.join(fig_dir, f"completeness_vs_mass_steps_{method}.png"),
               span=(M_MIN, M_MAX))

    in_mass = (logM >= M_MIN) & (logM < M_MAX)
    final = m["selected"] & in_mass

    # --------------------------------------------------------
    # 3-2. 確認：最終サンプルのうち、main sequence の下端より下にいる割合
    # --------------------------------------------------------
    frac_below = np.mean(dMS[final] < -MS_WIDTH)
    print(f"  最終サンプルのうち MS の下端（-{MS_WIDTH:g} dex）より下: {frac_below:.3f}")

    row = {"method": method, "M_MIN": M_MIN, "M_MAX": M_MAX, "M_MAX_SN": M_MAX_sn,
           "N_selected": int(m["selected"].sum()), "N_final": int(final.sum()),
           "frac_below_MS": frac_below}

    # --------------------------------------------------------
    # 3-3. 縦軸の量ごとに：地図 と その量の関数としての完全性
    #      分母：main sequence の下端より上（starburst を含む）
    # --------------------------------------------------------
    for name, Y in Y_QUANT.items():
        yv = q[Y["key"]]
        par = base & np.isfinite(yv)
        sel = par & m["lum"] & m["sn"] & m["sf"]

        # 地図
        xb, yb = np.arange(7.0, 12.01, 0.1), Y["bins2d"]
        N_par, _, _ = np.histogram2d(logM[par], yv[par], bins=[xb, yb])
        N_sel, _, _ = np.histogram2d(logM[sel], yv[sel], bins=[xb, yb])
        with np.errstate(divide="ignore", invalid="ignore"):
            comp = N_sel / N_par
        comp[N_par < NMIN_CELL] = np.nan

        fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
        norm = LogNorm(vmin=1, vmax=max(N_par.max(), 1))
        for ax, N_, title in zip(axes[:2], [N_par, N_sel], ["Parent", "Selected"]):
            im = ax.pcolormesh(xb, yb, np.where(N_ > 0, N_, np.nan).T,
                               norm=norm, cmap="viridis", shading="auto")
            ax.set_title(title)
        fig.colorbar(im, ax=axes[:2], label="Number of galaxies")
        im3 = axes[2].pcolormesh(xb, yb, comp.T, vmin=0, vmax=1,
                                 cmap="magma", shading="auto")
        axes[2].set_title(f"Completeness ({method})")
        fig.colorbar(im3, ax=axes[2], label=r"$N_{\rm sel}/N_{\rm parent}$")

        mm_plot = np.linspace(xb[0], xb[-1], 100)
        r_line = ridge_at(mm_plot)
        ms_line = {"SFR": r_line, "sSFR": r_line - mm_plot}.get(name)
        for ax in axes:
            if ms_line is not None:
                ax.plot(mm_plot, ms_line, color="w", lw=1.5)
                ax.plot(mm_plot, ms_line - MS_WIDTH, color="w", lw=1, ls="--")
                ax.plot(mm_plot, ms_line + MS_WIDTH, color="w", lw=1, ls="--")
            for v in (M_MIN, M_MAX):
                if np.isfinite(v):
                    ax.axvline(v, color="c", lw=1.5, ls=":")
            ax.set_xlim(xb[0], xb[-1]); ax.set_ylim(yb[0], yb[-1])
            ax.set_xlabel(r"$\log(M_*/M_\odot)$")
        axes[0].set_ylabel(Y["label"])
        plt.savefig(os.path.join(fig_dir, f"completeness_map_sm_{name}_{method}.png"),
                    bbox_inches="tight")
        plt.show()

        # その量の関数としての完全性
        yb1 = Y["bins1d"]
        yc = 0.5 * (yb1[1:] + yb1[:-1])

        den_y = par & above_ms & in_mass                       # 質量の上限あり
        curves_y = step_curves(yv, den_y, m, yb1)

        den_y_nolim = par & above_ms & (logM >= M_MIN)          # 比較：上限なし
        f_all_nolim = step_curves(yv, den_y_nolim, m, yb1)[2]

        y_lo, y_hi = complete_interval(yb1, curves_y[2], COMP_THRESH)
        row[f"{name}_lo"], row[f"{name}_hi"] = y_lo, y_hi

        hz = HIGHZ_VALUES[Y["key"]]
        n_near = int(np.sum(final & (np.abs(yv - hz) < HIGHZ_TOL)))
        row[f"{name}_N_near_highz"] = n_near

        print(f"  {name:9s}: 完全な範囲 {y_lo:+.1f} -- {y_hi:+.1f}  "
              f"| high-z の値 {hz:+.1f} ± {HIGHZ_TOL} の local 銀河: {n_near:,}")

        plot_steps(yc, curves_y, method, Y["label"],
                   rf"Fraction retained (${M_MIN:.1f} \leq \log M_* < {M_MAX:.1f}$)",
                   os.path.join(fig_dir, f"completeness_vs_{name}_steps_{method}.png"),
                   span=(y_lo, y_hi),
                   extra=(yc, f_all_nolim, rf"all cuts, $\log M_* \geq {M_MIN:.1f}$"))

    # --------------------------------------------------------
    # 3-4. 確認：main sequence の幅を変えたときの M_MIN と M_MAX
    # --------------------------------------------------------
    for w in (0.4, 0.6, 0.8):
        lo_, hi_ = mass_range(m, mbins, w)
        row[f"M_MIN_w{w:g}"], row[f"M_MAX_w{w:g}"] = lo_, hi_
        print(f"  幅 ±{w:g} dex: {lo_:.1f} <= log M* < {hi_:.1f}")

    summary.append(row)


# ============================================================
# 4. 3つの方法の比較（M* の関数）
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True)
for (method, (f_sn, f_sf, f_all, f_L, _)), col in zip(curves_by_method.items(),
                                                     ["C0", "C1", "C2"]):
    lo, hi = ranges_by_method[method]
    axes[0].plot(mcen, f_all, "o-", color=col, label=f"{method} ({lo:.1f}–{hi:.1f})")
    axes[1].plot(mcen, f_L,   "o-", color=col, label=method)
    for ax in axes:
        for v in (lo, hi):
            if np.isfinite(v):
                ax.axvline(v, color=col, ls=":", lw=1)
axes[0].set_title("All cuts (S/N + SF + $L$)")
axes[1].set_title(r"$L$ cut | SF")
for ax in axes:
    ax.axhline(COMP_THRESH, ls=":", color="gray")
    ax.set_xlabel(r"$\log(M_*/M_\odot)$")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=12)
axes[0].set_ylabel("Fraction of MS galaxies retained")
plt.savefig(os.path.join(fig_dir, "completeness_vs_mass_methods.png"), bbox_inches="tight")
plt.show()

summary_df = pd.DataFrame(summary)
print("\n===== Summary =====")
print(summary_df.T.to_string())
summary_df.to_csv(os.path.join(fig_dir, "completeness_summary.csv"), index=False)