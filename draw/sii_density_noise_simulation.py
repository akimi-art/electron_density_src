"""Simulate S/N=5 measurements of the [S II] density diagnostic."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyneb as pn


def main():
    pn.atomicData.setDataFile("s_ii_atom_RGJ19.dat")
    pn.atomicData.setDataFile("s_ii_coll_TZ10.dat")
    sulfur = pn.Atom("S", 2)
    rng = np.random.default_rng(42)
    true_density = np.geomspace(10, 1000, 60)
    n_draws = 50_000
    numerator = tuple(level - 1 for level in sulfur.getTransition(6716))
    denominator = tuple(level - 1 for level in sulfur.getTransition(6731))

    def ratio(density):
        emissivities = sulfur.getEmissivity(10_000, density)
        return emissivities[numerator] / emissivities[denominator]

    ideal = ratio(true_density)
    sigma = ideal / 5
    observed = rng.normal(ideal, sigma)
    log_grid = np.linspace(-2, 6, 1601)
    ratio_grid = ratio(10**log_grid)
    # Use the decreasing branch that contains the simulated densities.
    branch_end = np.argmin(ratio_grid) + 1
    log_grid, ratio_grid = log_grid[:branch_end], ratio_grid[:branch_end]
    if not np.all(np.diff(ratio_grid) < 0):
        raise ValueError("Expected a strictly decreasing diagnostic curve.")

    def invert(values):
        return 10**np.interp(values, ratio_grid[::-1], log_grid[::-1],
                             left=np.nan, right=np.nan)

    measured = invert(observed)
    draws = rng.normal(observed[:, None], sigma[:, None],
                       size=(true_density.size, n_draws))
    # Check the untruncated ratio distribution before any physical-range filtering.
    p16, p84 = np.percentile(draws, [16, 84], axis=1)
    density_low, density_high = invert(p84), invert(p16)
    usable = (density_low <= measured) & (measured <= density_high)
    if not np.any(usable):
        raise ValueError("No densities with two-sided uncertainties.")

    # For retained objects, propagate the draws inside the physical range.
    mean_samples = []
    for density_draws in invert(draws[usable]):
        valid_draws = density_draws[np.isfinite(density_draws)]
        mean_samples.append(rng.choice(valid_draws, size=n_draws))
    measured_mean = measured[usable].mean()
    mc_low, mc_median, mc_high = np.percentile(np.mean(mean_samples, axis=0), [16, 50, 84])
    # Center the Monte Carlo spread on the direct mean of the inferred densities.
    err_low, err_high = mc_median - mc_low, mc_high - mc_median
    mean_low, mean_high = measured_mean - err_low, measured_mean + err_high
    true_mean = true_density.mean()
    count = f"{usable.sum()}/{true_density.size}"

    plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix",
                         "font.size": 15, "axes.labelsize": 19, "pdf.fonttype": 42,
                         "xtick.direction": "in", "ytick.direction": "in"})
    fig, ax = plt.subplots(figsize=(8, 5.2), layout="constrained")
    ax.plot(10**log_grid, ratio_grid, color="black", lw=1.8,
            label=r"PyNeb ($T_{\rm e}=10{,}000\,\mathrm{K}$)")
    ax.errorbar(true_density, ideal, yerr=sigma, fmt="o", color="#2166ac",
                alpha=0.38, markersize=2, capsize=0, elinewidth=0.65,
                label="Input ratios (S/N = 5)")
    ax.axvline(true_mean, color="#187451", linestyle=":", lw=2,
               label=rf"True mean: {true_mean:.0f} $\mathrm{{cm}}^{{-3}}$")
    ax.axvspan(mean_low, mean_high, color="#b44429", alpha=0.15, linewidth=0)
    result_label = (rf"{measured_mean:.0f}^{{+{err_high:.0f}}}_{{-{err_low:.0f}}}"
                    r"\,\mathrm{cm}^{-3}")
    ax.axvline(measured_mean, color="#b44429", linestyle="--", lw=2,
               label=f"Inferred mean: $" + result_label + f"$ ({count})")
    ax.set(xscale="log", xlim=(1, 1e5), ylim=(0.36, 1.83),
           xlabel=r"$n_{\rm e}\;[\mathrm{cm}^{-3}]$",
           ylabel=r"$[\mathrm{S\,II}]\;F(6716)/F(6731)$")
    ax.legend(loc="upper right", frameon=False, fontsize=12)

    output = Path(__file__).resolve().parents[2] / "output" / "pdf"
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "sii_density_noise_simulation.pdf")
    fig.savefig(output / "sii_density_noise_simulation.png", dpi=300)
    plt.close(fig)

    print("n_true   R_ideal   R_obs   sigma_R   n_inferred   err_minus   err_plus")
    for i in range(true_density.size):
        uncertainty = (f"{measured[i]-density_low[i]:.1f}   {density_high[i]-measured[i]:.1f}"
                       if usable[i] else "excluded")
        print(f"{true_density[i]:7.1f}   {ideal[i]:.4f}   {observed[i]:.4f}   "
              f"{sigma[i]:.4f}   {measured[i]:.1f}   {uncertainty}")
    print(f"True mean: {true_mean:.1f}")
    print(f"Inferred mean: {measured_mean:.1f} -{err_low:.1f} +{err_high:.1f} ({count})")
    print(f"Raw Monte Carlo mean percentiles (16/50/84): {mc_low:.1f} "
          f"{mc_median:.1f} {mc_high:.1f}")


if __name__ == "__main__":
    main()
