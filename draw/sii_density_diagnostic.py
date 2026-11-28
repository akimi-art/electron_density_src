"""Plot the theoretical [S II] density diagnostic without observations."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np
import pyneb as pn


def main():
    output = Path(__file__).resolve().parents[2] / "output" / "pdf"
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "figure.figsize": (6, 6), "font.size": 28,
        "axes.labelsize": 28, "axes.titlesize": 28, "axes.linewidth": 2,
        "axes.grid": False, "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "xtick.major.size": 16, "ytick.major.size": 16,
        "xtick.major.width": 2, "ytick.major.width": 2,
        "xtick.minor.visible": True, "ytick.minor.visible": True,
        "xtick.minor.size": 4, "ytick.minor.size": 4,
        "xtick.minor.width": 1.5, "ytick.minor.width": 1.5,
        "xtick.labelsize": 28, "ytick.labelsize": 28,
        "font.family": "STIXGeneral", "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
    })

    log_density = np.linspace(0, 5, 601)
    density = 10**log_density
    pn.atomicData.setDataFile("s_ii_atom_RGJ19.dat")
    pn.atomicData.setDataFile("s_ii_coll_TZ10.dat")
    sulfur = pn.Atom("S", 2)
    fig, ax = plt.subplots(layout="constrained")

    for temperature, color in ((10_000, "black"), (15_000, "crimson")):
        numerator = sulfur.getEmissivity(temperature, density, wave=6716)
        denominator = sulfur.getEmissivity(temperature, density, wave=6731)
        ratio = numerator / denominator
        assert np.all(np.isfinite(ratio)) and np.all(np.diff(ratio) < 0)
        ax.plot(log_density, ratio, color=color, lw=2.5, linestyle="-",
                label=rf"$T_{{\rm e}}$ = {temperature:,} K")

    ax.set(xlim=(0, 5), ylim=(0.4, 1.56),
           xlabel=r"$\log(n_{\rm e}/\mathrm{cm}^{-3})$",
           ylabel=r"$R_{[\mathrm{S\,II}]}$")
    ax.set_yticks(np.arange(0.4, 1.51, 0.2))
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(4))
    ax.legend(loc="upper right", frameon=False, fontsize=20)

    fig.savefig(output / "sii_density_diagnostic.pdf")
    fig.savefig(output / "sii_density_diagnostic.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
