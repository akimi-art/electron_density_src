"""Compare the [O II] and [S II] electron-density diagnostics at 10,000 K."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyneb as pn


OUTPUT = Path(__file__).resolve().parents[2] / "output" / "pdf"
TEMPERATURE = 10_000


def main():
    pn.atomicData.setDataFile("s_ii_atom_RGJ19.dat")
    pn.atomicData.setDataFile("s_ii_coll_TZ10.dat")
    density = np.geomspace(1, 1e5, 601)
    diagnostics = (
        (pn.Atom("O", 2), 3729, 3726, "-", "O"),
        (pn.Atom("S", 2), 6716, 6731, "--", "S"),
    )

    with plt.rc_context({
        "font.family": "STIXGeneral", "mathtext.fontset": "stix",
        "font.size": 16, "axes.labelsize": 20, "axes.linewidth": 1.3,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True, "pdf.fonttype": 42,
    }):
        fig, ax = plt.subplots(figsize=(6.5, 4.8), layout="constrained")
        for atom, numerator, denominator, style, element in diagnostics:
            ratio = (
                atom.getEmissivity(TEMPERATURE, density, wave=numerator)
                / atom.getEmissivity(TEMPERATURE, density, wave=denominator)
            )
            ax.plot(density, ratio, color="black", linestyle=style, linewidth=2.1,
                    label=rf"$[\mathrm{{{element}\,II}}]\;F({numerator})/F({denominator})$")

        ax.set(xscale="log", xlim=(1, 1e5),
               xlabel=r"$n_{\rm e}\;[\mathrm{cm}^{-3}]$", ylabel=r"$R$")
        ax.tick_params(which="major", length=7)
        ax.tick_params(which="minor", length=3)
        ax.legend(loc="upper right", frameon=False, handlelength=2.8,
                  title=rf"$T_{{\rm e}}={TEMPERATURE:,}\,\mathrm{{K}}$",
                  alignment="left")

        OUTPUT.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUTPUT / "oii_sii_density_diagnostic.pdf")
        fig.savefig(OUTPUT / "oii_sii_density_diagnostic.png", dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    main()
