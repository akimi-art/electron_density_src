"""Plot the [O II] and [S II] doublets used for electron-density diagnostics."""

from pathlib import Path

import matplotlib.pyplot as plt
import pyneb as pn


OUTPUT = Path(__file__).resolve().parents[2] / "output" / "pdf"
COLORS = ("#0000ff", "#ff0000")


def draw_transitions(ax, atom, ion, wavelengths):
    transitions = sorted(
        [(wave, *atom.getTransition(wave)) for wave in wavelengths],
        key=lambda transition: transition[1],
        reverse=True,
    )
    ground = transitions[0][2]
    upper_levels = sorted(upper for _, upper, _ in transitions)
    # Schematic heights keep the fine-structure levels legible.
    heights = dict(zip(upper_levels, (0.79, 0.91)))
    heights[ground] = 0.17
    for level, height in heights.items():
        term = "4S" if level == ground else "2D"
        multiplicity, orbital = term
        twice_j = int(atom.getStatWeight(level=level)) - 1
        ax.hlines(height, 0.08, 0.65, color="#28323b", linewidth=1.7)
        ax.text(0.68, height,
                rf"$^{{{multiplicity}}}\mathrm{{{orbital}}}^{{\circ}}_{{{twice_j}/2}}$",
                fontsize=17, va="center")

    for index, ((wave, upper, lower), color) in enumerate(zip(transitions, COLORS)):
        x = (0.23, 0.49)[index]
        height = heights[upper]
        ax.annotate("", xy=(x, heights[lower]), xytext=(x, height),
                    arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.8,
                                "mutation_scale": 15, "shrinkA": 1, "shrinkB": 2})
        ax.text(x, 0.105, rf"$\lambda {wave}\,\AA$",
                color=color, fontsize=15, ha="center", va="top")

    ax.set_title(rf"$[\mathrm{{{ion}\,II}}]$", fontsize=23, pad=7)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.set_axis_off()


def main():
    pn.atomicData.setDataFile("s_ii_atom_RGJ19.dat")
    pn.atomicData.setDataFile("s_ii_coll_TZ10.dat")
    oxygen = pn.Atom("O", 2)
    sulfur = pn.Atom("S", 2)

    with plt.rc_context({"font.family": "STIXGeneral", "mathtext.fontset": "stix",
                         "pdf.fonttype": 42}):
        fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.65))
        fig.subplots_adjust(left=0.035, right=0.98, bottom=0.045, top=0.85, wspace=0.14)
        draw_transitions(axes[0], oxygen, "O", (3726, 3729))
        draw_transitions(axes[1], sulfur, "S", (6716, 6731))

        OUTPUT.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUTPUT / "oii_sii_density_transitions.pdf")
        fig.savefig(OUTPUT / "oii_sii_density_transitions.png", dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    main()
