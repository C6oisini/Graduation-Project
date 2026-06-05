from __future__ import annotations

import matplotlib.pyplot as plt

EN_FONT = "Times New Roman"
CN_FONT = "Songti SC"


def apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": [EN_FONT, CN_FONT],
            "font.serif": [EN_FONT, CN_FONT],
            "font.sans-serif": [EN_FONT, CN_FONT],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "svg.fonttype": "none",
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "axes.titlepad": 12,
            "axes.labelpad": 8,
        }
    )


def style_axis(ax, *, grid_axis: str = "both") -> None:
    ax.grid(True, axis=grid_axis, alpha=0.22, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
        spine.set_color("#4B5563")
    ax.tick_params(axis="both", which="major", pad=5, length=4, width=0.9)
