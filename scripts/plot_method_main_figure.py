# -*- coding: utf-8 -*-
"""Draw the manuscript-style main method figure.

The figure is a schematic-led composite: a split VLM pipeline leads into a
client-side hyperspherical vMF privacy layer, with compact guarantee panels.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


# Mandatory publication export settings: keep SVG/PDF text editable.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.size"] = 7.5
plt.rcParams["mathtext.fontset"] = "dejavusans"
plt.rcParams["axes.linewidth"] = 0.7
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["legend.frameon"] = False


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "asset"

PALETTE = {
    "blue": "#0F4D92",
    "blue_mid": "#3775BA",
    "blue_soft": "#E9F2FB",
    "aqua": "#DDEFEF",
    "teal": "#42949E",
    "teal_dark": "#236C73",
    "rose": "#E4CCD8",
    "rose_dark": "#9B4E68",
    "peach": "#F0E0D0",
    "orange": "#C77925",
    "green_soft": "#DDF3DE",
    "green": "#4E9D62",
    "red": "#B64342",
    "red_soft": "#F6CFCB",
    "gray0": "#FFFFFF",
    "gray1": "#F6F7F8",
    "gray2": "#D8D8D8",
    "gray3": "#8A8A8A",
    "gray4": "#4D4D4D",
    "black": "#272727",
}


def rounded_box(ax, x, y, w, h, fc, ec, lw=0.8, radius=0.012, z=1):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.006,rounding_size={radius}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def arrow(ax, p0, p1, color=None, lw=1.0, style="-", rad=0.0, ms=8, z=6):
    if color is None:
        color = PALETTE["black"]
    patch = FancyArrowPatch(
        p0,
        p1,
        arrowstyle="-|>",
        mutation_scale=ms,
        linewidth=lw,
        linestyle=style,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def panel_label(ax, x, y, label):
    ax.text(x, y, label, fontsize=8.5, fontweight="bold", color=PALETTE["black"], ha="left", va="top")


def section_title(ax, x, y, text, color=None):
    ax.text(
        x,
        y,
        text,
        fontsize=7.6,
        fontweight="bold",
        color=color or PALETTE["blue"],
        ha="left",
        va="top",
    )


def small_text(ax, x, y, text, *, color=None, ha="left", va="top", size=6.4, linespacing=1.18):
    ax.text(
        x,
        y,
        text,
        fontsize=size,
        color=color or PALETTE["gray4"],
        ha=ha,
        va=va,
        linespacing=linespacing,
    )


def draw_image_icon(ax, x, y, w, h):
    ax.add_patch(Rectangle((x, y), w, h, facecolor="#EEF5FF", edgecolor=PALETTE["blue_mid"], lw=0.7))
    ax.add_patch(Circle((x + w * 0.78, y + h * 0.78), w * 0.055, color="#F0B04A", ec="none"))
    ax.add_patch(
        Polygon(
            [(x + 0.10 * w, y + 0.18 * h), (x + 0.40 * w, y + 0.56 * h), (x + 0.62 * w, y + 0.18 * h)],
            closed=True,
            fc="#83BDF2",
            ec=PALETTE["blue_mid"],
            lw=0.6,
        )
    )
    ax.add_patch(
        Polygon(
            [(x + 0.35 * w, y + 0.18 * h), (x + 0.63 * w, y + 0.48 * h), (x + 0.90 * w, y + 0.18 * h)],
            closed=True,
            fc="#B7D7F4",
            ec=PALETTE["blue_mid"],
            lw=0.6,
        )
    )


def draw_text_icon(ax, x, y, w, h):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=PALETTE["gray0"], edgecolor=PALETTE["blue_mid"], lw=0.7))
    for i, width in enumerate([0.78, 0.58, 0.74, 0.46]):
        yy = y + h * (0.74 - i * 0.18)
        ax.plot([x + 0.12 * w, x + (0.12 + width) * w], [yy, yy], color=PALETTE["gray4"], lw=0.9)


def embedding_bars(ax, x, y, w, h, color, n=12, alpha=1.0):
    values = [0.28, 0.70, 0.44, 0.82, 0.54, 0.38, 0.74, 0.56, 0.36, 0.68, 0.48, 0.76]
    gap = w / n
    for i in range(n):
        bh = h * values[i % len(values)]
        ax.add_patch(
            Rectangle(
                (x + i * gap + gap * 0.24, y + (h - bh) / 2),
                gap * 0.52,
                bh,
                facecolor=color,
                edgecolor="none",
                alpha=alpha,
            )
        )


def token_row(ax, x, y, w, h):
    labels = ["sys", "task", "user", "private", "answer"]
    widths = [0.13, 0.15, 0.13, 0.18, 0.15]
    colors = [PALETTE["gray2"], PALETTE["gray2"], PALETTE["peach"], PALETTE["rose"], PALETTE["peach"]]
    xx = x
    for lab, ww, col in zip(labels, widths, colors):
        bw = w * ww
        rounded_box(ax, xx, y, bw, h, col, PALETTE["gray3"], lw=0.45, radius=0.004, z=4)
        ax.text(xx + bw / 2, y + h / 2, lab, fontsize=5.7, ha="center", va="center", color=PALETTE["black"])
        xx += bw + w * 0.015


def draw_sphere_mechanism(ax, cx, cy, r):
    ax.add_patch(Circle((cx, cy), r, facecolor=PALETTE["gray0"], edgecolor=PALETTE["blue"], lw=0.9, zorder=2))
    ax.add_patch(Arc((cx, cy), 2 * r, 0.68 * r, theta1=0, theta2=360, color=PALETTE["gray2"], lw=0.7, zorder=3))
    ax.add_patch(Arc((cx, cy), 1.25 * r, 2 * r, theta1=0, theta2=360, color=PALETTE["gray2"], lw=0.7, zorder=3))
    ax.plot([cx - r, cx + r], [cy, cy], color=PALETTE["gray2"], lw=0.55, zorder=3)
    ax.plot([cx, cx], [cy - r, cy + r], color=PALETTE["gray2"], lw=0.55, zorder=3)

    # Tangent plane and vectors.
    plane = Polygon(
        [(cx - 0.12 * r, cy + 0.60 * r), (cx + 0.96 * r, cy + 0.25 * r), (cx + 0.44 * r, cy - 0.03 * r), (cx - 0.55 * r, cy + 0.32 * r)],
        closed=True,
        fc=PALETTE["aqua"],
        ec=PALETTE["teal"],
        lw=0.7,
        alpha=0.75,
        zorder=4,
    )
    ax.add_patch(plane)
    arrow(ax, (cx, cy), (cx + 0.58 * r, cy + 0.24 * r), color=PALETTE["blue"], lw=1.1, ms=8, z=7)
    arrow(ax, (cx + 0.58 * r, cy + 0.24 * r), (cx + 0.27 * r, cy + 0.70 * r), color=PALETTE["orange"], lw=1.0, ms=8, z=7)
    arrow(ax, (cx, cy), (cx + 0.58 * r, cy + 0.54 * r), color=PALETTE["rose_dark"], lw=1.0, ms=8, z=7)
    ax.text(cx + 0.62 * r, cy + 0.19 * r, r"$\mu$", fontsize=7.0, color=PALETTE["blue"], ha="left", va="center")
    ax.text(cx + 0.26 * r, cy + 0.76 * r, r"$\hat n_\perp$", fontsize=6.6, color=PALETTE["orange"], ha="left", va="center")
    ax.text(cx + 0.63 * r, cy + 0.58 * r, r"$\mu'$", fontsize=7.0, color=PALETTE["rose_dark"], ha="left", va="center")


def draw_attacker(ax, x, y):
    ax.add_patch(Circle((x, y + 0.022), 0.012, fc=PALETTE["red_soft"], ec=PALETTE["red"], lw=0.6, zorder=7))
    ax.add_patch(Rectangle((x - 0.015, y - 0.015), 0.030, 0.025, fc=PALETTE["red_soft"], ec=PALETTE["red"], lw=0.6, zorder=7))
    ax.plot([x - 0.017, x + 0.017], [y + 0.007, y + 0.034], color=PALETTE["red"], lw=1.1, zorder=8)
    ax.plot([x + 0.017, x - 0.017], [y + 0.007, y + 0.034], color=PALETTE["red"], lw=1.1, zorder=8)


def guarantee_panel(ax, x, y, w, h, label, title, body, accent):
    rounded_box(ax, x, y, w, h, PALETTE["gray1"], PALETTE["gray2"], lw=0.75, radius=0.010, z=1)
    panel_label(ax, x + 0.012, y + h - 0.012, label)
    ax.plot([x + 0.014, x + 0.014], [y + 0.035, y + h - 0.045], color=accent, lw=2.0)
    section_title(ax, x + 0.030, y + h - 0.035, title, color=PALETTE["black"])
    small_text(ax, x + 0.030, y + h - 0.075, body, size=6.0)


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(7.25, 5.05), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # Compact title line for standalone use; the caption can replace it in paper.
    ax.text(
        0.035,
        0.968,
        "Semantic-preserving metric privacy for split VLM inference",
        fontsize=10.6,
        fontweight="bold",
        color=PALETTE["black"],
        ha="left",
        va="top",
    )
    ax.text(
        0.035,
        0.940,
        "Client-side vMF perturbation protects intermediate embeddings while preserving task semantics.",
        fontsize=6.5,
        color=PALETTE["gray4"],
        ha="left",
        va="top",
    )
    ax.plot([0.035, 0.965], [0.922, 0.922], color=PALETTE["gray2"], lw=0.6)

    panel_label(ax, 0.035, 0.898, "a")
    section_title(ax, 0.060, 0.898, "End-to-end privacy layer in split VLM inference")

    # Main schematic panels.
    left = (0.045, 0.355, 0.215, 0.500)
    center = (0.305, 0.315, 0.405, 0.585)
    right = (0.755, 0.355, 0.200, 0.500)
    rounded_box(ax, *left, PALETTE["gray0"], PALETTE["gray2"], lw=0.8)
    rounded_box(ax, *center, "#FBFCFD", PALETTE["blue_mid"], lw=0.95)
    rounded_box(ax, *right, PALETTE["gray0"], PALETTE["gray2"], lw=0.8)

    # Client side.
    section_title(ax, 0.065, 0.820, "Client side")
    draw_image_icon(ax, 0.070, 0.715, 0.070, 0.075)
    draw_text_icon(ax, 0.155, 0.715, 0.070, 0.075)
    small_text(ax, 0.105, 0.704, "image", ha="center", size=5.8)
    small_text(ax, 0.190, 0.704, "prompt", ha="center", size=5.8)
    rounded_box(ax, 0.075, 0.620, 0.150, 0.052, PALETTE["blue_soft"], PALETTE["blue_mid"], lw=0.6, radius=0.008)
    ax.text(0.150, 0.646, "local encoders", fontsize=6.4, fontweight="bold", color=PALETTE["blue"], ha="center", va="center")
    small_text(ax, 0.150, 0.628, "visual patches + tokens", ha="center", size=5.5)
    embedding_bars(ax, 0.080, 0.545, 0.140, 0.038, PALETTE["blue_mid"])
    small_text(ax, 0.150, 0.520, r"intermediate $x=r\mu$", ha="center", size=6.0)
    ax.text(0.060, 0.405, "raw image and text\nremain local", fontsize=5.9, color=PALETTE["gray4"], ha="left", va="top")

    # Center privacy layer.
    section_title(ax, 0.325, 0.855, "Geometric privacy layer")
    small_text(ax, 0.325, 0.826, r"Embedding directions lie on $\mathbb{S}^{d-1}$; privacy is measured by angular distance.", size=5.9)

    rounded_box(ax, 0.330, 0.580, 0.355, 0.210, PALETTE["aqua"], PALETTE["teal"], lw=0.8, radius=0.010)
    section_title(ax, 0.350, 0.760, "Orthogonal tangent-plane vMF perturbation", color=PALETTE["teal_dark"])
    draw_sphere_mechanism(ax, 0.430, 0.675, 0.055)
    ax.text(
        0.520,
        0.695,
        r"$P_\perp=I-\mu\mu^{T}$" "\n" r"$\lambda=\beta/\epsilon$" "\n" r"$y=r\mu'$",
        fontsize=7.0,
        color=PALETTE["black"],
        ha="left",
        va="center",
        linespacing=1.35,
    )
    small_text(ax, 0.520, 0.612, "direction is perturbed;\nnorm is restored", size=5.8)

    rounded_box(ax, 0.330, 0.445, 0.355, 0.092, PALETTE["peach"], PALETTE["orange"], lw=0.65, radius=0.008)
    section_title(ax, 0.350, 0.515, "Asymmetric budgets", color=PALETTE["orange"])
    ax.text(0.585, 0.506, r"$\epsilon_V=\epsilon_{sys}$" "\n" r"$\epsilon_T=\alpha\epsilon_{sys}$", fontsize=6.4, color=PALETTE["black"], ha="center", va="center", linespacing=1.1)
    small_text(ax, 0.350, 0.474, "stronger visual protection;\nweaker text perturbation", size=5.55)

    rounded_box(ax, 0.330, 0.350, 0.355, 0.075, PALETTE["green_soft"], PALETTE["green"], lw=0.65, radius=0.008)
    section_title(ax, 0.350, 0.407, "Sensitive masking", color=PALETTE["green"])
    token_row(ax, 0.515, 0.385, 0.135, 0.026)
    ax.text(0.350, 0.374, r"$y_f=M\odot y+(1-M)\odot x$", fontsize=6.45, color=PALETTE["black"], ha="left", va="center")

    # Cloud side.
    section_title(ax, 0.775, 0.820, "Cloud side")
    embedding_bars(ax, 0.785, 0.735, 0.145, 0.038, PALETTE["teal"], alpha=0.95)
    embedding_bars(ax, 0.785, 0.682, 0.145, 0.034, PALETTE["orange"], alpha=0.88)
    small_text(ax, 0.858, 0.658, "protected embeddings", ha="center", size=6.0)
    rounded_box(ax, 0.790, 0.575, 0.135, 0.058, PALETTE["blue_soft"], PALETTE["blue_mid"], lw=0.6, radius=0.008)
    ax.text(0.858, 0.604, "VLM decoder", fontsize=6.5, fontweight="bold", color=PALETTE["blue"], ha="center", va="center")
    rounded_box(ax, 0.790, 0.455, 0.135, 0.065, PALETTE["green_soft"], PALETTE["green"], lw=0.65, radius=0.008)
    ax.text(0.858, 0.492, "task output", fontsize=6.4, fontweight="bold", color=PALETTE["green"], ha="center", va="center")
    small_text(ax, 0.858, 0.470, "captioning / VQA", ha="center", size=5.5)
    draw_attacker(ax, 0.925, 0.695)
    small_text(ax, 0.935, 0.650, "attack\nblocked", ha="center", color=PALETTE["red"], size=5.55)
    arrow(ax, (0.920, 0.695), (0.890, 0.714), color=PALETTE["red"], lw=0.75, style="--", rad=0.25, ms=6)

    # Flow arrows.
    arrow(ax, (0.150, 0.610), (0.150, 0.585), color=PALETTE["gray4"], lw=0.9)
    arrow(ax, (0.260, 0.565), (0.305, 0.565), color=PALETTE["black"], lw=1.0, ms=9)
    small_text(ax, 0.282, 0.584, "split", ha="center", size=5.6)
    arrow(ax, (0.666, 0.548), (0.666, 0.538), color=PALETTE["teal_dark"], lw=0.85, ms=7)
    arrow(ax, (0.666, 0.445), (0.666, 0.425), color=PALETTE["teal_dark"], lw=0.85, ms=7)
    arrow(ax, (0.710, 0.565), (0.755, 0.565), color=PALETTE["black"], lw=1.0, ms=9)
    small_text(ax, 0.733, 0.584, "upload", ha="center", size=5.6)
    arrow(ax, (0.858, 0.660), (0.858, 0.633), color=PALETTE["gray4"], lw=0.9, ms=8)
    arrow(ax, (0.858, 0.575), (0.858, 0.520), color=PALETTE["gray4"], lw=0.9, ms=8)

    # Bottom guarantee row.
    guarantee_panel(
        ax,
        0.045,
        0.085,
        0.285,
        0.200,
        "b",
        "Norm-preserving geometry",
        r"Perturbation changes direction, not activation scale." "\n" r"$\|y\|_2=\|x\|_2$ stabilizes LayerNorm and attention.",
        PALETTE["teal"],
    )
    # Minimal equal-norm visual cue.
    ax.plot([0.085, 0.135], [0.125, 0.125], color=PALETTE["blue"], lw=2.2)
    ax.plot([0.085, 0.135], [0.105, 0.105], color=PALETTE["rose_dark"], lw=2.2)
    small_text(ax, 0.145, 0.130, r"$x$", size=5.6)
    small_text(ax, 0.145, 0.110, r"$y$", size=5.6)

    guarantee_panel(
        ax,
        0.365,
        0.085,
        0.285,
        0.200,
        "c",
        "Approximate metric privacy",
        r"Nearby semantic directions have close output laws." "\n" r"$\Pr[M(x)\in S]\leq e^{\kappa d_{geo}}\Pr[M(x')\in S]+2\delta_{TV}$.",
        PALETTE["blue"],
    )
    for i, color in enumerate([PALETTE["blue_mid"], PALETTE["rose_dark"]]):
        cx = 0.415 + i * 0.040
        ax.add_patch(Circle((cx, 0.125), 0.030, fc=color, ec="none", alpha=0.22))
        ax.add_patch(Circle((cx, 0.125), 0.006, fc=color, ec="none", alpha=0.90))

    guarantee_panel(
        ax,
        0.685,
        0.085,
        0.270,
        0.200,
        "d",
        "Utility-aware protection",
        "Asymmetric budgets and token masks protect private user content\nwhile preserving system prompts and downstream utility.",
        PALETTE["orange"],
    )
    token_row(ax, 0.725, 0.110, 0.180, 0.030)

    # Export bundle.
    out = ASSET_DIR / "method_main_figure"
    fig.savefig(f"{out}.svg", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(f"{out}.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(f"{out}.png", dpi=600, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(f"{out}.tiff", dpi=600, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


if __name__ == "__main__":
    main()
