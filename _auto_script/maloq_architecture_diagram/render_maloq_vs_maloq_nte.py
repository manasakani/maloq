#!/usr/bin/env python3
"""Render the SC26 MALOQ vs MALOQ-NTE layer-structure comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


BG = "#F5F7FB"
INK = "#172033"
MUTED = "#5E6A7D"
LINE = "#D7DDEA"
BLUE = "#2563EB"
BLUE_LIGHT = "#EAF1FF"
ORANGE = "#E98432"
ORANGE_LIGHT = "#FFF1E6"
TEAL = "#079A91"
TEAL_LIGHT = "#E4F8F5"
PURPLE = "#7656D6"
PURPLE_LIGHT = "#F0ECFF"
GREEN = "#2F9E65"
GREEN_LIGHT = "#EAF8F0"
WHITE = "#FFFFFF"


def rounded_box(
    ax,
    x,
    y,
    width,
    height,
    *,
    facecolor=WHITE,
    edgecolor=LINE,
    linewidth=1.2,
    radius=0.12,
    zorder=1,
):
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def put_text(
    ax,
    x,
    y,
    text,
    font,
    *,
    size=10,
    color=INK,
    ha="center",
    va="center",
    weight="normal",
    linespacing=1.18,
    zorder=5,
):
    ax.text(
        x,
        y,
        text,
        fontproperties=font,
        fontsize=size,
        color=color,
        ha=ha,
        va=va,
        fontweight=weight,
        linespacing=linespacing,
        zorder=zorder,
    )


def arrow(ax, x1, y1, x2, y2, *, color=MUTED, linewidth=1.5, zorder=3):
    patch = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=linewidth,
        color=color,
        shrinkA=0,
        shrinkB=0,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def flow_box(
    ax,
    x,
    y,
    width,
    height,
    label,
    font,
    *,
    facecolor,
    edgecolor,
    text_color=INK,
    size=8.8,
):
    rounded_box(
        ax,
        x,
        y,
        width,
        height,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.4,
        radius=0.10,
        zorder=2,
    )
    put_text(
        ax,
        x + width / 2,
        y + height / 2,
        label,
        font,
        size=size,
        color=text_color,
        weight="bold",
    )


def module_chain(ax, xs, y, labels, font, colors, *, height=0.48, size=7.2):
    widths = []
    for label in labels:
        width = max(0.75, 0.085 * max(len(line) for line in label.split("\n")) + 0.34)
        widths.append(width)
    for idx, (x, width, label, palette) in enumerate(
        zip(xs, widths, labels, colors)
    ):
        face, edge = palette
        flow_box(
            ax,
            x,
            y,
            width,
            height,
            label,
            font,
            facecolor=face,
            edgecolor=edge,
            size=size,
        )
        if idx + 1 < len(labels):
            arrow(
                ax,
                x + width + 0.04,
                y + height / 2,
                xs[idx + 1] - 0.05,
                y + height / 2,
                color="#8792A5",
                linewidth=1.1,
            )
    return widths


def chip(ax, x, y, width, label, font, *, facecolor, edgecolor, color=INK):
    rounded_box(
        ax,
        x,
        y,
        width,
        0.34,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.0,
        radius=0.16,
        zorder=2,
    )
    put_text(ax, x + width / 2, y + 0.17, label, font, size=7.1, color=color)


def render(output_dir: Path, font_path: Path | None) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    font = (
        FontProperties(fname=str(font_path))
        if font_path is not None and font_path.exists()
        else FontProperties(family="DejaVu Sans")
    )

    mpl.rcParams["svg.fonttype"] = "path"
    mpl.rcParams["savefig.facecolor"] = BG

    fig, ax = plt.subplots(figsize=(16, 9), dpi=160)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")

    put_text(
        ax,
        0.65,
        8.58,
        "MALOQ vs MALOQ-NTE: Layer Structure",
        font,
        size=24,
        ha="left",
        weight="bold",
    )
    put_text(
        ax,
        0.67,
        8.17,
        "SC26 QH9Stable matched-comparison configuration",
        font,
        size=9.5,
        color=MUTED,
        ha="left",
    )

    # Main comparison panels.
    rounded_box(
        ax,
        0.48,
        1.55,
        7.35,
        6.25,
        facecolor=WHITE,
        edgecolor="#BFD0F4",
        linewidth=1.5,
        radius=0.20,
    )
    rounded_box(
        ax,
        8.17,
        1.55,
        7.35,
        6.25,
        facecolor=WHITE,
        edgecolor="#BDE5E1",
        linewidth=1.5,
        radius=0.20,
    )

    # Headers.
    rounded_box(
        ax,
        0.66,
        7.08,
        2.28,
        0.52,
        facecolor=BLUE,
        edgecolor=BLUE,
        radius=0.15,
    )
    put_text(ax, 1.80, 7.34, "MALOQ", font, size=15, color=WHITE, weight="bold")
    put_text(
        ax,
        3.15,
        7.34,
        "Interleaved schedule",
        font,
        size=9.3,
        color=BLUE,
        ha="left",
        weight="bold",
    )

    rounded_box(
        ax,
        8.35,
        7.08,
        2.70,
        0.52,
        facecolor=TEAL,
        edgecolor=TEAL,
        radius=0.15,
    )
    put_text(
        ax, 9.70, 7.34, "MALOQ-NTE", font, size=15, color=WHITE, weight="bold"
    )
    put_text(
        ax,
        11.25,
        7.34,
        "Node-then-edge schedule",
        font,
        size=9.3,
        color=TEAL,
        ha="left",
        weight="bold",
    )

    # Schedule: MALOQ.
    put_text(
        ax,
        0.82,
        6.80,
        "MESSAGE-PASSING ORDER",
        font,
        size=7.3,
        color=MUTED,
        ha="left",
        weight="bold",
    )
    flow_box(
        ax,
        0.82,
        6.02,
        1.12,
        0.58,
        "Initial\nembedding",
        font,
        facecolor="#EEF1F6",
        edgecolor="#AAB4C5",
        size=8.0,
    )
    x_positions = [2.25, 3.10, 3.95, 4.80, 5.65, 6.50]
    labels = ["Node 1", "Edge 1", "Node 2", "Edge 2", "Node 3", "Edge 3"]
    for idx, (x, label) in enumerate(zip(x_positions, labels)):
        is_node = idx % 2 == 0
        flow_box(
            ax,
            x,
            6.02,
            0.70,
            0.58,
            label,
            font,
            facecolor=BLUE_LIGHT if is_node else ORANGE_LIGHT,
            edgecolor=BLUE if is_node else ORANGE,
            text_color=BLUE if is_node else ORANGE,
            size=7.4,
        )
        if idx == 0:
            arrow(ax, 1.98, 6.31, 2.18, 6.31)
        if idx < len(labels) - 1:
            arrow(ax, x + 0.72, 6.31, x_positions[idx + 1] - 0.07, 6.31)
    put_text(
        ax,
        4.12,
        5.72,
        "Each updated node state immediately feeds its paired edge block",
        font,
        size=8.1,
        color=MUTED,
    )

    # Schedule: NTE.
    put_text(
        ax,
        8.51,
        6.80,
        "MESSAGE-PASSING ORDER",
        font,
        size=7.3,
        color=MUTED,
        ha="left",
        weight="bold",
    )
    flow_box(
        ax,
        8.51,
        6.02,
        1.12,
        0.58,
        "Initial\nembedding",
        font,
        facecolor="#EEF1F6",
        edgecolor="#AAB4C5",
        size=8.0,
    )
    nte_positions = [9.95, 10.85, 11.75, 13.05, 13.95]
    nte_labels = ["Node 1", "Node 2", "Node 3", "Edge 1", "Edge 2"]
    for idx, (x, label) in enumerate(zip(nte_positions, nte_labels)):
        is_node = idx < 3
        flow_box(
            ax,
            x,
            6.02,
            0.75,
            0.58,
            label,
            font,
            facecolor=TEAL_LIGHT if is_node else PURPLE_LIGHT,
            edgecolor=TEAL if is_node else PURPLE,
            text_color=TEAL if is_node else PURPLE,
            size=7.4,
        )
        if idx == 0:
            arrow(ax, 9.67, 6.31, 9.88, 6.31)
        if idx < len(nte_labels) - 1:
            gap_end = nte_positions[idx + 1] - 0.07
            arrow(ax, x + 0.77, 6.31, gap_end, 6.31)
    ax.plot([12.75, 12.75], [5.91, 6.71], color="#AEB7C6", linewidth=1.0)
    put_text(
        ax,
        11.35,
        6.78,
        "NODE PHASE",
        font,
        size=6.3,
        color=TEAL,
        weight="bold",
    )
    put_text(
        ax,
        13.85,
        6.78,
        "EDGE PHASE",
        font,
        size=6.3,
        color=PURPLE,
        weight="bold",
    )
    put_text(
        ax,
        11.85,
        5.62,
        "All node blocks run before any edge block",
        font,
        size=8.1,
        color=MUTED,
    )

    # Block recipe headings.
    put_text(
        ax,
        0.82,
        5.29,
        "eSEN BLOCK RECIPE",
        font,
        size=7.3,
        color=MUTED,
        ha="left",
        weight="bold",
    )
    put_text(
        ax,
        8.51,
        5.29,
        "eSEN BLOCK RECIPE",
        font,
        size=7.3,
        color=MUTED,
        ha="left",
        weight="bold",
    )

    # MALOQ block internals.
    rounded_box(
        ax,
        0.80,
        3.20,
        6.70,
        1.90,
        facecolor="#FAFBFE",
        edgecolor="#D9E2F6",
        radius=0.14,
    )
    put_text(
        ax, 1.02, 4.73, "Message branch", font, size=7.8, color=BLUE, ha="left"
    )
    labels = ["RMS\nNorm", "SO(2)\nmessage", "tanh\ngate", "Residual\nadd"]
    xs = [1.00, 2.20, 3.48, 4.60]
    colors = [
        ("#EEF1F6", "#AAB4C5"),
        (BLUE_LIGHT, BLUE),
        (BLUE_LIGHT, BLUE),
        (GREEN_LIGHT, GREEN),
    ]
    module_chain(ax, xs, 4.12, labels, font, colors, height=0.48, size=7.0)
    put_text(
        ax, 1.02, 3.88, "Atomwise branch", font, size=7.8, color=BLUE, ha="left"
    )
    labels = ["RMS\nNorm", "Spectral FFN\nSO(3) Linear → gate → Linear", "Residual\nadd"]
    xs = [1.00, 2.20, 5.18]
    colors = [
        ("#EEF1F6", "#AAB4C5"),
        (BLUE_LIGHT, BLUE),
        (GREEN_LIGHT, GREEN),
    ]
    module_chain(ax, xs, 3.30, labels, font, colors, height=0.48, size=6.8)
    chip(
        ax,
        5.70,
        4.50,
        1.48,
        "No LayerScale",
        font,
        facecolor="#F3F4F7",
        edgecolor="#B9C1CE",
        color=MUTED,
    )

    # NTE block internals.
    rounded_box(
        ax,
        8.49,
        3.20,
        6.70,
        1.90,
        facecolor="#FAFEFD",
        edgecolor="#CFEAE7",
        radius=0.14,
    )
    put_text(
        ax, 8.71, 4.73, "Message branch", font, size=7.8, color=TEAL, ha="left"
    )
    labels = [
        "RMS\nNorm",
        "SO(2)\nmessage",
        "sigmoid\ngate",
        "Degree\nLayerScale",
        "Residual\nadd",
    ]
    xs = [8.69, 9.78, 10.98, 12.12, 13.42]
    colors = [
        ("#EEF1F6", "#AAB4C5"),
        (TEAL_LIGHT, TEAL),
        (TEAL_LIGHT, TEAL),
        (PURPLE_LIGHT, PURPLE),
        (GREEN_LIGHT, GREEN),
    ]
    module_chain(ax, xs, 4.12, labels, font, colors, height=0.48, size=6.7)
    put_text(
        ax, 8.71, 3.88, "Atomwise branch", font, size=7.8, color=TEAL, ha="left"
    )
    labels = [
        "RMS\nNorm",
        "Grid FFN\nSO(3) grid + SiLU MLP",
        "Degree\nLayerScale",
        "Residual\nadd",
    ]
    xs = [8.69, 9.78, 12.20, 13.55]
    colors = [
        ("#EEF1F6", "#AAB4C5"),
        (TEAL_LIGHT, TEAL),
        (PURPLE_LIGHT, PURPLE),
        (GREEN_LIGHT, GREEN),
    ]
    module_chain(ax, xs, 3.30, labels, font, colors, height=0.48, size=6.3)
    chip(
        ax,
        11.65,
        4.70,
        1.35,
        "Distance envelope",
        font,
        facecolor=ORANGE_LIGHT,
        edgecolor=ORANGE,
        color=ORANGE,
    )
    chip(
        ax,
        13.08,
        4.70,
        1.45,
        "Edge: scalar mod.",
        font,
        facecolor=PURPLE_LIGHT,
        edgecolor=PURPLE,
        color=PURPLE,
    )

    # Output/readout difference.
    flow_box(
        ax,
        1.05,
        2.18,
        2.70,
        0.56,
        "Output width 128\nIdentity projection",
        font,
        facecolor=BLUE_LIGHT,
        edgecolor=BLUE,
        text_color=BLUE,
        size=8.0,
    )
    chip(
        ax,
        4.00,
        2.29,
        2.55,
        "tanh · Spectral · no LayerScale",
        font,
        facecolor="#F3F4F7",
        edgecolor="#B9C1CE",
        color=MUTED,
    )

    flow_box(
        ax,
        8.74,
        2.18,
        2.70,
        0.56,
        "Readout 128 → 64\nSO(3) Linear",
        font,
        facecolor=TEAL_LIGHT,
        edgecolor=TEAL,
        text_color=TEAL,
        size=8.0,
    )
    chip(
        ax,
        11.70,
        2.29,
        3.05,
        "sigmoid · Grid · LayerScale init 1/64",
        font,
        facecolor=PURPLE_LIGHT,
        edgecolor=PURPLE,
        color=PURPLE,
    )

    # Common output head.
    arrow(ax, 3.68, 2.16, 5.20, 1.30, color=BLUE, linewidth=1.4)
    arrow(ax, 10.80, 2.16, 10.80, 1.30, color=TEAL, linewidth=1.4)
    rounded_box(
        ax,
        4.05,
        0.73,
        7.90,
        0.62,
        facecolor="#172033",
        edgecolor="#172033",
        radius=0.18,
        zorder=2,
    )
    put_text(
        ax,
        8.00,
        1.04,
        "Shared native MALOQ coupled-irrep head  →  orbital-matrix labels (H / D)",
        font,
        size=9.0,
        color=WHITE,
        weight="bold",
    )
    put_text(
        ax,
        8.00,
        0.36,
        "Reference config shown: MALOQ = Node 3 + Edge 3, output 128  |  "
        "MALOQ-NTE = Node 3 + Edge 2, readout 64. The do128 / Le3 ablation matches only width and edge depth.",
        font,
        size=6.9,
        color=MUTED,
    )

    png_path = output_dir / "maloq_vs_maloq_nte_layers.png"
    svg_path = output_dir / "maloq_vs_maloq_nte_layers.svg"
    fig.savefig(png_path, dpi=180, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return png_path, svg_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    parser.add_argument("--font", type=Path, default=None)
    args = parser.parse_args()
    png_path, svg_path = render(args.output_dir, args.font)
    print(png_path)
    print(svg_path)


if __name__ == "__main__":
    main()
