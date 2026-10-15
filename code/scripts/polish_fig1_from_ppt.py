from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle, Circle
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
PPTX_CANDIDATES = [
    ROOT / "paper_fig1_revised_clean_noresults_nocheckpoint.pptx",
    ROOT / "paper" / "source" / "figures" / "paper_fig1_revised.pptx",
]
OUT = ROOT / "paper" / "source" / "figures"


mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7.3,
    }
)


C = {
    "ink": "#111111",
    "muted": "#3E4A56",
    "panel": "#EEEEEE",
    "panel_edge": "#D7D7D7",
    "line": "#2F5EA7",
    "blue": "#6E93D4",
    "blue_dark": "#2D5A9D",
    "blue_soft": "#DDE8F6",
    "orange": "#F19A4B",
    "orange_soft": "#FFE1BF",
    "teal": "#9FC6D2",
    "teal_dark": "#5B94A3",
    "red": "#D47B6E",
    "red_soft": "#EEC2BA",
    "green": "#B6D7A8",
    "green_edge": "#4A7F3F",
    "ultra": "#D9A19B",
    "white": "#FFFFFF",
}


def load_road_image():
    for pptx in PPTX_CANDIDATES:
        if not pptx.exists():
            continue
        with ZipFile(pptx) as zf:
            for name in zf.namelist():
                if name == "ppt/media/image1.png":
                    return Image.open(BytesIO(zf.read(name))).convert("RGBA")
    raise FileNotFoundError("Could not find road scene image in Fig. 1 PPTX source.")


def box(ax, x, y, w, h, text="", fc="white", ec="#404D5C", lw=1.0, r=0.014, fs=7.2, weight="normal",
        color=None, ha="center", va="center", linespacing=1.05, z=2):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.004,rounding_size={r}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z,
    )
    ax.add_patch(patch)
    if text:
        tx = x + w / 2 if ha == "center" else x + 0.012
        ax.text(tx, y + h / 2, text, ha=ha, va=va, fontsize=fs, fontweight=weight,
                color=color or C["ink"], linespacing=linespacing, zorder=z + 1)
    return patch


def arrow(ax, a, b, color=None, lw=1.6, ms=11, rad=0.0, ls="-", z=5):
    ax.add_patch(
        FancyArrowPatch(
            a, b, arrowstyle="-|>", mutation_scale=ms, lw=lw, color=color or C["line"],
            linestyle=ls, connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2, zorder=z,
        )
    )


def label(ax, x, y, text, fs=7.5, weight="normal", color=None, ha="left", va="center", style="normal"):
    ax.text(x, y, text, fontsize=fs, fontweight=weight, color=color or C["ink"],
            ha=ha, va=va, fontstyle=style)


def small_graph(ax, cx, cy, s=1.0):
    pts = [
        (cx - 0.030*s, cy + 0.018*s),
        (cx + 0.000*s, cy + 0.034*s),
        (cx + 0.030*s, cy + 0.010*s),
        (cx + 0.003*s, cy - 0.026*s),
        (cx - 0.033*s, cy - 0.018*s),
    ]
    for i, j in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (1, 3)]:
        ax.plot([pts[i][0], pts[j][0]], [pts[i][1], pts[j][1]], lw=1.0, color="#6EA4C2", zorder=4)
    for k, (x, y) in enumerate(pts):
        col = "#70A5C7" if k != 2 else "#8DBB82"
        ax.add_patch(Circle((x, y), 0.010*s, fc=col, ec="white", lw=0.5, zorder=5))


def draw_panel_a(ax, road):
    x0, y0, w, h = 0.025, 0.145, 0.285, 0.725
    box(ax, x0, y0, w, h, fc=C["panel"], ec="none", r=0.020, z=0)
    label(ax, x0, 0.905, "(a) Scenario Generation and Protocol Freezing", fs=8.4, weight="bold")
    label(ax, x0 + w/2, 0.815, "Multi-candidate Scenarios\n(CARLA)", fs=8.2, weight="bold", ha="center")
    ax.imshow(road, extent=(x0 + 0.025, x0 + w - 0.025, 0.665, 0.775), aspect="auto", zorder=2)
    box(ax, x0 + 0.030, 0.608, w - 0.060, 0.048, "Fixed IDs + mask + target", fc="white", ec="#8292A0", lw=0.9, fs=6.7, r=0.006)

    box(ax, x0 + 0.030, 0.300, w - 0.060, 0.245, fc="white", ec="#4B5C6E", lw=1.1, r=0.008)
    label(ax, x0 + w/2, 0.512, "Protocol Graph Builder /\nFeature Encoder", fs=7.7, weight="bold", ha="center")
    ax.plot([x0 + w/2, x0 + w/2], [0.315, 0.482], color="#4B5C6E", lw=0.9)
    label(ax, x0 + 0.055, 0.465, "Node features (32D)", fs=6.7, weight="bold")
    label(ax, x0 + w/2 + 0.020, 0.465, "Edge features (8D)", fs=6.7, weight="bold")
    node_txt = "relative pose\nanchor frame\ndistance, velocity\nanchor/candidate flags"
    edge_txt = "relative type\nnear/front/...\nnormalized distance\nrelative angle"
    for i, line in enumerate(node_txt.splitlines()):
        y = 0.438 - i * 0.033
        ax.add_patch(Rectangle((x0 + 0.050, y - 0.008), 0.010, 0.010, fc=C["blue"], ec=C["blue"], lw=0.2))
        label(ax, x0 + 0.067, y, line, fs=6.1, weight="bold" if i == 0 else "normal")
    for i, line in enumerate(edge_txt.splitlines()):
        y = 0.438 - i * 0.033
        ax.add_patch(Rectangle((x0 + w/2 + 0.022, y - 0.008), 0.010, 0.010, fc=C["blue"], ec=C["blue"], lw=0.2))
        label(ax, x0 + w/2 + 0.039, y, line, fs=6.1, weight="bold" if i == 0 else "normal")

    arrow(ax, (x0 + w/2, 0.300), (x0 + w/2, 0.245), color=C["blue"], lw=1.8, ms=11)
    box(ax, x0 + 0.028, 0.205, w - 0.056, 0.046, "Frozen scene graphs + candidate_mask", fc=C["orange"], ec=C["orange"], lw=0.8, fs=6.6, r=0.006)


def draw_panel_b(ax):
    x0, y0, w, h = 0.335, 0.145, 0.310, 0.725
    box(ax, x0, y0, w, h, fc=C["panel"], ec="none", r=0.020, z=0)
    label(ax, x0, 0.905, "(b) Graph Encoder and Query-Conditioned Scoring", fs=8.4, weight="bold")

    box(ax, x0 + 0.070, 0.785, 0.210, 0.075, fc="white", ec="#1B1B1B", lw=1.0, r=0.008)
    small_graph(ax, x0 + 0.095, 0.823, s=0.92)
    label(ax, x0 + 0.138, 0.825, "Input protocol\ngraphs (candidates)", fs=7.2, weight="bold")
    arrow(ax, (x0 + 0.175, 0.785), (x0 + 0.175, 0.655), color="#111111", lw=1.2, ms=10)

    box(ax, x0 + 0.025, 0.610, 0.160, 0.120, fc="white", ec="#40546A", lw=1.2, r=0.020)
    label(ax, x0 + 0.050, 0.705, "GNN Backbone", fs=7.4, weight="bold")
    box(ax, x0 + 0.070, 0.625, 0.095, 0.072, "Shared\nGraph Layers", fc=C["blue"], ec="#111111", lw=0.9, r=0.010, fs=6.8, color="white")

    label(ax, x0 + 0.214, 0.690, "Shared graph encoder", fs=7.8, weight="bold")
    label(ax, x0 + 0.218, 0.645, "relation labels\n+ normalized geometry", fs=6.9, color=C["ink"])
    arrow(ax, (x0 + 0.185, 0.670), (x0 + 0.225, 0.670), color=C["blue"], lw=2.0, ms=12)

    box(ax, x0 + 0.135, 0.525, 0.155, 0.075, "Scoring head:\n$\\pi(G,q)$", fc=C["blue"], ec=C["blue"], lw=1.0, r=0.010, fs=8.0, color="white", weight="bold")
    arrow(ax, (x0 + 0.245, 0.600), (x0 + 0.245, 0.525), color="#111111", lw=1.1, ms=9)

    box(ax, x0 + 0.050, 0.455, 0.090, 0.060, "Ranking Loss\nPairwise hinge", fc=C["red"], ec="#8F4A40", lw=0.9, r=0.010, fs=6.1)
    arrow(ax, (x0 + 0.105, 0.610), (x0 + 0.105, 0.515), color="#111111", lw=1.1, ms=9)
    box(ax, x0 + 0.225, 0.405, 0.085, 0.085, "Flip Loss\n(Query-Flip\nmargin)", fc=C["teal"], ec="#111111", lw=1.0, r=0.010, fs=6.0, weight="bold")
    arrow(ax, (x0 + 0.140, 0.485), (x0 + 0.225, 0.448), color="#111111", lw=1.0, ms=9)
    arrow(ax, (x0 + 0.210, 0.525), (x0 + 0.260, 0.490), color="#111111", lw=1.0, ms=9)

    arrow(ax, (x0 + 0.212, 0.525), (x0 + 0.212, 0.250), color="#111111", lw=1.0, ms=9)
    box(ax, x0 + 0.045, 0.190, 0.190, 0.088, fc="#D8EEF3", ec="#111111", lw=1.0, r=0.010)
    label(ax, x0 + 0.140, 0.262, "3-Stage Fine-tuning Flow", fs=6.4, weight="bold", ha="center")
    xs = [x0 + 0.060, x0 + 0.118, x0 + 0.178]
    names = ["Stage1:\nQuery-only", "Stage2:\nHead", "Stage3:\nFull"]
    for i, (sx, nm) in enumerate(zip(xs, names)):
        box(ax, sx, 0.205, 0.048, 0.045, nm, fc="#E8F4F8", ec="#111111", lw=0.9, r=0.006, fs=5.7, weight="bold")
        if i < 2:
            arrow(ax, (sx + 0.048, 0.228), (xs[i+1], 0.228), color="#111111", lw=0.9, ms=7)
    box(ax, x0 + 0.245, 0.205, 0.070, 0.060, "Query-Flip\nfine-tuned selector", fc="#F6AD6B", ec="#F6AD6B", lw=0.8, r=0.006, fs=5.8, weight="bold")
    arrow(ax, (x0 + 0.235, 0.228), (x0 + 0.245, 0.228), color="#111111", lw=0.9, ms=7)


def draw_panel_c(ax):
    x0, y0, w, h = 0.665, 0.145, 0.310, 0.725
    box(ax, x0, y0, w, h, fc=C["panel"], ec="none", r=0.020, z=0)
    label(ax, x0, 0.905, "(c) Online Evaluation and Counterfactual Validation", fs=8.4, weight="bold")
    label(ax, x0 + w/2, 0.815, "Online Multi-candidate Scenarios\n& Counterfactual Evidence", fs=8.0, weight="bold", ha="center")

    box(ax, x0 + 0.020, 0.720, w - 0.040, 0.052, "Closed-loop evaluation\nshared controller; four selectors", fc="#DDE6F0", ec=C["blue_dark"], lw=1.8, r=0.006, fs=6.2)
    box(ax, x0 + 0.045, 0.605, 0.080, 0.090, "Selectors:\nGraph\nMLP\nRandom\nOracle", fc="#DDE6F0", ec=C["blue_dark"], lw=1.2, r=0.010, fs=6.0)
    ax.plot([x0 + 0.000, x0 + w], [0.595, 0.595], color="#111111", lw=1.1, ls=(0, (4, 4)))
    label(ax, x0 + w/2, 0.583, "Counterfactual pipeline", fs=7.8, weight="bold", ha="center")
    box(ax, x0 + 0.020, 0.522, w - 0.040, 0.040, "Counterfactual Pair Generator: same $G$, different $q_1 \\ne q_2$", fc="#E8EEF7", ec=C["blue_dark"], lw=1.0, r=0.005, fs=6.3)

    box(ax, x0 + 0.170, 0.410, 0.135, 0.105, fc="#D7E9D0", ec=C["blue_dark"], lw=1.1, r=0.010)
    gate_rows = [("Normal:  Δd<2.0m, Δθ<10°", C["green"]), ("Hard:    Δd<1.5m, Δθ<8°", "#EDC297"), ("Ultra:   Δd<1.0m, Δθ<5°", C["ultra"])]
    for i, (txt, col) in enumerate(gate_rows):
        box(ax, x0 + 0.178, 0.485 - i * 0.033, 0.118, 0.025, txt, fc=col, ec="#111111", lw=0.7, r=0.004, fs=5.7)
    label(ax, x0 + 0.070, 0.465, "Geometry\nMatching Gate", fs=7.2, ha="center")
    arrow(ax, (x0 + 0.120, 0.452), (x0 + 0.170, 0.452), color="#111111", lw=1.0, ms=8)

    label(ax, x0 + 0.020, 0.360, "$(G,q_1)$", fs=6.6, style="italic")
    label(ax, x0 + 0.112, 0.360, "$target_A$", fs=6.5, style="italic")
    label(ax, x0 + 0.020, 0.315, "$(G,q_2)$", fs=6.6, style="italic")
    label(ax, x0 + 0.112, 0.315, "$target_B$", fs=6.5, style="italic")
    arrow(ax, (x0 + 0.074, 0.360), (x0 + 0.104, 0.360), color="#111111", lw=0.9, ms=7)
    arrow(ax, (x0 + 0.074, 0.315), (x0 + 0.104, 0.315), color="#111111", lw=0.9, ms=7)
    box(ax, x0 + 0.145, 0.335, 0.052, 0.040, "$\\pi(G,q)$", fc=C["blue"], ec=C["blue_dark"], lw=1.0, r=0.005, fs=7.0)
    box(ax, x0 + 0.145, 0.290, 0.052, 0.040, "$\\pi(G,q)$", fc=C["blue"], ec=C["blue_dark"], lw=1.0, r=0.005, fs=7.0)
    box(ax, x0 + 0.235, 0.285, 0.075, 0.095, "Evaluation\nmetrics\nBoth Correct\nSwitch Rate\nAccuracy", fc=C["teal"], ec="#111111", lw=1.0, r=0.010, fs=5.8, weight="bold")
    arrow(ax, (x0 + 0.197, 0.332), (x0 + 0.235, 0.332), color=C["blue"], lw=1.3, ms=9)

    # Bottom control baselines, kept from the PPT but visually quieter.
    box(ax, x0 + 0.040, 0.165, 0.115, 0.105, fc="white", ec="#111111", lw=1.0, r=0.012)
    for i, txt in enumerate(["Geometry-only", "No-Query", "Query-Swap"]):
        box(ax, x0 + 0.058, 0.235 - i * 0.034, 0.079, 0.020, txt, fc="#AED0CC", ec="#111111", lw=0.8, r=0.004, fs=5.5)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    road = load_road_image()
    fig, ax = plt.subplots(figsize=(13.60, 6.65), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("auto")
    ax.axis("off")

    draw_panel_a(ax, road)
    draw_panel_b(ax)
    draw_panel_c(ax)

    arrow(ax, (0.305, 0.230), (0.335, 0.230), color=C["blue"], lw=2.1, ms=13)
    arrow(ax, (0.310, 0.720), (0.335, 0.670), color=C["blue"], lw=2.1, ms=13)
    arrow(ax, (0.625, 0.560), (0.665, 0.560), color=C["blue"], lw=2.1, ms=13)
    arrow(ax, (0.620, 0.230), (0.665, 0.205), color="#111111", lw=1.2, ms=9, ls=(0, (3, 3)))
    arrow(ax, (0.920, 0.285), (0.920, 0.155), color="#111111", lw=1.2, ms=9, ls=(0, (3, 3)))
    arrow(ax, (0.920, 0.155), (0.780, 0.155), color="#111111", lw=1.2, ms=9, ls=(0, (3, 3)))

    box(ax, 0.028, 0.035, 0.944, 0.075, fc=C["panel"], ec="none", r=0.016, z=0)
    label(ax, 0.500, 0.072, "Protocol Graph & Counterfactual Validation", fs=10.5, weight="bold", ha="center")

    for ext in ["png", "pdf", "svg"]:
        fig.savefig(OUT / f"figarc.{ext}", bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)


if __name__ == "__main__":
    main()
