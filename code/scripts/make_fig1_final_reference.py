from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
from xml.etree import ElementTree as ET

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "paper" / "source" / "figures"
SOURCE_PPTX = ROOT / "paper_fig1.pptx"
FINAL_PPTX = ROOT / "paper_fig1_final_match_reference.pptx"

P_NS = "http://schemas.openxmlformats.org/presentationml/2006/main"
A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
ET.register_namespace("p", P_NS)
ET.register_namespace("a", A_NS)
ET.register_namespace("r", R_NS)

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)

COL = {
    "panel": "#F3F3F3",
    "edge": "#7B91A6",
    "blue": "#5F89D8",
    "blue_dark": "#1D5FB5",
    "orange": "#F59A3A",
    "red": "#E7746B",
    "teal": "#A8D0DB",
    "control": "#DDF0EC",
    "green": "#C7DDB4",
    "hard": "#F2C08E",
    "ultra": "#DDA29C",
    "ink": "#050505",
}
FS_SCALE = 1.18


def load_road_image():
    with ZipFile(SOURCE_PPTX) as zf:
        return Image.open(BytesIO(zf.read("ppt/media/image1.png"))).convert("RGBA")


def rounded_box(
    ax,
    x,
    y,
    w,
    h,
    text="",
    fc="white",
    ec=None,
    lw=0.9,
    r=0.006,
    fs=8.5,
    weight="normal",
    ha="center",
    va="center",
    color=None,
    ls="-",
    z=2,
):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.004,rounding_size={r}",
        facecolor=fc,
        edgecolor=ec or COL["edge"],
        linewidth=lw,
        linestyle=ls,
        zorder=z,
    )
    ax.add_patch(patch)
    if text:
        tx = x + w / 2 if ha == "center" else x + 0.010
        ax.text(
            tx,
            y + h / 2,
            text,
            fontsize=fs * FS_SCALE,
            fontweight=weight,
            ha=ha,
            va=va,
            color=color or COL["ink"],
            linespacing=1.08,
            zorder=z + 1,
        )
    return patch


def text(ax, x, y, s, fs=9, weight="normal", ha="left", va="center", style="normal", color=None, z=5):
    ax.text(
        x,
        y,
        s,
        fontsize=fs * FS_SCALE,
        fontweight=weight,
        ha=ha,
        va=va,
        fontstyle=style,
        color=color or COL["ink"],
        linespacing=1.08,
        zorder=z,
    )


def arrow(ax, start, end, color=None, lw=1.1, ms=10, ls="-", z=6, rad=0.0):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=ms,
            linewidth=lw,
            linestyle=ls,
            color=color or COL["blue"],
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=1,
            shrinkB=1,
            zorder=z,
        )
    )


def graph_icon(ax, cx, cy, scale=1.0):
    pts = [(-0.030, 0.018), (0.000, 0.034), (0.032, 0.010), (0.004, -0.025), (-0.034, -0.014)]
    pts = [(cx + px * scale, cy + py * scale) for px, py in pts]
    for i, j in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (1, 3)]:
        ax.plot([pts[i][0], pts[j][0]], [pts[i][1], pts[j][1]], color="#6EA6C6", lw=1.0, zorder=5)
    for idx, (px, py) in enumerate(pts):
        ax.add_patch(
            Circle(
                (px, py),
                0.0075 * scale,
                facecolor="#8BBB78" if idx in (2, 4) else "#6CA7C6",
                edgecolor="white",
                linewidth=0.4,
                zorder=6,
            )
        )


def draw_road_panel(ax, road, x, y, w, h):
    rx, ry, rw, rh = x + 0.022, y + h - 0.205, w - 0.044, 0.150
    ax.imshow(road, extent=(rx, rx + rw, ry, ry + rh), aspect="auto", zorder=3)
    ax.add_patch(Rectangle((rx + 0.690 * rw, ry + 0.040 * rh), 0.285 * rw, 0.600 * rh, fc="#EFF3E8", ec="none", zorder=4))
    text(ax, rx + 0.830 * rw, ry + 0.325 * rh, "candidate\nset $C$ (size $K$)", fs=7.8, weight="bold", ha="center", z=5)
    for sy, ey in [(0.580, 0.780), (0.520, 0.700), (0.455, 0.615), (0.390, 0.535)]:
        arrow(
            ax,
            (rx + 0.835 * rw, ry + sy * rh),
            (rx + 0.695 * rw, ry + ey * rh),
            color="#2089C5",
            lw=0.9,
            ms=6,
            ls=(0, (3, 2)),
            z=7,
        )


def draw_left(ax, road):
    x, y, w, h = 0.006, 0.205, 0.310, 0.785
    rounded_box(ax, x, y, w, h, fc=COL["panel"], ec=COL["edge"], lw=0.7, r=0.014, z=0)
    text(ax, x + w / 2, y + h - 0.035, "Protocol-aligned multi-candidate scenes\n(CARLA)", fs=12.2, weight="bold", ha="center")
    draw_road_panel(ax, road, x, y, w, h)

    rounded_box(
        ax,
        x + 0.018,
        0.671,
        w - 0.036,
        0.056,
        "Protocol construction: fixed node ordering / IDs\n+ candidate mask + target index",
        fs=7.4,
        ec=COL["edge"],
        r=0.005,
    )
    rounded_box(
        ax,
        x + 0.018,
        0.570,
        w - 0.036,
        0.080,
        "Experimental scope (closed-loop):\n"
        "Main: $K=2$   |   Scale-300: $K\\in\\{2,3,5\\}$\n"
        "Hard-confusable: $K\\in\\{3,5,8\\}$",
        fs=7.1,
        ec=COL["edge"],
        lw=0.7,
        r=0.005,
        ls=(0, (4, 3)),
    )

    rounded_box(ax, x + 0.018, 0.286, w - 0.036, 0.255, fc="white", ec="#243C52", lw=1.0, r=0.006)
    text(ax, x + w / 2, 0.516, "Protocol Graph Builder /\nFeature Encoder", fs=11.0, weight="bold", ha="center")
    ax.plot([x + w / 2, x + w / 2], [0.291, 0.475], color="#243C52", lw=0.9, zorder=5)
    text(ax, x + 0.050, 0.456, "Node features (32D):", fs=7.8, weight="bold")
    text(ax, x + w / 2 + 0.020, 0.456, "Edge features (8D):", fs=7.8, weight="bold")
    left_items = [
        "relative pose\n($\\Delta x,\\Delta y,\\Delta yaw$)",
        "wrt ego/anchor\ndistance $d$",
        "velocity v,yaw",
        "flags:is_anchor,\nis_candidate",
    ]
    right_items = ["relative type\n(near/front/...)", "normalized\ndistance,", "relative angle"]
    for idx, item in enumerate(left_items):
        yy = 0.426 - idx * 0.040
        ax.add_patch(Rectangle((x + 0.032, yy - 0.006), 0.007, 0.010, fc="#4E79D0", ec="#4E79D0", zorder=5))
        text(ax, x + 0.047, yy, item, fs=7.0, weight="bold")
    for idx, item in enumerate(right_items):
        yy = 0.426 - idx * 0.060
        ax.add_patch(Rectangle((x + w / 2 + 0.018, yy - 0.006), 0.007, 0.010, fc="#4E79D0", ec="#4E79D0", zorder=5))
        text(ax, x + w / 2 + 0.033, yy, item, fs=7.0, weight="bold")

    arrow(ax, (x + w / 2, 0.286), (x + w / 2, 0.250), lw=1.8, ms=13)
    rounded_box(
        ax,
        x + 0.018,
        0.220,
        w - 0.036,
        0.052,
        "Frozen protocol graphs + candidate_mask + target index",
        fc=COL["orange"],
        ec=COL["orange"],
        fs=7.7,
        r=0.004,
    )


def draw_middle(ax):
    x, y, w, h = 0.322, 0.205, 0.350, 0.785
    rounded_box(ax, x, y, w, h, fc=COL["panel"], ec=COL["edge"], lw=0.7, r=0.014, z=0)

    rounded_box(ax, x + 0.045, 0.882, w - 0.080, 0.085, fc="white", ec="black", lw=1.0, r=0.009)
    graph_icon(ax, x + 0.090, 0.925, scale=1.20)
    text(ax, x + 0.130, 0.925, "Input protocol graphs (candidates)\n+ query embedding $q$", fs=10.1, weight="bold")
    arrow(ax, (x + w / 2, 0.882), (x + w / 2, 0.650), color="black", lw=1.0, ms=11)

    rounded_box(ax, x + 0.018, 0.668, 0.132, 0.160, fc="white", ec="#29465E", lw=1.0, r=0.017)
    text(ax, x + 0.044, 0.804, "GNN Backbone", fs=10.5, weight="bold")
    rounded_box(
        ax,
        x + 0.047,
        0.695,
        0.093,
        0.088,
        "FiLM-conditioned\nshared graph layers\n(message passing)",
        fc=COL["blue"],
        ec="black",
        lw=0.9,
        r=0.010,
        fs=7.3,
        color="white",
    )
    arrow(ax, (x + 0.150, 0.748), (x + 0.202, 0.748), lw=2.6, ms=15)
    text(ax, x + 0.215, 0.765, "Shared graph encoder", fs=10.2, weight="bold")
    text(ax, x + 0.215, 0.715, "relation labels\n+ normalized geometry", fs=8.1)
    arrow(ax, (x + 0.083, 0.668), (x + 0.083, 0.555), color="black", lw=1.0, ms=10)

    rounded_box(
        ax,
        x + 0.168,
        0.575,
        0.136,
        0.104,
        "Scoring head:\n$\\pi(G,q)$\n(FiLM conditioning)",
        fc=COL["blue"],
        ec=COL["blue"],
        fs=9.6,
        color="white",
        weight="bold",
        r=0.009,
    )
    arrow(ax, (x + w / 2, 0.575), (x + w / 2, 0.420), color="black", lw=1.0, ms=10)
    rounded_box(ax, x + 0.020, 0.448, 0.116, 0.093, "Ranking Loss\nPairwise Hinge, $m=1.0$", fc=COL["red"], ec="#A54940", fs=8.5, r=0.009)
    rounded_box(ax, x + 0.242, 0.407, 0.105, 0.112, "Flip Loss\nQuery-Flip\nconstraint,\nmargin m_flip", fc=COL["teal"], ec="black", fs=7.5, weight="bold", r=0.009)
    arrow(ax, (x + 0.136, 0.494), (x + 0.242, 0.464), color="black", lw=0.9, ms=9)
    arrow(ax, (x + 0.248, 0.575), (x + 0.286, 0.519), color="black", lw=0.9, ms=9)

    rounded_box(ax, x + 0.035, 0.235, 0.285, 0.126, fc="#E6F4F6", ec="black", lw=0.9, r=0.010)
    text(ax, x + 0.177, 0.337, "3-Stage Fine-tuning Flow", fs=9.6, weight="bold", ha="center")
    stages = [
        ("Stage 1:\nquery projection\n+ FiLM", x + 0.055, 0.080),
        ("Stage 2:\nscoring head", x + 0.162, 0.075),
        ("Stage 3:\nfull model", x + 0.255, 0.075),
    ]
    for idx, (label, sx, sw) in enumerate(stages):
        rounded_box(ax, sx, 0.252, sw, 0.070, label, fc="#EEF7FA", ec="black", fs=7.2, r=0.007)
        if idx < len(stages) - 1:
            arrow(ax, (sx + sw, 0.286), (stages[idx + 1][1], 0.286), color="black", lw=0.9, ms=8)


def draw_right(ax):
    x, y, w, h = 0.680, 0.205, 0.314, 0.785
    rounded_box(ax, x, y, w, h, fc=COL["panel"], ec=COL["edge"], lw=0.7, r=0.014, z=0)
    text(ax, x + w / 2, 0.958, "Online Evaluation & Counterfactual Validation", fs=12.0, weight="bold", ha="center")

    rounded_box(ax, x + 0.015, 0.778, 0.150, 0.126, "Evaluation protocol\n\n30 scenes $\\times$ 10 seeds\nsame protocol graph\nshared controller", fs=8.3, ec=COL["blue_dark"], r=0.007)
    rounded_box(ax, x + 0.185, 0.778, 0.125, 0.126, "Target selector\nOurs (GS),\nMLP+pooling,\nRandom,\nOracle", fs=8.3, ec=COL["blue_dark"], r=0.007)
    rounded_box(
        ax,
        x + 0.015,
        0.708,
        w - 0.030,
        0.055,
        "Evaluation metrics\nProtoAcc@1  $\\bullet$  SemNavSucc@8m / @4m  $\\bullet$  scene-clustered intervals",
        fs=6.55,
        ec=COL["blue_dark"],
        r=0.005,
    )
    ax.plot([x, x + w], [0.687, 0.687], color="black", lw=0.9, ls=(0, (4, 4)), zorder=4)
    text(ax, x + w / 2, 0.656, "Counterfactual pipeline", fs=11.7, weight="bold", ha="center")
    rounded_box(ax, x + 0.015, 0.607, w - 0.030, 0.048, "Counterfactual Pair Generator: same $G$, different $q_1\\neq q_2$", fs=7.8, ec=COL["blue_dark"], r=0.005)

    text(ax, x + 0.070, 0.505, "Geometry\nMatching Gate", fs=8.5, ha="center")
    arrow(ax, (x + 0.132, 0.512), (x + 0.198, 0.512), color="black", lw=0.9, ms=8)
    rounded_box(ax, x + 0.202, 0.418, 0.104, 0.137, fc="#E9F2E5", ec=COL["blue_dark"], lw=0.9, r=0.009)
    gates = [
        (r"Normal:   $\Delta d<2.0m, \Delta\theta<10^\circ$", COL["green"]),
        (r"Hard:      $\Delta d<1.5m, \Delta\theta<8^\circ$", COL["hard"]),
        (r"Ultra:     $\Delta d<1.0m, \Delta\theta<5^\circ$", COL["ultra"]),
    ]
    for idx, (label, fill) in enumerate(gates):
        rounded_box(ax, x + 0.209, 0.518 - idx * 0.043, 0.090, 0.034, label, fc=fill, ec="black", fs=5.7, r=0.003)

    text(ax, x + 0.018, 0.365, r"$(G,q_1)$", fs=8.0, style="italic")
    text(ax, x + 0.072, 0.365, r"$target_A$", fs=8.0, style="italic")
    text(ax, x + 0.018, 0.311, r"$(G,q_2)$", fs=8.0, style="italic")
    text(ax, x + 0.072, 0.311, r"$target_B$", fs=8.0, style="italic")
    arrow(ax, (x + 0.110, 0.365), (x + 0.155, 0.365), color="black", lw=0.9, ms=8)
    arrow(ax, (x + 0.110, 0.311), (x + 0.155, 0.311), color="black", lw=0.9, ms=8)
    rounded_box(ax, x + 0.158, 0.343, 0.050, 0.047, r"$\pi(G,q)$", fc=COL["blue"], ec=COL["blue_dark"], fs=8.7, r=0.005)
    rounded_box(ax, x + 0.158, 0.289, 0.050, 0.047, r"$\pi(G,q)$", fc=COL["blue"], ec=COL["blue_dark"], fs=8.7, r=0.005)
    ax.plot([x + 0.208, x + 0.224, x + 0.224, x + 0.242], [0.366, 0.366, 0.339, 0.339], color="black", lw=0.9, zorder=5)
    ax.plot([x + 0.208, x + 0.224, x + 0.224, x + 0.242], [0.312, 0.312, 0.339, 0.339], color="black", lw=0.9, zorder=5)
    arrow(ax, (x + 0.230, 0.339), (x + 0.247, 0.339), lw=1.6, ms=11)
    rounded_box(ax, x + 0.248, 0.283, 0.062, 0.138, "Evaluation metrics\n\n$\\bullet$Task accuracy\n$\\bullet$Both-correct rate\n$\\bullet$Switch rate", fc=COL["teal"], ec="black", fs=6.6, weight="bold", r=0.009)


def draw_bottom(ax):
    rounded_box(ax, 0.006, 0.012, 0.988, 0.180, fc=COL["panel"], ec=COL["edge"], lw=0.7, r=0.014, z=0)
    text(ax, 0.500, 0.160, "Protocol Graph & Query-Flip Controls", fs=12.7, weight="bold", ha="center")
    xs = [0.025, 0.195, 0.365, 0.540]
    ws = [0.155, 0.155, 0.165, 0.170]
    labels = [
        "Geometry-only:\nremove query dependence;\ntest geometry shortcuts",
        "No-Query:\nremove query signal $q$;\nexpect chance-level behavior",
        "Query-Swap:\nswap $q_1 \\leftrightarrow q_2$ on the same graph;\ntest target switching",
        "Candidate-order permutation:\npermute candidate ordering / IDs;\ntest index shortcuts",
    ]
    for bx, bw, label in zip(xs, ws, labels):
        rounded_box(ax, bx, 0.045, bw, 0.088, label, fc=COL["control"], ec="#2A86A2", fs=6.9, r=0.005)
    rounded_box(ax, 0.735, 0.045, 0.105, 0.140, fc="white", ec="black", lw=1.0, r=0.010)
    for idx, label in enumerate(["Geometry-only", "No-Query", "Query-Swap", "Cand-Permute"]):
        rounded_box(ax, 0.750, 0.150 - idx * 0.035, 0.075, 0.025, label, fc="#B9DAD6", ec="black", fs=6.6, r=0.003)
    rounded_box(
        ax,
        0.865,
        0.055,
        0.130,
        0.080,
        "Same-family Query-Flip\nshown; Split-C stress\nreported as a limitation.",
        fc="white",
        ec=COL["blue_dark"],
        lw=0.9,
        r=0.007,
        ls=(0, (4, 3)),
        fs=6.7,
    )


def save_fig():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    road = load_road_image()
    fig = plt.figure(figsize=(16, 9), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    draw_left(ax, road)
    draw_middle(ax)
    draw_right(ax)
    draw_bottom(ax)

    arrow(ax, (0.285, 0.795), (0.322, 0.795), lw=3.3, ms=17)
    arrow(ax, (0.285, 0.247), (0.322, 0.247), lw=2.7, ms=15)
    arrow(ax, (0.622, 0.600), (0.680, 0.600), lw=3.4, ms=18)
    arrow(ax, (0.735, 0.205), (0.735, 0.105), color="black", lw=0.9, ms=8, ls=(0, (4, 3)))
    arrow(ax, (0.895, 0.283), (0.895, 0.120), color="black", lw=0.9, ms=8, ls=(0, (4, 3)))
    arrow(ax, (0.895, 0.120), (0.840, 0.120), color="black", lw=0.9, ms=8, ls=(0, (4, 3)))

    for ext in ["png", "pdf", "svg"]:
        fig.savefig(FIG_DIR / f"figarc.{ext}", dpi=300, bbox_inches=None, pad_inches=0)
    fig.savefig(ROOT / "paper_fig1_final_match_reference.png", dpi=300, bbox_inches=None, pad_inches=0)
    plt.close(fig)


def create_pptx_with_full_slide_image():
    png = (FIG_DIR / "figarc.png").read_bytes()
    with ZipFile(SOURCE_PPTX, "r") as zin:
        entries = {info.filename: zin.read(info.filename) for info in zin.infolist()}

    slide_xml = ET.fromstring(entries["ppt/slides/slide1.xml"])
    rels_path = "ppt/slides/_rels/slide1.xml.rels"
    rels = ET.fromstring(entries[rels_path])
    image_name = "ppt/media/fig1_final_match_reference.png"
    rel_id = "rId999"
    entries[image_name] = png

    rel = ET.SubElement(rels, f"{{{REL_NS}}}Relationship")
    rel.set("Id", rel_id)
    rel.set("Type", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image")
    rel.set("Target", "../media/fig1_final_match_reference.png")

    sp_tree = slide_xml.find(f".//{{{P_NS}}}spTree")
    pic = ET.Element(f"{{{P_NS}}}pic")
    nv = ET.SubElement(pic, f"{{{P_NS}}}nvPicPr")
    c_nv_pr = ET.SubElement(nv, f"{{{P_NS}}}cNvPr")
    c_nv_pr.set("id", "9999")
    c_nv_pr.set("name", "Fig1 final reference")
    ET.SubElement(nv, f"{{{P_NS}}}cNvPicPr")
    ET.SubElement(nv, f"{{{P_NS}}}nvPr")
    blip_fill = ET.SubElement(pic, f"{{{P_NS}}}blipFill")
    blip = ET.SubElement(blip_fill, f"{{{A_NS}}}blip")
    blip.set(f"{{{R_NS}}}embed", rel_id)
    stretch = ET.SubElement(blip_fill, f"{{{A_NS}}}stretch")
    ET.SubElement(stretch, f"{{{A_NS}}}fillRect")
    sp_pr = ET.SubElement(pic, f"{{{P_NS}}}spPr")
    xfrm = ET.SubElement(sp_pr, f"{{{A_NS}}}xfrm")
    ET.SubElement(xfrm, f"{{{A_NS}}}off", {"x": "0", "y": "0"})
    ET.SubElement(xfrm, f"{{{A_NS}}}ext", {"cx": "12192000", "cy": "6858000"})
    geom = ET.SubElement(sp_pr, f"{{{A_NS}}}prstGeom", {"prst": "rect"})
    ET.SubElement(geom, f"{{{A_NS}}}avLst")
    sp_tree.append(pic)

    entries["ppt/slides/slide1.xml"] = ET.tostring(slide_xml, encoding="utf-8", xml_declaration=True)
    entries[rels_path] = ET.tostring(rels, encoding="utf-8", xml_declaration=True)
    with ZipFile(FINAL_PPTX, "w", ZIP_DEFLATED) as zout:
        for name, data in entries.items():
            zout.writestr(name, data)

    (FIG_DIR / FINAL_PPTX.name).write_bytes(FINAL_PPTX.read_bytes())


if __name__ == "__main__":
    save_fig()
    create_pptx_with_full_slide_image()
