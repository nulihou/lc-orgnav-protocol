from copy import deepcopy
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import shutil
from xml.etree import ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
PPTX = ROOT / "paper_fig1.pptx"
OUT_PPTX = ROOT / "paper_fig1_latest_protocol.pptx"
BACKUP = ROOT / "backups" / "paper_fig1_before_latest_protocol_edit_20260526.pptx"

NS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
}

ET.register_namespace("a", NS["a"])
ET.register_namespace("p", NS["p"])
ET.register_namespace("r", "http://schemas.openxmlformats.org/officeDocument/2006/relationships")


def sp_text(sp):
    return "".join(t.text or "" for t in sp.findall(".//a:t", NS))


def set_text(sp, text):
    tx_body = sp.find("p:txBody", NS)
    if tx_body is None:
        return
    body_pr = tx_body.find("a:bodyPr", NS)
    lst_style = tx_body.find("a:lstStyle", NS)

    # Keep the first paragraph/rPr style as a template where possible.
    first_p = tx_body.find("a:p", NS)
    p_pr_template = first_p.find("a:pPr", NS) if first_p is not None else None
    r_pr_template = first_p.find(".//a:rPr", NS) if first_p is not None else None

    for child in list(tx_body):
        if child.tag not in {f"{{{NS['a']}}}bodyPr", f"{{{NS['a']}}}lstStyle"}:
            tx_body.remove(child)

    for line in text.split("\n"):
        p = ET.SubElement(tx_body, f"{{{NS['a']}}}p")
        if p_pr_template is not None:
            p.append(deepcopy(p_pr_template))
        r = ET.SubElement(p, f"{{{NS['a']}}}r")
        if r_pr_template is not None:
            r.append(deepcopy(r_pr_template))
        t = ET.SubElement(r, f"{{{NS['a']}}}t")
        t.text = line
        ET.SubElement(p, f"{{{NS['a']}}}endParaRPr")


def set_xfrm(sp, x=None, y=None, cx=None, cy=None):
    xfrm = sp.find(".//a:xfrm", NS)
    if xfrm is None:
        return
    off = xfrm.find("a:off", NS)
    ext = xfrm.find("a:ext", NS)
    if off is not None:
        if x is not None:
            off.set("x", str(x))
        if y is not None:
            off.set("y", str(y))
    if ext is not None:
        if cx is not None:
            ext.set("cx", str(cx))
        if cy is not None:
            ext.set("cy", str(cy))


def get_xfrm(sp):
    xfrm = sp.find(".//a:xfrm", NS)
    if xfrm is None:
        return None
    off = xfrm.find("a:off", NS)
    ext = xfrm.find("a:ext", NS)
    if off is None or ext is None:
        return None
    return tuple(int(off.attrib[k]) for k in ("x", "y")) + tuple(int(ext.attrib[k]) for k in ("cx", "cy"))


def find_by_contains(root, needle):
    for sp in root.findall(".//p:sp", NS):
        if needle in sp_text(sp):
            return sp
    raise KeyError(f"Shape containing {needle!r} not found")


def main():
    if not PPTX.exists():
        raise FileNotFoundError(PPTX)
    BACKUP.parent.mkdir(parents=True, exist_ok=True)
    if not BACKUP.exists():
        shutil.copy2(PPTX, BACKUP)

    with ZipFile(PPTX, "r") as zin:
        entries = {info.filename: zin.read(info.filename) for info in zin.infolist()}

    root = ET.fromstring(entries["ppt/slides/slide1.xml"])

    replacements = {
            "(a) Scenario Generation and Protocol Freezing": "Protocol Scene (Main Protocol)\n-- Two Candidates",
            "Constraints:△d<2.0m,△θ<10°": "Protocol-aligned scene construction\nfixed node ordering • candidate mask • target index",
            "Frozen scene graphs + candidate_mask": "Frozen scene graphs + candidate_mask",
            "Input protocolgraphs(candidates)": "Input protocol\ngraphs (candidates)\n+ query embedding q",
            "Checkpoint": "",
            "LCORG-Flip-FT(v1)best.pt": "Query-Flip\nfine-tuned selector",
            "Online  multi-candidate  scenarios(N=150=30 scenes×5 seeds)": "Evaluation protocol\n30 scenes × 10 seeds\nsame protocol graph\nshared controller",
            "Target   selector:Ours(GS),Random,Oracle": "Target selector\nGraph selector (Ours)\nMLP+pooling\nRandom\nOracle",
            "SemNavSucc@8m": "Evaluation metrics",
            "GS(Ours):46.0%(CI 38.2-54.0)Random:24.3%     (Cl 18.1-31.8)Oracle:54.7%        (Cl 46.7-62.4)": "ProtoAcc@1 • SemNavSucc@8m\nSemNavSucc@4m • scene-clustered CIs\nScale-300 reported separately",
            "Evaluationmetrics·BothCorrect·SwitchRate·Accuracy": "Evaluation\nmetrics\n·Task accuracy\n·Both-correct rate\n·Switch rate",
            "Protocol Graph & Counterfactual Validation": "Protocol Graph & Shortcut Controls",
            "Query-SwapD": "Candidate permutation",
            "Geometry-only": "No-Query",
    }

    # More robust replacement by normalized existing text.
    for sp in root.findall(".//p:sp", NS):
        normalized = sp_text(sp).replace(" ", "").replace("\n", "")
        for old, new in replacements.items():
            old_norm = old.replace(" ", "").replace("\n", "")
            if old_norm and normalized == old_norm:
                set_text(sp, new)

    # Fix the three bottom shortcut-control pills explicitly by vertical order.
    for sp in root.findall(".//p:sp", NS):
        p = get_xfrm(sp)
        if p is None:
            continue
        x, y, cx, cy = p
        if 13100 <= x <= 13400 and 7000 <= y <= 7700:
            if y < 7250:
                set_text(sp, "No-Query")
            elif y < 7500:
                set_text(sp, "Query-Swap")
            else:
                set_text(sp, "Candidate permutation")

    # Replace bottom explanatory boxes if present in the newer design.
    for sp in root.findall(".//p:sp", NS):
        txt = sp_text(sp)
        if "Geometry-only" in txt and "shortcut" in txt:
            set_text(sp, "No-Query:\nremove query signal q;\nexpect chance-level behavior")
        elif "Use zero query" in txt:
            set_text(sp, "Query-Swap:\nswap q1 ↔ q2 on the same graph;\ntest target switching")
        elif "Swap q1" in txt or "Candidate-order" in txt:
            set_text(sp, "Candidate-order permutation:\npermute candidate ordering / IDs;\ntest index shortcuts")

    # Make the old results box behave as a compact metric note.
    try:
        metrics_title = find_by_contains(root, "Evaluation metrics")
        set_xfrm(metrics_title, x=13120, y=2870, cx=3900, cy=250)
    except KeyError:
        pass
    try:
        metrics_body = find_by_contains(root, "ProtoAcc@1")
        set_xfrm(metrics_body, x=13120, y=3130, cx=3900, cy=520)
    except KeyError:
        pass

    entries["ppt/slides/slide1.xml"] = ET.tostring(root, encoding="utf-8", xml_declaration=True)

    tmp_out = OUT_PPTX.with_suffix(".tmp.pptx")
    with ZipFile(tmp_out, "w", ZIP_DEFLATED) as zout:
        for filename, data in entries.items():
            zout.writestr(filename, data)
    tmp_out.replace(OUT_PPTX)


if __name__ == "__main__":
    main()
