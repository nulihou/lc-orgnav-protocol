import re
import sys
import textwrap
from pathlib import Path

import fitz


def ascii_clean(text: str) -> str:
    repl = {
        "\u2013": "-",
        "\u2014": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u00d7": "x",
        "\u00b1": "+/-",
        "\u2264": "<=",
        "\u2265": ">=",
        "\u2192": "->",
        "\u03bb": "lambda",
    }
    for k, v in repl.items():
        text = text.replace(k, v)
    return text.encode("ascii", errors="ignore").decode("ascii")


def strip_inline_markdown(text: str) -> str:
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*", r"\1", text)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    return text


def render_markdown_to_pdf(src: Path, dst: Path) -> None:
    raw = ascii_clean(src.read_text(encoding="utf-8"))
    lines = raw.splitlines()

    doc = fitz.open()
    page_rect = fitz.paper_rect("letter")
    margin_l = 48
    margin_t = 48
    margin_b = 48
    max_width = page_rect.width - 2 * margin_l
    y = margin_t
    page = doc.new_page(width=page_rect.width, height=page_rect.height)

    def new_page():
        nonlocal page, y
        page = doc.new_page(width=page_rect.width, height=page_rect.height)
        y = margin_t

    def ensure(space: float):
        if y + space > page_rect.height - margin_b:
            new_page()

    def draw_line(text: str, size: float = 9.0, font: str = "Times-Roman", indent: float = 0.0):
        nonlocal y
        ensure(size * 1.45)
        page.insert_text((margin_l + indent, y), text, fontsize=size, fontname=font, color=(0, 0, 0))
        y += size * 1.35

    def draw_wrapped(text: str, size: float = 9.0, font: str = "Times-Roman", indent: float = 0.0):
        chars = max(45, int(max_width / (size * 0.48)) - int(indent / 4))
        for part in textwrap.wrap(text, width=chars, break_long_words=False, replace_whitespace=True) or [""]:
            draw_line(part, size=size, font=font, indent=indent)

    for line in lines:
        stripped = line.strip()
        if not stripped:
            y += 4
            continue
        if re.match(r"^\|?[-: ]+\|[-|: ]+$", stripped):
            continue
        if stripped.startswith("#"):
            level = len(stripped) - len(stripped.lstrip("#"))
            title = stripped[level:].strip()
            y += 4
            draw_wrapped(title, size=14 if level == 1 else 11, font="Times-Bold")
            y += 3
            continue
        if stripped.startswith("|") and stripped.endswith("|"):
            cells = [strip_inline_markdown(c.strip()) for c in stripped.strip("|").split("|")]
            if len(cells) >= 3 and cells[0].lower().startswith("reviewer concern"):
                draw_wrapped("Reviewer concern / Revision closure / Claim scope", size=9.0, font="Times-Bold")
            elif len(cells) >= 3:
                draw_wrapped(f"Concern: {cells[0]}", size=8.4, font="Times-Bold", indent=8)
                draw_wrapped(f"Revision: {cells[1]}", size=8.4, indent=18)
                draw_wrapped(f"Claim boundary: {cells[2]}", size=8.4, indent=18)
                y += 2
            else:
                draw_wrapped(" | ".join(cells), size=8.4, indent=8)
            continue
        bullet = re.match(r"^([-*]|\d+\.)\s+(.*)$", stripped)
        if bullet:
            marker, body = bullet.group(1), strip_inline_markdown(bullet.group(2))
            draw_wrapped(f"{marker} {body}", size=9.0, indent=12)
            continue
        draw_wrapped(strip_inline_markdown(stripped), size=9.0)

    dst.parent.mkdir(parents=True, exist_ok=True)
    doc.save(dst)
    doc.close()


if __name__ == "__main__":
    render_markdown_to_pdf(Path(sys.argv[1]), Path(sys.argv[2]))
