"""Build the sample PDF policies in data/sample_knowledge_base from scripts/pdf_sources/*.md.

Real enterprise knowledge bases are mostly PDFs, so two of the sample documents
are shipped as PDFs (the rest are Markdown). Each top-level section starts on a
new page, so citations like "HR Handbook, Section 4.2, p. 4" are meaningful.

    python scripts/make_sample_pdfs.py          (needs: pip install reportlab)
"""

import re
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak

ROOT = Path(__file__).resolve().parent.parent
SOURCES = ROOT / "scripts" / "pdf_sources"
TARGETS = {
    "HR_Handbook.md": ROOT / "data" / "sample_knowledge_base" / "hr" / "HR_Handbook.pdf",
    "Account_Security_Policy.md": ROOT / "data" / "sample_knowledge_base" / "security" / "Account_Security_Policy.pdf",
}


def build(src: Path, dst: Path) -> None:
    styles = getSampleStyleSheet()
    title = ParagraphStyle("t", parent=styles["Title"], fontSize=20, spaceAfter=10)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=15, spaceBefore=6, spaceAfter=8)
    h3 = ParagraphStyle("h3", parent=styles["Heading3"], fontSize=12, spaceBefore=8, spaceAfter=4)
    body = ParagraphStyle("b", parent=styles["BodyText"], fontSize=10.5, leading=15)
    meta = ParagraphStyle("m", parent=body, textColor="#555555", fontSize=9.5)

    story, first_h2 = [], True
    for line in src.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        esc = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        if line.startswith("# "):
            story.append(Paragraph(esc[2:], title))
        elif line.startswith("## "):
            if not first_h2:
                story.append(PageBreak())
            first_h2 = False
            story.append(Paragraph(esc[3:], h2))
        elif line.startswith("### "):
            story.append(Paragraph(esc[4:], h3))
        elif re.match(r"^(Document owner|URL):", line):
            story.append(Paragraph(esc, meta))
        else:
            story.append(Paragraph(esc, body))
            story.append(Spacer(1, 4))

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")
        canvas.restoreState()

    dst.parent.mkdir(parents=True, exist_ok=True)
    SimpleDocTemplate(str(dst), pagesize=A4, leftMargin=2.2 * cm, rightMargin=2.2 * cm,
                      topMargin=2 * cm, bottomMargin=2 * cm,
                      title=src.stem.replace("_", " ")).build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"wrote {dst.relative_to(ROOT)}")


if __name__ == "__main__":
    for name, target in TARGETS.items():
        build(SOURCES / name, target)
