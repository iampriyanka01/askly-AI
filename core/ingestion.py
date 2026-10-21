"""
core/ingestion.py  -  STAGE 1: enterprise documents  ->  section-aware chunks
=============================================================================

Role in the pipeline
    Everything downstream (BM25, dense vectors, reranking, citations) works on
    the chunk `Document`s created here.

    data/knowledge_base/<category>/*.pdf|.md|.txt
        -> load_source_documents()   one Document per file, split into SECTIONS
        -> chunk_documents()         one Document per section (long sections are
                                     split further with overlap)

Why section-aware chunking?
    Helpdesk answers must cite *where* a fact came from ("HR Handbook, Section
    4.2, p. 4"). Splitting by the document's own headings keeps each chunk about
    one topic, gives every chunk a section number/title/page for its citation,
    and lets Askly suggest neighbouring sections as proactive next steps.

    Each chunk's text starts with a short context header -
        "HR Handbook > 4 Leave Policy > 4.2 Sick Leave"
    - so a chunk that never repeats the words "sick leave" in its body is still
    findable by them (contextual chunk headers help both BM25 and embeddings).

Supported formats
    .md / .markdown : optional YAML-ish front matter (title, category, url, owner,
                      last_updated) and #/##/### headings
    .pdf            : per-page text; numbered headings ("4.2 Sick Leave") are
                      detected; a "URL:" line in the document is picked up
    .txt            : treated like Markdown without front matter

Chunk metadata (travels into FAISS / Pinecone and back into citations)
    chunk_id, doc_id, title, category, section, section_number, section_title,
    parent_section, page, source (file name), path, url, chunk_index, content_hash

Public API
    load_source_documents(directory)      -> list[Document]  (whole files)
    chunk_documents(docs)                 -> list[Document]  (chunks)
    load_knowledge_base(directory)        -> list[Document]  (both steps)
    load_file(path, category=None)        -> list[Document]  (chunks for one upload)
    build_catalog(source_docs)            -> {doc_id: {...sections in order...}}
"""

import hashlib
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.config import CHUNK_SIZE, CHUNK_OVERLAP
from utils.utils import get_logger, load_pdf_pages, clean_text

logger = get_logger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".md", ".markdown", ".txt"}

# "4.2 Sick Leave" / "12 Code of Conduct" - a number, then a Capitalised title, no
# trailing full stop, short. Wrapped paragraph lines ("30 days of paid ...") start
# lowercase after the number, so they are not mistaken for headings.
_NUMBERED_HEADING = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){0,3})\s+([A-Z][A-Za-z0-9 ,&/()'\-?:]{2,75})$")
_PAGE_FOOTER = re.compile(r"^(page\s+\d+(\s+of\s+\d+)?|\d+)$", re.IGNORECASE)
_URL_LINE = re.compile(r"\bURL:\s*(\S+)", re.IGNORECASE)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _pretty_title(stem: str) -> str:
    return re.sub(r"\s+", " ", stem.replace("_", " ").replace("-", " ")).strip().title()


def _hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "section"


def _split_heading(heading: str) -> Tuple[Optional[str], str]:
    m = re.match(r"^(\d+(?:\.\d+)*)\.?\s+(.*)$", heading.strip())
    return (m.group(1), m.group(2).strip()) if m else (None, heading.strip())


def _parse_front_matter(text: str) -> Tuple[Dict[str, str], str]:
    """Minimal front-matter parser (key: value lines between --- fences)."""
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end == -1:
        return {}, text
    meta = {}
    for line in text[3:end].strip().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip().lower()] = v.strip().strip('"').strip("'")
    return meta, text[end + 4:].lstrip("\n")


def _new_section(heading: str, level: int, page: Optional[int], parent: Optional[str]) -> Dict[str, Any]:
    number, title = _split_heading(heading)
    return {"heading": heading.strip(), "number": number, "title": title, "level": level,
            "page": page, "parent": parent, "lines": []}


# ---------------------------------------------------------------------------
# format-specific section parsers
# ---------------------------------------------------------------------------

_MD_META = re.compile(r"^\s*(?:\*\*)?(document owner|policy owner|owner|last updated|distribution|compliance|"
                      r"version)(?:\*\*)?\s*:\s*(?:\*\*)?\s*(.+?)\s*$", re.IGNORECASE)
_FENCE = re.compile(r"^\s*(```|~~~)")


def _sections_from_markdown(body: str) -> Tuple[Optional[str], Dict[str, str], List[Dict[str, Any]]]:
    """Split Markdown into sections by #..###### headings.

    - lines inside ``` / ~~~ code fences are content, never headings (a YAML/bash
      block full of "# comments" must not explode into fake sections)
    - "**Document Owner**: X" / "Last Updated: Y" style lines become document metadata
    - a subtitle right under the title with no content of its own (e.g.
      "## Orion Digital Services, Inc.") is recorded as the organisation, not a section
    - horizontal rules (---) are dropped
    Returns (title, metadata, sections).
    """
    doc_title, sections, meta = None, [], {}
    current = _new_section("Overview", 2, None, None)
    parents: Dict[int, str] = {}
    in_fence = False
    for raw in body.splitlines():
        line = raw.rstrip()
        if _FENCE.match(line):
            in_fence = not in_fence
            current["lines"].append(line)
            continue
        if in_fence:
            current["lines"].append(line)
            continue
        m = re.match(r"^(#{1,6})\s+(.*?)\s*#*\s*$", line)
        if m:
            level, heading = len(m.group(1)), m.group(2).strip()
            if level == 1 and doc_title is None:
                doc_title = heading
                continue
            sections.append(current)
            level = max(level, 2)
            parents[level] = heading
            for deeper in [k for k in parents if k > level]:
                parents.pop(deeper)
            parent = parents.get(level - 1) if level > 2 else None
            current = _new_section(heading, level, None, parent)
            continue
        mm = _MD_META.match(line)
        if mm and len(sections) <= 2 and not meta.get(mm.group(1).lower()):
            key = {"document owner": "owner", "policy owner": "owner", "last updated": "last_updated"}.get(
                mm.group(1).lower(), mm.group(1).lower())
            meta[key] = mm.group(2).strip().strip("*").strip()
            continue
        if re.match(r"^\s*([-*_])\1{2,}\s*$", line):
            continue
        current["lines"].append(line)
    sections.append(current)
    kept = [s for s in sections if "\n".join(s["lines"]).strip()]
    # Organisation subtitle: first heading after the title with no text of its own.
    if sections and sections[0]["heading"] == "Overview" and not "\n".join(sections[0]["lines"]).strip() \
            and len(sections) > 1 and not "\n".join(sections[1]["lines"]).strip():
        meta.setdefault("organization", sections[1]["heading"])
    if doc_title and " — " in doc_title:          # "People & Policy Manual — Orion Digital Services, Inc."
        doc_title, org = doc_title.split(" — ", 1)
        meta.setdefault("organization", org.strip())
    return doc_title, meta, kept


def _sections_from_pdf(pages: List[str]) -> Tuple[Optional[str], Optional[str], List[Dict[str, Any]]]:
    """Split PDF page texts into sections using numbered headings; remember each
    section's starting page. Returns (title, url, sections)."""
    doc_title, url, sections = None, None, []
    current = _new_section("Overview", 2, 1, None)
    parents: Dict[int, str] = {}
    for page_no, page_text in enumerate(pages, start=1):
        for raw in page_text.splitlines():
            line = raw.strip()
            if not line or _PAGE_FOOTER.match(line):
                continue
            if url is None:
                m = _URL_LINE.search(line)
                if m:
                    url = m.group(1)
            if doc_title is None:
                doc_title = line
                continue
            m = _NUMBERED_HEADING.match(line)
            if m and not line.endswith("."):
                level = m.group(1).count(".") + 2  # "4" -> 2, "4.2" -> 3
                sections.append(current)
                parents[level] = line
                parent = parents.get(level - 1) if level > 2 else None
                current = _new_section(line, level, page_no, parent)
                continue
            if current["page"] is None:
                current["page"] = page_no
            current["lines"].append(line)
    sections.append(current)
    return doc_title, url, [s for s in sections if " ".join(s["lines"]).strip()]


# File-name keywords -> category, for knowledge bases dropped in as one flat folder.
# Checked in order: "it_support_guide" must hit "it" before "support" -> customer.
_FILENAME_CATEGORIES = [
    ("it", r"(^|[_\-\s])it([_\-\s]|$)|helpdesk|service.?desk"),
    ("security", r"security|password|infosec"),
    ("onboarding", r"onboarding|new.?hire|new.?starter"),
    ("hr", r"\bhr\b|(^|_)hr_|human|people|policy|handbook|leave|benefit|remote"),
    ("finance", r"financ|expense|travel|payroll|account"),
    ("customer", r"customer|client|faq"),
    ("operations", r"operation|facilit|office"),
    ("engineering", r"technical|engineering|architecture|api|developer|devops"),
]


def infer_category(filename: str) -> str:
    name = Path(filename).stem.lower()
    for cat, pattern in _FILENAME_CATEGORIES:
        if re.search(pattern, name):
            return cat
    return "general"


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def load_source_file(path: Path, category: Optional[str] = None, root: Optional[Path] = None) -> Optional[Document]:
    """Load ONE file into a Document whose metadata['sections'] lists its sections."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return None
    if category is None:
        category = path.parent.name.lower() if root and path.parent != root else infer_category(path.name)

    meta: Dict[str, Any] = {}
    try:
        if ext == ".pdf":
            pages = load_pdf_pages(str(path))
            if not any(p.strip() for p in pages):
                logger.warning(f"No extractable text in {path.name} (scanned PDF?) - skipped")
                return None
            title, url, sections = _sections_from_pdf(pages)
            meta.update({"url": url, "page_count": len(pages)})
        else:
            text = path.read_text(encoding="utf-8", errors="ignore")
            fm, body = _parse_front_matter(text)
            title, md_meta, sections = _sections_from_markdown(body)
            for k, v in md_meta.items():
                fm.setdefault(k, v)
            if not fm.get("url"):
                m = _URL_LINE.search(body)
                fm["url"] = m.group(1) if m else None
            meta.update(fm)
            title = fm.get("title") or title
    except Exception as e:
        logger.error(f"Failed to load {path}: {e}")
        return None

    for s in sections:
        s["text"] = clean_text("\n".join(s.pop("lines")))
    category = (meta.pop("category", None) or category or "general").lower()
    title = title or _pretty_title(path.stem)
    doc_id = f"{category}/{path.stem}"
    full_text = "\n\n".join(f"{s['heading']}\n{s['text']}" for s in sections)
    metadata = {
        "doc_id": doc_id, "title": title, "category": category,
        "source": path.name, "path": str(path), "url": meta.get("url"),
        "owner": meta.get("owner"), "last_updated": meta.get("last_updated"),
        "organization": meta.get("organization"),
        "page_count": meta.get("page_count"), "sections": sections,
    }
    return Document(page_content=full_text, metadata=metadata)


def load_source_documents(directory: str) -> List[Document]:
    """Load every supported file under `directory` (recursively). The first-level
    sub-folder name is the document's category (hr, it, security, ...)."""
    root = Path(directory)
    if not root.exists():
        logger.warning(f"Knowledge base folder not found: {root}")
        return []
    docs = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS and not path.name.startswith("."):
            rel = path.relative_to(root)
            category = rel.parts[0].lower() if len(rel.parts) > 1 else infer_category(path.name)
            doc = load_source_file(path, category=category, root=root)
            if doc:
                docs.append(doc)
    logger.info(f"Loaded {len(docs)} knowledge-base documents from {root}")
    return docs


def chunk_documents(docs: List[Document], chunk_size: int = CHUNK_SIZE,
                    chunk_overlap: int = CHUNK_OVERLAP) -> List[Document]:
    """One chunk per section; sections longer than `chunk_size` are split with
    overlap. Chunk ids are stable (doc + section + index), so re-indexing an
    unchanged file never re-embeds it."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                                              separators=["\n\n", "\n", ". ", " ", ""])
    chunks: List[Document] = []
    for doc in docs:
        md = doc.metadata
        seen_ids: Dict[str, int] = {}
        for order, section in enumerate(md.get("sections", [])):
            pieces = splitter.split_text(section["text"]) if len(section["text"]) > chunk_size else [section["text"]]
            path_parts = [md["title"]] + ([section["parent"]] if section.get("parent") else []) + [section["heading"]]
            header = " > ".join(path_parts)
            for i, piece in enumerate(pieces):
                base = f"{md['doc_id']}::{section['number'] or _slug(section['heading'])}"
                n = seen_ids.get(base, 0)
                seen_ids[base] = n + 1
                chunk_id = f"{base}::{n}"
                content = f"{header}\n{piece}"
                chunks.append(Document(page_content=content, metadata={
                    "chunk_id": chunk_id,
                    "doc_id": md["doc_id"],
                    "title": md["title"],
                    "category": md["category"],
                    "section": section["heading"],
                    "section_number": section["number"] or "",
                    "section_title": section["title"],
                    "parent_section": section.get("parent") or "",
                    "section_order": order,
                    "page": section.get("page") or 0,
                    "source": md["source"],
                    "path": md["path"],
                    "url": md.get("url") or "",
                    "chunk_index": i,
                    "content_hash": _hash(content),
                }))
    logger.info(f"Created {len(chunks)} chunks from {len(docs)} documents")
    return chunks


def load_knowledge_base(directory: str) -> Tuple[List[Document], List[Document]]:
    """Load + chunk the whole knowledge base. Returns (source_docs, chunks)."""
    docs = load_source_documents(directory)
    return docs, chunk_documents(docs)


def load_file(path: str, category: Optional[str] = None) -> Tuple[Optional[Document], List[Document]]:
    """Load + chunk a single (e.g. freshly uploaded) file."""
    doc = load_source_file(Path(path), category=category)
    return (doc, chunk_documents([doc])) if doc else (None, [])


def build_catalog(docs: List[Document]) -> Dict[str, Dict[str, Any]]:
    """{doc_id: {title, category, url, source, sections:[{heading, number, title, parent, page}]}}
    - the document structure used for proactive "related resources"."""
    catalog = {}
    for d in docs:
        md = d.metadata
        catalog[md["doc_id"]] = {
            "title": md["title"], "category": md["category"], "url": md.get("url"),
            "source": md["source"], "owner": md.get("owner"), "last_updated": md.get("last_updated"),
            "organization": md.get("organization"),
            "sections": [{k: s.get(k) for k in ("heading", "number", "title", "parent", "page")}
                         for s in md.get("sections", [])],
        }
    return catalog
