"""Read CV / JD files of various formats into plain text."""
from pathlib import Path


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def _read_docx(path: Path) -> str:
    import docx
    doc = docx.Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


_READERS = {
    ".pdf": _read_pdf,
    ".docx": _read_docx,
    ".txt": _read_text,
    ".md": _read_text,
}

SUPPORTED = set(_READERS)


def read_file(path: Path) -> str:
    """Return the text content of a single file, or '' if unsupported/empty."""
    reader = _READERS.get(path.suffix.lower())
    if reader is None:
        return ""
    try:
        return reader(path).strip()
    except Exception as e:  # noqa: BLE001
        return f"[Could not read {path.name}: {e}]"


def load_dir(directory: Path) -> list[dict]:
    """Load every supported file in a directory.

    Returns a list of {"name", "path", "text"} dicts, skipping empties.
    """
    directory = Path(directory)
    if not directory.exists():
        return []
    docs = []
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED:
            text = read_file(path)
            if text:
                docs.append({"name": path.name, "path": str(path), "text": text})
    return docs


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> list[str]:
    """Simple character-based chunker with overlap. Good enough for small corpora."""
    text = " ".join(text.split())  # normalize whitespace
    if len(text) <= chunk_size:
        return [text] if text else []
    chunks = []
    start = 0
    step = max(chunk_size - overlap, 1)
    while start < len(text):
        chunks.append(text[start : start + chunk_size])
        start += step
    return chunks
