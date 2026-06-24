"""Read CV / JD files of various formats into plain text."""
import io
from pathlib import Path


def _read_pdf_bytes(data: bytes) -> str:
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(data))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def _read_docx_bytes(data: bytes) -> str:
    import docx
    doc = docx.Document(io.BytesIO(data))
    return "\n".join(p.text for p in doc.paragraphs)


def _read_text_bytes(data: bytes) -> str:
    return data.decode("utf-8", errors="ignore")


_BYTE_READERS = {
    ".pdf": _read_pdf_bytes,
    ".docx": _read_docx_bytes,
    ".txt": _read_text_bytes,
    ".md": _read_text_bytes,
}

SUPPORTED = set(_BYTE_READERS)


def read_bytes(name: str, data: bytes) -> str:
    """Extract text from in-memory file bytes (for browser uploads / DB blobs).

    Dispatches by the extension in `name`. Returns '' if unsupported/empty.
    """
    reader = _BYTE_READERS.get(Path(name).suffix.lower())
    if reader is None:
        return ""
    try:
        return reader(data).strip()
    except Exception as e:  # noqa: BLE001
        return f"[Could not read {name}: {e}]"


def read_file(path: Path) -> str:
    """Return the text content of a single file on disk, or '' if unsupported."""
    path = Path(path)
    if path.suffix.lower() not in _BYTE_READERS:
        return ""
    try:
        return read_bytes(path.name, path.read_bytes())
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
