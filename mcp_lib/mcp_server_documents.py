import os
import re
import sys
import json
import hashlib
import faiss
import numpy as np
import requests
import trafilatura
import pymupdf4llm
import yaml
from pathlib import Path
from markitdown import MarkItDown
from mcp.server.fastmcp import FastMCP

from models import UrlInput, FilePathInput, MarkdownOutput, SearchDocumentsInput

mcp = FastMCP("documents")

ROOT = Path(__file__).parent.resolve()

# Load parameters from default_mcp_config.yaml
_cfg_path = ROOT / "default_mcp_config.yaml"
with _cfg_path.open("r", encoding="utf-8") as _f:
    _cfg = yaml.safe_load(_f).get("documents", {})

EMBED_URL     = _cfg.get("embed_url",        "http://localhost:11434/api/embeddings")
EMBED_MODEL   = _cfg.get("embed_model",       "nomic-embed-text")
TOP_K         = int(_cfg.get("top_k",         5))
CHUNK_SIZE    = int(_cfg.get("chunk_size",    256))
CHUNK_OVERLAP = int(_cfg.get("chunk_overlap", 40))
DOC_DIR       = ROOT / _cfg.get("documents_dir",  "documents")
INDEX_DIR     = ROOT / _cfg.get("faiss_index_dir", "faiss_index")


# --- Embedding & FAISS helpers ---

def get_embedding(text: str) -> np.ndarray:
    result = requests.post(EMBED_URL, json={"model": EMBED_MODEL, "prompt": text})
    result.raise_for_status()
    return np.array(result.json()["embedding"], dtype=np.float32)

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    words = text.split()
    for i in range(0, len(words), size - overlap):
        yield " ".join(words[i:i + size])

def ensure_faiss_ready():
    if not (INDEX_DIR / "index.bin").exists():
        process_documents()

def process_documents():
    """Index all documents under mcp_lib/documents/ into a FAISS store."""
    INDEX_DIR.mkdir(exist_ok=True)

    index_file = INDEX_DIR / "index.bin"
    meta_file  = INDEX_DIR / "metadata.json"
    cache_file = INDEX_DIR / "doc_index_cache.json"

    cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    metadata = json.loads(meta_file.read_text()) if meta_file.exists() else []
    index = faiss.read_index(str(index_file)) if index_file.exists() else None

    for file in DOC_DIR.glob("*.*"):
        fhash = hashlib.md5(file.read_bytes()).hexdigest()
        if cache.get(file.name) == fhash:
            continue

        ext = file.suffix.lower()
        if ext == ".pdf":
            text = pymupdf4llm.to_markdown(str(file))
        elif ext in {".html", ".htm"}:
            downloaded = trafilatura.fetch_url(file.read_text().strip())
            text = trafilatura.extract(downloaded) or ""
        else:
            text = MarkItDown().convert(str(file)).text_content

        for chunk in chunk_text(text):
            vec = get_embedding(chunk).reshape(1, -1)
            if index is None:
                index = faiss.IndexFlatL2(vec.shape[1])
            index.add(vec)
            metadata.append({"doc": file.name, "chunk": chunk, "chunk_id": len(metadata)})

        cache[file.name] = fhash

    if index is not None:
        faiss.write_index(index, str(index_file))
        meta_file.write_text(json.dumps(metadata))
        cache_file.write_text(json.dumps(cache))


# --- Tools ---

@mcp.tool()
def search_stored_documents_rag(input: SearchDocumentsInput) -> list[str]:
    """Search indexed local documents and return relevant text chunks."""
    ensure_faiss_ready()
    try:
        index = faiss.read_index(str(INDEX_DIR / "index.bin"))
        metadata = json.loads((INDEX_DIR / "metadata.json").read_text())
        query_vec = get_embedding(input.query).reshape(1, -1)
        _, indices = index.search(query_vec, k=TOP_K)
        return [
            f"{metadata[i]['chunk']}\n[Source: {metadata[i]['doc']}, ID: {metadata[i]['chunk_id']}]"
            for i in indices[0]
        ]
    except Exception as e:
        return [f"ERROR: {e}"]


@mcp.tool()
def convert_webpage_url_into_markdown(input: UrlInput) -> MarkdownOutput:
    """Fetch a webpage and return its clean text content as markdown."""
    downloaded = trafilatura.fetch_url(input.url)
    if not downloaded:
        return MarkdownOutput(markdown="Failed to download the webpage.")
    markdown = trafilatura.extract(
        downloaded,
        include_comments=False,
        include_tables=True,
        output_format="markdown",
    ) or ""
    return MarkdownOutput(markdown=markdown)


@mcp.tool()
def extract_pdf(input: FilePathInput) -> MarkdownOutput:
    """Convert a PDF file to markdown text."""
    if not os.path.exists(input.file_path):
        return MarkdownOutput(markdown=f"File not found: {input.file_path}")
    markdown = pymupdf4llm.to_markdown(input.file_path)
    return MarkdownOutput(markdown=markdown)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "dev":
        mcp.run()
    else:
        mcp.run(transport="stdio")
