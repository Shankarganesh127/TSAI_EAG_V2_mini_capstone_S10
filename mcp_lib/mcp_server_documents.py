import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
import pymupdf4llm
import requests
import trafilatura
from markitdown import MarkItDown
from mcp.server.fastmcp import FastMCP

try:
    from .models import UrlInput, FilePathInput, MarkdownOutput, SearchDocumentsInput
    from .server_utils import load_mcp_config, run_mcp_server
except ImportError:
    from models import UrlInput, FilePathInput, MarkdownOutput, SearchDocumentsInput
    from server_utils import load_mcp_config, run_mcp_server

mcp = FastMCP("documents")

ROOT = Path(__file__).parent.resolve()

# Load parameters from default_mcp_config.yaml
_cfg_path = ROOT / "default_mcp_config.yaml"
_cfg = load_mcp_config(_cfg_path).get("documents", {})

EMBED_URL     = _cfg.get("embed_url",        "http://localhost:11434/api/embeddings")
EMBED_MODEL   = _cfg.get("embed_model",       "nomic-embed-text")
TOP_K         = int(_cfg.get("top_k",         5))
MIN_SIMILARITY = float(_cfg.get("min_similarity", 0.35))
CHUNK_SIZE    = int(_cfg.get("chunk_size",    256))
CHUNK_OVERLAP = int(_cfg.get("chunk_overlap", 40))
DOC_DIR       = ROOT / _cfg.get("documents_dir",  "documents")
INDEX_DIR     = ROOT / _cfg.get("faiss_index_dir", "faiss_index")


@dataclass(frozen=True)
class CorpusPaths:
    index: Path
    metadata: Path
    manifest: Path


def _corpus_paths() -> CorpusPaths:
    return CorpusPaths(
        index=INDEX_DIR / "index.bin",
        metadata=INDEX_DIR / "metadata.json",
        manifest=INDEX_DIR / "corpus_manifest.json",
    )


# --- Embedding & FAISS helpers ---

def get_embedding(text: str) -> np.ndarray:
    result = requests.post(
        EMBED_URL,
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=30,
    )
    result.raise_for_status()
    vector = np.array(result.json()["embedding"], dtype=np.float32)
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("chunk size must be positive and overlap must be smaller than size")
    words = text.split()
    for i in range(0, len(words), size - overlap):
        yield " ".join(words[i:i + size])

def ensure_faiss_ready():
    process_documents()


def _document_files() -> list[Path]:
    if not DOC_DIR.exists():
        return []
    return sorted(
        path for path in DOC_DIR.rglob("*")
        if path.is_file()
        and not any(
            part.startswith(".") for part in path.relative_to(DOC_DIR).parts
        )
    )


def _corpus_fingerprint(files: list[Path]) -> str:
    digest = hashlib.sha256()
    for file in files:
        digest.update(str(file.relative_to(DOC_DIR)).replace("\\", "/").encode("utf-8"))
        digest.update(file.read_bytes())
    return digest.hexdigest()


def _extract_text(file: Path) -> str:
    ext = file.suffix.lower()
    if ext == ".pdf":
        return pymupdf4llm.to_markdown(str(file))
    if ext in {".html", ".htm"}:
        return trafilatura.extract(file.read_text(encoding="utf-8", errors="ignore")) or ""
    if ext in {".txt", ".md", ".csv", ".json", ".yaml", ".yml"}:
        return file.read_text(encoding="utf-8", errors="ignore")
    return MarkItDown().convert(str(file)).text_content


def _indexing_signature() -> dict:
    return {
        "embedding_model": EMBED_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
    }


def _load_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _corpus_is_current(
    paths: CorpusPaths,
    manifest: dict,
    fingerprint: str,
) -> bool:
    return (
        manifest.get("fingerprint") == fingerprint
        and manifest.get("indexing_signature") == _indexing_signature()
        and paths.index.exists()
        and paths.metadata.exists()
    )


def _build_corpus(files: list[Path]) -> tuple[list[dict], list[np.ndarray]]:
    metadata: list[dict] = []
    vectors: list[np.ndarray] = []
    for file in files:
        relative = str(file.relative_to(DOC_DIR)).replace("\\", "/")
        for chunk in chunk_text(_extract_text(file)):
            if not chunk.strip():
                continue
            vectors.append(get_embedding(chunk))
            metadata.append({
                "doc": relative,
                "chunk": chunk,
                "chunk_id": len(metadata),
            })
    return metadata, vectors


def _create_faiss_index(vectors: list[np.ndarray]):
    if not vectors:
        return None
    matrix = np.asarray(vectors, dtype=np.float32)
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    return index


def _persist_corpus(
    paths: CorpusPaths,
    metadata: list[dict],
    index,
    manifest: dict,
) -> None:
    metadata_tmp = paths.metadata.with_suffix(".json.tmp")
    metadata_tmp.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if index is None:
        if paths.index.exists():
            paths.index.unlink()
    else:
        index_tmp = paths.index.with_suffix(".bin.tmp")
        faiss.write_index(index, str(index_tmp))
        index_tmp.replace(paths.index)

    metadata_tmp.replace(paths.metadata)
    paths.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

def process_documents():
    """Rebuild the document corpus when any source file changes."""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)

    paths = _corpus_paths()
    files = _document_files()
    fingerprint = _corpus_fingerprint(files)
    manifest = _load_json(paths.manifest, {})
    if _corpus_is_current(paths, manifest, fingerprint):
        return {"changed": False, "files": len(files), "chunks": manifest.get("chunks", 0)}

    metadata, vectors = _build_corpus(files)
    index = _create_faiss_index(vectors)
    updated_manifest = {
        "fingerprint": fingerprint,
        "files": len(files),
        "chunks": len(metadata),
        "indexing_signature": _indexing_signature(),
    }
    _persist_corpus(paths, metadata, index, updated_manifest)
    return {"changed": True, "files": len(files), "chunks": len(metadata)}


# --- Tools ---

def search_documents(query: str) -> list[str]:
    """Search indexed local documents and return grounded source chunks."""
    ensure_faiss_ready()
    try:
        paths = _corpus_paths()
        if not paths.index.exists():
            return []
        index = faiss.read_index(str(paths.index))
        metadata = _load_json(paths.metadata, [])
        query_vec = get_embedding(query).reshape(1, -1)
        scores, indices = index.search(query_vec, k=min(TOP_K, len(metadata)))
        return [
            f"{metadata[i]['chunk']}\n[Source: {metadata[i]['doc']}, "
            f"ID: {metadata[i]['chunk_id']}, Similarity: {float(score):.3f}]"
            for score, i in zip(scores[0], indices[0])
            if i >= 0 and float(score) >= MIN_SIMILARITY
        ]
    except Exception as e:
        return [f"ERROR: {e}"]


@mcp.tool()
def search_stored_documents_rag(input: SearchDocumentsInput) -> list[str]:
    """Search indexed local documents and return relevant text chunks."""
    return search_documents(input.query)


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
    run_mcp_server(mcp)
