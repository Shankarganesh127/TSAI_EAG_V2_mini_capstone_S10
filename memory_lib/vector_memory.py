from __future__ import annotations

import asyncio
import hashlib
import json
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from .config import MemoryConfig, get_default_memory_config


@dataclass(frozen=True)
class MemoryHit:
    query: str
    answer: str
    score: float
    session_id: str
    created_at: str
    log_path: str = ""


@dataclass(frozen=True)
class SyncReport:
    changed: bool
    source_files: int
    indexed_records: int


class SessionVectorMemory:
    """Persistent FAISS-backed memory for successful query/answer turns."""

    def __init__(self, embedder: Any, config: MemoryConfig | None = None):
        if embedder is None or not callable(getattr(embedder, "embed", None)):
            raise ValueError("SessionVectorMemory requires an object with an embed() method")
        self.embedder = embedder
        self.config = config or get_default_memory_config()
        self.base_dir = Path(self.config.base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.records_path = self.base_dir / "records.json"
        self.index_path = self.base_dir / "memory.faiss"
        self.manifest_path = self.base_dir / "source_manifest.json"
        self._lock = threading.RLock()
        self._records = self._load_records()
        self._index = self._load_or_rebuild_index()

    async def sync_session_logs(self, logs_dir: str | Path) -> SyncReport:
        """Rebuild the index when the source session-log database changes."""
        return await asyncio.to_thread(self._sync_session_logs, Path(logs_dir))

    def _sync_session_logs(self, logs_dir: Path) -> SyncReport:
        files = sorted(logs_dir.rglob("*.json")) if logs_dir.exists() else []
        fingerprint = self._source_fingerprint(logs_dir, files)
        manifest = self._load_manifest()
        if (
            manifest.get("fingerprint") == fingerprint
            and manifest.get("source_root") == str(logs_dir.resolve())
            and manifest.get("embedding_signature") == self._embedding_signature()
        ):
            return SyncReport(False, len(files), len(self._records))

        source_records = self._extract_session_records(files)
        records: list[dict[str, Any]] = []
        vectors: list[np.ndarray] = []
        for record in source_records:
            vector = self._embed(record["query"])
            record["embedding"] = vector[0].tolist()
            records.append(record)
            vectors.append(vector[0])

        index = self._build_index(vectors)

        with self._lock:
            self._persist_snapshot(records, index)
            self._records = records
            self._index = index
            self._save_manifest({
                "fingerprint": fingerprint,
                "source_root": str(logs_dir.resolve()),
                "source_files": len(files),
                "indexed_records": len(records),
                "embedding_signature": self._embedding_signature(),
                "rebuilt_at": datetime.now(timezone.utc).isoformat(),
            })
        return SyncReport(True, len(files), len(records))

    def _embedding_signature(self) -> dict[str, str]:
        config = getattr(self.embedder, "config", None)
        return {
            "provider": str(getattr(config, "embedding_provider", "unknown")),
            "model": str(getattr(config, "embedding_model", "unknown")),
        }

    @staticmethod
    def _source_fingerprint(logs_dir: Path, files: list[Path]) -> str:
        digest = hashlib.sha256()
        for path in files:
            try:
                relative = path.relative_to(logs_dir)
            except ValueError:
                relative = path
            digest.update(str(relative).replace("\\", "/").encode("utf-8"))
            digest.update(path.read_bytes())
        return digest.hexdigest()

    def _load_manifest(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            return {}
        try:
            with self.manifest_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    @staticmethod
    def _extract_session_records(files: list[Path]) -> list[dict[str, Any]]:
        records_by_query: dict[str, dict[str, Any]] = {}
        for path in files:
            try:
                with path.open("r", encoding="utf-8") as handle:
                    session = json.load(handle)
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"Cannot read session log {path}: {exc}") from exc
            session_id = str(session.get("session_id", ""))
            fallback_query = str(session.get("query", ""))
            for stage in session.get("stages", []):
                stage_name = stage.get("stage")
                if stage_name == "agent_result":
                    query = str(stage.get("query", fallback_query)).strip()
                    answer = str(stage.get("answer", "")).strip()
                elif stage_name == "session_end" and stage.get("final_answer"):
                    query = fallback_query.strip()
                    answer = str(stage.get("final_answer", "")).strip()
                else:
                    continue
                if not query or not answer or query == "interactive session":
                    continue
                record = {
                    "query": query,
                    "answer": answer,
                    "session_id": session_id,
                    "created_at": str(stage.get("timestamp", session.get("started_at", ""))),
                    "log_path": str(path),
                }
                key = " ".join(query.lower().split())
                previous = records_by_query.get(key)
                if previous is None or record["created_at"] >= previous["created_at"]:
                    records_by_query[key] = record
        return sorted(records_by_query.values(), key=lambda item: item["created_at"])

    def _load_records(self) -> list[dict[str, Any]]:
        if not self.records_path.exists():
            return []
        try:
            with self.records_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Cannot load vector-memory records: {exc}") from exc
        if not isinstance(data, list):
            raise RuntimeError("Vector-memory records must be a JSON list")
        return data

    def _load_or_rebuild_index(self):
        if not self._records:
            return None
        vectors = np.asarray([record["embedding"] for record in self._records], dtype="float32")
        faiss.normalize_L2(vectors)
        # Records contain the canonical embeddings. Reconstructing prevents a
        # partially completed two-file update from pairing stale vectors with
        # newer metadata after a process interruption.
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        faiss.write_index(index, str(self.index_path))
        return index

    @staticmethod
    def _build_index(vectors: list[np.ndarray]):
        if not vectors:
            return None
        matrix = np.asarray(vectors, dtype="float32")
        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)
        return index

    def _embed(self, text: str) -> np.ndarray:
        result = self.embedder.embed(text)
        if not result or not result[0]:
            raise RuntimeError("Embedding provider returned no vector")
        vector = np.asarray([result[0]], dtype="float32")
        if vector.ndim != 2 or vector.shape[1] == 0:
            raise RuntimeError("Embedding provider returned an invalid vector")
        faiss.normalize_L2(vector)
        return vector

    async def search(self, query: str) -> list[MemoryHit]:
        return await asyncio.to_thread(self._search_sync, query)

    def _search_sync(self, query: str) -> list[MemoryHit]:
        vector = self._embed(query)
        with self._lock:
            if self._index is None or not self._records:
                return []
            if vector.shape[1] != self._index.d:
                raise RuntimeError("Embedding dimension changed; rebuild the memory store")
            limit = min(self.config.top_k, len(self._records))
            scores, indices = self._index.search(vector, limit)
            hits = []
            for score, position in zip(scores[0], indices[0]):
                if position < 0 or float(score) < self.config.similarity_threshold:
                    continue
                record = self._records[int(position)]
                hits.append(self._to_memory_hit(record, float(score)))
            return hits

    @staticmethod
    def _to_memory_hit(record: dict[str, Any], score: float) -> MemoryHit:
        return MemoryHit(
            query=record["query"],
            answer=record["answer"],
            score=score,
            session_id=record["session_id"],
            created_at=record["created_at"],
            log_path=record.get("log_path", ""),
        )

    async def add(
        self, query: str, answer: str, session_id: str, log_path: str = ""
    ) -> None:
        await asyncio.to_thread(self._add_sync, query, answer, session_id, log_path)

    def _add_sync(self, query: str, answer: str, session_id: str, log_path: str) -> None:
        vector = self._embed(query)
        record = {
            "query": query, "answer": answer, "session_id": session_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "log_path": log_path,
            "embedding": vector[0].tolist(),
        }
        with self._lock:
            if self._index is None:
                self._index = faiss.IndexFlatIP(vector.shape[1])
            elif vector.shape[1] != self._index.d:
                raise RuntimeError("Embedding dimension changed; rebuild the memory store")
            self._index.add(vector)
            self._records.append(record)
            self._persist()

    def _persist(self) -> None:
        index_tmp = self.index_path.with_suffix(".faiss.tmp")
        self._write_json_atomic(self.records_path, self._records)
        faiss.write_index(self._index, str(index_tmp))
        index_tmp.replace(self.index_path)

    def _persist_snapshot(self, records: list[dict[str, Any]], index) -> None:
        self._write_json_atomic(self.records_path, records)
        if index is not None:
            index_tmp = self.index_path.with_suffix(".faiss.tmp")
            faiss.write_index(index, str(index_tmp))
            index_tmp.replace(self.index_path)
        elif self.index_path.exists():
            self.index_path.unlink()

    def _save_manifest(self, manifest: dict[str, Any]) -> None:
        self._write_json_atomic(self.manifest_path, manifest)

    @staticmethod
    def _write_json_atomic(path: Path, data: Any) -> None:
        temporary = path.with_suffix(f"{path.suffix}.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
        temporary.replace(path)

    @staticmethod
    def describe_hit(hit: MemoryHit) -> dict[str, Any]:
        return asdict(hit)
