from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from .chunking import ChunkingConfig
from .config import Settings
from .documents import discover_files
from .embeddings import BailianEmbeddings


@dataclass(slots=True)
class IndexBundle:
    vector_store: Chroma
    collection_name: str
    fingerprint: str
    chunk_count: int
    source_count: int
    created: bool


def _source_state(data_dir: Path) -> list[dict[str, object]]:
    state: list[dict[str, object]] = []
    for path in discover_files(data_dir):
        stat = path.stat()
        state.append(
            {
                "source": path.relative_to(data_dir).as_posix(),
                "size": stat.st_size,
                "modified_ns": stat.st_mtime_ns,
            }
        )
    return state


def corpus_fingerprint(
    settings: Settings,
    chunking: ChunkingConfig,
) -> tuple[str, list[dict[str, object]]]:
    sources = _source_state(settings.data_dir)
    payload = {
        "sources": sources,
        "chunking": chunking.version,
        "embedding_model": settings.embedding_model,
        "embedding_dimensions": settings.embedding_dimensions,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), sources


def _read_manifest(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _write_manifest(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def build_or_load_index(
    *,
    settings: Settings,
    chunks: list[Document],
    chunking: ChunkingConfig,
    force: bool = False,
) -> IndexBundle:
    if not chunks:
        raise ValueError("没有可写入向量库的文本片段")

    settings.index_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = settings.index_dir / "manifest.json"
    fingerprint, sources = corpus_fingerprint(settings, chunking)
    collection_name = f"{settings.collection_prefix}_{fingerprint[:12]}"
    manifest = _read_manifest(manifest_path)

    embeddings = BailianEmbeddings(
        api_key=settings.api_key,
        base_url=settings.base_url,
        model=settings.embedding_model,
        dimensions=settings.embedding_dimensions,
    )

    is_current = bool(
        manifest
        and manifest.get("fingerprint") == fingerprint
        and manifest.get("collection_name") == collection_name
        and manifest.get("status") == "complete"
    )

    if is_current and not force:
        store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=str(settings.index_dir),
            collection_metadata={"hnsw:space": "cosine"},
        )
        return IndexBundle(
            vector_store=store,
            collection_name=collection_name,
            fingerprint=fingerprint,
            chunk_count=int(manifest.get("chunk_count", len(chunks))),
            source_count=int(manifest.get("source_count", len(sources))),
            created=False,
        )

    # The index is derived data. Rebuilding this fingerprint is safe and prevents duplicates
    # after an interrupted indexing attempt.
    candidate = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(settings.index_dir),
        collection_metadata={"hnsw:space": "cosine"},
    )
    candidate.delete_collection()
    store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(settings.index_dir),
        collection_metadata={"hnsw:space": "cosine"},
    )

    ids = [str(chunk.metadata["chunk_id"]) for chunk in chunks]
    store.add_documents(documents=chunks, ids=ids)

    _write_manifest(
        manifest_path,
        {
            "status": "complete",
            "fingerprint": fingerprint,
            "collection_name": collection_name,
            "source_count": len(sources),
            "chunk_count": len(chunks),
            "embedding_model": settings.embedding_model,
            "embedding_dimensions": settings.embedding_dimensions,
            "chunking_version": chunking.version,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "sources": sources,
        },
    )
    return IndexBundle(
        vector_store=store,
        collection_name=collection_name,
        fingerprint=fingerprint,
        chunk_count=len(chunks),
        source_count=len(sources),
        created=True,
    )

