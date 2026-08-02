from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from langchain_chroma import Chroma

from .chunking import ChunkingConfig, split_documents
from .config import Settings
from .documents import IngestionReport, load_documents
from .embeddings import BailianEmbeddings
from .indexing import IndexBundle, build_or_load_index
from .rag import DocumentRAGService
from .uploads import UploadReport, UploadedFileLike, load_uploaded_documents


@dataclass(slots=True)
class Runtime:
    settings: Settings
    report: IngestionReport
    index: IndexBundle
    service: DocumentRAGService


@dataclass(slots=True)
class UploadRuntime:
    settings: Settings
    report: UploadReport
    chunk_count: int
    service: DocumentRAGService


def create_runtime(
    settings: Settings | None = None,
    *,
    force_rebuild: bool = False,
) -> Runtime:
    settings = settings or Settings.from_env()
    settings.validate()
    chunking = ChunkingConfig()
    documents, report = load_documents(settings.data_dir)
    chunks = split_documents(documents, chunking)
    index = build_or_load_index(
        settings=settings,
        chunks=chunks,
        chunking=chunking,
        force=force_rebuild,
    )
    service = DocumentRAGService(settings, index.vector_store)
    return Runtime(
        settings=settings,
        report=report,
        index=index,
        service=service,
    )


def create_upload_runtime(
    files: list[UploadedFileLike],
    settings: Settings | None = None,
) -> UploadRuntime:
    settings = settings or Settings.from_env()
    settings.validate_api()
    documents, report = load_uploaded_documents(files)
    chunks = split_documents(documents, ChunkingConfig())

    embeddings = BailianEmbeddings(
        api_key=settings.api_key,
        base_url=settings.base_url,
        model=settings.embedding_model,
        dimensions=settings.embedding_dimensions,
    )
    collection_name = f"uploaded_docs_{uuid4().hex[:16]}"
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"},
    )
    vector_store.add_documents(
        documents=chunks,
        ids=[str(chunk.metadata["chunk_id"]) for chunk in chunks],
    )
    return UploadRuntime(
        settings=settings,
        report=report,
        chunk_count=len(chunks),
        service=DocumentRAGService(settings, vector_store),
    )
