from __future__ import annotations

import hashlib
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    chunk_size: int = 700
    chunk_overlap: int = 120

    @property
    def version(self) -> str:
        return f"recursive-char-{self.chunk_size}-{self.chunk_overlap}-zh-v1"


SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", "、", " ", ""]


def split_documents(
    documents: list[Document],
    config: ChunkingConfig | None = None,
) -> list[Document]:
    config = config or ChunkingConfig()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        length_function=len,
        add_start_index=True,
        separators=SEPARATORS,
    )
    chunks = splitter.split_documents(documents)

    for ordinal, chunk in enumerate(chunks):
        identity = "|".join(
            [
                str(chunk.metadata.get("document_id", "")),
                str(chunk.metadata.get("page_number", "")),
                str(chunk.metadata.get("start_index", "")),
                chunk.page_content,
            ]
        )
        chunk.metadata["chunk_id"] = hashlib.sha256(
            identity.encode("utf-8")
        ).hexdigest()[:24]
        chunk.metadata["chunk_ordinal"] = ordinal
        chunk.metadata["chunking_version"] = config.version
    return chunks

