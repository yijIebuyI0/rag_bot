from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Protocol

from langchain_core.documents import Document
from pypdf import PdfReader

from .documents import SUPPORTED_SUFFIXES, normalize_text


class UploadedFileLike(Protocol):
    name: str

    def getvalue(self) -> bytes: ...


@dataclass(slots=True)
class UploadReport:
    file_names: list[str] = field(default_factory=list)
    document_count: int = 0
    skipped_pages: int = 0
    errors: list[tuple[str, str]] = field(default_factory=list)

    @property
    def successful_file_count(self) -> int:
        return len(self.file_names) - len(self.errors)


def _decode_text(payload: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return payload.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("无法识别文件编码")


def _load_pdf(name: str, payload: bytes) -> tuple[list[Document], int]:
    reader = PdfReader(BytesIO(payload))
    if reader.is_encrypted:
        raise ValueError("PDF 已加密，无法读取")

    document_id = hashlib.sha256(payload).hexdigest()[:20]
    total_pages = len(reader.pages)
    documents: list[Document] = []
    skipped_pages = 0
    for page_index, page in enumerate(reader.pages):
        content = normalize_text(page.extract_text() or "")
        if not content:
            skipped_pages += 1
            continue
        documents.append(
            Document(
                page_content=content,
                metadata={
                    "source": name,
                    "file_name": name,
                    "file_type": "pdf",
                    "page_number": page_index + 1,
                    "total_pages": total_pages,
                    "document_id": document_id,
                },
            )
        )
    if not documents:
        raise ValueError("没有提取到文字；该文件可能是扫描版 PDF")
    return documents, skipped_pages


def _load_text(name: str, suffix: str, payload: bytes) -> list[Document]:
    content = normalize_text(_decode_text(payload))
    if not content:
        raise ValueError("文件内容为空")
    return [
        Document(
            page_content=content,
            metadata={
                "source": name,
                "file_name": name,
                "file_type": suffix.lstrip("."),
                "document_id": hashlib.sha256(payload).hexdigest()[:20],
            },
        )
    ]


def load_uploaded_documents(
    files: list[UploadedFileLike],
) -> tuple[list[Document], UploadReport]:
    if not files:
        raise ValueError("请先上传至少一个文档")

    report = UploadReport(file_names=[Path(file.name).name for file in files])
    documents: list[Document] = []

    for uploaded in files:
        name = Path(uploaded.name).name
        suffix = Path(name).suffix.lower()
        try:
            if suffix not in SUPPORTED_SUFFIXES:
                raise ValueError("仅支持 PDF、MD 和 TXT")
            payload = uploaded.getvalue()
            if not payload:
                raise ValueError("文件为空")
            if suffix == ".pdf":
                loaded, skipped = _load_pdf(name, payload)
                report.skipped_pages += skipped
            else:
                loaded = _load_text(name, suffix, payload)
            documents.extend(loaded)
        except Exception as exc:
            report.errors.append((name, str(exc)))

    if not documents:
        details = "；".join(f"{name}: {message}" for name, message in report.errors)
        raise RuntimeError(f"没有成功读取任何文档。{details}")
    report.document_count = len(documents)
    return documents, report
