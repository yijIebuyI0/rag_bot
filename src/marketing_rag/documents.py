from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader


SUPPORTED_SUFFIXES = {".pdf", ".md", ".txt"}


@dataclass(slots=True)
class IngestionReport:
    source_files: list[Path] = field(default_factory=list)
    document_count: int = 0
    skipped_pages: int = 0
    errors: list[tuple[str, str]] = field(default_factory=list)

    @property
    def successful_file_count(self) -> int:
        return len(self.source_files) - len(self.errors)


def normalize_text(text: str) -> str:
    """Remove extraction noise without flattening paragraph boundaries."""

    text = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def discover_files(data_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in data_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )


def _relative_source(path: Path, data_dir: Path) -> str:
    return path.relative_to(data_dir).as_posix()


def load_pdf(path: Path, data_dir: Path) -> tuple[list[Document], int]:
    reader = PdfReader(str(path))
    if reader.is_encrypted:
        raise ValueError("PDF 已加密，无法读取")

    source = _relative_source(path, data_dir)
    document_id = _file_digest(path)[:20]
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
                    "source": source,
                    "file_name": path.name,
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


def _read_text(path: Path) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("无法识别文件编码")


def load_text_document(path: Path, data_dir: Path) -> list[Document]:
    content = normalize_text(_read_text(path))
    if not content:
        raise ValueError("文件内容为空")
    return [
        Document(
            page_content=content,
            metadata={
                "source": _relative_source(path, data_dir),
                "file_name": path.name,
                "file_type": path.suffix.lower().lstrip("."),
                "document_id": _file_digest(path)[:20],
            },
        )
    ]


def load_documents(data_dir: Path) -> tuple[list[Document], IngestionReport]:
    files = discover_files(data_dir)
    if not files:
        raise FileNotFoundError(f"没有发现 PDF、MD 或 TXT 文件：{data_dir}")

    report = IngestionReport(source_files=files)
    documents: list[Document] = []

    for path in files:
        try:
            if path.suffix.lower() == ".pdf":
                loaded, skipped = load_pdf(path, data_dir)
                report.skipped_pages += skipped
            else:
                loaded = load_text_document(path, data_dir)
            documents.extend(loaded)
        except Exception as exc:  # A broken file must not block the corpus.
            report.errors.append((_relative_source(path, data_dir), str(exc)))

    if not documents:
        details = "；".join(f"{name}: {message}" for name, message in report.errors)
        raise RuntimeError(f"没有成功加载任何资料。{details}")

    report.document_count = len(documents)
    return documents, report

