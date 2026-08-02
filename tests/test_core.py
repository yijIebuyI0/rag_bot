from __future__ import annotations

import unittest
from dataclasses import dataclass

from langchain_core.documents import Document

from marketing_rag.chunking import ChunkingConfig, split_documents
from marketing_rag.documents import normalize_text
from marketing_rag.uploads import load_uploaded_documents


@dataclass
class FakeUpload:
    name: str
    payload: bytes

    def getvalue(self) -> bytes:
        return self.payload


class NormalizeTextTests(unittest.TestCase):
    def test_normalizes_noise_but_keeps_paragraphs(self) -> None:
        value = normalize_text("第一段  有空格\r\n\r\n\r\n第二段\x00")
        self.assertEqual(value, "第一段 有空格\n\n第二段")


class ChunkingTests(unittest.TestCase):
    def test_chunks_keep_citation_metadata(self) -> None:
        content = "。".join([f"这是第{i}条营销知识" for i in range(80)])
        document = Document(
            page_content=content,
            metadata={
                "source": "strategy.pdf",
                "file_name": "strategy.pdf",
                "page_number": 7,
                "document_id": "doc-1",
            },
        )
        chunks = split_documents(
            [document], ChunkingConfig(chunk_size=120, chunk_overlap=20)
        )

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.metadata["page_number"] == 7 for chunk in chunks))
        self.assertTrue(all(chunk.metadata.get("chunk_id") for chunk in chunks))
        self.assertEqual(len({chunk.metadata["chunk_id"] for chunk in chunks}), len(chunks))


class UploadTests(unittest.TestCase):
    def test_markdown_upload_keeps_file_metadata(self) -> None:
        documents, report = load_uploaded_documents(
            [FakeUpload("知识库.md", "# 标题\n这是上传内容。".encode("utf-8"))]
        )
        self.assertEqual(report.successful_file_count, 1)
        self.assertEqual(documents[0].metadata["file_name"], "知识库.md")
        self.assertEqual(documents[0].metadata["file_type"], "md")
        self.assertIn("这是上传内容", documents[0].page_content)


if __name__ == "__main__":
    unittest.main()
