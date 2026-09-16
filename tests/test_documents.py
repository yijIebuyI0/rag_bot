from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from marketing_rag.documents import (
    discover_files,
    load_documents,
    load_text_document,
    normalize_text,
)


class DiscoverFilesTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_discovers_supported_files_recursively(self) -> None:
        (self.root / "sub").mkdir()
        (self.root / "a.md").write_text("甲", encoding="utf-8")
        (self.root / "sub" / "b.txt").write_text("乙", encoding="utf-8")
        (self.root / "sub" / "c.pdf").write_bytes(b"%PDF-1.4 broken")
        (self.root / "d.docx").write_bytes(b"not supported")

        found = discover_files(self.root)

        self.assertEqual(
            [path.name for path in found], ["a.md", "b.txt", "c.pdf"]
        )

    def test_load_documents_reports_broken_files_without_aborting(self) -> None:
        (self.root / "good.md").write_text("可用内容", encoding="utf-8")
        (self.root / "broken.pdf").write_bytes(b"%PDF-1.4 not a real pdf")

        documents, report = load_documents(self.root)

        self.assertEqual(report.successful_file_count, 1)
        self.assertEqual(report.document_count, 1)
        self.assertEqual(len(report.errors), 1)
        self.assertEqual(report.errors[0][0], "broken.pdf")
        self.assertEqual(documents[0].metadata["file_name"], "good.md")

    def test_load_documents_fails_when_nothing_can_be_loaded(self) -> None:
        (self.root / "broken.pdf").write_bytes(b"%PDF-1.4 not a real pdf")

        with self.assertRaises(RuntimeError) as context:
            load_documents(self.root)

        self.assertIn("broken.pdf", str(context.exception))

    def test_load_documents_fails_on_empty_corpus(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_documents(self.root)


class TextDocumentTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_gb18030_files_are_readable(self) -> None:
        path = self.root / "gb.txt"
        path.write_bytes("中文资料".encode("gb18030"))

        documents = load_text_document(path, self.root)

        self.assertEqual(documents[0].page_content, "中文资料")
        self.assertEqual(documents[0].metadata["file_type"], "txt")
        self.assertEqual(documents[0].metadata["source"], "gb.txt")
        self.assertTrue(documents[0].metadata["document_id"])

    def test_empty_file_is_rejected(self) -> None:
        path = self.root / "empty.md"
        path.write_text("   \n\n", encoding="utf-8")

        with self.assertRaises(ValueError):
            load_text_document(path, self.root)

    def test_normalize_text_keeps_paragraph_boundaries(self) -> None:
        value = normalize_text("甲\t\t乙\n\n\n\n丙")

        self.assertEqual(value, "甲 乙\n\n丙")


if __name__ == "__main__":
    unittest.main()
