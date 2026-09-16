from __future__ import annotations

import os
import gc
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from chromadb.api.client import SharedSystemClient
from langchain_core.embeddings import Embeddings

from marketing_rag.chunking import ChunkingConfig, split_documents
from marketing_rag.config import Settings
from marketing_rag.documents import load_documents
from marketing_rag.indexing import build_or_load_index, corpus_fingerprint


class StubEmbeddings(Embeddings):
    """Deterministic local embeddings so indexing never reaches the network."""

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs

    @staticmethod
    def _vector(text: str) -> list[float]:
        seed = sum(text.encode("utf-8")) % 97
        return [float((seed + index) % 13) + 1.0 for index in range(8)]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


class IndexingTests(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["ANONYMIZED_TELEMETRY"] = "False"
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.data_dir = self.root / "data"
        self.data_dir.mkdir()
        self.source_file = self.data_dir / "策略.md"
        self.source_file.write_text(
            "# 投放策略\n今年的投放重点是一线城市。", encoding="utf-8"
        )
        self.settings = Settings(
            project_dir=self.root,
            data_dir=self.data_dir,
            index_dir=self.root / "index",
            api_key="test-key",
            base_url="https://example.invalid/v1",
            chat_model="test-chat",
            embedding_model="test-embed",
            embedding_dimensions=8,
            top_k=4,
            max_context_chars=1000,
            temperature=0.0,
        )
        self.chunking = ChunkingConfig(chunk_size=120, chunk_overlap=20)
        self._patcher = mock.patch(
            "marketing_rag.indexing.BailianEmbeddings", StubEmbeddings
        )
        self._patcher.start()
        self.addCleanup(self._patcher.stop)
        # Chroma keeps the persisted segment files open; release them before
        # TemporaryDirectory tries to delete the folder (Windows file locking).
        self.addCleanup(self._tmp.cleanup)
        self.addCleanup(self._release_chroma)

    @staticmethod
    def _release_chroma() -> None:
        SharedSystemClient.clear_system_cache()
        gc.collect()

    def _chunks(self) -> list:
        documents, _ = load_documents(self.data_dir)
        return split_documents(documents, self.chunking)

    def test_build_creates_index_and_reuses_complete_manifest(self) -> None:
        chunks = self._chunks()

        first = build_or_load_index(
            settings=self.settings, chunks=chunks, chunking=self.chunking
        )

        self.assertTrue(first.created)
        self.assertEqual(first.chunk_count, len(chunks))
        self.assertEqual(first.source_count, 1)
        self.assertTrue((self.settings.index_dir / "manifest.json").exists())
        self.assertTrue(first.collection_name.startswith("marketing_knowledge_"))

        hits = first.vector_store.similarity_search("投放重点", k=1)
        self.assertEqual(len(hits), 1)
        self.assertIn("一线城市", hits[0].page_content)

        second = build_or_load_index(
            settings=self.settings, chunks=chunks, chunking=self.chunking
        )

        self.assertFalse(second.created)
        self.assertEqual(second.collection_name, first.collection_name)
        self.assertEqual(second.fingerprint, first.fingerprint)
        self.assertEqual(second.chunk_count, len(chunks))

    def test_force_rebuild_recreates_the_collection(self) -> None:
        chunks = self._chunks()
        first = build_or_load_index(
            settings=self.settings, chunks=chunks, chunking=self.chunking
        )

        forced = build_or_load_index(
            settings=self.settings, chunks=chunks, chunking=self.chunking, force=True
        )

        self.assertTrue(forced.created)
        self.assertEqual(forced.collection_name, first.collection_name)
        self.assertEqual(
            forced.vector_store._collection.count(), len(chunks)  # noqa: SLF001
        )

    def test_source_change_invalidates_the_index(self) -> None:
        chunks = self._chunks()
        first = build_or_load_index(
            settings=self.settings, chunks=chunks, chunking=self.chunking
        )

        self.source_file.write_text(
            "# 投放策略\n今年的投放重点是一线城市和下沉市场。", encoding="utf-8"
        )
        updated_chunks = self._chunks()
        second = build_or_load_index(
            settings=self.settings, chunks=updated_chunks, chunking=self.chunking
        )

        self.assertTrue(second.created)
        self.assertNotEqual(second.fingerprint, first.fingerprint)
        self.assertNotEqual(second.collection_name, first.collection_name)

    def test_fingerprint_tracks_chunking_configuration(self) -> None:
        first, sources = corpus_fingerprint(self.settings, self.chunking)
        second, _ = corpus_fingerprint(
            self.settings, ChunkingConfig(chunk_size=64, chunk_overlap=8)
        )

        self.assertEqual(len(sources), 1)
        self.assertNotEqual(first, second)

    def test_build_rejects_empty_chunk_list(self) -> None:
        with self.assertRaises(ValueError):
            build_or_load_index(
                settings=self.settings, chunks=[], chunking=self.chunking
            )


if __name__ == "__main__":
    unittest.main()
