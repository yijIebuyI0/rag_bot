from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from marketing_rag.config import Settings


ENV_KEYS = (
    "DASHSCOPE_API_KEY",
    "DASHSCOPE_BASE_URL",
    "RAG_DATA_DIR",
    "RAG_INDEX_DIR",
    "RAG_CHAT_MODEL",
    "RAG_EMBEDDING_MODEL",
    "RAG_EMBEDDING_DIMENSIONS",
    "RAG_TOP_K",
    "RAG_MAX_CONTEXT_CHARS",
    "RAG_TEMPERATURE",
)


class SettingsTests(unittest.TestCase):
    def setUp(self) -> None:
        self._saved = {key: os.environ.pop(key, None) for key in ENV_KEYS}
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self._tmp.cleanup()

    def test_defaults_point_inside_project_dir(self) -> None:
        os.environ["DASHSCOPE_API_KEY"] = "test-key"

        settings = Settings.from_env(self.root)

        self.assertEqual(settings.data_dir, (self.root / "data").resolve())
        self.assertEqual(settings.index_dir, (self.root / "chroma_db").resolve())
        self.assertEqual(settings.chat_model, "deepseek-v4-flash")
        self.assertEqual(settings.embedding_model, "text-embedding-v4")
        self.assertEqual(settings.embedding_dimensions, 1024)
        self.assertEqual(settings.top_k, 6)
        self.assertEqual(settings.max_context_chars, 12_000)
        self.assertEqual(settings.temperature, 0.2)
        self.assertEqual(settings.collection_prefix, "marketing_knowledge")
        self.assertFalse(settings.base_url.endswith("/"))

    def test_relative_env_paths_resolve_against_project_dir(self) -> None:
        os.environ["RAG_DATA_DIR"] = "knowledge"
        os.environ["RAG_INDEX_DIR"] = ".index"

        settings = Settings.from_env(self.root)

        self.assertEqual(settings.data_dir, (self.root / "knowledge").resolve())
        self.assertEqual(settings.index_dir, (self.root / ".index").resolve())

    def test_numeric_env_values_override_defaults(self) -> None:
        os.environ["RAG_TOP_K"] = "3"
        os.environ["RAG_TEMPERATURE"] = "0.7"
        os.environ["RAG_EMBEDDING_DIMENSIONS"] = "512"
        os.environ["RAG_MAX_CONTEXT_CHARS"] = "800"
        os.environ["RAG_CHAT_MODEL"] = "custom-chat"

        settings = Settings.from_env(self.root)

        self.assertEqual(settings.top_k, 3)
        self.assertAlmostEqual(settings.temperature, 0.7)
        self.assertEqual(settings.embedding_dimensions, 512)
        self.assertEqual(settings.max_context_chars, 800)
        self.assertEqual(settings.chat_model, "custom-chat")

    def test_validate_reports_missing_key_and_missing_data_dir(self) -> None:
        settings = Settings.from_env(self.root)

        with self.assertRaises(ValueError) as context:
            settings.validate()

        message = str(context.exception)
        self.assertIn("DASHSCOPE_API_KEY", message)
        self.assertIn("资料目录不存在", message)

    def test_validate_api_ignores_corpus_state(self) -> None:
        os.environ["DASHSCOPE_API_KEY"] = "test-key"
        settings = Settings.from_env(self.root)

        self.assertIsNone(settings.validate_api())

    def test_validate_passes_when_data_dir_exists(self) -> None:
        os.environ["DASHSCOPE_API_KEY"] = "test-key"
        (self.root / "data").mkdir()
        settings = Settings.from_env(self.root)

        self.assertIsNone(settings.validate())


if __name__ == "__main__":
    unittest.main()
