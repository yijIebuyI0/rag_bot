from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

from langchain_core.documents import Document

from marketing_rag.config import Settings
from marketing_rag.rag import DocumentRAGService, SourceExcerpt


def make_settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "project_dir": Path("."),
        "data_dir": Path("data"),
        "index_dir": Path("index"),
        "api_key": "test-key",
        "base_url": "https://example.invalid/v1",
        "chat_model": "test-chat",
        "embedding_model": "test-embed",
        "embedding_dimensions": 8,
        "top_k": 4,
        "max_context_chars": 12_000,
        "temperature": 0.0,
    }
    values.update(overrides)
    return Settings(**values)  # type: ignore[arg-type]


def make_document(
    content: str,
    *,
    chunk_id: str,
    file_name: str = "策略.pdf",
    page_number: int | None = 3,
) -> Document:
    metadata: dict[str, object] = {
        "chunk_id": chunk_id,
        "source": f"docs/{file_name}",
        "file_name": file_name,
        "start_index": 0,
    }
    if page_number is not None:
        metadata["page_number"] = page_number
    return Document(page_content=content, metadata=metadata)


class FakeVectorStore:
    def __init__(self, results: list[tuple[Document, float]]) -> None:
        self.results = results
        self.calls: list[tuple[str, int]] = []

    def similarity_search_with_relevance_scores(
        self, query: str, k: int
    ) -> list[tuple[Document, float]]:
        self.calls.append((query, k))
        return self.results


class FakeCompletionClient:
    """Stands in for ``openai.OpenAI`` without touching the network."""

    def __init__(self, content: str = "结论见 [资料1]。") -> None:
        self.content = content
        self.requests: list[dict[str, object]] = []
        create = self._create
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))

    def _create(self, **kwargs: object) -> object:
        self.requests.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))],
            usage=SimpleNamespace(prompt_tokens=123, completion_tokens=45),
        )


class RetrieveTests(unittest.TestCase):
    def test_retrieve_dedupes_chunks_and_clamps_scores(self) -> None:
        store = FakeVectorStore(
            [
                (make_document("第一段", chunk_id="c1"), 0.82),
                (make_document("第一段副本", chunk_id="c1"), 0.5),
                (make_document("第二段", chunk_id="c2"), 1.7),
            ]
        )
        service = DocumentRAGService(make_settings(top_k=3), store)  # type: ignore[arg-type]

        sources = service.retrieve("  今年的投放重点是什么  ")

        self.assertEqual([source.number for source in sources], [1, 2])
        self.assertEqual([source.chunk_id for source in sources], ["c1", "c2"])
        self.assertEqual([source.content for source in sources], ["第一段", "第二段"])
        self.assertAlmostEqual(sources[1].score, 1.0)
        self.assertEqual(store.calls, [("今年的投放重点是什么", 3)])

    def test_retrieve_rejects_blank_question(self) -> None:
        service = DocumentRAGService(make_settings(), FakeVectorStore([]))  # type: ignore[arg-type]

        with self.assertRaises(ValueError):
            service.retrieve("   ")

    def test_citation_label_falls_back_to_whole_document(self) -> None:
        source = SourceExcerpt(
            number=2,
            source="docs/策略.md",
            file_name="策略.md",
            page_number=None,
            content="全文内容",
            score=0.9,
            chunk_id="c1",
        )

        self.assertEqual(source.citation_label, "资料2｜策略.md｜全文")


class AnswerTests(unittest.TestCase):
    def test_answer_without_hits_skips_the_model_call(self) -> None:
        service = DocumentRAGService(make_settings(), FakeVectorStore([]))  # type: ignore[arg-type]
        client = FakeCompletionClient()
        service.client = client  # type: ignore[assignment]

        response = service.answer("公司有几个品牌？")

        self.assertEqual(response.sources, ())
        self.assertIn("现有资料不足以回答", response.answer)
        self.assertIsNone(response.input_tokens)
        self.assertEqual(client.requests, [])

    def test_answer_sends_context_and_returns_token_usage(self) -> None:
        store = FakeVectorStore([(make_document("投放重点是一线城市。", chunk_id="c1"), 0.91)])
        service = DocumentRAGService(make_settings(), store)  # type: ignore[arg-type]
        client = FakeCompletionClient()
        service.client = client  # type: ignore[assignment]

        response = service.answer("今年的投放重点是什么？")

        self.assertEqual(response.answer, "结论见 [资料1]。")
        self.assertEqual(response.input_tokens, 123)
        self.assertEqual(response.output_tokens, 45)
        self.assertEqual(len(response.sources), 1)
        self.assertEqual(len(client.requests), 1)

        request = client.requests[0]
        self.assertEqual(request["model"], "test-chat")
        self.assertEqual(request["temperature"], 0.0)
        messages = request["messages"]  # type: ignore[assignment]
        self.assertEqual(messages[0]["role"], "system")
        user_prompt = messages[1]["content"]
        self.assertIn("今年的投放重点是什么？", user_prompt)
        self.assertIn("[资料1]", user_prompt)
        self.assertIn("投放重点是一线城市。", user_prompt)
        self.assertIn("第 3 页", user_prompt)

    def test_context_is_truncated_to_max_context_chars(self) -> None:
        settings = make_settings(max_context_chars=1)
        service = DocumentRAGService(settings, FakeVectorStore([]))  # type: ignore[arg-type]
        sources = (
            SourceExcerpt(
                number=1,
                source="docs/a.md",
                file_name="a.md",
                page_number=None,
                content="甲",
                score=0.9,
                chunk_id="c1",
            ),
            SourceExcerpt(
                number=2,
                source="docs/b.md",
                file_name="b.md",
                page_number=None,
                content="乙",
                score=0.8,
                chunk_id="c2",
            ),
        )

        context = service._build_context(sources)

        self.assertIn("[资料1]", context)
        self.assertNotIn("[资料2]", context)


if __name__ == "__main__":
    unittest.main()
