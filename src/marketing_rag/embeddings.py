from __future__ import annotations

from collections.abc import Iterable

from langchain_core.embeddings import Embeddings
from openai import OpenAI


class BailianEmbeddings(Embeddings):
    """LangChain embedding adapter for Bailian's OpenAI-compatible endpoint."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str = "text-embedding-v4",
        dimensions: int = 1024,
        batch_size: int = 10,
    ) -> None:
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=60,
            max_retries=3,
        )
        self.model = model
        self.dimensions = dimensions
        self.batch_size = min(max(batch_size, 1), 10)

    @staticmethod
    def _batches(values: list[str], size: int) -> Iterable[list[str]]:
        for start in range(0, len(values), size):
            yield values[start : start + size]

    def _request(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
            dimensions=self.dimensions,
            encoding_format="float",
        )
        ordered = sorted(response.data, key=lambda item: item.index)
        vectors = [item.embedding for item in ordered]
        if len(vectors) != len(texts):
            raise RuntimeError("百炼返回的向量数量与输入文本数量不一致")
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for batch in self._batches(texts, self.batch_size):
            vectors.extend(self._request(batch))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError("检索问题不能为空")
        return self._request([text])[0]

