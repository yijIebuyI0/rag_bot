from __future__ import annotations

from dataclasses import dataclass

from langchain_chroma import Chroma
from langchain_core.documents import Document
from openai import OpenAI

from .config import Settings


SYSTEM_PROMPT = """你是一个严谨的文档问答助手。

回答规则：
1. 只能使用给定资料作答，不得用常识补齐资料中没有的信息。
2. 先给直接结论，再解释方法、条件或执行步骤。
3. 每个关键事实后必须标注对应资料编号，例如 [资料1]。
4. 多份资料存在差异时，明确说明差异，不要擅自合并。
5. 资料不足时直接回答“现有资料不足以回答”，并说明还缺少什么。
6. 使用简洁、清晰的中文；除非用户要求，否则不要写空泛的背景介绍。
"""


@dataclass(frozen=True, slots=True)
class SourceExcerpt:
    number: int
    source: str
    file_name: str
    page_number: int | None
    content: str
    score: float
    chunk_id: str

    @property
    def citation_label(self) -> str:
        page = f"第 {self.page_number} 页" if self.page_number else "全文"
        return f"资料{self.number}｜{self.file_name}｜{page}"


@dataclass(frozen=True, slots=True)
class RAGResponse:
    question: str
    answer: str
    sources: tuple[SourceExcerpt, ...]
    input_tokens: int | None = None
    output_tokens: int | None = None


def _document_key(document: Document) -> str:
    return str(
        document.metadata.get("chunk_id")
        or (
            f"{document.metadata.get('source')}|"
            f"{document.metadata.get('page_number')}|"
            f"{document.metadata.get('start_index')}"
        )
    )


class DocumentRAGService:
    def __init__(self, settings: Settings, vector_store: Chroma) -> None:
        self.settings = settings
        self.vector_store = vector_store
        self.client = OpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
            timeout=90,
            max_retries=3,
        )

    def retrieve(self, question: str) -> tuple[SourceExcerpt, ...]:
        question = question.strip()
        if not question:
            raise ValueError("问题不能为空")

        raw_results = self.vector_store.similarity_search_with_relevance_scores(
            question,
            k=self.settings.top_k,
        )
        seen: set[str] = set()
        sources: list[SourceExcerpt] = []

        for document, score in raw_results:
            key = _document_key(document)
            if key in seen:
                continue
            seen.add(key)
            page_value = document.metadata.get("page_number")
            sources.append(
                SourceExcerpt(
                    number=len(sources) + 1,
                    source=str(document.metadata.get("source", "未知来源")),
                    file_name=str(
                        document.metadata.get("file_name", "未知文件")
                    ),
                    page_number=int(page_value) if page_value is not None else None,
                    content=document.page_content,
                    score=max(0.0, min(1.0, float(score))),
                    chunk_id=key,
                )
            )
        return tuple(sources)

    def _build_context(self, sources: tuple[SourceExcerpt, ...]) -> str:
        blocks: list[str] = []
        current_size = 0
        for source in sources:
            page = f"第 {source.page_number} 页" if source.page_number else "全文"
            block = (
                f"[资料{source.number}]\n"
                f"文件：{source.file_name}\n"
                f"位置：{page}\n"
                f"原文：{source.content.strip()}"
            )
            if blocks and current_size + len(block) > self.settings.max_context_chars:
                break
            blocks.append(block)
            current_size += len(block)
        return "\n\n".join(blocks)

    def answer(self, question: str) -> RAGResponse:
        question = question.strip()
        sources = self.retrieve(question)
        if not sources:
            return RAGResponse(
                question=question,
                answer="现有资料不足以回答：没有检索到相关原文片段。",
                sources=(),
            )

        context = self._build_context(sources)
        user_prompt = f"""请根据下列企业资料回答问题。

问题：{question}

检索资料：
{context}

请确保结论可追溯，并在对应句子末尾使用 [资料N] 标注来源。"""

        completion = self.client.chat.completions.create(
            model=self.settings.chat_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.settings.temperature,
            max_tokens=1800,
        )
        answer = completion.choices[0].message.content or "模型未返回答案"
        usage = completion.usage
        return RAGResponse(
            question=question,
            answer=answer.strip(),
            sources=sources,
            input_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
            output_tokens=getattr(usage, "completion_tokens", None)
            if usage
            else None,
        )


# 保留旧名称，避免命令行索引模式的现有引用失效。
MarketingRAGService = DocumentRAGService
