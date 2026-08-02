from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def _as_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


def _as_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value else default


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime configuration loaded from the project-level .env file."""

    project_dir: Path
    data_dir: Path
    index_dir: Path
    api_key: str
    base_url: str
    chat_model: str
    embedding_model: str
    embedding_dimensions: int
    top_k: int
    max_context_chars: int
    temperature: float
    collection_prefix: str = "marketing_knowledge"

    @classmethod
    def from_env(cls, project_dir: Path | None = None) -> "Settings":
        root = (project_dir or Path(__file__).resolve().parents[2]).resolve()
        load_dotenv(root / ".env", override=False)

        data_value = os.getenv("RAG_DATA_DIR", str(root / "data"))
        index_value = os.getenv("RAG_INDEX_DIR", str(root / "chroma_db"))

        data_dir = Path(data_value).expanduser()
        index_dir = Path(index_value).expanduser()
        if not data_dir.is_absolute():
            data_dir = root / data_dir
        if not index_dir.is_absolute():
            index_dir = root / index_dir

        return cls(
            project_dir=root,
            data_dir=data_dir.resolve(),
            index_dir=index_dir.resolve(),
            api_key=os.getenv("DASHSCOPE_API_KEY", "").strip(),
            base_url=os.getenv(
                "DASHSCOPE_BASE_URL",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            ).rstrip("/"),
            chat_model=os.getenv("RAG_CHAT_MODEL", "deepseek-v4-flash"),
            embedding_model=os.getenv(
                "RAG_EMBEDDING_MODEL", "text-embedding-v4"
            ),
            embedding_dimensions=_as_int("RAG_EMBEDDING_DIMENSIONS", 1024),
            top_k=_as_int("RAG_TOP_K", 6),
            max_context_chars=_as_int("RAG_MAX_CONTEXT_CHARS", 12_000),
            temperature=_as_float("RAG_TEMPERATURE", 0.2),
        )

    def validate(self) -> None:
        problems: list[str] = []
        if not self.api_key:
            problems.append(".env 中缺少 DASHSCOPE_API_KEY")
        if not self.base_url:
            problems.append(".env 中缺少 DASHSCOPE_BASE_URL")
        if not self.data_dir.exists():
            problems.append(f"资料目录不存在：{self.data_dir}")
        if not self.data_dir.is_dir():
            problems.append(f"资料路径不是文件夹：{self.data_dir}")
        if problems:
            raise ValueError("；".join(problems))

    def validate_api(self) -> None:
        problems: list[str] = []
        if not self.api_key:
            problems.append(".env 中缺少 DASHSCOPE_API_KEY")
        if not self.base_url:
            problems.append(".env 中缺少 DASHSCOPE_BASE_URL")
        if problems:
            raise ValueError("；".join(problems))
