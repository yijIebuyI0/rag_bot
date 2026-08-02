from __future__ import annotations

import html
import sys
from dataclasses import asdict
from pathlib import Path

import streamlit as st

# 让入口脚本在复制到新目录后，也能直接找到 src/marketing_rag。
PROJECT_DIR = Path(__file__).resolve().parent
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from marketing_rag.rag import SourceExcerpt
from marketing_rag.runtime import UploadRuntime, create_upload_runtime


st.set_page_config(
    page_title="RAG 文档问答 Bot",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)


APP_CSS = """
<style>
:root {
  --ink: #172033;
  --muted: #667085;
  --blue: #2563eb;
  --green: #0f9f73;
  --line: #dbe3ef;
  --surface: #ffffff;
  --canvas: #f7f9fc;
}

.stApp { background: var(--canvas); }
html, body, [class*="css"] {
  font-family: "Segoe UI", "Microsoft YaHei", "PingFang SC", sans-serif;
  color: var(--ink);
}
.block-container { max-width: 980px; padding-top: 2.2rem; }

.bot-header {
  padding-bottom: 1.1rem;
  border-bottom: 1px solid var(--line);
  margin-bottom: 1.2rem;
}
.bot-title {
  font-size: 2rem;
  font-weight: 750;
  letter-spacing: -.025em;
  margin: 0 0 .35rem;
}
.bot-subtitle { color: var(--muted); line-height: 1.65; }

.pipeline {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  border: 1px solid var(--line);
  background: var(--surface);
  margin: 1rem 0 1.5rem;
}
.pipeline span {
  padding: .7rem .8rem;
  color: var(--muted);
  font-size: .78rem;
  border-right: 1px solid var(--line);
  text-align: center;
}
.pipeline span:last-child { border-right: 0; }
.pipeline b { color: var(--blue); margin-right: .28rem; }

.empty-state {
  border: 1px dashed #b8c5d9;
  background: rgba(255,255,255,.68);
  padding: 2rem;
  text-align: center;
  color: var(--muted);
}
.empty-state strong { display: block; color: var(--ink); margin-bottom: .4rem; }

.source-card {
  border-left: 3px solid var(--blue);
  padding: .6rem .8rem;
  margin-bottom: .7rem;
  background: #f8faff;
}
.source-card strong { color: var(--ink); }
.source-meta { color: var(--muted); font-size: .78rem; margin-top: .2rem; }

[data-testid="stSidebar"] {
  background: #f0f4fa;
  border-right: 1px solid var(--line);
}
[data-testid="stChatMessage"] {
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 8px;
}
.stButton > button { border-radius: 6px; }
.stButton > button[kind="primary"] { font-weight: 700; }

@media (max-width: 700px) {
  .pipeline { grid-template-columns: repeat(2, 1fr); }
  .pipeline span:nth-child(2) { border-right: 0; }
  .pipeline span:nth-child(-n+2) { border-bottom: 1px solid var(--line); }
}
</style>
"""

st.markdown(APP_CSS, unsafe_allow_html=True)


def render_header() -> None:
    st.markdown(
        """
        <div class="bot-header">
          <div class="bot-title">RAG 文档问答 Bot</div>
          <div class="bot-subtitle">
            上传自己的文档，构建本地向量知识库，然后用中文提问。每条答案都可以回到原文核对。
          </div>
        </div>
        <div class="pipeline" aria-label="RAG 处理流程">
          <span><b>01</b>文档切分</span>
          <span><b>02</b>生成向量</span>
          <span><b>03</b>语义检索</span>
          <span><b>04</b>引用回答</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sources(raw_sources: list[dict[str, object]]) -> None:
    if not raw_sources:
        return
    st.caption("回答依据")
    for raw in raw_sources:
        source = SourceExcerpt(**raw)
        page = f"第 {source.page_number} 页" if source.page_number else "全文"
        with st.expander(f"来源 {source.number} · {source.file_name} · {page}"):
            st.markdown(
                f"""
                <div class="source-card">
                  <strong>{html.escape(source.file_name)}</strong>
                  <div class="source-meta">{page} · 语义相关度 {source.score:.1%}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.write(source.content)


def current_file_signature(files: list[object]) -> tuple[tuple[str, int], ...]:
    return tuple((file.name, file.size) for file in files)


render_header()

with st.sidebar:
    st.header("上传文档")
    uploaded_files = st.file_uploader(
        "选择 PDF、Markdown 或 TXT",
        type=["pdf", "md", "txt"],
        accept_multiple_files=True,
        help="PDF 需要包含可复制的文本；扫描版 PDF 暂不支持。",
    )

    if uploaded_files:
        st.caption(f"已选择 {len(uploaded_files)} 个文件")
        for uploaded in uploaded_files:
            st.write(f"• {uploaded.name}")

    build_clicked = st.button(
        "构建知识库",
        type="primary",
        use_container_width=True,
        disabled=not uploaded_files,
    )

    if build_clicked:
        with st.spinner("正在读取、切分并生成向量……"):
            try:
                runtime = create_upload_runtime(list(uploaded_files))
            except Exception as exc:
                st.error(f"构建失败：{exc}")
            else:
                st.session_state.rag_runtime = runtime
                st.session_state.file_signature = current_file_signature(
                    list(uploaded_files)
                )
                st.session_state.messages = []
                st.success("知识库已就绪")

    runtime: UploadRuntime | None = st.session_state.get("rag_runtime")
    if runtime:
        st.divider()
        st.caption("当前知识库")
        st.write(f"**{runtime.report.successful_file_count}** 个文档")
        st.write(f"**{runtime.chunk_count}** 个文本片段")
        st.caption(
            f"{runtime.settings.embedding_model} · {runtime.settings.chat_model}"
        )
        if st.button("清空对话", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        if runtime.report.errors:
            with st.expander(f"{len(runtime.report.errors)} 个文件未读取"):
                for file_name, message in runtime.report.errors:
                    st.write(f"{file_name}：{message}")

    st.divider()
    st.caption("文档内容会发送给百炼向量模型；检索片段会发送给回答模型。")

runtime = st.session_state.get("rag_runtime")
if "messages" not in st.session_state:
    st.session_state.messages = []

if not runtime:
    st.markdown(
        """
        <div class="empty-state">
          <strong>先从左侧上传文档</strong>
          支持同时上传多个 PDF、MD 和 TXT 文件。构建完成后，聊天输入框会自动出现。
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

if uploaded_files:
    selected_signature = current_file_signature(list(uploaded_files))
    if selected_signature != st.session_state.get("file_signature"):
        st.info("上传文件已经变化；点击“构建知识库”后，新文件才会用于回答。")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_sources(message.get("sources", []))

question = st.chat_input("针对已上传文档提问")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("正在检索文档……"):
            try:
                response = runtime.service.answer(question)
            except Exception as exc:
                st.error(f"回答失败：{exc}")
            else:
                st.markdown(response.answer)
                serialized_sources = [asdict(source) for source in response.sources]
                render_sources(serialized_sources)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": response.answer,
                        "sources": serialized_sources,
                    }
                )
