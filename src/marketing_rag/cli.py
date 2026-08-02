from __future__ import annotations

import argparse
import sys

from .runtime import create_runtime


def build_main() -> None:
    parser = argparse.ArgumentParser(description="构建企业营销知识库索引")
    parser.add_argument("--force", action="store_true", help="强制重建当前索引")
    args = parser.parse_args()

    runtime = create_runtime(force_rebuild=args.force)
    action = "已创建" if runtime.index.created else "已复用"
    print(f"{action}向量索引：{runtime.index.collection_name}")
    print(f"资料文件：{runtime.index.source_count}")
    print(f"文本片段：{runtime.index.chunk_count}")
    if runtime.report.errors:
        print("加载失败：")
        for source, message in runtime.report.errors:
            print(f"- {source}: {message}")


def ask_main() -> None:
    parser = argparse.ArgumentParser(description="向企业营销知识库提问")
    parser.add_argument("question", nargs="+", help="要查询的问题")
    args = parser.parse_args()
    question = " ".join(args.question)

    runtime = create_runtime()
    response = runtime.service.answer(question)
    print(response.answer)
    print("\n来源：")
    for source in response.sources:
        print(f"- {source.citation_label}（相关度 {source.score:.1%}）")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        sys.argv.pop(1)
        build_main()
    else:
        ask_main()

