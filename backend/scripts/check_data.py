"""查看 Milvus 中已导入的数据

用法:
  python scripts/check_data.py              # 查看数据概况 + 样例
  python scripts/check_data.py --query 头痛  # 相似度检索测试
  python scripts/check_data.py --limit 5     # 指定样例条数
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pymilvus import connections, Collection, utility

from app.config import settings


def connect():
    connections.connect(uri=settings.milvus_uri)


def show_overview(limit: int = 3):
    """打印集合概况 + 样例数据"""
    connect()

    name = settings.milvus_collection
    if not utility.has_collection(name):
        print(f"集合 '{name}' 不存在，请先运行 ingest")
        return

    col = Collection(name)
    col.load()
    total = col.num_entities

    print("=" * 60)
    print(f"集合名:     {name}")
    print(f"数据条数:   {total}")
    print(f"字段信息:")
    for f in col.schema.fields:
        print(f"  - {f.name} ({f.dtype})")

    # 按科室统计（count 聚合查询，避免单次 query 超过 Milvus 16384 上限）
    print(f"\n科室分布:")
    departments = ["内科", "男科", "妇产科", "肿瘤科", "儿科", "外科"]
    for dept in departments:
        r = col.query(
            expr=f"department == '{dept}'",
            output_fields=["count(*)"],
        )
        cnt = r[0].get("count(*)", 0) if r else 0
        print(f"  {dept}: {cnt} 条")

    # 样例数据
    print(f"\n样例数据 (前 {limit} 条):")
    print("-" * 60)
    samples = col.query(
        expr="pk >= 0",
        output_fields=["text", "department", "title", "source"],
        limit=limit,
    )
    for i, s in enumerate(samples, 1):
        print(f"[{i}] 科室: {s.get('department', '')}")
        print(f"    标题: {s.get('title', '')}")
        print(f"    来源: {s.get('source', '')}")
        text = s.get("text", "")
        print(f"    内容: {text[:150]}{'...' if len(text) > 150 else ''}")
        print()


def search_query(query: str, k: int = 5):
    """相似度检索测试"""
    from app.vectorstore import get_vectorstore

    vs = get_vectorstore()

    # 直接用 LangChain 的 similarity_search_with_score
    results = vs.similarity_search_with_score(query, k=k)

    print("=" * 60)
    print(f"检索: '{query}'  (Top-{k})")
    print("=" * 60)
    for i, (doc, score) in enumerate(results, 1):
        meta = doc.metadata or {}
        text = doc.page_content
        print(f"[{i}] 相似度: {score:.4f}  科室: {meta.get('department', '')}")
        print(f"    标题: {meta.get('title', '')}")
        print(f"    内容: {text[:150]}{'...' if len(text) > 150 else ''}")
        print()


def main():
    parser = argparse.ArgumentParser(description="查看 Milvus 中的医疗问答数据")
    parser.add_argument("--query", type=str, default=None,
                        help="相似度检索测试 (如: '头痛怎么办')")
    parser.add_argument("--limit", type=int, default=3,
                        help="样例数据条数 (默认 3)")
    parser.add_argument("--k", type=int, default=5,
                        help="检索返回条数 (默认 5)")
    args = parser.parse_args()

    if args.query:
        search_query(args.query, k=args.k)
    else:
        show_overview(limit=args.limit)


if __name__ == "__main__":
    main()
