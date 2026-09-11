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

from app.config import settings


def show_overview(limit: int = 3):
    """打印集合概况 + 样例数据"""
    from pymilvus import MilvusClient

    name = settings.milvus_collection
    client = MilvusClient(uri=settings.milvus_uri)
    if not client.has_collection(name):
        print(f"集合 '{name}' 不存在，请先运行 ingest")
        return

    client.load_collection(name)
    stats = client.get_collection_stats(name)
    total = int(stats.get("row_count", 0)) if stats else 0
    desc = client.describe_collection(name)

    print("=" * 60)
    print(f"集合名:     {name}")
    print(f"数据条数:   {total}")
    print(f"字段信息:")
    for f in desc.get("fields", []):
        print(f"  - {f.get('name')} ({f.get('type')})")
    if desc.get("functions"):
        for fn in desc["functions"]:
            print(f"  BM25 Function: {fn.get('name')} "
                  f"{fn.get('input_field_names')} -> {fn.get('output_field_names')}")

    # 按科室统计（metadata 是 JSON 字段，用 JSON path 过滤）
    print(f"\n科室分布:")
    for dept in ["内科", "男科", "妇产科", "肿瘤科", "儿科", "外科"]:
        r = client.query(
            collection_name=name,
            filter=f'metadata["department"] == "{dept}"',
            output_fields=["count(*)"],
        )
        cnt = r[0].get("count(*)", 0) if r else 0
        print(f"  {dept}: {cnt} 条")

    # 样例数据
    print(f"\n样例数据 (前 {limit} 条):")
    print("-" * 60)
    samples = client.query(
        collection_name=name,
        filter="id >= 0",
        output_fields=["content", "metadata"],
        limit=limit,
    )
    for i, s in enumerate(samples, 1):
        meta = s.get("metadata") or {}
        print(f"[{i}] 科室: {meta.get('department', '')}")
        print(f"    标题: {meta.get('title', '')}")
        print(f"    来源: {meta.get('source', '')}")
        text = s.get("content", "")
        print(f"    内容: {text[:150]}{'...' if len(text) > 150 else ''}")
        print()


def search_query(query: str, k: int = 5, mode: str = "hybrid"):
    """检索测试：走生产同一条链路（服务端 BM25 + 稠密向量）"""
    from app.vectorstore import hybrid_search

    results = hybrid_search(query, k=k, mode=mode)

    print("=" * 60)
    print(f"检索: '{query}'  (mode={mode}, Top-{k})")
    print("=" * 60)
    for i, doc in enumerate(results, 1):
        meta = doc.metadata or {}
        print(f"[{i}] 分数: {meta.get('retrieval_score', 0):.4f}  科室: {meta.get('department', '')}")
        print(f"    标题: {meta.get('title', '')}")
        print(f"    内容: {doc.page_content[:150]}{'...' if len(doc.page_content) > 150 else ''}")
        print()


def main():
    parser = argparse.ArgumentParser(description="查看 Milvus 中的医疗问答数据")
    parser.add_argument("--query", type=str, default=None,
                        help="检索测试 (如: '头痛怎么办')")
    parser.add_argument("--mode", type=str, default="hybrid",
                        choices=["hybrid", "vector", "bm25"],
                        help="检索模式 (默认 hybrid: BM25 + 向量)")
    parser.add_argument("--limit", type=int, default=3,
                        help="样例数据条数 (默认 3)")
    parser.add_argument("--k", type=int, default=5,
                        help="检索返回条数 (默认 5)")
    args = parser.parse_args()

    if args.query:
        search_query(args.query, k=args.k, mode=args.mode)
    else:
        show_overview(limit=args.limit)


if __name__ == "__main__":
    main()
