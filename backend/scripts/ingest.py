"""数据批量入库脚本（Milvus 内置 BM25 版：自动建库 + 入库）

用法:
  # 冒烟测试（每科 500 条）
  python scripts/ingest.py --cleaned --limit 500

  # 全量入库（清洗后 JSON，60.5 万条）
  python scripts/ingest.py --cleaned

  # 只入指定科室
  python scripts/ingest.py --cleaned --limit 500 --dept 内科

注意:
  - 必须先启动 Milvus 2.5+ (docker compose up -d)
  - 默认每次运行都会删除并重建 medical_qa collection（含 BM25 Function）
  - sparse 字段由 Milvus 服务端自动生成，无需客户端计算，也不再生成 pkl 缓存
"""

import argparse
import sys
from pathlib import Path

# 把 backend 目录加入 sys.path 以便导入 app
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tqdm import tqdm
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    Function,
    FunctionType,
    connections,
    utility,
)

from app.config import settings
from app.data_loader import iter_medical_documents, iter_cleaned_documents
from app.vectorstore import get_embedder

EMBEDDING_DIM = 1024  # BAAI/bge-large-zh-v1.5 输出维度


def ensure_collection(recreate: bool = True) -> Collection:
    """确保 medical_qa 存在（新 schema：dense + BM25 sparse + metadata）"""
    connections.connect(alias="default", uri=settings.milvus_uri)
    name = settings.milvus_collection

    if utility.has_collection(name):
        if not recreate:
            return Collection(name)
        print(f"[建库] 删除旧 collection: {name}")
        Collection(name).drop()

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
        FieldSchema(
            name="content",
            dtype=DataType.VARCHAR,
            max_length=65535,
            enable_analyzer=True,
            analyzer_params={"type": "chinese"},
        ),
        FieldSchema(name="sparse", dtype=DataType.SPARSE_FLOAT_VECTOR),
        FieldSchema(name="metadata", dtype=DataType.JSON),
    ]
    schema = CollectionSchema(fields, description="医疗问答: dense(bge) + BM25(sparse)")
    schema.add_function(
        Function(
            name="text_bm25",
            function_type=FunctionType.BM25,
            input_field_names=["content"],
            output_field_names=["sparse"],
        )
    )

    coll = Collection(name, schema)
    coll.create_index("embedding", {"index_type": "AUTOINDEX", "metric_type": "IP"})
    coll.create_index(
        "sparse",
        {
            "index_type": "SPARSE_INVERTED_INDEX",
            "metric_type": "BM25",
            "params": {"bm25_k1": 1.2, "bm25_b": 0.75},
        },
    )
    print(f"[建库] collection {name} 创建成功（BM25 Function 已启用）")
    return coll


def insert_batch(coll: Collection, embedder, docs: list) -> int:
    """批嵌入 + 批量写入（embedding 走 GPU 批处理）"""
    texts = [d.page_content for d in docs]
    vecs = embedder.embed_documents(texts)
    rows = [
        {
            "content": d.page_content,
            "embedding": vec,
            "metadata": dict(d.metadata or {}),
        }
        for d, vec in zip(docs, vecs)
    ]
    coll.insert(rows)
    return len(rows)


def main():
    parser = argparse.ArgumentParser(
        description="医疗数据入库 Milvus (自动建库 + BM25 内置)"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="每个科室文件最多入库多少条 (默认全部)"
    )
    parser.add_argument(
        "--dept",
        type=str,
        default=None,
        help="只入库指定科室 (按目录名匹配，如 '内科')",
    )
    parser.add_argument("--batch", type=int, default=500, help="每批条数（嵌入+写入）")
    parser.add_argument(
        "--cleaned",
        action="store_true",
        help="使用清洗后的 JSON 数据 (backend/data/by_department/)",
    )
    parser.add_argument(
        "--no-recreate", action="store_true", help="不重建 collection，直接追加写入"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("医疗问答数据入库 Milvus (BM25 内置版)")
    print(f"  数据来源:   {'清洗后JSON' if args.cleaned else '原始CSV'}")
    print(f"  limit/file: {args.limit}")
    print(f"  科室过滤:   {args.dept or '全部'}")
    print(f"  批大小:     {args.batch}")
    print("=" * 60)

    coll = ensure_collection(recreate=not args.no_recreate)

    # 流式加载避免占太多内存
    def filtered_iter():
        if args.cleaned:
            doc_iter = iter_cleaned_documents(limit_per_file=args.limit)
        else:
            doc_iter = iter_medical_documents(limit_per_file=args.limit)
        for doc in doc_iter:
            if args.dept and args.dept not in doc.metadata.get("department", ""):
                continue
            yield doc

    embedder = get_embedder()  # 加载 bge 模型（首次较慢）

    batch = []
    total = 0
    try:
        for doc in tqdm(filtered_iter(), desc="入库中", unit="条"):
            batch.append(doc)
            if len(batch) >= args.batch:
                total += insert_batch(coll, embedder, batch)
                batch.clear()
        # 写入剩余
        if batch:
            total += insert_batch(coll, embedder, batch)
    except KeyboardInterrupt:
        print(f"\n[已中断] 已写入 {total} 条，重新运行即可自动重建后全量重入")
        return

    print(f"\n[完成] 共写入 {total} 条文档到 Milvus")
    print("[完成] BM25 索引由 Milvus 服务端维护，无需生成 pkl 缓存")


if __name__ == "__main__":
    main()
