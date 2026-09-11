"""【已废弃，请勿使用】仅生成 BM25 文档缓存（不写入 Milvus）

⚠️ 本脚本已废弃：混合检索已改为 Milvus 服务端 BM25（sparse 字段由 BM25
   Function 自动生成），检索链路不再读取 bm25_docs.pkl。
   保留此文件仅为历史参考，重新生成缓存不会对检索产生任何效果。
   检索实现见 app/vectorstore.py 的 hybrid_search()。

原用途：当 bm25_docs.pkl 丢失或损坏时使用此脚本快速重建。

用法:
  # 使用清洗后的 JSON 数据（by_department/），全量
  python scripts/gen_bm25_cache.py --cleaned

  # 限量 1000 条/文件（快速测试）
  python scripts/gen_bm25_cache.py --cleaned --limit 1000

  # 使用原始 CSV 数据
  python scripts/gen_bm25_cache.py
"""
import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.data_loader import iter_medical_documents, iter_cleaned_documents


def main():
    parser = argparse.ArgumentParser(description="生成 BM25 文档缓存")
    parser.add_argument("--limit", type=int, default=None,
                        help="每个文件最多加载多少条 (默认全部)")
    parser.add_argument("--cleaned", action="store_true", default=True,
                        help="使用清洗后的 JSON 数据 (by_department/)")
    parser.add_argument("--dept", type=str, default=None,
                        help="只加载指定科室")
    args = parser.parse_args()

    print("=" * 60)
    print("生成 BM25 文档缓存")
    print(f"  数据来源:   {'清洗后JSON' if args.cleaned else '原始CSV'}")
    print(f"  limit/file: {args.limit}")
    print(f"  科室过滤:   {args.dept or '全部'}")
    print("=" * 60)

    def filtered_iter():
        if args.cleaned:
            doc_iter = iter_cleaned_documents(limit_per_file=args.limit)
        else:
            doc_iter = iter_medical_documents(limit_per_file=args.limit)
        for doc in doc_iter:
            if args.dept and args.dept not in doc.metadata.get("department", ""):
                continue
            yield doc

    print("[BM25] 正在收集文档...")
    all_docs = list(filtered_iter())
    print(f"[BM25] 共 {len(all_docs)} 条文档")

    bm25_cache_path = Path(__file__).resolve().parent.parent / "data" / "bm25_docs.pkl"
    bm25_cache_path.parent.mkdir(parents=True, exist_ok=True)

    with open(bm25_cache_path, "wb") as f:
        pickle.dump(all_docs, f)

    print(f"[BM25] 已保存缓存到 {bm25_cache_path}")
    print("[完成] 请重启后端服务")


if __name__ == "__main__":
    main()
