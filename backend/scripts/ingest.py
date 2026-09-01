"""数据批量入库脚本

用法:
  # 全量入库 (79万条, 耗时较长)
  python scripts/ingest.py

  # 每个科室限量入库 1000 条 (Demo 快速跑通)
  python scripts/ingest.py --limit 1000

  # 只入指定科室
  python scripts/ingest.py --limit 500 --dept 内科

注意: 必须先启动 Milvus (docker compose up -d)
"""
import argparse
import sys
from pathlib import Path

# 把 backend 目录加入 sys.path 以便导入 app
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tqdm import tqdm

from app.data_loader import iter_medical_documents
from app.vectorstore import get_vectorstore


def main():
    parser = argparse.ArgumentParser(description="医疗数据入库 Milvus")
    parser.add_argument("--limit", type=int, default=None,
                        help="每个科室文件最多入库多少条 (默认全部)")
    parser.add_argument("--dept", type=str, default=None,
                        help="只入库指定科室 (按目录名匹配，如 '内科')")
    parser.add_argument("--batch", type=int, default=500,
                        help="每批写入 Milvus 的文档数")
    args = parser.parse_args()

    print("=" * 60)
    print("医疗问答数据入库 Milvus")
    print(f"  limit/file: {args.limit}")
    print(f"  科室过滤:   {args.dept or '全部'}")
    print(f"  批大小:     {args.batch}")
    print("=" * 60)

    # 先加载到内存流式处理避免占太多内存
    def filtered_iter():
        for doc in iter_medical_documents(limit_per_file=args.limit):
            if args.dept and args.dept not in doc.metadata.get("source", ""):
                continue
            yield doc

    vs = get_vectorstore()

    batch = []
    total = 0
    try:
        for doc in tqdm(filtered_iter(), desc="入库中", unit="条"):
            batch.append(doc)
            if len(batch) >= args.batch:
                vs.add_documents(batch)
                total += len(batch)
                batch.clear()
        # 写入剩余
        if batch:
            vs.add_documents(batch)
            total += len(batch)
    except KeyboardInterrupt:
        print(f"\n[已中断] 已写入 {total} 条")
        return

    print(f"\n[完成] 共写入 {total} 条文档到 Milvus")


if __name__ == "__main__":
    main()
