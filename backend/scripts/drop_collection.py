"""清空 Milvus 中的 medical_qa 集合（全量入库前使用，避免与测试数据重复）

用法:
  python scripts/drop_collection.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pymilvus import connections, utility

from app.config import settings

connections.connect(uri=settings.milvus_uri)

name = settings.milvus_collection
if utility.has_collection(name):
    utility.drop_collection(name)
    print(f"[完成] 集合 '{name}' 已删除")
else:
    print(f"[跳过] 集合 '{name}' 不存在")

# 验证
print(f"当前集合还存在: {utility.has_collection(name)}")
print("下次运行 ingest.py 时会自动重建集合并写入数据")
