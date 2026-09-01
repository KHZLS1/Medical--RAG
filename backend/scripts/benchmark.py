"""分环节测速：定位入库瓶颈（GPU embedding vs Milvus 写入）"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.data_loader import iter_medical_documents
from app.vectorstore import get_embedder

# ===== 1. 加载 1000 条数据 =====
print("加载 1000 条数据...")
docs = []
for doc in iter_medical_documents(limit_per_file=200):
    docs.append(doc)
    if len(docs) >= 1000:
        break
texts = [d.page_content for d in docs]
print(f"实际加载 {len(docs)} 条\n")

# ===== 2. 首批 embedding（含模型加载 + GPU 预热，不计入） =====
embedder = get_embedder()
print("首批 embedding（含模型加载+GPU预热，耗时忽略）...")
t0 = time.time()
embedder.embed_documents(texts[:32])
print(f"  首批耗时 {time.time()-t0:.1f}s\n")

# ===== 3. 正式测纯 embedding 速度 =====
rest = texts[32:]
print(f"正式测速 {len(rest)} 条（纯 GPU embedding，不写 Milvus）...")
t0 = time.time()
embedder.embed_documents(rest)
dt = time.time() - t0
speed = len(rest) / dt
print(f"  耗时 {dt:.1f}s → {speed:.1f} 条/秒\n")

# ===== 4. 再测一轮（验证稳态） =====
t0 = time.time()
embedder.embed_documents(rest)
dt2 = time.time() - t0
speed2 = len(rest) / dt2
print(f"  第二轮耗时 {dt2:.1f}s → {speed2:.1f} 条/秒\n")

# ===== 5. 结论 =====
print("=" * 50)
print("诊断结论:")
final = max(speed, speed2)
print(f"  纯 embedding 速度: {final:.0f} 条/秒")
print(f"  整体入库速度:     65 条/秒")
if final > 150:
    gap = final / 65
    print(f"  → embedding 比 整体入库快 {gap:.1f} 倍")
    print("  → 瓶颈在 Milvus 写入或 LangChain 封装层")
else:
    print("  → 瓶颈就在 GPU embedding 本身")
    print("  → 可能原因: 外接显卡带宽 / 未用 fp16 / 驱动问题")
