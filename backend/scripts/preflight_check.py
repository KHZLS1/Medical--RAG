"""入库前置检查（只读，不做任何修改）

全量重入库前跑一遍，确认环境、数据、依赖、资源都就绪。

用法:
  cd backend
  python scripts/preflight_check.py
"""
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OK, WARN, FAIL = "[ OK ]", "[WARN]", "[FAIL]"
issues: list[str] = []


def line(status: str, msg: str) -> None:
    print(f"{status} {msg}")
    if status is FAIL:
        issues.append(msg)


def section(title: str) -> None:
    print(f"\n{'=' * 62}\n{title}\n{'=' * 62}")


# ---------- 1. 配置 ----------
section("1. 配置 (.env)")
try:
    from app.config import settings

    print(f"  Milvus      : {settings.milvus_uri}")
    print(f"  Collection  : {settings.milvus_collection}")
    print(f"  Embedding   : {settings.embedding_model}")
    print(f"  设备        : {settings.embedding_device}")
    if settings.embedding_device != "cuda":
        line(WARN, f"EMBEDDING_DEVICE={settings.embedding_device}（CPU 会慢 20~40 倍）")
    else:
        line(OK, "向量化设备 = cuda")
    if not settings.database_url:
        line(WARN, "DATABASE_URL 未配置（只影响 MySQL 会话/文档表，不影响入库）")
    else:
        line(OK, "DATABASE_URL 已配置")
except Exception as e:
    line(FAIL, f"配置加载失败: {e}")


# ---------- 2. Milvus ----------
section("2. Milvus 服务与现有数据")
try:
    host, port = settings.milvus_host, settings.milvus_port
    with socket.create_connection((host, port), timeout=5):
        line(OK, f"{host}:{port} 可连接")
except Exception as e:
    line(FAIL, f"Milvus 不可达 ({host}:{port}): {e}\n         请先执行: docker compose up -d")

try:
    from pymilvus import Collection, MilvusClient, utility

    client = MilvusClient(uri=settings.milvus_uri)
    name = settings.milvus_collection
    if client.has_collection(name):
        stats = client.get_collection_stats(name)
        rows = int(stats.get("row_count", 0))
        desc = client.describe_collection(name)
        dim = None
        for f in desc.get("fields", []):
            if f.get("name") == "vector":
                dim = f.get("params", {}).get("dim")
        print(f"  现有集合 '{name}': {rows:,} 条向量, dim={dim}")
        line(OK, f"待删除的旧数据规模: {rows:,} 条")
        idx = client.list_indexes(name)
        print(f"  索引: {idx}")
    else:
        line(WARN, f"集合 '{name}' 不存在（将直接新建）")
except Exception as e:
    line(WARN, f"无法读取集合信息: {e}")


# ---------- 3. 清洗数据 ----------
section("3. 清洗后数据 (data/by_department)")
import json

data_dir = settings.cleaned_data_dir_resolved
if not data_dir.exists():
    line(FAIL, f"目录不存在: {data_dir}")
else:
    files = sorted(data_dir.glob("*.json"))
    if not files:
        line(FAIL, f"目录下没有 JSON 文件: {data_dir}")
    else:
        total = 0
        chars = 0
        for f in files:
            try:
                with open(f, encoding="utf-8") as fh:
                    recs = json.load(fh)
                total += len(recs)
                for r in recs:
                    chars += len((r.get("instruction") or "") + (r.get("input") or "") + (r.get("output") or ""))
            except Exception as e:
                line(FAIL, f"{f.name} 解析失败: {e}")
        line(OK, f"{len(files)} 个科室文件 / {total:,} 条记录 / {chars / 1e6:.1f}M 字符")
        print(f"  预计入库向量数: 约 {total:,} 条（不切分）")


# ---------- 4. 检索 schema（服务端 BM25） ----------
section("4. 检索 schema（服务端 BM25，无需 pkl 缓存）")
try:
    from pymilvus import MilvusClient

    client = MilvusClient(uri=settings.milvus_uri)
    name = settings.milvus_collection
    if not client.has_collection(name):
        line(WARN, f"集合 '{name}' 不存在，请先入库: python scripts/ingest.py --cleaned")
    else:
        desc = client.describe_collection(name)
        field_names = {f.get("name") for f in desc.get("fields", [])}
        need = {"embedding", "content", "sparse", "metadata"}
        missing = need - field_names
        if missing:
            line(FAIL, f"集合缺少字段 {sorted(missing)} —— 检索会失败，需用 ingest.py 重建")
        else:
            line(OK, "字段齐全: embedding / content / sparse / metadata")
        funcs = desc.get("functions") or []
        if funcs:
            for fn in funcs:
                print(f"  BM25 Function: {fn.get('name')} {fn.get('input_field_names')} -> {fn.get('output_field_names')}")
        else:
            line(FAIL, "集合没有 BM25 Function，sparse 字段不会自动生成（需重建 collection）")
        print("  检索方式: pymilvus hybrid_search（BM25 + 稠密向量 + WeightedRanker）")
except Exception as e:
    line(WARN, f"无法校验检索 schema: {e}")


# ---------- 5. 依赖 ----------
section("5. Python 依赖")
deps = {
    "torch": "深度学习框架",
    "sentence_transformers": "Embedding 推理",
    "FlagEmbedding": "Reranker 精排",
    "pymilvus": "Milvus 客户端（检索 + 写入）",
    "langchain_huggingface": "HuggingFace Embedding 包装",
    "jieba": "评估脚本分词",
}
for mod, desc in deps.items():
    try:
        __import__(mod)
        line(OK, f"{mod} ({desc})")
    except Exception as e:
        line(FAIL, f"{mod} 缺失 ({desc}): {type(e).__name__}")

try:
    import torch

    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  显存: 空闲 {free / 1024**3:.2f}GB / 共 {total / 1024**3:.2f}GB")
        if free / 1024**3 < 2.5:
            line(WARN, "空闲显存 < 2.5GB，建议关闭其他占用 GPU 的程序")
        else:
            line(OK, "显存充足")
        dtype = getattr(torch.cuda, "get_device_properties", None)
    else:
        line(FAIL, "CUDA 不可用")
except Exception:
    pass

try:
    from pathlib import Path as _P

    cache = _P.home() / ".cache" / "huggingface" / "hub" / "models--BAAI--bge-large-zh-v1.5"
    if cache.exists():
        line(OK, "bge-large-zh-v1.5 模型已缓存（无需联网下载）")
    else:
        line(WARN, "模型未在本地缓存，首次运行需联网下载约 1.3GB")
except Exception:
    pass


# ---------- 6. 资源 ----------
section("6. 资源")
import shutil

try:
    free_gb = shutil.disk_usage(ROOT).free / 1024**3
    if free_gb < 10:
        line(WARN, f"C 盘剩余 {free_gb:.1f}GB，建议 ≥10GB")
    else:
        line(OK, f"C 盘剩余 {free_gb:.1f}GB")
except Exception:
    pass

# 后端服务是否占用
for p in (8000, 5173):
    with socket.socket() as s:
        s.settimeout(1)
        if s.connect_ex(("127.0.0.1", p)) == 0:
            line(WARN, f"端口 {p} 有服务在运行 —— 入库前建议停止后端/前端，避免检索到半成品数据")
        else:
            line(OK, f"端口 {p} 空闲")


# ---------- 汇总 ----------
section("结论")
if issues:
    print(f"发现 {len(issues)} 个阻塞项:")
    for i in issues:
        print(f"  - {i}")
else:
    print("无阻塞项，可以开始入库。")
