"""逐步测试后端各环节，定位卡住的原因"""
import sys
import json
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# --- 测试 1: health 接口 ---
print("=" * 50)
print("测试 1: 调用 /api/health")
import urllib.request
import urllib.error
try:
    start = time.time()
    resp = urllib.request.urlopen("http://127.0.0.1:8000/api/health", timeout=15)
    print(f"  状态: {resp.status}, 耗时: {time.time()-start:.2f}s")
    print(f"  响应: {resp.read().decode()[:200]}")
except urllib.error.HTTPError as e:
    print(f"  HTTP 错误: {e.code} {e.reason}")
except Exception as e:
    print(f"  失败: {type(e).__name__}: {e}")

# --- 测试 2: ingest 接口 ---
print("\n" + "=" * 50)
print("测试 2: 调用 /api/ingest (limit=10)")
try:
    data = json.dumps({"limit_per_file": 10}).encode("utf-8")
    req = urllib.request.Request(
        "http://127.0.0.1:8000/api/ingest",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    start = time.time()
    resp = urllib.request.urlopen(req, timeout=60)
    print(f"  状态: {resp.status}, 耗时: {time.time()-start:.2f}s")
    print(f"  响应: {resp.read().decode()[:300]}")
except urllib.error.HTTPError as e:
    print(f"  HTTP 错误: {e.code} {e.reason}")
    print(f"  body: {e.read().decode()[:300]}")
except Exception as e:
    print(f"  失败: {type(e).__name__}: {e}")

# --- 测试 3: chat 接口 ---
print("\n" + "=" * 50)
print("测试 3: 调用 /api/chat (小孩发烧)")
try:
    data = json.dumps({"question": "小孩发烧39度怎么办"}).encode("utf-8")
    req = urllib.request.Request(
        "http://127.0.0.1:8000/api/chat",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    start = time.time()
    resp = urllib.request.urlopen(req, timeout=30)
    print(f"  状态: {resp.status}, 耗时: {time.time()-start:.2f}s")
    # 读取前几行 SSE 数据
    body = resp.read(500).decode("utf-8", errors="replace")
    print(f"  前500字节: {body[:300]}")
except urllib.error.HTTPError as e:
    print(f"  HTTP 错误: {e.code} {e.reason}")
    print(f"  body: {e.read().decode()[:300]}")
except Exception as e:
    print(f"  失败: {type(e).__name__}: {e}")

print("\n" + "=" * 50)
print("测试完成")
