"""测试通过 Vite proxy 访问 chat 接口，确认 SSE 流式是否正常"""
import json
import urllib.request
import time

url = "http://127.0.0.1:5173/api/chat"
data = json.dumps({"question": "高血压能吃党参吗"}).encode("utf-8")

req = urllib.request.Request(
    url,
    data=data,
    headers={"Content-Type": "application/json"},
    method="POST"
)

print(f"请求 URL: {url}")
print("等待响应...")
start = time.time()

try:
    resp = urllib.request.urlopen(req, timeout=30)
    print(f"状态码: {resp.status}")
    print(f"Content-Type: {resp.headers.get('Content-Type')}")
    print()

    # 逐块读取，统计多久收到第一个 token
    first_token_time = None
    token_count = 0
    total_bytes = 0

    while True:
        chunk = resp.read(256)
        if not chunk:
            break
        total_bytes += len(chunk)
        text = chunk.decode("utf-8", errors="replace")
        # 查找第一个 token
        if first_token_time is None and '"type": "token"' in text and '"data": ""' not in text:
            first_token_time = time.time() - start

        # 统计 token 事件数
        token_count += text.count('"type": "token"')

        # 打印前几行
        if total_bytes < 500:
            print(text[:200], end="")
            if total_bytes >= 500:
                print("\n...(后续省略)...")

    print()
    print("=" * 50)
    print(f"总耗时: {time.time()-start:.2f}s")
    print(f"首 token 时间: {first_token_time:.2f}s" if first_token_time else "首 token 时间: 未检测到")
    print(f"收到 token 事件数: {token_count}")
    print(f"总字节数: {total_bytes}")

except urllib.error.HTTPError as e:
    print(f"HTTP 错误: {e.code} {e.reason}")
    print(f"响应体: {e.read().decode()[:500]}")
except Exception as e:
    print(f"错误: {type(e).__name__}: {e}")
