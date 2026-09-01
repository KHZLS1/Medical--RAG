"""文档上传接口测试

测试三个接口:
  1. POST   /api/upload          上传文档
  2. GET    /api/uploads         查看已上传文档列表
  3. DELETE /api/uploads/{name}   删除已上传文档

用法:
  cd backend
  python ../test/3.upload-test.py
"""
import sys
from pathlib import Path

import requests

BASE_URL = "http://localhost:8000"
TEST_DIR = Path(__file__).resolve().parent


def test_upload():
    """测试上传接口"""
    print("=" * 50)
    print("[1] 测试上传接口  POST /api/upload")
    print("=" * 50)

    # 创建一个临时测试文件
    test_file = TEST_DIR / "upload_test_sample.txt"
    test_file.write_text(
        "高血压常见症状包括头痛、头晕、心悸等。\n"
        "建议定期监测血压，限制钠盐摄入，保持适度运动。\n"
        "如血压持续升高，请及时就医。",
        encoding="utf-8",
    )

    with open(test_file, "rb") as f:
        resp = requests.post(
            f"{BASE_URL}/api/upload",
            files={"file": (test_file.name, f, "text/plain")},
        )

    print(f"  状态码: {resp.status_code}")
    print(f"  响应:   {resp.json()}")
    return resp.status_code == 200


def test_list():
    """测试查看列表接口"""
    print("\n" + "=" * 50)
    print("[2] 测试查看列表  GET /api/uploads")
    print("=" * 50)

    resp = requests.get(f"{BASE_URL}/api/uploads")

    print(f"  状态码: {resp.status_code}")
    data = resp.json()
    files = data.get("files", [])
    print(f"  已上传文件数: {len(files)}")
    for f in files:
        print(f"    - {f['filename']}  ({f['size_mb']}MB, {f['extension']})")

    return resp.status_code == 200


def test_delete():
    """测试删除接口"""
    print("\n" + "=" * 50)
    print("[3] 测试删除接口  DELETE /api/uploads/{filename}")
    print("=" * 50)

    filename = "upload_test_sample.txt"
    resp = requests.delete(f"{BASE_URL}/api/uploads/{filename}")

    print(f"  状态码: {resp.status_code}")
    print(f"  响应:   {resp.json()}")

    # 再查一次确认已删除
    resp2 = requests.get(f"{BASE_URL}/api/uploads")
    remaining = [f["filename"] for f in resp2.json().get("files", [])]
    if filename in remaining:
        print(f"  [失败] 文件 {filename} 仍然存在")
        return False
    else:
        print(f"  [确认] 文件 {filename} 已被删除")
        return resp.status_code == 200


def test_upload_invalid():
    """测试上传非法文件类型（应返回 400）"""
    print("\n" + "=" * 50)
    print("[4] 测试非法文件类型上传 (应返回 400)")
    print("=" * 50)

    test_file = TEST_DIR / "bad_file.exe"
    test_file.write_bytes(b"fake content")

    with open(test_file, "rb") as f:
        resp = requests.post(
            f"{BASE_URL}/api/upload",
            files={"file": (test_file.name, f, "application/octet-stream")},
        )

    print(f"  状态码: {resp.status_code} (期望 400)")
    print(f"  响应:   {resp.json()}")

    test_file.unlink()
    return resp.status_code == 400


def main():
    # 先检查服务是否启动
    try:
        resp = requests.get(f"{BASE_URL}/api/health", timeout=3)
        if resp.status_code != 200:
            print("后端服务未就绪，请先启动:")
            print("  cd backend")
            print("  python -m uvicorn app.main:app --port 8000 --reload")
            return
    except requests.ConnectionError:
        print("无法连接后端服务，请先启动:")
        print("  cd backend")
        print("  python -m uvicorn app.main:app --port 8000 --reload")
        return

    print(f"后端服务正常: {resp.json()}\n")

    results = []
    results.append(("上传", test_upload()))
    results.append(("查看列表", test_list()))
    results.append(("删除", test_delete()))
    results.append(("非法文件拦截", test_upload_invalid()))

    # 清理临时文件
    test_file = TEST_DIR / "upload_test_sample.txt"
    if test_file.exists():
        test_file.unlink()

    print("\n" + "=" * 50)
    print("测试结果汇总")
    print("=" * 50)
    for name, passed in results:
        status = "通过" if passed else "失败"
        print(f"  {name}: {status}")


if __name__ == "__main__":
    main()
