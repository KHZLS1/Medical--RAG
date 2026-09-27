"""旧接口回归自测（阶段二验证清单 #4）

覆盖阶段二**没有动过**的既有接口，确认行为与改动前一致：
  GET    /api/health
  GET    /api/conversations                 会话列表
  POST   /api/conversations                 建会话
  GET    /api/conversations/{id}            会话详情
  PUT    /api/conversations/{id}            改标题
  GET    /api/conversations/{id}/messages   消息列表（含 feedback 回填字段）
  POST   /api/chat                          流式问答（SSE 事件齐全）
  POST   /api/feedback → DELETE → GET /stats 反馈幂等、撤回、汇总分页
  GET    /api/uploads                       已上传文档列表
  DELETE /api/conversations/{id}            清理本次自建会话

⚠️ 刻意**不测** /api/ingest 与 /api/ingest-stream
   项目规约要求向量入库一律走 scripts/ingest.py，禁止调 /api/ingest 接口。
   这两个端点会真往 Milvus 写数据，回归测试不该触发。
   （本脚本旧版正是调了 /api/ingest(limit_per_file=10)，属历史遗留的危险用法，
     已移除；要看入库结果请查 GET /api/uploads。）

跑法（后端已启动）：
  python scripts/test_backend_api.py [--base http://localhost:8000]

退出码：全 PASS → 0；有 FAIL → 1（可直接接 CI）。
"""
import argparse
import json
import sys

import requests

BASE = "http://localhost:8000"
TIMEOUT = 180        # 单轮 SSE 总时长上限（改写 + 检索 + 生成）

_fails: list[str] = []


def check(name, cond, detail=""):
    mark = "PASS" if cond else "FAIL"
    print(f"  [{mark}] {name}" + (f"\n         ↳ {detail}" if detail and not cond else ""))
    if not cond:
        _fails.append(name)
    return cond


def post_chat(question: str, conversation_id: int | None = None):
    """POST /api/chat 并解析 SSE，返回 (events, conversation_id)"""
    events = []
    conv_id = conversation_id
    resp = requests.post(
        f"{BASE}/api/chat",
        json={"question": question, "conversation_id": conversation_id},
        stream=True,
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data:"):
            continue
        try:
            evt = json.loads(line[5:].strip())
        except json.JSONDecodeError:
            continue
        events.append(evt)
        if evt.get("type") == "conversation_id":
            conv_id = evt["data"]
    return events, conv_id


def main():
    global BASE
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE)
    args = ap.parse_args()
    BASE = args.base

    print("=" * 60)
    print("旧接口回归自测（验证清单 #4）")
    print("=" * 60)

    # ===== 1. health =====
    print("\n== 1. GET /api/health ==")
    r = requests.get(f"{BASE}/api/health", timeout=15)
    check("HTTP 200", r.status_code == 200, str(r.status_code))
    check("status=ok", r.json().get("status") == "ok", r.text[:200])

    # ===== 2. 会话 CRUD =====
    print("\n== 2. 会话 CRUD ==")
    r = requests.get(f"{BASE}/api/conversations", timeout=30)
    check("列表 HTTP 200", r.status_code == 200, str(r.status_code))
    check("含 conversations 数组", isinstance(r.json().get("conversations"), list),
          r.text[:200])

    r = requests.post(f"{BASE}/api/conversations",
                      json={"title": "[回归自测] 临时会话"}, timeout=30)
    check("创建 HTTP 200", r.status_code == 200, r.text[:200])
    conv_id = r.json().get("id")
    check("返回整数 id", isinstance(conv_id, int), r.text[:200])

    r = requests.get(f"{BASE}/api/conversations/{conv_id}", timeout=30)
    check("详情 HTTP 200", r.status_code == 200, str(r.status_code))
    check("标题写入正确", r.json().get("title") == "[回归自测] 临时会话", r.text[:200])

    r = requests.put(f"{BASE}/api/conversations/{conv_id}",
                     json={"title": "[回归自测] 改名"}, timeout=30)
    check("改标题 HTTP 200", r.status_code == 200, str(r.status_code))
    check("标题已更新", r.json().get("title") == "[回归自测] 改名", r.text[:200])

    r = requests.get(f"{BASE}/api/conversations/999999999", timeout=30)
    check("不存在的会话 → 404", r.status_code == 404, str(r.status_code))

    # ===== 3. 流式问答（SSE）=====
    print("\n== 3. POST /api/chat 流式问答 ==")
    events, conv_id = post_chat("小孩发烧39度怎么办", conv_id)
    types = [e["type"] for e in events]
    check("有 token 流", "token" in types, str(types[:15]))
    check("有 sources", "sources" in types, str(types[:15]))
    check("有 trace", "trace" in types, str(types[:15]))
    check("有 rewrite（医学路径）", "rewrite" in types, str(types[:15]))
    check("有 message_id（可提交反馈）", "message_id" in types, str(types[:15]))
    msg_id = next((e["data"] for e in events if e["type"] == "message_id"), None)

    # ===== 4. 消息列表 =====
    print("\n== 4. GET /api/conversations/{id}/messages ==")
    r = requests.get(f"{BASE}/api/conversations/{conv_id}/messages", timeout=30)
    check("HTTP 200", r.status_code == 200, str(r.status_code))
    msgs = r.json().get("messages", [])
    roles = {m.get("role") for m in msgs}
    check("含 user 与 assistant", {"user", "assistant"} <= roles, str(roles))
    check("assistant 消息带 sources",
          any(m["role"] == "assistant" and m.get("sources") for m in msgs),
          str([(m["role"], bool(m.get("sources"))) for m in msgs]))
    check("每条都带 feedback 字段（前端恢复评价状态用）",
          all("feedback" in m for m in msgs))

    # ===== 5. 反馈：幂等 / 汇总 / 撤回 =====
    print("\n== 5. 反馈闭环 ==")
    check("拿到 message_id", isinstance(msg_id, int), str(msg_id))
    if isinstance(msg_id, int):
        r = requests.post(f"{BASE}/api/feedback",
                          json={"message_id": msg_id, "thumbs": "down",
                                "comment": "[回归自测] 差评"}, timeout=30)
        check("提交差评 HTTP 200", r.status_code == 200, r.text[:200])
        fb_id = r.json().get("feedback_id")

        # 幂等：同一 message_id 再提交 → 同一条记录，不新增
        r2 = requests.post(f"{BASE}/api/feedback",
                           json={"message_id": msg_id, "thumbs": "up"}, timeout=30)
        check("幂等：重复提交不新建（feedback_id 不变）",
              r2.status_code == 200 and r2.json().get("feedback_id") == fb_id,
              f"{r.status_code}/{r.text[:120]} vs {r2.status_code}/{r2.text[:120]}")
        check("幂等：thumbs 被更新为 up", r2.json().get("thumbs") == "up",
              r2.text[:200])

        r = requests.post(f"{BASE}/api/feedback",
                          json={"message_id": msg_id, "thumbs": "sideways"}, timeout=30)
        check("非法 thumbs → 400", r.status_code == 400, str(r.status_code))

        r = requests.get(f"{BASE}/api/feedback/stats", timeout=30)
        check("stats HTTP 200", r.status_code == 200, str(r.status_code))
        stats = r.json()
        check("stats 含分页与计数字段",
              {"total", "up", "down", "down_rate", "with_correction",
               "down_items", "offset", "has_more"} <= set(stats),
              str(sorted(stats.keys())))

        r = requests.delete(f"{BASE}/api/feedback/{msg_id}", timeout=30)
        check("撤回 HTTP 200", r.status_code == 200, r.text[:200])

        r = requests.get(f"{BASE}/api/conversations/{conv_id}/messages", timeout=30)
        target = next((m for m in r.json()["messages"] if m.get("id") == msg_id), None)
        check("撤回后该条 feedback 归零",
              target is not None and target.get("feedback") is None,
              str(target and target.get("feedback")))

    # ===== 6. 已上传文档列表 =====
    print("\n== 6. GET /api/uploads ==")
    r = requests.get(f"{BASE}/api/uploads", timeout=30)
    check("HTTP 200", r.status_code == 200, str(r.status_code))
    check("含 files 数组", isinstance(r.json().get("files"), list), r.text[:200])

    # ===== 7. 清理自建会话 =====
    print("\n== 7. 清理（DELETE 自建会话）==")
    r = requests.delete(f"{BASE}/api/conversations/{conv_id}", timeout=30)
    check("删除 HTTP 200", r.status_code == 200, r.text[:200])
    r = requests.get(f"{BASE}/api/conversations/{conv_id}", timeout=30)
    check("删除后详情 404", r.status_code == 404, str(r.status_code))

    # ===== 汇总 =====
    print("\n" + "=" * 60)
    if _fails:
        print(f"结论：失败 {len(_fails)} 项 —— {'; '.join(_fails)}")
        print("=" * 60)
        return 1
    print("结论：全部通过 —— 旧接口行为与阶段二改动前一致")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())