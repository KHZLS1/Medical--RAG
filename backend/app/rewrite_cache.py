"""改写结果磁盘缓存：让查询改写可复现，并省掉重复的 LLM 调用

为什么需要它
------------
评估指标 `hit_rate` / `mrr` **只依赖检索集**，而检索集只由 `enhanced_query` 决定
（`eval_rag.py: compute_hit_rate` 只判断标准答案前 80 字是否出现在召回文档里，
judge=false 时完全不调用 LLM）。也就是说，评估链路里**唯一的随机源就是查询改写**。

实测（2026-09-25）：同一份代码、同一测试集连跑两次，`hit_rate` 会差 4 题（0.08）。
单次评估因此分不出 4 题以内的改动 —— 那些数字不可比。

⚠️ **重要更正（2026-09-25 晚）**：一开始以为"把改写温度降到 0.0 就确定了"，**这是错的**。
实测（关缓存、同 Prompt、同问题、连算两次）：**50 题里有一半左右改写串不一致**，
最坏的连关键词都换了（"卵巢癌 治疗 预后 生存率…" vs "卵巢癌 治疗 预后 食欲不振…"）。
零温只是"贪婪解码"，而 DeepSeek 这类 MoE 模型的输出还受路由/批大小影响 ⇒ **temperature=0 ≠ 可复现**。

**⇒ 本模块是本项目目前唯一的可复现性来源。** 也就是说：缓存命中时结果逐字一致（这才让
指标可比），但**一旦指纹变化（改 Prompt/模型/温度）导致整表作废，参考点就跟着重置**。
所以「改 Prompt 之后必须重跑并重新冻结参考点」不是形式主义。

把改写结果固化到磁盘后：同一批问题永远得到同一个 query → 指标可复现；
顺带把每题一次改写调用也省掉（评估里 50 题就是 50 次调用）。

失效策略
--------
文件头部记录 `fingerprint`（改写 Prompt + 模型 + 温度 的指纹）。任一不符 → 整个
缓存作废重算。prompt 一改就自动重算，不会拿着旧结果骗自己。
条目数超过上限时按插入顺序淘汰最旧的（Python dict 保持插入序）。

并发说明
--------
后端与评估脚本可能同时读写同一文件。写入用「临时文件 + os.replace」原子替换，
读不到/写失败都只告警、不抛异常 —— 缓存是加速手段，绝不能成为故障点。
"""
import hashlib
import json
import os
import tempfile
from pathlib import Path

DEFAULT_MAX_ENTRIES = 5000


class RewriteCache:
    """改写结果的磁盘缓存。

    只认 (kind, history_key, question) 三元组做键 —— kind 区分单轮/多轮两套 Prompt，
    history_key 是多轮时的历史文本。取值为 dict（而非 RewriteResult），
    避免本模块反向依赖 query_rewriter 造成循环导入。
    """

    def __init__(self, path: str | Path, fingerprint: str,
                 max_entries: int = DEFAULT_MAX_ENTRIES):
        self.path = Path(path)
        self.fingerprint = fingerprint
        self.max_entries = max_entries
        self._entries: dict[str, dict] = {}
        self.hits = 0
        self.misses = 0
        self._load()

    # ---------- 键与值 ----------
    @staticmethod
    def make_key(kind: str, history_key: str, question: str) -> str:
        raw = f"{kind}\x00{history_key}\x00{question}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]

    def get(self, kind: str, history_key: str, question: str) -> dict | None:
        item = self._entries.get(self.make_key(kind, history_key, question))
        if not item:
            self.misses += 1
            return None
        self.hits += 1
        return item

    def put(self, kind: str, history_key: str, question: str, query: str,
            degraded: bool = False, reason: str = "", act: str = "new_question") -> None:
        self._entries[self.make_key(kind, history_key, question)] = {
            "q": question,          # 存原问题，文件可读、便于人工核对
            "query": query,
            "degraded": degraded,
            "reason": reason,
            "act": act,             # 对话行为（阶段一），随改写结果一起固化
        }
        while len(self._entries) > self.max_entries:
            self._entries.pop(next(iter(self._entries)))   # 淘汰最旧的
        self._save()

    @property
    def stats(self) -> dict:
        return {"size": len(self._entries), "hits": self.hits, "misses": self.misses,
                "path": str(self.path)}

    # ---------- 落盘 ----------
    def _load(self) -> None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except Exception as e:
            print(f"[改写缓存] 读取失败，按空缓存处理: {e}")
            return

        if (data.get("meta") or {}).get("fingerprint") != self.fingerprint:
            print("[改写缓存] Prompt/模型/温度指纹已变 → 旧缓存作废，本次全部重算")
            return
        entries = data.get("entries")
        if isinstance(entries, dict):
            self._entries = entries

    def _save(self) -> None:
        payload = {
            "meta": {
                "fingerprint": self.fingerprint,
                "count": len(self._entries),
                "note": "按 (kind,history,question) 缓存改写结果；改 Prompt/模型/温度会自动作废",
            },
            "entries": self._entries,
        }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # 先写同目录临时文件再原子替换：避免并发读到半个 JSON
            fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=1)
                os.replace(tmp, self.path)
            except Exception:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise
        except Exception as e:
            print(f"[改写缓存] 写入失败（不影响本次改写）: {e}")
