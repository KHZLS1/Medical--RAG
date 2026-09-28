"""T62-B3 离线自测：SSE 流的**空闲**看门狗（`app.main._watchdog_iter`）

不需要 Milvus / LLM / MySQL：只测那个纯 async 生成器。

为什么这个测试非要有
--------------------
它是"慢但在吐字的流不能被误杀"这条结论的**护栏**。前端曾经用"整轮 120s"做判据，
结果 2026-09-28 端到端验收里 8 轮问答有 3 轮在第 120 秒整被掐断 —— 而连接其实一直是
活的（`sse_starlette` 每 15s 发 `: ping`），只是 provider 尾延迟到了分钟级
（实测同一句话三次：1.4s / 33.6s / 254.1s）。
把"整轮计时"和"空闲计时"搞混是这段逻辑唯一但致命的错误，所以必须用一条
**总时长超过阈值、但每条间隔都没超过**的流把它钉住。

跑法（backend 目录、RAG 环境）：

    python scripts/test_sse_idle.py

退出码 0 = 全绿。
"""
import asyncio
import sys
from contextlib import suppress
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app.main as m

_FAILED: list[str] = []


def check(name, cond, detail=""):
    mark = "PASS" if cond else "FAIL"
    print(f"  [{mark}] {name}" + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        _FAILED.append(name)
    return cond


class Wrap:
    """可观测 aclose 的异步流包装（用来验证超时后底层生成器被关掉）。"""

    def __init__(self, agen):
        self._agen = agen
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        return await self._agen.__anext__()

    async def aclose(self):
        self.closed = True
        with suppress(Exception):
            await self._agen.aclose()


def stream_of(items, gap: float = 0.0):
    """把 items 做成异步流；gap 是**每两条之间**的等待（用来制造"慢但在吐字"）。"""
    async def gen():
        for it in items:
            if gap:
                await asyncio.sleep(gap)
            yield it
    return gen()


def with_stall(items, gap: float = 0.0):
    """先吐 items，然后**永久卡住**（模拟 provider 不再吐字）。"""
    async def gen():
        for it in items:
            if gap:
                await asyncio.sleep(gap)
            yield it
        await asyncio.Event().wait()   # 永不 set ⇒ 卡死
    return gen()


async def collect(stream, idle):
    out = []
    async for payload, timed_out in m._watchdog_iter(stream, idle):
        out.append((payload, timed_out))
    return out


async def main():
    print("== B3-1 正常流：全部取到，且没有收尾标记 ==")
    got = await collect(stream_of(["a", "b", "c"]), idle=1.0)
    check("三条都取到", [p for p, _ in got] == ["a", "b", "c"], str(got))
    check("没有 timed_out", not any(t for _, t in got), str(got))

    print("== B3-2 卡住不动 → 到点收尾（只发一次收尾标记）==")
    got = await collect(with_stall(["a", "b"]), idle=0.25)
    check("先拿到已吐出的两条", [p for p, _ in got][:2] == ["a", "b"], str(got))
    check("末尾有一个 timed_out 标记", got[-1] == (None, True), str(got))
    check("收尾标记只出现一次", sum(1 for _, t in got if t) == 1, str(got))

    print("== B3-3 ⚠️ 关键：idle 是**每条重新计时**，不是整轮计时 ==")
    # 总时长 ~0.6s > idle 0.25s，但每条间隔 0.12s < 0.25s ⇒ 必须一条都不丢。
    # 若有人把实现改成"整轮计时"，这条立刻变红。
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    got = await collect(stream_of(list(range(5)), gap=0.12), idle=0.25)
    elapsed = loop.time() - t0
    check("总耗时确实超过了 idle（否则这条测试没意义）", elapsed > 0.25, f"{elapsed:.3f}s")
    check("慢但在吐字的流一条都没被误杀",
          [p for p, _ in got] == list(range(5)), str(got))
    check("没有被判超时", not any(t for _, t in got), str(got))

    print("== B3-4 idle<=0 → 不设超时（保留原行为）==")
    got = await collect(stream_of(["x"], gap=0.3), idle=0.0)
    check("间隔虽长但 idle=0 不掐", [p for p, _ in got] == ["x"], str(got))
    check("没有 timed_out", not any(t for _, t in got), str(got))

    print("== B3-5 超时后底层生成器被关掉（不留悬着的生成器）==")
    wrapped = Wrap(with_stall(["a"], gap=0.05))
    got = await collect(wrapped, idle=0.2)
    check("收到收尾标记", got[-1] == (None, True), str(got))
    check("aclose 被调用", wrapped.closed, str(wrapped.closed))

    print("== B3-6 空流 → 直接结束，不误报超时 ==")
    got = await collect(stream_of([]), idle=0.2)
    check("空流不产出任何条目（含收尾标记）", got == [], str(got))

    print("== B3-7 兜底文案与配置 ==")
    check("配置默认 90s", m.settings.sse_idle_timeout_sec == 90.0,
          str(m.settings.sse_idle_timeout_sec))
    check("「一个 token 都没到」与「吐了一半」是两段不同文案",
          m.IDLE_TIMEOUT_NO_CONTENT != m.IDLE_TIMEOUT_PARTIAL)
    check("两段文案都指向急诊",
          all("120" in s and "急诊" in s
              for s in (m.IDLE_TIMEOUT_NO_CONTENT, m.IDLE_TIMEOUT_PARTIAL)))

    print()
    print("离线自测完成。")
    if _FAILED:
        print(f"FAILED {len(_FAILED)} 项：")
        for n in _FAILED:
            print(f"  - {n}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
