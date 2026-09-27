"""LangGraph 检查点：SQLite 落盘，跨请求恢复图状态

为什么 SQLite 而不是 MySQL：
  langgraph 官方只有 sqlite / postgres 两类 saver，MySQL 无官方实现。
  checkpointer 只存图状态（与业务库职责不同），单独一个 sqlite 文件
  是官方支持的形态，免去自己维护 community saver 的成本。

为什么用 AsyncSqliteSaver 而不是同步 SqliteSaver：
  生产链路全程走 graph.astream（SSE 流式），LangGraph 的异步执行路径只调
  checkpointer 的 async 方法——同步 SqliteSaver 的 aget/aput 直接抛
  NotImplementedError（实测症状：SSE 只剩一条 error 事件，无任何 token）。
  AsyncSqliteSaver 依赖 aiosqlite，且构造时必须已在事件循环里（内部调
  get_running_loop），所以单例只能在首个请求内惰性创建，不能 import 期建。
  它的同步方法只允许在**其它线程**调用（事件循环线程里会抛 InvalidStateError），
  配套约定：一切读写都走 async 接口（main.py 的挂起检测用 aget_state）。
  多 worker / 高并发部署再换 PostgresSaver（见实施计划 §4）。

另外提供 `delete_thread()`：删会话时同步清掉该 thread 的快照，别让独立于 MySQL
的 sqlite 无限堆积（阶段二 §4 的记账项）。
"""
import asyncio
from pathlib import Path

import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from .config import settings

_saver: AsyncSqliteSaver | None = None
_init_lock: asyncio.Lock | None = None


async def get_checkpointer() -> AsyncSqliteSaver:
    """AsyncSqliteSaver 单例：事件循环内惰性创建，锁防并发重复建连接。"""
    global _saver, _init_lock
    if _saver is not None:
        return _saver
    if _init_lock is None:
        # 检查与赋值之间无 await，事件循环内不会交错，懒建锁是安全的
        _init_lock = asyncio.Lock()
    async with _init_lock:
        if _saver is None:
            db_path = (
                Path(__file__).resolve().parent.parent / settings.graph_checkpointer_db_path
            )
            db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = await aiosqlite.connect(db_path)
            _saver = AsyncSqliteSaver(conn)
    return _saver


async def delete_thread(conversation_id: int) -> bool:
    """删除某会话的 checkpoint 快照（删除会话时调用）。返回是否真的执行了清理。

    为什么需要它：checkpointer 的 sqlite 是**独立于业务库**的存储 —— 在 MySQL 里
    删掉会话不会波及它，不清理就会一直堆积（每轮都把 docs / context 全文写进去，
    实测已到 9.4MB）。这正是实施计划_阶段二 §4 里记下的那笔账。

    刻意**吞掉异常**：快照残留只是占空间，而"删会话失败"是功能故障。清理层绝不能
    反过来把业务操作拖垮（与 rewrite_cache 的"缓存失败只告警"同一纪律）。
    """
    if not settings.graph_checkpointer_enabled:
        return False          # 没开 checkpointer 就没快照可清（也避免为此建库文件）
    try:
        from .graph import thread_config   # 延迟导入：thread_id 口径只维护一份
        thread_id = thread_config(conversation_id)["configurable"]["thread_id"]
        saver = await get_checkpointer()
        await saver.adelete_thread(thread_id)
        return True
    except Exception as e:
        print(f"[checkpoint] 清理会话 {conversation_id} 的快照失败（不影响会话删除）: "
              f"{type(e).__name__}: {e}")
        return False
