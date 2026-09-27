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
