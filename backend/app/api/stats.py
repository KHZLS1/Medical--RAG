"""展示大屏数据接口

`GET /api/stats/overview` 一次返回大屏五屏所需的全部数据。做成单接口而不是五个：
展厅大屏是无人值守轮询，五个接口意味着五倍的超时/半成功状态组合，
而前端还要自己拼装"哪些板块是新鲜的、哪些是兜底快照" —— 一个接口一个时间戳，
新鲜度判断就只有一个口径。

各板块均可独立失败（DB 挂了 usage/feedback 为 None、Milvus 挂了 ingested 为 None），
整体仍返回 200。大屏宁可显示"该板块无数据"，也不能因为一个依赖挂掉就整页白屏。
"""
import asyncio

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from .. import stats as stats_source
from ..database import get_db

router = APIRouter(prefix="/api/stats", tags=["展示大屏"])


@router.get("/overview")
async def overview(top: int = 10, db: Session = Depends(get_db)):
    """大屏总览。

    Args:
        top: 「覆盖分布」页返回的科室数量，其余合并为 other_records。
             默认 10：语料分布是典型长尾（头部科室几十万条、尾部只有几条），
             232 根柱子画出来全长一样高，不如只讲头部 + 长尾合计。
    """
    # 语料扫描是几十秒的重 IO，必须离开事件循环；ingested 要连 Milvus（可能超时）。
    # 两者都只做文件/网络 IO，不碰 db —— db 会话仍留在事件循环线程里用，
    # 与项目其它接口一致，避免跨线程使用 Session。
    corpus, ingested, evaluation = await asyncio.gather(
        asyncio.to_thread(stats_source.corpus_stats),
        asyncio.to_thread(stats_source.ingested_count),
        asyncio.to_thread(stats_source.evaluation_stats),
    )

    return stats_source.build_overview(
        db, top, corpus=corpus, ingested=ingested, evaluation=evaluation
    )