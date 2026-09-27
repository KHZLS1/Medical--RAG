"""生成展示大屏的兜底快照

为什么需要它
------------
展厅大屏是无人值守的：后端挂了、Milvus 没起、网络断了，页面都不能白屏 ——
现场没人会去重启服务。所以前端内置一份"最后一次已知良好"的数据快照，
接口请求失败时继续轮播，并在角落标明"当前显示的是快照 + 生成时间"。

快照必须由本脚本从真实数据生成，不能手写数字：手写的那份过两周就和真实情况脱节，
而且没有任何机制提醒它过期了 —— 大屏会一直自信地展示旧数字。

离线可用：DB / Milvus 连不上时对应板块写 null，其余照常生成
（语料规模与评估指标只读本地文件，不依赖任何服务）。

用法:
  cd backend
  python scripts/gen_dashboard_snapshot.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import stats as stats_source

# 写进前端源码目录，由 Vite 打包进产物，随前端一起发到展厅机器
OUT_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "frontend" / "src" / "data" / "dashboard-fallback.json"
)


def _open_session():
    """拿一个 DB 会话；拿不到就返回 None（对应板块留空，不影响其余部分）"""
    try:
        from app.database import SessionLocal

        return SessionLocal()
    except Exception as e:
        print(f"[警告] 数据库不可用，使用量/反馈板块将为空: {type(e).__name__}: {e}")
        return None


def main():
    db = _open_session()
    try:
        payload = stats_source.build_overview(db, top=10)
    finally:
        if db is not None:
            db.close()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    c = payload["corpus"]
    ev = payload["evaluation"]
    print(f"[完成] 已写入 {OUT_PATH}")
    print(f"  语料库   {c['total_records']:,} 条 / {c['departments']} 个科室")
    print(f"  已入库   {c['ingested'] if c['ingested'] is not None else 'Milvus 未连接'}")
    print(f"  评估     {ev['timestamp'] if ev else '无评估结果'}")
    print(f"  使用量   {payload['usage']}")
    print(f"  反馈     {payload['feedback']}")


if __name__ == "__main__":
    main()