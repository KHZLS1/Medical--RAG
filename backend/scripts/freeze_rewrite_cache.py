"""把当前改写缓存冻结成「入库快照」——A/B 对比的唯一输入来源。

为什么需要它
------------
`app/rewrite_cache.py` 的指纹机制是「Prompt / 模型 / 温度一改，整表作废」。日常这是好事
（不会拿旧结果骗自己），但做 A/B 时反噬：改一版改写 Prompt，指纹就变，两边的输入
不再是同一份 —— 「哪版更好」里混进了「喂的改写词不一样」，结论不可信。

冻结集把**某一次运行**的结果固定成一个文件：

  · 加载时跳过指纹校验（`REWRITE_CACHE_FROZEN=true`）；
  · 只读（不回写），新指纹的条目不会混进来；
  · 用 `git add -f` 入库，换机器 / 清空 data/ 后仍是同一份输入。

⚠️ 冻结模式要求**全命中**。miss 会真调 LLM，而结果不落盘 ⇒ 那几题本轮不可复现。
   跑完评估务必看 `rewrite_cache_stats()` 的 `misses`，非 0 说明冻结集不完整。

用法（backend 目录）
--------------------
  python scripts/freeze_rewrite_cache.py              # 缓存 → data/frozen/rewrite_frozen.json
  python scripts/freeze_rewrite_cache.py --check      # 只对比，不写入
  python scripts/freeze_rewrite_cache.py --from <p>   # 指定源缓存

冻结之后
--------
  git add -f backend/data/frozen/rewrite_frozen.json
  在 .env 里设 REWRITE_CACHE_FROZEN=true
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings


def _load(path: Path) -> tuple[dict, str]:
    """读缓存文件，返回 (entries, fingerprint)；读不到返回 ({}, "")。"""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}, ""
    except Exception as e:
        print(f"[警告] 解析失败 {path}: {type(e).__name__}: {e}")
        return {}, ""
    return data.get("entries") or {}, (data.get("meta") or {}).get("fingerprint", "")


def main() -> int:
    ap = argparse.ArgumentParser(description="冻结改写缓存为入库快照")
    ap.add_argument("--from", dest="src", default=None,
                    help="源缓存路径（默认取 settings.rewrite_cache_path）")
    ap.add_argument("--check", action="store_true",
                    help="只对比冻结集与当前缓存，不写入任何文件")
    args = ap.parse_args()

    src = Path(args.src) if args.src else settings.rewrite_cache_path_resolved
    dst = settings.rewrite_frozen_path_resolved

    src_entries, src_fp = _load(src)
    dst_entries, dst_fp = _load(dst)

    print(f"源缓存   : {src}")
    print(f"  条目 {len(src_entries)}  指纹 {src_fp or '(无)'}")
    print(f"冻结集   : {dst}")
    print(f"  条目 {len(dst_entries)}  指纹 {dst_fp or '(无)'}")

    if args.check:
        only_src = set(src_entries) - set(dst_entries)
        only_dst = set(dst_entries) - set(src_entries)
        print(f"\n[对比] 仅在源缓存里: {len(only_src)} 条")
        print(f"[对比] 仅在冻结集里: {len(only_dst)} 条")
        if only_src:
            print("[提示] 源缓存里有冻结集没有的条目 —— 若这批题会进评估，冻结集需重新生成")
        if only_dst:
            print("[提示] 冻结集里有源缓存没有的条目（换过 Prompt 指纹后属正常）")
        return 0

    if not src_entries:
        print(f"\n[ERROR] 源缓存为空或不存在，无可冻结：{src}")
        print("        先跑一次评估把改写结果攒出来，再执行本脚本。")
        return 1

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)

    # 回读校验：确保写出来的文件确实能按冻结模式加载
    written, written_fp = _load(dst)
    ok = len(written) == len(src_entries)
    print(f"\n[写入] {dst}")
    print(f"[回读] 条目 {len(written)}  指纹 {written_fp or '(无)'}  → {'OK' if ok else '条数不符！'}")

    print("\n下一步：")
    print("  1. git add -f backend/data/frozen/rewrite_frozen.json")
    print("  2. .env 里设 REWRITE_CACHE_FROZEN=true")
    print("  3. 跑评估后检查 rewrite_cache_stats() 的 misses —— 必须为 0，否则冻结集不完整")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
