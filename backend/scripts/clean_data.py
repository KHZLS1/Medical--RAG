#!/usr/bin/env python3
"""医疗数据清洗脚本

读取 GBK 编码的医疗 CSV → 清洗 → 输出 JSON 指令微调格式

清洗内容:
  1. GBK → UTF-8 编码转换（解决乱码）
  2. 去重（文件内 + 跨文件，按 input+output 去重；可选按 input 合并同问多答）
  3. 科室过滤（移除非医学科室数据）
  4. 文本规范化（去多余空白/换行/控制字符、繁体转简体、全角转半角）
  5. 空值/过短过滤（答案 < MIN_ANSWER_LEN）
  6. 垃圾/无意义问题过滤（"无/不知道/请问"等精确词、问题 < MIN_ASK_LEN）
  7. 噪音剔除（答案含 HTML 标签 / URL）

输出格式:
  [
      {
          "instruction": "<title>",
          "input": "<ask>",
          "output": "<answer>",
          "history": null
      },
      ...
  ]

用法:
  cd backend
  python scripts/clean_data.py                              # 全量清洗（合并到一个文件）
  python scripts/clean_data.py --split-by-dept             # 按科室分文件输出
  python scripts/clean_data.py --split-by-dept --format jsonl  # 分科室+JSONL
  python scripts/clean_data.py --limit 500                 # 每文件限量500条（测试）
  python scripts/clean_data.py --output result.json        # 指定输出路径
  python scripts/clean_data.py --keep-dept                 # 输出保留department字段
  python scripts/clean_data.py --dedup-strategy first      # 同问多答只留第一条（默认 merge 合并）
"""

import csv
import json
import re
import argparse
from pathlib import Path
from collections import Counter


# ============================================================
# 配置常量
# ============================================================

# 输入目录（上传的 GBK 编码 CSV）
DEFAULT_INPUT_DIR = Path(__file__).resolve().parent.parent / "uploads"

# 默认输出路径
DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "data" / "cleaned_data.json"

# 需要跳过的文件（冗余/样例）
SKIP_FILES = {"样例_内科5000-6000.csv"}

# 非医学科室（这些"科室"实际是疾病名或生活类目，需要过滤）
NON_MEDICAL_DEPARTMENTS = {
    "减肥",
    "健身",
    "体检",
    "生活疾病",
    "①安定",
    "动脉导管未闭",
    "复杂先心病",
}

# 科室名标准化映射（同义合并）
DEPT_NORMALIZE = {
    "神经内科": "神经科",
    "精神疾病": "精神心理",
}

# 繁体→简体映射（常见医疗用字，可按需扩展）
TRAD_TO_SIMP = {
    "龜": "龟",
    "體": "体",
    "腎": "肾",
    "膽": "胆",
    "腸": "肠",
    "腦": "脑",
    "膿": "脓",
    "瘡": "疮",
    "癬": "癣",
    "濕": "湿",
    "熱": "热",
    "腫": "肿",
    "藥": "药",
    "醫": "医",
    "診": "诊",
    "療": "疗",
    "癥": "症",
    "發": "发",
    "節": "节",
    "關": "关",
    "網": "网",
    "細": "细",
    "經": "经",
    "織": "织",
    "線": "线",
    "產": "产",
    "婦": "妇",
    "嬰": "婴",
    "兒": "儿",
    "養": "养",
    "營": "营",
    "補": "补",
    "虛": "虚",
    "實": "实",
    "陰": "阴",
    "陽": "阳",
    "氣": "气",
    "脈": "脉",
    "臟": "脏",
    "宮": "宫",
    "頸": "颈",
    "質": "质",
    "構": "构",
    "機": "机",
    "檢": "检",
    "驗": "验",
    "類": "类",
    "調": "调",
    "論": "论",
    "證": "证",
    "緩": "缓",
    "輕": "轻",
    "過": "过",
    "當": "当",
    "還": "还",
    "進": "进",
    "遠": "远",
    "運": "运",
    "傳": "传",
    "個": "个",
    "們": "们",
    "這": "这",
    "國": "国",
    "會": "会",
    "說": "说",
    "請": "请",
    "謝": "谢",
    "對": "对",
    "開": "开",
    "間": "间",
    "後": "后",
    "從": "从",
    "給": "给",
    "讓": "让",
    "裡": "里",
    "內": "内",
    "兩": "两",
    "點": "点",
    "號": "号",
    "術": "术",
    "復": "复",
    "況": "况",
    "狀": "状",
    "態": "态",
    "動": "动",
    "靜": "静",
    "須": "须",
    "導": "导",
    "數": "数",
    "滿": "满",
    "覺": "觉",
    "觸": "触",
    "觀": "观",
    "釋": "释",
    "針": "针",
    "鐵": "铁",
    "銅": "铜",
    "銀": "银",
    "鉛": "铅",
    "鈣": "钙",
    "鈉": "钠",
    "鉀": "钾",
    "鎂": "镁",
    "鋅": "锌",
    "據": "据",
}

# 最短答案长度（低于此值视为无效）
MIN_ANSWER_LEN = 20

# 最短有效问题长度（低于此值视为无意义问题）
MIN_ASK_LEN = 5

# 垃圾/无意义问题：整句精确匹配（可按需扩展，如"那""好"等单字变体）
JUNK_ASK_RE = re.compile(
    r"^(无|無|没有|没有了|不知道|不清楚|不懂|不会|不会了|"
    r"请问|请教|你好|您好|在吗|好的|谢谢|嗯|哦|噢|啊|呀|"
    r"null|none|nan|na|n/a)$",
    re.IGNORECASE,
)

# 答案噪音：HTML 标签 / 链接残留
HTML_TAG_RE = re.compile(r"<[a-zA-Z/][^>]*>")
URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)


# ============================================================
# 文本清洗函数
# ============================================================


def clean_text(text: str) -> str:
    """文本规范化：繁简转换 + 全半角 + 去控制字符 + 合并空白"""
    if not text:
        return ""

    # 1. 繁体→简体
    for trad, simp in TRAD_TO_SIMP.items():
        text = text.replace(trad, simp)

    # 2. 全角字母/数字/空格→半角
    chars = []
    for ch in text:
        code = ord(ch)
        if 0xFF01 <= code <= 0xFF5E:
            chars.append(chr(code - 0xFEE0))
        elif code == 0x3000:
            chars.append(" ")
        else:
            chars.append(ch)
    text = "".join(chars)

    # 3. 换行符/制表符→空格
    text = re.sub(r"[\r\n\t]+", " ", text)

    # 4. 去除控制字符（保留可打印 ASCII + 中文）
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)

    # 5. 多个空格→单个
    text = re.sub(r" +", " ", text)

    return text.strip()


def is_junk_ask(ask: str) -> bool:
    """是否为垃圾/无意义问题（"无""不知道""请问"等整句精确匹配）"""
    return bool(JUNK_ASK_RE.match(ask))


def has_noise(text: str) -> bool:
    """是否含噪音（HTML 标签 / URL 残留）"""
    return bool(HTML_TAG_RE.search(text) or URL_RE.search(text))


def normalize_department(dept: str) -> str:
    """科室名标准化"""
    dept = dept.strip()
    return DEPT_NORMALIZE.get(dept, dept)


def is_valid_department(dept: str) -> bool:
    """是否为有效医学科室"""
    return dept not in NON_MEDICAL_DEPARTMENTS


# ============================================================
# CSV 读取与清洗
# ============================================================


def clean_csv_file(file_path: Path, limit: int | None = None):
    """清洗单个 GBK CSV 文件，返回清洗后的记录列表和统计"""
    records = []
    stats = Counter()

    with open(file_path, "r", encoding="gb18030", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            stats["raw"] += 1
            if limit and stats["raw"] > limit:
                break

            dept = clean_text(row.get("department", ""))
            dept = normalize_department(dept)

            if not is_valid_department(dept):
                stats["filtered_dept"] += 1
                continue

            title = clean_text(row.get("title", ""))
            ask = clean_text(row.get("ask", ""))
            answer = clean_text(row.get("answer", ""))

            if not ask or not answer:
                stats["filtered_empty"] += 1
                continue

            if len(answer) < MIN_ANSWER_LEN:
                stats["filtered_short"] += 1
                continue

            # 二次清洗：垃圾/无意义问题
            if is_junk_ask(ask):
                stats["filtered_junk_ask"] += 1
                continue

            if len(ask) < MIN_ASK_LEN:
                stats["filtered_short_ask"] += 1
                continue

            # 二次清洗：答案噪音（HTML/URL）
            if has_noise(answer):
                stats["filtered_noise"] += 1
                continue

            records.append(
                {
                    "department": dept,
                    "instruction": title,
                    "input": ask,
                    "output": answer,
                }
            )

    return records, stats


# ============================================================
# 去重
# ============================================================


def deduplicate(records: list[dict], strategy: str = "merge"):
    """按 (input, output) 去重保留首次出现；再按策略处理同问多答。

    strategy:
      - merge   （默认）: 相同 input 的多个不同 output 合并为一条，answer 以换行拼接
      - first   : 相同 input 只保留第一条
      - keep-all: 不去重同问多答，全部保留（仅按 input+output 精确去重）

    返回 (unique_records, dup_count, merged_groups, merged_extra_removed)
      dup_count: 完全重复 (input+output) 删除条数
      merged_groups: 同问多答被处理（合并/去重）的组数（keep-all 时为 0）
      merged_extra_removed: 因同问多答处理额外减少的条数
    """
    # 第一层：按 (input, output) 精确去重
    seen = set()
    unique = []
    dup_count = 0

    for rec in records:
        key = (rec["input"], rec["output"])
        if key in seen:
            dup_count += 1
            continue
        seen.add(key)
        unique.append(rec)

    if strategy == "keep-all":
        return unique, dup_count, 0, 0

    # 第二层：处理同问多答
    by_input: dict[str, list[dict]] = {}
    for rec in unique:
        by_input.setdefault(rec["input"], []).append(rec)

    result = []
    merged_groups = 0
    merged_extra_removed = 0

    for recs in by_input.values():
        if len(recs) == 1:
            result.append(recs[0])
            continue

        merged_groups += 1
        merged_extra_removed += len(recs) - 1

        if strategy == "first":
            result.append(recs[0])
            continue

        # merge：合并多条 answer，保留首条元信息
        base = dict(recs[0])
        base["output"] = "\n\n".join(r["output"] for r in recs)
        result.append(base)

    return result, dup_count, merged_groups, merged_extra_removed


# ============================================================
# 输出
# ============================================================


def write_json(records: list[dict], output_path: Path, keep_dept: bool):
    """输出为 JSON 数组"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    items = []
    for rec in records:
        item = {
            "instruction": rec["instruction"],
            "input": rec["input"],
            "output": rec["output"],
            "history": None,
        }
        if keep_dept:
            item["department"] = rec["department"]
        items.append(item)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=4)

    # 打印样例
    print("\n===== 输出样例（前 3 条）=====")
    for item in items[:3]:
        print(json.dumps(item, ensure_ascii=False, indent=4))


def write_by_department(
    records: list[dict], output_dir: Path, fmt: str, keep_dept: bool
):
    """按科室分文件输出，每个科室一个 JSON/JSONL 文件"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 按科室分组
    dept_groups: dict[str, list[dict]] = {}
    for rec in records:
        dept_groups.setdefault(rec["department"], []).append(rec)

    file_list = []
    for dept, dept_records in sorted(dept_groups.items(), key=lambda x: -len(x[1])):
        # 科室名做文件名（去掉不安全字符）
        safe_name = re.sub(r'[\\/:*?"<>|]', "_", dept).strip()
        ext = "json" if fmt == "json" else "jsonl"
        file_path = output_dir / f"cleaned_{safe_name}.{ext}"

        if fmt == "json":
            write_json(dept_records, file_path, keep_dept)
        else:
            write_jsonl(dept_records, file_path, keep_dept)

        file_list.append((dept, len(dept_records), file_path))
        print(f"  {dept}: {len(dept_records)} 条 → {file_path.name}")

    return file_list


def write_jsonl(records: list[dict], output_path: Path, keep_dept: bool):
    """输出为 JSONL（每行一个 JSON 对象，适合大数据量）"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            item = {
                "instruction": rec["instruction"],
                "input": rec["input"],
                "output": rec["output"],
                "history": None,
            }
            if keep_dept:
                item["department"] = rec["department"]
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # 打印样例
    print("\n===== 输出样例（前 3 条）=====")
    for rec in records[:3]:
        item = {
            "instruction": rec["instruction"],
            "input": rec["input"],
            "output": rec["output"],
            "history": None,
        }
        if keep_dept:
            item["department"] = rec["department"]
        print(json.dumps(item, ensure_ascii=False, indent=4))


# ============================================================
# 主入口
# ============================================================


def main():
    parser = argparse.ArgumentParser(
        description="医疗数据清洗：GBK CSV → JSON 指令格式"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(DEFAULT_INPUT_DIR),
        help="输入目录（GBK CSV 文件所在）",
    )
    parser.add_argument(
        "--output", type=str, default=str(DEFAULT_OUTPUT), help="输出文件路径"
    )
    parser.add_argument(
        "--format",
        choices=["json", "jsonl"],
        default="json",
        help="输出格式：json 数组 或 jsonl 逐行",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="每个文件最多处理多少条（测试用）"
    )
    parser.add_argument(
        "--keep-dept",
        action="store_true",
        help="输出中保留 department 字段（默认不含）",
    )
    parser.add_argument(
        "--dedup-strategy",
        choices=["merge", "first", "keep-all"],
        default="merge",
        help="同问多答处理策略：merge 合并多答（默认）/ first 只留第一条 / keep-all 全部保留",
    )
    parser.add_argument(
        "--split-by-dept",
        action="store_true",
        help="按科室分文件输出（每个科室一个文件）",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output)

    # 1. 扫描输入文件
    csv_files = sorted(input_dir.glob("*.csv"))
    csv_files = [f for f in csv_files if f.name not in SKIP_FILES]

    if not csv_files:
        print(f"[错误] 在 {input_dir} 下未找到 CSV 文件")
        return

    print("=" * 60)
    print("医疗数据清洗")
    print(f"  输入目录:  {input_dir}")
    print(f"  输出路径:  {output_path}")
    print(f"  输出格式:  {args.format}")
    print(f"  文件数:    {len(csv_files)}（跳过 {len(SKIP_FILES)} 个冗余文件）")
    if args.limit:
        print(f"  限量:      每文件 {args.limit} 条")
    print("=" * 60)

    # 2. 逐文件清洗
    all_records = []
    total_stats = Counter()

    for csv_path in csv_files:
        print(f"\n[处理] {csv_path.name}")
        records, stats = clean_csv_file(csv_path, limit=args.limit)

        all_records.extend(records)
        total_stats.update(stats)

        print(f"  原始: {stats['raw']}")
        print(f"  科室过滤: {stats['filtered_dept']}")
        print(f"  空值过滤: {stats['filtered_empty']}")
        print(f"  答案过短: {stats['filtered_short']}")
        print(f"  垃圾问题: {stats['filtered_junk_ask']}")
        print(f"  问题过短: {stats['filtered_short_ask']}")
        print(f"  噪音过滤: {stats['filtered_noise']}")
        print(f"  有效保留: {len(records)}")

    # 3. 跨文件去重
    print(f"\n===== 去重 =====")
    print(f"  去重前: {len(all_records)}")
    all_records, dup_count, merged_groups, merged_extra = deduplicate(
        all_records, args.dedup_strategy
    )
    print(f"  完全重复删除: {dup_count}")
    print(
        f"  同问多答({args.dedup_strategy}): {merged_groups} 组，额外减少 {merged_extra} 条"
    )
    print(f"  去重后: {len(all_records)}")

    # 4. 输出
    print(f"\n===== 写入文件 =====")
    if args.split_by_dept:
        # 分科室输出到子目录
        split_dir = output_path.parent / "by_department"
        file_list = write_by_department(
            all_records, split_dir, args.format, args.keep_dept
        )
        print(f"\n  共 {len(file_list)} 个科室文件，输出目录: {split_dir}")
    else:
        if args.format == "json":
            write_json(all_records, output_path, args.keep_dept)
        else:
            write_jsonl(all_records, output_path, args.keep_dept)
        print(f"\n  文件: {output_path}")

    # 5. 科室分布统计
    print(f"\n===== 科室分布（Top 15）=====")
    dept_counter = Counter(r["department"] for r in all_records)
    for dept, count in dept_counter.most_common(15):
        print(f"  {dept}: {count}")
    print(f"  ...共 {len(dept_counter)} 个科室")

    # 6. 汇总
    print(f"\n{'=' * 60}")
    print("清洗汇总")
    print(f"  原始总量:     {total_stats['raw']}")
    print(f"  科室过滤:     {total_stats['filtered_dept']}")
    print(f"  空值过滤:     {total_stats['filtered_empty']}")
    print(f"  答案过短:     {total_stats['filtered_short']}")
    print(f"  垃圾问题:     {total_stats['filtered_junk_ask']}")
    print(f"  问题过短:     {total_stats['filtered_short_ask']}")
    print(f"  噪音过滤:     {total_stats['filtered_noise']}")
    print(f"  完全重复删除: {dup_count}")
    print(
        f"  同问多答({args.dedup_strategy}): {merged_groups} 组，额外减少 {merged_extra} 条"
    )
    print(f"  最终输出:     {len(all_records)}")
    print(f"  保留率:       {len(all_records) / max(total_stats['raw'], 1) * 100:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
