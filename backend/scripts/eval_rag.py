"""
RAG 回答质量评估脚本

用法:
    cd backend
    python scripts/eval_rag.py build --num 100        # 构建测试集
    python scripts/eval_rag.py eval --mode vector      # 评估纯向量
    python scripts/eval_rag.py eval --mode bm25        # 评估纯 BM25
    python scripts/eval_rag.py eval --mode hybrid       # 评估混合检索
    python scripts/eval_rag.py eval --mode hybrid_rerank # 评估混合+Reranker
    python scripts/eval_rag.py eval --mode hybrid_rerank --judge  # 含LLM裁判
    python scripts/eval_rag.py tune                    # 权重网格搜索
    python scripts/eval_rag.py summary                # 汇总对比

常态化(基线回归):
    python scripts/eval_rag.py eval --mode hybrid_rerank --save-baseline  # 首次: 固化基线
    python scripts/eval_rag.py eval --mode hybrid_rerank --bm25-weight 0.5 --tag w50
        # 改参数重跑, 结束后自动与基线对比并给出回归判定
    python scripts/eval_rag.py compare --mode hybrid_rerank              # 最新归档 vs 基线(不重跑)
    python scripts/eval_rag.py baseline                                  # 查看已有基线
    python scripts/eval_rag.py baseline --mode hybrid_rerank --promote   # 最近一次结果提升为基线

依赖: pip install numpy (其余已在 requirements.txt 中)
"""
import sys
import json
import time
import random
import hashlib
import argparse
from datetime import datetime

import numpy as np
from pathlib import Path

# 添加 backend 目录到 Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jieba
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.vectorstore import get_embedder, hybrid_search
from app.reranker import rerank_documents
from app.query_rewriter import rewrite_query, rewrite_cache_stats
from app.rag_chain import _format_docs_with_sources, build_generation_chain
from app.llm import get_llm
from app.config import settings

# ===== 路径配置 =====
BACKEND_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BACKEND_DIR / "data" / "by_department"
EVAL_DIR = BACKEND_DIR / "data" / "eval"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

# ========== 常态化: 基线固化与回归对比 ==========
HISTORY_DIR = EVAL_DIR / "history"     # 每次评估的归档, 永不覆盖
BASELINE_DIR = EVAL_DIR / "baselines"  # 每个模式一个基线文件

# 指标 -> (中文名, 回归阈值, 方向)  延迟用相对阈值(10%)
METRIC_LABELS = [
    ("avg_hit_rate", "命中率", 0.02, "higher"),
    ("avg_mrr", "MRR", 0.02, "higher"),
    ("avg_similarity", "相似度", 0.01, "higher"),
    ("avg_coverage", "覆盖率", 0.02, "higher"),
    ("avg_llm_score", "LLM评分", 0.10, "higher"),
    ("avg_latency", "延迟", 0.10, "relative_lower"),
]


def _testset_md5(path):
    """测试集指纹: 测试集一变, 对比就失效, 用于比对前校验"""
    return hashlib.md5(Path(path).read_bytes()).hexdigest()[:8]


def _meta_snapshot(mode, testset_path, bm25_weight, vector_weight, k,
                   use_judge, tag=None, use_rewrite=True):
    """记录本次评估的环境快照: 没有这些信息, 基线数字毫无意义"""
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "mode": mode,
        "tag": tag,
        "k": k,
        "bm25_weight": bm25_weight,
        "vector_weight": vector_weight,
        "reranker_top_k": getattr(settings, "reranker_top_k", None),
        "judge": use_judge,
        "rewrite": use_rewrite,
        "testset": Path(testset_path).name,
        "testset_md5": _testset_md5(testset_path),
    }


def _archive_run(meta, summary, details):
    """每次评估都归档, 文件名: 时间__模式[__标签].json"""
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    ts = meta["timestamp"].replace("-", "").replace(":", "")
    name = f"{ts}__{meta['mode']}"
    if meta.get("tag"):
        name += f"__{meta['tag']}"
    path = HISTORY_DIR / f"{name}.json"
    path.write_text(
        json.dumps({"meta": meta, "summary": summary, "details": details},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def _history_files(mode=None):
    """列出归档(按时间排序)"""
    if not HISTORY_DIR.exists():
        return []
    files = []
    for p in HISTORY_DIR.glob("*.json"):
        parts = p.stem.split("__")
        if len(parts) >= 2 and (mode is None or parts[1] == mode):
            files.append(p)
    return sorted(files)


def _baseline_path(mode):
    return BASELINE_DIR / f"{mode}.json"


def _save_baseline(mode, meta, summary, details):
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    p = _baseline_path(mode)
    p.write_text(
        json.dumps({"meta": meta, "summary": summary, "details": details},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return p


def load_baseline(mode):
    p = _baseline_path(mode)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _per_question_drops(base_details, cand_details, metric="mrr",
                        top=5, thresh=0.02):
    """按题对比, 找出指标回退最大的题目(定位回归用)"""
    bmap = {d.get("index"): d for d in (base_details or []) if "error" not in d}
    drops = []
    for d in (cand_details or []):
        if "error" in d:
            continue
        b = bmap.get(d.get("index"))
        if not b:
            continue
        delta = d.get(metric, 0) - b.get(metric, 0)
        if delta < -thresh:
            drops.append((delta, b.get(metric, 0), d))
    drops.sort(key=lambda x: x[0])
    return drops[:top]


def compare_summaries(base_meta, base_summary, cand_meta, cand_summary,
                      base_details=None, cand_details=None, top=5):
    """打印基线 vs 本次的对比表和回归判定"""
    bm, cm = base_meta or {}, cand_meta or {}
    print(f"\n{'='*60}")
    print("基线回归对比")
    print(f"{'='*60}")
    print(f"基线: {bm.get('timestamp', '?')} | "
          f"BM25={bm.get('bm25_weight', '?')} "
          f"Vector={bm.get('vector_weight', '?')} k={bm.get('k', '?')} | "
          f"judge={bm.get('judge', '?')}")
    comparable = bm.get("testset_md5") == cm.get("testset_md5")
    if not comparable:
        print(f"警告: 测试集不一致 (基线 {base_summary.get('count', '?')} 条 "
              f"vs 本次 {cand_summary.get('count', '?')} 条), "
              f"差异不可作为回归依据!")

    n_reg, n_imp = 0, 0
    print(f"\n{'指标':<8} {'基线':>9} {'本次':>9} {'变化':>9}  判定")
    print("-" * 52)
    for key, label, thresh, direction in METRIC_LABELS:
        if key not in base_summary or key not in cand_summary:
            continue
        b, c = base_summary[key], cand_summary[key]
        delta = round(c - b, 4)
        if not comparable:
            print(f"{label:<8} {b:>9.4f} {c:>9.4f} {delta:>+9.4f}  -")
            continue
        if direction == "relative_lower":
            ref = b if b else thresh
            reg, imp = delta > ref * thresh, delta < -ref * thresh
        else:
            reg, imp = delta < -thresh, delta > thresh
        n_reg += reg
        n_imp += imp
        verdict = "回归" if reg else ("改善" if imp else "持平")
        print(f"{label:<8} {b:>9.4f} {c:>9.4f} {delta:>+9.4f}  {verdict}")

    # 定位回归题目 (仅测试集一致时才有意义)
    if (base_details and cand_details
            and bm.get("testset_md5") == cm.get("testset_md5")):
        drops = _per_question_drops(base_details, cand_details, "mrr", top)
        if drops:
            print(f"\nMRR 回退最大的 {len(drops)} 题 (基线 -> 本次):")
            for delta, b_val, d in drops:
                print(f"  #{d.get('index')} mrr {b_val:.2f} -> "
                      f"{d.get('mrr', 0):.2f}  {d.get('question', '')}")

    print()
    if not comparable:
        print("结论: 两次评估的测试集不同, 以上差异不可作为回归依据, 不做判定。"
              "（要比效果请用同一测试集；小样本试跑可加 --no-compare 跳过对比）")
    elif n_reg == 0 and n_imp == 0:
        print("结论: 与基线持平。")
    elif n_reg == 0:
        print(f"结论: 无回归, {n_imp} 项改善。确认有效后执行: "
              f"python scripts/eval_rag.py baseline "
              f"--mode {cm.get('mode', '?')} --promote")
    else:
        print(f"结论: {n_reg} 项回归 / {n_imp} 项改善。"
              f"先看上面的回退题目定位原因；"
              f"若本次是有意的配置变更(换模型/换参数)，"
              f"确认无实质退化后可执行 baseline --mode {cm.get('mode', '?')} --promote "
              f"覆盖旧基线。")
    return {"regressions": n_reg, "improvements": n_imp,
            "comparable": comparable}


def list_baselines():
    if not BASELINE_DIR.exists() or not list(BASELINE_DIR.glob("*.json")):
        print("尚无基线。先固化一个: "
              "python scripts/eval_rag.py eval --mode <mode> --save-baseline")
        return
    print(f"{'模式':<16} {'时间':<20} {'BM25':>5} {'Vec':>5} {'k':>4}  "
          f"{'Hit':<8} {'MRR':<8} 相似度")
    print("-" * 78)
    for p in sorted(BASELINE_DIR.glob("*.json")):
        data = json.loads(p.read_text(encoding="utf-8"))
        m, s = data.get("meta", {}), data.get("summary", {})
        print(f"{p.stem:<16} {str(m.get('timestamp', '?')):<20} "
              f"{str(m.get('bm25_weight', '?')):>5} "
              f"{str(m.get('vector_weight', '?')):>5} "
              f"{str(m.get('k', '?')):>4}  "
              f"{str(s.get('avg_hit_rate', '-')):<8} "
              f"{str(s.get('avg_mrr', '-')):<8} "
              f"{s.get('avg_similarity', '-')}")


def promote_baseline(mode, from_file=None):
    """把某次归档(默认最近一次)提升为该模式的基线"""
    if from_file:
        src = Path(from_file)
        if not src.exists():
            print(f"文件不存在: {src}")
            return
    else:
        hist = _history_files(mode)
        if not hist:
            print(f"模式 {mode} 没有历史归档可提升, 先运行 eval。")
            return
        src = hist[-1]
    data = json.loads(src.read_text(encoding="utf-8"))
    p = _save_baseline(mode, data["meta"], data["summary"],
                       data.get("details", []))
    print(f"基线已更新: {p}")
    print(f"来源归档: {src.name}")


def cmd_compare(mode, top=5):
    """最新一次归档 vs 基线 (纯看历史, 不重跑)"""
    baseline = load_baseline(mode)
    if not baseline:
        print(f"模式 {mode} 尚无基线。先固化: "
              f"eval --mode {mode} --save-baseline")
        return
    hist = _history_files(mode)
    if not hist:
        print(f"模式 {mode} 无历史归档, 请先运行 eval。")
        return
    latest = hist[-1]
    cand = json.loads(latest.read_text(encoding="utf-8"))
    print(f"本次取最新归档: {latest.name}")
    compare_summaries(baseline["meta"], baseline["summary"],
                      cand.get("meta", {}), cand["summary"],
                      baseline.get("details"), cand.get("details"), top)

# ========== Step 1: 构建测试集 ==========
def build_testset(num=100, seed=42, out=None):
    """从 by_department 数据中随机抽样构建测试集

    out: 输出路径；默认 data/eval/eval_testset.json。
         想建小样本试跑时务必指定 out，避免覆盖正式测试集。
    """
    random.seed(seed)
    all_items = []

    for f in DATA_DIR.glob("*.json"):
        with open(f, "r", encoding="utf-8") as fp:
            items = json.load(fp)
        dept = f.stem.replace("cleaned_", "")
        for item in items:
            if not item.get("input") or not item.get("output"):
                continue
            all_items.append(
                {
                    "question": item["input"],
                    "answer": item["output"],
                    "department": dept,
                }
            )

    print(f"总共 {len(all_items)} 条数据，抽样 {num} 条")
    sample = random.sample(all_items, min(num, len(all_items)))

    output = Path(out) if out else EVAL_DIR / "eval_testset.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as fp:
        json.dump(sample, fp, ensure_ascii=False, indent=2)

    print(f"测试集已保存: {output}")
    return output

# ========== Step 2: 按模式检索 ==========
def retrieve_by_mode(question, mode="hybrid_rerank",
                     bm25_weight=0.3, vector_weight=0.7, k=20,
                     timings=None, use_rewrite=True):
    """按指定检索模式获取文档

    全部走 pymilvus 原生混合检索（服务端 BM25 + bge 稠密向量），
    与生产环境 app.vectorstore.hybrid_search 完全同一条链路。
    timings: 传入 dict 时记录各阶段耗时（排查慢在哪一步）
    use_rewrite: 是否做查询改写（关闭可对比改写对召回的影响）
    """
    # 统一查询改写（与生产环境一致）
    t0 = time.time()
    enhanced_query = rewrite_query(question) if use_rewrite else question
    if timings is not None:
        timings["rewrite_s"] = round(time.time() - t0, 2)

    search_mode = "hybrid" if mode.startswith("hybrid") else mode
    t0 = time.time()
    docs = hybrid_search(
        enhanced_query,
        k=k,
        mode=search_mode,
        bm25_weight=bm25_weight,
        vector_weight=vector_weight,
    )
    if timings is not None:
        timings["search_s"] = round(time.time() - t0, 2)

    if mode == "hybrid_rerank":
        # Reranker 精排（用原始 question 做精细匹配），Top-K 与生产配置保持一致
        t0 = time.time()
        docs = rerank_documents(question, docs, top_k=settings.reranker_top_k)
        if timings is not None:
            timings["rerank_s"] = round(time.time() - t0, 2)

    context, sources = _format_docs_with_sources(docs)
    return docs, context, sources

# ========== Step 3: 评估指标 ==========
def compute_similarity(answer, ground_truth):
    """用 bge embedding 计算答案语义相似度 (0-1)"""
    embedder = get_embedder()
    embeddings = embedder.embed_documents([answer, ground_truth])
    a, b = np.array(embeddings[0]), np.array(embeddings[1])
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def compute_coverage(answer, ground_truth):
    """关键词覆盖率 (0-1)：jieba 分词后计算重叠率"""
    gt_words = set(w for w in jieba.cut(ground_truth) if len(w.strip()) > 1)
    ans_words = set(w for w in jieba.cut(answer) if len(w.strip()) > 1)
    if not gt_words:
        return 0.0
    return len(gt_words & ans_words) / len(gt_words)


def compute_hit_rate(docs, ground_truth):
    """检索命中率: 标准答案是否在检索结果中 (0 or 1)"""
    gt_snippet = ground_truth[:80]
    for doc in docs:
        if gt_snippet in doc.page_content:
            return 1
    return 0


def compute_mrr(docs, ground_truth):
    """MRR: 标准答案在检索结果中的倒数排名 (0-1)"""
    gt_snippet = ground_truth[:80]
    for i, doc in enumerate(docs, 1):
        if gt_snippet in doc.page_content:
            return 1.0 / i
    return 0.0


def llm_judge(question, system_answer, ground_truth):
    """用 LLM 作为裁判打分 (1-5)，temperature=0 保证一致性"""
    from langchain_openai import ChatOpenAI
    judge_llm = ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        temperature=0,
        max_tokens=10,
    )
    prompt = ChatPromptTemplate.from_template(
        "请评价以下医疗问答的系统回答质量。\n\n"
        "【问题】{question}\n\n"
        "【标准答案】{ground_truth}\n\n"
        "【系统回答】{system_answer}\n\n"
        "【评分标准】\n"
        "5分: 完全覆盖标准答案核心信息，准确无误\n"
        "4分: 覆盖大部分核心信息，有小遗漏\n"
        "3分: 覆盖部分核心信息，有明显遗漏\n"
        "2分: 仅覆盖少量信息，或不准确\n"
        "1分: 与标准答案几乎无关或完全错误\n\n"
        "请只返回一个数字(1-5)，不要返回其他内容。"
    )
    chain = prompt | judge_llm | StrOutputParser()
    result = chain.invoke({
        "question": question,
        "ground_truth": ground_truth,
        "system_answer": system_answer,
    }).strip()
    try:
        return int(result)
    except ValueError:
        return 3


# ========== Step 4: 生成回答 ==========
def generate_answer(question, context):
    """用医疗 Prompt 生成回答（复用现有生成链）

    评估场景无多轮历史，user_statement 置空，与生产 prompt 变量对齐。
    """
    chain = build_generation_chain()
    return chain.invoke({
        "context": context,
        "question": question,
        "user_statement": "（无）",
    })


# ========== Step 5: 评估单个模式 ==========
def evaluate_mode(mode, testset_path=None, bm25_weight=0.3,
                  vector_weight=0.7, use_judge=False, k=20,
                  tag=None, save_baseline=False, auto_compare=True,
                  verbose=False, use_rewrite=True):
    """评估指定检索模式，输出逐条结果和汇总"""
    if testset_path is None:
        testset_path = EVAL_DIR / "eval_testset.json"

    with open(testset_path, "r", encoding="utf-8") as f:
        testset = json.load(f)

    meta = _meta_snapshot(mode, testset_path, bm25_weight, vector_weight,
                          k, use_judge, tag, use_rewrite)

    results = []
    total = len(testset)

    print(f"\n{'='*60}")
    print(f"评估模式: {mode} | 权重: BM25={bm25_weight}, Vector={vector_weight}")
    print(f"测试集: {total} 条 | LLM裁判: {'是' if use_judge else '否'} | "
          f"查询改写: {'开' if use_rewrite else '关'}"
          + (f" | 标签: {tag}" if tag else ""))
    print("提示: 首题含 Reranker 模型加载(约 1-2 分钟), 每题约 1 分钟, "
          "总时长 ≈ 题数 × 1 分钟")
    print(f"{'='*60}\n", flush=True)

    for i, item in enumerate(testset, 1):
        question = item["question"]
        ground_truth = item["answer"]

        try:
            timings = {} if verbose else None
            t0 = time.time()
            # 检索
            docs, context, sources = retrieve_by_mode(
                question, mode, bm25_weight, vector_weight, k, timings,
                use_rewrite
            )
            # 检索指标
            hit = compute_hit_rate(docs, ground_truth)
            mrr = compute_mrr(docs, ground_truth)
            # 生成回答
            t_gen = time.time()
            answer = generate_answer(question, context)
            if timings is not None:
                timings["gen_s"] = round(time.time() - t_gen, 2)
            elapsed = time.time() - t0
            # 答案指标
            sim = compute_similarity(answer, ground_truth)
            cov = compute_coverage(answer, ground_truth)

            result = {
                "index": i,
                "mode": mode,
                "question": question[:50],
                "hit_rate": hit,
                "mrr": round(mrr, 4),
                "similarity": round(sim, 4),
                "coverage": round(cov, 4),
                # 存下生成答案, 便于事后人工核对指标异常(如覆盖率骤降)
                "answer": answer,
                "latency": round(elapsed, 2),
            }
            if timings is not None:
                result["timings"] = timings
            if use_judge:
                score = llm_judge(question, answer, ground_truth)
                result["llm_score"] = score

            results.append(result)
            status = "OK" if "error" not in result else "ERR"
            print(f"[{i}/{total}] {status} hit={hit} mrr={mrr:.2f} "
                  f"sim={sim:.4f} cov={cov:.4f} {elapsed:.1f}s", end="")
            if use_judge:
                print(f" judge={result['llm_score']}", end="")
            print(flush=True)
            if timings is not None:
                print("        阶段: "
                      + " ".join(f"{k}={v}s" for k, v in timings.items()),
                      flush=True)

        except Exception as e:
            print(f"[{i}/{total}] ERROR: {e}")
            results.append({
                "index": i, "mode": mode,
                "question": question[:50], "error": str(e),
            })

    # 汇总统计
    valid = [r for r in results if "error" not in r]
    summary = {
        "mode": mode,
        "count": len(valid),
        "avg_hit_rate": round(float(np.mean([r["hit_rate"] for r in valid])), 4) if valid else 0,
        "avg_mrr": round(float(np.mean([r["mrr"] for r in valid])), 4) if valid else 0,
        "avg_similarity": round(float(np.mean([r["similarity"] for r in valid])), 4) if valid else 0,
        "avg_coverage": round(float(np.mean([r["coverage"] for r in valid])), 4) if valid else 0,
        "avg_latency": round(float(np.mean([r["latency"] for r in valid])), 2) if valid else 0,
    }
    if use_judge and valid and "llm_score" in valid[0]:
        summary["avg_llm_score"] = round(
            float(np.mean([r["llm_score"] for r in valid])), 2)

    print(f"\n--- {mode} 汇总 ---")
    for k, v in summary.items():
        if k != "mode":
            print(f"  {k}: {v}")

    # 改写缓存命中率：判断这次重跑是否复用了旧改写。
    # hits=50/misses=0 → 与上次逐字相同的 query，指标可比；
    # misses>0 → 有题目是新改写的，与历史结果对比要留噪声余量。
    try:
        stats = rewrite_cache_stats()
        if stats.get("enabled"):
            print(f"  改写缓存: {stats['hits']} 命中 / {stats['misses']} 未命中"
                  f" / 共 {stats['size']} 条 ({stats['path']})")
        else:
            print("  改写缓存: 已关闭（REWRITE_CACHE_ENABLED=false）→ 指标不可复现")
    except Exception as e:            # 缓存只是观测项，不能影响评估
        print(f"  改写缓存: 统计失败 ({e})")

    # 保存结果 (latest, 兼容 summary 命令)
    output = EVAL_DIR / f"eval_{mode}_results.json"
    with open(output, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "summary": summary, "details": results},
                  f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {output}")

    # 归档历史 (永不覆盖, 常态化的数据底座)
    arch = _archive_run(meta, summary, results)
    print(f"已归档: {arch}")

    # 自动回归对比: 该模式有基线就和基线比
    baseline = load_baseline(mode)
    if auto_compare and baseline:
        compare_summaries(baseline["meta"], baseline["summary"],
                          meta, summary,
                          baseline.get("details"), results)

    # 固化/更新基线
    if save_baseline:
        p = _save_baseline(mode, meta, summary, results)
        print(f"基线已{'更新' if baseline else '固化'}: {p}")

    return summary


# ========== Step 6: 权重网格搜索 ==========
def tune_weights(testset_path=None, k=20):
    """网格搜索 BM25_WEIGHT 和 VECTOR_WEIGHT 最佳配比"""
    if testset_path is None:
        testset_path = EVAL_DIR / "eval_testset.json"

    with open(testset_path, "r", encoding="utf-8") as f:
        testset = json.load(f)

    # 权重网格: BM25 从 0.0 到 1.0，步长 0.1
    weights = [(round(i * 0.1, 1), round(1 - i * 0.1, 1))
               for i in range(11)]

    results = []
    total_w = len(weights)

    for idx, (bm25_w, vec_w) in enumerate(weights, 1):
        print(f"\n[{idx}/{total_w}] BM25={bm25_w} Vector={vec_w}")
        hit_rates, mrrs, sims, covs = [], [], [], []

        for i, item in enumerate(testset, 1):
            question = item["question"]
            ground_truth = item["answer"]
            try:
                docs, context, _ = retrieve_by_mode(
                    question, "hybrid", bm25_w, vec_w, k
                )
                hit_rates.append(compute_hit_rate(docs, ground_truth))
                mrrs.append(compute_mrr(docs, ground_truth))
                answer = generate_answer(question, context)
                sims.append(compute_similarity(answer, ground_truth))
                covs.append(compute_coverage(answer, ground_truth))
            except Exception as e:
                print(f"  [{i}] ERROR: {e}")

        r = {
            "bm25_weight": bm25_w,
            "vector_weight": vec_w,
            "avg_hit_rate": round(float(np.mean(hit_rates)), 4) if hit_rates else 0,
            "avg_mrr": round(float(np.mean(mrrs)), 4) if mrrs else 0,
            "avg_similarity": round(float(np.mean(sims)), 4) if sims else 0,
            "avg_coverage": round(float(np.mean(covs)), 4) if covs else 0,
        }
        results.append(r)
        print(f"  hit={r['avg_hit_rate']} mrr={r['avg_mrr']} "
              f"sim={r['avg_similarity']} cov={r['avg_coverage']}")

    # 保存
    output = EVAL_DIR / "weight_tuning.json"
    with open(output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    # 归档 (tune 跑一次很贵, 别让下次覆盖)
    tune_meta = {"timestamp": datetime.now().isoformat(timespec="seconds"),
                 "mode": "tune"}
    arch = _archive_run(tune_meta, {"count": len(results)}, results)
    print(f"已归档: {arch}")

    # 打印汇总表
    print(f"\n{'='*60}")
    print("权重调优结果汇总")
    print(f"{'='*60}")
    print(f"{'BM25':<8} {'Vector':<8} {'Hit':<8} {'MRR':<8} "
          f"{'Sim':<8} {'Cov':<8}")
    print("-" * 54)
    for r in results:
        print(f"{r['bm25_weight']:<8} {r['vector_weight']:<8} "
              f"{r['avg_hit_rate']:<8} {r['avg_mrr']:<8} "
              f"{r['avg_similarity']:<8} {r['avg_coverage']:<8}")

    best_sim = max(results, key=lambda x: x["avg_similarity"])
    best_mrr = max(results, key=lambda x: x["avg_mrr"])
    print(f"\n最佳(相似度): BM25={best_sim['bm25_weight']} "
          f"Vector={best_sim['vector_weight']}")
    print(f"最佳(MRR):    BM25={best_mrr['bm25_weight']} "
          f"Vector={best_mrr['vector_weight']}")

    # 把最优权重落盘到 tuned_weights.json，后端自动覆盖默认值（重启生效）
    tuned_path = BACKEND_DIR / settings.tuned_weights_path
    tuned_path.parent.mkdir(parents=True, exist_ok=True)
    tuned_path.write_text(
        json.dumps({
            "bm25_weight": best_sim["bm25_weight"],
            "vector_weight": best_sim["vector_weight"],
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"最优权重已写入: {tuned_path}（重启后端生效）")
    print(f"结果已保存: {output}")
    return results


# ========== Step 7: 汇总对比 ==========
def print_summary():
    """汇总所有评估结果，输出对比表"""
    print(f"\n{'='*70}")
    print("RAG 评估结果汇总")
    print(f"{'='*70}")

    modes = ["vector", "bm25", "hybrid", "hybrid_rerank"]
    all_s = []
    for mode in modes:
        path = EVAL_DIR / f"eval_{mode}_results.json"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            all_s.append(data["summary"])

    if not all_s:
        print("未找到评估结果，请先运行 eval 命令。")
        return

    headers = ["模式", "Hit", "MRR", "相似度", "覆盖率", "延迟(s)"]
    has_judge = any("avg_llm_score" in s for s in all_s)
    if has_judge:
        headers.append("LLM评分")
    print(f"\n{' | '.join(h.ljust(10) for h in headers)}")
    print("-" * (len(headers) * 12))
    for s in all_s:
        row = [s["mode"], f"{s['avg_hit_rate']:.4f}",
               f"{s['avg_mrr']:.4f}", f"{s['avg_similarity']:.4f}",
               f"{s['avg_coverage']:.4f}", f"{s['avg_latency']:.1f}"]
        if has_judge and "avg_llm_score" in s:
            row.append(f"{s['avg_llm_score']:.2f}")
        print(f"{' | '.join(c.ljust(10) for c in row)}")

    best = max(all_s, key=lambda x: x["avg_similarity"])
    print(f"\n推荐模式(按相似度): {best['mode']}")


# ========== CLI 入口 ==========
def main():
    parser = argparse.ArgumentParser(
        description="RAG 回答质量评估脚本")
    sub = parser.add_subparsers(dest="command")

    # build - 构建测试集
    p_build = sub.add_parser("build", help="构建测试集")
    p_build.add_argument("--num", type=int, default=100, help="抽样数量")
    p_build.add_argument("--seed", type=int, default=42, help="随机种子")
    p_build.add_argument("--out", type=str, default=None,
        help="输出路径(默认 data/eval/eval_testset.json; 建小样本时务必指定, "
             "避免覆盖正式测试集)")

    # eval - 评估指定模式
    p_eval = sub.add_parser("eval", help="评估指定检索模式")
    p_eval.add_argument("--mode", required=True,
        choices=["vector", "bm25", "hybrid", "hybrid_rerank"],
        help="检索模式")
    p_eval.add_argument("--testset", type=str, default=None,
        help="测试集路径(默认 data/eval/eval_testset.json)")
    p_eval.add_argument("--judge", action="store_true",
        help="启用 LLM 裁判评分(消耗额外 API 额度)")
    p_eval.add_argument("--bm25-weight", type=float, default=0.3,
        help="BM25 权重(默认 0.3)")
    p_eval.add_argument("--vector-weight", type=float, default=0.7,
        help="向量权重(默认 0.7)")
    p_eval.add_argument("--k", type=int, default=20,
        help="召回数量(默认 20)")
    p_eval.add_argument("--tag", type=str, default=None,
        help="本次评估标签, 如 w50 / k10")
    p_eval.add_argument("--save-baseline", action="store_true",
        help="把本次结果固化为该模式的基线")
    p_eval.add_argument("--no-compare", action="store_true",
        help="跳过与基线的自动对比")
    p_eval.add_argument("--verbose", action="store_true",
        help="每题打印各阶段耗时(rewrite/search/rerank/gen), 排查慢在哪一步")
    p_eval.add_argument("--no-rewrite", action="store_true",
        help="关闭查询改写(对比改写对召回的影响)")

    # tune - 权重调优
    p_tune = sub.add_parser("tune", help="权重网格搜索")
    p_tune.add_argument("--testset", type=str, default=None)

    # compare - 最新归档 vs 基线
    p_cmp = sub.add_parser("compare", help="最新一次评估与基线对比(不重跑)")
    p_cmp.add_argument("--mode", required=True,
        choices=["vector", "bm25", "hybrid", "hybrid_rerank"])
    p_cmp.add_argument("--top", type=int, default=5,
        help="列出回退最大的前 N 题(默认 5)")

    # baseline - 基线管理
    p_base = sub.add_parser("baseline", help="查看/提升基线")
    p_base.add_argument("--mode", type=str, default=None,
        help="模式(查看时省略则列出全部)")
    p_base.add_argument("--promote", action="store_true",
        help="把该模式最近一次归档提升为基线")
    p_base.add_argument("--from-file", type=str, default=None,
        help="指定归档文件提升为基线")

    # summary - 汇总
    sub.add_parser("summary", help="汇总所有评估结果")

    args = parser.parse_args()

    if args.command == "build":
        build_testset(args.num, args.seed, args.out)
    elif args.command == "eval":
        evaluate_mode(args.mode, args.testset,
                      bm25_weight=args.bm25_weight,
                      vector_weight=args.vector_weight,
                      use_judge=args.judge, k=args.k,
                      tag=args.tag, save_baseline=args.save_baseline,
                      auto_compare=not args.no_compare,
                      verbose=args.verbose,
                      use_rewrite=not args.no_rewrite)
    elif args.command == "tune":
        tune_weights(args.testset)
    elif args.command == "summary":
        print_summary()
    elif args.command == "compare":
        cmd_compare(args.mode, args.top)
    elif args.command == "baseline":
        if args.promote:
            if not args.mode:
                parser.error("--promote 需要 --mode")
            promote_baseline(args.mode, args.from_file)
        else:
            list_baselines()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()