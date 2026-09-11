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

依赖: pip install numpy (其余已在 requirements.txt 中)
"""
import sys
import json
import time
import random
import argparse
import numpy as np
from pathlib import Path

# 添加 backend 目录到 Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jieba
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.vectorstore import get_embedder, hybrid_search
from app.reranker import rerank_documents
from app.query_rewriter import rewrite_query
from app.rag_chain import _format_docs_with_sources, build_generation_chain
from app.llm import get_llm
from app.config import settings

# ===== 路径配置 =====
BACKEND_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BACKEND_DIR / "data" / "by_department"
EVAL_DIR = BACKEND_DIR / "data" / "eval"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

# ========== Step 1: 构建测试集 ==========
def build_testset(num=100, seed=42):
    """从 by_department 数据中随机抽样构建测试集"""
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

    output = EVAL_DIR / "eval_testset.json"
    with open(output, "w", encoding="utf-8") as fp:
        json.dump(sample, fp, ensure_ascii=False, indent=2)

    print(f"测试集已保存: {output}")
    return output

# ========== Step 2: 按模式检索 ==========
def retrieve_by_mode(question, mode="hybrid_rerank",
                     bm25_weight=0.3, vector_weight=0.7, k=20):
    """按指定检索模式获取文档

    全部走 pymilvus 原生混合检索（服务端 BM25 + bge 稠密向量），
    与生产环境 app.vectorstore.hybrid_search 完全同一条链路。
    """
    # 统一查询改写（与生产环境一致）
    enhanced_query = rewrite_query(question)

    search_mode = "hybrid" if mode.startswith("hybrid") else mode
    docs = hybrid_search(
        enhanced_query,
        k=k,
        mode=search_mode,
        bm25_weight=bm25_weight,
        vector_weight=vector_weight,
    )

    if mode == "hybrid_rerank":
        # Reranker 精排（用原始 question 做精细匹配）
        docs = rerank_documents(question, docs, top_k=5)

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
    """用 DeepSeek 作为裁判打分 (1-5)，temperature=0 保证一致性"""
    from langchain_openai import ChatOpenAI
    judge_llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=settings.deepseek_api_key,
        base_url="https://api.deepseek.com",
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
    """用 DeepSeek + 医疗 Prompt 生成回答（复用现有生成链）"""
    chain = build_generation_chain()
    return chain.invoke({"context": context, "question": question})


# ========== Step 5: 评估单个模式 ==========
def evaluate_mode(mode, testset_path=None, bm25_weight=0.3,
                  vector_weight=0.7, use_judge=False, k=20):
    """评估指定检索模式，输出逐条结果和汇总"""
    if testset_path is None:
        testset_path = EVAL_DIR / "eval_testset.json"

    with open(testset_path, "r", encoding="utf-8") as f:
        testset = json.load(f)

    results = []
    total = len(testset)

    print(f"\n{'='*60}")
    print(f"评估模式: {mode} | 权重: BM25={bm25_weight}, Vector={vector_weight}")
    print(f"测试集: {total} 条 | LLM裁判: {'是' if use_judge else '否'}")
    print(f"{'='*60}\n")

    for i, item in enumerate(testset, 1):
        question = item["question"]
        ground_truth = item["answer"]

        try:
            t0 = time.time()
            # 检索
            docs, context, sources = retrieve_by_mode(
                question, mode, bm25_weight, vector_weight, k
            )
            # 检索指标
            hit = compute_hit_rate(docs, ground_truth)
            mrr = compute_mrr(docs, ground_truth)
            # 生成回答
            answer = generate_answer(question, context)
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
                "latency": round(elapsed, 2),
            }
            if use_judge:
                score = llm_judge(question, answer, ground_truth)
                result["llm_score"] = score

            results.append(result)
            status = "OK" if "error" not in result else "ERR"
            print(f"[{i}/{total}] {status} hit={hit} mrr={mrr:.2f} "
                  f"sim={sim:.4f} cov={cov:.4f} {elapsed:.1f}s", end="")
            if use_judge:
                print(f" judge={result['llm_score']}", end="")
            print()

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

    # 保存结果
    output = EVAL_DIR / f"eval_{mode}_results.json"
    with open(output, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "details": results},
                  f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {output}")
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

    # eval - 评估指定模式
    p_eval = sub.add_parser("eval", help="评估指定检索模式")
    p_eval.add_argument("--mode", required=True,
        choices=["vector", "bm25", "hybrid", "hybrid_rerank"],
        help="检索模式")
    p_eval.add_argument("--testset", type=str, default=None,
        help="测试集路径(默认 data/eval/eval_testset.json)")
    p_eval.add_argument("--judge", action="store_true",
        help="启用 LLM 裁判评分(消耗额外 API 额度)")

    # tune - 权重调优
    p_tune = sub.add_parser("tune", help="权重网格搜索")
    p_tune.add_argument("--testset", type=str, default=None)

    # summary - 汇总
    sub.add_parser("summary", help="汇总所有评估结果")

    args = parser.parse_args()

    if args.command == "build":
        build_testset(args.num, args.seed)
    elif args.command == "eval":
        evaluate_mode(args.mode, args.testset, use_judge=args.judge)
    elif args.command == "tune":
        tune_weights(args.testset)
    elif args.command == "summary":
        print_summary()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()