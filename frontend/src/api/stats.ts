// 展示大屏数据源：GET /api/stats/overview，失败时退到离线快照
//
// 为什么要兜底：这块屏是给展厅无人值守轮播用的。后端挂了、Milvus 没起来、
// 网线被踢了 —— 任何一个都不该让大屏变成一片报错。所以接口拉不到时读
// `data/dashboard-fallback.json`（由 backend/scripts/gen_dashboard_snapshot.py
// 生成），继续轮播一份"最后一次已知"的数字，并在角上明确标注这是快照。
//
// 形状由后端 stats.py 的 build_overview 唯一定义，这里只是镜像；快照与接口
// 同形状，所以前端两条路径共用同一套渲染代码。

import fallback from '../data/dashboard-fallback.json'

export interface DeptCount {
  department: string
  count: number
}

export interface CorpusStats {
  /** 清洗后磁盘语料条数（语料库规模） */
  total_records: number
  departments: number
  /** Milvus 实际可检索条数；连不上时为 null */
  ingested: number | null
  top_departments: DeptCount[]
  other_records: number
  failed_files: number
}

export interface RetrievalStats {
  bm25_weight: number
  vector_weight: number
  reranker_top_k: number
  /** 单路召回条数；读不到 vectorstore 时为 null */
  recall_k: number | null
  /** tuned=调参产物，config=.env 配置 */
  weights_source: 'tuned' | 'config'
}

export interface EvaluationStats {
  file: string
  timestamp: string | null
  mode: string | null
  recall_k: number | null
  bm25_weight: number | null
  vector_weight: number | null
  reranker_top_k: number | null
  testset: string | null
  questions: number | null
  hit_rate: number | null
  mrr: number | null
  coverage: number | null
  similarity: number | null
  latency_s: number | null
}

export interface UsageStats {
  conversations: number
  questions: number
  answers: number
  clarifications: number
}

export interface FeedbackSummary {
  total: number
  up: number
  down: number
  down_rate: number
  with_correction: number
}

export interface DashboardOverview {
  generated_at: string
  corpus: CorpusStats
  retrieval: RetrievalStats
  evaluation: EvaluationStats | null
  usage: UsageStats | null
  feedback: FeedbackSummary | null
}

export interface OverviewResult {
  data: DashboardOverview
  /** true=接口实时数据，false=离线快照兜底 */
  live: boolean
}

/**
 * 拉取大屏总览。
 *
 * 不抛异常 —— 调用方只有"渲染"一件事要做，多一条 catch 分支没有意义。
 * 失败信息通过 live=false 传达，由界面负责标注数据来源。
 */
export async function fetchOverview(top = 10): Promise<OverviewResult> {
  try {
    const res = await fetch(`/api/stats/overview?top=${top}`, { cache: 'no-store' })
    if (!res.ok) throw new Error(`HTTP ${res.status}`)
    const data = (await res.json()) as DashboardOverview
    if (!data?.corpus) throw new Error('返回体缺少 corpus 字段')
    return { data, live: true }
  } catch (e) {
    console.warn('[大屏] 接口不可用，改用离线快照:', e)
    return { data: fallback as DashboardOverview, live: false }
  }
}